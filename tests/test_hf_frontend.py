"""The ternary decode ladder at the frontend boundary: packed uint8,
ternary-valued float, all-zero, and the refusals (asymmetric,
many-valued), plus the absmean transform for QAT master weights.
Pure numpy; no torch, no network."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ankhdjet.frontend.hf import (
    absmean_quantize, decode_ternary, is_group_ternary,
    unpack_ternary_uint8,
)

pytestmark = pytest.mark.package


def _pack(src: np.ndarray) -> np.ndarray:
    """Reference packer for the HF lane convention (inverse of
    unpack_ternary_uint8): lane i holds contiguous block i of axis 0."""
    n4 = src.shape[0]
    assert n4 % 4 == 0
    n = n4 // 4
    enc = (src + 1).astype(np.uint8)
    out = np.zeros((n,) + src.shape[1:], dtype=np.uint8)
    for lane in range(4):
        out |= enc[lane * n : (lane + 1) * n] << (2 * lane)
    return out


def test_packed_uint8_round_trip():
    rng = np.random.default_rng(7)
    src = rng.choice([-1, 0, 1], size=(64, 12)).astype(np.int8)
    decoded = decode_ternary(_pack(src))
    assert decoded is not None
    W, scale = decoded
    assert scale == 1.0
    assert np.array_equal(W, src)
    assert np.array_equal(unpack_ternary_uint8(_pack(src)), src)


def test_ternary_float_decodes_with_embedded_scale():
    rng = np.random.default_rng(8)
    tern = rng.choice([-1, 0, 1], size=(16, 8)).astype(np.float32)
    s = np.float32(0.0071)
    decoded = decode_ternary(tern * s)
    assert decoded is not None
    W, scale = decoded
    assert np.array_equal(W, tern.astype(np.int8))
    assert scale == pytest.approx(float(s))


def test_unit_valued_float_has_scale_one():
    W, scale = decode_ternary(np.array([[1.0, -1.0, 0.0]], dtype=np.float32))
    assert scale == 1.0
    assert np.array_equal(W, [[1, -1, 0]])


def test_all_zero_tensor_decodes():
    W, scale = decode_ternary(np.zeros((4, 4), dtype=np.float32))
    assert scale == 1.0
    assert not W.any()


def test_asymmetric_values_refused():
    assert decode_ternary(np.array([-0.5, 0.0, 0.3], dtype=np.float32)) is None


def test_many_valued_tensor_refused():
    assert decode_ternary(np.array([-0.5, -0.1, 0.0, 0.5], dtype=np.float32)) is None
    rng = np.random.default_rng(9)
    assert decode_ternary(rng.normal(size=(8, 8)).astype(np.float32)) is None


def test_absmean_matches_b158_transform():
    raw = np.array([[0.5, -0.4], [0.05, 0.0]], dtype=np.float32)
    W, scale = absmean_quantize(raw)
    assert scale == pytest.approx(float(np.abs(raw).mean()))
    assert np.array_equal(W, [[1, -1], [0, 0]])


def test_absmean_zero_tensor():
    W, scale = absmean_quantize(np.zeros((3, 3), dtype=np.float32))
    assert scale == 1.0 and not W.any()


def test_group_scaled_ternary_is_diagnosed_not_decoded():
    rng = np.random.default_rng(11)
    tern = rng.choice([-1, 0, 1], size=(16, 512)).astype(np.float32)
    scales = rng.uniform(0.01, 0.1, size=(16, 4))  # one scale per 128-col group
    raw = tern * np.repeat(scales, 128, axis=1).astype(np.float32)
    assert decode_ternary(raw) is None
    assert is_group_ternary(raw)


def test_gaussian_tensor_is_not_group_ternary():
    rng = np.random.default_rng(12)
    assert not is_group_ternary(rng.normal(size=(16, 512)).astype(np.float32))


def _qwen36_27b_config() -> dict:
    """The Qwen3.6-27B config, as its ternary derivatives keep it: the
    language model nested under text_config, 64 blocks of which every
    fourth is full attention and the rest gated-delta linear attention."""
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
        "vision_config": {"depth": 27, "hidden_size": 1152},
        "text_config": {
            "model_type": "qwen3_5_text", "hidden_size": 5120, "intermediate_size": 17408,
            "num_hidden_layers": 64, "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
            "attn_output_gate": True, "full_attention_interval": 4,
            "layer_types": ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)],
            "linear_num_key_heads": 16, "linear_key_head_dim": 128,
            "linear_num_value_heads": 48, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
            "vocab_size": 248320, "max_position_embeddings": 262144,
        },
    }


def test_hybrid_attention_config_builds_the_checkpoint_shapes():
    """Every projection of a hybrid-attention config lands in the IR at the
    shape the checkpoint stores it (the Qwen3.6-27B safetensors index): the
    gated q_proj at twice the head columns, the joined
    q/k/v projection of a linear block, its gate, decay and output projections;
    only the full-attention blocks count toward the KV cache."""
    from ankhdjet.frontend.hf import parse_hf_config, build_ir_from_arch, block_projections
    arch = parse_hf_config(_qwen36_27b_config(), name="qwen36_27b")
    assert arch.hf_prefix == "model.language_model" and arch.num_hidden_layers == 64
    assert arch.n_full_attention_layers == 16 and arch.n_linear_attention_layers == 48
    assert arch.layer_type(3) == "full_attention" and arch.layer_type(0) == "linear_attention"
    full = {name: (path, n, m) for name, path, n, m in block_projections(arch, 3)}
    assert full["b3_q"] == ("model.language_model.layers.3.self_attn.q_proj", 5120, 12288)
    assert full["b3_k"][1:] == (5120, 1024) and full["b3_v"][1:] == (5120, 1024)
    assert full["b3_o"] == ("model.language_model.layers.3.self_attn.o_proj", 6144, 5120)
    lin = {name: (path, n, m) for name, path, n, m in block_projections(arch, 0)}
    assert lin["b0_qkv"] == ("model.language_model.layers.0.linear_attn.in_proj_qkv", 5120, 10240)
    assert lin["b0_z"][1:] == (5120, 6144) and lin["b0_a"][1:] == (5120, 48) and lin["b0_b"][1:] == (5120, 48)
    assert lin["b0_o"] == ("model.language_model.layers.0.linear_attn.out_proj", 6144, 5120)
    assert lin["b0_down"] == ("model.language_model.layers.0.mlp.down_proj", 17408, 5120)
    ir = build_ir_from_arch(arch)
    assert len(ir.layers) == 48 * 8 + 16 * 7 + 1
    backbone = sum(l.input_dim * l.output_dim for l in ir.layers if l.name != "lm_head")
    assert backbone == 48 * (5120 * (10240 + 6144 + 48 + 48) + 6144 * 5120 + 3 * 5120 * 17408) \
        + 16 * (5120 * (12288 + 1024 + 1024) + 6144 * 5120 + 3 * 5120 * 17408)
    assert 24.3e9 < backbone < 24.4e9
    assert arch.linear_state_values == 48 * (48 * 128 * 128 + 4 * 10240)


def test_hybrid_config_refusals_and_interval_fallback():
    from ankhdjet.frontend.hf import parse_hf_config
    cfg = _qwen36_27b_config()
    cfg["text_config"].pop("layer_types")            # the interval alone names the pattern
    arch = parse_hf_config(cfg)
    assert arch.n_full_attention_layers == 16 and arch.layer_type(7) == "full_attention"
    bad = _qwen36_27b_config(); bad["text_config"]["layer_types"][0] = "sliding_attention"
    with pytest.raises(ValueError, match="sliding_attention"):
        parse_hf_config(bad)
    bad = _qwen36_27b_config(); bad["text_config"].pop("linear_num_value_heads")
    with pytest.raises(ValueError, match="head geometry"):
        parse_hf_config(bad)


def test_llama_family_config_is_unchanged():
    """A llama-family config keeps its projection names, shapes, module
    paths and KV accounting: every block has a cache, no state."""
    from ankhdjet.frontend.hf import parse_hf_config, build_ir_from_arch, block_projections
    cfg = {"hidden_size": 2560, "intermediate_size": 6912, "num_hidden_layers": 30, "num_attention_heads": 20,
           "num_key_value_heads": 5, "vocab_size": 128256, "max_position_embeddings": 4096}
    arch = parse_hf_config(cfg, name="bitnet")
    assert arch.hf_prefix == "model" and not arch.layer_types and arch.n_full_attention_layers == 30
    assert arch.linear_state_values == 0 and arch.q_out_dim == 2560 and arch.head_dim == 128
    rows = block_projections(arch, 2)
    assert [r[0] for r in rows] == ["b2_q", "b2_k", "b2_v", "b2_o", "b2_gate", "b2_up", "b2_down"]
    assert rows[0][1] == "model.layers.2.self_attn.q_proj" and rows[1][2:] == (2560, 640) and rows[6][2:] == (6912, 2560)
    ir = build_ir_from_arch(arch)
    assert len(ir.layers) == 30 * 7 + 1 and ir.layers[-1].name == "lm_head"


def test_group_scaled_ternary_decodes_with_its_group():
    """A tensor whose rows hold {-s_g, 0, +s_g} per run of 128 inputs decodes
    to its sign mask and an (out, in/128) scale array, the group found as the
    largest consistent candidate; a tensor with one scale per 64 is found at
    64; a tensor whose magnitudes differ inside every run is refused."""
    from ankhdjet.frontend.hf import decode_group_ternary
    rng = np.random.default_rng(21)
    signs = rng.choice([-1, 0, 1], size=(6, 512)).astype(np.float32)
    s128 = rng.uniform(0.005, 0.02, size=(6, 4)).astype(np.float32)
    raw = signs * np.repeat(s128, 128, axis=1)
    W, s, g = decode_group_ternary(raw)
    assert g == 128 and np.array_equal(W, signs.astype(np.int8))
    assert s.shape == (6, 4) and np.allclose(s, s128)
    s64 = rng.uniform(0.005, 0.02, size=(6, 8)).astype(np.float32)
    W, s, g = decode_group_ternary(signs * np.repeat(s64, 64, axis=1))
    assert g == 64 and s.shape == (6, 8) and np.allclose(s, s64)
    # an all-zero run carries scale 0 and stays consistent
    raw[2, 128:256] = 0
    W, s, g = decode_group_ternary(raw)
    assert g == 128 and s[2, 1] == 0 and not W[2, 128:256].any()
    assert decode_group_ternary(rng.normal(size=(6, 512)).astype(np.float32)) is None
    assert decode_group_ternary(_pack(signs[:4].astype(np.int8))) is None        # packed storage is the other ladder rung
    assert decode_group_ternary(raw[:, :500]) is None                          # no candidate divides the width


def test_binary_affine_decodes_per_row():
    """A tensor whose rows each hold exactly two values bias +/- scale (no
    zero bucket, FBI-LLM's folded BinaryLinearWscales 'row' pattern) decodes
    to its sign mask plus one (scale, bias) pair per row; a tensor with a
    zero bucket (ternary) or three-plus values is left to the ternary
    ladder/refused, and packed uint8 storage is not this ladder's rung."""
    from ankhdjet.frontend.hf import decode_binary_affine
    rng = np.random.default_rng(7)
    sign = rng.choice([0, 1], size=(6, 512)).astype(np.int8)
    scale = rng.uniform(0.01, 0.05, size=(6, 1)).astype(np.float32)
    bias = rng.uniform(-0.2, 0.2, size=(6, 1)).astype(np.float32)
    raw = np.where(sign == 1, bias + scale, bias - scale).astype(np.float32)
    out_sign, s, b, g = decode_binary_affine(raw)
    assert g == 512 and np.array_equal(out_sign, sign)
    assert np.allclose(s, scale) and np.allclose(b, bias)

    # a tensor with a zero bucket is ternary-shaped, not this ladder's job
    assert decode_binary_affine(
        np.array([[-0.5, 0.0, 0.5]], dtype=np.float32)) is None
    # noise has more than two values per row
    assert decode_binary_affine(rng.normal(size=(6, 512)).astype(np.float32)) is None
    # packed storage belongs to the ternary uint8 rung
    assert decode_binary_affine(_pack(np.ones((4, 8), dtype=np.int8))) is None


def test_binary_affine_decodes_sub_row_groups():
    """A per-group (not whole-row) affine scale/bias is found at its own
    group size, same as decode_group_ternary's largest-consistent-candidate
    search."""
    from ankhdjet.frontend.hf import decode_binary_affine
    rng = np.random.default_rng(11)
    sign = rng.choice([0, 1], size=(4, 256)).astype(np.int8)
    scale = rng.uniform(0.01, 0.05, size=(4, 2)).astype(np.float32)
    bias = rng.uniform(-0.2, 0.2, size=(4, 2)).astype(np.float32)
    lo = np.repeat(bias - scale, 128, axis=1)
    hi = np.repeat(bias + scale, 128, axis=1)
    raw = np.where(sign == 1, hi, lo).astype(np.float32)
    out_sign, s, b, g = decode_binary_affine(raw)
    assert g == 128 and s.shape == (4, 2)
    assert np.array_equal(out_sign, sign)
    assert np.allclose(s, scale) and np.allclose(b, bias)


def test_binary_sign_sibling_decodes_latent_weight():
    """FBI-LLM's actual released convention (confirmed against a real
    LiqunMa/FBI-LLM_130M checkpoint -- see
    docs/binary_and_matmulfree_investigation.md): `weight` is the
    continuous, never-folded training latent (every element distinct),
    and the real effective weight is wscale*sign(weight)+wbias with
    wscale/wbias broadcast per INPUT FEATURE (one value per column of
    the stored (out, in) tensor, shared across every output row) --
    the opposite IR axis from decode_binary_affine's row/per-tensor
    case. This decoder does no pattern search: it is a direct read."""
    from ankhdjet.frontend.hf import decode_binary_sign_sibling
    rng = np.random.default_rng(3)
    out_f, in_f = 24, 16
    latent = rng.normal(size=(out_f, in_f)).astype(np.float32)  # never collapses to 2 values
    wscale = rng.uniform(0.01, 0.05, size=(1, in_f)).astype(np.float32)
    wbias = rng.uniform(-0.1, 0.1, size=(1, in_f)).astype(np.float32)

    sign, s, b = decode_binary_sign_sibling(latent, wscale, wbias)
    assert sign.shape == latent.shape
    assert set(np.unique(sign).tolist()).issubset({0, 1})
    assert np.array_equal(s, wscale) and np.array_equal(b, wbias)

    w_eff_direct = wscale * np.sign(latent) + wbias
    w_eff_decoded = b + (sign.astype(np.float32) * 2 - 1) * s
    assert np.allclose(w_eff_direct, w_eff_decoded)
    # column-wise granularity: every column collapses to exactly 2 values
    for c in range(in_f):
        assert np.unique(w_eff_decoded[:, c]).size == 2

    # the raw latent (many distinct values per row/column) correctly
    # fails every pattern-search decoder -- there is nothing to find
    assert decode_ternary(latent) is None
    from ankhdjet.frontend.hf import decode_binary_affine, decode_group_ternary
    assert decode_group_ternary(latent) is None
    assert decode_binary_affine(latent) is None


def test_group_scaled_real_row_from_the_qwen36_release():
    """A real k_proj row of a group-scaled ternary release, if its checkpoint
    tensor was fetched into the scratchpad, decodes at group 128."""
    import os
    p = Path(os.environ.get("ANKHDJET_SCRATCH", "")) / "kproj_l3.npy"
    if not p.exists():
        pytest.skip("no fetched tensor")
    from ankhdjet.frontend.hf import decode_group_ternary
    W, s, g = decode_group_ternary(np.load(p))
    assert g == 128 and W.shape == (1024, 5120) and s.shape == (1024, 40) and (s > 0).all()
