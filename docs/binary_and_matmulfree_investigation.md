# Binary weights and MatMul-free architectures: investigation

Scoping record for two things the frontend cannot ingest today, plus an
honest accounting of what's actually been proven to work versus what's
only shape-tested. Not a commitment to build either; a reference for
whoever picks this up.

## 1. Binary (2-value) weights — case study: FBI-LLM

Every layer of the stack assumes ternary `{-1, 0, +1}`, not "N discrete
levels":

- `ankhdjet/frontend/hf.py`: `decode_ternary` (274-298) rejects any
  tensor with more than 3 unique raw values and requires the positive
  and negative magnitudes to match; `decode_group_ternary` (304-324)
  and `is_group_ternary` (327-347) generalize the *scale grid* to
  per-group `±s`, but still require a `0` bucket in every group;
  `unpack_ternary_uint8` (254-271) hardcodes the 2-bit-packed
  `{-1,0,1}` convention.
- `ankhdjet/frontend/ir.py`: `QuantScheme` (28-30) has only
  `TERNARY`/`FLOAT`; `WeightTensor.__post_init__` (41-48) hard-asserts
  `vals.issubset({-1, 0, 1})` for any `TERNARY` tensor and has no field
  for a per-group bias/offset.
- `ankhdjet/backend/wmat.py` (33, 43, 55) re-asserts the same value set
  on emission, and the NOR-tile MAC cell in `rtl/`/`macro_grid.py` is
  built to accumulate ternary charge, not a generic level count.

Checked a real release against this: `LiqunMa/FBI-LLM_130M` (paper:
"Scaling Up Fully Binarized LLMs from Scratch via Autoregressive
Distillation"). Its `config.json` is plain `model_type: llama`,
`architectures: [LlamaForCausalLM]` — no new architecture dispatch
needed, unlike part 2. But the weight encoding doesn't fit the
existing ternary ladder at all:

- Training code (`qat/learnable_binarizer.py`, `BinaryLinearWscales`):
  `forward` does `w = STEBinary().apply(self.weight)` (sign, ±1), then
  `w = self.wscale * w + self.wbias` — an **affine** transform, not a
  symmetric scale. `to_regular_linear()` folds this into a plain
  tensor for release: `binary_weight * wscale.data + wbias.data`.
- Result: each row (or column/tensor, depending on `scaling_pattern`)
  of a released `q_proj`/`gate_proj`/etc. matrix holds **exactly two**
  distinct float values, `{bias − scale, bias + scale}`, with **no
  zero level** and no requirement that the two levels be symmetric
  around zero. `decode_ternary`/`decode_group_ternary` both require a
  zero bucket and reject anything with an unaccounted-for bias, so
  this pattern fails every existing detector as-is, not just the
  "too many unique values" check.
- Storage is `pytorch_model.bin` (pickle, float32), not `.safetensors`.

**Implemented** (software stack, 2026-10-04): items 1-3 below are done
and tested — `decode_binary_affine` (`ankhdjet/frontend/hf.py`),
`QuantScheme.BINARY` + `WeightTensor.bias` (`ankhdjet/frontend/ir.py`,
threaded through `ModelIR.save`/`.load`), wired into `load_weights`'s
decode ladder as the third rung after `decode_ternary`/
`decode_group_ternary`. `wmat.py`/`macro_grid.py` refuse `BINARY`
tensors with a precise diagnosis rather than mis-emitting them as
ternary. Tests: `tests/test_hf_frontend.py`
(`test_binary_affine_decodes_per_row`,
`test_binary_affine_decodes_sub_row_groups`), `test_ir_persistence.py`
(bias round-trip), `test_wmat_emit.py`
(`test_binary_scheme_refused_at_emission`).

What building this would take, and what turned out cheaper than
expected once the RTL was actually sketched out (below):
1. ~~New detector~~ **done**: per group, find exactly 2 unique values,
   recover `scale=(v2-v1)/2`, `bias=(v1+v2)/2`, store a sign bitmask
   plus the `(scale, bias)` pair.
2. ~~New `QuantScheme.BINARY`~~ **done**, with a `bias` field on
   `WeightTensor` that didn't exist before.
3. ~~Decode ladder wiring~~ **done**.
4. `wmat.py` emission + the RTL/NOR-tile backend: refusal is
   implemented; real emission is not, and the earlier framing here
   overstated the blocker for the common case. For FBI-LLM's dominant
   **row-granularity** pattern (one scale+bias per OUTPUT COLUMN,
   `scaling_pattern='row'`), the sign bit never floats a drain — it's
   always BL+ or BL-, which is a strict *subset* of the ternary mask
   vocabulary the signed-off cell already implements. A behavioral
   (simulation-only, not signed-off) model in
   `rtl/column/cirom_affine_binary_beh.sv` demonstrates this: it
   reuses `cirom_nor_tile` completely unmodified for the hit-count
   accumulate (`HAS_VIA_NEG = ~HAS_VIA_POS`), adds one array-wide
   running activation-total accumulator (new, but shared by every
   column, not per-column), and replaces the existing single shared
   per-tensor Q8.8 requantize multiply with M independent per-column
   Q8.8 multiply-adds — exactly the cost `docs/array_architecture.md`'s
   "shared scale multiplier" density note already identifies as the
   dominant wrapper lever, just paid per-column instead of once.
   Verified bit-exact against `ankhdjet.reference.nor.binary_affine_matmul_nor`
   via Verilator across 4 shapes (`tests/test_cirom_affine_binary_beh.py`).
   **This reference model is not wired into any signed-off chip/macro
   RTL, has not been through layout/DRC/LVS, and is not a claim that
   row-granularity binary support is production-ready** — only that,
   for this specific sub-case, no new bitcell is needed, which
   contradicts what this document originally said. The **sub-row
   group-granularity** case (bias/scale varying *within* a row, the
   general case `decode_binary_affine` also decodes) does not reduce
   this way: a per-group bias needs a per-group running activation
   total, not one shared across the whole row, i.e. new per-group
   periphery analogous to the analog variant's existing TriMLA group
   accumulators. That case remains unimplemented and is the part of
   the original "needs a hardware change" claim that still holds.
5. Caveat: this is FBI-LLM's specific affine-binary pattern. A
   hypothetical symmetric binary release with a zero bucket (unlikely
   by construction, since `sign()` never outputs exactly 0) would be
   a much smaller change confined to the frontend.

**Correction (2026-10-04), from testing against the real checkpoint:**
everything above item 5 was built on an *assumed* shape for the folded
checkpoint. Downloading `LiqunMa/FBI-LLM_130M` and inspecting the real
`pytorch_model.bin` disproved two parts of that assumption at once:

- **The checkpoint is never folded.** `weight` is the continuous
  training-time latent (every element of a column distinct — checked:
  2048 unique values in one `gate_proj` column), not the two-value
  `to_regular_linear()` output. The actual effective weight is
  `wscale*sign(weight) + wbias`, computed at forward time. There is
  nothing to *detect* — `decode_binary_affine`'s pattern search
  correctly returns `None` on it (verified), but that's because this
  isn't a decode-by-inference problem at all: the checkpoint names the
  affine parameters explicitly as `.wscale`/`.wbias` sibling tensors
  (the same convention as BitNet's `.weight_scale`), so the right move
  is a direct read, not a search.
- **The granularity is the other axis.** `q_proj.weight` is `(768,
  768)`, ambiguous on a square matrix, but `gate_proj.weight` is
  `(2048, 768)` with `wscale`/`wbias` shape `(1, 768)` — `768` matches
  `in_features`, not `out_features`. Confirmed on `down_proj` too
  (`(768, 2048)` weight, `(1, 2048)` scale/bias). So scale/bias vary
  **per input feature**, broadcast over every output row — FBI-LLM's
  `'column'` pattern, not the `'row'` pattern items 1-5 above assumed.
  This flips where the cheap accumulator lives: the per-row scale
  needs to multiply the *activation* before the row-sweep accumulate,
  not the hit-count *after* it, and there's no M-wide post-multiply to
  reuse the ternary tile's cheap bit-serial AND-based hit count for.

**Built for the corrected shape:**
- `ankhdjet/frontend/hf.py`'s `decode_binary_sign_sibling` — a direct
  read (`sign = raw >= 0`, `scale`/`bias` straight from the sibling
  tensors), wired into `load_weights` ahead of the ternary ladder
  whenever `.wscale`/`.wbias` siblings are present. Verified against
  three real layers of `LiqunMa/FBI-LLM_130M` (`gate_proj`,
  `down_proj`, `q_proj`): reconstructed `wscale*sign(weight)+wbias`
  matches the direct computation exactly on all three.
- `WeightTensor.scale_axis` (`ir.py`): `"output"` (the existing
  row/per-tensor/group-ternary convention, unchanged) vs `"input"`
  (this case) — explicit, not inferred from shape, since silently
  guessing the wrong axis here would silently compute the wrong model.
- `rtl/column/cirom_input_axis_affine_beh.sv` +
  `ankhdjet.reference.nor.input_axis_affine_matmul` — a new behavioral
  reference (not signed-off, not cycle-accurate: it sweeps one row per
  cycle rather than K-cycle bit-serial, since a wide signed per-row
  product doesn't decompose into the existing single-bit AND the way
  a plain activation bit does). Bit-exact via Verilator across 4
  shapes (`tests/test_input_axis_affine_beh.py`). The module's header
  states plainly that recovering bit-serial activation handling for
  this axis is an open question, not solved here.
**Update (2026-10-04): the full real-checkpoint pipeline now runs.**
`load_weights` only reads `.safetensors`, and FBI-LLM ships pickle
(`pytorch_model.bin`) — rather than add pickle loading to the trusted
frontend, `tools/convert_pickle_checkpoint.py` is a new, isolated,
one-time conversion script: it is the *only* place in the toolchain
that calls `torch.load` (with `weights_only=True`, PyTorch's
allowlisted-tensors-only mode — not a full sandbox, so only point it
at checkpoints from sources you trust), and it writes a plain
`model.safetensors` + copied `config.json`/tokenizer files that
`load_weights` reads completely normally. `load_weights` (and nothing
else) gained a small local-directory branch: if `repo_id` is already a
directory, it skips `snapshot_download` and reads straight from it.

Ran end to end against the real checkpoint:
`convert_pickle_checkpoint.py` on the downloaded
`LiqunMa/FBI-LLM_130M` snapshot, then
`load_weights('/path/to/converted', ...)`. Result: all 12 layers'
7 projections each (84 total: q/k/v/o/gate/up/down) decode as
`BINARY` with `scale_axis="input"` through the ordinary
`parse_hf_config`/`block_projections` path — no architecture-specific
code was needed, confirming the earlier guess that FBI-LLM's plain
llama-shaped config and module names "just work." `lm_head` correctly
falls through to the existing non-ternary placeholder path (it's kept
full-precision, same convention as BitNet's tied bf16 head). Spot
reconstruction check on a real layer
(`model.layers.0.mlp.gate_proj`): the IR's `sign`/`scale`/`bias`
reproduce `wscale*sign(weight)+wbias` with **zero** numeric
difference against the original tensors. This is now a permanent,
gated regression
(`tests/test_hf_frontend.py::test_fbi_llm_sign_sibling_decodes_through_load_weights`,
skips cleanly without a local conversion present, mirroring the
existing Qwen3.6 real-row test's gating convention).

**Update (2026-10-04, part 2): items 1-4 above are now done.**

1. **`scale_axis` is now recorded through the manifest.** `GridManifest`
   gained a `scale_axis` field (`macro_grid.py`), set from
   `wt.scale_axis` and written into `manifest.json` alongside
   `scale_bias_file`, so a downstream consumer reading `scale_bias.npz`
   back no longer has to guess which axis it's for.
   `tests/test_wmat_emit.py::test_binary_manifest_records_input_axis`
   covers it.
2. **`verilog.py` now has a real emitter for the input axis**:
   `emit_layer_input_axis_affine()` bakes a real layer's per-row
   `scale`/`bias` as Q8.8 constants into `cirom_input_axis_affine_beh`
   (passed as the `HAS_VIA_POS`-style parameter below, not a port —
   see point 4), followed by one `requantize` instance per column at
   unit scale (reused unmodified; it's doing only the relu+clip
   activation stage here, since the affine combine already happened
   inside the axis module) for the standard K-bit output bus
   convention. Bit-exact via Verilator across 3 shapes
   (`tests/test_input_axis_layer_emit.py`).
3. **The "open" bit-serial question is resolved, not left open**:
   it doesn't help, and the module's existing structure (one
   time-shared multiplier, N row-cycles, M simple conditional
   adders) is the right choice, not a placeholder. Reasoning: a
   bit-serial decomposition would need, per cycle, a full N-wide
   signed adder tree per column gated by that cycle's activation bit
   (since the per-cell quantity `sign_pm[r,c]*scale[r]` is a real
   signed number, not ±1, bit-serial-izing it doesn't turn it into a
   cheap AND the way a single activation bit does for the ternary
   case) — K cycles of that is strictly more total work than N cycles
   of one shared multiply + M cheap adds. The output-axis case's
   bit-serial AND-based hit count is cheap specifically *because* the
   per-cell weight is ±1; losing that is the real, structural cost of
   the input axis, and no cycle restructuring recovers it.
4. **A real decoded FBI-LLM layer now runs through the RTL.**
   `tests/test_fbi_llm_real_layer_rtl.py` decodes
   `model.layers.0.self_attn.q_proj` via `load_weights` on the
   converted checkpoint, slices it to 128×64 (same convention as the
   existing real-weight BitNet test, `test_bitnet_block0_all_projections.py`
   — Verilator's per-literal width limit rejects the full 768×768
   mask even as a parameter override, so real-weight RTL validation in
   this codebase is always a slice, never the full matrix), and
   confirms the generated RTL's output matches
   `input_axis_affine_matmul` bit-exact on real sign/scale/bias with
   synthetic activations.

   One fix discovered along the way:
   `cirom_input_axis_affine_beh.sv`'s sign mask had to move from an
   input *port* to a `HAS_VIA_POS`-style compile-time *parameter*
   (matching `cirom_nor_tile`'s convention) — Verilator's width limit
   for a literal in a bare port-connection expression is far lower
   than for a parameter override, and the mask is mask-programmed
   (fab-time-fixed) anyway, so this is also the architecturally
   correct representation, not just a workaround.

**Update (2026-10-04, part 3): both remaining items are done.**

- **Multi-layer pipeline chaining**: `emit_pipeline_input_axis()`
  chains N input-axis layers through a one-cycle valid->start
  handshake, mirroring `emit_pipeline`'s structure for the ternary
  case — but simpler, since no `between_layer`-equivalent stage is
  needed between stages: each `emit_layer_input_axis_affine` module
  already applies its own activation+clip internally, so a layer's
  `out_flat` bus just gets unpacked back into the next layer's
  `act[N]` array on the handshake register. Bit-exact via Verilator
  on 2- and 3-layer synthetic chains
  (`tests/test_input_axis_pipeline_emit.py`).
- **Full-model compile against the real checkpoint**: ran
  `ankhdjet.backend.macro_grid.emit_model` against the real decoded
  `LiqunMa/FBI-LLM_130M` (all 84 projection layers, `lm_head` skipped
  as off-fabric, same convention as BitNet's tied bf16 head). Result:
  5,184 macros, 84,934,656 weights (exact match: 12 × (4×768×768 +
  2×2048×768 + 768×2048)), **0 padded** (64×256 macro geometry
  divides every FBI-LLM_130M projection shape exactly), 85.27 MB of
  mask programs. Then ran the project's independent
  `ankhdjet.backend.verify.verify_model` audit — the same bit-for-bit
  reassembly-from-disk check `docs/results.md` reports for BitNet —
  and it initially reported false failures on all 84 layers, because
  `verify.py` predates `BINARY` and compared the IR's `{0,1}` sign
  mask directly against the emitted `{-1,+1}` mask without the sign
  mapping, and also applied the ternary-only "padding must be
  all-zero" check to a scheme where every position is always BL+/BL-
  (never floating), so that check doesn't apply at all. Fixed both in
  `verify.py` (now scheme-aware); re-ran: **84/84 layers bit-exact
  from disk.** Both fixes are covered by new synthetic tests in
  `tests/test_verify.py`, and the full real-checkpoint run is a
  permanent gated regression
  (`tests/test_fbi_llm_full_model_compile.py`).

Physical signoff remains explicitly out of scope, unchanged from the
start of this investigation.

**Update (2026-10-04, part 4): a real synthesis-quality tile
controller, modeled on the signed-off `cirom_dig_ctrl.sv`.**

Everything built in parts 1-3 above (`cirom_input_axis_affine_beh.sv`,
`emit_layer_input_axis_affine`, `emit_pipeline_input_axis`) is
explicitly a *behavioral reference*: 64-bit casts everywhere, unpacked
array ports, built to prove the math is right as cheaply as possible
in simulation, not to go through synthesis. Taking this to an actual
SKY130 chip flow needs something closer to `cirom_dig_ctrl.sv` (the
signed-off Darga tile's full ternary MAC on die) — careful bit widths,
host-loadable activation store, no shortcuts.

`rtl/tt_digital/cirom_dig_ctrl_affine.sv` is that: modeled directly on
`cirom_dig_ctrl.sv`'s FSM skeleton and host interface, extended with
the one new thing this case needs (a per-row Q8.8 scale/bias multiply
folded into the row sweep, plus one shared bias accumulator added back
in before streamout — see the file's header for the derivation).
Feature set is deliberately reduced from `cirom_dig_ctrl.sv` (only the
on-die activation store, only MVM mode; the streamed/raw-row-read
modes are omitted) to keep the first version tractable. One real width
bug surfaced and got fixed while building it: the result byte-stream
mux initially assumed exactly 2 bytes per accumulator
(`cirom_dig_ctrl.sv`'s own fixed convention, valid there since ternary
hit-counts stay under 16 bits) — this design's accumulated quantity is
a sum of `ACT_W × SCALE_W` products, which needs more than 2 bytes at
real sizes, so the mux is now generic over `GRPB = ACC_W/OUT_W` bytes.

Validated two ways:
- **Functional regression** (`tests/test_cirom_dig_ctrl_affine.py`):
  the controller driving a real array interface
  (`cirom_array_beh`, the same behavioral macro stand-in the
  signed-off chip's own bench uses — not a new stand-in), compared
  bit-exact against `input_axis_affine_matmul` across 3 shapes
  including multi-pass column grouping (`N_ACC < N_COLS`).
- **Synthesis**: `yosys synth` (generic cells; no SKY130 standard-cell
  mapping, since `PDK_ROOT` isn't set up in this environment — see
  below) elaborates cleanly at both a small test size (2,557 cells)
  and at a real FBI-LLM layer's full dimensions (`N_ROWS=768,
  N_COLS=256, N_ACC=32`: 12,669 cells, ~24s, one expected benign
  warning — Yosys converting the small `acc[]` array to discrete
  registers, the same thing it does for `cirom_dig_ctrl.sv`'s
  identically-shaped array). `check` reports 0 problems at both sizes.

**What a real SKY130 LibreLane run still needs, and why it didn't
happen:** this environment has `yosys` and `verilator` but not
`openroad`, `opensta`, `magic`, `netgen`, or a `PDK_ROOT` (`~/.ciel`
SKY130 PDK) — `librelane/*/run_librelane.sh`'s own comments describe
installing these via `nix build` as "first-time setup." That's a
real, slow (likely tens of minutes, multi-GB), environment-changing
install, so it wasn't done without asking first. Beyond tooling, a
LibreLane run also needs: a chip-top RTL instantiating this controller
against the real hardened macro's scalar-port contract (mirroring
`cirom_chip_digital.sv`'s `` `ANKHDJET_ARRAY_MODULE `` binding, not yet
written), and a `config.json`/macro-placement/pin-order/SDC set
(mirroring `librelane/cirom_chip_digital/`, not yet written). Both are
ordinary extensions of what exists, not open design questions — the
open questions (synthesis-quality RTL, functional regression) are the
ones this update closed.

`config.json` for `ridger/MMfreeLM-370M`: `model_type: hgrn_bit`,
`architectures: [HGRNBitForCausalLM]`, with fields (`attn_mode:
fused_recurrent`, `num_heads`, `hidden_ratio`, `expand_ratio`,
`conv_size`, `share_conv_kernel`, `use_lower_bound`) that share almost
nothing with the llama-family or Qwen3.5/3.6 shapes the frontend
already understands. This is a linear-RNN token mixer (HGRN:
Hierarchically Gated Recurrent Network) with no self-attention matmul
at all, plus a ternary MLP.

- `parse_hf_config` (hf.py, 129-165) has no dispatch on `model_type`
  or `architectures` at all — it's pure field-presence extraction
  assuming llama-shaped keys (`hidden_size`, `num_attention_heads`,
  `intermediate_size`, `vocab_size`), with one special-cased branch
  for Qwen3.5/3.6 hybrid attention: `layer_types` entries restricted
  to exactly `{"full_attention", "linear_attention"}` (138-146),
  feeding `block_projections` (168-198) which builds the gated-delta
  projection set (`in_proj_qkv/z/a/b`, `out_proj`). That's shaped for
  Qwen's gated-delta-net, not HGRN's recurrent gate + short causal
  conv — different module names, different math.
- Outcome today: `parse_hf_config` raises `KeyError` on the missing
  llama-shaped keys before it gets anywhere near weight loading. Even
  if those were patched around, `load_weights`'s missing-projection
  check (hf.py, ~422-427) would fail next — HGRN-bit's state dict uses
  its own module names for the recurrent mixer and ternary MLP, not
  `self_attn.q_proj`/`mlp.gate_proj`.

What building this would take:
1. A genuinely new config-parsing branch mapping HGRN-bit's fields
   (`hidden_ratio`, `expand_ratio`, `num_heads`, `conv_size`) to IR
   dimensions, instead of assuming `num_attention_heads`/
   `intermediate_size` exist.
2. A new `block_projections` variant for the recurrent-gate mixer +
   short conv, built from the real checkpoint's actual state-dict key
   names (not yet inspected against real weights — only the config
   was fetched, not the tensors).
3. The MLP/mixer weights are plausibly still ternary per the paper's
   own quantization scheme, which is good news: once the config and
   module-name mapping exist, the *existing* ternary decode ladder
   might apply directly to the value tensors with no new numeric
   decoder. This is unverified — no real HGRN-bit checkpoint's tensor
   values have been downloaded and inspected in this investigation.
4. The Qwen3.6 work already introduced a "recurrent state kept beside
   the KV cache" concept for linear-attention blocks — conceptually
   reusable for HGRN's recurrent hidden state, but the wiring is new:
   today's `layer_types` dispatch only recognizes
   `{"full_attention", "linear_attention"}`, and HGRN-bit is uniformly
   recurrent (no hybrid per-block split), so even the enum doesn't fit
   without a new case.
5. Backend: since the weights are plausibly still 3-level ternary (not
   the FBI-LLM affine-binary case), this is likely almost entirely a
   **frontend parsing problem** — a new config/module-name mapping —
   not a hardware encoding problem. Unlike part 1, no RTL/MAC change
   is expected to be required, pending confirmation against real
   weights.

## 3. What's actually confirmed working today

Only one model has true end-to-end proof; most of the rest of the
"supported" surface is shape-tested against synthetic configs, not
decoded from real checkpoints, in this repo's own test suite.

**Hardware-validated (real weights → IR → RTL, bit-exact):**
`microsoft/bitnet-b1.58-2B-4T` (2.084B ternary fabric weights,
packed-u8 storage, llama-family BitNet). Evidence:
`tests/test_microsoft_bitnet_pytorch_match.py` decodes a real `q_proj`
tensor via `load_weights` and matches `transformers`' own
`unpack_weights`+`F.linear`; `tests/test_bitnet_block0_all_projections.py`
takes all 7 block-0 projection shapes through `emit_layer_nor` →
Verilator, bit-exact against the Python NOR reference;
`tests/test_microsoft_bitnet_nor_layer.py` does single-layer NOR-tile
validation. `docs/results.md` records the full pipeline going further
still: all 210 layers compiled to 128,400 mask-program macros
(2,084,044,800 weights, 2.11 GB `.wmat`), an independent `ankhdjet
verify` pass reassembling all 210 layers bit-for-bit against the
checkpoint, and a 3-way "architectural twin" (transformers reference,
from-scratch numpy, and the fabric's own bit-serial decomposition)
agreeing token-for-token on a real greedy decode. **This is the only
model taken all the way to mask-program/RTL validation.**

**Config/shape-tested only (never decoded from real weights in-repo):**
A synthetic Qwen3.6-27B-class hybrid-attention config
(`tests/test_hf_frontend.py:108-153`) proves the parser produces
correct shapes and a ~24.3B-param count — no real tensor is decoded. A
test that *would* check a real Qwen3.6 tensor
(`test_group_scaled_real_row_from_the_qwen36_release`) exists but
`pytest.skip`s unless a scratch `.npy` file is already present
locally — no evidence in CI or docs that it has been run for real. A
generic llama-family config (hidden_size=2560, 30 layers,
`test_llama_family_config_is_unchanged`) is likewise synthetic-only.

**`docs/results.md`'s "Ternary checkpoint coverage" table** — the
closest thing to a real coverage report, 12 real HF releases across 5
orgs, 0.113B–6.979B params:

| Evidence level | Count | Releases |
|---|---|---|
| Full real-tensor decode (some with full emission/audit) | 5 | TriLM_190M_Unpacked (+ full 6,912-macro emission, 112/112 audit), Falcon3-1B-1.58bit, Falcon-E-1B-Base, **bitnet-b1.58-2B-4T** (+ full 210-layer emission/verify/twin), Llama3-8B-1.58-100B-tokens |
| Config-only (weights never fetched/decoded) | 6 | bitnet_b1_58-large (679M), Falcon-E-3B-Base, bitnet_b1_58-3B, TriLM_3.9B_Unpacked, Falcon3-7B-1.58bit, Ternary-Bonsai-8B-unpacked |
| Refused | 1 | Ternary-Bonsai-1.7B-unpacked (group-scaled fp16, "per-group scales exceed the per-tensor requantize contract") |

**This table is stale on the refused row.** It hasn't been regenerated
since the public-release squash commit (`1dce69a`); commit `82feab7`
("Decode group-scaled ternary storage in the frontend...") was written
specifically to handle exactly the group-scaled-fp16 pattern
Ternary-Bonsai uses. Both Ternary-Bonsai releases would very likely
decode successfully today — the table needs a re-run, not a read, for
that row and probably for the other `config-only` rows now that more
of the ladder exists.

**Bottom line on sizes:** real-tensor decode is demonstrated from
0.113B (TriLM_190M) up to 6.979B (Llama3-8B-1.58) params. Everything
above is config-only or stale. The single size taken to hardware/RTL
validation is 2.08B ternary fabric weights
(bitnet-b1.58-2B-4T) — nothing larger has been pushed that far.
~3.9B params and below estimate inside one reticle-class ASAP7 die;
~7B releases estimate just past one reticle — that's an area-estimator
finding, not an ingestion limit.

**CI caveat:** the real-checkpoint tests gate on a local HF cache
(`_hf_cache_has_repo`) and skip cleanly when it's absent. Neither
`nightly.yml` nor `fast-check.yml` references an HF model name or
`HF_HOME` — none of this is fetched or enforced in CI. "Confirmed"
above means confirmed when someone runs it locally with the cache
populated, not continuously verified per commit.

All of this — part 3 included — is still inside the ternary family.
Binary (part 1) and non-attention architectures (part 2) are gaps
nothing in the current test suite or results table touches.
