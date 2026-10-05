"""Bit-exact Verilator validation of the new row-granularity
affine-binary RTL path (see docs/binary_and_matmulfree_investigation.md):
ankhdjet.backend.verilog.emit_layer_affine_binary (wraps cirom_nor_tile +
cirom_act_total_beh) feeding rtl/between_layer/between_layer_affine.sv
(per-channel requantize_affine). NOT signed-off RTL -- new, additive
files only; nothing in rtl/chip or rtl/grid is touched.

Oracle: ternary_matmul_nor on the sign bits as a degenerate +-1 ternary
tensor (no zero weight -> the hit-count accumulate is identical) for
`hit`, raw sum(act) for `act_total`, then requantize_affine for the
final per-channel K-bit output.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ankhdjet.backend.verilog import (
    _column_widths, emit_layer_affine_binary, quantize_q88,
)
from ankhdjet.frontend.ir import Layer, LayerType, QuantScheme, WeightTensor
from ankhdjet.reference.between import requantize_affine
from ankhdjet.reference.nor import ternary_matmul_nor
from tests._verilator_runner import build_and_run

TILE_SV   = REPO_ROOT / "rtl" / "column" / "cirom_nor_tile.sv"
ACTTOT_SV = REPO_ROOT / "rtl" / "column" / "cirom_act_total_beh.sv"
REQA_SV   = REPO_ROOT / "rtl" / "between_layer" / "requantize_affine.sv"
BLA_SV    = REPO_ROOT / "rtl" / "between_layer" / "between_layer_affine.sv"


TB = r"""
`timescale 1ns/1ps
module tb_affine_layer;
    localparam int N        = {N};
    localparam int M        = {M};
    localparam int K        = {K};
    localparam int WACC     = {WACC};
    localparam int SCALE_W  = 16;
    localparam int Q_FRAC   = 8;
    localparam int KOUT     = 8;

    logic clk, rst_n, start, valid;
    logic [K*N-1:0]              act_flat;
    logic [K-1:0]                act_byte [0:N-1];
    logic signed [WACC*M-1:0]    hit_flat;
    logic [WACC-1:0]             act_total;
    logic [KOUT*M-1:0]           out_flat;

    ankhdjet_layer_{LID} u_layer (
        .clk(clk), .rst_n(rst_n), .start(start),
        .act_flat(act_flat),
        .acc_flat(hit_flat), .act_total(act_total), .valid(valid)
    );

    between_layer_affine #(
        .M(M), .WACC(WACC), .SCALE_W(SCALE_W), .Q_FRAC(Q_FRAC), .K(KOUT),
        .SCALE_Q_FLAT({SCALE_LIT}),
        .BIAS_Q_FLAT({BIAS_LIT}),
        .ACTIVATION(0)
    ) u_bla (
        .hit_flat(hit_flat), .act_total(act_total), .out_flat(out_flat)
    );

    always #5 clk = ~clk;
    int i, j, cycles;

    initial begin
        $readmemh("act.hex", act_byte);
        for (i = 0; i < N; i = i + 1) begin
            act_flat[K*(i+1)-1 -: K] = act_byte[i];
        end
        clk = 0; rst_n = 0; start = 0;
        #23 rst_n = 1;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        cycles = 0;
        while (!valid && cycles < 200000) begin
            @(negedge clk);
            cycles++;
        end
        if (!valid) begin $display("TIMEOUT"); $finish; end
        for (j = 0; j < M; j = j + 1) begin
            $display("OUT[%0d]=%0d", j, out_flat[KOUT*(j+1)-1 -: KOUT]);
        end
        $display("CYCLES=%0d", cycles);
        $finish;
    end
endmodule
"""


def _flat_q88_literal(values: np.ndarray, width: int = 16) -> str:
    n = len(values)
    parts = [f"{int(v) & ((1 << width) - 1):0{width}b}" for v in values]
    bits = "".join(reversed(parts))
    return f"{n * width}'b{bits}"


def _write_act_hex(activations: np.ndarray, path: Path) -> None:
    path.write_text("\n".join(f"{int(a):x}" for a in activations) + "\n")


def run_case(workdir: Path, n: int, m: int, k_bits: int,
              seed: int = 0x5EED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sign = rng.integers(0, 2, size=(n, m)).astype(np.int8)
    act = rng.integers(0, 1 << k_bits, size=n, dtype=np.int64)
    scale_f = rng.uniform(-4.0, 4.0, size=m)
    bias_f = rng.uniform(-1.0, 1.0, size=m)
    scale_q88 = np.array([quantize_q88(x) for x in scale_f], dtype=np.int64)
    bias_q88 = np.array([quantize_q88(x) for x in bias_f], dtype=np.int64)

    layer = Layer(
        name="affine0", layer_type=LayerType.LINEAR,
        weights={"weight": WeightTensor(name="weight", data=sign,
                                        scheme=QuantScheme.BINARY)},
        input_dim=n, output_dim=m,
    )
    rtl_src = emit_layer_affine_binary(layer, k_bits=k_bits, subcol_rows=n)

    _, wacc = _column_widths(n, k_bits)
    lid = "affine0"
    tb_src = TB.format(
        N=n, M=m, K=k_bits, WACC=wacc, LID=lid,
        SCALE_LIT=_flat_q88_literal(scale_q88),
        BIAS_LIT=_flat_q88_literal(bias_q88),
    )
    (workdir / "layer.sv").write_text(rtl_src)
    (workdir / "tb.sv").write_text(tb_src)
    _write_act_hex(act, workdir / "act.hex")

    stdout = build_and_run(
        workdir=workdir,
        sources=[TILE_SV, ACTTOT_SV, REQA_SV, BLA_SV,
                workdir / "layer.sv", workdir / "tb.sv"],
        top="tb_affine_layer", build_timeout=300.0, run_timeout=300.0,
    )

    measured = np.zeros(m, dtype=np.int64)
    for line in stdout.splitlines():
        mo = re.match(r"\s*OUT\[(\d+)\]=(-?\d+)", line)
        if mo:
            measured[int(mo.group(1))] = int(mo.group(2))
    if "TIMEOUT" in stdout:
        raise RuntimeError(f"timeout:\n{stdout[-1500:]}")

    sign_pm = (sign.astype(np.int64) * 2 - 1)
    hit = ternary_matmul_nor(sign_pm, act, k_bits=k_bits, subcol_rows=n)
    act_total = int(np.sum(act))
    expected = requantize_affine(hit, act_total, scale_q88, bias_q88,
                                 q_frac=8, k_bits=8, activation="relu")
    return measured, expected


def main() -> int:
    cases = [
        # (n, m, k_bits)
        (16, 4, 4),
        (32, 8, 8),
        (64, 8, 8),
        (128, 16, 8),
    ]
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="affine_layer_") as td:
        work = Path(td)
        for idx, (n, m, k) in enumerate(cases):
            measured, expected = run_case(work, n, m, k)
            ok = np.array_equal(measured, expected)
            print(f"[{'ok' if ok else 'FAIL'}] case {idx}  N={n} M={m} K={k} "
                  f"match={int((measured==expected).sum())}/{m}")
            if not ok:
                mis = np.where(measured != expected)[0]
                for j in mis[:3]:
                    print(f"    out[{j}]: exp={expected[j]} got={measured[j]}")
                all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
