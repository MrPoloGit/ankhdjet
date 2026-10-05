"""Bit-exact Verilator validation of cirom_input_axis_affine_beh, the
INPUT-AXIS affine-binary REFERENCE model (scale/bias per input row,
broadcast over output columns -- FBI-LLM's actual released convention;
see docs/binary_and_matmulfree_investigation.md and the module's own
header comment for why this is NOT signed-off RTL and not
cycle-accurate). Compares against
ankhdjet.reference.nor.input_axis_affine_matmul on random sign masks
and per-row Q8.8 scale/bias.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ankhdjet.reference.nor import input_axis_affine_matmul
from tests._verilator_runner import build_and_run

BEH_SV = REPO_ROOT / "rtl" / "column" / "cirom_input_axis_affine_beh.sv"


TB = r"""
`timescale 1ns/1ps
module tb_input_axis;
    localparam int N       = {N};
    localparam int M       = {M};
    localparam int K       = {K};
    localparam int SCALE_W = 16;
    localparam int Q_FRAC  = 8;
    localparam int WACC    = {WACC};

    logic clk, rst_n, start, valid;
    logic [K-1:0]              act      [0:N-1];
    logic signed [SCALE_W-1:0] scale_q  [0:N-1];
    logic signed [SCALE_W-1:0] bias_q   [0:N-1];
    logic signed [WACC-1:0]    result   [0:M-1];

    cirom_input_axis_affine_beh #(
        .N(N), .M(M), .K(K), .SCALE_W(SCALE_W), .Q_FRAC(Q_FRAC), .WACC(WACC),
        .HAS_VIA_POS({SIGN_LIT})
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .act(act), .scale_q(scale_q), .bias_q(bias_q),
        .result(result), .valid(valid)
    );

    always #5 clk = ~clk;
    int i, j, cycles;

    initial begin
        $readmemh("act.hex", act);
        $readmemh("scale.hex", scale_q);
        $readmemh("bias.hex", bias_q);
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
            $display("OUT[%0d]=%0d", j, result[j]);
        end
        $display("CYCLES=%0d", cycles);
        $finish;
    end
endmodule
"""


def _write_hex_rows(values: np.ndarray, path: Path, width_hex: int) -> None:
    path.write_text(
        "\n".join(f"{int(v) & ((1 << (4 * width_hex)) - 1):0{width_hex}x}"
                  for v in values) + "\n")


def _bits_literal(bits: np.ndarray) -> str:
    n = bits.size
    s = "".join("1" if int(bits[n - 1 - i]) else "0" for i in range(n))
    return f"{n}'b{s}"


def run_case(workdir: Path, n: int, m: int, k_bits: int,
              seed: int = 0xC0DE) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sign = rng.integers(0, 2, size=(n, m)).astype(np.int64)
    act = rng.integers(0, 1 << k_bits, size=n, dtype=np.int64)
    scale_q = rng.integers(-2000, 2000, size=n, dtype=np.int64)
    bias_q = rng.integers(-2000, 2000, size=n, dtype=np.int64)

    expected = input_axis_affine_matmul(sign, act, scale_q, bias_q, q_frac=8)

    wacc = max(32, int(np.ceil(np.log2(
        n * (1 << k_bits) * 4096 + 1))) + 2)
    sign_lit = _bits_literal(sign.flatten())   # row r, col c -> bit r*M+c
    tb_src = TB.format(N=n, M=m, K=k_bits, WACC=wacc, SIGN_LIT=sign_lit)
    (workdir / "tb.sv").write_text(tb_src)
    _write_hex_rows(act, workdir / "act.hex", width_hex=max(1, (k_bits + 3) // 4))
    _write_hex_rows(scale_q, workdir / "scale.hex", width_hex=4)
    _write_hex_rows(bias_q, workdir / "bias.hex", width_hex=4)

    stdout = build_and_run(
        workdir=workdir, sources=[BEH_SV, workdir / "tb.sv"],
        top="tb_input_axis", build_timeout=300.0, run_timeout=300.0,
    )

    measured = np.zeros(m, dtype=np.int64)
    for line in stdout.splitlines():
        mo = re.match(r"\s*OUT\[(\d+)\]=(-?\d+)", line)
        if mo:
            measured[int(mo.group(1))] = int(mo.group(2))
    if "TIMEOUT" in stdout:
        raise RuntimeError(f"timeout:\n{stdout[-1500:]}")
    return measured, expected


def main() -> int:
    cases = [
        # (n, m, k_bits)
        (8, 4, 4),
        (16, 8, 8),
        (32, 8, 8),
        (64, 16, 8),
    ]
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="input_axis_") as td:
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
