"""Bit-exact Verilator validation of cirom_affine_binary_beh, the
row-granularity affine-binary REFERENCE model (NOT signed-off RTL --
see docs/binary_and_matmulfree_investigation.md and the module's own
header comment). Compares against
ankhdjet.reference.nor.binary_affine_matmul_nor on random sign masks
and per-column Q8.8 scale/bias at multiple shapes, the same way
test_cirom_nor_tile.py validates the ternary tile.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ankhdjet.reference.nor import binary_affine_matmul_nor
from tests._verilator_runner import build_and_run

TILE_SV = REPO_ROOT / "rtl" / "column" / "cirom_nor_tile.sv"
BEH_SV = REPO_ROOT / "rtl" / "column" / "cirom_affine_binary_beh.sv"


TB = r"""
`timescale 1ns/1ps
module tb_affine;
    localparam int N = {N};
    localparam int M = {M};
    localparam int K = {K};
    localparam int SUBCOL = {SUBCOL};
    localparam int WACC = {WACC};

    logic clk, rst_n, start, valid;
    logic [K*N-1:0]       act_flat;
    logic [K-1:0]         act_byte  [0:N-1];
    logic signed [15:0]   scale_q88 [0:M-1];
    logic signed [15:0]   bias_q88  [0:M-1];
    logic signed [63:0]   result    [0:M-1];

    cirom_affine_binary_beh #(
        .N(N), .M(M), .K(K), .SUBCOL_ROWS(SUBCOL), .WACC(WACC),
        .HAS_VIA_POS({POS_LIT})
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .act_flat(act_flat),
        .scale_q88(scale_q88), .bias_q88(bias_q88),
        .result(result), .valid(valid)
    );

    always #5 clk = ~clk;

    int i, j, cycles;

    initial begin
        $readmemh("act.hex", act_byte);
        $readmemh("scale.hex", scale_q88);
        $readmemh("bias.hex", bias_q88);
        for (i = 0; i < N; i = i + 1) begin
            act_flat[K*(i+1)-1 -: K] = act_byte[i];
        end
        clk = 0; rst_n = 0; start = 0;
        #23 rst_n = 1;
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;
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


def _bits_literal(bits: np.ndarray) -> str:
    n = bits.size
    s = "".join("1" if int(bits[n - 1 - i]) else "0" for i in range(n))
    return f"{n}'b{s}"


def _write_hex_rows(values: np.ndarray, path: Path, width_hex: int) -> None:
    path.write_text(
        "\n".join(f"{int(v) & ((1 << (4 * width_hex)) - 1):0{width_hex}x}"
                  for v in values) + "\n")


def run_case(workdir: Path, n: int, m: int, k_bits: int,
              subcol: int, seed: int = 0xB16) -> tuple[np.ndarray, int, np.ndarray]:
    rng = np.random.default_rng(seed)
    sign = rng.integers(0, 2, size=(n, m)).astype(np.int64)
    act = rng.integers(0, 1 << k_bits, size=n, dtype=np.int64)
    scale_q88 = rng.integers(-2000, 2000, size=m, dtype=np.int64)
    bias_q88 = rng.integers(-2000, 2000, size=m, dtype=np.int64)

    expected = binary_affine_matmul_nor(
        sign, act, scale_q88, bias_q88, k_bits=k_bits, subcol_rows=subcol)

    pos_bits = sign.flatten()  # w_idx = row*M + c, matching cirom_nor_tile

    wacc = max(8, int(np.ceil(np.log2((n * (1 << k_bits)) + 1))) + 1)
    tb_src = TB.format(
        N=n, M=m, K=k_bits, SUBCOL=subcol, WACC=wacc,
        POS_LIT=_bits_literal(pos_bits),
    )
    (workdir / "tb.sv").write_text(tb_src)
    _write_hex_rows(act, workdir / "act.hex", width_hex=max(1, (k_bits + 3) // 4))
    _write_hex_rows(scale_q88, workdir / "scale.hex", width_hex=4)
    _write_hex_rows(bias_q88, workdir / "bias.hex", width_hex=4)

    stdout = build_and_run(
        workdir=workdir, sources=[TILE_SV, BEH_SV, workdir / "tb.sv"],
        top="tb_affine", build_timeout=300.0, run_timeout=300.0,
    )

    measured = np.zeros(m, dtype=np.int64)
    cycles = -1
    for line in stdout.splitlines():
        mo = re.match(r"\s*OUT\[(\d+)\]=(-?\d+)", line)
        if mo:
            measured[int(mo.group(1))] = int(mo.group(2))
        mc = re.match(r"\s*CYCLES=(\d+)", line)
        if mc:
            cycles = int(mc.group(1))
    if "TIMEOUT" in stdout:
        raise RuntimeError(f"timeout:\n{stdout[-1500:]}")
    return measured, cycles, expected


def main() -> int:
    cases = [
        # (n, m, k_bits, subcol)
        (16, 4, 4, 16),
        (32, 8, 8, 16),
        (64, 8, 8, 32),
        (128, 16, 8, 64),
    ]
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="affine_") as td:
        work = Path(td)
        for idx, (n, m, k, sub) in enumerate(cases):
            measured, cycles, expected = run_case(work, n, m, k, sub)
            ok = np.array_equal(measured, expected)
            print(f"[{'ok' if ok else 'FAIL'}] case {idx}  N={n} M={m} K={k} "
                  f"SUBCOL={sub}  cycles={cycles}  match={int((measured==expected).sum())}/{m}")
            if not ok:
                mis = np.where(measured != expected)[0]
                for j in mis[:3]:
                    print(f"    out[{j}]: exp={expected[j]} got={measured[j]}")
                all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
