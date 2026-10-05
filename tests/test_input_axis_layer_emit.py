"""Bit-exact Verilator validation of
ankhdjet.backend.verilog.emit_layer_input_axis_affine: a real IR layer
(scale_axis="input") wired to cirom_input_axis_affine_beh, followed by
a unit-scale requantize per column for the relu+clip activation stage.
NOT signed-off RTL (see docs/binary_and_matmulfree_investigation.md).

Oracle: ankhdjet.reference.nor.input_axis_affine_matmul, then relu +
clip to out_k_bits manually (what a unit-scale requantize does).
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ankhdjet.backend.verilog import emit_layer_input_axis_affine, quantize_q88
from ankhdjet.frontend.ir import Layer, LayerType, QuantScheme, WeightTensor
from ankhdjet.reference.nor import input_axis_affine_matmul
from tests._verilator_runner import build_and_run

BEH_SV = REPO_ROOT / "rtl" / "column" / "cirom_input_axis_affine_beh.sv"
REQ_SV = REPO_ROOT / "rtl" / "between_layer" / "requantize.sv"


TB = r"""
`timescale 1ns/1ps
module tb_input_axis_layer;
    localparam int N    = {N};
    localparam int M    = {M};
    localparam int K    = {K};
    localparam int KOUT = 8;

    logic clk, rst_n, start, valid;
    logic [K-1:0]    act      [0:N-1];
    logic [KOUT*M-1:0] out_flat;

    ankhdjet_layer_{LID} u_layer (
        .clk(clk), .rst_n(rst_n), .start(start),
        .act(act), .out_flat(out_flat), .valid(valid)
    );

    always #5 clk = ~clk;
    int i, j, cycles;

    initial begin
        $readmemh("act.hex", act);
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


def _write_act_hex(activations: np.ndarray, path: Path, k_bits: int) -> None:
    width_hex = max(1, (k_bits + 3) // 4)
    path.write_text(
        "\n".join(f"{int(a):0{width_hex}x}" for a in activations) + "\n")


def run_case(workdir: Path, n: int, m: int, k_bits: int,
              seed: int = 0xABCD) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sign = rng.integers(0, 2, size=(n, m)).astype(np.int8)
    act = rng.integers(0, 1 << k_bits, size=n, dtype=np.int64)
    scale_f = rng.uniform(-0.1, 0.1, size=n)
    bias_f = rng.uniform(-0.05, 0.05, size=n)
    scale_q88 = np.array([quantize_q88(x) for x in scale_f], dtype=np.int64)
    bias_q88 = np.array([quantize_q88(x) for x in bias_f], dtype=np.int64)

    layer = Layer(
        name="ia0", layer_type=LayerType.LINEAR,
        weights={"weight": WeightTensor(
            name="weight", data=sign, scheme=QuantScheme.BINARY,
            scale=scale_f.reshape(n, 1).astype(np.float32),
            bias=bias_f.reshape(n, 1).astype(np.float32),
            scale_axis="input")},
        input_dim=n, output_dim=m,
    )
    rtl_src = emit_layer_input_axis_affine(layer, k_bits=k_bits)

    lid = "ia0"
    tb_src = TB.format(N=n, M=m, K=k_bits, LID=lid)
    (workdir / "layer.sv").write_text(rtl_src)
    (workdir / "tb.sv").write_text(tb_src)
    _write_act_hex(act, workdir / "act.hex", k_bits)

    stdout = build_and_run(
        workdir=workdir, sources=[BEH_SV, REQ_SV, workdir / "layer.sv", workdir / "tb.sv"],
        top="tb_input_axis_layer", build_timeout=300.0, run_timeout=300.0,
    )

    measured = np.zeros(m, dtype=np.int64)
    for line in stdout.splitlines():
        mo = re.match(r"\s*OUT\[(\d+)\]=(-?\d+)", line)
        if mo:
            measured[int(mo.group(1))] = int(mo.group(2))
    if "TIMEOUT" in stdout:
        raise RuntimeError(f"timeout:\n{stdout[-1500:]}")

    raw = input_axis_affine_matmul(sign, act, scale_q88, bias_q88, q_frac=8)
    expected = np.clip(np.where(raw < 0, 0, raw), 0, 255).astype(np.int64)
    return measured, expected


def main() -> int:
    cases = [
        # (n, m, k_bits)
        (8, 4, 4),
        (16, 8, 8),
        (32, 8, 8),
    ]
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="input_axis_layer_") as td:
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
