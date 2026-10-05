"""Bit-exact Verilator validation of
ankhdjet.backend.verilog.emit_pipeline_input_axis: a 3-layer chain of
input-axis affine-binary layers (see
docs/binary_and_matmulfree_investigation.md), each with its own real
per-row scale/bias, through the valid->start handshake between them.
NOT signed-off RTL.

Oracle: chain input_axis_affine_matmul -> relu+clip (what each layer's
internal unit-scale requantize does) -> feed as the next layer's
activations, matching test_pipeline_emit.py's style for the ternary
case.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ankhdjet.backend.verilog import emit_pipeline_input_axis, quantize_q88
from ankhdjet.frontend.ir import Layer, LayerType, QuantScheme, WeightTensor
from ankhdjet.reference.nor import input_axis_affine_matmul
from tests._verilator_runner import build_and_run

BEH_SV = REPO_ROOT / "rtl" / "column" / "cirom_input_axis_affine_beh.sv"
REQ_SV = REPO_ROOT / "rtl" / "between_layer" / "requantize.sv"


TB = r"""
`timescale 1ns/1ps
module tb_input_axis_pipeline;
    localparam int N0    = {N0};
    localparam int MLAST = {MLAST};
    localparam int K     = {K};

    logic clk, rst_n, start, valid;
    logic [K-1:0]      act      [0:N0-1];
    logic [K*MLAST-1:0] out_flat;

    ankhdjet_pipeline_{NAME} dut (
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
        for (j = 0; j < MLAST; j = j + 1) begin
            $display("OUT[%0d]=%0d", j, out_flat[K*(j+1)-1 -: K]);
        end
        $display("CYCLES=%0d", cycles);
        $finish;
    end
endmodule
"""


def _make_layer(name: str, n: int, m: int, k_bits: int, seed: int) -> tuple[Layer, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sign = rng.integers(0, 2, size=(n, m)).astype(np.int8)
    scale_f = rng.uniform(-0.05, 0.05, size=n)
    bias_f = rng.uniform(-0.02, 0.02, size=n)
    layer = Layer(
        name=name, layer_type=LayerType.LINEAR,
        weights={"weight": WeightTensor(
            name="weight", data=sign, scheme=QuantScheme.BINARY,
            scale=scale_f.reshape(n, 1).astype(np.float32),
            bias=bias_f.reshape(n, 1).astype(np.float32),
            scale_axis="input")},
        input_dim=n, output_dim=m,
    )
    scale_q88 = np.array([quantize_q88(x) for x in scale_f], dtype=np.int64)
    bias_q88 = np.array([quantize_q88(x) for x in bias_f], dtype=np.int64)
    return layer, scale_q88, bias_q88


def run_case(workdir: Path, dims: list[tuple[int, int]], k_bits: int,
              seed: int = 0xF00D) -> tuple[np.ndarray, np.ndarray]:
    layers = []
    signs = []
    scale_q88s = []
    bias_q88s = []
    for i, (n, m) in enumerate(dims):
        L, sq, bq = _make_layer(f"pl{i}", n, m, k_bits, seed + i)
        layers.append(L)
        signs.append(L.weights["weight"].data)
        scale_q88s.append(sq)
        bias_q88s.append(bq)

    rng = np.random.default_rng(seed + 100)
    n0, _ = dims[0]
    act = rng.integers(0, 1 << k_bits, size=n0, dtype=np.int64)

    name = "chain"
    rtl_src = emit_pipeline_input_axis(layers, name=name, k_bits=k_bits)
    (workdir / "pipeline.sv").write_text(rtl_src)
    _, m_last = dims[-1]
    tb_src = TB.format(N0=n0, MLAST=m_last, K=k_bits, NAME=name)
    (workdir / "tb.sv").write_text(tb_src)
    width_hex = max(1, (k_bits + 3) // 4)
    (workdir / "act.hex").write_text(
        "\n".join(f"{int(a):0{width_hex}x}" for a in act) + "\n")

    stdout = build_and_run(
        workdir=workdir,
        sources=[BEH_SV, REQ_SV, workdir / "pipeline.sv", workdir / "tb.sv"],
        top="tb_input_axis_pipeline", build_timeout=300.0, run_timeout=300.0,
    )
    if "TIMEOUT" in stdout:
        raise RuntimeError(f"timeout:\n{stdout[-1500:]}")
    measured = np.zeros(m_last, dtype=np.int64)
    for line in stdout.splitlines():
        mo = re.match(r"\s*OUT\[(\d+)\]=(-?\d+)", line)
        if mo:
            measured[int(mo.group(1))] = int(mo.group(2))

    cur_act = act
    for sign, sq, bq in zip(signs, scale_q88s, bias_q88s):
        raw = input_axis_affine_matmul(sign, cur_act, sq, bq, q_frac=8)
        cur_act = np.clip(np.where(raw < 0, 0, raw), 0, (1 << k_bits) - 1).astype(np.int64)
    return measured, cur_act


def main() -> int:
    cases = [
        [(8, 6), (6, 4)],
        [(16, 12), (12, 8), (8, 5)],
    ]
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="input_axis_pipeline_") as td:
        work = Path(td)
        for idx, dims in enumerate(cases):
            measured, expected = run_case(work, dims, k_bits=8)
            ok = np.array_equal(measured, expected)
            print(f"[{'ok' if ok else 'FAIL'}] case {idx}  dims={dims} "
                  f"match={int((measured==expected).sum())}/{len(expected)}")
            if not ok:
                mis = np.where(measured != expected)[0]
                for j in mis[:3]:
                    print(f"    out[{j}]: exp={expected[j]} got={measured[j]}")
                all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
