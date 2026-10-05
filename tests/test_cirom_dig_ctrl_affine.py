"""Chip/tile-level functional regression for cirom_dig_ctrl_affine (the
synthesis-quality INPUT-AXIS affine-binary tile controller -- see
docs/binary_and_matmulfree_investigation.md and the module's own
header comment), mirroring rtl/chip/sim/run_sim.sh's rigor: the
controller driving a real array interface (cirom_array_beh, the same
behavioral macro stand-in the signed-off chip bench uses), compared
bit-exact against ankhdjet.reference.nor.input_axis_affine_matmul.

Unlike every other RTL file built earlier in this investigation
(cirom_input_axis_affine_beh.sv and friends: 64-bit casts, unpacked
array ports, explicitly NOT synthesis-tuned), this one is modeled
directly on the signed-off cirom_dig_ctrl.sv and is meant to actually
go through synthesis.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ankhdjet.backend.verilog import quantize_q88
from ankhdjet.reference.nor import input_axis_affine_matmul
from tests._verilator_runner import build_and_run

CTRL_SV = REPO_ROOT / "rtl" / "tt_digital" / "cirom_dig_ctrl_affine.sv"
ARR_SV = REPO_ROOT / "rtl" / "grid" / "cirom_array_beh.sv"


TB = r"""
`timescale 1ns/1ps
module tb_affine_ctrl;
    localparam int N_ROWS  = {N_ROWS};
    localparam int N_COLS  = {N_COLS};
    localparam int N_ACC   = {N_ACC};
    localparam int ACC_W   = {ACC_W};
    localparam int ACT_W   = 8;
    localparam int SCALE_W = 16;
    localparam int OUT_W   = 8;
    localparam int CFG_BITS = 16;
    localparam int GRPB    = ACC_W / OUT_W;
    localparam int PASSES  = N_COLS / N_ACC;
    localparam int NBYTES  = N_ACC * GRPB * PASSES;

    logic clk = 0, rst_n = 0;
    logic [7:0] ui = '0;
    logic act_wr = 0, start = 0, cfg_mode = 0, cfg_in = 0;
    wire  [N_ROWS-1:0] wl;
    wire  pre_n;
    wire  [N_COLS-1:0] blp, bln;
    wire  [OUT_W-1:0] result;
    wire  result_valid, busy, done;

    cirom_dig_ctrl_affine #(
        .N_ROWS(N_ROWS), .N_COLS(N_COLS), .N_ACC(N_ACC), .ACC_W(ACC_W),
        .ACT_W(ACT_W), .SCALE_W(SCALE_W), .OUT_W(OUT_W), .CFG_BITS(CFG_BITS),
        .SCALE_Q({SCALE_LIT}), .BIAS_Q({BIAS_LIT})
    ) dut (
        .clk(clk), .rst_n(rst_n), .ui(ui), .act_wr(act_wr), .start(start),
        .cfg_mode(cfg_mode), .cfg_in(cfg_in),
        .wl(wl), .pre_n(pre_n), .blp(blp), .bln(bln),
        .result(result), .result_valid(result_valid), .busy(busy), .done(done)
    );

    cirom_array_beh #(
        .MR(N_ROWS), .MC(N_COLS), .WPOS("wpos.memh"), .WNEG("wneg.memh")
    ) arr (
        .wl(wl), .pre_n(pre_n), .blp(blp), .bln(bln)
    );

    always #5 clk = ~clk;

    logic [7:0] act_byte [0:N_ROWS-1];
    logic [7:0] rx [0:NBYTES-1];
    int i, k, cycles;

    task automatic cfg_load(input [15:0] word);
        for (int b = 15; b >= 0; b--) begin
            cfg_mode <= 1'b1;
            cfg_in   <= word[b];
            @(posedge clk);
        end
        cfg_mode <= 1'b0; cfg_in <= 1'b0;
        @(posedge clk);
    endtask

    initial begin
        $readmemh("act.hex", act_byte);
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);
        cfg_load(16'h0022);   // strobe_delay=2, pre_width=2, bypass_pre=0

        for (i = 0; i < N_ROWS; i = i + 1) begin
            ui     <= act_byte[i];
            act_wr <= 1'b1;
            @(posedge clk);
        end
        act_wr <= 1'b0;
        @(posedge clk);

        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        k = 0; cycles = 0;
        while (k < NBYTES && cycles < 2000000) begin
            @(posedge clk);
            if (result_valid) begin rx[k] = result; k = k + 1; end
            cycles = cycles + 1;
        end
        if (k < NBYTES) begin $display("TIMEOUT"); $finish; end
        wait (done);

        for (i = 0; i < NBYTES / GRPB; i = i + 1) begin
            logic signed [ACC_W-1:0] v;
            for (int bb = 0; bb < GRPB; bb = bb + 1)
                v[bb*OUT_W +: OUT_W] = rx[i*GRPB + bb];
            $display("OUT[%0d]=%0d", i, v);
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


def _write_array_memh(sign: np.ndarray, workdir: Path) -> None:
    n, m = sign.shape
    width_hex = (m + 3) // 4
    pos_lines, neg_lines = [], []
    for r in range(n):
        p = n_ = 0
        for c in range(m):
            if sign[r, c] == 1:
                p |= 1 << c
            else:
                n_ |= 1 << c
        pos_lines.append(f"{p:0{width_hex}x}")
        neg_lines.append(f"{n_:0{width_hex}x}")
    (workdir / "wpos.memh").write_text("\n".join(pos_lines) + "\n")
    (workdir / "wneg.memh").write_text("\n".join(neg_lines) + "\n")


def run_case(workdir: Path, n_rows: int, n_cols: int, n_acc: int,
              seed: int = 0x1234) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sign = rng.integers(0, 2, size=(n_rows, n_cols)).astype(np.int8)
    act = rng.integers(0, 256, size=n_rows, dtype=np.int64)
    scale_f = rng.uniform(-0.08, 0.08, size=n_rows)
    bias_f = rng.uniform(-0.03, 0.03, size=n_rows)
    scale_q88 = np.array([quantize_q88(x) for x in scale_f], dtype=np.int64)
    bias_q88 = np.array([quantize_q88(x) for x in bias_f], dtype=np.int64)

    acc_w = 32
    tb_src = TB.format(
        N_ROWS=n_rows, N_COLS=n_cols, N_ACC=n_acc, ACC_W=acc_w,
        SCALE_LIT=_flat_q88_literal(scale_q88),
        BIAS_LIT=_flat_q88_literal(bias_q88),
    )
    (workdir / "tb.sv").write_text(tb_src)
    _write_array_memh(sign, workdir)
    width_hex = 2
    (workdir / "act.hex").write_text(
        "\n".join(f"{int(a):0{width_hex}x}" for a in act) + "\n")

    stdout = build_and_run(
        workdir=workdir, sources=[CTRL_SV, ARR_SV, workdir / "tb.sv"],
        top="tb_affine_ctrl", build_timeout=300.0, run_timeout=300.0,
    )
    if "TIMEOUT" in stdout:
        raise RuntimeError(f"timeout:\n{stdout[-1500:]}")
    measured = np.zeros(n_cols, dtype=np.int64)
    for line in stdout.splitlines():
        mo = re.match(r"\s*OUT\[(\d+)\]=(-?\d+)", line)
        if mo:
            measured[int(mo.group(1))] = int(mo.group(2))

    expected = input_axis_affine_matmul(sign, act, scale_q88, bias_q88, q_frac=0)
    # q_frac=0: the tile streams the RAW pre-shift accumulate (see the
    # module's header); the Q_FRAC shift is a downstream concern.
    return measured, expected


def main() -> int:
    cases = [
        # (n_rows, n_cols, n_acc)
        (16, 8, 8),     # single pass
        (16, 8, 4),     # two passes
        (32, 12, 4),    # three passes, odd column count
    ]
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="dig_ctrl_affine_") as td:
        work = Path(td)
        for idx, (n_rows, n_cols, n_acc) in enumerate(cases):
            measured, expected = run_case(work, n_rows, n_cols, n_acc)
            ok = np.array_equal(measured, expected)
            print(f"[{'ok' if ok else 'FAIL'}] case {idx}  "
                  f"N_ROWS={n_rows} N_COLS={n_cols} N_ACC={n_acc} "
                  f"match={int((measured==expected).sum())}/{n_cols}")
            if not ok:
                mis = np.where(measured != expected)[0]
                for j in mis[:5]:
                    print(f"    out[{j}]: exp={expected[j]} got={measured[j]}")
                all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
