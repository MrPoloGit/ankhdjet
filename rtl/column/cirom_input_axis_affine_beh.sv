// Behavioral REFERENCE model only, for the INPUT-AXIS affine-binary
// case: scale/bias vary per INPUT ROW (activation index), broadcast
// over every output column -- FBI-LLM's actual released convention,
// confirmed against a real checkpoint (LiqunMa/FBI-LLM_130M). This is
// the OPPOSITE axis from cirom_affine_binary_beh.sv's row/per-tensor
// case (which was built on an assumption the real release does not
// use -- see docs/binary_and_matmulfree_investigation.md). NOT part of
// the signed-off chip/macro RTL; not through layout, DRC, or LVS.
//
//   dot[c] = sum_r act[r] * (bias[r] + sign_pm[r,c] * scale[r])
//          = sum_r (act[r] * bias[r])
//            + sum_r sign_pm[r,c] * (act[r] * scale[r])
//
// Both terms need scale[r]/bias[r] multiplied into the per-row
// activation BEFORE the row sweep's accumulate -- this is the real
// architectural cost of this axis, and it is materially worse than
// the output-axis case: that case needed one array-wide shared
// accumulator plus M final multiplies. This axis needs the per-row
// scale multiply on the activation CRITICAL PATH itself (N multiplies
// per dot product, not M), and cirom_nor_tile's cheap bit-serial
// AND-based hit count no longer applies, since the thing being
// accumulated is a wide signed product, not a single activation bit.
// This reference therefore does NOT preserve the K-cycle bit-serial
// convention the rest of this codebase uses: it sweeps one row per
// cycle (N cycles total), establishing correctness of the math, not a
// cycle-accurate or area-accounted design. A real implementation would
// still need to decide how (or whether) to recover bit-serial
// activation handling for this axis -- that question is open.
`default_nettype none

module cirom_input_axis_affine_beh #(
    parameter int       N         = 64,
    parameter int       M         = 8,
    parameter int       K         = 8,
    parameter int       SCALE_W   = 16,
    parameter int       Q_FRAC    = 8,
    parameter int       WACC      = 48,
    // Mask-programmed sign, same convention as cirom_nor_tile's
    // HAS_VIA_POS: compile-time, not a runtime signal, and passed as a
    // parameter (not a port) because Verilator's literal-width limit
    // for a bare port-connection expression is far lower than for a
    // parameter override -- real multi-hundred-thousand-bit real-weight
    // layers (see tests/test_fbi_llm_real_layer_rtl.py) need the
    // parameter path, same as cirom_nor_tile's real-weight tests do.
    parameter [N*M-1:0] HAS_VIA_POS = {(N*M){1'b0}}   // row r, col c: bit r*M+c, 1 -> BL+
)(
    input  logic                       clk,
    input  logic                       rst_n,
    input  logic                       start,
    input  logic [K-1:0]               act      [N],   // unsigned per-row activation
    input  logic signed [SCALE_W-1:0]  scale_q  [N],   // per-row Q8.8
    input  logic signed [SCALE_W-1:0]  bias_q   [N],   // per-row Q8.8
    output logic signed [WACC-1:0]     result   [M],
    output logic                       valid
);
    localparam int R_W = $clog2(N + 1);

    logic [R_W-1:0]         r_idx;
    logic                   running;
    logic signed [WACC-1:0] hit [M];
    logic signed [WACC-1:0] bias_acc;

    integer c;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_idx <= '0; running <= 1'b0; valid <= 1'b0; bias_acc <= '0;
            for (c = 0; c < M; c++) hit[c] <= '0;
        end else begin
            valid <= 1'b0;
            if (start) begin
                r_idx <= '0; running <= 1'b1; bias_acc <= '0;
                for (c = 0; c < M; c++) hit[c] <= '0;
            end else if (running) begin
                bias_acc <= bias_acc + WACC'(64'(act[r_idx]) * 64'(bias_q[r_idx]));
                for (c = 0; c < M; c++) begin
                    if (HAS_VIA_POS[r_idx * M + c])
                        hit[c] <= hit[c] + WACC'(64'(act[r_idx]) * 64'(scale_q[r_idx]));
                    else
                        hit[c] <= hit[c] - WACC'(64'(act[r_idx]) * 64'(scale_q[r_idx]));
                end
                if (r_idx == N[R_W-1:0] - 1'b1) begin
                    running <= 1'b0;
                    valid   <= 1'b1;
                end
                r_idx <= r_idx + 1'b1;
            end
        end
    end

    genvar gc;
    generate
        for (gc = 0; gc < M; gc++) begin : g_out
            assign result[gc] = (hit[gc] + bias_acc) >>> Q_FRAC;
        end
    endgenerate

endmodule

`default_nettype wire
