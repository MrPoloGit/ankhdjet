// Behavioral REFERENCE model only, for the row-granularity affine-binary
// case (FBI-LLM's folded BinaryLinearWscales: one scale+bias per OUTPUT
// COLUMN, not per element -- see
// docs/binary_and_matmulfree_investigation.md). This module is NOT part
// of the signed-off chip/macro RTL (rtl/chip, rtl/grid), is not wired
// into cirom_chip_digital/analog, and has not been through layout, DRC,
// or LVS. It exists to show what cirom_nor_tile.sv is missing for this
// case and where the real cost lands:
//
//   dot[c] = sum_r act[r] * (bias[c] + sign_pm[r,c] * scale[c])
//          = bias[c] * sum_r(act[r])
//            + scale[c] * sum_r(sign_pm[r,c] * act[r])
//
// `sign_pm` (+-1 from the {0,1} sign bit) never floats a drain -- every
// position routes to BL+ (sign=1) or BL- (sign=0), a strict subset of
// the ternary mask vocabulary cirom_nor_tile.sv already implements, so
// it is reused here completely unmodified (HAS_VIA_NEG is just the
// complement of HAS_VIA_POS). `sum_r(sign_pm*act)` is exactly that
// tile's existing hit-count accumulate (acc_flat). The only new
// hardware is `sum_r(act[r])`, ONE array-wide running total shared by
// every column (act_total below), and the final affine combine, which
// replaces today's single shared per-tensor Q8.8 scale multiply
// (docs/array_architecture.md's "shared scale multiplier" density
// note) with M independent Q8.8 multiply-adds -- the real, quantified
// periphery cost of this case.
`default_nettype none

module cirom_affine_binary_beh #(
    parameter int          N           = 256,
    parameter int          M           = 64,
    parameter int          K           = 8,
    parameter int          SUBCOL_ROWS = 128,
    parameter int          WACC        = 24,
    parameter [N*M-1:0]    HAS_VIA_POS = {(N*M){1'b0}}  // sign=1 -> BL+, sign=0 -> BL-
) (
    input  logic                      clk,
    input  logic                      rst_n,
    input  logic                      start,
    input  logic [K*N-1:0]            act_flat,
    input  logic signed [15:0]        scale_q88 [M],   // per-column Q8.8
    input  logic signed [15:0]        bias_q88  [M],   // per-column Q8.8
    output logic signed [63:0]        result [M],
    output logic                      valid
);
    localparam [N*M-1:0] HAS_VIA_NEG = ~HAS_VIA_POS;

    logic signed [WACC*M-1:0] acc_flat;
    logic                     hit_valid;

    // Reused unmodified: the hit-count accumulate is identical to the
    // ternary tile's, since sign never floats a drain.
    cirom_nor_tile #(
        .N(N), .M(M), .K(K), .SUBCOL_ROWS(SUBCOL_ROWS), .WACC(WACC),
        .HAS_VIA_POS(HAS_VIA_POS), .HAS_VIA_NEG(HAS_VIA_NEG)
    ) hit_tile (
        .clk(clk), .rst_n(rst_n), .start(start),
        .act_flat(act_flat), .acc_flat(acc_flat), .valid(hit_valid)
    );

    // Mirrors cirom_nor_tile's own K*SUBCOL_ROWS timing (duplicated
    // rather than extracted from it, to avoid touching the signed-off
    // module) so act_total lands on the same cycle hit_valid does.
    localparam int N_SUB = (N + SUBCOL_ROWS - 1) / SUBCOL_ROWS;
    localparam int K_W   = $clog2(K + 1);
    localparam int R_W   = $clog2(SUBCOL_ROWS + 1);
    logic [K_W-1:0]  k_idx;
    logic [R_W-1:0]  r_idx;
    logic            running;
    logic [K*N-1:0]  act_latched;
    logic [WACC-1:0] act_total;

    integer s2;
    int row2, bit_idx2;
    logic [WACC-1:0] act_hit;
    always_comb begin
        act_hit = '0;
        row2 = 0; bit_idx2 = 0;
        for (s2 = 0; s2 < N_SUB; s2++) begin
            row2 = s2 * SUBCOL_ROWS + r_idx;
            if (row2 < N) begin
                bit_idx2 = K * row2 + k_idx;
                if (act_latched[bit_idx2]) act_hit = act_hit + 1;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            k_idx <= '0; r_idx <= '0; running <= 1'b0;
            act_latched <= '0; act_total <= '0;
        end else begin
            if (start) begin
                act_latched <= act_flat;
                k_idx <= '0; r_idx <= '0; running <= 1'b1; act_total <= '0;
            end else if (running) begin
                act_total <= act_total + (act_hit <<< k_idx);
                if (r_idx == SUBCOL_ROWS - 1) begin
                    r_idx <= '0;
                    if (k_idx == K - 1) running <= 1'b0;
                    else k_idx <= k_idx + 1'b1;
                end else r_idx <= r_idx + 1'b1;
            end
        end
    end

    // Final affine combine: M independent Q8.8 multiply-adds. Widths
    // are generous (64-bit intermediates) since this is a correctness
    // reference, not an area-optimized design -- a to-silicon version
    // would size this per the real WACC/activation range.
    genvar gc;
    generate
        for (gc = 0; gc < M; gc++) begin : g_affine
            logic signed [63:0] bias_term, scale_term;
            always_comb begin
                bias_term  = (64'(bias_q88[gc])
                              * 64'($signed({1'b0, act_total}))) >>> 8;
                scale_term = (64'(scale_q88[gc])
                              * 64'($signed(acc_flat[WACC*(gc+1)-1 -: WACC]))) >>> 8;
                result[gc] = bias_term + scale_term;
            end
        end
    endgenerate

    assign valid = hit_valid;

endmodule

`default_nettype wire
