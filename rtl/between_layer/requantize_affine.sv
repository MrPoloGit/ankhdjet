// Per-channel affine requantize: the row-granularity affine-binary
// counterpart of requantize.sv (see
// docs/binary_and_matmulfree_investigation.md). NOT wired into any
// signed-off chip/macro RTL.
//
// Datapath (combinational):
//   product = hit * SCALE_Q + act_total * BIAS_Q   (both signed Q-format)
//   shifted = product >>> Q_FRAC
//   if ACTIVATION==0 (relu): shifted = max(shifted, 0)
//   out     = clip(shifted, 0, 2**K - 1)
//
// `hit` is cirom_nor_tile's existing per-column accumulate (unchanged);
// `act_total` is the one new array-wide accumulator
// (cirom_act_total_beh.sv), shared across every channel's instance of
// this module -- the only per-array (not per-channel) new hardware the
// row-granularity case needs. SCALE_Q/BIAS_Q are per-channel Q8.8
// (signed, since neither needs to be positive), in place of
// requantize.sv's single shared SCALE_Q.

`default_nettype none

module requantize_affine #(
    parameter int WACC       = 24,
    parameter int SCALE_W    = 16,
    parameter int Q_FRAC     = 8,
    parameter int K          = 8,
    parameter signed [SCALE_W-1:0] SCALE_Q = {{(SCALE_W-1){1'b0}}, 1'b1} << Q_FRAC,
    parameter signed [SCALE_W-1:0] BIAS_Q  = '0,
    parameter int ACTIVATION = 0
)(
    input  logic signed [WACC-1:0] hit,
    input  logic        [WACC-1:0] act_total,   // unsigned
    output logic [K-1:0]           out
);
    localparam int MAX_UNSIGNED = (1 << K) - 1;

    // 64-bit intermediates: this is a correctness reference, not an
    // area-optimized design -- a to-silicon version would size these
    // per the real WACC/SCALE_W range (see requantize.sv for that
    // narrower, shared-scale style).
    logic signed [63:0] scale_term, bias_term, product, shifted;
    logic signed [63:0] activated, clipped;

    always_comb begin
        scale_term = 64'(hit) * 64'(SCALE_Q);
        bias_term  = 64'($signed({1'b0, act_total})) * 64'(BIAS_Q);
        product    = scale_term + bias_term;
        shifted    = product >>> Q_FRAC;

        if (ACTIVATION == 0 && shifted < 0)
            activated = '0;
        else
            activated = shifted;

        if (activated > 64'(MAX_UNSIGNED))
            clipped = 64'(MAX_UNSIGNED);
        else if (activated < 64'sd0)
            clipped = 64'sd0;
        else
            clipped = activated;
    end

    assign out = clipped[K-1:0];

endmodule

`default_nettype wire
