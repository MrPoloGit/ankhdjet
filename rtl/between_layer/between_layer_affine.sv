// M-channel affine between-layer block: the row-granularity
// affine-binary counterpart of between_layer.sv (see
// docs/binary_and_matmulfree_investigation.md). NOT wired into any
// signed-off chip/macro RTL.
//
// Where between_layer.sv applies one SCALE_Q shared by every channel,
// this applies a PER-CHANNEL SCALE_Q/BIAS_Q (flat Q8.8 literals, same
// convention as HAS_VIA_POS elsewhere in this codebase) to a shared
// act_total input -- the real, quantified periphery cost of the
// row-granularity affine-binary case: M independent multiply-adds in
// place of one shared multiply.
//
// Bus layout:
//   hit_flat: channel j at bits [WACC*(j+1)-1 -: WACC], signed
//   out_flat: channel j at bits [K*(j+1)-1 -: K],       unsigned
//   SCALE_Q_FLAT/BIAS_Q_FLAT: channel j at bits [SCALE_W*(j+1)-1 -: SCALE_W]

`default_nettype none

module between_layer_affine #(
    parameter int M          = 16,
    parameter int WACC       = 24,
    parameter int SCALE_W    = 16,
    parameter int Q_FRAC     = 8,
    parameter int K          = 8,
    parameter [M*SCALE_W-1:0] SCALE_Q_FLAT = '0,
    parameter [M*SCALE_W-1:0] BIAS_Q_FLAT  = '0,
    parameter int ACTIVATION = 0
)(
    input  logic [WACC*M-1:0] hit_flat,
    input  logic [WACC-1:0]   act_total,
    output logic [K*M-1:0]    out_flat
);
    genvar j;
    generate
        for (j = 0; j < M; j = j + 1) begin : ch
            logic signed [WACC-1:0] hit_j;
            logic [K-1:0]           out_j;
            assign hit_j = $signed(hit_flat[WACC*(j+1)-1 -: WACC]);
            requantize_affine #(
                .WACC      (WACC),
                .SCALE_W   (SCALE_W),
                .Q_FRAC    (Q_FRAC),
                .K         (K),
                .SCALE_Q   ($signed(SCALE_Q_FLAT[SCALE_W*(j+1)-1 -: SCALE_W])),
                .BIAS_Q    ($signed(BIAS_Q_FLAT [SCALE_W*(j+1)-1 -: SCALE_W])),
                .ACTIVATION(ACTIVATION)
            ) u_rq (
                .hit(hit_j),
                .act_total(act_total),
                .out(out_j)
            );
            assign out_flat[K*(j+1)-1 -: K] = out_j;
        end
    endgenerate
endmodule

`default_nettype wire
