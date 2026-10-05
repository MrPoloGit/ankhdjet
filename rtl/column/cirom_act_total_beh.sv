// Array-wide running activation total, timed to match cirom_nor_tile's
// own K*SUBCOL_ROWS sweep exactly (same N/K/SUBCOL_ROWS parameters,
// same start/valid handshake) so the two can be instantiated side by
// side and their outputs combined downstream on the same cycle.
//
// This is the one new per-array accumulator the row-granularity
// affine-binary case needs (see
// docs/binary_and_matmulfree_investigation.md): shared by every output
// column, not per-column, since dot[c] = bias[c]*act_total +
// scale[c]*hit[c] and hit[c] is cirom_nor_tile's existing accumulate.
// Behavioral reference only -- not wired into any signed-off
// chip/macro RTL.
`default_nettype none

module cirom_act_total_beh #(
    parameter int N           = 256,
    parameter int K           = 8,
    parameter int SUBCOL_ROWS = 128,
    parameter int WACC        = 24
) (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                start,
    input  logic [K*N-1:0]      act_flat,
    output logic [WACC-1:0]     act_total,
    output logic                valid
);
    localparam int N_SUB = (N + SUBCOL_ROWS - 1) / SUBCOL_ROWS;
    localparam int K_W   = $clog2(K + 1);
    localparam int R_W   = $clog2(SUBCOL_ROWS + 1);

    logic [K_W-1:0] k_idx;
    logic [R_W-1:0] r_idx;
    logic           running;
    logic [K*N-1:0] act_latched;

    integer s;
    int row, bit_idx;
    logic [WACC-1:0] act_hit;
    always_comb begin
        act_hit = '0;
        row = 0; bit_idx = 0;
        for (s = 0; s < N_SUB; s++) begin
            row = s * SUBCOL_ROWS + r_idx;
            if (row < N) begin
                bit_idx = K * row + k_idx;
                if (act_latched[bit_idx]) act_hit = act_hit + 1;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            k_idx <= '0; r_idx <= '0; running <= 1'b0;
            act_latched <= '0; act_total <= '0; valid <= 1'b0;
        end else begin
            valid <= 1'b0;
            if (start) begin
                act_latched <= act_flat;
                k_idx <= '0; r_idx <= '0; running <= 1'b1; act_total <= '0;
            end else if (running) begin
                act_total <= act_total + (act_hit <<< k_idx);
                if (r_idx == SUBCOL_ROWS - 1) begin
                    r_idx <= '0;
                    if (k_idx == K - 1) begin
                        running <= 1'b0;
                        valid   <= 1'b1;
                    end else begin
                        k_idx <= k_idx + 1'b1;
                    end
                end else begin
                    r_idx <= r_idx + 1'b1;
                end
            end
        end
    end

endmodule

`default_nettype wire
