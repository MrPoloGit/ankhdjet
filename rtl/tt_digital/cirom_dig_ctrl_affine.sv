// Digital-tier controller for the INPUT-AXIS affine-binary case (see
// docs/binary_and_matmulfree_investigation.md): scale/bias vary per
// INPUT ROW, broadcast over every output column -- FBI-LLM's actual
// released convention, confirmed against a real checkpoint. Modeled
// directly on cirom_dig_ctrl.sv (the signed-off Darga tile's full
// ternary MAC on die), carrying the same FSM skeleton, host-loaded
// activation store, and column-group (N_ACC) passing, extended with
// the one new thing this case needs: a per-row Q8.8 scale/bias
// multiply folded into the accumulate, plus one shared bias
// accumulator (not per-column) added back in before streamout.
//
//   dot[c] = sum_r act[r] * (bias[r] + sign_pm[r,c] * scale[r])
//          = sum_r (act[r] * bias[r])  +  sum_r sign_pm[r,c] * (act[r] * scale[r])
//
// SCALE_Q/BIAS_Q are compile-time constants (flat Q8.8 parameters, one
// per row, same convention as a mask-programmed via pattern): they are
// PART of the layer's weights, fixed at synthesis time, not
// host-loaded the way activations are -- unlike the ternary tile,
// where the per-tensor scale is applied entirely off-tile, this tile
// must apply its per-row scale/bias during the row sweep itself, since
// that is the only place the per-row quantity is available cheaply.
//
// Streamed `result` is the RAW accumulate (acc[i] + bias_acc, each
// column's hit-weighted sum plus the shared bias-weighted sum), still
// in Q8.8 units -- NOT yet shifted by Q_FRAC or clipped. That final
// shift+clip is left to a downstream `requantize` stage at unit scale,
// same separation of concerns as the ternary tile (whose per-tensor
// scale is also applied off-tile): this file's job is the per-row MAC
// only. Unlike the ternary tile (ACC_W <= 16, 2 streamed bytes), the
// accumulated quantity here is a sum of ACT_W x SCALE_W products, so
// ACC_W can need more than 2 bytes; the byte-streamout logic below is
// generic over GRPB = ACC_W/OUT_W bytes per accumulator (ACC_W must be
// a multiple of OUT_W).
//
// Feature set reduced from cirom_dig_ctrl.sv on purpose, to keep this
// first affine-binary tile controller tractable: only the on-die
// activation store (STORE_ACTS=1 behavior, not parameterized away) and
// only MVM mode (no raw row-read mode) are implemented. Both are
// straightforward to add back following cirom_dig_ctrl.sv's pattern if
// a bench needs them.
`default_nettype none

module cirom_dig_ctrl_affine #(
    parameter int N_ROWS   = 64,
    parameter int N_COLS   = 32,
    parameter int N_ACC    = 8,
    parameter int ACC_W    = 32,   // must be a multiple of OUT_W
    parameter int ACT_W    = 8,
    parameter int SCALE_W  = 16,
    parameter int OUT_W    = 8,
    parameter int CFG_BITS = 16,
    // Per-row Q8.8 constants, row r at bits [SCALE_W*(r+1)-1 -: SCALE_W].
    // Mask-programmed (compile-time), not host-loaded.
    parameter [N_ROWS*SCALE_W-1:0] SCALE_Q = {(N_ROWS*SCALE_W){1'b0}},
    parameter [N_ROWS*SCALE_W-1:0] BIAS_Q  = {(N_ROWS*SCALE_W){1'b0}}
)(
    input  logic                      clk,
    input  logic                      rst_n,
    // host command
    input  logic [7:0]                ui,          // one activation byte per write
    input  logic                      act_wr,
    input  logic                      start,
    // serial config (same scheme as cirom_dig_ctrl.sv)
    input  logic                      cfg_mode,
    input  logic                      cfg_in,
    // array interface (digital sense: raw bitlines in, sampled here)
    output logic [N_ROWS-1:0]         wl,
    output logic                      pre_n,
    input  logic [N_COLS-1:0]         blp,
    input  logic [N_COLS-1:0]         bln,
    // serialized result out (raw Q8.8 accumulate, not yet shifted/clipped)
    output logic [OUT_W-1:0]          result,
    output logic                      result_valid,
    output logic                      busy,
    output logic                      done
);
    localparam int RB     = $clog2(N_ROWS);
    localparam int PASSES = N_COLS / N_ACC;
    localparam int GRPB   = ACC_W / OUT_W;   // bytes per accumulator (exact: ACC_W%OUT_W==0)
    localparam int AW     = N_ACC > 1 ? $clog2(N_ACC) : 1;
    localparam int BW     = GRPB  > 1 ? $clog2(GRPB)  : 1;
    localparam int PW     = PASSES > 1 ? $clog2(PASSES) : 1;

    typedef enum logic [3:0] {
        S_IDLE, S_PRECHG, S_EVAL, S_SAMPLE, S_ACC, S_NEXT, S_SHIFT,
        S_NEXTGRP, S_DONE
    } state_t;
    state_t state, next;

    // config register (same scheme and fields as cirom_dig_ctrl.sv)
    logic [CFG_BITS-1:0] cfg_reg;
    wire  [3:0] cfg_strobe_delay = cfg_reg[3:0];
    wire  [3:0] cfg_pre_width    = cfg_reg[7:4];
    wire        cfg_bypass_pre   = cfg_reg[8];
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)        cfg_reg <= '0;
        else if (cfg_mode) cfg_reg <= {cfg_reg[CFG_BITS-2:0], cfg_in};
    end

    logic [RB-1:0]           row_q;
    logic [3:0]              cnt;
    logic [N_COLS-1:0]       pos_hit, neg_hit;
    logic signed [ACC_W-1:0] acc [0:N_ACC-1];
    logic signed [ACC_W-1:0] bias_acc;
    logic [AW-1:0]           acc_idx_q;   // which accumulator is streaming
    logic [BW-1:0]           byte_q;      // which byte of that accumulator
    logic [PW-1:0]           grp_q;       // which column-group pass
    wire                     last_row  = (row_q == N_ROWS - 1);
    wire                     last_grp  = (grp_q == PASSES - 1);
    wire                     last_byte = (byte_q == GRPB - 1);
    wire                     last_acc  = (acc_idx_q == N_ACC - 1);

    // On-die activation store: N_ROWS x ACT_W, one host byte per row
    // (ACT_W is a full byte here, matching the K=8 reference activation
    // precision this case was validated at -- unlike cirom_dig_ctrl.sv's
    // two-4-bit-nibbles-per-byte packing). Same unrolled
    // constant-part-select write/read discipline: plain registers, no
    // memory inference.
    logic [N_ROWS*ACT_W-1:0] act_mem;
    logic [RB-1:0]           act_ptr;
    integer b;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_ptr <= '0;
        end else if (act_wr && state == S_IDLE) begin
            for (b = 0; b < N_ROWS; b = b + 1)
                if (act_ptr == b[RB-1:0])
                    act_mem[b*ACT_W +: ACT_W] <= ui[ACT_W-1:0];
            act_ptr <= act_ptr + 1'b1;
        end else if (start && state == S_IDLE) begin
            act_ptr <= '0;
        end
    end
    wire [ACT_W-1:0] act_cur = act_mem[row_q*ACT_W +: ACT_W];

    // Per-row Q8.8 scale/bias lookup (mask-programmed constants).
    wire signed [SCALE_W-1:0] scale_cur = $signed(SCALE_Q[row_q*SCALE_W +: SCALE_W]);
    wire signed [SCALE_W-1:0] bias_cur  = $signed(BIAS_Q [row_q*SCALE_W +: SCALE_W]);
    // One shared multiply per row cycle, reused by every column in the
    // active group (the real saving this architecture relies on: the
    // multiply is on the row, not the column).
    wire signed [ACT_W+SCALE_W:0] scale_term = $signed({1'b0, act_cur}) * scale_cur;
    wire signed [ACT_W+SCALE_W:0] bias_term  = $signed({1'b0, act_cur}) * bias_cur;

    integer i;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; row_q <= '0; cnt <= '0;
            pos_hit <= '0; neg_hit <= '0;
            byte_q <= '0; acc_idx_q <= '0; grp_q <= '0;
            bias_acc <= '0;
            for (i = 0; i < N_ACC; i = i + 1) acc[i] <= '0;
        end else begin
            state <= next;
            case (state)
                S_IDLE: if (start) begin
                    row_q     <= '0;
                    cnt       <= cfg_pre_width;
                    byte_q    <= '0;
                    acc_idx_q <= '0;
                    grp_q     <= '0;
                    bias_acc  <= '0;
                    for (i = 0; i < N_ACC; i = i + 1) acc[i] <= '0;
                end
                S_PRECHG: if (cnt == 0) cnt <= cfg_strobe_delay; else cnt <= cnt - 1'b1;
                S_EVAL:   if (cnt != 0) cnt <= cnt - 1'b1;
                S_SAMPLE: begin
                    pos_hit <= ~blp;   // discharged bitline reads as a hit
                    neg_hit <= ~bln;
                end
                S_ACC: begin
                    bias_acc <= bias_acc + ACC_W'(bias_term);
                    for (i = 0; i < N_ACC; i = i + 1) begin
                        case ({pos_hit[grp_q*N_ACC + i], neg_hit[grp_q*N_ACC + i]})
                            2'b10: acc[i] <= acc[i] + ACC_W'(scale_term);
                            2'b01: acc[i] <= acc[i] - ACC_W'(scale_term);
                            default: ;
                        endcase
                    end
                end
                S_NEXT: begin row_q <= row_q + 1'b1; cnt <= cfg_pre_width; end
                S_SHIFT: begin
                    if (last_byte) begin
                        byte_q    <= '0;
                        acc_idx_q <= acc_idx_q + 1'b1;
                    end else begin
                        byte_q <= byte_q + 1'b1;
                    end
                end
                S_NEXTGRP: begin
                    grp_q     <= grp_q + 1'b1;
                    row_q     <= '0;
                    cnt       <= cfg_pre_width;
                    byte_q    <= '0;
                    acc_idx_q <= '0;
                    bias_acc  <= '0;
                    for (i = 0; i < N_ACC; i = i + 1) acc[i] <= '0;
                end
                default: ;
            endcase
        end
    end

    always_comb begin
        next = state;
        case (state)
            S_IDLE:   if (start) next = cfg_bypass_pre ? S_EVAL : S_PRECHG;
            S_PRECHG: if (cnt == 0) next = S_EVAL;
            S_EVAL:   if (cnt == 0) next = S_SAMPLE;
            S_SAMPLE: next = S_ACC;
            S_ACC:    if (last_row) next = S_SHIFT; else next = S_NEXT;
            S_NEXT:   next = cfg_bypass_pre ? S_EVAL : S_PRECHG;
            S_SHIFT:  if (last_byte && last_acc) begin
                          if (last_grp) next = S_DONE;
                          else next = S_NEXTGRP;
                      end
            S_NEXTGRP: next = cfg_bypass_pre ? S_EVAL : S_PRECHG;
            S_DONE:   next = S_IDLE;
            default:  next = S_IDLE;
        endcase
    end

    // wordline: one-hot during precharge-evaluate-sample of the active row
    wire reading = (state == S_PRECHG) || (state == S_EVAL) || (state == S_SAMPLE);
    always_comb begin
        wl = '0;
        if (reading) wl[row_q] = 1'b1;
    end
    assign pre_n = (state == S_PRECHG) ? 1'b0 : 1'b1;

    // result mux: GRPB bytes of (acc[acc_idx_q] + bias_acc), low byte
    // first. ACC_W is an exact multiple of OUT_W (parameter contract),
    // so no sign-extension padding is needed the way cirom_dig_ctrl.sv's
    // fixed 2-byte mux needs it.
    wire signed [ACC_W-1:0] acc_sel = acc[acc_idx_q] + bias_acc;
    assign result        = acc_sel[byte_q*OUT_W +: OUT_W];
    assign result_valid  = (state == S_SHIFT);
    assign busy          = (state != S_IDLE);
    assign done           = (state == S_DONE);

endmodule

`default_nettype wire
