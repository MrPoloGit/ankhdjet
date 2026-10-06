// TinyTapeout top for the FBI-LLM affine-binary tile: the same
// mask-programmed CiROM array macro Darga/Azara use, read by the same
// clocked digital bitline sampling, but driven by the INPUT-AXIS
// affine-binary tile controller (cirom_dig_ctrl_affine) instead of
// the ternary one (cirom_dig_ctrl) -- see
// docs/binary_and_matmulfree_investigation.md for the scheme and
// rtl/chip/cirom_chip_digital_affine.sv (this tile's non-TT sibling,
// already through a full DRC 0 / LVS 0 / timing-clean LibreLane run)
// for the controller's own signoff history.
//
// SCALE_Q/BIAS_Q below are the SAME real values cirom_chip_digital_affine.sv
// uses: the first 64 input-row Q8.8 scale/bias constants of
// LiqunMa/FBI-LLM_130M's block-0 self_attn.q_proj, read directly off
// the converted checkpoint. As with that file: this is the real
// per-row affine periphery constants for a real slice of a real
// layer, driving the real signed-off array macro -- not a claim that
// this tile computes the full q_proj matmul (N_COLS here is 32,
// matching the macro's shape, not q_proj's real 768 output columns,
// and uio has no path for live model data to enter besides ui_in).
//
// Pin map (one fewer control bit than Darga: this controller has no
// raw-row-read mode, only matrix-vector multiply):
//   ui_in[7:0]   activation byte while loading (one byte per row, 64
//                writes load the full activation vector)
//   uio_in[0]    act_wr: write ui_in into the activation store
//   uio_in[1]    start
//   uio_in[2]    cfg_mode (serial config: strobe_delay/pre_width/bypass_pre)
//   uio_in[3]    cfg_in
//   uo_out[7:0]  result byte stream (N_ACC accumulators x ACC_W/8 bytes
//                each, low byte first, in ascending column order)
//   uio_out[5]   result_valid
//   uio_out[6]   busy
//   uio_out[7]   done
`default_nettype none

module tt_um_cirom_fbi_llm (
`ifdef USE_POWER_PINS
    inout  wire       VGND,
    inout  wire       VDPWR,
`endif
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);
    localparam int N_ROWS   = 64;
    localparam int N_COLS   = 32;
    localparam int N_ACC    = 32;   // single pass: N_ACC == N_COLS
    localparam int ACC_W    = 32;
    localparam int ACT_W    = 8;
    localparam int SCALE_W  = 16;
    localparam int OUT_W    = 8;
    localparam int CFG_BITS = 16;

    // Real Q8.8 constants: LiqunMa/FBI-LLM_130M block-0 self_attn.q_proj,
    // input rows 0..63 (row r at bits [SCALE_W*(r+1)-1 -: SCALE_W]) --
    // identical to rtl/chip/cirom_chip_digital_affine.sv's.
    localparam [N_ROWS*SCALE_W-1:0] SCALE_Q =
        1024'b0000000000100000000000000010001000000000001000110000000000100101000000000010010100000000001011000000000000100100000000000010110000000000000111110000000000101111000000000010010000000000001000010000000000011111000000000010010000000000001010010000000000100001000000000010011100000000001001100000000000100001000000000010001000000000001000100000000000011110000000000001101100000000001001000000000000100010000000000001111000000000001001000000000000100010000000000010011000000000001000010000000000100110000000000010100000000000001001100000000000100010000000000010010100000000001000100000000000100110000000000010001000000000001000000000000000011111000000000010010100000000001001110000000000011111000000000010101000000000001010000000000000100101000000000010000100000000001000000000000000100101000000000010010100000000001000110000000000101000000000000010100000000000001000110000000000100000000000000010110100000000001000100000000000100000000000000001011100000000000011100000000000100001000000000011000100000000001000100000000000101001;
    localparam [N_ROWS*SCALE_W-1:0] BIAS_Q =
        1024'b1111111111111110111111111111111111111111111111110000000000000001000000000000001000000000000000101111111111111110111111111111101111111111111110101111111111111111000000000000011000000000000000110000000000000010000000000000000000000000000001001111111111111100111111111111011000000000000001000000000000000001111111111111111000000000000001110000000000000011000000000000000000000000000000011111111111111110111111111111101011111111111110000000000000000001111111111111111111111111111111010000000000001110000000000000010111111111111111110000000000000100111111111111110000000000000000001111111111111001000000000000010111111111111111010000000000000101111111111111111111111111111111110000000000001110000000000000011011111111111111111111111111111110111111111111111111111111111111101111111111111110111111111111111111111111111111101111111111111110111111111111111011111111111111010000000000000011111111111111111100000000000000100000000000000111000000000000001100000000000000010000000000000001000000000000000000000000000001001111111111111111;

    wire [N_ROWS-1:0] wl;
    wire              pre_n;
    wire [N_COLS-1:0] blp, bln;
    wire              result_valid, busy, done;

    wire _unused = &{ena, uio_in[7:4], 1'b0};

    cirom_dig_ctrl_affine #(
        .N_ROWS(N_ROWS), .N_COLS(N_COLS), .N_ACC(N_ACC), .ACC_W(ACC_W),
        .ACT_W(ACT_W), .SCALE_W(SCALE_W), .OUT_W(OUT_W), .CFG_BITS(CFG_BITS),
        .SCALE_Q(SCALE_Q), .BIAS_Q(BIAS_Q)
    ) u_ctrl (
        .clk(clk), .rst_n(rst_n),
        .ui(ui_in),
        .act_wr(uio_in[0]),
        .start(uio_in[1]),
        .cfg_mode(uio_in[2]),
        .cfg_in(uio_in[3]),
        .wl(wl), .pre_n(pre_n),
        .blp(blp), .bln(bln),
        .result(uo_out),
        .result_valid(result_valid),
        .busy(busy),
        .done(done)
    );

    cirom_dig_afe #(
        .N_ROWS(N_ROWS), .N_COLS(N_COLS)
    ) u_afe (
`ifdef USE_POWER_PINS
        .VPWR(VDPWR), .VGND(VGND),
`endif
        .wl(wl), .pre_n(pre_n),
        .blp(blp), .bln(bln)
    );

    assign uio_out = {done, busy, result_valid, 5'b0};
    assign uio_oe  = 8'b1110_0000;

endmodule

`default_nettype wire
