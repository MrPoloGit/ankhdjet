// cirom_chip_digital_affine: chip-level top for the INPUT-AXIS
// affine-binary case (see docs/binary_and_matmulfree_investigation.md),
// wiring cirom_dig_ctrl_affine (the synthesis-quality tile controller)
// to the SAME hardened array macro cirom_chip_digital.sv uses -- the
// array needs nothing new for this case (a binary cell never floats a
// drain, a strict subset of the ternary mask vocabulary the macro
// already implements), so this chip reuses macro_array_pc_64x32_test0
// unchanged, just driven by different periphery.
//
// SCALE_Q/BIAS_Q below are REAL values: the first 64 input-row Q8.8
// scale/bias constants of microsoft... no -- LiqunMa/FBI-LLM_130M's
// block-0 self_attn.q_proj, read directly off the converted checkpoint
// (see tools/convert_pickle_checkpoint.py and
// tests/test_fbi_llm_real_layer_rtl.py for how). This is NOT a claim
// that this chip computes the real q_proj matmul end to end (N_COLS
// here is 32, matching the existing macro's shape, not q_proj's real
// 768 output columns, and no activation source feeds real model
// data) -- it is the real per-row affine periphery constants for a
// real slice of a real layer, driving the real signed-off array
// macro, which is the part of this chip that is not a reference
// model.
//
// NOT SIGNED OFF: this file has not been through floorplan, DRC, LVS,
// or STA. See docs/binary_and_matmulfree_investigation.md for exactly
// what's missing (LibreLane config exists alongside this file; the
// toolchain run that would exercise it does not, in the environment
// this was built in).
`default_nettype none

`ifndef ANKHDJET_ARRAY_MODULE
`define ANKHDJET_ARRAY_MODULE macro_array_pc_64x32_test0
`endif

module cirom_chip_digital_affine (
`ifdef USE_POWER_PINS
    inout  wire       VPWR,
    inout  wire       VGND,
`endif
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] ui,          // activation byte (load) / unused
    input  wire       act_wr,
    input  wire       start,
    input  wire       cfg_mode,
    input  wire       cfg_in,
    output wire [7:0] result,
    output wire       result_valid,
    output wire       busy,
    output wire       done
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
    // input rows 0..63 (row r at bits [SCALE_W*(r+1)-1 -: SCALE_W]).
    localparam [N_ROWS*SCALE_W-1:0] SCALE_Q =
        1024'b0000000000100000000000000010001000000000001000110000000000100101000000000010010100000000001011000000000000100100000000000010110000000000000111110000000000101111000000000010010000000000001000010000000000011111000000000010010000000000001010010000000000100001000000000010011100000000001001100000000000100001000000000010001000000000001000100000000000011110000000000001101100000000001001000000000000100010000000000001111000000000001001000000000000100010000000000010011000000000001000010000000000100110000000000010100000000000001001100000000000100010000000000010010100000000001000100000000000100110000000000010001000000000001000000000000000011111000000000010010100000000001001110000000000011111000000000010101000000000001010000000000000100101000000000010000100000000001000000000000000100101000000000010010100000000001000110000000000101000000000000010100000000000001000110000000000100000000000000010110100000000001000100000000000100000000000000001011100000000000011100000000000100001000000000011000100000000001000100000000000101001;
    localparam [N_ROWS*SCALE_W-1:0] BIAS_Q =
        1024'b1111111111111110111111111111111111111111111111110000000000000001000000000000001000000000000000101111111111111110111111111111101111111111111110101111111111111111000000000000011000000000000000110000000000000010000000000000000000000000000001001111111111111100111111111111011000000000000001000000000000000001111111111111111000000000000001110000000000000011000000000000000000000000000000011111111111111110111111111111101011111111111110000000000000000001111111111111111111111111111111010000000000001110000000000000010111111111111111110000000000000100111111111111110000000000000000001111111111111001000000000000010111111111111111010000000000000101111111111111111111111111111111110000000000001110000000000000011011111111111111111111111111111110111111111111111111111111111111101111111111111110111111111111111111111111111111101111111111111110111111111111111011111111111111010000000000000011111111111111111100000000000000100000000000000111000000000000001100000000000000010000000000000001000000000000000000000000000001001111111111111111;

    wire [N_ROWS-1:0] wl;
    wire              pre_n;
    wire [N_COLS-1:0] blp, bln;

`ifdef USE_POWER_PINS
    wire vpwr = VPWR;
    wire vgnd = VGND;
`endif

    cirom_dig_ctrl_affine #(
        .N_ROWS(N_ROWS), .N_COLS(N_COLS), .N_ACC(N_ACC), .ACC_W(ACC_W),
        .ACT_W(ACT_W), .SCALE_W(SCALE_W), .OUT_W(OUT_W), .CFG_BITS(CFG_BITS),
        .SCALE_Q(SCALE_Q), .BIAS_Q(BIAS_Q)
    ) u_ctrl (
        .clk(clk), .rst_n(rst_n), .ui(ui), .act_wr(act_wr), .start(start),
        .cfg_mode(cfg_mode), .cfg_in(cfg_in),
        .wl(wl), .pre_n(pre_n), .blp(blp), .bln(bln),
        .result(result), .result_valid(result_valid),
        .busy(busy), .done(done)
    );

    `ANKHDJET_ARRAY_MODULE u_macro (
        .PRE_N(pre_n),
        .WL_0(wl[0]), .WL_1(wl[1]), .WL_2(wl[2]), .WL_3(wl[3]), .WL_4(wl[4]), .WL_5(wl[5]), .WL_6(wl[6]), .WL_7(wl[7]),
        .WL_8(wl[8]), .WL_9(wl[9]), .WL_10(wl[10]), .WL_11(wl[11]), .WL_12(wl[12]), .WL_13(wl[13]), .WL_14(wl[14]), .WL_15(wl[15]),
        .WL_16(wl[16]), .WL_17(wl[17]), .WL_18(wl[18]), .WL_19(wl[19]), .WL_20(wl[20]), .WL_21(wl[21]), .WL_22(wl[22]), .WL_23(wl[23]),
        .WL_24(wl[24]), .WL_25(wl[25]), .WL_26(wl[26]), .WL_27(wl[27]), .WL_28(wl[28]), .WL_29(wl[29]), .WL_30(wl[30]), .WL_31(wl[31]),
        .WL_32(wl[32]), .WL_33(wl[33]), .WL_34(wl[34]), .WL_35(wl[35]), .WL_36(wl[36]), .WL_37(wl[37]), .WL_38(wl[38]), .WL_39(wl[39]),
        .WL_40(wl[40]), .WL_41(wl[41]), .WL_42(wl[42]), .WL_43(wl[43]), .WL_44(wl[44]), .WL_45(wl[45]), .WL_46(wl[46]), .WL_47(wl[47]),
        .WL_48(wl[48]), .WL_49(wl[49]), .WL_50(wl[50]), .WL_51(wl[51]), .WL_52(wl[52]), .WL_53(wl[53]), .WL_54(wl[54]), .WL_55(wl[55]),
        .WL_56(wl[56]), .WL_57(wl[57]), .WL_58(wl[58]), .WL_59(wl[59]), .WL_60(wl[60]), .WL_61(wl[61]), .WL_62(wl[62]), .WL_63(wl[63]),
        .BLP_0(blp[0]), .BLP_1(blp[1]), .BLP_2(blp[2]), .BLP_3(blp[3]), .BLP_4(blp[4]), .BLP_5(blp[5]), .BLP_6(blp[6]), .BLP_7(blp[7]), .BLP_8(blp[8]), .BLP_9(blp[9]), .BLP_10(blp[10]), .BLP_11(blp[11]), .BLP_12(blp[12]), .BLP_13(blp[13]), .BLP_14(blp[14]), .BLP_15(blp[15]), .BLP_16(blp[16]), .BLP_17(blp[17]), .BLP_18(blp[18]), .BLP_19(blp[19]), .BLP_20(blp[20]), .BLP_21(blp[21]), .BLP_22(blp[22]), .BLP_23(blp[23]), .BLP_24(blp[24]), .BLP_25(blp[25]), .BLP_26(blp[26]), .BLP_27(blp[27]), .BLP_28(blp[28]), .BLP_29(blp[29]), .BLP_30(blp[30]), .BLP_31(blp[31]),
        .BLN_0(bln[0]), .BLN_1(bln[1]), .BLN_2(bln[2]), .BLN_3(bln[3]), .BLN_4(bln[4]), .BLN_5(bln[5]), .BLN_6(bln[6]), .BLN_7(bln[7]), .BLN_8(bln[8]), .BLN_9(bln[9]), .BLN_10(bln[10]), .BLN_11(bln[11]), .BLN_12(bln[12]), .BLN_13(bln[13]), .BLN_14(bln[14]), .BLN_15(bln[15]), .BLN_16(bln[16]), .BLN_17(bln[17]), .BLN_18(bln[18]), .BLN_19(bln[19]), .BLN_20(bln[20]), .BLN_21(bln[21]), .BLN_22(bln[22]), .BLN_23(bln[23]), .BLN_24(bln[24]), .BLN_25(bln[25]), .BLN_26(bln[26]), .BLN_27(bln[27]), .BLN_28(bln[28]), .BLN_29(bln[29]), .BLN_30(bln[30]), .BLN_31(bln[31])
`ifdef USE_POWER_PINS
        , .VPWR(vpwr), .VGND(vgnd)
`endif
    );

endmodule

`default_nettype wire
