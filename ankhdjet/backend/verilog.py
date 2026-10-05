"""Verilog backend: emits synthesizable modules from a ternary IR.

`emit_layer_nor` wraps a single `cirom_nor_tile` parameterized with the
layer's mask-programmed via positions. `emit_pipeline` chains N LINEAR
layers through between_layer requantize stages and returns a coherent
`ankhdjet_pipeline_<name>` top with valid/start handshake between layers.

`PipelineConfig` is the config bag the area model + report tools consume;
it does not currently drive codegen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ankhdjet.backend.idents import check_unique, sv_ident
from ankhdjet.frontend.ir import Layer, LayerType, QuantScheme


def _column_widths(n: int, k_bits: int) -> tuple[int, int]:
    """Return (WCNT, WACC) for a column with N rows and K-bit activations."""
    wcnt = max(1, math.ceil(math.log2(n + 1)))
    max_mag = n * ((1 << k_bits) - 1)
    wacc = math.ceil(math.log2(max_mag + 1)) + 1
    return wcnt, wacc


def emit_layer_nor(layer: Layer, k_bits: int = 8,
                    subcol_rows: int = 64) -> str:
    """Emit one ankhdjet_layer_<name> module wrapping a `cirom_nor_tile`.

    NOR-array architecture: 1 NMOS per ternary weight, mask-customized
    via-1 routing to BL+/BL-/floating, column-tiled row-sequential
    readout. Cycles per dot product = k_bits * min(N, subcol_rows).
    The module name is the sv_ident-legalized layer name; the original
    spelling is kept in the header comment.
    """
    if layer.layer_type != LayerType.LINEAR:
        raise NotImplementedError(f"unsupported layer type {layer.layer_type}")
    wt = layer.weights["weight"]
    if wt.scheme != QuantScheme.TERNARY:
        raise ValueError(f"expected ternary weights, got {wt.scheme}")
    W = wt.data
    if W.ndim != 2:
        raise ValueError(f"weight matrix must be 2-D, got {W.shape}")

    n, m = W.shape
    _, wacc = _column_widths(n, k_bits)
    lid = sv_ident(layer.name)

    # Pack HAS_VIA_POS / HAS_VIA_NEG flat bit-vectors. Bit index r*M + c.
    pos_bits = []
    neg_bits = []
    for r in range(n):
        for c in range(m):
            pos_bits.append("1" if W[r, c] == 1 else "0")
            neg_bits.append("1" if W[r, c] == -1 else "0")
    # Verilog literal is MSB-first
    pos_lit = f"{n*m}'b{''.join(reversed(pos_bits))}"
    neg_lit = f"{n*m}'b{''.join(reversed(neg_bits))}"

    lines: list[str] = []
    lines.append(f"// Auto-generated NOR-array layer: {layer.name}")
    lines.append(f"// Shape: N={n} x M={m}, K={k_bits}, SUBCOL_ROWS={subcol_rows}")
    lines.append(f"// Cycles per dot product: {k_bits * min(n, subcol_rows)}")
    lines.append("")
    lines.append("`default_nettype none")
    lines.append("")
    lines.append(f"module ankhdjet_layer_{lid} (")
    lines.append("    input  logic                       clk,")
    lines.append("    input  logic                       rst_n,")
    lines.append("    input  logic                       start,")
    lines.append(f"    input  logic [{k_bits * n - 1}:0]  act_flat,")
    lines.append(f"    output logic signed [{wacc * m - 1}:0] acc_flat,")
    lines.append("    output logic                       valid")
    lines.append(");")
    lines.append("    cirom_nor_tile #(")
    lines.append(f"        .N({n}),")
    lines.append(f"        .M({m}),")
    lines.append(f"        .K({k_bits}),")
    lines.append(f"        .SUBCOL_ROWS({subcol_rows}),")
    lines.append(f"        .WACC({wacc}),")
    lines.append(f"        .HAS_VIA_POS({pos_lit}),")
    lines.append(f"        .HAS_VIA_NEG({neg_lit})")
    lines.append("    ) u_nor (")
    lines.append("        .clk(clk), .rst_n(rst_n), .start(start),")
    lines.append("        .act_flat(act_flat),")
    lines.append("        .acc_flat(acc_flat), .valid(valid)")
    lines.append("    );")
    lines.append("endmodule")
    lines.append("")
    lines.append("`default_nettype wire")
    return "\n".join(lines) + "\n"


def quantize_q88(x: float, q_frac: int = 8, width: int = 16) -> int:
    """Round a float to a signed Q(width-1-q_frac).q_frac fixed-point
    integer, clipped to the representable range (saturate, don't wrap)."""
    lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
    return int(max(lo, min(hi, round(x * (1 << q_frac)))))


def emit_layer_affine_binary(layer: Layer, k_bits: int = 8,
                             subcol_rows: int = 64) -> str:
    """Emit one ankhdjet_layer_<name> module for the row-granularity
    affine-binary case (see docs/binary_and_matmulfree_investigation.md):
    wraps a `cirom_nor_tile` (hit accumulate, reused unmodified -- the
    sign mask never floats a drain) alongside a `cirom_act_total_beh`
    (the one new array-wide accumulator this case needs), exposing both
    raw outputs for a downstream `between_layer_affine` stage to combine
    with the layer's per-column scale/bias. NOT wired into any
    signed-off chip/macro RTL; the layer's actual scale/bias (from
    `decode_binary_affine`) are consumed by the caller when building
    `between_layer_affine`'s SCALE_Q_FLAT/BIAS_Q_FLAT, not by this
    module.
    """
    if layer.layer_type != LayerType.LINEAR:
        raise NotImplementedError(f"unsupported layer type {layer.layer_type}")
    wt = layer.weights["weight"]
    if wt.scheme != QuantScheme.BINARY:
        raise ValueError(f"expected binary weights, got {wt.scheme}")
    W = wt.data
    if W.ndim != 2:
        raise ValueError(f"weight matrix must be 2-D, got {W.shape}")
    if not set(np.unique(W).tolist()).issubset({0, 1}):
        raise ValueError("binary weight data must be in {0, 1}")

    n, m = W.shape
    _, wacc = _column_widths(n, k_bits)
    lid = sv_ident(layer.name)

    # HAS_VIA_POS = sign (1 -> BL+); HAS_VIA_NEG is its complement, since
    # a binary cell never floats a drain. Bit index r*M + c, matching
    # cirom_nor_tile's convention.
    pos_bits = []
    for r in range(n):
        for c in range(m):
            pos_bits.append("1" if W[r, c] == 1 else "0")
    pos_lit = f"{n*m}'b{''.join(reversed(pos_bits))}"
    neg_lit = f"~({pos_lit})"

    lines: list[str] = []
    lines.append(f"// Auto-generated affine-binary layer: {layer.name}")
    lines.append(f"// Shape: N={n} x M={m}, K={k_bits}, SUBCOL_ROWS={subcol_rows}")
    lines.append("// Reference model only -- see "
                 "docs/binary_and_matmulfree_investigation.md")
    lines.append("")
    lines.append("`default_nettype none")
    lines.append("")
    lines.append(f"module ankhdjet_layer_{lid} (")
    lines.append("    input  logic                       clk,")
    lines.append("    input  logic                       rst_n,")
    lines.append("    input  logic                       start,")
    lines.append(f"    input  logic [{k_bits * n - 1}:0]  act_flat,")
    lines.append(f"    output logic signed [{wacc * m - 1}:0] acc_flat,")
    lines.append(f"    output logic [{wacc - 1}:0]        act_total,")
    lines.append("    output logic                       valid")
    lines.append(");")
    lines.append("    logic hit_valid, act_valid;")
    lines.append("    cirom_nor_tile #(")
    lines.append(f"        .N({n}), .M({m}), .K({k_bits}),")
    lines.append(f"        .SUBCOL_ROWS({subcol_rows}), .WACC({wacc}),")
    lines.append(f"        .HAS_VIA_POS({pos_lit}),")
    lines.append(f"        .HAS_VIA_NEG({neg_lit})")
    lines.append("    ) u_nor (")
    lines.append("        .clk(clk), .rst_n(rst_n), .start(start),")
    lines.append("        .act_flat(act_flat),")
    lines.append("        .acc_flat(acc_flat), .valid(hit_valid)")
    lines.append("    );")
    lines.append("    cirom_act_total_beh #(")
    lines.append(f"        .N({n}), .K({k_bits}),")
    lines.append(f"        .SUBCOL_ROWS({subcol_rows}), .WACC({wacc})")
    lines.append("    ) u_act (")
    lines.append("        .clk(clk), .rst_n(rst_n), .start(start),")
    lines.append("        .act_flat(act_flat),")
    lines.append("        .act_total(act_total), .valid(act_valid)")
    lines.append("    );")
    lines.append("    assign valid = hit_valid && act_valid;")
    lines.append("endmodule")
    lines.append("")
    lines.append("`default_nettype wire")
    return "\n".join(lines) + "\n"


def emit_layer_input_axis_affine(layer: Layer, k_bits: int = 8,
                                 out_k_bits: int = 8, q_frac: int = 8) -> str:
    """Emit one ankhdjet_layer_<name> module for the INPUT-AXIS
    affine-binary case: scale/bias vary per input row, broadcast over
    every output column -- FBI-LLM's actual released convention,
    confirmed against a real checkpoint (see
    docs/binary_and_matmulfree_investigation.md). Wraps
    `cirom_input_axis_affine_beh` with the layer's real per-row
    scale/bias baked in as Q8.8 constants, followed by one
    `requantize` instance per column at unit scale (no further
    multiply -- the affine combine already happened inside
    cirom_input_axis_affine_beh) purely for the relu+clip-to-K-bit
    activation stage, so this layer's output composes with the
    existing between_layer/requantize bus convention. NOT signed-off
    RTL and not cycle-accurate (see that module's header for why
    bit-serial-over-K doesn't help this axis: the per-row quantity is
    a real signed product, not a single bit, so one shared
    time-multiplexed multiply over N row-cycles does strictly less
    total work than K*N bit-serial masked adds would).
    """
    if layer.layer_type != LayerType.LINEAR:
        raise NotImplementedError(f"unsupported layer type {layer.layer_type}")
    wt = layer.weights["weight"]
    if wt.scheme != QuantScheme.BINARY:
        raise ValueError(f"expected binary weights, got {wt.scheme}")
    if wt.scale_axis != "input":
        raise ValueError(
            f"expected scale_axis='input', got {wt.scale_axis!r} -- "
            f"use emit_layer_affine_binary for the output-axis case")
    W = wt.data
    if W.ndim != 2:
        raise ValueError(f"weight matrix must be 2-D, got {W.shape}")
    if not set(np.unique(W).tolist()).issubset({0, 1}):
        raise ValueError("binary weight data must be in {0, 1}")
    n, m = W.shape
    scale = np.asarray(wt.scale).reshape(n)
    bias = np.asarray(wt.bias).reshape(n)

    wacc = max(32, int(math.ceil(math.log2(
        n * (1 << k_bits) * 4096 + 1))) + 2)
    lid = sv_ident(layer.name)

    sign_bits = []
    for r in range(n):
        for c in range(m):
            sign_bits.append("1" if W[r, c] == 1 else "0")
    sign_lit = f"{n*m}'b{''.join(reversed(sign_bits))}"

    def _q88_lit(x: float) -> str:
        v = quantize_q88(x, q_frac)
        return f"-16'sd{-v}" if v < 0 else f"16'sd{v}"

    scale_lits = ", ".join(_q88_lit(x) for x in scale)
    bias_lits = ", ".join(_q88_lit(x) for x in bias)

    lines: list[str] = []
    lines.append(f"// Auto-generated input-axis affine-binary layer: {layer.name}")
    lines.append(f"// Shape: N={n} x M={m}, K={k_bits}, WACC={wacc}")
    lines.append("// Reference model only -- see "
                 "docs/binary_and_matmulfree_investigation.md")
    lines.append("")
    lines.append("`default_nettype none")
    lines.append("")
    lines.append(f"module ankhdjet_layer_{lid} (")
    lines.append("    input  logic                       clk,")
    lines.append("    input  logic                       rst_n,")
    lines.append("    input  logic                       start,")
    lines.append(f"    input  logic [{k_bits-1}:0]        act [{n}],")
    lines.append(f"    output logic [{out_k_bits * m - 1}:0] out_flat,")
    lines.append("    output logic                       valid")
    lines.append(");")
    lines.append(f"    localparam logic signed [15:0] SCALE_Q [{n}] = '{{{scale_lits}}};")
    lines.append(f"    localparam logic signed [15:0] BIAS_Q  [{n}] = '{{{bias_lits}}};")
    lines.append(f"    logic signed [{wacc-1}:0] result [{m}];")
    lines.append("    cirom_input_axis_affine_beh #(")
    lines.append(f"        .N({n}), .M({m}), .K({k_bits}), .SCALE_W(16),")
    lines.append(f"        .Q_FRAC({q_frac}), .WACC({wacc}),")
    lines.append(f"        .HAS_VIA_POS({sign_lit})")
    lines.append("    ) u_axis (")
    lines.append("        .clk(clk), .rst_n(rst_n), .start(start),")
    lines.append("        .act(act), .scale_q(SCALE_Q), .bias_q(BIAS_Q),")
    lines.append("        .result(result), .valid(valid)")
    lines.append("    );")
    lines.append("    genvar gq;")
    lines.append("    generate")
    lines.append(f"        for (gq = 0; gq < {m}; gq = gq + 1) begin : g_rq")
    lines.append(f"            logic [{out_k_bits-1}:0] out_j;")
    lines.append("            requantize #(")
    lines.append(f"                .WACC({wacc}), .SCALE_W(16), .Q_FRAC({q_frac}),")
    lines.append(f"                .K({out_k_bits}), .SCALE_Q(16'd{1 << q_frac}),")
    lines.append("                .ACTIVATION(0)")
    lines.append("            ) u_rq (")
    lines.append("                .acc(result[gq]),")
    lines.append("                .out(out_j)")
    lines.append("            );")
    lines.append(f"            assign out_flat[{out_k_bits}*(gq+1)-1 -: {out_k_bits}] = out_j;")
    lines.append("        end")
    lines.append("    endgenerate")
    lines.append("endmodule")
    lines.append("")
    lines.append("`default_nettype wire")
    return "\n".join(lines) + "\n"


def emit_pipeline_input_axis(
    layers: list[Layer],
    name: str = "pipeline",
    k_bits: int = 8,
    out_k_bits: int = 8,
    q_frac: int = 8,
) -> str:
    """Chain N input-axis affine-binary LINEAR layers (see
    docs/binary_and_matmulfree_investigation.md) through a one-cycle
    valid->start handshake, mirroring `emit_pipeline`'s structure for
    the ternary case. Unlike that case, no `between_layer`-equivalent
    stage sits between layers: `emit_layer_input_axis_affine`'s module
    already applies its own activation+clip (one `requantize` per
    column at unit scale) internally, so a layer's `out_flat` IS the
    next layer's activation input, modulo unpacking the flat K-bit bus
    back into the unpacked `act[N]` array the next layer's port
    expects.

    All layers must have `scheme=BINARY` and `scale_axis="input"`, and
    layer i's output columns (M) must equal layer i+1's input rows
    (N). Module names are `sv_ident`-legalized; collisions are refused.
    """
    if not layers:
        raise ValueError("emit_pipeline_input_axis requires at least one layer")
    for L in layers:
        if L.layer_type != LayerType.LINEAR:
            raise NotImplementedError(f"unsupported layer type {L.layer_type}")
        wt = L.weights["weight"]
        if wt.scheme != QuantScheme.BINARY:
            raise ValueError(f"layer {L.name}: expected BINARY, got {wt.scheme}")
        if wt.scale_axis != "input":
            raise ValueError(
                f"layer {L.name}: expected scale_axis='input', got {wt.scale_axis!r}")
        if wt.data.ndim != 2:
            raise ValueError(f"layer {L.name}: weight must be 2-D")
    check_unique([L.name for L in layers])

    n_layers = len(layers)
    n_stages = max(0, n_layers - 1)
    dims = [L.weights["weight"].data.shape for L in layers]
    for i in range(n_stages):
        if dims[i][1] != dims[i + 1][0]:
            raise ValueError(
                f"layer {layers[i].name}.M={dims[i][1]} != "
                f"layer {layers[i+1].name}.N={dims[i+1][0]}")

    parts = [emit_layer_input_axis_affine(L, k_bits=k_bits, out_k_bits=out_k_bits,
                                          q_frac=q_frac)
             for L in layers]

    n0, _ = dims[0]
    _, m_last = dims[-1]
    lines: list[str] = []
    lines.append(f"// Auto-generated input-axis pipeline: {name} ({n_layers} layers)")
    for i, L in enumerate(layers):
        ni, mi = dims[i]
        lines.append(f"//   L{i}: {L.name}  N={ni} M={mi}")
    lines.append("// Reference model only -- see "
                 "docs/binary_and_matmulfree_investigation.md")
    lines.append("")
    lines.append("`default_nettype none")
    lines.append("")
    lines.append(f"module ankhdjet_pipeline_{sv_ident(name)} (")
    lines.append("    input  logic                       clk,")
    lines.append("    input  logic                       rst_n,")
    lines.append("    input  logic                       start,")
    lines.append(f"    input  logic [{k_bits-1}:0]        act [{n0}],")
    lines.append(f"    output logic [{out_k_bits * m_last - 1}:0] out_flat,")
    lines.append("    output logic                       valid")
    lines.append(");")

    n_, m_ = dims[0]
    lines.append(f"    wire [{out_k_bits * m_ - 1}:0] out_0;")
    lines.append("    wire                            valid_0;")
    lines.append(f"    ankhdjet_layer_{sv_ident(layers[0].name)} u_l0 (")
    lines.append("        .clk(clk), .rst_n(rst_n), .start(start),")
    lines.append("        .act(act),")
    lines.append("        .out_flat(out_0), .valid(valid_0)")
    lines.append("    );")
    lines.append("")

    for i in range(n_stages):
        ni1, mi1 = dims[i + 1]
        lines.append(f"    // Stage {i}: unpack out_{i} into L{i+1}'s act array + handshake")
        lines.append(f"    logic [{k_bits-1}:0] act_{i+1} [{ni1}];")
        lines.append(f"    logic                start_{i+1};")
        lines.append(f"    integer uq{i};")
        lines.append("    always_ff @(posedge clk or negedge rst_n) begin")
        lines.append("        if (!rst_n) begin")
        lines.append(f"            for (uq{i} = 0; uq{i} < {ni1}; uq{i} = uq{i} + 1) act_{i+1}[uq{i}] <= '0;")
        lines.append(f"            start_{i+1} <= 1'b0;")
        lines.append(f"        end else if (valid_{i}) begin")
        lines.append(f"            for (uq{i} = 0; uq{i} < {ni1}; uq{i} = uq{i} + 1)")
        lines.append(f"                act_{i+1}[uq{i}] <= out_{i}[{out_k_bits}*(uq{i}+1)-1 -: {out_k_bits}];")
        lines.append(f"            start_{i+1} <= 1'b1;")
        lines.append("        end else begin")
        lines.append(f"            start_{i+1} <= 1'b0;")
        lines.append("        end")
        lines.append("    end")
        lines.append(f"    wire [{out_k_bits * mi1 - 1}:0] out_{i+1};")
        lines.append(f"    wire                            valid_{i+1};")
        lines.append(f"    ankhdjet_layer_{sv_ident(layers[i+1].name)} u_l{i+1} (")
        lines.append(f"        .clk(clk), .rst_n(rst_n), .start(start_{i+1}),")
        lines.append(f"        .act(act_{i+1}),")
        lines.append(f"        .out_flat(out_{i+1}), .valid(valid_{i+1})")
        lines.append("    );")
        lines.append("")

    last = n_layers - 1
    lines.append(f"    assign out_flat = out_{last};")
    lines.append(f"    assign valid    = valid_{last};")
    lines.append("endmodule")
    lines.append("")
    lines.append("`default_nettype wire")

    parts.append("\n".join(lines) + "\n")
    return "\n".join(parts)


def emit_pipeline(
    layers: list[Layer],
    name: str = "pipeline",
    k_bits: int = 8,
    subcol_rows: int | list[int] = 64,
    scales_q: list[int] | None = None,
    scale_w: int = 16,
    q_frac: int = 8,
    activation: int = 0,
) -> str:
    """Emit a complete N-layer pipeline: N ankhdjet_layer_<L> modules plus one
    `ankhdjet_pipeline_<name>` top that chains them through between_layer with
    a one-cycle valid->start handshake between consecutive linear layers.

    All layers must be LINEAR with ternary weights, and layer i's M must
    equal layer i+1's N. The N-1 between_layer stages take per-stage
    SCALE_Q from `scales_q` (None -> unit scale 1<<q_frac for each).
    `subcol_rows` may be a scalar (applied to all layers) or a list with
    one entry per layer. Module names are sv_ident-legalized; layer
    names whose legalized identifiers collide are refused.
    """
    if not layers:
        raise ValueError("emit_pipeline requires at least one layer")
    for L in layers:
        if L.layer_type != LayerType.LINEAR:
            raise NotImplementedError(f"unsupported layer type {L.layer_type}")
        wt = L.weights["weight"]
        if wt.scheme != QuantScheme.TERNARY:
            raise ValueError(f"layer {L.name}: expected ternary, got {wt.scheme}")
        if wt.data.ndim != 2:
            raise ValueError(f"layer {L.name}: weight must be 2-D")
    check_unique([L.name for L in layers])
    n_layers = len(layers)
    n_stages = max(0, n_layers - 1)
    if scales_q is None:
        scales_q = [1 << q_frac] * n_stages
    if len(scales_q) != n_stages:
        raise ValueError(
            f"scales_q length {len(scales_q)} != n_layers-1 ({n_stages})")

    if isinstance(subcol_rows, int):
        subcol_list = [subcol_rows] * n_layers
    else:
        subcol_list = list(subcol_rows)
        if len(subcol_list) != n_layers:
            raise ValueError(
                f"subcol_rows length {len(subcol_list)} != n_layers ({n_layers})")

    dims: list[tuple[int, int, int]] = []
    for L in layers:
        n, m = L.weights["weight"].data.shape
        _, wacc = _column_widths(n, k_bits)
        dims.append((n, m, wacc))
    for i in range(n_stages):
        if dims[i][1] != dims[i + 1][0]:
            raise ValueError(
                f"layer {layers[i].name}.M={dims[i][1]} != "
                f"layer {layers[i+1].name}.N={dims[i+1][0]}")

    parts: list[str] = []
    for L, sc in zip(layers, subcol_list):
        parts.append(emit_layer_nor(L, k_bits=k_bits, subcol_rows=sc))

    n0, _, _ = dims[0]
    _, m_last, wacc_last = dims[-1]
    lines: list[str] = []
    lines.append(f"// Auto-generated pipeline: {name} ({n_layers} layers)")
    for i, L in enumerate(layers):
        ni, mi, wacci = dims[i]
        lines.append(f"//   L{i}: {L.name}  N={ni} M={mi} WACC={wacci}")
    lines.append("")
    lines.append("`default_nettype none")
    lines.append("")
    lines.append(f"module ankhdjet_pipeline_{sv_ident(name)} (")
    lines.append("    input  logic                       clk,")
    lines.append("    input  logic                       rst_n,")
    lines.append("    input  logic                       start,")
    lines.append(f"    input  logic [{k_bits * n0 - 1}:0]  act_flat,")
    lines.append(f"    output logic signed [{wacc_last * m_last - 1}:0] acc_flat,")
    lines.append("    output logic                       valid")
    lines.append(");")

    n_, m_, wacc_ = dims[0]
    lines.append(f"    wire signed [{wacc_ * m_ - 1}:0] acc_0;")
    lines.append("    wire                              valid_0;")
    lines.append(f"    ankhdjet_layer_{sv_ident(layers[0].name)} u_l0 (")
    lines.append("        .clk(clk), .rst_n(rst_n), .start(start),")
    lines.append("        .act_flat(act_flat),")
    lines.append("        .acc_flat(acc_0), .valid(valid_0)")
    lines.append("    );")
    lines.append("")

    for i in range(n_stages):
        ni, mi, wacci = dims[i]
        ni1, mi1, wacci1 = dims[i + 1]
        sq = int(scales_q[i])
        lines.append(f"    // Stage {i}: between_layer + handshake to L{i+1}")
        lines.append(f"    wire [{k_bits * mi - 1}:0] bl_out_{i};")
        lines.append("    between_layer #(")
        lines.append(f"        .M({mi}), .WACC({wacci}), .SCALE_W({scale_w}),")
        lines.append(f"        .Q_FRAC({q_frac}), .K({k_bits}),")
        lines.append(f"        .SCALE_Q({scale_w}'d{sq}),")
        lines.append(f"        .ACTIVATION({activation})")
        lines.append(f"    ) u_bl_{i} (")
        lines.append(f"        .acc_flat(acc_{i}),")
        lines.append(f"        .out_flat(bl_out_{i})")
        lines.append("    );")
        lines.append(f"    logic [{k_bits * mi - 1}:0] act_{i+1};")
        lines.append(f"    logic                       start_{i+1};")
        lines.append("    always_ff @(posedge clk or negedge rst_n) begin")
        lines.append("        if (!rst_n) begin")
        lines.append(f"            act_{i+1}   <= '0;")
        lines.append(f"            start_{i+1} <= 1'b0;")
        lines.append(f"        end else if (valid_{i}) begin")
        lines.append(f"            act_{i+1}   <= bl_out_{i};")
        lines.append(f"            start_{i+1} <= 1'b1;")
        lines.append("        end else begin")
        lines.append(f"            start_{i+1} <= 1'b0;")
        lines.append("        end")
        lines.append("    end")
        lines.append(f"    wire signed [{wacci1 * mi1 - 1}:0] acc_{i+1};")
        lines.append(f"    wire                                valid_{i+1};")
        lines.append(f"    ankhdjet_layer_{sv_ident(layers[i+1].name)} u_l{i+1} (")
        lines.append(f"        .clk(clk), .rst_n(rst_n), .start(start_{i+1}),")
        lines.append(f"        .act_flat(act_{i+1}),")
        lines.append(f"        .acc_flat(acc_{i+1}), .valid(valid_{i+1})")
        lines.append("    );")
        lines.append("")

    last = n_layers - 1
    lines.append(f"    assign acc_flat = acc_{last};")
    lines.append(f"    assign valid    = valid_{last};")
    lines.append("endmodule")
    lines.append("")
    lines.append("`default_nettype wire")

    parts.append("\n".join(lines) + "\n")
    return "\n".join(parts)


@dataclass
class PipelineConfig:
    """Parameters consumed by the area model + report tools.

    Captures the architectural knobs that affect area/throughput:
      tile_cols     columns sharing one popcount tree (1 = per-column)
      pipelined     inter-layer lockstep epoch scheduling
      bit_parallel  K-bit-wide AND gates (1-cycle vs K-cycle dot product)
      nor_array     1T NOR architecture (vs 2T/2BL legacy)
      subcol_rows   rows per sub-column when nor_array=True

    scales_q is a per-between-layer Q-format scale factor list
    (L-1 entries for an L-layer model); when None, unit scale is used.

    readout selects the NOR-array readout tier for estimation:
    "analog" (per-sub-column StrongARM comparators; matches the
    published SKY130 anchors) or "digital" (full-swing bitline
    samplers, the default tier). tile_cols/bit_parallel
    only apply to the legacy parallel-WL architecture
    (nor_array=False), which 1T cells cannot implement.
    """
    k_bits: int = 8
    scale_w: int = 16
    q_frac: int = 8
    scales_q: list[int] | None = None
    tile_cols: int = 1
    pipelined: bool = False
    bit_parallel: bool = False
    nor_array: bool = True
    subcol_rows: int = 64
    readout: str = "analog"

    def scale_q_for(self, between_layer_idx: int) -> int:
        if self.scales_q is None:
            return 1 << self.q_frac
        return self.scales_q[between_layer_idx]
