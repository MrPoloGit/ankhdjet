"""Bit-exact Python reference for the NOR-array CiROM architecture.

Architecture: mask-programmable NOR ROM (structural family with production
precedent WO2025217724A1 and academic precedent BitROM arXiv 2509.08542)
with 1 NMOS per ternary weight, via-1 drain customisation, column-tiled
sense, and row-sequential one-hot WL readout. The earlier
`ankhdjet.reference.ternary_matmul` modeled an unimplementable parallel-WL
abstraction; this file is the new ground truth.

Per-dot-product execution model:
    for sub in 0..ceil(N/SUBCOL_ROWS)-1:                    # column tiling
        precharge BL+ and BL- to VDD on every sub-column
        for r in 0..SUBCOL_ROWS-1:
            assert WL[sub*SUBCOL_ROWS + r] (one-hot inside sub-column)
            for each output column c:
                pos_hit[c] = (W[sub*SUBCOL_ROWS+r, c] == +1)   # via-1 to BL+
                neg_hit[c] = (W[sub*SUBCOL_ROWS+r, c] == -1)   # via-1 to BL-
                # multiply by activation bit for the current bit-slice k
                acc[c] += (pos_hit - neg_hit) * (act[r] >> k & 1) << k
            re-precharge before next row
The total cycle count per dot product is K * N (one row per cycle, K bit
slices). With column tiling at SUBCOL_ROWS rows per sub-column run in
parallel, the cycle count drops to K * ceil(N / SUBCOL_PARALLEL).
"""

from __future__ import annotations

import numpy as np

DEFAULT_SUBCOL_ROWS = 64  # as-built bitline depth; 256-row is the production-path geometry


def ternary_matmul_nor(
    W: np.ndarray,
    act: np.ndarray,
    k_bits: int = 8,
    subcol_rows: int = DEFAULT_SUBCOL_ROWS,
) -> np.ndarray:
    """Compute `W.T @ act` using NOR-array semantics.

    `W` is (N, M) ternary in {-1, 0, +1}. `act` is (N,) unsigned int with
    values in [0, 2**k_bits). Returns (M,) signed int = sum_i W[i,c] * act[i].

    The result is identical to a plain integer matmul; this reference exists
    to anchor the cycle-accurate behaviour of cirom_nor_subcol/tile.sv.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2D (got {W.shape})")
    if set(np.unique(W).tolist()) - {-1, 0, 1}:
        raise ValueError("W must be ternary in {-1, 0, +1}")
    n, m = W.shape
    if act.shape != (n,):
        raise ValueError(f"act shape {act.shape} != (N={n},)")
    if subcol_rows <= 0:
        raise ValueError("subcol_rows must be > 0")

    acc = np.zeros(m, dtype=np.int64)
    for sub_start in range(0, n, subcol_rows):
        sub_end = min(sub_start + subcol_rows, n)
        for r in range(sub_start, sub_end):
            row_w = W[r]
            for k in range(k_bits):
                a_bit = (int(act[r]) >> k) & 1
                if a_bit == 0:
                    continue
                # pos_hit = (row_w == +1) -> add 2^k
                # neg_hit = (row_w == -1) -> sub 2^k
                acc += (row_w.astype(np.int64) << k) * a_bit
    return acc


def binary_affine_matmul_nor(
    sign: np.ndarray,
    act: np.ndarray,
    scale_q88: np.ndarray,
    bias_q88: np.ndarray,
    k_bits: int = 8,
    subcol_rows: int = DEFAULT_SUBCOL_ROWS,
) -> np.ndarray:
    """Row-granularity affine-binary dot product (FBI-LLM's folded
    BinaryLinearWscales: one (scale, bias) pair per OUTPUT COLUMN, not
    per element -- see docs/binary_and_matmulfree_investigation.md).

    `sign` is (N, M) in {0, 1}: 1 routes to BL+, 0 to BL- -- every
    position is always one of the two vias, never floating, which is
    why this needs no new bitcell: it is a strict subset of the ternary
    mask vocabulary ternary_matmul_nor already models. `scale_q88` and
    `bias_q88` are (M,) signed Q8.8 fixed point (the project's existing
    per-tensor requantize convention, here one per column instead of
    shared).

    Reduction identity this relies on (sign_pm = 2*sign - 1 in
    {-1, +1}):
        dot[c] = sum_r act[r] * (bias[c] + sign_pm[r,c] * scale[c])
               = bias[c] * sum_r(act[r])
                 + scale[c] * sum_r(sign_pm[r,c] * act[r])
    The second term is exactly ternary_matmul_nor's existing hit-count
    accumulate (computed the same way below, bit-serially per
    sub-column); the first term is one array-wide running activation
    total shared by every column. Only the final affine combine is new
    per-column hardware -- M independent multiply-adds in place of one
    shared per-tensor scale.
    """
    sign = np.asarray(sign)
    if sign.ndim != 2:
        raise ValueError(f"sign must be 2D (got {sign.shape})")
    if set(np.unique(sign).tolist()) - {0, 1}:
        raise ValueError("sign must be binary in {0, 1}")
    n, m = sign.shape
    act = np.asarray(act, dtype=np.int64)
    if act.shape != (n,):
        raise ValueError(f"act shape {act.shape} != (N={n},)")
    scale_q88 = np.asarray(scale_q88, dtype=np.int64)
    bias_q88 = np.asarray(bias_q88, dtype=np.int64)
    if scale_q88.shape != (m,) or bias_q88.shape != (m,):
        raise ValueError(f"scale_q88/bias_q88 must be shape (M={m},)")
    if subcol_rows <= 0:
        raise ValueError("subcol_rows must be > 0")

    sign_pm = sign.astype(np.int64) * 2 - 1      # {0,1} -> {-1,+1}
    hit = np.zeros(m, dtype=np.int64)
    for sub_start in range(0, n, subcol_rows):
        sub_end = min(sub_start + subcol_rows, n)
        for r in range(sub_start, sub_end):
            row_s = sign_pm[r]
            for k in range(k_bits):
                a_bit = (int(act[r]) >> k) & 1
                if a_bit:
                    hit += row_s << k
    act_total = int(np.sum(act))
    # Q8.8 affine; floor division by 256 matches a signed >>> 8 in the RTL.
    bias_term = np.floor_divide(bias_q88 * act_total, 256)
    scale_term = np.floor_divide(scale_q88 * hit, 256)
    return bias_term + scale_term


def input_axis_affine_matmul(
    sign: np.ndarray,
    act: np.ndarray,
    scale_q: np.ndarray,
    bias_q: np.ndarray,
    q_frac: int = 8,
) -> np.ndarray:
    """INPUT-AXIS affine-binary dot product: scale/bias vary per INPUT
    ROW (activation index), broadcast over every output column --
    FBI-LLM's actual released convention, confirmed against a real
    checkpoint (see docs/binary_and_matmulfree_investigation.md and
    rtl/column/cirom_input_axis_affine_beh.sv). The opposite IR axis
    from `binary_affine_matmul_nor`'s row/per-tensor case.

        dot[c] = sum_r act[r] * (bias[r] + sign_pm[r,c] * scale[r])
               = sum_r(act[r] * bias[r])
                 + sum_r sign_pm[r,c] * (act[r] * scale[r])

    `sign` is (N, M) in {0, 1}. `scale_q`/`bias_q` are (N,) signed Q8.8
    (one per input row, not per output column). Not bit-serial over K
    -- this reference establishes the math is right, not a
    cycle-accurate accounting; see the RTL file's header for why this
    axis cannot reuse the cheap bit-serial AND-based hit count the
    output-axis case does.
    """
    sign = np.asarray(sign)
    if sign.ndim != 2:
        raise ValueError(f"sign must be 2D (got {sign.shape})")
    if set(np.unique(sign).tolist()) - {0, 1}:
        raise ValueError("sign must be binary in {0, 1}")
    n, m = sign.shape
    act = np.asarray(act, dtype=np.int64)
    if act.shape != (n,):
        raise ValueError(f"act shape {act.shape} != (N={n},)")
    scale_q = np.asarray(scale_q, dtype=np.int64)
    bias_q = np.asarray(bias_q, dtype=np.int64)
    if scale_q.shape != (n,) or bias_q.shape != (n,):
        raise ValueError(f"scale_q/bias_q must be shape (N={n},)")

    sign_pm = sign.astype(np.int64) * 2 - 1      # {0,1} -> {-1,+1}
    hit = np.zeros(m, dtype=np.int64)
    for r in range(n):
        hit += sign_pm[r] * int(act[r]) * int(scale_q[r])
    bias_acc = int(np.sum(act * bias_q))
    return np.floor_divide(hit + bias_acc, 1 << q_frac)


def cycles_per_dot_product(n: int, k_bits: int = 8,
                            subcol_rows: int = DEFAULT_SUBCOL_ROWS,
                            subcol_parallel: bool = True) -> int:
    """Total cycle count for one dot product.

    With `subcol_parallel=True` (default), all ceil(n/subcol_rows)
    sub-columns are sensed in parallel each cycle, so the wall-clock cycle
    count is k_bits * subcol_rows. Without it (single shared sense amp),
    the cycle count is k_bits * n.
    """
    if subcol_parallel:
        return k_bits * min(n, subcol_rows)
    return k_bits * n


__all__ = ["ternary_matmul_nor", "binary_affine_matmul_nor",
           "input_axis_affine_matmul",
           "cycles_per_dot_product", "DEFAULT_SUBCOL_ROWS"]
