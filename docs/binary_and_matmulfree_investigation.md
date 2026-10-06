# Binary weights and MatMul-free architectures: investigation

Scoping record for two things the frontend cannot ingest today, plus an
honest accounting of what's actually been proven to work versus what's
only shape-tested. Not a commitment to build either; a reference for
whoever picks this up.

## 1. Binary (2-value) weights — case study: FBI-LLM

Every layer of the stack assumes ternary `{-1, 0, +1}`, not "N discrete
levels":

- `ankhdjet/frontend/hf.py`: `decode_ternary` (274-298) rejects any
  tensor with more than 3 unique raw values and requires the positive
  and negative magnitudes to match; `decode_group_ternary` (304-324)
  and `is_group_ternary` (327-347) generalize the *scale grid* to
  per-group `±s`, but still require a `0` bucket in every group;
  `unpack_ternary_uint8` (254-271) hardcodes the 2-bit-packed
  `{-1,0,1}` convention.
- `ankhdjet/frontend/ir.py`: `QuantScheme` (28-30) has only
  `TERNARY`/`FLOAT`; `WeightTensor.__post_init__` (41-48) hard-asserts
  `vals.issubset({-1, 0, 1})` for any `TERNARY` tensor and has no field
  for a per-group bias/offset.
- `ankhdjet/backend/wmat.py` (33, 43, 55) re-asserts the same value set
  on emission, and the NOR-tile MAC cell in `rtl/`/`macro_grid.py` is
  built to accumulate ternary charge, not a generic level count.

Checked a real release against this: `LiqunMa/FBI-LLM_130M` (paper:
"Scaling Up Fully Binarized LLMs from Scratch via Autoregressive
Distillation"). Its `config.json` is plain `model_type: llama`,
`architectures: [LlamaForCausalLM]` — no new architecture dispatch
needed, unlike part 2. But the weight encoding doesn't fit the
existing ternary ladder at all:

- Training code (`qat/learnable_binarizer.py`, `BinaryLinearWscales`):
  `forward` does `w = STEBinary().apply(self.weight)` (sign, ±1), then
  `w = self.wscale * w + self.wbias` — an **affine** transform, not a
  symmetric scale. `to_regular_linear()` folds this into a plain
  tensor for release: `binary_weight * wscale.data + wbias.data`.
- Result: each row (or column/tensor, depending on `scaling_pattern`)
  of a released `q_proj`/`gate_proj`/etc. matrix holds **exactly two**
  distinct float values, `{bias − scale, bias + scale}`, with **no
  zero level** and no requirement that the two levels be symmetric
  around zero. `decode_ternary`/`decode_group_ternary` both require a
  zero bucket and reject anything with an unaccounted-for bias, so
  this pattern fails every existing detector as-is, not just the
  "too many unique values" check.
- Storage is `pytorch_model.bin` (pickle, float32), not `.safetensors`.

**Implemented** (software stack, 2026-10-04): items 1-3 below are done
and tested — `decode_binary_affine` (`ankhdjet/frontend/hf.py`),
`QuantScheme.BINARY` + `WeightTensor.bias` (`ankhdjet/frontend/ir.py`,
threaded through `ModelIR.save`/`.load`), wired into `load_weights`'s
decode ladder as the third rung after `decode_ternary`/
`decode_group_ternary`. `wmat.py`/`macro_grid.py` refuse `BINARY`
tensors with a precise diagnosis rather than mis-emitting them as
ternary. Tests: `tests/test_hf_frontend.py`
(`test_binary_affine_decodes_per_row`,
`test_binary_affine_decodes_sub_row_groups`), `test_ir_persistence.py`
(bias round-trip), `test_wmat_emit.py`
(`test_binary_scheme_refused_at_emission`).

What building this would take, and what turned out cheaper than
expected once the RTL was actually sketched out (below):
1. ~~New detector~~ **done**: per group, find exactly 2 unique values,
   recover `scale=(v2-v1)/2`, `bias=(v1+v2)/2`, store a sign bitmask
   plus the `(scale, bias)` pair.
2. ~~New `QuantScheme.BINARY`~~ **done**, with a `bias` field on
   `WeightTensor` that didn't exist before.
3. ~~Decode ladder wiring~~ **done**.
4. `wmat.py` emission + the RTL/NOR-tile backend: refusal is
   implemented; real emission is not, and the earlier framing here
   overstated the blocker for the common case. For FBI-LLM's dominant
   **row-granularity** pattern (one scale+bias per OUTPUT COLUMN,
   `scaling_pattern='row'`), the sign bit never floats a drain — it's
   always BL+ or BL-, which is a strict *subset* of the ternary mask
   vocabulary the signed-off cell already implements. A behavioral
   (simulation-only, not signed-off) model in
   `rtl/column/cirom_affine_binary_beh.sv` demonstrates this: it
   reuses `cirom_nor_tile` completely unmodified for the hit-count
   accumulate (`HAS_VIA_NEG = ~HAS_VIA_POS`), adds one array-wide
   running activation-total accumulator (new, but shared by every
   column, not per-column), and replaces the existing single shared
   per-tensor Q8.8 requantize multiply with M independent per-column
   Q8.8 multiply-adds — exactly the cost `docs/array_architecture.md`'s
   "shared scale multiplier" density note already identifies as the
   dominant wrapper lever, just paid per-column instead of once.
   Verified bit-exact against `ankhdjet.reference.nor.binary_affine_matmul_nor`
   via Verilator across 4 shapes (`tests/test_cirom_affine_binary_beh.py`).
   **This reference model is not wired into any signed-off chip/macro
   RTL, has not been through layout/DRC/LVS, and is not a claim that
   row-granularity binary support is production-ready** — only that,
   for this specific sub-case, no new bitcell is needed, which
   contradicts what this document originally said. The **sub-row
   group-granularity** case (bias/scale varying *within* a row, the
   general case `decode_binary_affine` also decodes) does not reduce
   this way: a per-group bias needs a per-group running activation
   total, not one shared across the whole row, i.e. new per-group
   periphery analogous to the analog variant's existing TriMLA group
   accumulators. That case remains unimplemented and is the part of
   the original "needs a hardware change" claim that still holds.
5. Caveat: this is FBI-LLM's specific affine-binary pattern. A
   hypothetical symmetric binary release with a zero bucket (unlikely
   by construction, since `sign()` never outputs exactly 0) would be
   a much smaller change confined to the frontend.

**Correction (2026-10-04), from testing against the real checkpoint:**
everything above item 5 was built on an *assumed* shape for the folded
checkpoint. Downloading `LiqunMa/FBI-LLM_130M` and inspecting the real
`pytorch_model.bin` disproved two parts of that assumption at once:

- **The checkpoint is never folded.** `weight` is the continuous
  training-time latent (every element of a column distinct — checked:
  2048 unique values in one `gate_proj` column), not the two-value
  `to_regular_linear()` output. The actual effective weight is
  `wscale*sign(weight) + wbias`, computed at forward time. There is
  nothing to *detect* — `decode_binary_affine`'s pattern search
  correctly returns `None` on it (verified), but that's because this
  isn't a decode-by-inference problem at all: the checkpoint names the
  affine parameters explicitly as `.wscale`/`.wbias` sibling tensors
  (the same convention as BitNet's `.weight_scale`), so the right move
  is a direct read, not a search.
- **The granularity is the other axis.** `q_proj.weight` is `(768,
  768)`, ambiguous on a square matrix, but `gate_proj.weight` is
  `(2048, 768)` with `wscale`/`wbias` shape `(1, 768)` — `768` matches
  `in_features`, not `out_features`. Confirmed on `down_proj` too
  (`(768, 2048)` weight, `(1, 2048)` scale/bias). So scale/bias vary
  **per input feature**, broadcast over every output row — FBI-LLM's
  `'column'` pattern, not the `'row'` pattern items 1-5 above assumed.
  This flips where the cheap accumulator lives: the per-row scale
  needs to multiply the *activation* before the row-sweep accumulate,
  not the hit-count *after* it, and there's no M-wide post-multiply to
  reuse the ternary tile's cheap bit-serial AND-based hit count for.

**Built for the corrected shape:**
- `ankhdjet/frontend/hf.py`'s `decode_binary_sign_sibling` — a direct
  read (`sign = raw >= 0`, `scale`/`bias` straight from the sibling
  tensors), wired into `load_weights` ahead of the ternary ladder
  whenever `.wscale`/`.wbias` siblings are present. Verified against
  three real layers of `LiqunMa/FBI-LLM_130M` (`gate_proj`,
  `down_proj`, `q_proj`): reconstructed `wscale*sign(weight)+wbias`
  matches the direct computation exactly on all three.
- `WeightTensor.scale_axis` (`ir.py`): `"output"` (the existing
  row/per-tensor/group-ternary convention, unchanged) vs `"input"`
  (this case) — explicit, not inferred from shape, since silently
  guessing the wrong axis here would silently compute the wrong model.
- `rtl/column/cirom_input_axis_affine_beh.sv` +
  `ankhdjet.reference.nor.input_axis_affine_matmul` — a new behavioral
  reference (not signed-off, not cycle-accurate: it sweeps one row per
  cycle rather than K-cycle bit-serial, since a wide signed per-row
  product doesn't decompose into the existing single-bit AND the way
  a plain activation bit does). Bit-exact via Verilator across 4
  shapes (`tests/test_input_axis_affine_beh.py`). The module's header
  states plainly that recovering bit-serial activation handling for
  this axis is an open question, not solved here.
**Update (2026-10-04): the full real-checkpoint pipeline now runs.**
`load_weights` only reads `.safetensors`, and FBI-LLM ships pickle
(`pytorch_model.bin`) — rather than add pickle loading to the trusted
frontend, `tools/convert_pickle_checkpoint.py` is a new, isolated,
one-time conversion script: it is the *only* place in the toolchain
that calls `torch.load` (with `weights_only=True`, PyTorch's
allowlisted-tensors-only mode — not a full sandbox, so only point it
at checkpoints from sources you trust), and it writes a plain
`model.safetensors` + copied `config.json`/tokenizer files that
`load_weights` reads completely normally. `load_weights` (and nothing
else) gained a small local-directory branch: if `repo_id` is already a
directory, it skips `snapshot_download` and reads straight from it.

Ran end to end against the real checkpoint:
`convert_pickle_checkpoint.py` on the downloaded
`LiqunMa/FBI-LLM_130M` snapshot, then
`load_weights('/path/to/converted', ...)`. Result: all 12 layers'
7 projections each (84 total: q/k/v/o/gate/up/down) decode as
`BINARY` with `scale_axis="input"` through the ordinary
`parse_hf_config`/`block_projections` path — no architecture-specific
code was needed, confirming the earlier guess that FBI-LLM's plain
llama-shaped config and module names "just work." `lm_head` correctly
falls through to the existing non-ternary placeholder path (it's kept
full-precision, same convention as BitNet's tied bf16 head). Spot
reconstruction check on a real layer
(`model.layers.0.mlp.gate_proj`): the IR's `sign`/`scale`/`bias`
reproduce `wscale*sign(weight)+wbias` with **zero** numeric
difference against the original tensors. This is now a permanent,
gated regression
(`tests/test_hf_frontend.py::test_fbi_llm_sign_sibling_decodes_through_load_weights`,
skips cleanly without a local conversion present, mirroring the
existing Qwen3.6 real-row test's gating convention).

**Update (2026-10-04, part 2): items 1-4 above are now done.**

1. **`scale_axis` is now recorded through the manifest.** `GridManifest`
   gained a `scale_axis` field (`macro_grid.py`), set from
   `wt.scale_axis` and written into `manifest.json` alongside
   `scale_bias_file`, so a downstream consumer reading `scale_bias.npz`
   back no longer has to guess which axis it's for.
   `tests/test_wmat_emit.py::test_binary_manifest_records_input_axis`
   covers it.
2. **`verilog.py` now has a real emitter for the input axis**:
   `emit_layer_input_axis_affine()` bakes a real layer's per-row
   `scale`/`bias` as Q8.8 constants into `cirom_input_axis_affine_beh`
   (passed as the `HAS_VIA_POS`-style parameter below, not a port —
   see point 4), followed by one `requantize` instance per column at
   unit scale (reused unmodified; it's doing only the relu+clip
   activation stage here, since the affine combine already happened
   inside the axis module) for the standard K-bit output bus
   convention. Bit-exact via Verilator across 3 shapes
   (`tests/test_input_axis_layer_emit.py`).
3. **The "open" bit-serial question is resolved, not left open**:
   it doesn't help, and the module's existing structure (one
   time-shared multiplier, N row-cycles, M simple conditional
   adders) is the right choice, not a placeholder. Reasoning: a
   bit-serial decomposition would need, per cycle, a full N-wide
   signed adder tree per column gated by that cycle's activation bit
   (since the per-cell quantity `sign_pm[r,c]*scale[r]` is a real
   signed number, not ±1, bit-serial-izing it doesn't turn it into a
   cheap AND the way a single activation bit does for the ternary
   case) — K cycles of that is strictly more total work than N cycles
   of one shared multiply + M cheap adds. The output-axis case's
   bit-serial AND-based hit count is cheap specifically *because* the
   per-cell weight is ±1; losing that is the real, structural cost of
   the input axis, and no cycle restructuring recovers it.
4. **A real decoded FBI-LLM layer now runs through the RTL.**
   `tests/test_fbi_llm_real_layer_rtl.py` decodes
   `model.layers.0.self_attn.q_proj` via `load_weights` on the
   converted checkpoint, slices it to 128×64 (same convention as the
   existing real-weight BitNet test, `test_bitnet_block0_all_projections.py`
   — Verilator's per-literal width limit rejects the full 768×768
   mask even as a parameter override, so real-weight RTL validation in
   this codebase is always a slice, never the full matrix), and
   confirms the generated RTL's output matches
   `input_axis_affine_matmul` bit-exact on real sign/scale/bias with
   synthetic activations.

   One fix discovered along the way:
   `cirom_input_axis_affine_beh.sv`'s sign mask had to move from an
   input *port* to a `HAS_VIA_POS`-style compile-time *parameter*
   (matching `cirom_nor_tile`'s convention) — Verilator's width limit
   for a literal in a bare port-connection expression is far lower
   than for a parameter override, and the mask is mask-programmed
   (fab-time-fixed) anyway, so this is also the architecturally
   correct representation, not just a workaround.

**Update (2026-10-04, part 3): both remaining items are done.**

- **Multi-layer pipeline chaining**: `emit_pipeline_input_axis()`
  chains N input-axis layers through a one-cycle valid->start
  handshake, mirroring `emit_pipeline`'s structure for the ternary
  case — but simpler, since no `between_layer`-equivalent stage is
  needed between stages: each `emit_layer_input_axis_affine` module
  already applies its own activation+clip internally, so a layer's
  `out_flat` bus just gets unpacked back into the next layer's
  `act[N]` array on the handshake register. Bit-exact via Verilator
  on 2- and 3-layer synthetic chains
  (`tests/test_input_axis_pipeline_emit.py`).
- **Full-model compile against the real checkpoint**: ran
  `ankhdjet.backend.macro_grid.emit_model` against the real decoded
  `LiqunMa/FBI-LLM_130M` (all 84 projection layers, `lm_head` skipped
  as off-fabric, same convention as BitNet's tied bf16 head). Result:
  5,184 macros, 84,934,656 weights (exact match: 12 × (4×768×768 +
  2×2048×768 + 768×2048)), **0 padded** (64×256 macro geometry
  divides every FBI-LLM_130M projection shape exactly), 85.27 MB of
  mask programs. Then ran the project's independent
  `ankhdjet.backend.verify.verify_model` audit — the same bit-for-bit
  reassembly-from-disk check `docs/results.md` reports for BitNet —
  and it initially reported false failures on all 84 layers, because
  `verify.py` predates `BINARY` and compared the IR's `{0,1}` sign
  mask directly against the emitted `{-1,+1}` mask without the sign
  mapping, and also applied the ternary-only "padding must be
  all-zero" check to a scheme where every position is always BL+/BL-
  (never floating), so that check doesn't apply at all. Fixed both in
  `verify.py` (now scheme-aware); re-ran: **84/84 layers bit-exact
  from disk.** Both fixes are covered by new synthetic tests in
  `tests/test_verify.py`, and the full real-checkpoint run is a
  permanent gated regression
  (`tests/test_fbi_llm_full_model_compile.py`).

Physical signoff remains explicitly out of scope, unchanged from the
start of this investigation.

**Update (2026-10-04, part 4): a real synthesis-quality tile
controller, modeled on the signed-off `cirom_dig_ctrl.sv`.**

Everything built in parts 1-3 above (`cirom_input_axis_affine_beh.sv`,
`emit_layer_input_axis_affine`, `emit_pipeline_input_axis`) is
explicitly a *behavioral reference*: 64-bit casts everywhere, unpacked
array ports, built to prove the math is right as cheaply as possible
in simulation, not to go through synthesis. Taking this to an actual
SKY130 chip flow needs something closer to `cirom_dig_ctrl.sv` (the
signed-off Darga tile's full ternary MAC on die) — careful bit widths,
host-loadable activation store, no shortcuts.

`rtl/tt_digital/cirom_dig_ctrl_affine.sv` is that: modeled directly on
`cirom_dig_ctrl.sv`'s FSM skeleton and host interface, extended with
the one new thing this case needs (a per-row Q8.8 scale/bias multiply
folded into the row sweep, plus one shared bias accumulator added back
in before streamout — see the file's header for the derivation).
Feature set is deliberately reduced from `cirom_dig_ctrl.sv` (only the
on-die activation store, only MVM mode; the streamed/raw-row-read
modes are omitted) to keep the first version tractable. One real width
bug surfaced and got fixed while building it: the result byte-stream
mux initially assumed exactly 2 bytes per accumulator
(`cirom_dig_ctrl.sv`'s own fixed convention, valid there since ternary
hit-counts stay under 16 bits) — this design's accumulated quantity is
a sum of `ACT_W × SCALE_W` products, which needs more than 2 bytes at
real sizes, so the mux is now generic over `GRPB = ACC_W/OUT_W` bytes.

Validated two ways:
- **Functional regression** (`tests/test_cirom_dig_ctrl_affine.py`):
  the controller driving a real array interface
  (`cirom_array_beh`, the same behavioral macro stand-in the
  signed-off chip's own bench uses — not a new stand-in), compared
  bit-exact against `input_axis_affine_matmul` across 3 shapes
  including multi-pass column grouping (`N_ACC < N_COLS`).
- **Synthesis**: `yosys synth` (generic cells; no SKY130 standard-cell
  mapping, since `PDK_ROOT` isn't set up in this environment — see
  below) elaborates cleanly at both a small test size (2,557 cells)
  and at a real FBI-LLM layer's full dimensions (`N_ROWS=768,
  N_COLS=256, N_ACC=32`: 12,669 cells, ~24s, one expected benign
  warning — Yosys converting the small `acc[]` array to discrete
  registers, the same thing it does for `cirom_dig_ctrl.sv`'s
  identically-shaped array). `check` reports 0 problems at both sizes.

**Update (2026-10-04, part 5): the chip-top RTL and LibreLane config
now exist, and an actual LibreLane invocation narrowed the remaining
blockers to two concrete, known things.**

- `rtl/chip/cirom_chip_digital_affine.sv`: a new chip top mirroring
  `cirom_chip_digital.sv`'s structure, wiring `cirom_dig_ctrl_affine`
  to the exact same hardened array macro
  (`macro_array_pc_64x32_test0`) via the same `` `ANKHDJET_ARRAY_MODULE ``
  binding — no new macro needed, confirming again that the array
  itself is unaffected by this quantization scheme. `SCALE_Q`/`BIAS_Q`
  are real values: the first 64 input rows of
  `LiqunMa/FBI-LLM_130M`'s block-0 `self_attn.q_proj`, read directly
  off the converted checkpoint. Synthesizes cleanly through `yosys`
  with the real macro's blackbox Verilog in the design (16,130 cells
  in the full hierarchy, macro correctly kept as an opaque blackbox,
  one benign warning, same as the standalone controller check).
- `librelane/cirom_chip_digital_affine/`: a full config set
  (`config.json`, macro placement, pin order, SDC — the SDC is
  `cirom_chip_digital.sv`'s own file, reused verbatim since it's
  generic) mirroring `librelane/cirom_chip_digital/` exactly, except
  `DIE_AREA` (1200×400 vs. the original 850×300 — a rough, unoptimized
  guess reflecting that this periphery is meaningfully bigger than the
  original's simple raw-readout FSM, not a floorplan result).

**Actually invoking `run_librelane.sh` was informative, not just
blocked:**
- The SKY130 PDK **auto-downloaded** — LibreLane fetches it itself on
  first use, no separate `ciel`/`PDK_ROOT` setup needed. That removes
  one of the blockers listed in the previous update entirely.
- The config loaded and validated cleanly (the same deprecation
  warnings `cirom_chip_digital`'s own config produces — nothing
  specific to this design).
- It failed at exactly one point: the array macro's **GDS file
  doesn't exist**
  (`cell/sky130/macro/build/macro_array_pc_64x32_test0.gds`) — it's a
  build artifact of `tools/gen_macro.sh`, not checked into the repo,
  and generating it needs Magic.

**What's left, now narrower:**
1. `magic` (to generate the macro's GDS via `tools/gen_macro.sh 64 32
   test0`) and `openroad`/`opensta` (for the flow itself) are not
   installed. Checked: none are available via Homebrew. Both are
   `nix build`s from source on this darwin machine — confirmed via
   `nix build --dry-run`, which showed Magic alone pulling in a
   from-source Python 3.14 + Tcl/Tk + xcbuild closure, and OpenROAD a
   much larger one (Qt5, SWIG, dozens more). Realistically tens of
   minutes to hours combined, with real odds of a darwin-specific
   build hiccup — a deliberate choice to stop and ask before running,
   not a limitation worth hiding.
2. Once those exist: run `tools/gen_macro.sh 64 32 test0`, then
   `bash librelane/cirom_chip_digital_affine/run_librelane.sh`, and
   see what synthesis/floorplan/PnR/DRC/LVS/STA actually report — none
   of that has run yet.

**Update (2026-10-04, part 6): all of item 1-2 above turned out to be
solvable in minutes, not hours — a pre-pulled Docker image
(`ghcr.io/librelane/librelane:3.1.0.dev3`) already had the entire
toolchain (OpenROAD, OpenSTA, Magic, Netgen, Yosys, KLayout) built in.
The nix-build estimate above was real for the from-source path, but
unnecessary once the right image was found. A real flow run happened,
and real signoff numbers came back — not a clean pass, but a genuine
first attempt, not a blocked one.**

Generated the macro's GDS via `tools/gen_cells.sh` + `tools/gen_macro.sh
64 32 test0` inside the container (mounting the repo and a host-cached
SKY130 PDK snapshot): DRC=0 throughout, LVS clean at the macro's own
level. Ran the full flow
(`librelane/cirom_chip_digital_affine/config.json`) end to end,
~49 minutes: synthesis → floorplan → placement → CTS → routing all
completed (701mm total wire length, ~113K vias). Final signoff:
**DRC passed**; **LVS failed** (2052 errors, cascading from a device-count
mismatch that originates inside the macro's own hierarchical
comparison, not obviously from the new periphery RTL); **STA setup
violations**, but only in the slow process corners. Two real,
unrelated config bugs were found and fixed along the way (an
overly-broad `Checker.YosysUnmappedCells` false-positive on the
intentional hard macro, downgraded to a warning via
`ERROR_ON_UNMAPPED_CELLS: false`; and an ambiguous pin-order regex,
`result.*` matching `result_valid` too — fixed to `result\[.*\]`).

**The PDK-version-mismatch hypothesis for the LVS failure was wrong.**
The macro's GDS had been generated against one cached SKY130 snapshot
(`8afc8346...`) while the flow itself auto-downloaded and used a
different one (`74c0e6b...`); the two snapshots' Magic tech files do
differ in real ways (confirmed by diff), which looked like a
plausible, well-motivated root cause. Regenerated the macro against
the matching `74c0e6b` snapshot and reran the full flow: **the LVS
error count was exactly identical (370 power grid violations, 2052 LVS
errors) to the previous run.** Identical counts across a changed input
is strong evidence the regeneration had no effect on the actual
mismatch — this was not the root cause, or at least not the whole of
it.

What's actually indicated instead: `docs/lvs_root_cause.md` documents
a prior investigation into exactly this failure *class* in this
project — custom macro pins not binding correctly during Magic
extraction (abstract vs. flat extraction mode, pin label datatype,
geometry that collapses during abstraction) — for the analog chip's
sense macro. The symptom pattern matches closely: netgen's report
shows it "flattening unmatched subcells" inside the macro's own
hierarchy before the device-count mismatch appears, meaning the
discrepancy may originate in how *this* macro extracts in *this*
chip's specific wiring, not in the new periphery logic at all. A
control test — running the already-signed-off `cirom_chip_digital`
(same macro, unchanged) through this exact same Docker toolchain — is
in progress to isolate whether this is a bug specific to the new
design or a toolchain/extraction-mode regression that would also hit
the known-good chip.

**Resolved: the control test came back, and it settles the
question.** Ran `librelane/cirom_chip_digital/config.json`
(the already-signed-off, taped-out chip, completely unchanged) through
the identical Docker toolchain. Result: **DRC passed; LVS failed, 2088
errors** — the same magnitude and the same pattern (DRC clean, LVS
broken) as the new `cirom_chip_digital_affine` design's 2052. A chip
with no changes, that has real signoff history, fails the same way
under this toolchain. That rules out the new chip-top RTL, the new
LibreLane config, and the regenerated macro as the cause of anything —
**this is a toolchain/extraction-version regression** (most plausibly
in how this specific Docker image's bundled Magic/Netgen versions
interact with the macro's hierarchical SPICE comparison, matching
`docs/lvs_root_cause.md`'s documented failure class for this project:
custom macro pins not binding correctly during extraction), not a
defect in anything built during this investigation. Synthesis,
floorplan, placement, CTS, and routing are all real, clean results
against real SKY130 for a design that didn't exist before this
session. The LVS gap is a pre-existing toolchain compatibility issue
that would need its own investigation (likely: try an older/different
LibreLane Docker image pin, or work through the `lvs_root_cause.md`
extraction-mode playbook against this specific image's Magic version)
— out of scope to chase further without confirming which exact
tool-version combination the chip was originally signed off against.

**Update: tried the obvious next thing, and it wasn't the fix
either.** `ankhdjet`'s own `pyproject.toml` pins `librelane==3.0.3`,
not the `3.1.0.dev3` Docker image used above — a real, specific
version mismatch, not a guess. Pulled `ghcr.io/librelane/librelane:3.0.3`
(kept as a separate, additional image alongside `3.1.0.dev3`, not a
replacement) and reran the same control test
(`cirom_chip_digital`, unchanged) through it:

| Image | Magic | Netgen | DRC | LVS |
|---|---|---|---|---|
| `librelane:3.1.0.dev3` | 8.3.674 | 1.5.320 | Passed | Failed, 2088 errors |
| `librelane:3.0.3` (ankhdjet's pinned version) | 8.3.623 | 1.5.316 | Passed | Failed, 2052 errors |

**Still fails, at the version ankhdjet actually depends on.** This
rules out "wrong LibreLane point release" as the explanation — the
LVS break is present across at least two LibreLane/Magic/Netgen
combinations, on the unmodified signed-off chip, which means the real
discrepancy is most likely the **SKY130 PDK snapshot** itself (both
runs auto-fetch whatever snapshot each LibreLane version defaults to
today, which is almost certainly newer than whatever snapshot was
used when this chip was actually signed off — PDK device models and
extraction decks do change between skywater-pdk/open_pdks releases),
not the EDA tool binaries. Pinning the PDK snapshot used at original
signoff (if that version is recoverable — not yet attempted) is the
next concrete thing to try, rather than further LibreLane version
hunting. Not resolved as of this update; both Docker images are kept
available side by side for whoever picks this up next.

**Update: real LVS debugging, following `docs/lvs_root_cause.md`'s own
playbook, with real progress but still not resolved.**

- **Checked whether the recoverable signoff metadata exists at all:
  it doesn't.** The original per-commit history was squashed into a
  single "public release" commit (`1dce69a`); `docs/results.md`
  records signoff *results* (timing, DRC/LVS pass) but never a PDK
  snapshot hash or tool version. Trial-and-error snapshot hunting
  would mean guessing blindly across historical `open_pdks` releases,
  each guess costing another ~15-50 min run — not pursued without
  deciding that's worth it first.
- **LibreLane's own `--smoke-test` passes LVS cleanly** in this exact
  Docker image. This matters: it proves Magic, Netgen, and OpenROAD
  are not broken in this environment for an ordinary design. The
  failure is specific to this chip's custom hard macro, not the
  toolchain as a whole.
- **Found the actual mechanism.** Netgen's LVS report
  (`70-netgen-lvs/reports/lvs.netgen.rpt`) shows, for the array macro
  specifically: `Class macro_array_pc_64x32_test0 (0): Merged 1984
  parallel devices`, then `sky130_fd_pr__nfet_01v8 (2048->64)` on the
  layout side against `(2048)` (no merge) on the schematic side.
  **2048 → 64 is exactly the array's row count** (64 rows × 32
  columns) — every bitcell in a row is extracting as topologically
  identical to netgen's matcher, collapsing all 32 column-instances
  per row into one, while the independently-generated `.lvs.spice`
  reference keeps all 2048 textually distinct. This is a real,
  specific, reproducible mechanism, not a vague "LVS is broken."
- **Ruled out, with direct evidence, not guesses:**
  - *Wrong netgen setup file*: LibreLane's LVS step doesn't hardcode a
    different netgen config the way it first looked — its bundled
    `netgen/setup.tcl` is a one-line wrapper,
    `source $::env(NETGEN_SETUP)`, which does read the `NETGEN_SETUP`
    config variable. Overrode it explicitly to the PDK's own
    `sky130A_setup.tcl` (confirmed the override actually took effect
    in the run's recorded config) and reran: **identical result**,
    2088 errors, the same 1984-device merge. The PDK's own setup file
    has the same behavior LibreLane's default does — this was never
    actually the divergence.
- **The one concrete difference still standing**: `gen_macro.sh`'s
  own standalone netgen self-check (which passes on this exact macro)
  calls `netgen -batch lvs <circuit1> <circuit2> <setupfile>
  <logfile>` with no extra flags. LibreLane's internal LVS step always
  adds `-blackbox -json` — hardcoded in its Python (`netgen.py`), not
  exposed as a config variable, so it can't be turned off from
  `config.json` the way `NETGEN_SETUP` could. Netgen's `-blackbox`
  mode changes how it falls back when subcircuits don't cleanly
  resolve, and is the most plausible remaining mechanism for why the
  same macro, same PDK, same setup file passes standalone but not
  embedded in the full chip's LVS run. **Not yet tested** — would
  require patching/mounting a modified copy of LibreLane's bundled
  `netgen.py` or its invocation into the container, a real, separate
  experiment, not a config change.

**Honest summary of effort spent**: four full LibreLane runs (~15-50
min each), two ~5GB Docker images, one disk-space cleanup, and direct
inspection of the raw LVS netlist comparison — three specific,
testable hypotheses raised and disproven with evidence (PDK version,
LibreLane version, netgen setup file), one real, specific mechanism
identified (row-wise bitcell merging via `-blackbox`'s likely
interaction with repeated-topology devices), and one concrete,
scoped-but-unattempted next experiment (patch out `-blackbox`). This
is now squarely the same category of effort `docs/lvs_root_cause.md`
documents taking real, sustained work for a *different* macro in this
project — not a quick-fix situation.

`config.json` for `ridger/MMfreeLM-370M`: `model_type: hgrn_bit`,
`architectures: [HGRNBitForCausalLM]`, with fields (`attn_mode:
fused_recurrent`, `num_heads`, `hidden_ratio`, `expand_ratio`,
`conv_size`, `share_conv_kernel`, `use_lower_bound`) that share almost
nothing with the llama-family or Qwen3.5/3.6 shapes the frontend
already understands. This is a linear-RNN token mixer (HGRN:
Hierarchically Gated Recurrent Network) with no self-attention matmul
at all, plus a ternary MLP.

- `parse_hf_config` (hf.py, 129-165) has no dispatch on `model_type`
  or `architectures` at all — it's pure field-presence extraction
  assuming llama-shaped keys (`hidden_size`, `num_attention_heads`,
  `intermediate_size`, `vocab_size`), with one special-cased branch
  for Qwen3.5/3.6 hybrid attention: `layer_types` entries restricted
  to exactly `{"full_attention", "linear_attention"}` (138-146),
  feeding `block_projections` (168-198) which builds the gated-delta
  projection set (`in_proj_qkv/z/a/b`, `out_proj`). That's shaped for
  Qwen's gated-delta-net, not HGRN's recurrent gate + short causal
  conv — different module names, different math.
- Outcome today: `parse_hf_config` raises `KeyError` on the missing
  llama-shaped keys before it gets anywhere near weight loading. Even
  if those were patched around, `load_weights`'s missing-projection
  check (hf.py, ~422-427) would fail next — HGRN-bit's state dict uses
  its own module names for the recurrent mixer and ternary MLP, not
  `self_attn.q_proj`/`mlp.gate_proj`.

What building this would take:
1. A genuinely new config-parsing branch mapping HGRN-bit's fields
   (`hidden_ratio`, `expand_ratio`, `num_heads`, `conv_size`) to IR
   dimensions, instead of assuming `num_attention_heads`/
   `intermediate_size` exist.
2. A new `block_projections` variant for the recurrent-gate mixer +
   short conv, built from the real checkpoint's actual state-dict key
   names (not yet inspected against real weights — only the config
   was fetched, not the tensors).
3. The MLP/mixer weights are plausibly still ternary per the paper's
   own quantization scheme, which is good news: once the config and
   module-name mapping exist, the *existing* ternary decode ladder
   might apply directly to the value tensors with no new numeric
   decoder. This is unverified — no real HGRN-bit checkpoint's tensor
   values have been downloaded and inspected in this investigation.
4. The Qwen3.6 work already introduced a "recurrent state kept beside
   the KV cache" concept for linear-attention blocks — conceptually
   reusable for HGRN's recurrent hidden state, but the wiring is new:
   today's `layer_types` dispatch only recognizes
   `{"full_attention", "linear_attention"}`, and HGRN-bit is uniformly
   recurrent (no hybrid per-block split), so even the enum doesn't fit
   without a new case.
5. Backend: since the weights are plausibly still 3-level ternary (not
   the FBI-LLM affine-binary case), this is likely almost entirely a
   **frontend parsing problem** — a new config/module-name mapping —
   not a hardware encoding problem. Unlike part 1, no RTL/MAC change
   is expected to be required, pending confirmation against real
   weights.

## 3. What's actually confirmed working today

Only one model has true end-to-end proof; most of the rest of the
"supported" surface is shape-tested against synthetic configs, not
decoded from real checkpoints, in this repo's own test suite.

**Hardware-validated (real weights → IR → RTL, bit-exact):**
`microsoft/bitnet-b1.58-2B-4T` (2.084B ternary fabric weights,
packed-u8 storage, llama-family BitNet). Evidence:
`tests/test_microsoft_bitnet_pytorch_match.py` decodes a real `q_proj`
tensor via `load_weights` and matches `transformers`' own
`unpack_weights`+`F.linear`; `tests/test_bitnet_block0_all_projections.py`
takes all 7 block-0 projection shapes through `emit_layer_nor` →
Verilator, bit-exact against the Python NOR reference;
`tests/test_microsoft_bitnet_nor_layer.py` does single-layer NOR-tile
validation. `docs/results.md` records the full pipeline going further
still: all 210 layers compiled to 128,400 mask-program macros
(2,084,044,800 weights, 2.11 GB `.wmat`), an independent `ankhdjet
verify` pass reassembling all 210 layers bit-for-bit against the
checkpoint, and a 3-way "architectural twin" (transformers reference,
from-scratch numpy, and the fabric's own bit-serial decomposition)
agreeing token-for-token on a real greedy decode. **This is the only
model taken all the way to mask-program/RTL validation.**

**Config/shape-tested only (never decoded from real weights in-repo):**
A synthetic Qwen3.6-27B-class hybrid-attention config
(`tests/test_hf_frontend.py:108-153`) proves the parser produces
correct shapes and a ~24.3B-param count — no real tensor is decoded. A
test that *would* check a real Qwen3.6 tensor
(`test_group_scaled_real_row_from_the_qwen36_release`) exists but
`pytest.skip`s unless a scratch `.npy` file is already present
locally — no evidence in CI or docs that it has been run for real. A
generic llama-family config (hidden_size=2560, 30 layers,
`test_llama_family_config_is_unchanged`) is likewise synthetic-only.

**`docs/results.md`'s "Ternary checkpoint coverage" table** — the
closest thing to a real coverage report, 12 real HF releases across 5
orgs, 0.113B–6.979B params:

| Evidence level | Count | Releases |
|---|---|---|
| Full real-tensor decode (some with full emission/audit) | 5 | TriLM_190M_Unpacked (+ full 6,912-macro emission, 112/112 audit), Falcon3-1B-1.58bit, Falcon-E-1B-Base, **bitnet-b1.58-2B-4T** (+ full 210-layer emission/verify/twin), Llama3-8B-1.58-100B-tokens |
| Config-only (weights never fetched/decoded) | 6 | bitnet_b1_58-large (679M), Falcon-E-3B-Base, bitnet_b1_58-3B, TriLM_3.9B_Unpacked, Falcon3-7B-1.58bit, Ternary-Bonsai-8B-unpacked |
| Refused | 1 | Ternary-Bonsai-1.7B-unpacked (group-scaled fp16, "per-group scales exceed the per-tensor requantize contract") |

**This table is stale on the refused row.** It hasn't been regenerated
since the public-release squash commit (`1dce69a`); commit `82feab7`
("Decode group-scaled ternary storage in the frontend...") was written
specifically to handle exactly the group-scaled-fp16 pattern
Ternary-Bonsai uses. Both Ternary-Bonsai releases would very likely
decode successfully today — the table needs a re-run, not a read, for
that row and probably for the other `config-only` rows now that more
of the ladder exists.

**Bottom line on sizes:** real-tensor decode is demonstrated from
0.113B (TriLM_190M) up to 6.979B (Llama3-8B-1.58) params. Everything
above is config-only or stale. The single size taken to hardware/RTL
validation is 2.08B ternary fabric weights
(bitnet-b1.58-2B-4T) — nothing larger has been pushed that far.
~3.9B params and below estimate inside one reticle-class ASAP7 die;
~7B releases estimate just past one reticle — that's an area-estimator
finding, not an ingestion limit.

**CI caveat:** the real-checkpoint tests gate on a local HF cache
(`_hf_cache_has_repo`) and skip cleanly when it's absent. Neither
`nightly.yml` nor `fast-check.yml` references an HF model name or
`HF_HOME` — none of this is fetched or enforced in CI. "Confirmed"
above means confirmed when someone runs it locally with the cache
populated, not continuously verified per commit.

All of this — part 3 included — is still inside the ternary family.
Binary (part 1) and non-attention architectures (part 2) are gaps
nothing in the current test suite or results table touches.

## Update: LVS root cause found — a real layout-hierarchy bug, not a tool/flag issue

Following up on the earlier "three hypotheses disproven" update (PDK
version, LibreLane version, `NETGEN_SETUP` override — all tested with
direct evidence and all ruled out), the leading remaining hypothesis
at that point was that LibreLane's `Netgen.LVS` step hardcodes
`-blackbox -json` (confirmed in its Python source,
`librelane/steps/netgen.py:255`) while `tools/gen_macro.sh`'s own
passing self-check never passes those flags. That hypothesis is now
**also ruled out** — not by testing the flags directly, but because
tracing the actual net-level mismatch in `lvs.netgen.rpt` found the
real mechanism, and it has nothing to do with netgen's comparison
flags.

**The mechanism.** In the mismatch report for
`macro_array_pc_64x32_test0`, net `BLP_0` (one of the macro's real
per-column bitline ports) shows, on the layout (circuit1) side: 1
pfet, **0 nfets**. On the schematic (circuit2) side: 1 pfet, **32
nfets** (the 32 bitcells of that column, across all rows minus the
ones using BLN). The 64 `WL_<r>` nets show the mirror symptom: 1
nfet gate connection on the layout side vs. 32 on the schematic side
— for every one of the 64 rows, uniformly. 64 rows x (32-1=31
collapsed per row) matches the earlier-observed 2048->64 "merged
parallel devices" count exactly.

Tracing why: the macro's extracted hierarchical SPICE
(`68-magic-spiceextraction/cirom_chip_digital.spice`) declares the
leaf bitcell as `.subckt P2_bitcell_v4 G S VSUBS` — **three ports**.
Internally: `X0 S G D VSUBS sky130_fd_pr__nfet_01v8` — the transistor's
real drain terminal is wired to a net literally named `D`, which is
**not one of the subckt's three ports**. It's stranded inside the
cell boundary: never promoted out to the array level, so it can never
reach the column's real `BLP_c`/`BLN_c` net in the extracted graph.

This is directly confirmed against the project's own canonical
single-cell reference,
`cell/sky130/bitcell_v4/bitcell_v4_schematic.spice`:
`.subckt bitcell_v4 D S G` — **drain is a real, named, exposed port**
there. So somewhere between the clean standalone bitcell and the
macro's hierarchical extraction inside the full chip, the drain port
gets dropped.

**Why the macro's own self-check (`tools/gen_macro.sh`) never sees
this:** its netgen-LVS step does `flatten chk_$NAME` on the macro's
own `.mag` *before* extracting — collapsing all hierarchy first, so
Magic derives every net from flat physical geometry and the
port-promotion question never arises. The full-chip LibreLane flow
does the opposite: Magic's `Magic.SpiceExtraction` step extracts the
whole chip *without* flattening the macro (by design — flattening a
macro at full-chip scale is what hierarchical macro support exists to
avoid), so hierarchical port-promotion has to work correctly across
every cell boundary the net crosses, and for this macro it doesn't.

**Why hierarchy survives into the chip flow at all, even though
`gen_macro_array_pc.tcl` calls `flatten -dolabels $MACRO_NAME`:** that
flatten call runs *before* the GDS write, and the macro's build
pipeline still shows real sub-cell structure
(`P2_bitcell_v4`, `P2_v4_array_64x32_test0`) when the chip-level flow
reads its GDS. `gen_macro.sh`'s self-check re-flattens the already
"flattened" macro again (`flatten chk_$NAME`) immediately before its
own extraction — a redundant-looking step that turns out to be load
bearing: it implies the macro's saved `.mag`/`.gds` is not actually
fully flat, and the self-check papers over that by flattening a
second time right before verifying. The full-chip flow has no
equivalent step.

One more concrete, consistent data point: `macro/sky130/gen_abstracts.py`'s
`relabel_pins_to_pin_datatype()` — which moves the macro's pin labels
onto the GDS PIN datatype so the chip flow's Magic can recognize them
as ports — is explicitly documented as "Top-cell only — subcell-internal
labels must not become ports." That caveat only makes sense if the
GDS it operates on still has real subcells at the point this runs,
which corroborates the above directly from the generator's own code
comments, not just from reading the LVS report.

**Net effect:** this is a genuine macro-generation/extraction-hierarchy
issue, specific to how `macro_array_pc_64x32_test0`'s GDS carries
(non-flat) internal structure into the full-chip flow, not a config
flag, PDK version, or netgen setup file. It affects the **unmodified,
previously-signed-off `cirom_chip_digital`** exactly as much as the
new `cirom_chip_digital_affine` design — both instantiate the same
hard macro the same way — so it is not something introduced by any of
this binary-support work.

**Untested next step, concretely scoped:** LibreLane's `Netgen.LVS`
step exposes `LVS_FLATTEN_CELLS` (`librelane/steps/netgen.py:153`,
"A list of cell names to be flattened while running LVS") and
`LVS_IGNORE_CELLS`. Passing
`LVS_FLATTEN_CELLS: ["macro_array_pc_64x32_test0"]` (or the specific
inner cells, `P2_v4_array_64x32_test0` / `P2_bitcell_v4`) in
`config.json` is a one-line, directly-supported config change — no
script patching — and is the next thing to try. Caveat going in,
stated honestly: this flattens netgen's in-memory netlist graph
*after* Magic has already extracted it; if Magic's extraction itself
already dropped the drain connection (rather than just naming it
confusingly), flattening the already-broken netlist will not recover
a connection that was never captured. Whether `LVS_FLATTEN_CELLS` is
a real fix or another dead end can only be answered by running it.

**Tested, and it's a dead end, exactly per the caveat above.** Ran
`Netgen.LVS` only (`--only Netgen.LVS`, resumed from the cached
`68-magic-spiceextraction` state via `-i`, ~30s) against the
unmodified control chip with
`-c 'LVS_FLATTEN_CELLS=["macro_array_pc_64x32_test0"]'`. Result:
byte-identical `macro_array_pc_64x32_test0` subcircuit report —
same "Merged 1984 parallel devices", same `(2048->64)` vs. `(2048)`
mismatch, same `BLP_0`/`WL_0` fanout counts — and the same final
verdict, "Top level cell failed pin matching." Confirms the theory:
`LVS_FLATTEN_CELLS` flattens netgen's already-extracted netlist
graph; it has no way to recover a connection Magic's own extraction
never captured in the first place. All three original hypotheses
(PDK version, LibreLane version, `NETGEN_SETUP`) plus this fourth one
(`LVS_FLATTEN_CELLS`) are now disproven with direct evidence.

**Where this actually stands:** the root cause is real and
understood precisely (the macro's leaf bitcell subckt is missing its
drain port — `P2_bitcell_v4`'s 3 ports vs. the canonical
`bitcell_v4`'s 4 — somewhere between `gen_macro_array_pc.tcl`'s
`flatten` and the GDS the chip flow reads back in). But every fix
available from flow configuration has now been exhausted. What's left
is actual layout-pipeline work: either (a) make
`gen_macro_array_pc.tcl`'s GDS export genuinely fully flat (figure
out why hierarchy survives the `flatten` call into the GDS at all),
or (b) find and use whatever knob (if any) makes LibreLane's own
`Magic.SpiceExtraction` step flatten this macro's layout *before*
extracting, mirroring what `tools/gen_macro.sh`'s passing self-check
already does. Both are real engineering, not config changes, and
both apply equally to the already-signed-off `cirom_chip_digital`
— this was never a defect in any of this binary-support work.

## Resolution: LVS 0, confirmed on both chips -- two real fixes, one of them load-bearing

The `(a)` path above was pursued to completion. Two distinct, real
bugs were found and fixed; only the second one turned out to be what
was actually blocking LVS.

**Fix 1 (real bug, not the blocker): `gen_macro_array_pc.tcl`'s
`flatten` call could never succeed.** It called
`flatten -dolabels $MACRO_NAME` -- `-dolabels` is not a real Magic
flatten option (the valid set is `-nolabels`, `-nosubcircuits`,
`-noports`, `-novendor`, `-dotoplabels`, `-doproperty`, `-dobox`,
`-doinplace`; confirmed against Magic's own command reference inside
the LibreLane image). Worse, `$MACRO_NAME` was already the name of
the cell being edited at that point in the script (loaded and
`cellname rename`d to it earlier), and Magic's `flatten` requires the
destination cell to not already exist. Fixed by flattening into a
distinct temporary name and renaming the *old* hierarchical cell out
of the way before claiming `$MACRO_NAME` for the new flat one
(`cell/sky130/macro/gen_macro_array_pc.tcl`). Verified with KLayout:
the macro's GDS went from 2 child-cell references (real, unflattened
hierarchy: `v4_array_64x32_test0`, `precharge_row32`, and
`bitcell_v4` nested further inside) to 0 -- genuinely flat, one cell,
as intended. This is a real, worth-keeping fix, but re-running the
macro's own LVS self-check against this fix *alone* produced the
exact same mismatch as before. Netgen auto-flattens non-matching
subcircuits during its own comparison regardless of whether the
source GDS was already flat, so this bug was never what LVS was
actually tripping on.

**Fix 2 (the actual blocker): the local `test0` build was stale --
mask programming had never been run against the real weight
matrix.** Direct inspection of `cell/sky130/macro/build/v4_array_64x32_test0.mag`
against `..._wlbl.mag` (its own pre-mask-programming input) showed
them *byte-identical except for a 1-second-different timestamp* --
zero `via1`/`via2`/`via3` instances anywhere in the file. Every one
of the 2048 bitcells was genuinely, electrically floating: confirmed
in the extracted netlist, where all 2048 drain terminals landed on
isolated per-instance local nets (`v4_array_..._0/bitcell_v4_N.D`),
none on `BLP_c`/`BLN_c`. Separately, the LVS reference schematic
(`macro/sky130/abstracts/macro_array_pc_64x32_test0.lvs.spice`) had
been generated from a dense fallback pattern (alternating `BLP`/`BLN`
by column parity, zero `nc_<r>_<c>` floating-net entries) instead of
the real `weights/test0.wmat` matrix (500 `+1`, 513 `-1`, **1035
`0`** -- just over half the array is zero-weight). Both sides of the
comparison were wrong, in different, incompatible ways; that
mismatch is what every Magic/LibreLane/netgen hypothesis chased above
was actually a symptom of.

Fixed by rebuilding both sides consistently from the real weights
file:
```
ANKHDJET_WEIGHTS_FILE=weights/test0.wmat ANKHDJET_WEIGHTS=test0 \
    magic ... < gen_mask_programming.tcl     # +1=500 -1=513 0=1035, DRC=0
magic ... < gen_macro_array_pc.tcl           # macro assembly (Fix 1 applied)
python3 macro/sky130/gen_abstracts.py 64 32 test0 --weights-file weights/test0.wmat
```
The macro's own standalone LVS self-check then passed cleanly:
**"Circuits match uniquely."** (1141 devices, 195 nets, exact match).

**Confirmed end to end on full, from-scratch LibreLane runs (no
resumed/cached state) on both chips:**

| Design | Magic DRC | Netgen LVS |
|---|---|---|
| `cirom_chip_digital` (control, unmodified) | `Check for Magic DRC errors clear.` | `Circuits match uniquely.` / `design__lvs_error__count: 0` |
| `cirom_chip_digital_affine` (FBI-LLM affine-binary target) | `Check for Magic DRC errors clear.` | `Circuits match uniquely.` / `Check for LVS errors clear.` |

Both runs end with the same pre-existing "370 power grid violations"
deferred warning that is *also* present in the original signed-off
`sanity_check` run's log -- confirmed identical, not a regression,
and the tool's own message says to ignore it when LVS passes, which
it now does on both chips.

**One new, separate, not-yet-investigated item on the affine chip
only:** its full run's deferred-error summary also reported setup
timing violations in the slow corners (`max_ss_100C_1v60`,
`min_ss_100C_1v60`, `nom_ss_100C_1v60`). This is unrelated to LVS/DRC
signoff and was already flagged earlier in this doc as a lower-priority,
unstarted item for the new design. Not pursued yet.

**Net result:** the FBI-LLM affine-binary chip now has a complete,
LVS-clean, DRC-clean placed-and-routed layout through full signoff,
on the same real hard macro and the same flow the project's own
prior silicon used. Nothing about the binary-support RTL/IR/backend
work (`cirom_dig_ctrl_affine.sv`, the affine IR fields, the decoders)
needed to change at all -- the entire blocker was upstream, in stale
local build artifacts for a macro shared by both chips.
