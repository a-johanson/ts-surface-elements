# Spatial Grid for Relaxation — Design Spec

Source of truth for the spatial-grid acceleration of the relax pass. Multiple
agent sessions may implement this feature from this document; the step list is
sequential and each step depends on its predecessors. Update this spec when a
resolved design decision changes or a step is completed.

## Problem

The relax compute shader (`src/stipple/shaders/relax-shader.ts:71`) currently
runs an O(n²) inner loop: every point iterates all other points to accumulate
a curvature-aware repulsion force gated by an inflated-distance cutoff and a
midpoint SDF line-of-sight check. At `POINT_COUNT = 4 * 1024` this is
affordable, but it blocks scaling the stipple density to tens of thousands of
points.

## Goal

Replace the O(n²) neighbor scan with a **uniform spatial grid** rebuilt each
frame, with cell size equal to the relaxation radius. Each point then iterates
only its own cell plus the 26 neighboring cells (27-cell 3×3×3 neighborhood) —
O(1) neighbor lookup amortized.

The grid is built with the standard GPU approach: per-point cell index as key
and point index as value, **radix-sorted** (1 bit per pass, 32 passes) so all
points in the same cell are contiguous, then a **cell-start/count table**
gives O(1) range lookup per cell.

## Non-goals

- Multi-block / global-scan radix sort (deferred until single-workgroup sort is
  demonstrably too slow — see *Switch points*).
- Wider-radix passes (4-bit) — start with 1-bit for clarity; optimize later.
- CPU readback or grid visualization tooling (only ad-hoc verification).
- Per-frame GPU bounding-box compute (see decision 5 — dropped in favor of a
  fixed CPU-known SCENE_BBOX).

## Assumptions

- `POINT_COUNT = 4096` today; target scaling to tens of thousands (not
  millions).
- Default relaxation `radius = 0.3` (`DEFAULT_RELAX_PARAMS`).
- Scene bounding box is small enough that grid dimensions per axis fit in ~10
  bits (e.g. `bbox extent / radius ≈ 20` per axis), so a packed 32-bit linear
  cell index comfortably holds three ~10-bit cell coordinates.
- The WebGPU `"subgroups"` feature is available on the target machine (Step 1
  verifies this and hard-fails otherwise).
- A single explicit `SCENE_BBOX` constant, chosen once to safely contain the
  animated SDF surface across all animation phases, is shared by the seed
  pipeline (rejection sampling) and the spatial grid (origin + dims). It
  replaces the former `SEED_PARAMS.bboxMin/bboxMax` in `main.ts` (completed
  in Step 3).

## Resolved design decisions

### 1. Grid rebuild frequency: once per frame

The grid is rebuilt **once per frame**, after `reproject` and before the relax
substep loop. It is reused across all relax substeps in that frame (up to
`MAX_SUBSTEPS = 4`).

**Rationale.** Per-substep point drift is `dt · ‖F‖ ≈ 0.01 · 1 = 0.01` units,
vs. cell size `radius = 0.3` — ~3% of a cell per substep, ~12% across a full
4-substep frame. A neighbor pair can only be *missed* if both points move
toward each other by a combined amount exceeding one cell width, which does
not occur within a single frame. Moreover, the pairs most at risk of straddling
a cell boundary sit near `d_infl ≈ radius`, where the linear-decay envelope
`(1 − d_infl/radius) → 0`, so the missed force contribution is near-zero.

**Switch condition.** If visual clustering artifacts appear near fast-moving
`smin` blend folds (where per-substep drift is largest), move rebuild to
per-substep by calling `spatialGrid.dispatch` inside the substep loop, right
before each `relax.dispatch`. The cost is ~`MAX_SUBSTEPS`× the sort work per
frame.

### 2. Sort scale: single-workgroup

The radix sort runs as **a single workgroup** (≤256 invocations, each thread
owning `N/256` elements). The subgroup-exclusive scan (Step 2) computes the
per-subgroup prefix; one `workgroupBarrier` merges subgroup partials in
`var<workgroup>` for the workgroup-wide prefix. Sort data (keys, values) lives
in **storage buffers**, not workgroup memory — so the sort is bounded by
per-thread sequential work, not the 16 KB `maxComputeWorkgroupStorageSize`
default. Workgroup memory is used only for scan partials (~1 KB).

**Scaling ceiling.** Single-workgroup remains efficient up to roughly
`256 × 256 = 65 536` points; beyond that per-thread sequential work per pass
dominates and latency grows linearly with `N`.

**Switch condition.** Move to a **multi-workgroup** sort (split the array across
workgroups with a global prefix-scan kernel between every radix pass, plus
atomic reductions) when `POINT_COUNT` exceeds ~32k–65k or when per-pass latency
becomes a frame-budget bottleneck. This is a significant rewrite; do not
attempt before the single-workgroup version is measured.

### 3. Module structure: one `spatial-grid-pipeline.ts`

A new `src/stipple/spatial-grid-pipeline.ts` owns the entire grid build:
cell-index computation, radix sort, and cell-range table. Its shaders live
under `src/stipple/shaders/spatial-grid/`. One
`dispatch(encoder, readFromA)` entry point records the full build into the
command encoder. The `radius` (cell size) is fixed at construction — a
per-dispatch radius would make `gridDims` stale relative to the pre-sized
`cellStart`/`cellCount` buffers. High cohesion, one orchestration call site
in `main.ts`.

### 4. Step 1 scope: feature gate only

Step 1 only requires the `"subgroups"` feature and hard-fails if absent. No
shader work. This lets the user confirm machine support (which is uncertain)
before building anything on top. A trivial subgroup smoke-test shader is
explicitly deferred — the first real subgroup usage is Step 2's scan helper,
which doubles as the compile/exec validation.

### 5. Grid bbox: fixed SCENE_BBOX constant (no per-frame GPU bbox)

The grid uses a single fixed CPU-known `SCENE_BBOX` constant directly as its
origin and dimensions, shared with the seed pipeline (replaces the former
`SEED_PARAMS.bboxMin/bboxMax`). **No per-frame GPU bounding-box compute.**

**Rationale.** The original request asked to "infer the bbox from the points"
via a subgroup min/max reduction. On analysis, a per-frame GPU bbox creates a
CPU/GPU synchronization problem — the CPU needs the bbox extent to size the
`cellStart`/`cellCount` buffers and fill `gridParams`, but reading back a
GPU-computed bbox every frame would stall the frame loop
(`mapAsync` + `device.queue.onSubmittedWorkDone()`), destroying CPU/GPU
parallelism. A conservative CPU-known constant sidesteps this entirely: no
readback, no per-frame-dims complexity, no actual-vs-conservative mismatch.

**Cost of the fixed grid.** Empty cells (cells the animated surface never
reaches) are ~free:
- `cellStart`/`cellCount` buffer memory: `numCells × 8` bytes. For SCENE_BBOX
  `6×4×4` at `radius = 0.3`: `gridDims = 20×14×14 ≈ 3920` cells → ~31 KB.
  Fixed, trivial.
- Per-frame clear dispatch: reset `cellStart → UINT_MAX`, `cellCount → 0`
  (~3920 / 256 ≈ 16 workgroups). Sub-microsecond.
- Relax empty-cell reads: for each of 27 neighbor cells, relax reads
  `cellStart[c]` (4 bytes, cached — the whole ~16 KB buffer fits L1). If
  `UINT_MAX`, branch-skip immediately. ~55k wasted *cached* reads hidden by
  GPU parallelism. Sub-microsecond.

**Tradeoff vs. per-frame GPU bbox.** Dropping the GPU bbox loses tight per-frame
binning (the grid is sized to the worst-case extent every frame, so more cells
are permanently empty), but as shown above that costs ~zero. It also skips the
subgroup min/max reduction that the original request mentioned as a learning
exercise — but the harder, more substantive subgroup learning (the exclusive
scan in the radix sort, Step 2) is retained.

**Deviation from the original request.** The original request asked for the
bbox to be "inferred from the points" using a subgroup min/max search. This
spec deviates: the bbox is a fixed constant. The deviation is documented here
so a future agent understands the reasoning. If the scene ever becomes hard to
bound conservatively, add the GPU subgroup min/max bbox compute back (see
*Switch points*).

### 6. Safety net: shader-side clamp on cell coords

The cell-index shader (Step 3) clamps each axis of the computed cell coord to
`[0, gridDim-1]` before packing to the linear key. This prevents out-of-bounds
storage writes if `SCENE_BBOX` is too small.

**Failure mode if SCENE_BBOX is too small.** The clamp itself prevents
crashes/corruption/validation errors; its only job is OOB-write prevention.
The sole consequence of a too-small SCENE_BBOX is **performance
degradation**: boundary cells become over-full because out-of-range points
collapse into them, so the relax inner loop iterates more points per
boundary cell (most skipped by the distance cutoff, `d_infl > r`). No
visible artifact, just slower.

The clamp is *correctness-preserving*. A clamped point Q (beyond the
boundary, in cell `gridDim-1`) can only be within `radius` of an in-range
point P if P is within `radius` of the boundary — which places P in cell
`gridDim-1` or `gridDim-2`. Both cells' 27-neighborhoods include cell
`gridDim-1` (where Q lives), so the pair is always tested. If P is further
in (cell `gridDim-3` or beyond), Q is at least `radius + cellSize` away —
beyond the cutoff and correctly skipped. So no neighbor pair within
`radius` is ever missed by the clamp; the only cost is wasted iterations
on over-full boundary cells.

## WebGPU subgroups feature requirement

- **Feature name:** `"subgroups"` (see W3C WebGPU §25.18). There is also an
  optional `"subgroup-size-control"` companion; not required by this spec.
- **WGSL directive:** `enable subgroups;` at the top of every shader that uses
  subgroup builtins/ops.
- **Subgroup builtins** (verified against the WGSL CRD, 21 Sep 2026, in Step 2):
  `@builtin(subgroup_invocation_id)`, `@builtin(subgroup_size)`,
  `@builtin(subgroup_id)`, `@builtin(num_subgroups)`. The last two are needed
  by the workgroup-scan merge step (identifies which subgroup a lane belongs
  to and how many subgroups exist).
- **Subgroup ops** (verified in Step 2): `subgroupExclusiveAdd`,
  `subgroupInclusiveAdd`, `subgroupAdd`, `subgroupBroadcast`,
  `subgroupBroadcastFirst`, `subgroupBallot`, `subgroupMin`, `subgroupMax`,
  etc. All names confirmed against §17.12 of the WGSL CRD.

If the feature is unavailable, `createGpuContext` logs a clear `console.error`
and throws — the app must not attempt a fallback path.

## Architecture & data flow

The grid build is inserted into the existing per-frame pipeline between
`reproject` and `relax`:

```
reproject (in-place on readFromA buffer, refreshes shared normals)
   │
   ▼
spatial grid build  ◄── reads points[readFromA], writes grid buffers
   │   1. cell-index computation (SCENE_BBOX origin+dims, clamped) → keys[], values[]
   │   2. clear cellStart/cellCount
   │   3. radix sort (32 binary splits, ping-pong A↔B)
   │   4. cell-start/count table
   ▼
relax (substepped, ping-pong)  ◄── reads points[readFromA] + grid buffers
   │
   ▼
shading → render
```

The grid is built from the **same buffer relax will read** (`readFromA` after
reproject), so cell assignments match the positions relax iterates. The grid
buffers are read-only during relax and rebuilt wholesale next frame. No
ping-pong of grid buffers across frames — they are overwritten in place each
build.

### Placement in the frame loop (`src/main.ts`)

After `reproject.dispatch(encoder, time, readFromA)` and before the substep
loop:

```ts
spatialGrid.dispatch(encoder, readFromA);
// then the existing substep loop, unchanged
for (let s = 0; s < substeps; s++) { relax.dispatch(...); readFromA = !readFromA; }
```

The grid is reused by all substeps in the frame (see decision 1).

## Buffer inventory

All buffers are created by `SpatialGridPipeline` upfront at construction
(Step 3). Sizes assume `POINT_COUNT = N`, `numCells = gridDimX * gridDimY *
gridDimZ` derived from `ceil(SCENE_BBOX extent / radius)` per axis.

| Buffer             | Size            | Usage                              | Owner / writer          |
|--------------------|-----------------|------------------------------------|-------------------------|
| `gridParamsBuffer` | uniform, ~48 B  | `UNIFORM \| COPY_DST`             | spatial-grid (CPU write, once at startup) |
| `keysA`, `keysB`   | `N × 4` B       | `STORAGE \| COPY_DST`             | spatial-grid (sort ping-pong) |
| `valuesA`, `valuesB` | `N × 4` B     | `STORAGE \| COPY_DST`             | spatial-grid (sort ping-pong) |
| `cellStart`        | `numCells × 4` B | `STORAGE \| COPY_DST`            | spatial-grid (cell-ranges, cleared per frame) |
| `cellCount`        | `numCells × 4` B | `STORAGE \| COPY_DST`            | spatial-grid (cell-ranges, cleared per frame) |

`gridParamsBuffer` holds `bboxMin` (vec3f), `bboxMax` (vec3f), `gridDims`
(vec3u), `cellSize` (f32), `pointCount` (u32) — all constants written once at
startup. Empty cells get `cellStart = UINT_MAX` and `cellCount = 0`; relax
skips them.

### Relax bind-group additions

`RelaxPipeline`'s bind group gains (in addition to the existing params, points
read/write, normals): `gridParamsBuffer`, sorted `keys`, sorted `values`,
`cellStart`, `cellCount`. Because the sort result lands in buffer A (32 passes,
even count), the relax bind group always binds the A pair as the sorted output —
regardless of the point-buffer ping-pong direction, since the grid is rebuilt
each frame from the current `readFromA` and its sorted output is always in A.
The `radius` already exists in `RelaxParams` and doubles as the cell size — no
new param needed; relax reads `bboxMin`/`bboxMax`/`gridDims` from
`gridParamsBuffer`.

## Implementation steps

Sequential. Stop after each step for review. Later steps will not compile/run
until predecessors are complete.

### Step 1 — Require the `subgroups` feature

Modify `src/webgpu.ts` `createGpuContext`:
- Call `adapter.requestDevice({ requiredFeatures: ["subgroups"] })`.
- After device creation, assert `device.features.has("subgroups")`.
- On failure: `console.error` a clear message naming the missing feature, then
  `throw new Error(...)`.
- No change to the `GpuContext` interface (features are exposed on `device`).

**Verify:** `npm run dev`, open the page. If it boots without the error, the
machine supports subgroups. If it throws, stop and reconsider the approach
(subgroups may be behind a flag or unsupported on the GPU/driver).

### Step 2 — WGSL subgroup scan primitive

New `src/stipple/shaders/subgroup-common.ts` exporting a WGSL string with:
- `enable subgroups;`
- A **subgroup-level exclusive prefix-sum** helper (`subgroupExclusiveScanU32`)
  built directly on `subgroupExclusiveAdd` — no `subgroupBroadcast` needed,
  since `subgroupExclusiveAdd` returns the per-lane exclusive prefix directly.
- A **workgroup-level exclusive scan** helper (`workgroupExclusiveScanU32`)
  that runs the subgroup scan per subgroup, stores subgroup partials in
  `var<workgroup>` (last lane of each subgroup writes), performs a
  `workgroupBarrier`, has thread 0 sequentially scan the partials array
  (≤64 entries) in place for the cross-subgroup exclusive prefix, barriers
  again, and combines with each lane's subgroup-exclusive result. Two barriers
  total (not one as originally stated) — the second ensures all lanes see the
  scanned partials.

This is the reusable foundation for the radix split (Step 4). In this step,
also verify the exact subgroup builtin/function names against the WGSL spec
and update this spec if any name differs from the list above.

**Completed:** names verified against WGSL CRD (21 Sep 2026); all match.
`subgroup_id` and `num_subgroups` were added to the builtins list (originally
omitted, needed by the workgroup-scan merge). `subgroupBroadcast` was not
needed. `WORKGROUP_SIZE` (256) and `MAX_SUBGROUPS` (64) constants are defined
in this module.

### Step 3 — Cell-index computation

`src/stipple/shaders/spatial-grid/cell-index-shader.ts` + the first piece of
`src/stipple/spatial-grid-pipeline.ts`:
- Read `points[readFromA]` and `gridParamsBuffer` (`bboxMin`, `gridDims`,
  `cellSize = radius`).
- Per point `i`: 3D cell coord
  `c = clamp(floor((p - bboxMin) / cellSize), vec3u(0), gridDims - 1u)`, packed
  to linear `u32` key `= c.x + gridDimX * (c.y + gridDimY * c.z)`.
- Write `keysA[i] = key`, `valuesA[i] = i`.
- **Clamp** (decision 6): each axis clamped to `[0, gridDim-1]` to prevent OOB
  if `SCENE_BBOX` is too small.
- Extract `SCENE_BBOX` as a shared named constant in `main.ts` (replaces
  `SEED_PARAMS.bboxMin/bboxMax`); pass it into both `SeedPipeline` and
  `SpatialGridPipeline`. Compute `gridDims` CPU-side from
  `ceil(SCENE_BBOX extent / radius)`.
- `SpatialGridPipeline` constructor takes `(device, points, sceneBBox, radius)`
  and creates all grid buffers upfront (7 buffers per the inventory). The
  `radius` is fixed at construction; `dispatch(encoder, readFromA)` takes no
  radius param (a per-dispatch radius would make `gridDims` stale relative to
  pre-sized buffers).
- `SeedPipeline` constructor changes to `(device, sceneBBox, band)`. The
  `SeedParams` TS interface is dropped; the WGSL `SeedParams` uniform keeps
  bbox (the shader needs it for rejection sampling). `band` is set to
  `DEFAULT_RELAX_PARAMS.radius` in `main.ts` so seed density matches the relax
  interaction scale. `dispatch` drops the `params` argument.

### Step 4 — Stable binary split (one bit)

`src/stipple/shaders/spatial-grid/radix-split-shader.ts` — the core primitive:
- Bit position `b` passed via uniform.
- Each thread owns `E = N/256` elements. Per element compute
  `pred = ((key >> b) & 1u) == 0u ? 1u : 0u`.
- Workgroup-wide exclusive scan of `pred` (Step 2 helper) → each element's
  destination among the 0-bits.
- `totalZeros` = scan total (last element's inclusive result).
- For a 0-bit element at global index `g`: dest = `exclusiveScan(pred)[g]`.
- For a 1-bit element at global index `g`: dest = `g - exclusiveScan(pred)[g]
  + totalZeros`.
- Scatter `(key, value)` from the read pair into the write pair at `dest`.
- Ping-pong bind groups A→B and B→A. Stable: 0-bits preserve order, then
  1-bits preserve order.

### Step 5 — Radix sort driver (32 passes)

`SpatialGridPipeline.dispatch` records one compute pass with 32
`dispatchWorkgroups` calls, bit `0..31`, alternating the A/B bind group each
pass. 32 is even, so the final sorted `(keys, values)` lands back in the A
pair. Explicit per-bit dispatches (rather than an in-shader loop) so each pass
is inspectable while learning; a single dispatch with an internal loop is a
later optimization (see *Switch points*).

### Step 6 — Cell-start/count table

`src/stipple/shaders/spatial-grid/cell-ranges-shader.ts` over the sorted
`keysA`:
- **Clear pass** (per frame): reset `cellStart → UINT_MAX`, `cellCount → 0`
  for all `numCells`. Implement as a small compute dispatch (one invocation
  per cell) or via `writeBuffer` if simpler. Must run before the ranges pass.
- **Ranges pass**: one invocation per point `i`. If `i == 0` or
  `keysA[i] != keysA[i-1]`, this is the first point of cell `keysA[i]` — store
  `cellStart[keysA[i]] = i`. For every `i`:
  `atomicAdd(cellCount[keysA[i]], 1)`.

Empty cells keep `cellStart = UINT_MAX`, `cellCount = 0`; relax skips them.

### Step 7 — Rewrite relax shader to use the grid

Edit `src/stipple/shaders/relax-shader.ts:71` — replace the O(n²) `for j`
loop:
- Recompute point `i`'s cell (same formula as Step 3, with the same clamp)
  from its position + `gridParamsBuffer`.
- Loop the 27 neighbor cells (3×3×3 offset). For each neighbor cell, read
  `cellStart`/`cellCount`; if `cellStart == UINT_MAX` skip; else iterate
  `[start, start + count)` in the sorted arrays, fetch
  `j = valuesA[sortedIdx]`, read `points_in[j]` and `normals_in[j]`.
- Run the **unchanged** pairwise force logic: curvature-inflated distance
  cutoff, midpoint SDF line-of-sight, linear-decay envelope, tangent-plane
  projection, Euler step, Newton re-projection.

Extend the relax bind group with: `gridParamsBuffer`, `keysA`, `valuesA`,
`cellStart`, `cellCount`. Update `RelaxPipeline` constructor and both
ping-pong bind groups.

### Step 8 — Wire into frame loop + update docs

- `src/main.ts`: construct
  `SpatialGridPipeline(gpu.device, points, SCENE_BBOX, DEFAULT_RELAX_PARAMS.radius)`
  (already done in Step 3). After `reproject.dispatch` and before the
  substep loop, call `spatialGrid.dispatch(encoder, readFromA)`. Pass the
  grid buffers into `RelaxPipeline` (constructor or a setter).
- Update `AGENTS.md` structural overview: add a `spatial-grid-pipeline.ts`
  line and update the data-flow description (reproject → **grid build** →
  relax → shading → render). Add `shaders/spatial-grid/` and
  `shaders/subgroup-common.ts` lines.
- Run `npm run lint`.

## Verification

| Step | How to confirm |
|------|----------------|
| 1    | Page boots without the subgroups error. |
| 5    | (Optional) CPU readback of `keysA` is non-decreasing. |
| 7    | Visual: points still relax to an even Poisson-disc distribution, no clustering or holes; relax dispatch GPU time drops vs. the O(n²) version. If SCENE_BBOX is too small, expect only performance degradation (over-full boundary cells); correctness is preserved by the clamp. |
| 8    | `npm run lint` clean; full frame loop runs without errors. |

## Switch points / future work

- **Per-substep rebuild** — if clustering appears near fast `smin` folds (see
  decision 1).
- **Multi-workgroup radix sort** — past ~32k–65k points or when sort latency
  dominates (see decision 2).
- **Single-dispatch in-shader sort loop** — collapse the 32 per-bit dispatches
  into one dispatch with an internal 32-iteration loop and `workgroupBarrier`
  between passes. Saves dispatch overhead only; do after correctness is
  confirmed.
- **Wider radix (4-bit, 8 passes)** — fewer passes once 1-bit correctness is
  proven; requires a 16-bucket histogram + scan instead of a binary split.
- **GPU-derived bbox** — if the scene becomes hard to bound conservatively with
  a constant, add the GPU subgroup min/max bbox compute back (the original
  request's Step 3). Requires either a one-time startup readback to size
  buffers, or conservative-max sizing + clamp (as now).

## Risks / open questions

- **Subgroup builtin/function names** — verified against the WGSL CRD (21 Sep
  2026) in Step 2. All names match §17.12 and §13.3.1.1.17–20; `subgroup_id`
  and `num_subgroups` were added to the builtins list (originally omitted).
- **Storage-buffer scatter performance** — scattering via storage buffers (not
  workgroup memory) is slower per access but unbounded by the 16 KB workgroup
  limit. Acceptable for v1; revisit if sort latency is high.
- **Cell-index packing width / cell-count cap** — three ~10-bit cell coords
  in a `u32` is fine for tens of thousands of cells. The `SpatialGridPipeline`
  constructor enforces a `MAX_CELLS = 2²⁰ = 1_048_576` cap and throws if
  exceeded, which guards both the u32 key packing width and the
  `cellStart`/`cellCount` memory (capped at ~8 MB). The target scale is tens
  of thousands of points; exceeding the cap means far more cells than points
  (e.g. >1M cells for <65k points means most cells are permanently empty),
  indicating a misconfigured radius or bounding box rather than a legitimate
  workload. If grid resolution ever legitimately grows beyond ~2¹⁰ per axis,
  widen the packing or use a 64-bit key and raise the cap.
- **SCENE_BBOX correctness** — if the constant is too small for the animated
  scene, out-of-range points collapse into boundary cells (clamped, decision 6).
  The clamp is correctness-preserving (no missed neighbor pairs within
  `radius`), so the only cost is performance degradation from over-full
  boundary cells. Not a crash or corruption; update the constant if the
  scene geometry changes.

## AGENTS.md update (part of Step 8)

Add to the structural overview:
- `src/stipple/shaders/subgroup-common.ts` — shared WGSL subgroup scan helpers
  (subgroup-level and workgroup-level exclusive prefix sum), reused by the
  radix split.
- `src/stipple/spatial-grid-pipeline.ts` — the per-frame spatial grid build
  (cell index → radix sort → cell ranges). Runs once per frame after
  reproject, before relax; reads the buffer relax will read; grid buffers are
  read-only during relax and rebuilt wholesale next frame. Uses a fixed
  CPU-known `SCENE_BBOX` (shared with the seed pipeline) as origin + dims.
- `src/stipple/shaders/spatial-grid/` — the cell-index, radix-split, and
  cell-ranges shaders.

Update the data-flow description in `src/main.ts`'s bullet to include the grid
build step between reproject and relax.
