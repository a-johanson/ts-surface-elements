# Surface Hatch Lines / Streamlines — Design Spec

Source of truth for the hatch-line / streamline feature that connects
distributed surface points into world-space line segments along a principal
direction. Multiple agent sessions may implement this feature from this
document; the step list is sequential and each step depends on its
predecessors. Update this spec when a resolved design decision changes or a
step is completed.

## Problem

After the stipple pipeline distributes points on the animated SDF surface
(seed → reproject → relax → shading), the points are well-spaced but
disconnected — there is no visual structure linking neighboring points into
lines. We want to connect each point to its single best-aligned neighbor
along a principal direction (orthogonal to the light direction projected
onto the tangent plane), producing hatch lines or streamlines that
organically follow the surface curvature. The entire computation and
rendering should happen on the GPU with no CPU readback.

## Goal

Add two new pipeline stages after the existing shading pass:

1. A **link compute pass** that, for each point, uses the spatial grid to
   find the best-aligned neighbor within the relax radius, subdivides the
   path to that neighbor into surface-projected segments, and writes the
   curve points to a storage buffer.

2. A **link render pass** that draws the curve segments as camera-facing
   quads with soft anti-aliased edges and per-segment occlusion culling,
   composited under the existing stipple points.

The feature is always on (no feature gate); lines are drawn before the
point render so points sit on top.

## Non-goals (deferred to future work)

- **Mitering** — overlapping segments at joints to eliminate hairline
  gaps. Step 5 addresses this; v1 tolerates small gaps.
- **Luminance modulation** — lines do not darken in shadow in v1
  (luminance is computed by the shading pass but not consumed by the link
  renderer). Can be added once the basic rendering looks right.
- **Dedup of mutual links** — if point `i` links to `j` and `j` links to
  `i`, both segments render. Dedup is not needed for correctness and may
  even look denser/nicer; revisit if it causes visual artifacts.
- **Depth testing** — no depth attachment; occlusion is resolved via
  `rayClearance` toward the eye (Step 4), not via the rasterizer's depth
  buffer.
- **Stale-grid reuse** — the link pass rebuilds the spatial grid from
  post-relax positions. Reusing the pre-relax grid (built before the relax
  substep loop) is a worthwhile performance optimization to try later;
  points move <0.05/frame vs 0.3 cell size, so most neighbors are still
  found. See *Switch points*.

## Assumptions

- `POINT_COUNT = 6 * 1024` (current `main.ts` constant).
- `RELAX_RADIUS = 0.3` (current cell size / relax interaction radius).
- The spatial grid (`SpatialGridPipeline`) and its `GridBuffers` interface
  are available and already wired into the frame loop.
- The shared `normalsBuffer` is fresh (written by the shading pass every
  frame) and matches the post-relax point buffer.
- `lightDir(time)` is available in `SDF_COMMON` and returns a normalized
  light direction.
- `tangentFrame(n)` is available in `SDF_COMMON` and returns an
  orthonormal tangent basis.
- `projectToSurface(p, time, iters, alpha)` is available in `SDF_COMMON`.
- `rayClearance(ro, rd, max_dist, time)` is available in `SDF_COMMON` and
  returns the minimum signed clearance between a ray and the SDF surface.
- `viewProjection(eye, target, up, fov, aspect)` is available in
  `src/stipple/camera.ts` and returns a 16-element column-major
  `mat4x4f` array.
- The `"subgroups"` feature is available (already required by the spatial
  grid radix sort).

## Resolved design decisions

### 1. Post-relax grid rebuild

The link pass needs a spatial grid built from **post-relax positions**.
The existing grid, built before the relax substep loop, reflects
pre-relax positions and is stale by up to ~12% of a cell per frame. The
link pass calls `spatialGrid.dispatch(encoder, readFromA)` a second time
(after the shading pass, where `readFromA` now points at the buffer relax
most recently wrote). This reuses the existing `SpatialGridPipeline`
infrastructure — no new grid code, just one extra dispatch call in
`main.ts`. The rebuild and the link dispatch run only on frames where
time is playing — while paused, no link input changes, so both are
skipped (see *Placement in the frame loop*).

**Cost.** One extra cell-index + radix-sort + cell-ranges build per frame
(~14 small dispatches for 6K points). Negligible.

**Future optimization.** Reusing the stale pre-relax grid is a worthwhile
experiment — per-frame point drift (~0.05 units) is well under the cell
size (0.3), so most neighbor pairs are still found. If visual quality is
acceptable with the stale grid, the second rebuild can be dropped. See
*Switch points*. For now, the spec mandates the correct (rebuild) path.

### 2. Single compute pass: scan + subdivide + project

The link compute shader does **both** the neighbor scan and the curve
subdivision in a single pass. After finding the best neighbor `j` for
point `i`, the shader immediately subdivides `lerp(p_i, p_j, t)` for `t ∈
[0, 1]` in `N_SEGMENTS` steps and Newton-projects each sample onto the
SDF surface, writing `N_SEGMENTS + 1` `vec4f` positions per point (`.xyz`
= position, `.w` = validity flag). This eliminates the need for an
intermediate `u32` index buffer and a second compute pass.

**Step 1 exception.** Step 1 (the initial validation step) writes a
simple `u32` index per point and renders a flat line-list — no
subdivision, no projection. This is intentionally minimal to verify the
neighbor-finding logic produces visually sensible connections. Step 2
replaces the `u32` index buffer with the subdivided curve buffer.

**Cost.** `N_SEGMENTS` Newton projections per point (~4 `map()` + gradient
evals each). For `N_SEGMENTS = 4` and 6K points this is ~24K SDF
evaluations — negligible compared to the shading pass which already does
~4 `rayClearance` traces per point.

### 3. Principal direction: orthogonal to the light's tangent projection

For each point `i` with normal `n_i`, the principal direction is:

```wgsl
let L = lightDir(time);
let L_tan = L - dot(L, n_i) * n_i;       // project light onto tangent plane
let principal = normalize(cross(n_i, L_tan));  // orthogonal direction
```

This gives a direction in the tangent plane that is perpendicular to the
shadow direction — hatch lines run "across the grain" of the lighting.
When `L_tan ≈ 0` (light nearly parallel to the normal), no principal
direction exists and the point gets no link (`UINT_MAX` / `.w = 0.0`).

### 4. Scoring: alignment discounted by distance

Only neighbors in the **forward hemisphere** (`alignment > 0`, i.e. the
neighbor is roughly in the principal direction, not behind) are
considered. The score is:

```wgsl
let dir = (p_j - p_i) / dist;
let alignment = dot(dir, principal);       // ∈ (0, 1]
let score = alignment - DISTANCE_WEIGHT * dist;
```

`DISTANCE_WEIGHT` is a compile-time WGSL constant (see decision 8). With
`DISTANCE_WEIGHT * RELAX_RADIUS ≈ 0.5`, a near
point at 0.4 alignment beats a far point at 0.9 alignment — matching the
desired tradeoff where proximity matters as much as alignment. The
highest-scoring neighbor wins; ties are broken by first-found (deterministic
given the grid iteration order).

**Thin-wall filter.** Like the relax shader, the link shader skips pairs
whose midpoint is deep inside the surface with opposing normals
(`map(midpoint) < -0.05 * radius && dot(n_i, n_j) < -0.2`), since these
likely sit on opposite sides of a thin wall.

### 5. Output: per-point curve buffer (not ping-ponged)

The link compute writes a single `linkCurves` buffer of size
`point_count * (N_SEGMENTS + 1) * 16` bytes (`vec4f` per curve point). It
is **not ping-ponged** — it is written once per frame by the link compute
and read by the link render. Same pattern as `shadingBuffer`. The link
compute reads from whichever point buffer relax most recently wrote
(determined by `readFromA` at call time), so it needs two read bind groups
(A and B), like the shading pipeline.

In Step 1, the output is a simpler `linksBuffer` of `u32 × point_count`
(best neighbor index or `UINT_MAX`). Step 2 replaces it with the curve
buffer.

### 6. Rendering: camera-facing quads (Step 3+)

From Step 3 onward, each sub-segment is rendered as a **camera-facing
quad** (billboarded around the segment axis). The vertex shader computes:

```wgsl
let seg_dir = normalize(p1 - p0);
let view_dir = normalize(eye - center);
let side = normalize(cross(seg_dir, view_dir));
let offset = ±side * LINK_HALF_WIDTH;
```

where `center = 0.5 * (p0 + p1)`. The quad corners are offset in world
space by `±side * LINK_HALF_WIDTH`, so lines have consistent world-space
width that foreshortens with viewing angle (same philosophy as the point
renderer's `POINT_RADIUS_WORLD`). The fragment shader applies a soft
anti-aliased edge using `fwidth` on the side coordinate (same AA pattern
as the point shader).

**Step 1 and Step 2 use a simple `line-list`** with native `line-list`
topology — no quads, no billboard math. This is the minimal path to
verify the connections look right. Step 3 upgrades to camera-facing
quads for artistic control over line width and anti-aliasing.

### 7. Occlusion: per-segment ray clearance (Step 4)

Occlusion is handled **per sub-segment** using the existing
`rayClearance` infrastructure. The link compute shader traces from each
curve point (biased along the local normal) toward the eye and records the
minimum clearance. The render shader culls or fades segments whose
clearance is negative (occluded by the surface between the segment and the
camera).

This mirrors the shading pass's per-corner clearance approach but applied
to curve points. The cost is one `rayClearance` trace per curve point per
frame — for `N_SEGMENTS + 1 = 5` points and 6K stipple points, that is
~30K traces, comparable to the shading pass's existing ~24K corner traces.

**Step 1 uses a coarse visibility check**: a segment is drawn if at least
one of its two endpoints has positive maximum corner clearance in the
existing `shadingBuffer` (i.e. at least one endpoint's stipple splat is
visible to the camera). This is free (no extra compute) and sufficient
to validate the connection-finding logic.

### 8. Compile-time WGSL constants

All tunable parameters are compile-time constants defined **directly in
WGSL** (e.g. `const N_SEGMENTS: u32 = 4u;`). Their values are **not**
interpolated into the WGSL string from TypeScript — the WGSL source
contains the literal values. The TS side documents the same values in the
table below for reference, but the WGSL is the single source of truth at
runtime. Changing a value requires editing the WGSL source and
recompiling. These can be promoted to uniform buffer fields later if
runtime tuning is needed.

| Constant | Default | Description |
|---|---|---|
| `N_SEGMENTS` | `4` | Curve subdivision count per link |
| `DISTANCE_WEIGHT` | `1.5` | Scoring: alignment-vs-distance tradeoff (`score = alignment - weight * dist`) |
| `LINK_HALF_WIDTH` | `0.015` | Quad half-width in world units (camera-facing quad rendering) |
| `LINK_COLOR` | `vec3f(0.6, 0.8, 1.0)` | Line fragment color (soft blue) |
| `LINK_ALPHA` | `0.4` | Line fragment base opacity |
| `LINK_NEWTON_ITERS` | `3` | Newton iterations per curve-point projection |
| `LINK_NEWTON_ALPHA` | `0.7` | Newton step size (conservative for smin blend regions) |
| `LINK_RADIUS` | `= RELAX_RADIUS` | Search radius for neighbor scan (same as cell size) |
| `LINK_TAN_EPS` | `1e-4` | Threshold below which the light is considered parallel to the normal (`L_tan ≈ 0`) — no principal direction, no link |
| `COINCIDENT_EPS` | `1e-7` | Skip neighbors closer than this (link direction `diff / dist` undefined at zero distance; same threshold as the relax shader's coincident guard) |

These can be promoted to uniforms later if runtime tuning is needed.

### 9. No mutual-link dedup

If point `i` links to `j` and `j` links to `i`, both segments are drawn.
This is intentional — it produces denser, more organic-looking lines, and
the overlap is visually harmless since the segments are nearly coincident.
Dedup would require either a second pass to canonicalize pairs or an
atomic compare-and-swap protocol, both of which add complexity for no
visual benefit. Revisit if mutual links cause visual artifacts.

## Architecture & data flow

The link stages are inserted into the per-frame pipeline after shading and
before the render pass:

```
reproject → grid build → relax (substeps) → shading
   │
   ▼
grid rebuild (from post-relax buffer)          ← NEW (reuses SpatialGridPipeline)
   │
   ▼
link compute (scan + subdivide + project)      ← NEW — writes linkCurves buffer
   │  reads: points[readFromA], normalsBuffer, grid buffers
   │  writes: linkCurves (not ping-ponged)
   │
   ▼
render pass:
   link render (camera-facing quads)            ← NEW — drawn first
   point render (existing)                      — drawn on top
```

The grid rebuild reads from the same buffer relax most recently wrote
(`readFromA` after the substep loop), so cell assignments match the final
relaxed positions. The link compute reads the same buffer plus the fresh
grid and the shared normals buffer (written by the shading pass). The
`linkCurves` buffer is written once per frame and read by the link render
inside the same render pass as the point render.

While time is paused, the grid rebuild and link compute are skipped
(positions, normals, and animation time are unchanged, so the links from
the last playing frame stay valid); the link render still runs every
frame since its visibility handling is view-dependent.

### Placement in the frame loop (`src/main.ts`)

After the shading dispatch and before the render pass:

```ts
shading.dispatch(encoder, eye, time, shadingReadsA);

// Grid rebuild + link compute run only on frames where the simulation
// stepped; while paused the links cannot change.
if (playedThisFrame) {
    // --- Grid rebuild from post-relax positions (for link pass) ---
    spatialGrid.dispatch(encoder, shadingReadsA);

    // --- Link compute: find connections + subdivide + project ---
    link.dispatch(encoder, eye, time, shadingReadsA);
}

// --- Render pass: link render (under) + point render (top) ---
const pass = encoder.beginRenderPass({ /* ... */ });
linkRender.render(pass, eye, gpu.canvas, renderReadsA, points.count);
pointRender.render(pass, eye, gpu.canvas, renderReadsA, points.count);
pass.end();
```

## Buffer inventory

### Step 1 (simple line-list)

| Buffer | Size | Usage | Owner / writer |
|---|---|---|---|
| `linksBuffer` | `point_count * 4` B | `STORAGE` | link-pipeline (compute, once per frame) |
| `linkRenderUniform` | `80` B (`mat4x4f` + `vec3f` eye + pad) | `UNIFORM \| COPY_DST` | link-render-pipeline (CPU write per frame) |

### Step 2+ (subdivided curves)

| Buffer | Size | Usage | Owner / writer |
|---|---|---|---|
| `linkCurves` | `point_count * (N_SEGMENTS + 1) * 16` B | `STORAGE` | link-pipeline (compute, once per frame) |
| `linkRenderUniform` | `80` B (`mat4x4f` + `vec3f` eye + pad) | `UNIFORM \| COPY_DST` | link-render-pipeline (CPU write per frame) |

With `N_SEGMENTS = 4` and `POINT_COUNT = 6144`, `linkCurves` is
`6144 * 5 * 16 = 491,520 B ≈ 480 KB` — trivial.

## Module structure

| File | Responsibility |
|---|---|
| `src/stipple/link-pipeline.ts` | Owns the link compute pass: shader module, compute pipeline, params uniform, ping-pong read bind groups (A/B), and the output buffer (`linksBuffer` in Step 1, `linkCurves` from Step 2). |
| `src/stipple/shaders/link-shader.ts` | WGSL compute shader. Includes `SDF_COMMON`, duplicates `GridParams` (same as relax/shading — WGSL has no cross-module struct import). Exports `LINK_SHADER` (Step 1) and later `buildLinkShader(...)` (Step 2+) — constants are defined directly in the WGSL (see decision 8). |
| `src/stipple/link-render-pipeline.ts` | Owns the link render pipeline: shader module, render pipeline, uniform buffer, and bind group. From Step 3, no ping-pong bind groups (the curve buffer is not ping-ponged). |
| `src/stipple/shaders/link-render-shader.ts` | WGSL vertex/fragment shader. Step 1: `line-list` topology, reads point buffer + links buffer. Step 2: reads `linkCurves` buffer. Step 3: camera-facing quads. Step 4: occlusion culling. |

All new shaders are re-exported through `src/stipple/shaders/index.ts`.

## Implementation steps

Sequential. Stop after each step for manual review by the user. Later steps will not compile
or produce correct output until predecessors are complete. Run `npm run
lint` after each step that touches TS files.

### Step 1 — Link compute + simple line-list render (validation)

**Status:** completed.

**Goal:** verify the neighbor-finding logic produces visually sensible
connections with minimal rendering infrastructure.

**New files:**
- `src/stipple/shaders/link-shader.ts` — exports `LINK_SHADER` (Step 1
  variant: outputs `u32` best-neighbor index per point, no subdivision).
- `src/stipple/link-pipeline.ts` — `LinkPipeline` class. Constructor takes
  `(device, points: PointBuffers, grid: GridBuffers)`. Creates the
  `linksBuffer` (`u32 × point_count`, `STORAGE`, pre-filled with
  `UINT_MAX` so a frame rendered before the link compute has ever
  dispatched — e.g. paused at startup — draws no lines instead of
  linking every point to index 0), the compute pipeline, and two read
  bind groups (A/B — reads `points.bufferA` or `bufferB`,
  `normalsBuffer`, and the 4 grid buffers). `dispatch(encoder, time,
  readFromA)` writes the params uniform (`time`, `point_count`) and
  records one compute pass. `DISTANCE_WEIGHT` and `LINK_RADIUS` are
  compile-time WGSL constants (decision 8).
- `src/stipple/shaders/link-render-shader.ts` — exports
  `LINK_RENDER_SHADER` (Step 1 variant: `line-list` topology).
- `src/stipple/link-render-pipeline.ts` — `LinkRenderPipeline` class.
  Constructor takes `(device, points: PointBuffers, links: GPUBuffer,
  config: CameraConfig, format: GPUTextureFormat)`. Creates the render
  pipeline (`line-list` topology), uniform buffer (`mat4x4f` + `eye` +
  pad = 80 B), and two bind groups (A/B — binds: uniform,
  `points.bufferA` or `bufferB`, `links`, `shadingBuffer`).
  `render(pass, eye, canvas, readFromA, pointCount)` writes the
  view-projection uniform and calls `pass.draw(2, pointCount)`.

**Shader details — link compute (Step 1):**

The compute shader includes `SDF_COMMON` and duplicates `GridParams`
(byte-for-byte match with relax shader). Bindings:

| Binding | Type | Description |
|---|---|---|
| 0 | uniform `LinkParams` | `time: f32, point_count: u32` (8 B) |
| 1 | storage, read `points_in` | `array<Point>` — the post-relax point buffer |
| 2 | storage, read `normals_in` | `array<vec4f>` — shared normals buffer |
| 3 | uniform `GridParams` | Grid params (bbox, dims, cell size) |
| 4 | storage, read `sorted_values` | `array<u32>` |
| 5 | storage, read `cell_start` | `array<u32>` |
| 6 | storage, read `cell_count` | `array<u32>` |
| 7 | storage, read_write `links_out` | `array<u32>` — best neighbor index or `UINT_MAX` |

The neighbor scan mirrors the relax shader's 27-cell loop exactly (same
cell recomputation, same `UINT_MAX` empty-cell sentinel, same thin-wall
filter), plus a coincident-pair skip below `COINCIDENT_EPS` — the link
direction `diff / dist` is undefined at zero distance (see decision 8
for both guard constants). For each valid neighbor `j` in the forward
hemisphere (`alignment > 0`), compute `score = alignment -
DISTANCE_WEIGHT * dist` and track the maximum. After the scan, write
`best_j` (or `UINT_MAX` if no valid neighbor was found, including the
`L_tan ≈ 0` case, gated by `LINK_TAN_EPS`).

**Shader details — link render (Step 1):**

`line-list` topology, `draw(2, pointCount)` — 2 vertices per instance
(one line segment per point). Bindings:

| Binding | Type | Description |
|---|---|---|
| 0 | uniform `LinkRenderUniform` | `view_proj: mat4x4f, eye: vec3f, _pad: f32` (80 B) |
| 1 | storage, read `points` | `array<Point>` — the post-relax point buffer |
| 2 | storage, read `links` | `array<u32>` |
| 3 | storage, read `shading` | `array<ShadingSample>` |

Vertex shader: `vid` selects endpoint 0 or 1. Endpoint 0 = `points[inst]`
(the point itself); endpoint 1 = `points[links[inst]]` (the linked
neighbor). Both endpoints read from the same post-relax buffer: the
pipeline has two bind groups (A/B, bound at construction) and `render`
selects the buffer via the `readFromBufferA` flag — the same ping-pong
pattern as the point renderer. If `links[inst] == UINT_MAX`, push
offscreen. For visibility: read `shading[inst].clearance` and
`shading[links[inst]].clearance`; if both endpoints have
`max(clearance) <= 0.0` (fully occluded), push offscreen. Otherwise
project the endpoint via `view_proj` and output `clip_pos`.

Fragment shader: solid color `LINK_COLOR` with `LINK_ALPHA` (compile-time
WGSL constants — see decision 8).

**Frame loop wiring (`src/main.ts`):**

After `shading.dispatch` — but only on frames where time is playing
(while paused the post-relax positions, normals, and animation time are
unchanged, so the links computed on the last playing frame stay valid):
1. Call `spatialGrid.dispatch(encoder, shadingReadsA)` — rebuild grid
   from post-relax positions.
2. Call `link.dispatch(encoder, time, shadingReadsA)` — run link compute.
Unconditionally, in the render pass, before `pointRender.render` (the
occlusion culling is view-dependent, so lines keep updating with the
camera while paused):
3. Call `linkRender.render(pass, eye, gpu.canvas, renderReadsA,
   points.count)`.

Construct `LinkPipeline` and `LinkRenderPipeline` in `bootstrap()`,
after `spatialGrid` and before the frame loop starts. The
`LinkRenderPipeline` needs both `points.bufferA` and `points.bufferB`
(bound at construction; `render` selects via the `readFromBufferA` flag).

**Verify:** `npm run dev`, open the page. Lines should appear connecting
nearby points roughly orthogonal to the light direction. Rotate the
camera and observe that lines fade when both endpoints are occluded
(behind the surface). The visual will be crude (thin 1px lines, no
anti-aliasing, no curvature following) but should confirm the
neighbor-finding logic is working. If connections look random or missing,
debug the scoring / principal direction computation before proceeding.

### Step 2 — Curve subdivision + projected line-list render

**Goal:** make lines follow the surface curvature by subdividing each
link into `N_SEGMENTS` surface-projected segments.

**Changes:**
- `link-shader.ts`: replace the `u32 links_out` output with a
  `linkCurves_out: array<vec4f>` output of size
  `point_count * (N_SEGMENTS + 1)`. After finding `best_j`, subdivide
  `lerp(p_i, p_j, t)` for `t ∈ {0, 1/N, 2/N, ..., 1}` and Newton-project
  each sample onto the surface via `projectToSurface`. Write
  `linkCurves[i * STRIDE + k] = vec4f(projected_pos, 1.0)` for valid
  links, `vec4f(0.0, 0.0, 0.0, 0.0)` for invalid (`.w = 0.0` flag).
  All tunable constants (`N_SEGMENTS`, `DISTANCE_WEIGHT`, etc.) are
  defined directly in the WGSL as `const` declarations (see decision 8).
- `link-pipeline.ts`: replace `linksBuffer` with `linkCurves` buffer
  (`point_count * (N_SEGMENTS + 1) * 16` B). Update the bind group
  (binding 7 changes from `array<u32>` to `array<vec4f>` — same buffer,
  different WGSL type annotation). The `GridBuffers` interface is passed
  at construction (same as Step 1). Export `linkCurves` buffer for the
  render pipeline to bind.
- `link-render-shader.ts`: read `linkCurves` instead of `points +
  links`. Each instance `inst` maps to point `i = inst / N_SEGMENTS`,
  segment `s = inst % N_SEGMENTS`. Endpoint 0 =
  `linkCurves[i * STRIDE + s]`, endpoint 1 =
  `linkCurves[i * STRIDE + s + 1]`. If either `.w == 0.0`, push
  offscreen. `draw(2, pointCount * N_SEGMENTS)`.
- `link-render-pipeline.ts`: bind group now binds `linkCurves` (single
  buffer, no ping-pong needed for the curve data) plus the uniform. Drop
  the `pointsA` / `pointsB` / `links` / `shading` bindings (Step 2
  visibility uses the `.w` flag only; the `shadingBuffer`-based visibility
  from Step 1 is dropped — per-segment occlusion arrives in Step 4).
  `render(pass, eye, canvas, pointCount)` — no `readFromBufferA` needed.

**Verify:** Lines should now curve along the surface instead of cutting
through it. On highly curved regions the difference should be clearly
visible. Lines that previously penetrated the surface now hug it.

### Step 3 — Camera-facing quad rendering with soft AA

**Goal:** replace the crude `line-list` with billboarded quads for
artistic control over line width and anti-aliasing.

**Changes:**
- `link-render-shader.ts`: switch from `line-list` to `triangle-list`
  with `draw(6, pointCount * N_SEGMENTS)` (6 vertices = 2 triangles per
  quad, instanced per sub-segment). Vertex shader: read the two curve
  endpoints for this sub-segment, compute `seg_dir`, `view_dir`, `side`
  as in decision 6. Offset the quad corners by `±side *
  LINK_HALF_WIDTH` in world space. Pass the side coordinate (`-1..+1`)
  as a varying for AA. Fragment shader: `d = abs(side_coord)`,
  `alpha = smoothstep(1.0 - half_band, 1.0 + half_band, ...)` — soft
  edge using `fwidth(d)` (same AA pattern as the point shader's ring
  edge). Multiply by `LINK_ALPHA`.
- `link-render-pipeline.ts`: update pipeline topology to `triangle-list`.
  Blend state: `src-alpha` / `one-minus-src-alpha` (same as point
  renderer). No ping-pong bind groups needed (curve buffer is shared).
- The uniform grows to include `eye: vec3f` (already present from Step 1;
  now actually used by the vertex shader for the billboard computation).

**Verify:** Lines should now have visible width that foreshortens with
viewing angle, with smooth anti-aliased edges. Rotate the camera to
confirm the billboard behavior. Lines should still follow the surface
curvature (from Step 2).

### Step 4 — Per-segment occlusion

**Goal:** cull or fade line segments that are occluded by the surface
between the segment and the camera, so lines on the back of the shape
are not visible through it.

**Changes:**
- `link-shader.ts`: after computing each curve point `p_k`, compute the
  local normal `n_k = normalize(sdfGradient(p_k, time))`, bias by
  `SHADOW_BIAS`, trace `rayClearance(p_k + n_k * SHADOW_BIAS, eye_dir,
  dist_eye, time)` toward the eye, and store the clearance in the `.w`
  component of each curve point (replacing the binary validity flag).
  Invalid links (no neighbor found) still get `.w = 0.0` — a sentinel
  that is both "invalid" and "fully occluded", which the render shader
  treats the same way (push offscreen). Valid but occluded segments get
  `.w < 0.0`; visible segments get `.w > 0.0`.
  - **Param change:** the link params uniform grows to include `eye:
    vec3f` (needed for the clearance trace). Layout: `eye: vec3f, time:
    f32, point_count: u32, distance_weight: f32, link_radius: f32,
    _pad: u32` (32 B). `dispatch` gains an `eye` parameter.
  - **Cost:** `N_SEGMENTS + 1` `rayClearance` traces per point per frame.
    For `N_SEGMENTS = 4`, 6K points: ~30K traces. Comparable to the
    shading pass's existing ~24K corner traces. Acceptable.
- `link-render-shader.ts`: the vertex shader reads `.w` as clearance
  instead of a binary flag. Push offscreen if `.w <= 0.0` (invalid or
  fully occluded). Optionally pass clearance as a varying and let the
  fragment shader apply a soft fade near the occlusion boundary (same
  `smoothstep` pattern as the point shader's `alpha_clearance`), for a
  smoother transition than a hard cull.
- `link-pipeline.ts`: `dispatch` signature becomes `dispatch(encoder,
  eye, time, readFromA)`. Update params uniform write.

**Verify:** Rotate the camera. Lines on the far side of the shape should
disappear (occluded by the surface). Lines near silhouette edges should
show a soft fade. Lines on the near side should be fully visible.

### Step 5 — Mitering / gap elimination

**Goal:** eliminate hairline gaps at segment joints where two
sub-segments of the same curve meet.

**Approach:** The simplest fix is to **overlap sub-segments by one
sample**: render segment `k` from `p_k` to `p_{k+1}` and segment `k+1`
from `p_{k+1}` to `p_{k+2}`. The shared endpoint `p_{k+1}` is
identical, so if the quad width is sufficient and the camera-facing
offset is consistent, the quads should meet without gaps. However,
because each quad is independently billboarded (the `side` vector
depends on the segment direction, which differs between adjacent
sub-segments), there can be small wedge-shaped gaps at joints.

**Fix:** extend each quad by `LINK_HALF_WIDTH` along the segment axis at
both ends, so adjacent quads overlap by `2 * LINK_HALF_WIDTH`. This is a
pure vertex-shader change — offset the corner positions along
`±seg_dir * LINK_HALF_WIDTH` in addition to `±side *
LINK_HALF_WIDTH`. The fragment shader's AA edge still applies on the
sides; the axial overlap is hidden by the adjacent quad.

If this produces visible overdraw artifacts (e.g. doubled alpha at
joints), switch to a **miter joint**: compute the bisector of the two
adjacent segment directions at each joint and extend the quad corners to
meet at the miter point. This requires reading the adjacent segment's
direction (one extra curve point fetch per vertex) and is more complex;
defer until the overlap approach is confirmed insufficient.

**Verify:** Look closely at segment joints (zoom in). There should be no
visible gaps or dark seams. Rotate the camera to check from multiple
angles.

### Step 6 — Wire into frame loop + update AGENTS.md

**This step is integrated into each preceding step** (the frame loop
wiring is part of each step's `main.ts` changes). This final step is a
checkpoint to:
- Run `npm run lint` and confirm it is clean.
- Update `AGENTS.md` structural overview: add `link-pipeline.ts`,
  `link-render-pipeline.ts`, `link-shader.ts`, `link-render-shader.ts`
  to the module list. Update the `src/main.ts` bullet and the data-flow
  description to include the grid rebuild + link compute + link render
  stages.
- Update the `src/stipple` data-flow line to include the link stages.

## Verification

| Step | How to confirm |
|---|---|
| 1 | Lines appear connecting nearby points, roughly orthogonal to the light direction. Lines fade when both endpoints are occluded. Connections look sensible (not random). |
| 2 | Lines curve along the surface in high-curvature regions instead of cutting through. No visible penetration of the SDF surface. |
| 3 | Lines have visible width with smooth anti-aliased edges. Width foreshortens with viewing angle. Billboard behavior correct when rotating. |
| 4 | Lines on the far side of the shape are culled. Soft fade near silhouette edges. No lines visible through the surface. |
| 5 | No gaps at segment joints. No doubled-alpha seams. Correct from multiple viewing angles. |
| 6 | `npm run lint` clean. AGENTS.md updated. Full frame loop runs without errors. |

## Switch points / future work

- **Stale-grid reuse** — skip the post-relax grid rebuild and reuse the
  pre-relax grid (built before the relax substep loop). Points move
  <0.05/frame vs 0.3 cell size, so most neighbors are still found. If
  visual quality is acceptable, this saves one grid build per frame.
  Measure the quality impact before committing.
- **Luminance modulation** — modulate line alpha or color by the
  per-point luminance from `shadingBuffer` so lines darken in shadow.
  Requires binding `shadingBuffer` in the link render pipeline and
  interpolating luminance across the curve (or sampling it per curve
  point in the compute shader).
- **Density modulation**: With hatching, we typically do not encode brightness via luminance modulation of the lines but via density 
  modulation. Maybe we can try to achieve that later by modulating the forces in the force kernel of the relax step by the luminance of
  the points so that points have less spacing in darker regions.
- **Mutual-link dedup** — if mutual links (i→j and j→i) cause visual
  artifacts, add a canonicalization pass that keeps only `min(i, j)` as
  the link source.
- **Runtime-tunable parameters** — promote `DISTANCE_WEIGHT`,
  `LINK_HALF_WIDTH`, `LINK_COLOR`, `LINK_ALPHA`, `N_SEGMENTS` to uniform
  buffer fields (or a GUI) for live experimentation.
- **Variable segment count** — allocate a fixed maximum segment count
  per link but use fewer segments for short links (scaled by distance).
  Wastes some buffer space but avoids over-tessellating short links.
- **Miter joints** — if the overlap approach (Step 5) produces visible
  artifacts, upgrade to proper miter joints with bisector computation.
- **Depth attachment** — add a depth attachment to the render pass for
  hardware occlusion instead of the per-segment `rayClearance` trace.
  This would make the link compute pass cheaper (no occlusion traces)
  but requires a depth pre-pass or sharing the depth buffer with the
  debug render's ray-marched SDF. Significant architectural change.

## Risks / open questions

- **Visual quality of the scoring function** — the `alignment -
  DISTANCE_WEIGHT * dist` scoring is a heuristic. It may produce links
  that look good in flat regions but behave oddly in high-curvature
  areas (e.g. the principal direction changes rapidly, so adjacent
  points link to very different neighbors, producing jagged lines).
  Step 1's validation is designed to surface this before investing in
  the nicer rendering. If the connections look bad, the scoring function
  or the principal direction definition should be revised before
  proceeding.
- **`rayClearance` cost in the link compute** — Step 4 adds
  `N_SEGMENTS + 1` `rayClearance` traces per point per frame. If this
  is a frame-budget bottleneck, reduce `N_SEGMENTS` or trace only every
  other curve point (interpolating clearance between traced points).
- **Curve point normals** — Step 4 computes `sdfGradient` per curve
  point for the clearance trace bias. This is 4 extra `map()` evals per
  curve point. If the gradient is too noisy in smin blend regions, use
  the nearest stipple point's normal instead (cheaper but less accurate
  for the bias).
- **Line-list topology support** — Step 1 uses native `line-list`
  topology, which WebGPU supports but with implementation-defined line
  width (typically 1px). This is intentional for the validation step;
  Step 3 replaces it with camera-facing quads for full control.
