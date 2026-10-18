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

2. A **link render pass** that draws the curve segments as screen-space
   stroke quads (geometry built in 2D pixel coordinates) with soft
   anti-aliased edges and per-segment occlusion culling, composited
   under the existing stipple points.

The feature is always on (no feature gate); lines are drawn before the
point render so points sit on top.

## Non-goals (deferred to future work)

- **Mitering** — overlapping segments at joints to eliminate hairline
  gaps. Step 5 addresses this; v1 tolerates small gaps.
- **Line luminance modulation** — lines do not darken in shadow in v1
  (luminance is computed by the shading pass but not consumed by the link
  renderer for line color/alpha). Can be added once the basic rendering
  looks right. Point *density* modulation is handled separately in
  Step 1d.
- **Depth testing** — no depth attachment; occlusion is resolved via
  `rayClearance` toward the eye (Step 4), not via the rasterizer's depth
  buffer.
- **Stale-grid reuse** — the link pass rebuilds the spatial grid from
  post-relax positions. Reusing the pre-relax grid (built before the relax
  substep loop) is a worthwhile performance optimization to try later;
  points move <0.05/frame vs 0.3 cell size, so most neighbors are still
  found. See *Switch points*.

## Assumptions

- `POINT_COUNT` between `6 * 1024` and `16 * 1024` (currently
  `10 * 1024` in `main.ts`).
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
- `rayClearance(ro, rd, max_dist, threshold, time)` is available in
  `SDF_COMMON` and returns the minimum signed clearance between a ray and
  the SDF surface; `threshold` caps the reported positive clearance and
  sets the origin ignore-band (callers pass their primitive's scale).
- The shading buffer's `luminance` field holds the view-independent
  per-point luminance (`lit * lambert`), written by the reproject pass
  (Step 1c).
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

### 2. Single compute pass: scan → dedup → subdivide + project

The link compute does the neighbor scan, the dedup decision, and the
curve emission in a single pass of three dispatches (see decision 9 for
the dispatch breakdown). The subdivision runs in the third dispatch —
**after** the dedup decision — so dropped links never emit curve
points: the scan finds the best neighbor `j` for point `i` and records
it in the `links` buffer; the third dispatch (dedup+subdiv) then
evaluates the survival predicate and, for surviving links, samples a
cubic Hermite segment between the two stipple endpoints with C1
chain-window tangents (decision 11), Newton-projects each interior
sample onto the SDF surface, and writes `N_SEGMENTS + 1` `vec4f`
positions per point (`.xyz` = position, `.w` = validity flag) into
`linkCurves`; dropped or absent links zero their slice, so every slice
is rewritten each frame (the buffer persists across frames and keeps
rendering while paused). The `links` buffer is never mutated after the
scan — it stays the raw scan output, which is what makes the
subdivision's cross-link chain-window reads race-free (decision 11).

**Historical note.** This decision originally placed the subdivision
inline in the scan dispatch and eliminated the `u32` index buffer;
both were overtaken by decision 9 (owner election + dedup), which needs
the `u32` `links` buffer as working state and requires curve emission
to happen after the dedup decision so deduped-out edges never render.
Step 2c later replaced the chord subdivision (`lerp` + project) with C1
Hermite sampling and removed the in-place clearing of losing `links`
entries (decision 11).

**Step 1 exception.** Step 1 (the initial validation step) writes a
simple `u32` index per point and renders a flat line-list — no
subdivision, no projection. This is intentionally minimal to verify the
neighbor-finding logic produces visually sensible connections. Step 2
moves the curve buffer in alongside the (now internal) `u32` index
buffer.

**Cost.** `N_SEGMENTS − 1` Newton projections per surviving link
(~4 `map()` + gradient evals each — the endpoints are the stipple
points and are written exactly); dropped links only zero their slice.
Negligible compared to the shading pass's per-point visibility trace
(one `rayClearance` per point since Step 1c).

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

In Step 1, the output was a simpler `linksBuffer` of `u32 × point_count`
(best neighbor index or `UINT_MAX`); since Step 2, that buffer is
internal working state for the dedup machinery (decision 9) and the
render-facing output is the curve buffer. From Step 2, the
dedup+subdiv dispatch also reads the same ping-pong-selected point
buffer (for `p_i`/`p_j`), so it carries A/B bind groups of its own.

### 6. Rendering: screen-space capsule stroke quads (Step 3+)

From Step 3 onward, each sub-segment is rendered as a **screen-space
stroke quad**: the stroke geometry is built entirely in 2D pixel
coordinates (*stroke space* — one unit per canvas pixel, sharing NDC's
y-up orientation; the difference from the framebuffer's y-down
orientation is an isometry and does not affect distances). The vertex
stage perspective-projects the two endpoint curve points to pixel
coordinates, extrudes the quad as the bounding box of the 2D capsule
around the projected segment, and outputs it with a **constant clip
`w` and depth — the segment center's**:

```wgsl
out.clip_pos = vec4f(ndcFromPx(frag_px) * w_c, z_c, w_c);
```

so interpolation across the quad is exactly linear in screen space and
the interpolated stroke position tracks each fragment's true screen
position. The per-instance stroke geometry — projected endpoints
`p0_px` / `p1_px` and per-endpoint pixel radii `r0_px` / `r1_px` —
travels as flat varyings; the fragment position `frag_px` interpolates
in stroke pixels.

**Line shape — 2D capsule SDF in stroke pixels.** The fragment shader
evaluates a 2D segment distance field: the distance from the fragment's
pixel position to the closest point on the projected segment (clamped
to the endpoints), minus a stroke radius interpolated between the two
endpoint radii by the clamped closest-point parameter — a signed
distance whose zero contour is a **capsule** in screen space (tapered
body, semicircular rounded caps). Anti-aliased via `fwidth(sdist) *
AA_WIDTH` (same pattern as the point shader's disc SDF; `AA_WIDTH`
matches the point shader's so line and dot edges have consistent
softness), giving a ~1px soft edge independent of zoom and viewing
angle. `LINK_ALPHA = 1.0` — the line interior is fully opaque; only
the ~1px AA edge blends. This opacity is what makes the mitering trick
work (decision 11 / Step 5).

**Stroke width — world-space half-width, projected to pixels.** The
line keeps a world-space half-width (`LINK_HALF_WIDTH`, sized relative
to `POINT_RADIUS_WORLD` — lines read thinner than the dots they
connect; same philosophy as the point renderer), but converts it to a
pixel stroke radius **at each endpoint's own depth**. The projected
cross-section radius of a 3D tube is exactly proportional to the
inverse projective depth — the pixel displacement per world unit along
a view-perpendicular direction is a depth-independent pixel scale
divided by clip `w`:

```wgsl
let r_px = LINK_HALF_WIDTH * px_scale / clip.w;
```

`px_scale` is measured by transforming a view-perpendicular direction
`perp = normalize(cross(view_dir, up))` (as a direction, `w = 0`; axis
fallback when `abs(view_dir.y) > 0.9`) through the view-projection and
scaling to pixels:

```wgsl
let perp_clip = u.view_proj * vec4f(perp, 0.0);
let px_scale = length(perp_clip.xy * (0.5 * canvas_size));
```

A view-perpendicular reference direction is used — not the projected
segment itself — because the projected segment's on-screen length
collapses to zero exactly in the near-parallel case this shader must
survive, while `perp` lies in the camera's xy plane, so `perp_clip.xy`
is always non-zero and `px_scale` always finite. The radius tapers
linearly along the segment body (interpolated by the clamped
closest-point parameter) and is held constant within each rounded cap
— the stroke reads with the same pixel width its 3D tube would
project: shrinking with distance, and collapsing toward its rounded
cap as the segment turns parallel to the view.

**Quad sizing — pixels throughout.** The quad half-extent is:

```wgsl
let quad_half = max(r0_px, r1_px) + AA_PIXELS;
```

It extends along the projected axis by `quad_half` beyond each endpoint
(so the rounded caps fit) and across by `±quad_half` (the larger
endpoint radius plus the AA margin on each side). At extreme zoom-out
the stroke radius collapses sub-pixel and the padding takes over — the
stroke settles to a stable ~`AA_PIXELS`-pixel minimum width that never
flickers.

**Why screen space — continuity at grazing angles.** Each capsule's
rounded cap extends the stroke radius past the shared endpoint in
*every* screen direction, so consecutive sub-segments always overlap at
their joints and the union renders as a rounded, welded miter with no
hairline gaps — at any viewing angle. Shared curve points also share a
projected depth, so neighboring capsules agree on the joint's stroke
radius from both sides. A world-space capsule cannot do this: its
along-axis reach past the shared endpoint foreshortens by the sine of
the angle to the view, starving exactly those joint overlaps when a
segment is viewed near-parallel (and its view-compressed distance
field blows the `fwidth`-derived AA band up by the same factor, fading
the caps out). The near-parallel case degenerates gracefully instead:
the projected segment collapses, the capsule becomes a disc, and
consecutive discs overlap into a continuous blob along the collapsed
chain.

**Guards.** The vertex shader pushes a sub-segment offscreen when:

- **Occluded** (Step 4) — `max(e0.w, e1.w) <= 0.0`: neither endpoint
  curve point has positive clearance toward the eye.
- **Behind the camera** — `clip0.w <= 0 || clip1.w <= 0`: an endpoint
  with non-positive clip `w` divides by a negative `w` and projects to
  a mirrored, garbage screen position. This also covers the segment
  center, whose clip `w` is the affine midpoint of the endpoints'.
- **Eye on the segment** — `dist_eye < EYE_EPS`: this close, the view
  direction (and with it the perpendicular basis) is numerically
  undefined — position differences at scene scale are noise-dominated —
  and the stroke would enclose the camera.
- **Collapsed projection** — `dot(px1 - px0, px1 - px0) <
  DEGENERATE_PX2` (coincident curve points or a view-parallel
  segment): the 2D basis falls back to a fixed axis, both radii
  flatten to the larger one, and the fragment's distance field
  degenerates to a plain point distance (disc).

**Uniform.** 96 B layout: `mat4x4f` view-projection (64) +
`vec3f` eye (16-aligned) + `vec2f` canvas_size (8) + pad (4) = 96 B.
`canvas_size` converts between NDC and pixel coordinates and scales
the pixel measurements; `eye` provides the view direction for the
view-perpendicular pixel scale.

**Historical note.** Step 3 originally rendered **world-space
camera-facing quads** — billboarded around the segment axis
(`side = normalize(cross(seg_dir, view_dir))`, guarded against the
parallel-segment degeneracy), with the capsule SDF evaluated against
world-space positions and the quad oversized to
`LINK_HALF_WIDTH + AA_PIXELS * wpp`, where `wpp` was a measured
world-units-per-pixel factor. It was replaced by the screen-space
approach above: the world-space capsule's along-axis reach past a
shared endpoint foreshortens by the sine of the angle to the view,
starving the joint overlaps near grazing angles, and its
view-compressed distance field blows the `fwidth`-derived AA band up
by the same factor, fading the caps out.

**Step 1 and Step 2 use a simple `line-list`** with native `line-list`
topology — no quads, no projection math. This is the minimal path to
verify the connections look right. Step 3 upgrades to the screen-space
capsule stroke quads described here.

### 7. Occlusion: per-segment ray clearance (Step 4)

Occlusion is handled **per sub-segment** using the existing
`rayClearance` infrastructure. The link compute shader traces from each
curve point (biased along the local normal) toward the eye and records the
minimum clearance. The render shader culls or fades segments whose
clearance is negative (occluded by the surface between the segment and the
camera).

This reuses the same `rayClearance` infrastructure the shading pass uses
for its per-point visibility trace (one center ray per point since
Step 1c), applied to curve points. The cost is one `rayClearance` trace
per curve point per frame — for `N_SEGMENTS + 1 = 5` points and 6K
stipple points, that is ~30K traces, about 5× the shading pass's
per-point visibility traces.

**Step 1 uses a coarse visibility check**: a segment is drawn if at least
one of its two endpoints has positive clearance in the existing
`shadingBuffer` (since Step 1c a single center-ray value per point; i.e.
at least one endpoint's stipple dot is visible to the camera). This is
free (no extra compute) and sufficient to validate the
connection-finding logic.

### 8. Compile-time WGSL constants

Shader constants are compile-time WGSL `const` definitions, not uniform
buffer fields — no runtime plumbing is added for values the shaders
never need to change while running. Which side owns a constant follows
from where the value is needed:

- **Needed only in WGSL** — define it directly in the shader source
  with its literal value (e.g. `const DISTANCE_WEIGHT: f32 = 1.5;`).
  The WGSL is the single source of truth and changing the value is a
  one-line shader edit. No TS mirror, no interpolation, no uniform —
  this is the preferred, lowest-friction case.
- **Also needed on the TypeScript side** (sizing a buffer, computing a
  draw's instance count, anything a pipeline constructor must know) —
  define it once in TypeScript and interpolate the value into the WGSL
  string. The TS definition is the single source of truth, so the two
  sides cannot drift; never duplicate the value as a literal in WGSL.

### 9. Target dedup: at most one incoming edge per point (owner election)

Without dedup, every point has at most one **outgoing** edge (its own
best-neighbor pick) but unbounded **incoming** edges — popular targets
become fans/stars, which destroys the polyline reading. The link stage
canonicalizes by target so each point has at most one incoming edge;
combined with the single outgoing edge, every point's degree is ≤ 2 and
the surviving graph is a disjoint union of paths and cycles — the
hatch-line/streamline topology.

**Mechanism (owner election).** Chosen over radix-sorting the
`(target, source)` edge pairs and keeping the first edge of each
equal-target run: a stable sort of source-ordered edges keeps the
smallest source index first, and `atomicMin` elects exactly that winner
with ~10× less memory traffic, no single-workgroup serialization, and
no extra sort machinery. Three dispatches inside one compute pass:

1. Clear `owners` to `UINT_MAX`.
2. The link compute scan writes `links[i] = best_j` and, for valid
   links, `atomicMin(&owners[best_j], i)` — after the dispatch,
   `owners[j]` is the smallest source index targeting `j`.
3. The **dedup+subdiv** dispatch evaluates, per point, whether `i`'s
   edge survived the election (`owners[links[i]] == i`) and suppresses
   `i`'s curve emission when it did not — since Step 2c it does **not**
   clear `links[i]` to `UINT_MAX` anymore: `links` stays the raw scan
   output so the subdivision's chain-window reads are race-free
   (decision 11). It **additionally breaks mutual 2-cycles**: when
   `owners[j] == i` (i's edge survived) **and** `owners[i] == j` (j's
   surviving edge targets i) **and** `i > j`, the edge is dropped (in
   Step 1b–2b by clearing `links[i]`; since Step 2c by skipping the
   curve emission — the predicate itself is unchanged). This reads only
   the stable `owners` buffer (written by the prior scan+elect
   dispatch) plus the raw `links` slots, so it is race-free within the
   dispatch — no new pass, buffer, or dispatch. Both i and j evaluate
   the predicate; exactly the larger index drops, so one segment of
   each mutual pair survives. It then emits the render-facing geometry
   (decision 2): surviving links are emitted as `N_SEGMENTS + 1`
   Hermite-sampled, surface-projected curve points (decision 11)
   written to `linkCurves`; dropped or absent links zero their slice.

All three run inside `LinkPipeline.dispatch` (one compute pass, three
dispatches — WebGPU orders dispatches within a pass, so each stage's
writes are visible to the next), so `main.ts` is unchanged. Winner
criterion: smallest source index (deterministic). Mutual links (2-cycles
i↔j) are removed by the fused mutual-break in the dedup+subdiv
dispatch — no separate chain-filtering pass is needed. The `links`
buffer is internal working state — the raw scan output since Step 2c
(decision 11); since Step 2 the render consumes the curve buffer
instead (decision 5).

**Cost.** Two extra N-sized dispatches plus one atomic per point —
~120 KB traffic, no contention hot spots. Negligible next to the
shading pass.

### 10. No chain cutting

An earlier revision of this spec included chain-length filtering:
dropping polylines shorter than a minimum link count and bisecting
those longer than a maximum, via per-round pointer-chasing passes over
a `prev`/`keep` buffer pair. It was implemented and tested, and
reverted: the cuts jumped frame-to-frame as chain membership shifted
with the animated surface, so polylines flickered between lengths and
the image read as unstable and visually noisy rather than structured.

Chains are now left unfiltered — the link stage runs only owner clear
→ scan+elect → dedup (with the fused mutual-break from decision 9),
and whatever graph that produces renders as-is, including single-link
fragments and long polylines. The mutual-pair case is still handled
(decision 9), since it is cheap, race-free, and stable; general chain
reshaping is not.

### 11. C1 chain-window Hermite subdivision (Step 2c)

Step 2's curve was the chord polyline `lerp(p_i, p_j, t)` Newton-
projected at `N_SEGMENTS + 1` samples — it follows the surface but is
angular: consecutive chords meet at sharp turning angles inside a link
and, worse, at the joints where one link's curve hands off to the next
(the incoming and outgoing tangents at a joint were unrelated, since
the two links were chosen independently). Step 2c replaces the
subdivision with cubic Hermite segments whose endpoint tangents are
**chain-window averages**, making the rendered chain of curve segments
C1 (tangent-continuous; curvature may still jump at joints — the
desired wavy-but-not-rubber-hose character).

**Chain windows.** The link graph is a union of paths and cycles
(decision 9); write its stipple vertices in chain order as `q_0 … q_n`
(each `q_m` is `points[idx_m]`, with `idx_{m+1} = links[idx_m]` along
the chain). Segment `σ_m = (q_m, q_{m+1})` — in implementation terms,
the surviving link of the point `idx_m` — computes:

- the tangent at its start vertex `q_m` from the window
  `{q_{m-1}, q_m, q_{m+1}}`,
- the tangent at its end vertex `q_{m+1}` from the window
  `{q_m, q_{m+1}, q_{m+2}}`.

Both adjacent segments evaluate the tangent at their shared vertex
from the same three stipple points, and the tangent rule is a **pure
function of that window** — so both compute identical tangent vectors
and the chain is C1 (parametrically, since the magnitude rule is also
window-pure).

**Tangent rule** (window-pure; robust to the uneven stipple spacing
produced by the luminance density modulation):

```wgsl
d_in  = v - pred;  d_out = succ - v;              // incident chords
u_in  = d_in / |d_in|;  u_out = d_out / |d_out|;  // unit directions
dir   = normalize(u_in + u_out);                  // averaged direction
mag   = LINK_TANGENT_SCALE * (|d_in| + |d_out|) / 2
t     = mag * dir
```

Unit chords are averaged *before* normalizing, so bunched knots cannot
overshoot (the failure mode that would force centripetal
parameterization in a plain Catmull-Rom). `LINK_TANGENT_SCALE` (see the
decision 8 table) rounds the joints more broadly as it grows and hugs
the chords as it shrinks. Guards are window-pure, so C1 survives them:
one incident chord shorter than `COINCIDENT_EPS` → follow the other;
near-antiparallel chords (hairpin) → the window's through-direction
`normalize(succ − pred)`; a fully degenerate window → zero tangent (the
Hermite reduces to a plain interpolation). Chain ends (no valid
predecessor/successor) use the one-sided chord tangent
`LINK_TANGENT_SCALE · (q_{m+1} − q_m)` — the same vector serves the
start of the first segment and the end of the last.

**Race-free windows inside the subdiv dispatch.** The window vertices
come from stipple-point positions, not from neighbor curve slices, so
no invocation depends on concurrently written curve data — except one:
the end-vertex window needs `q_{m+2} = points[links[idx_{m+1}]]`, a
*different* invocation's `links` slot. This is race-free because
`links` is never mutated after the scan: the dedup dispatch used to
clear losing entries in place (Step 1b–2b); Step 2c removes that write
and moves the survival predicate to the consumers — the dispatch itself
still evaluates it to gate curve emission, and the link render's
interim coarse cull (Step 2b) reads `links[i]` only for surviving
links, where the raw value equals the post-dedup value anyway.
Validation of the window candidates uses only finalized `links` +
`owners` state:

- predecessor of `q_m`: `a = owners[i]`; valid iff `a != UINT_MAX` and
  not mutual-dropped (`!(owners[a] == i && a > i)`) — else chain start,
  one-sided.
- successor of `q_{m+1}`: `k = links[j]` (`j = links[i]`, the own raw
  slot); valid iff `k != UINT_MAX && k != i && owners[k] == j` — the
  `k == i` guard rejects the degenerate double-back window (mutual pair
  or dropped back-pointer, where the window would average to a zero
  tangent), and `owners[k] == j` rejects a successor that lost its
  target's election — else chain end, one-sided.

**Sampling and projection.** The Hermite segment is evaluated at
`N_SEGMENTS + 1` parameter values `s ∈ {0, 1/N, …, 1}`; the endpoints
(`s = 0, 1`) are the stipple points themselves and are written exactly
(both adjacent segments write the identical endpoint position, so
segments join seamlessly), while interior samples are Newton-projected
via `projectToSurface`. `N_SEGMENTS` is a pure tessellation knob — the
curve's *shape* is fixed by the windows and `LINK_TANGENT_SCALE`,
independent of the sample count (Step 2's concern that higher
subdivision revealed tighter corners no longer applies). Cost is
`N_SEGMENTS − 1` projections per surviving link — below Step 2's
`N_SEGMENTS + 1` (the endpoints are free).

**Not addressed here.** Vertices stay pinned to the stipple points, so
chain-level zigzag (noisy picks making the chain itself wander) is
untouched — the *rendering* through the joints is smooth, the chain
shape is exactly what the picks produced. Smoothing the
principal-direction field before the scan (or best-continuation
election, see *Switch points*) remains the lever for that, if
curve-level smoothness proves insufficient.

## Architecture & data flow

The link stages are inserted into the per-frame pipeline after shading and
before the render pass:

``
reproject (+ per-point luminance, Step 1c) → grid build → relax (substeps)
   → shading (normals refresh + center-ray visibility clearance)
   │
   ▼
grid rebuild (from post-relax buffer)          (playing frames only; reuses SpatialGridPipeline)
   │
   ▼
link compute (owner clear → scan+elect → dedup+subdiv)
   │                                            (playing frames only)
   │  reads: points[readFromA], normalsBuffer, grid buffers
   │  writes: linkCurves (not ping-ponged; .w = validity placeholder)
   │  (one compute pass, 3 dispatches — decisions 2 and 9)
   │
   ▼
link clearance (per-curve-point rayClearance toward the eye)
   │                                            (every frame — view-dependent)
   │  reads: linkCurves, normalsBuffer, links, owners
   │  writes: linkCurves .w (occlusion clearance; 0.0 for dropped links)
   │  (decision 7)
   │
   ▼
render pass:
   link render (screen-space stroke quads, 2D capsule SDF;
   clearance cull)                             ← drawn first
   point render (existing)                     — drawn on top
```

The grid rebuild reads from the same buffer relax most recently wrote
(`readFromA` after the substep loop), so cell assignments match the final
relaxed positions. The link compute reads the same buffer plus the fresh
grid and the shared normals buffer (written by the shading pass). The
`linkCurves` buffer is written by the dedup+subdiv dispatch on playing
frames (positions + validity placeholder), has its `.w` clearance
refreshed by the clearance dispatch every frame, and is read by the link
render inside the same render pass as the point render.

The relax substeps additionally read the shared `shadingBuffer` for
per-neighbor luminance to modulate point density (Step 1d). Since
Step 1c, the luminance is written by the **reproject pass** earlier in
the same frame (before the relax substeps), so relax reads fresh
same-frame values rather than last frame's. The buffer is not
ping-ponged, so the binding is the same for both relax read bind groups
(A/B).

While time is paused, the grid rebuild and the link compute pass are
skipped (positions, normals, and animation time are unchanged, so the
links and curve geometry from the last playing frame stay valid); the
clearance dispatch and the link render still run every frame since both
are view-dependent — the clearance traces keep the line occlusion
tracking the camera against the frozen curve geometry.

### Placement in the frame loop (`src/main.ts`)

After the shading dispatch and before the render pass:

```ts
shading.dispatch(encoder, eye, time, shadingReadsA);

// Grid rebuild + link compute run only on frames where the simulation
// stepped; while paused the links cannot change. The clearance pass
// and the link render run every frame — both are view-dependent.
if (playedThisFrame) {
    // --- Grid rebuild from post-relax positions (for link pass) ---
    spatialGrid.dispatch(encoder, shadingReadsA);

    // --- Link compute: find connections → dedup → subdivide + project ---
    link.dispatch(encoder, eye, time, shadingReadsA);
}

// --- Link clearance: per-curve-point occlusion, every frame ---
link.dispatchClearance(encoder, eye, time);

// --- Render pass: link render (under) + point render (top) ---
const pass = encoder.beginRenderPass({ /* ... */ });
linkRender.render(pass, eye, target, gpu.canvas, points.count);
pointRender.render(pass, eye, target, gpu.canvas, renderReadsA, points.count);
pass.end();
```

## Buffer inventory

### Step 1 (simple line-list)

| Buffer | Size | Usage | Owner / writer |
|---|---|---|---|
| `linksBuffer` | `point_count * 4` B | `STORAGE` | link-pipeline (compute, once per frame) |
| `ownersBuffer` | `point_count * 4` B | `STORAGE` | link-pipeline (compute: clear + atomicMin election + read in dedup, once per frame) |
| `linkRenderUniform` | `96` B (`mat4x4f` + `vec3f` eye + `vec2f` canvas_size + pad) | `UNIFORM \| COPY_DST` | link-render-pipeline (CPU write per frame) |

### Step 2+ (subdivided curves)

| Buffer | Size | Usage | Owner / writer |
|---|---|---|---|
| `linksBuffer` | `point_count * 4` B | `STORAGE` | link-pipeline (internal: raw scan output — best-neighbor index or `UINT_MAX`, never mutated after the scan; consumers evaluate the survival predicate from `links` + `owners`) |
| `ownersBuffer` | `point_count * 4` B | `STORAGE` | link-pipeline (internal: clear + atomicMin election + read in dedup+subdiv, once per frame) |
| `linkCurves` | `point_count * (N_SEGMENTS + 1) * 16` B | `STORAGE` | link-pipeline (compute, once per frame) |
| `linkRenderUniform` | `96` B (`mat4x4f` + `vec3f` eye + `vec2f` canvas_size + pad) | `UNIFORM \| COPY_DST` | link-render-pipeline (CPU write per frame) |

With `N_SEGMENTS = 8` and `POINT_COUNT` between `6 * 1024` and
`16 * 1024`, `linkCurves` is `POINT_COUNT * 9 * 16` B ≈ 0.86 MB –
2.25 MB — trivial.

## Module structure

| File | Responsibility |
|---|---|
| `src/stipple/link-pipeline.ts` | Owns the link compute pass: shader module, compute pipelines, params uniform, ping-pong read bind groups (A/B) for the scan and the dedup+subdiv stage, the render-facing output buffer (`linksBuffer` in Step 1, `linkCurves` from Step 2), and the internal working buffers (`linksBuffer` from Step 2, `ownersBuffer`). |
| `src/stipple/shaders/link-shader.ts` | WGSL compute shader. Includes `SDF_COMMON`, duplicates `GridParams` (same as relax/shading — WGSL has no cross-module struct import). Exports `LINK_SHADER` (one module, one entry point per stage) and `LINK_N_SEGMENTS` (the TS-owned sub-segment count, interpolated into the WGSL — decision 8's TS-owned rule). Constants needed only in WGSL are defined directly in the shader source (see decision 8). |
| `src/stipple/link-render-pipeline.ts` | Owns the link render pipeline: shader module, render pipeline (`triangle-list` topology, instanced 6-vertex quads), the 96 B uniform buffer (`mat4x4f` view-projection + `vec3f` eye + `vec2f` canvas_size + pad), and a single bind group (uniform + curve buffer — nothing ping-ponged). |
| `src/stipple/shaders/link-render-shader.ts` | WGSL vertex/fragment shader. Step 1: `line-list` topology, reads point buffer + links buffer. Step 2: reads `linkCurves` buffer. Step 3: screen-space capsule stroke quads (2D pixel-coordinate geometry, projected 2D capsule SDF fragment, `fwidth` AA, degeneracy guards — see decision 6). Step 4: occlusion culling. |

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
   compile-time WGSL constants (decision 8). Step 1b adds the
   `ownersBuffer`, the owner-clear and dedup pipelines, and the
   three-dispatch pass — see below.
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
| 8 | storage, read_write `owners_elect` | `array<atomic<u32>>` — per-target elected source index (Step 1b owner election) |

The neighbor scan mirrors the relax shader's 27-cell loop exactly (same
cell recomputation, same `UINT_MAX` empty-cell sentinel, same thin-wall
filter), plus a coincident-pair skip below `COINCIDENT_EPS` — the link
direction `diff / dist` is undefined at zero distance. For each valid neighbor
`j` in the forward hemisphere (`alignment > 0`), compute `score = alignment -
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
| 3 | storage, read `shading` | `array<PointShading>` |

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

### Step 1b — Target dedup (owner election)

**Status:** completed.

**Goal:** canonicalize links by target so each point has at most one
incoming edge — with the single outgoing edge, degree ≤ 2 and the
surviving graph becomes a union of paths and cycles instead of fans at
popular targets. See decision 9.

**Changes:**
- `link-shader.ts`: adds `link_owner_clear_cs` (clears `owners` to
  `UINT_MAX`) and `link_dedup_cs` (clears `links[i]` when
  `owners[links[i]] != i`, and additionally breaks mutual 2-cycles per
  decision 9); `link_cs` gains the
  `atomicMin(&owners[best_j], i)` election. The owners buffer is bound
  at binding 8 in the scan (`atomic` read_write) and dedup (read)
  entries, and at binding 1 in the clear entry — bindings may repeat
  across entry points since each pipeline derives its layout from its
  own entry point's statically used resources.
- `link-pipeline.ts`: creates the `ownersBuffer` (`u32 × point_count`,
  `STORAGE`, internal) and three compute pipelines from one shader
  module. `dispatch` records one compute pass with three dispatches:
  owner clear → scan+elect → dedup.
- `main.ts`: unchanged — the `dispatch` signature and call site stay
  the same.

**Verify:** `npm run dev`. No fans/stars: every point has at most one
incoming and one outgoing edge, so links form chains. No mutual-link
pairs (both segments of an i↔j pair render) — the dedup dispatch's
fused mutual-break drops the larger-index edge of each 2-cycle. The
number of rendered segments drops (edges = distinct targets). If lines
become too sparse, revisit the winner criterion (best-continuation
election, see *Switch points*).

### Step 1c — Simplified point renderer + pre-relax luminance

**Status:** completed.

**Goal:** demote the point renderer from the elaborate corner-clearance
splat renderer to a simple fixed-size dot overlay, and restructure the
shading work so the view-independent luminance is computed *before*
relax. Step 1d's density modulation then reads fresh same-frame values
instead of last frame's, and the first-frame transient disappears.

**Point renderer — fixed-size dots.** The tangent-plane quad stays
(normals-buffer orientation, `POINT_RADIUS_WORLD` half-extent, 6-vertex
triangle list, src-alpha blending), but the fragment shader paints a
plain anti-aliased **filled disc** of fixed radius instead of a
luminance-scaled ring, and there are no per-corner clearance values and
no per-fragment clearance clipping. Dots therefore have the same size
in lit and shadowed regions. The vertex shader keeps a coarse occlusion
cull: a point whose **center-ray** clearance (from the shading buffer)
is ≤ 0 is pushed offscreen — one comparison, not a corner check.

**Shading pass slimmed to normals + visibility clearance.** The pass
keeps the post-relax normals refresh and now traces a single center-ray
`rayClearance` per point (instead of center + up to four corner traces,
each with a `projectToSurface`). It no longer computes luminance, and
its writes are member-wise (`shading_out[i].clearance` only) so the
reproject-written luminance in the same buffer is not clobbered. The
back-face guard (`dot(n, eye_dir) <= 0`) writes an explicitly occluded
clearance (a negative value) instead of being traced. The pass remains
view-dependent and runs every frame, including while paused.

The per-point shading record is a **single packed 16-byte value**
(`PointShading` in `SDF_COMMON`): the view-independent `luminance` and
the center-ray `clearance` as named `f32` fields plus two spare slots —
every per-point buffer element is 16 bytes, and each pass writes one
field. Compared to the old per-corner maximum, the cull tightens
slightly near silhouettes (a point whose center is occluded but whose
corners peek out is now culled) — acceptable for a stopgap that Step 4
replaces with per-segment traces.

**Luminance moved into the reproject pass.** Luminance
(`lit * lambert`) is view-independent — it needs only position, normal,
and the light, never the eye — so it can be computed before relax. The
reproject pass already evaluates each point's projected position and
fresh surface normal; after projecting, it computes
`lambert = max(dot(n, lightDir(time)), 0.0)`, traces
`softShadow(p + n·SHADOW_BIAS, light, time)` only when `lambert > 0`,
and writes `shading_out[i].luminance = lit * lambert` (member-wise,
leaving the `clearance` field — owned by the shading pass — untouched). The pass stays view-independent and gated on
`playedThisFrame`, like relax itself. The old view-dependent luminance
semantics (zeroed for back-facing points, consumed by the retired ring
renderer) is retired; the `luminance` field now holds the
view-independent value. First-frame transient: none — reproject
writes luminance before relax's first dispatch, so relax never reads
the zero-initialized buffer.

**Changes:**
- `point-shader.ts`: drop the `luminance`/`clearance` varyings and the
  ring; render an AA filled disc at a fixed radius (new WGSL constant
  `POINT_DOT_RADIUS`, a fraction of the quad half-extent); keep the
  center-clearance offscreen cull in the vertex shader.
- `shading-shader.ts`: drop `cornerClearance` and luminance; trace one
  center-ray clearance; member-wise `clearance` writes.
- `reproject-shader.ts`: add the `shading_out` binding (binding 3) and
  the luminance computation/write.
- `reproject-pipeline.ts`: add `shadingBuffer` to both bind groups.
- `link-render-shader.ts`: the coarse visibility check reads the scalar
  `clearance` field directly (the `max()` over four components goes
  away).
- `main.ts`: add `DRAW_POINTS = true` (const, beside `DRAW_DEBUG`) and
  guard the `pointRender.render` call; update the frame-loop TSDoc.
- `AGENTS.md`: update the structural-overview bullets for
  `reproject-pipeline.ts`, `shading-pipeline.ts`,
  `point-render-pipeline.ts`, and `point-buffers.ts`.

**Verify:** `npm run dev`. Points render as identical-size dots in lit
and shadowed regions (no ring, no luminance-dependent size). Dots
disappear when `DRAW_POINTS = false`; links are unaffected. Dots on the
far side of the shape are still culled via the center-ray clearance.
Run `npm run lint` (the step touches TS files).

### Step 1d — Luminance-driven density modulation in the relax kernel

**Status:** completed.

**Goal:** make points pack more densely in unlit (low-luminance) regions
and spread out in lit regions, so that hatch lines — which connect
neighboring points — are denser in shadow and sparser in light,
encoding brightness via density (the classical hatching signal) rather
than via line color or alpha.

**Mechanism — j-based effective radius.** The relax kernel's force is
`force += (1 - d_E/r) * diff/d_E`, gated by `d_E > r`. To modulate
density, the effective interaction radius is computed **per neighbor j**
from j's luminance:

```wgsl
let lum_j = shading_in[j].luminance;
let r_eff_j = r * mix(MIN_RADIUS_SCALE, 1.0, lum_j);
```

and substitutes `r_eff_j` for `r` in both the distance gate and the
kernel envelope:

```wgsl
if (d_E > r_eff_j) { continue; }                      // was: d_E > r
// ... coincident kick and thin-wall filter unchanged (keep original r) ...
force += (1.0 - d_E / r_eff_j) * diff / d_E;           // was: (1.0 - d_E / r)
```

The thin-wall filter and coincident-pair kick keep the original `r` —
they are geometric predicates (surface thickness, degenerate pair
separation), not spacing controls. `lum_i` is not read; the modulation
is purely j-based, so point i's own luminance never enters its own
force accumulation.

Dark neighbors (low `lum_j`) get a small effective radius — they only
repel within a small shell, so points can pack close around them. Lit
neighbors get the full radius — normal spacing. The asymmetry is key:
when point i is processed, the force from a lit neighbor uses
`r_eff = r` (full push), while the force from a dark neighbor uses
`r_eff = r * MIN_RADIUS_SCALE` (weak push). At a lit/dark boundary,
dark points feel strong repulsion from their lit neighbors (pushed
deeper into the dark region) but weak repulsion from their dark
neighbors (can sit close) — net migration into dark regions. In a
uniform dark interior, all neighbors are dark → all radii small →
equilibrium spacing scales with `r * MIN_RADIUS_SCALE` → up to
`(1/MIN_RADIUS_SCALE)²`× denser than lit regions. Points are conserved
(no creation/destruction); density redistribution is emergent from the
asymmetric force.

**Luminance source — computed by the reproject pass (Step 1c).** The
reproject pass already computes each point's projected position and
fresh surface normal before relax runs; Step 1c extended it to also
compute the view-independent luminance `lit * lambert` (where
`lit = softShadow(...)` ∈ [0,1] is shadow-aware and `lambert =
max(dot(n, light), 0.0)` ∈ [0,1], with the `softShadow` trace skipped
when `lambert <= 0`) and write it to the shading buffer's `luminance`
field. The
values are view-independent (they depend only on position, normal, and
the light — never on the camera), so density tracks the light and stays
stable while orbiting. Because the reproject pass dispatches before the
relax substeps in the same frame, relax reads **fresh same-frame**
values — no cross-frame staleness, no first-frame transient (see
below). Luminance is computed from pre-relax positions; points drift
<0.05/frame vs 0.3 cell size, so the within-frame drift is negligible.

**Why not alternatives:**

- *i-based radius* (compute `r_eff` from point i's own luminance):
  produces migration in the wrong direction — lit points flee dark
  neighbors (full radius felt), dark points don't move (reduced radius
  → weak forces felt) → lit regions empty, dark regions don't fill.
  Rejected.
- *Drift force* (`F_drift = -∇lum`): the 27-cell gradient estimate is
  too local (0.3-unit scale) and noisy to drive smooth migration.
  Rejected.
- *Dedicated pre-relax shadow pass*: an extra compute pass per frame
  (~6K `softShadow` traces) for fresh, shadow-aware luminance.
  Superseded — Step 1c folded the luminance computation into the
  existing reproject pass, which already evaluates the projected
  position and normal, at zero extra dispatches.

**Grid safety.** `r_eff_j ≤ r = cell_size` always (we only shrink,
never grow), so the existing 27-cell neighborhood search always covers
the full interaction range. No grid changes.

**First-frame transient.** None: the reproject pass writes luminance
before relax's first-ever dispatch (both run on the first playing
frame), so relax never reads the zero-initialized shading buffer.

**New compile-time WGSL constant** (defined in the relax shader,
following decision 8's pattern — literal value in WGSL, documented
here for reference):

| Constant | Default | Description |
|---|---|---|
| `MIN_RADIUS_SCALE` | `0.5` | Minimum effective radius scale at `lum=0`. Dark regions pack up to `(1/0.5)² = 4×` denser than lit regions. Tune visually. |

**Relax shader bindings after Step 1d:**

| Binding | Type | Description |
|---|---|---|
| 0 | uniform `RelaxParams` | `dt, radius, time, point_count` (16 B) |
| 1 | storage, read `points_in` | `array<Point>` — the read-side point buffer |
| 2 | storage, read_write `points_out` | `array<Point>` — the write-side point buffer |
| 3 | storage, read `normals_in` | `array<vec4f>` — shared normals buffer |
| 4 | uniform `GridParams` | Grid params (bbox, dims, cell size) |
| 5 | storage, read `sorted_values` | `array<u32>` |
| 6 | storage, read `cell_start` | `array<u32>` |
| 7 | storage, read `cell_count` | `array<u32>` |
| 8 | storage, read `shading_in` | `array<PointShading>` — shared shading buffer (reads the `luminance` field per neighbor) — **NEW** |

**Verify:** `npm run dev`. Points should visibly cluster in shadowed
regions (surfaces facing away from the light, and in cast shadows) and
spread out in lit regions. Rotate the camera — the clustering should be
**view-independent** (it tracks the light, not the camera). Pause time
and orbit: clusters stay fixed to the surface, not the view. Hatch lines
(from Steps 1/1b) should appear denser in dark regions. If points
collapse into tight blobs in deep shadow, raise `MIN_RADIUS_SCALE`
(toward 1.0 for subtler effect). If the effect is too weak, lower it
(toward 0.3 for ~11× density).

### Step 2 — Curve subdivision + projected line-list render

**Status:** completed.

**Goal:** make lines follow the surface curvature by subdividing each
link into `N_SEGMENTS` surface-projected segments.

**Changes:**
- `link-shader.ts`: add the curve constants (`N_SEGMENTS`,
  `LINK_NEWTON_ITERS`, `LINK_NEWTON_ALPHA` — literal values in WGSL,
  see decision 8) and a `curves_out: array<vec4f>` binding (binding 9)
  statically used only by the third dispatch, which is renamed
  `link_dedup_cs` → `link_dedup_subdiv_cs`. The dispatch evaluates the
  dedup predicate as before (owner election + fused mutual-break) and
  keeps `links[i]` in sync; for surviving links it subdivides
  `lerp(p_i, p_j, t)` for `t ∈ {0, 1/N, 2/N, ..., 1}` and
  Newton-projects each sample onto the surface via `projectToSurface`,
  writing `curves_out[i * STRIDE + k] = vec4f(projected_pos, 1.0)`;
  dropped or absent links zero their slice — every slice is rewritten
  every frame (the buffer persists across frames and keeps rendering
  while paused). The scan dispatch is unchanged.
- `link-pipeline.ts`: keep `linksBuffer` (now internal, private) as the
  dedup machinery's working state and add the `linkCurves` buffer
  (`point_count * (N_SEGMENTS + 1) * 16` B, zero-initialized — `.w =
  0.0` renders nothing before the link pass has ever dispatched). The
  dedup+subdiv dispatch now reads the post-relax point buffer (for
  `p_i`/`p_j`), so its bind group becomes ping-pong-dependent: A/B
  variants selected by `readFromA`. Export `linkCurves` (plus a TS
  mirror of `N_SEGMENTS` for the render instance count; WGSL remains
  the source of truth — decision 8) for the render pipeline.
- `link-render-shader.ts`: read `linkCurves` instead of
  `points + links` (bind group shrinks to uniform + curve buffer; the
  `N_SEGMENTS`/`STRIDE` consts are mirrored in this module). Each
  instance `inst` maps to point `i = inst / N_SEGMENTS`, segment
  `s = inst % N_SEGMENTS`. Endpoint 0 = `linkCurves[i * STRIDE + s]`,
  endpoint 1 = `linkCurves[i * STRIDE + s + 1]`. If either `.w == 0.0`,
  push offscreen. `draw(2, pointCount * N_SEGMENTS)`.
- `link-render-pipeline.ts`: constructor takes the `linkCurves` buffer
  (no ping-pong bind groups — the curve data is shared); the Step 1
  bindings (`pointsA` / `pointsB` / `links` / `shading`) are dropped
  (Step 2 visibility uses the `.w` flag only; the `shadingBuffer`-based
  visibility from Step 1 is dropped — per-segment occlusion arrives in
  Step 4, so back-side lines are temporarily visible through the
  surface). `render(pass, eye, target, canvas, pointCount)` — the
  look-at `target` must stay (the view projection needs it); only the
  `readFromBufferA` argument goes away.
- `main.ts`: construct the render pipeline with `link.linkCurves` and
  drop the `readFromBufferA` argument from the render call. The link
  compute call site is unchanged.

**Verify:** `npm run lint` (TS files touched), then `npm run dev`.
Lines should now curve along the surface instead of cutting through it.
On highly curved regions the difference should be clearly visible.
Lines that previously penetrated the surface now hug it. No fans/stars
or mutual-link pairs reappear (dedup still effective — check with
`DRAW_POINTS = false`). Back-side lines whose stipple endpoints are
both occluded are culled again by the Step 2b interim endpoint check;
mid-segment and single-endpoint occlusion remain until Step 4.

**Superseded (Step 2c).** The chord subdivision (`lerp` + project) and
the in-place clearing of losing `links` entries are replaced by C1
Hermite sampling with chain-window tangents — see Step 2c and
decision 11. The `N_SEGMENTS` TS mirror becomes the interpolated,
single-source `LINK_N_SEGMENTS` constant (decision 8's TS-owned rule). The
buffer layout, render pipeline, and instance mapping are otherwise
unchanged.

### Step 2b — Interim endpoint-visibility cull

**Status:** completed.

**Goal:** restore back-face/occlusion culling for the lines between
Step 2 and Step 4. Step 2 dropped the Step 1 shading-based visibility
check together with the render's point/links/shading bindings, leaving
the view-independent `.w` validity flag as the only cull — so
back-facing and otherwise occluded lines rendered through the shape.

**Mechanism.** The link render re-binds the internal `links` buffer and
the shared `shadingBuffer` and applies the Step 1 coarse check in the
vertex shader: a sub-segment is pushed offscreen when the curve slice
is invalid (`.w == 0.0` on either endpoint — checked first, so the
`links[i]` read below only happens for surviving links and `j` is
always a valid point index) or when **both** endpoint stipple points
have non-positive center-ray clearance
(`shading[i].clearance <= 0.0 && shading[links[i]].clearance <= 0.0`).
All `N_SEGMENTS` sub-segments of a link cull or draw together (they
share `i` and `links[i]`) — the same granularity as Step 1. The check
stays render-side and reads the shading buffer, which the shading pass
refreshes every frame — so culling remains correct while time is
paused and the camera orbits.

**Changes:**
- `link-render-shader.ts`: re-add `SDF_COMMON` (for `PointShading`);
  bindings 2 (`links`, read) and 3 (`shading`, read); coarse cull in
  the vertex shader after the validity check.
- `link-render-pipeline.ts`: constructor takes `links` and `shading`
  buffers; bind group grows to four entries (uniform, curves, links,
  shading). Still a single bind group — nothing ping-ponged.
- `link-pipeline.ts`: `linksBuffer` becomes public readonly again
  (internal working state, exposed for the render's interim cull).
- `main.ts`: passes `link.linksBuffer` and `points.shadingBuffer` to
  the render pipeline constructor.

**Not fixed until Step 4:** segments with exactly one occluded endpoint
still draw, and mid-segment occlusion (a curve hidden behind a bulge
while both stipple endpoints face the camera) still draws. Step 4's
per-curve-point `rayClearance` traces replace this check and drop the
`links`/`shading` render bindings again.

**Verify:** `npm run lint` (TS files touched), then `npm run dev`.
Back-side lines disappear again as in Step 1. Rotate the camera while
time is paused — culling updates live. Step 2's surface-following
behavior unchanged.

### Step 2c — C1 Hermite subdivision with chain-window tangents

**Status:** completed.

**Goal:** fix the joint kinks. Step 2's curves follow the surface, but
the rendered polyline is angular — consecutive chords meet at sharp
turning angles, and at each joint the incoming and outgoing tangents
are unrelated because the two links were chosen independently. The fix
replaces the chord subdivision with cubic Hermite segments whose
endpoint tangents are averaged over each endpoint vertex's chain
window, making the chain of curve segments C1 (decision 11).

**Mechanism.** The subdiv dispatch computes each endpoint tangent from
the endpoint vertex's three-point stipple window — the predecessor via
`owners`, the successor via the raw `links` slot of the target, both
validated against the dedup rules with one-sided chord-tangent
fallbacks — and samples the resulting Hermite segment, Newton-projecting
the interior samples (the endpoints are the stipple points, written
exactly, so adjacent segments join seamlessly). Because the tangent at
a shared vertex is a pure function of that vertex's window, adjacent
segments compute identical tangent vectors → C1 with no cross-link
write, extra dispatch, or extra buffer. `links` is no longer cleared
in place — it stays the raw scan output, which is what makes the
chain-window read `links[links[i]]` race-free; the survival predicate
moves to the consumers (the dispatch still evaluates it to gate curve
emission; the render's interim cull reads `links[i]` only for
surviving links, where the raw value equals the post-dedup value).
See decision 11 for the full specification.

**Changes:**
- `shaders/link-shader.ts`: the subdiv dispatch drops the `links`
  write-back; adds the chain-window fetches, the window-pure tangent
  helpers, and Hermite sampling + `projectToSurface` for interior
  samples. New WGSL constant `LINK_TANGENT_SCALE`.
  The `links_out` read/write binding gains a read-only alias
  (`links_in`, same buffer, binding 7) for the subdiv entry, mirroring
  the `owners_in` pattern. `N_SEGMENTS` is interpolated from the TS
  constant `LINK_N_SEGMENTS`, defined and exported here (decision 8's
  TS-owned rule).
- `shaders/link-render-shader.ts`: `N_SEGMENTS` is interpolated from
  the same `LINK_N_SEGMENTS` constant (removing the manual mirror).
- `link-pipeline.ts` / `link-render-pipeline.ts`: `LINK_N_SEGMENTS` is
  imported from the shader layer (via the `shaders` barrel) instead of
  defined in `link-pipeline.ts`; no functional change (buffer sizes,
  bind groups, dispatch order unchanged).
- `link-render-shader.ts` / `link-render-pipeline.ts`: otherwise
  unchanged — the interim cull reads `links[i]` only after the `.w`
  validity check, i.e. only for surviving links, whose raw `links`
  value equals the post-dedup value.
- `docs/hatch-lines-spec.md`: decisions 2, 8, 9 updated; decision 11
  added.

**Verify:** `npm run lint` (TS files touched), then `npm run dev`.
Joints between consecutive links are tangent-continuous — no more
kinks at stipple points; the polyline reads as smooth strokes threading
the stipple field. Curvature can still change abruptly at joints (C1,
not C2) — acceptable. Chain ends terminate naturally at stipple
points. If joints look over- or under-rounded, tune `LINK_TANGENT_SCALE`
in the WGSL (higher = rounder). If the curve still reads faceted
between joints, raise `LINK_N_SEGMENTS` (shape does not change — only
tessellation fidelity; buffer, compute, and render stay in sync by
construction). No fans/stars or mutual-link pairs reappear (dedup still
effective — check with `DRAW_POINTS = false`).

### Step 3 — Screen-space stroke-quad rendering with soft AA

**Status:** completed.

**Goal:** replace the crude `line-list` with quads for artistic control
over line width and anti-aliasing.

**Changes:**
- `link-render-shader.ts`: switch from `line-list` to `triangle-list`
  with `draw(6, pointCount * N_SEGMENTS)` (6 vertices = 2 triangles per
  quad, instanced per sub-segment). The stroke geometry is built
  entirely in 2D pixel coordinates (stroke space): the vertex shader
  perspective-projects the two endpoint curve points to pixel
  coordinates, converts the world-space half-width `LINK_HALF_WIDTH`
  to a per-endpoint pixel stroke radius via the inverse projective
  depth (`px_scale` measured along a view-perpendicular direction —
  see decision 6), and extrudes the quad as the bounding box of the
  2D capsule around the projected segment
  (`quad_half = max(r0_px, r1_px) + AA_PIXELS`, extended along the
  projected axis so the rounded caps fit — see Step 5). The quad is
  output with the segment center's constant clip `w` and depth, so
  interpolation is exactly linear in screen space. The fragment
  shader evaluates the 2D segment distance field in stroke pixels
  against the depth-tapered radius, AA'd via `fwidth(sdist) *
  AA_WIDTH` (same pattern as the point shader's disc SDF, ~1px edge,
  zoom- and angle-independent). Four guards in the vertex shader:
  occlusion cull (`max(e0.w, e1.w) <= 0.0` — Step 4), behind-camera
  (`clip.w <= 0` on either endpoint), eye-on-segment
  (`dist_eye < EYE_EPS`), and collapsed projection
  (`DEGENERATE_PX2` — fixed fallback axis, radii flattened to the
  larger one, disc-shaped distance field). `LINK_ALPHA = 1.0` (opaque
  interior; only the AA edge blends). New WGSL-only constants
  (`LINK_HALF_WIDTH`, `LINK_COLOR`, `LINK_ALPHA`, `AA_WIDTH`,
  `AA_PIXELS`, `DEGENERATE_PX2`, `EYE_EPS`) — see decision 8.
- `link-render-pipeline.ts`: topology `triangle-list`; blend state
  `src-alpha` / `one-minus-src-alpha` (same as point renderer). Uniform
  grows 80 → 96 B to add `vec2f canvas_size` (NDC ↔ pixel coordinate
  conversion and the pixel-scale measurement); `writeUniform` writes
  `canvas.width` / `canvas.height`. No ping-pong bind groups needed
  (curve buffer is shared).

**Historical note.** Step 3 originally rendered **world-space
camera-facing quads** — billboarded around the segment axis with a
world-space capsule SDF and a quad oversized by a measured
world-units-per-pixel factor (`wpp`). It was replaced by the
screen-space approach (see decision 6): the world-space capsule's
along-axis reach past a shared endpoint foreshortens with the view
angle, starving the joint overlaps near grazing angles, and its
view-compressed distance field blew the `fwidth`-derived AA band up by
the same factor, fading the caps out.

**Verify:** Lines read with the pixel width their 3D tube would
project — shrinking with distance, collapsing toward the rounded cap
as the segment turns parallel to the view — with ~1px zoom-independent
AA edges. At extreme zoom-out the stroke settles to a stable
`~AA_PIXELS`-pixel minimum width (no sub-pixel flicker). Rotating
through all camera orientations produces no freeze and no garbage
geometry; near-parallel sub-segments collapse gracefully to
overlapping discs, so chains stay continuous at grazing angles. Lines
still follow the surface curvature (from Step 2).

### Step 4 — Per-segment occlusion

**Goal:** cull or fade line segments that are occluded by the surface
between the segment and the camera, so lines on the back of the shape
are not visible through it.

**Changes:**
- `link-shader.ts`: after computing each curve point `p_k`, compute the
  local normal `n_k = normalize(sdfGradient(p_k, time))`, bias by
  `SHADOW_BIAS`, trace `rayClearance(p_k + n_k * SHADOW_BIAS, eye_dir,
  dist_eye, threshold_k, time)` toward the eye, and store the clearance
  in the `.w` component of each curve point (replacing the binary
  validity flag). The per-curve-point threshold `threshold_k` is the
  local curve spacing — the longest adjacent sub-segment length, floored
  at `PROBE_STEP` — so the trace's origin ignore-band covers the curve's
  own surface neighborhood; the shading pass passes the splat-radius
  `CLEARANCE_THRESHOLD` for the stipple points instead.
  Invalid links (no neighbor found) still get `.w = 0.0` — a sentinel
  that is both "invalid" and "fully occluded", which the render shader
  treats the same way (push offscreen). Valid but occluded segments get
  `.w < 0.0`; visible segments get `.w > 0.0`.
  - **Param change:** the link params uniform grows to include `eye:
    vec3f` (needed for the clearance trace). Layout: `eye: vec3f, time:
    f32, point_count: u32, distance_weight: f32, link_radius: f32,
    _pad: u32` (32 B). `dispatch` gains an `eye` parameter.
  - **Cost:** `N_SEGMENTS + 1` `rayClearance` traces per point per frame.
    For `N_SEGMENTS = 4`, 6K points: ~30K traces — about 5× the shading
    pass's per-point visibility traces (one per point since Step 1c).
    Acceptable.
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

**Status:** completed. Solved for free by the Step 3 capsule SDF —
no additional code beyond what Step 3 already shipped.

**Goal:** eliminate hairline gaps at segment joints where two
sub-segments of the same curve meet.

**Approach (supersedes the overlap / miter-bisector plan):** the
Step 3 line shape is a capsule (tapered body with semicircular rounded
caps — decision 6), and the quad extends along the projected segment
axis by `quad_half` (in stroke pixels) beyond each endpoint to fit
those caps. Because adjacent sub-segments share endpoints exactly (the
Hermite subdivision writes identical endpoints — decision 11), the
rounded caps of neighboring segments overlap at each joint. With
`LINK_ALPHA = 1.0` (opaque interior, only the ~1px AA edge blends),
the overlap is invisible — opaque over opaque produces no darkening
and no doubled alpha — and the union of the two capsules renders as a
**rounded miter / bent-sausage join** with no hairline gaps:

- Collinear segments: the two opposite-half-disc caps form a full
  disc → seamless.
- Bent segments: the union is continuous; the outside of the bend is
  rounded (the natural rounded-miter look); the inside may have a
  slight concavity but no gap.
- The ~1px AA edges sit at each segment's own capsule boundary,
  inside the neighbor's opaque interior, so they do not overlap-blend
  at joints.
- Chain ends terminate with a rounded cap too — no special handling.

No miter geometry, no SDF union of two segments, no extra varyings,
no extra dispatch. The earlier "extend each quad by `LINK_HALF_WIDTH`
along the axis" and "miter joint bisector" alternatives are
superseded: Step 3's axis extension (by `quad_half`, not
`LINK_HALF_WIDTH`) already provides the overlap, and the opacity
makes it seamless. Screen-space rendering strengthens this further:
each cap extends the stroke radius past the shared endpoint in
*every* screen direction (decision 6), so the overlap survives grazing
angles where a world-space capsule's along-axis reach would
foreshorten away.

**Verify:** Look closely at segment joints (zoom in). No gaps, no
dark seams, no doubled alpha. Rounded miters at bends; rounded caps
at chain ends. Correct from multiple viewing angles. Toggle
`DRAW_POINTS = false` in `main.ts` to inspect the lines alone.

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
| 1b | No fans/stars at popular targets: every point has at most one incoming and one outgoing edge, so links form chains. No mutual-link pairs (both segments of an i↔j pair render) — the dedup dispatch's fused mutual-break drops the larger-index edge. Segment count drops (edges = distinct targets) but lines stay dense enough. |
| 1c | Points render as fixed-size dots irrespective of luminance or shadow. `DRAW_POINTS = false` in `main.ts` hides them; links are unaffected. Dots behind the surface are still culled via the center-ray clearance. Shading pass does one `rayClearance` trace per point. |
| 1d | Points visibly cluster in shadowed regions and spread out in lit regions. Clustering is view-independent (tracks the light, not the camera — pause time and orbit to confirm). Hatch lines appear denser in dark regions. No point collapse in deep shadow. |
| 2 | Lines curve along the surface in high-curvature regions instead of cutting through. No visible penetration of the SDF surface. No fans/stars or mutual pairs reappear (dedup still effective). |
| 2b | Back-side lines disappear again as in Step 1 (both endpoints occluded → culled). Culling updates live while paused and orbiting. Segments with one occluded endpoint or mid-segment occlusion still draw until Step 4. |
| 2c | Joint kinks are gone — consecutive curve segments meet with matching tangents. Chain ends terminate naturally at stipple points. `LINK_TANGENT_SCALE` visibly controls joint rounding; `LINK_N_SEGMENTS` changes tessellation fidelity but not shape (buffer, compute, and render stay in sync by construction). |
| 3 | Lines read with the pixel width their 3D tube would project — shrinking with distance, collapsing toward the rounded cap as the segment turns parallel to the view — with ~1px zoom-independent AA edges. At extreme zoom-out the stroke settles to a stable `~AA_PIXELS`-pixel minimum width (no sub-pixel flicker). Rotating through all camera orientations produces no freeze, no garbage geometry, and no gaps at grazing-angle joints (near-parallel sub-segments collapse to overlapping discs). Capsule shape with rounded caps visible at chain ends. |
| 4 | Lines on the far side of the shape are culled. Soft fade near silhouette edges. No lines visible through the surface. |
| 5 | No gaps at segment joints. No doubled-alpha seams (opaque interior, `LINK_ALPHA = 1`). Rounded miters at bends; rounded caps at chain ends. Correct from multiple viewing angles. |
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
- **Best-continuation link election** — replace the smallest-source-index
  owner election with a packed `(quantized continuation-alignment score,
  source index)` `atomicMin` key, so the suitor whose edge best continues
  the line through the target wins (smoother polylines through kinks
  than an arbitrary index tie-break).
- **Runtime-tunable parameters** — promote `DISTANCE_WEIGHT`,
  `LINK_HALF_WIDTH`, `LINK_COLOR`, `LINK_TANGENT_SCALE`, `AA_PIXELS`
  to uniform buffer fields (or a GUI) for live experimentation
  (`N_SEGMENTS` is structural — decision 8 — but could join as a
  uniform if the curve buffer is sized for its maximum).
- **Variable segment count** — allocate a fixed maximum segment count
  per link but use fewer segments for short links (scaled by distance).
  Wastes some buffer space but avoids over-tessellating short links.
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
