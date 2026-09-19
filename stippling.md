# Screen-space stippling of a ray-marched SDF scene

Design document for the stippling pipeline that replaces the former n-body
simulation. This document captures the agreed plan so future work can pick
up from it without re-deriving the design.

## Goal

Render a ray-marched SDF scene, derive a screen-space density field from
the resulting image, and iteratively redistribute a set of 2D points so
that their density follows the scene's density — producing a stippled
rendering. Initial version renders, for debugging, the density texture
as grayscale with red billboard quads overlaid at point positions.

## Pipeline overview

Each frame runs four passes, in order:

1. **Density pass** (compute, writes to storage texture) — one workgroup
   (8×8) per 64 pixels; each invocation ray-marches the SDF scene using
   the orbit camera's eye position and frustum-corner rays. Writes one
   `f32` per canvas pixel into an `r32float` storage texture:
   `-1.0` on ray miss (background), `1 - Lambert` (range `[0, 1]`) on hit.
   A compute shader (not a fragment render pass) is used because
   `r32float` is not a renderable color format in WebGPU (see below).
2. **Relax pass** (compute, two dispatches) — redistributes the 2D points
   so their density matches the density texture, using an
   attraction/repulsion model (see below).
3. **Debug render pass** (render to canvas) — blits the density texture
   as grayscale (`max(d, 0)`, so background maps to black).
4. **Point render pass** (same render pass as #3, second draw) — draws
   red billboard quads at point positions in screen UV space, additive
   blended, soft circular discs.

A fifth, one-shot pass runs at bootstrap and on canvas resize:

0. **Seed pass** (compute) — GPU-side rejection sampling of the initial
   point distribution, using the density texture as the acceptance
   probability. Runs after the first density render of the frame.

## Density encoding

- Texture format: `r32float`, **always nearest sampling**.
  - No `Float32Filterable` feature requirement.
  - No interpolation edge cases at the figure boundary.
- Encoding:
  - `-1.0` — ray missed the scene (background).
  - `[0.0, 1.0]` — ray hit; value is `1 - Lambert` luminance under a
    fixed light direction `normalize(vec3(0.5, 0.8, 0.6))`.
- A density of `0` on a hit pixel (fully lit surface) is cleanly
  distinguishable from background's `-1` — no epsilon floor hack needed.
- If `createTexture` with `r32float` fails, the error is surfaced rather
  than silently downgraded.

### Why a compute shader, not a fragment render pass

`r32float` is **not a renderable color-attachment format** in the WebGPU
core spec (only `rgba8*`, `bgra8*`, `rgba16float`, `r10g10b10a2*`, and
`rgb10a2*` are guaranteed renderable). To still obtain a true 32-bit
float density texture, the density pass runs as a **compute shader**
that writes to an `r32float` **storage texture**
(`texture_storage_2d<r32float, write>`). Storage textures support
`r32float` in the core spec — no feature flags required.

The same `GPUTexture` (created with both `STORAGE_BINDING` and
`TEXTURE_BINDING` usage) is later bound as `texture_2d<f32>` (nearest)
by the debug blit, seed, and relax pipelines. Writing as a storage
texture and reading as a sampled texture within the same command encoder
is explicitly supported by WebGPU — passes are ordered, so the write
completes before any read.

## SDF scene

Initial scene is a CSG union (smooth-min blend) of 2-3 primitives — e.g.
a sphere + torus + box — chosen so the silhouette has clear inside/outside
regions and interesting concavities for stipple density. Authoring lives
entirely in WGSL inside `src/stipple/shaders.ts`.

## Initial point distribution — GPU rejection sampling

No CPU readback. A compute shader maps one invocation per point index `i`
and performs rejection sampling against the density texture:

```
const N_ATTEMPTS = 32;
base = i * N_ATTEMPTS * 3;            // integer stride per point
accepted = false;
temp = vec2(0.5);                      // fallback, overwritten below
last_hit = vec2(0.5);                  // last d ≥ 0 point, if any
last_hit_valid = false;

for (a in 0..N_ATTEMPTS) {
    s = base + a * 3;
    x = pcg_to_float(s + 0);
    y = pcg_to_float(s + 1);
    r = pcg_to_float(s + 2);           // uniform ∈ [0, 1]
    d = textureSample(density, nearest, vec2(x, y)).r;

    if (a == 0) { temp = vec2(x, y); } // first attempt, always (even if bg)

    if (d >= 0.0) {
        last_hit = vec2(x, y);
        last_hit_valid = true;
        if (r < d) {                   // accept with probability d
            write(vec2(x, y));
            accepted = true;
            break;
        }
    }
}

if (!accepted) {
    write(last_hit_valid ? last_hit : temp);
}
```

- PCG hash on `uint -> uint`, divided by `2^32` for the float. State is
  fully determined by `base` — no per-invocation state carried across
  frames, so reseeding is just re-dispatching.
- Background (`d = -1`) is never accepted (`r < -1` is impossible) and
  never updates `last_hit`.
- `temp` (= first attempted point, even if background) is the ultimate
  fallback so the shader always writes something random — relaxation
  pulls stragglers inside.
- Because points are accepted with probability `d`, the initial
  distribution already follows the density, giving relaxation a head
  start.

## Relaxation force model

Points live in `[0, 1]²` UV space. Each frame, two compute dispatches:

1. **Sample-densities dispatch** (workgroup 64): each invocation `i`
   samples `d_i = density(p_i)` once from the density texture and writes
   into a small `densities: array<f32>` storage buffer (16 KB for 4K
   points).
2. **Relax dispatch**: each invocation `i` reads `p_i, d_i` and loops
   over all `j ≠ i` reading `p_j, d_j` from storage buffers — pure
   buffer reads, no texture sampling in the hot loop. O(n²) ALU, same
   shape as the former n-body compute.

For particle `i` at `p_i` with sampled `d_i`, neighbor `j` at `p_j`
with `d_j`, let `r = p_j - p_i`, `r̂ = r / (|r| + ε)`:

- Both inside (`d_i ≥ 0, d_j ≥ 0`):
  repel `F += k_rep · (1 - α · d_i · d_j) / (r² + ε²) · r̂`.
  Weaker repulsion where density is high (dark) → tighter packing there.
- `i` inside, `j` outside (`d_j < 0`):
  mild push `F += k_push / (r² + ε²) · r̂`.
- `i` outside (`d_i < 0`), `j` inside (`d_j ≥ 0`):
  attract `F += k_att / (r² + ε²) · (-r̂)`.
- Both outside: no force (let them drift; insiders pull them in).

Integrate with semi-implicit Euler and per-frame velocity damping:

```
v ← v + F · dt
p ← p + v · dt
v *= damping
p = clamp(p, [0, 1]²)
```

Parameters (exposed via a uniform buffer, tunable at runtime later):

| name     | default | meaning                                              |
| -------- | ------- | ---------------------------------------------------- |
| `dt`     | 0.005   | time step per frame                                  |
| `k_rep`  | TBD     | repulsion strength between two inside particles      |
| `k_att`  | TBD     | attraction strength, outside → inside                |
| `k_push` | TBD     | mild push, inside particle away from outside         |
| `α`      | TBD     | density-coupling factor for repulsion weakening      |
| `ε`      | TBD     | Plummer-style softening, prevents singularities      |
| `damping`| 0.9     | per-frame velocity damping                           |

Light direction (fixed in shader): `normalize(vec3(0.5, 0.8, 0.6))`.

Point count for v1: **4096**.

## Module structure

```
src/
  main.ts                          # orchestrates the passes per frame
  webgpu.ts                        # unchanged
  orbit-controls.ts                # unchanged
  mat4.ts                          # unchanged (builds ray-gen frustum uniforms)
  stipple/
    shaders.ts                     # all WGSL: SDF scene, ray-march FS,
                                   # sample-density CS, seed CS, relax CS,
                                   # debug-blit FS, point VS/FS
    point-buffers.ts               # 2D point layout (vec2 pos + vec2 vel),
                                   # ping-pong storage, densities aux buffer.
                                   # No readback.
    density-pipeline.ts            # compute pass: ray-march SDF → r32float
                                   # storage texture. Recreates texture on
                                   # resize.
    seed-pipeline.ts               # one-shot compute: GPU rejection sampling
                                   # → point buffer. Runs at bootstrap + resize.
    relax-pipeline.ts              # two-dispatch compute: sample-densities +
                                   # relax, ping-pong. Runs every frame.
    debug-render-pipeline.ts       # render-to-canvas: blit density as grayscale
    point-render-pipeline.ts       # render-to-canvas: red billboard quads at
                                   # point UVs
```

Each pipeline owns its shader module, pipeline object, bind groups, and
any uniform buffers — same ownership convention as the former
`nbody/*-pipeline.ts` files. `main.ts` stays thin: sync size → (reseed
if needed) → density pass → relax pass → debug render → point render →
submit.

## Build order

Each step is independently verifiable in the browser before moving on.

1. Delete `src/nbody/`, scaffold `src/stipple/shaders.ts` with the SDF
   scene + ray-march fragment shader. Get the density pass rendering
   directly to the canvas (no points yet) to verify the SDF.
2. Add `point-buffers.ts` + `seed-pipeline.ts` (GPU rejection sampling) +
   `point-render-pipeline.ts`; render 4K red dots over the density
   texture (no relaxation yet — dots stay where the seed put them).
3. Add `relax-pipeline.ts` with the two-dispatch sample-densities + relax
   flow; wire into the frame loop.
4. Tune force parameters; split debug render from point render if
   needed.
5. Update `AGENTS.md` structural overview; run `npm run lint`.

## Notes and open items

- If convergence of the attraction/repulsion model is poor, weighted
  Lloyd relaxation (Secord 2002) can be swapped in behind the same
  `RelaxPipeline` interface — the rest of the pipeline wouldn't change.
- The density texture is recreated on canvas resize. Because the density
  field changes shape on resize, points are reseeded at that point too
  (the seed pipeline is re-dispatched).
- The seed shader must run after a density render, since it samples the
  density texture. At bootstrap and on resize, `main.ts` forces a
  density render before dispatching the seed shader.
