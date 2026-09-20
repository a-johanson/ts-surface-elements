/**
 * WGSL shader source strings for the surface-stippling pipeline.
 *
 * Points live in 3D world space on the surface of the SDF scene. The
 * shared {@link SDF_COMMON} block — interpolated into every shader that
 * needs the SDF — owns the scene description (`map`), gradient/normal
 * helpers, and a Newton-Raphson surface projection routine.
 */

/**
 * Shared WGSL block: SDF scene, gradient, normal, and surface projection.
 *
 * Interpolated into the seed, relax, shading, debug-render, and
 * point-render shaders. Contains only functions, structs, and
 * compile-time constants — no bindings.
 *
 * - `Point` — the per-point storage layout (pos only, 16 bytes). Surface
 *   normals live in a separate shared buffer, written by the seed pass
 *   (once, at bootstrap) and by the shading pass (every frame, after
 *   relax).
 * - `map(p)` — the scene signed distance field.
 * - `sdfGradient(p)` — central-difference gradient (6 taps); used by
 *   Newton projection so the magnitude is the true distance-field
 *   gradient (tetrahedron normals only give a direction).
 * - `calcNormal(p)` — normalized gradient; used by the debug render for
 *   Lambert shading.
 * - `projectToSurface(p, iters, alpha)` — Newton-Raphson steps
 *   `p -= alpha * (f(p) / |∇f|²) * ∇f` until `|f(p)| < SURF_EPS` or the
 *   iteration budget is exhausted. For a true distance field (`|∇f| ≈ 1`)
 *   and `alpha = 1`, each step lands on the surface in one go; smaller
 *   `alpha` guards against overshoot in `smin` blend regions where the
 *   field is not a perfect distance function.
 * - `rayMarch(ro, rd)` — sphere-traces `map()` along `rd` from `ro` up to
 *   `MAX_DIST` with `MAX_STEPS` iterations. Returns the hit distance, or
 *   `-1.0` on miss. Shared by the debug render (per-fragment SDF
 *   visualization) and the shading pass (per-point visibility and shadow
 *   tests).
 */
export const SDF_COMMON = /* wgsl */ `
struct Point {
    pos: vec4f,
};

const MAX_DIST: f32 = 50.0;
const SURF_EPS: f32 = 0.001;
const MAX_STEPS: i32 = 96;

fn rayMarch(ro: vec3f, rd: vec3f) -> f32 {
    var t = 0.0;
    for (var i: i32 = 0; i < MAX_STEPS; i = i + 1) {
        let p = ro + rd * t;
        let d = map(p);
        if (d < SURF_EPS) {
            return t;
        }
        t = t + d;
        if (t > MAX_DIST) {
            break;
        }
    }
    return -1.0;
}

fn sdSphere(p: vec3f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdTorus(p: vec3f, t: vec2f) -> f32 {
    let q = vec2f(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

fn sdBox(p: vec3f, b: vec3f) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn map(p: vec3f) -> f32 {
    let d1 = sdSphere(p - vec3f(-1.2, 0.0, 0.0), 1.5);
    let d2 = sdTorus(p - vec3f(1.2, 0.0, 0.0), vec2f(1.0, 0.35));
    let d3 = sdBox(p - vec3f(0.0, 1.6, 0.0), vec3f(1.0, 0.4, 1.0));
    let d12 = smin(d1, d2, 0.6);
    return smin(d12, d3, 0.6);
}

fn sdfGradient(p: vec3f) -> vec3f {
    let e = 0.0005;
    return vec3f(
        map(p + vec3f(e, 0.0, 0.0)) - map(p + vec3f(-e, 0.0, 0.0)),
        map(p + vec3f(0.0, e, 0.0)) - map(p + vec3f(0.0, -e, 0.0)),
        map(p + vec3f(0.0, 0.0, e)) - map(p + vec3f(0.0, 0.0, -e)),
    ) / (2.0 * e);
}

fn calcNormal(p: vec3f) -> vec3f {
    return normalize(sdfGradient(p));
}

fn projectToSurface(p_in: vec3f, iters: i32, alpha: f32) -> vec3f {
    var p = p_in;
    for (var i: i32 = 0; i < iters; i = i + 1) {
        let f = map(p);
        if (abs(f) < SURF_EPS) {
            break;
        }
        let g = sdfGradient(p);
        let gl2 = dot(g, g);
        if (gl2 < 1e-10) {
            break;
        }
        p = p - alpha * (f / gl2) * g;
    }
    return p;
}
`;

/**
 * Debug render shader — ray-marches the SDF per fragment and writes
 * grayscale Lambert shading directly to the canvas color attachment.
 *
 * Replaces the former two-stage density-compute + blit approach. With
 * surface-space points nothing consumes a density texture, so the
 * `r32float` storage texture and separate blit pipeline are gone; the
 * SDF is visualized by a single full-screen fragment shader.
 *
 * Background (ray miss) maps to black; hit fragments map to
 * `[0, 1]` gray via `1 - Lambert`.
 */
export const DEBUG_RENDER_SHADER = /* wgsl */ `
${SDF_COMMON}

struct CameraRays {
    eye: vec3f,
    half_width: f32,
    forward: vec3f,
    half_height: f32,
    right: vec3f,
    _pad0: f32,
    up: vec3f,
    _pad1: f32,
    resolution: vec2u,
    _pad2: vec2u,
};

@group(0) @binding(0) var<uniform> camera: CameraRays;

@vertex
fn blit_vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    let pos = array<vec2f, 6>(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0),
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0),
    );
    return vec4f(pos[vid], 0.0, 1.0);
}

@fragment
fn blit_fs(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let res = camera.resolution;
    let uv = frag_coord.xy / vec2f(res);
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    let rd = normalize(
        camera.forward
        + ndc.x * camera.half_width * camera.right
        + ndc.y * camera.half_height * camera.up
    );

    let t = rayMarch(camera.eye, rd);
    if (t < 0.0) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    let p = camera.eye + rd * t;
    let n = calcNormal(p);
    let light_dir = normalize(vec3f(0.5, 0.8, 0.6));
    let lambert = max(dot(n, light_dir), 0.0);
    return vec4f(lambert, lambert, lambert, 1.0);
}
`;

/**
 * Seed compute shader — 3D rejection sampling in a bounding box.
 *
 * One invocation per point. Generates pseudo-random candidates uniformly
 * in `[bbox_min, bbox_max]³` via a PCG hash and accepts the first whose
 * `|map(p)| < band`. Accepted (and fallback) points are projected onto
 * the surface with Newton-Raphson before being written.
 *
 * Fallback: if no candidate lands inside the band, the candidate with the
 * smallest `|map(p)|` is chosen (tracked via a running minimum) and
 * projected — relaxation will pull stragglers into place.
 *
 * `N_ATTEMPTS` is a compile-time knob (paired with `MAX_STEPS` /
 * `SURF_EPS`); the bbox and band are uniform-side so the scene can be
 * re-tuned without recompiling.
 *
 * In addition to positions, the seed pass writes the matching surface
 * normal for each point into the shared `normals_out` buffer. The normals
 * buffer is shared (not ping-ponged) and must match whichever point buffer
 * relax reads; since seed writes bufferA, it also seeds the normals from
 * bufferA. Subsequent frames have their normals refreshed by the shading
 * pass, which runs after relax and writes normals from the buffer relax
 * just produced — so the invariant "normals match the buffer relax reads"
 * is preserved across the ping-pong swap.
 */
export const SEED_SHADER = /* wgsl */ `
${SDF_COMMON}

struct SeedParams {
    bbox_min: vec3f,
    band: f32,
    bbox_max: vec3f,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: SeedParams;
@group(0) @binding(1) var<storage, read_write> points_out: array<Point>;
@group(0) @binding(2) var<storage, read_write> normals_out: array<vec4f>;

const N_ATTEMPTS: u32 = 256u;
const SEED_NEWTON_ITERS: i32 = 16;
const SEED_ALPHA: f32 = 0.5;

fn pcg_hash(seed: u32) -> u32 {
    let state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcg_rand(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967295.0;
}

@compute @workgroup_size(64)
fn seed_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let base = i * N_ATTEMPTS * 3u;
    var accepted = false;
    var best_p = (params.bbox_min + params.bbox_max) * 0.5;
    var best_d = 1e9;

    for (var a: u32 = 0u; a < N_ATTEMPTS; a = a + 1u) {
        let s = base + a * 3u;
        let rx = pcg_rand(s + 0u);
        let ry = pcg_rand(s + 1u);
        let rz = pcg_rand(s + 2u);
        let p = mix(params.bbox_min, params.bbox_max, vec3f(rx, ry, rz));
        let d = abs(map(p));

        if (d < best_d) {
            best_d = d;
            best_p = p;
        }

        if (d < params.band) {
            let projected = projectToSurface(p, SEED_NEWTON_ITERS, SEED_ALPHA);
            points_out[i].pos = vec4f(projected, 0.0);
            normals_out[i] = vec4f(normalize(sdfGradient(projected)), 0.0);
            accepted = true;
            break;
        }
    }

    if (!accepted) {
        let projected = projectToSurface(best_p, SEED_NEWTON_ITERS, SEED_ALPHA);
        points_out[i].pos = vec4f(projected, 0.0);
        normals_out[i] = vec4f(normalize(sdfGradient(projected)), 0.0);
    }
}
`;

/**
 * Relax compute shader — O(n²) surface-aware, curvature-aware repulsion.
 *
 * One invocation per point `i`. For every other point `j`, a Euclidean
 * cutoff and a midpoint SDF line-of-sight check gate the pairwise force.
 * The distance used for the cutoff and the linear-decay envelope is
 * inflated by surface-curvature divergence:
 *
 *     d_infl = d_E · (1 + ½·‖n_i − n_j‖²)
 *
 * where `n_i` and `n_j` are both read from the shared normals buffer,
 * which is written by the shading pass (and seeded once at bootstrap) from
 * the same buffer relax reads — so the normals match the current
 * positions. Pairs across narrow gaps, high-curvature regions, or
 * self-folding `smin` geometry thus see an
 * effectively larger separation, suppressing cross-sheet repulsion that
 * would otherwise corrupt the Poisson-disc distribution.
 *
 * Distance usage:
 *  - cutoff `d_infl > radius`        → skip (inflated)
 *  - line-of-sight `alpha·d_E²`      → skip (Euclidean)
 *  - decay `(1 - d_infl/radius)`     → inflated
 *  - direction `(p_i - p_j) / d_E`   → Euclidean (unit)
 *
 * The accumulated force is projected onto the tangent plane at `pi` and
 * integrated with a direct Euler position step `x* = x + dt·F_tan`. The
 * new position is re-projected onto the surface with a few Newton steps;
 * since per-frame drift is small, `RELAX_NEWTON_ITERS = 4` with
 * `alpha = 1` suffices.
 */
export const RELAX_SHADER = /* wgsl */ `
${SDF_COMMON}

struct RelaxParams {
    dt: f32,
    radius: f32,
    alpha: f32,
    _pad0: f32,
    point_count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<uniform> params: RelaxParams;
@group(0) @binding(1) var<storage, read> points_in: array<Point>;
@group(0) @binding(2) var<storage, read_write> points_out: array<Point>;
@group(0) @binding(3) var<storage, read> normals_in: array<vec4f>;

const RELAX_NEWTON_ITERS: i32 = 4;
const RELAX_ALPHA: f32 = 1.0;

@compute @workgroup_size(64)
fn relax_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let p_i = points_in[i].pos.xyz;
    let n_i = normals_in[i].xyz;

    var force = vec3f(0.0);
    let r = params.radius;
    let alpha = params.alpha;

    for (var j: u32 = 0u; j < params.point_count; j = j + 1u) {
        if (j == i) {
            continue;
        }
        let p_j = points_in[j].pos.xyz;
        let diff = p_i - p_j;
        let d_E = length(diff);
        if (d_E == 0.0) {
            continue;
        }

        let n_j = normals_in[j].xyz;
        let delta_n = n_i - n_j;
        let d_infl = d_E * (1.0 + 0.5 * dot(delta_n, delta_n));

        if (d_infl > r) {
            continue;
        }

        let m = (p_i + p_j) * 0.5;
        if (abs(map(m)) > alpha * d_E * d_E) {
            continue;
        }

        force = force + (1.0 - d_infl / r) * diff / d_E;
    }

    force = force - dot(force, n_i) * n_i;

    let drifted = p_i + params.dt * force;
    let new_pos = projectToSurface(drifted, RELAX_NEWTON_ITERS, RELAX_ALPHA);

    points_out[i].pos = vec4f(new_pos, 0.0);
}
`;

/**
 * Shading compute shader — per-point visibility, shadow, and normal
 * refresh.
 *
 * One invocation per point. Runs every frame after the relax pass, reading
 * whichever point buffer relax most recently wrote. Computes the surface
 * normal from the (now final) position and writes it into the shared
 * normals buffer — so the normals buffer always matches the buffer that
 * the *next* relax pass will read (preserving the relax invariant that
 * normals match the read-side buffer across the ping-pong swap).
 *
 * Visibility: sphere-traces from `p + n·bias` toward the eye. The point
 * is visible iff the march either misses (`t < 0`) or reaches the eye
 * without hitting the surface (`t >= dist_to_eye`).
 *
 * Shadow: sphere-traces from `p + n·bias` toward the light direction. If
 * the march hits the surface, the point is in shadow (luminance 0);
 * otherwise Lambert shading applies.
 *
 * Output packing (`shading_out[i]`, `vec4f`):
 *  - `x` — visibility (`1.0` visible, `0.0` occluded).
 *  - `y` — luminance (`visible * lit * lambert`, in `[0, 1]`).
 *  - `z`, `w` — unused.
 *
 * The point renderer reads this buffer to discard occluded points in the
 * vertex shader and modulate the fragment color by luminance.
 */
export const SHADING_SHADER = /* wgsl */ `
${SDF_COMMON}

struct ShadingParams {
    eye: vec3f,
    point_count: u32,
    light_dir: vec3f,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: ShadingParams;
@group(0) @binding(1) var<storage, read> points_in: array<Point>;
@group(0) @binding(2) var<storage, read_write> normals_out: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> shading_out: array<vec4f>;

const SHADOW_BIAS: f32 = 0.004;

@compute @workgroup_size(64)
fn shading_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let p = points_in[i].pos.xyz;
    let n = normalize(sdfGradient(p));
    normals_out[i] = vec4f(n, 0.0);

    let origin = p + n * SHADOW_BIAS;

    let to_eye = params.eye - origin;
    let dist_eye = length(to_eye);
    let eye_dir = to_eye / dist_eye;
    let t_eye = rayMarch(origin, eye_dir);
    let visible = select(0.0, 1.0, t_eye < 0.0 || t_eye >= dist_eye);

    let t_light = rayMarch(origin, params.light_dir);
    let lit = select(0.0, 1.0, t_light < 0.0);

    let lambert = max(dot(n, params.light_dir), 0.0);
    let luminance = visible * lit * lambert;

    shading_out[i] = vec4f(visible, luminance, 0.0, 0.0);
}
`;

/**
 * Point render shader — draws stipple points as screen-space billboard
 * quads projected from 3D world space.
 *
 * The vertex shader projects each point's world position with
 * `view_proj`, then offsets the quad corners in NDC scaled by `clip.w` so
 * the disc has a constant pixel radius regardless of depth. Points flagged
 * as occluded in the shading buffer are pushed offscreen in the vertex
 * shader so no fragments are rasterized for them. The fragment shader
 * paints a soft red disc modulated by the per-point luminance from the
 * shading buffer, with alpha blending.
 */
export const POINT_SHADER = /* wgsl */ `
${SDF_COMMON}

struct PointUniform {
    view_proj: mat4x4f,
    resolution: vec2u,
    point_radius_px: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> u: PointUniform;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read> shading: array<vec4f>;

struct VertexOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) luminance: f32,
};

@vertex
fn point_vs(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOut {
    let corner = array<vec2f, 6>(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0),
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0),
    );

    let sh = shading[iid];
    var out: VertexOut;
    out.uv = corner[vid];
    out.luminance = sh.y;

    if (sh.x < 0.5) {
        out.clip_pos = vec4f(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let world = points[iid].pos.xyz;
    let clip = u.view_proj * vec4f(world, 1.0);

    let ndc = clip.xy / clip.w;
    let ndc_per_pixel = vec2f(2.0 / f32(u.resolution.x), 2.0 / f32(u.resolution.y));
    let offset = corner[vid] * u.point_radius_px * ndc_per_pixel;
    let new_ndc = ndc + offset;

    out.clip_pos = vec4f(new_ndc * clip.w, clip.z, clip.w);
    return out;
}

@fragment
fn point_fs(in: VertexOut) -> @location(0) vec4f {
    let dist = length(in.uv);
    if (dist > 1.0) {
        discard;
    }
    let alpha = smoothstep(1.0, 0.75, dist);
    let lum = in.luminance;
    return vec4f(0.75, 0.1, 0.0, alpha);
}
`;
