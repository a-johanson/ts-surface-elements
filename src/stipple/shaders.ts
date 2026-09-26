/**
 * WGSL shader source strings for the surface-stippling pipeline.
 *
 * Points live in 3D world space on the surface of the time-animated SDF
 * scene. The shared {@link SDF_COMMON} block — interpolated into every
 * shader that needs the SDF — owns the scene description (`map`), a
 * time-parameterized gradient/normal helper set, and a Newton-Raphson
 * surface projection routine. All SDF helpers thread a `time` parameter so
 * the surface can be re-evaluated at the correct animation phase.
 */

/**
 * Shared WGSL block: time-animated SDF scene, gradient, normal, surface
 * projection, and ray-march.
 *
 * Interpolated into the seed, reproject, relax, shading, debug-render, and
 * point-render shaders. Contains only functions, structs, and compile-time
 * constants — no bindings.
 *
 * - `Point` — the per-point storage layout (pos only, 16 bytes). Surface
 *   normals live in a separate shared buffer, written by the seed pass
 *   (once, at bootstrap at t=0), by the reproject pass (every frame, before
 *   relax), and by the shading pass (every frame, after relax).
 * - `map(p, time)` — the scene signed distance field, parameterized by
 *   `time` for smooth morphing.
 * - `sdfGradient(p, time)` — tetrahedron-pattern gradient (4 taps); used by
 *   Newton projection so the magnitude is the true distance-field gradient.
 * - `calcNormal(p, time)` — normalized gradient; used by the debug render
 *   for Lambert shading.
 * - `projectToSurface(p, time, iters, alpha)` — Newton-Raphson steps
 *   `p -= alpha * (f(p) / |∇f|²) * ∇f` until `|f(p)| < SURF_EPS` or the
 *   iteration budget is exhausted. For a true distance field (`|∇f| ≈ 1`)
 *   and `alpha = 1`, each step lands on the surface in one go; smaller
 *   `alpha` guards against overshoot in `smin` blend regions where the
 *   field is not a perfect distance function.
 * - `rayMarch(ro, rd, time)` — sphere-traces `map()` along `rd` from `ro`
 *   up to `MAX_DIST` with `MAX_STEPS` iterations. Returns the hit distance,
 *   or `-1.0` on miss. Shared by the debug render (per-fragment SDF
 *   visualization) and the shading pass (per-point visibility tests).
 * - `softShadow(ro, rd, time)` — Aaltonen penumbra estimation: sphere-
 *   traces `map()` toward the light and returns a `[0, 1]` factor (1 =
 *   fully lit, 0 = fully occluded). At each step it triangulates the
 *   current and previous unbounding spheres to estimate the closest
 *   surface approach to the ray, catching penumbras that fall between
 *   sample positions (especially near sharp corners). Shared by the
 *   debug render (per-fragment self-shadowing) and the shading pass
 *   (per-point soft shadow).
 */
export const SDF_COMMON = /* wgsl */ `
struct Point {
    pos: vec4f,
};

const SURF_EPS: f32 = 0.001;

const RAYMARCH_MAX_DIST: f32 = 50.0;
const RAYMARCH_STEP_SCALE: f32 = 1.0;
const RAYMARCH_MAX_STEPS: i32 = 250;

const SHADOW_BIAS: f32 = 0.005;
const SHADOW_MAX_DIST: f32 = 5.0;
const SHADOW_W: f32 = 0.4;
const SHADOW_MAX_STEPS: i32 = 128;

fn rayMarch(ro: vec3f, rd: vec3f, time: f32) -> f32 {
    var t = 0.0;
    for (var i: i32 = 0; i < RAYMARCH_MAX_STEPS; i = i + 1) {
        let p = ro + rd * t;
        let d = map(p, time);
        if (d < SURF_EPS) {
            return t;
        }
        t += RAYMARCH_STEP_SCALE * d;
        if (t > RAYMARCH_MAX_DIST) {
            break;
        }
    }
    return -1.0;
}

fn softShadow(ro: vec3f, rd: vec3f, time: f32) -> f32 {
    var res = 1.0;
    var pd = 1e20;
    var t = SHADOW_BIAS;
    for (var i: i32 = 0; i < SHADOW_MAX_STEPS && t < SHADOW_MAX_DIST; i = i + 1) {
        let d = map(ro + rd * t, time);
        if (d < SURF_EPS) {
            return 0.0;
        }
        let y = d * d / (2.0 * pd);
        let cd = sqrt(d * d - y * y);
        res = min(res, cd / (SHADOW_W * max(0.0, t - y)));
        pd = d;
        t = t + d;
    }
    return res;
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

fn map(p: vec3f, time: f32) -> f32 {
    let r1 = 1.5 + 0.2 * sin(time);
    let r2 = 1.0 + 0.15 * cos(time * 0.7);
    let box_y = 1.6 + 0.3 * sin(time * 0.5);
    let d1 = sdSphere(p - vec3f(-1.2, 0.0, 0.0), r1);
    let d2 = sdTorus(p - vec3f(1.2, 0.0, 0.0), vec2f(r2, 0.35));
    let d3 = sdBox(p - vec3f(0.0, box_y, 0.0), vec3f(1.0, 0.4, 1.0));
    let d12 = smin(d1, d2, 0.6);
    return smin(d12, d3, 0.6);
}

fn sdfGradient(p: vec3f, time: f32) -> vec3f {
    const h: f32 = 0.0005;
    const k = vec3f(1.0, -1.0, -1.0);
    return (
        k.xyy * map(p + k.xyy * h, time) +
        k.yyx * map(p + k.yyx * h, time) +
        k.yxy * map(p + k.yxy * h, time) +
        k.xxx * map(p + k.xxx * h, time)
    ) / (4.0 * h);
}

fn calcNormal(p: vec3f, time: f32) -> vec3f {
    return normalize(sdfGradient(p, time));
}

fn projectToSurface(p_in: vec3f, time: f32, iters: i32, alpha: f32) -> vec3f {
    var p = p_in;
    for (var i: i32 = 0; i < iters; i = i + 1) {
        let f = map(p, time);
        if (abs(f) < SURF_EPS) {
            break;
        }
        let g = sdfGradient(p, time);
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
 * Debug render shader — ray-marches the time-animated SDF per fragment and
 * writes grayscale Lambert shading modulated by Aaltonen soft shadows
 * directly to the canvas color attachment.
 *
 * Background (ray miss) maps to black; hit fragments map to `[0, 1]` gray
 * via `1 - Lambert`, scaled by the soft-shadow penumbra factor.
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
    time: f32,
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

    let t = rayMarch(camera.eye, rd, camera.time);
    if (t < 0.0) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    let p = camera.eye + rd * t;
    let n = calcNormal(p, camera.time);
    let light_dir = normalize(vec3f(0.5, 0.8, 0.6));
    let lambert = max(dot(n, light_dir), 0.0);
    let shadow = softShadow(p + n * SHADOW_BIAS, light_dir, camera.time);
    let lum = lambert * shadow;
    return vec4f(lum, lum, lum, 1.0);
}
`;

/**
 * Seed compute shader — 3D rejection sampling in a bounding box.
 *
 * One invocation per point. Generates pseudo-random candidates uniformly
 * in `[bbox_min, bbox_max]³` via a PCG hash and accepts the first whose
 * `|map(p, 0)| < band`. Accepted (and fallback) points are projected onto
 * the surface at t=0 with Newton-Raphson before being written.
 *
 * Seeding is a one-shot bootstrap at t=0 — the SDF is evaluated at its
 * initial configuration. Subsequent surface motion is handled per-frame by
 * the reproject pass.
 *
 * In addition to positions, the seed pass writes the matching surface
 * normal for each point into the shared `normals_out` buffer. The normals
 * buffer is shared (not ping-ponged) and must match whichever point buffer
 * relax reads; since seed writes bufferA, it also seeds the normals from
 * bufferA. Subsequent frames have their normals refreshed by the reproject
 * pass (before relax) and the shading pass (after relax).
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
const SEED_TIME: f32 = 0.0;

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
        let d = abs(map(p, SEED_TIME));

        if (d < best_d) {
            best_d = d;
            best_p = p;
        }

        if (d < params.band) {
            let projected = projectToSurface(p, SEED_TIME, SEED_NEWTON_ITERS, SEED_ALPHA);
            points_out[i].pos = vec4f(projected, 0.0);
            normals_out[i] = vec4f(normalize(sdfGradient(projected, SEED_TIME)), 0.0);
            accepted = true;
            break;
        }
    }

    if (!accepted) {
        let projected = projectToSurface(best_p, SEED_TIME, SEED_NEWTON_ITERS, SEED_ALPHA);
        points_out[i].pos = vec4f(projected, 0.0);
        normals_out[i] = vec4f(normalize(sdfGradient(projected, SEED_TIME)), 0.0);
    }
}
`;

/**
 * Reproject compute shader — re-projects points onto the current animated
 * surface and refreshes the shared normals buffer.
 *
 * One invocation per point. Runs every frame *before* relax. Reads each
 * point's position (from the buffer relax is about to read), Newton-
 * projects it onto the surface at the current animation time with a
 * generous iteration budget, writes the projected position back in-place,
 * and writes the matching surface normal into the shared normals buffer.
 *
 * This ensures that relax starts with on-surface positions and matching
 * normals even when the SDF has moved since the last frame. Without this
 * pass, relax would read stale positions (off the new surface) and stale
 * normals (describing the old surface), corrupting the curvature-aware
 * repulsion kernel.
 *
 * In-place `read_write` on the point buffer is safe because each invocation
 * touches only index `i`. Two bind groups (A/B) cover the ping-pong
 * alternation.
 */
export const REPROJECT_SHADER = /* wgsl */ `
${SDF_COMMON}

struct ReprojectParams {
    time: f32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: ReprojectParams;
@group(0) @binding(1) var<storage, read_write> points: array<Point>;
@group(0) @binding(2) var<storage, read_write> normals_out: array<vec4f>;

const REPROJECT_NEWTON_ITERS: i32 = 16;
const REPROJECT_ALPHA: f32 = 0.5;

@compute @workgroup_size(64)
fn reproject_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let p = points[i].pos.xyz;
    let projected = projectToSurface(p, params.time, REPROJECT_NEWTON_ITERS, REPROJECT_ALPHA);
    points[i].pos = vec4f(projected, 0.0);
    normals_out[i] = vec4f(normalize(sdfGradient(projected, params.time)), 0.0);
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
 * which is written by the reproject pass (before relax) and the shading
 * pass (after relax) from the same buffer relax reads — so the normals
 * match the current positions. Pairs across narrow gaps, high-curvature
 * regions, or self-folding `smin` geometry thus see an effectively larger
 * separation, suppressing cross-sheet repulsion that would otherwise
 * corrupt the Poisson-disc distribution.
 *
 * Distance usage:
 *  - cutoff `d_infl > radius`        → skip (inflated)
 *  - line-of-sight `alpha·d_E²`      → skip (Euclidean)
 *  - decay `(1 - d_infl/radius)`     → inflated
 *  - direction `(p_i - p_j) / d_E`   → Euclidean (unit)
 *
 * The accumulated force is projected onto the tangent plane at `pi` and
 * integrated with a direct Euler position step `x* = x + dt·F_tan`, where
 * `dt` is the per-frame capped wall-clock delta threaded from the frame
 * loop (not a constant). The new position is re-projected onto the
 * surface with a few Newton steps; since per-frame drift is small,
 * `RELAX_NEWTON_ITERS = 4` with `alpha = 1` suffices.
 */
export const RELAX_SHADER = /* wgsl */ `
${SDF_COMMON}

struct RelaxParams {
    dt: f32,
    radius: f32,
    alpha: f32,
    time: f32,
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
    let time = params.time;

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
        if (abs(map(m, time)) > alpha * d_E * d_E) {
            continue;
        }

        force = force + (1.0 - d_infl / r) * diff / d_E;
    }

    force = force - dot(force, n_i) * n_i;

    let drifted = p_i + params.dt * force;
    let new_pos = projectToSurface(drifted, time, RELAX_NEWTON_ITERS, RELAX_ALPHA);

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
 * the *next* reproject + relax cycle will read (preserving the relax
 * invariant that normals match the read-side buffer across the ping-pong
 * swap).
 *
 * Visibility: sphere-traces from `p + n·bias` toward the eye. The point
 * is visible iff the march either misses (`t < 0`) or reaches the eye
 * without hitting the surface (`t >= dist_to_eye`).
 *
 * Shadow: Aaltonen soft-shadow penumbra estimation from `p + n·bias`
 * toward the light direction, returning a `[0, 1]` factor that modulates
 * the Lambert term to produce graded penumbras instead of a binary
 * shadow cut.
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
    time: f32,
};

@group(0) @binding(0) var<uniform> params: ShadingParams;
@group(0) @binding(1) var<storage, read> points_in: array<Point>;
@group(0) @binding(2) var<storage, read_write> normals_out: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> shading_out: array<vec4f>;

@compute @workgroup_size(64)
fn shading_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let time = params.time;
    let p = points_in[i].pos.xyz;
    let n = normalize(sdfGradient(p, time));
    normals_out[i] = vec4f(n, 0.0);

    let origin = p + n * SHADOW_BIAS;

    let to_eye = params.eye - origin;
    let dist_eye = length(to_eye);
    let eye_dir = to_eye / dist_eye;

    if (dot(n, eye_dir) <= 0.0) {
        shading_out[i] = vec4f(0.0);
        return;
    }

    let t_eye = rayMarch(origin, eye_dir, time);
    if (t_eye >= 0.0 && t_eye < dist_eye) {
        shading_out[i] = vec4f(0.0);
        return;
    }

    let lit = softShadow(origin, params.light_dir, time);
    let lambert = max(dot(n, params.light_dir), 0.0);
    shading_out[i] = vec4f(1.0, lit * lambert, 0.0, 0.0);
}
`;

/**
 * Point render shader — draws stipple points as tangent-plane quads
 * oriented from the shared normals buffer and projected via a
 * view-projection matrix uniform.
 *
 * The vertex shader builds an orthonormal tangent basis (t1, t2) from
 * the pre-computed surface normal at each point and offsets the quad
 * corners in world space by `POINT_RADIUS_WORLD`, so quads lie flat on
 * the SDF surface and foreshorten naturally with viewing angle. Points
 * flagged as occluded in the shading buffer are pushed offscreen in the
 * vertex shader so no fragments are rasterized for them. The fragment
 * shader paints a white ring outline whose radius scales linearly with
 * per-point luminance (collapsing to a point at zero luminance); the
 * line width is configurable via `LINE_WIDTH` and the ring never
 * exceeds the quad boundary.
 */
export const POINT_SHADER = /* wgsl */ `
${SDF_COMMON}

struct PointUniform {
    view_proj: mat4x4f,
};

@group(0) @binding(0) var<uniform> u: PointUniform;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read> shading: array<vec4f>;
@group(0) @binding(3) var<storage, read> normals: array<vec4f>;

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
    const POINT_RADIUS_WORLD: f32 = 0.03;

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

    let p = points[iid].pos.xyz;
    let n = normals[iid].xyz;
    let up = mix(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), f32(abs(n.y) > 0.99));
    let t1 = normalize(cross(up, n));
    let t2 = cross(n, t1);
    let c = corner[vid];
    let world = p + (c.x * t1 + c.y * t2) * POINT_RADIUS_WORLD;
    out.clip_pos = u.view_proj * vec4f(world, 1.0);
    return out;
}

@fragment
fn point_fs(in: VertexOut) -> @location(0) vec4f {
    const LINE_WIDTH: f32 = 0.35;
    const HALF_W: f32 = LINE_WIDTH * 0.5;
    const AA: f32 = LINE_WIDTH * 0.25;
    const MIN_R: f32 = 0.75 * HALF_W;
    const MAX_R: f32 = 1.0 - (HALF_W + AA);
    let r = (MAX_R - MIN_R) * in.luminance + MIN_R;
    let dist = length(in.uv);
    let d = abs(dist - r);
    if (d > HALF_W + AA) {
        discard;
    }
    let alpha = smoothstep(HALF_W + AA, HALF_W, d);
    return vec4f(1.0, 1.0, 1.0, alpha);
}
`;
