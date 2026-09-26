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
 * - `softShadow(ro, rd, time)` — penumbra estimation: sphere-
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
    let d1 = sdSphere(p - vec3f(-1.2, 0.0, 0.0), r1);
    let d2 = sdTorus(p - vec3f(1.2, 0.0, 0.0), vec2f(r2, 0.35));
    let d3 = sdSphere(p - vec3f(1.0, 0.4, 1.0), 0.8 * r1);
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
