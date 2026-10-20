/**
 * WGSL shader source strings for the surface-stippling pipeline.
 *
 * Points live in 3D world space on the surface of the time-animated SDF
 * scene. The shared {@link SDF_COMMON} block — interpolated into every
 * shader that needs the SDF — owns the scene description (`map`), an animated
 * light direction (`lightDir`), a time-parameterized gradient/normal
 * helper set, and a Newton-Raphson surface projection routine. All SDF helpers thread a `time` parameter so
 * the surface can be re-evaluated at the correct animation phase.
 */

/**
 * Shared WGSL block: time-animated SDF scene, gradient, normal, surface
 * projection, and ray-march.
 *
 * Interpolated into the seed, reproject, relax, normals-refresh,
 * debug-render, point-render, and hatch shaders. Contains only functions,
 * structs, and compile-time constants — no bindings.
 *
 * - `Point` — the per-point storage layout (pos only, 16 bytes). Surface
 *   normals live in a separate shared buffer, written by the seed pass
 *   (once, at bootstrap at t=0), by the reproject pass (every playing
 *   frame, before relax), and by the normals-refresh pass (every playing
 *   frame, after relax).
 * - `POINT_RADIUS_WORLD` / `tangentFrame(n)` — the canonical splat radius
 *   and tangent-basis construction used by the point renderer to offset
 *   its quad corners onto the tangent plane, so quads lie flat on the
 *   curved surface.
 * - `map(p, time)` — the scene signed distance field, parameterized by
 *   `time` for smooth morphing.
 * - `lightDir(time)` — the animated light direction.
 * - `sdfGradient(p, time)` — tetrahedron-pattern gradient (4 taps); used by
 *   Newton projection so the magnitude is the true distance-field gradient.
 * - `calcNormal(p, time)` — normalized gradient; used by the debug render
 *   for Lambert shading and by the normals-refresh pass.
 * - `projectToSurface(p, time, iters, alpha)` — Newton-Raphson steps
 *   `p -= alpha * (f(p) / |∇f|²) * ∇f` until `|f(p)| < SURF_EPS` or the
 *   iteration budget is exhausted. For a true distance field (`|∇f| ≈ 1`)
 *   and `alpha = 1`, each step lands on the surface in one go; smaller
 *   `alpha` guards against overshoot where the field is not a perfect
 *   distance function, e.g. across `min` seams between scene primitives.
 * - `rayMarch(ro, rd, time)` — sphere-traces `map()` along `rd` from `ro`
 *   up to `MAX_DIST` with `MAX_STEPS` iterations. Returns the hit distance,
 *   or `-1.0` on miss. Shared by the debug render (per-fragment SDF
 *   visualization).
 * - `rayClearance(ro, rd, max_dist, threshold, time)` — sphere-traces
 *   `map()` along `rd` from `ro` over `[0, max_dist]` and returns the
 *   minimum signed clearance between the ray and the SDF surface.
 *   Negative values are penetrations (the ray crossed into the surface);
 *   small positive values are grazes (the ray passed close without
 *   hitting). Combines a signed `map()` sample with a `softShadow`-style
 *   triangulated closest-approach estimate between consecutive
 *   unbounding spheres, so the minimum is tracked accurately even when
 *   the surface passes closest to the ray between sample positions.
 *   `threshold` is the caller's clearance scale, playing two roles: an
 *   unobstructed trace returns exactly `threshold` (the positive
 *   clearance saturates there), and the origin's own surface is excluded
 *   by gating recording on travel distance — samples within `threshold`
 *   of the origin are not recorded, since within that band the only
 *   surface is the one the point sits on. The hatch clearance stage
 *   passes each stroke point's local stroke spacing.
 * - `softShadow(ro, rd, time)` — penumbra estimation: sphere-
 *   traces `map()` toward the light and returns a `[0, 1]` factor (1 =
 *   fully lit, 0 = fully occluded). At each step it triangulates the
 *   current and previous unbounding spheres to estimate the closest
 *   surface approach to the ray, catching penumbras that fall between
 *   sample positions (especially near sharp corners). Shared by the
 *   debug render (per-fragment self-shadowing) and the reproject pass
 *   (per-point soft shadow for luminance).
 */
export const SDF_COMMON = /* wgsl */ `
struct Point {
    pos: vec4f,
};

struct TangentFrame {
    t1: vec3f,
    t2: vec3f,
};

const PI = radians(180.0);
const TAU = radians(360.0);

const SURF_EPS: f32 = 0.001;
const PROBE_STEP: f32 = 5.0 * SURF_EPS;

const RAYMARCH_MAX_DIST: f32 = 50.0;
const RAYMARCH_STEP_SCALE: f32 = 0.7;
const RAYMARCH_MAX_STEPS: u32 = 250;

const SHADOW_BIAS: f32 = 5.0 * SURF_EPS;
const SHADOW_MAX_DIST: f32 = 5.0;
const SHADOW_W: f32 = 0.4;
const SHADOW_MAX_STEPS: u32 = 128;

const POINT_RADIUS_WORLD: f32 = 0.03;

fn rayMarch(ro: vec3f, rd: vec3f, time: f32) -> f32 {
    var t = 0.0;
    for (var i: u32 = 0u; i < RAYMARCH_MAX_STEPS && t <= RAYMARCH_MAX_DIST; i += 1u) {
        let p = ro + rd * t;
        let d = map(p, time);
        if (d < SURF_EPS) {
            return t;
        }
        t += RAYMARCH_STEP_SCALE * d;
    }
    return -1.0;
}

fn rayClearance(
    ro: vec3f,
    rd: vec3f,
    max_dist: f32,
    threshold: f32,
    time: f32,
) -> f32 {
    if max_dist <= SURF_EPS {
        return threshold;
    }

    var t = min(PROBE_STEP, max_dist);
    var min_dist = threshold;

    // Ignore the initial positive threshold band until the ray has
    // traveled at least the threshold — within that band of the origin,
    // the only surface is the origin's own.
    var tracking = false;

    // Previous sample's signed distance, for the softShadow-style
    // triangulated closest-approach estimate between samples.
    var prev_dist = 1e20;

    for (var i = 0u; i < RAYMARCH_MAX_STEPS && t < max_dist; i += 1u) {
        let dist = map(ro + rd * t, time);

        if dist < 0.0 {
            // Penetration always counts, including near the origin.
            min_dist = min(min_dist, dist);
            tracking = true;
            // Reset so the next positive sample's triangulation degenerates
            // (cd ≈ dist) instead of triangulating against a stale
            // pre-penetration sphere.
            prev_dist = 1e20;
        } else if tracking {
            min_dist = min(min_dist, dist);

            // Triangulate the closest surface approach to the ray in the
            // interval between the previous and current unbounding
            // spheres — the surface can graze the ray between sample
            // positions, which a plain min(dist) would miss. Mirrors the
            // penumbra estimate in softShadow, but tracks the minimum
            // clearance itself instead of a shadow factor.
            let apex_offset = dist * dist / (2.0 * prev_dist);
            let closest_approach = sqrt(max(0.0, dist * dist - apex_offset * apex_offset));
            min_dist = min(min_dist, closest_approach);

            prev_dist = dist;
        } else if t >= threshold {
            // The ray has escaped the initial surface neighborhood.
            tracking = true;
        }

        var step: f32;

        if abs(dist) <= SURF_EPS {
            // Move through the numerically ambiguous surface region.
            step = PROBE_STEP;
        } else if !tracking {
            step = max(dist * RAYMARCH_STEP_SCALE, PROBE_STEP);
        } else {
            // Continue in either direction using the unsigned distance.
            step = abs(dist) * RAYMARCH_STEP_SCALE;
        }

        step = min(step, max(max_dist - t, PROBE_STEP));

        t += step;
    }

    if t < max_dist - SURF_EPS {
        return min(min_dist, 0.0);
    }

    return min_dist;
}

fn tangentFrame(n: vec3f) -> TangentFrame {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.99);
    let t1 = normalize(cross(up, n));
    return TangentFrame(t1, cross(n, t1));
}

fn softShadow(ro: vec3f, rd: vec3f, time: f32) -> f32 {
    var res = 1.0;
    var pd = 1e20;
    var t = SHADOW_BIAS;
    for (var i: u32 = 0u; i < SHADOW_MAX_STEPS && t < SHADOW_MAX_DIST; i += 1u) {
        let d = map(ro + rd * t, time);
        if (d < SURF_EPS) {
            return 0.0;
        }
        let y = d * d / (2.0 * pd);
        let cd = sqrt(d * d - y * y);
        res = min(res, cd / (SHADOW_W * max(0.0, t - y)));
        pd = d;
        t = t + RAYMARCH_STEP_SCALE * d;
    }
    return res;
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn sdTorus(p: vec3f, major: f32, minor: f32) -> f32 {
    let q = vec2f(length(p.xz) - major, p.y);
    return length(q) - minor;
}

// fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
//     let ab = b - a;
//     let ap = p - a;
//     let t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
//     return length(ap - ab * t) - r;
// }

fn opTwist(p: vec3f, freq: f32, offset: f32) -> vec3f {
    let angle = freq * p.x + offset;
    let c = cos(angle);
    let s = sin(angle);
    return vec3f(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

fn lightDir(time: f32) -> vec3f {
    return normalize(vec3f(0.8, 2.0, 1.0));
}

fn map(p_in: vec3f, time: f32) -> f32 {
    const TORUS_MAJOR: f32 = 1.25;
    const TORUS_MINOR: f32 = 0.35;
    const LINK_SHIFT: f32 = 0.82;
    const TWIST_RATE: f32 = 0.3;
    const F_TWIST: f32 = 0.9;
    const TWIST_MID: f32 = -0.5 * LINK_SHIFT;

    let rate = TWIST_RATE * sin(F_TWIST * time);
    let p = opTwist(p_in, rate, -rate * TWIST_MID);

    let d_a = sdTorus(p, TORUS_MAJOR, TORUS_MINOR);
    let p_b = vec3f(p.x + LINK_SHIFT, p.z, -p.y);
    let d_b = sdTorus(p_b, TORUS_MAJOR, TORUS_MINOR);
    return min(d_a, d_b);
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
    for (var i: i32 = 0; i < iters; i += 1) {
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
