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
 * Interpolated into the seed, reproject, relax, shading, debug-render, and
 * point-render shaders. Contains only functions, structs, and compile-time
 * constants — no bindings.
 *
 * - `Point` — the per-point storage layout (pos only, 16 bytes). Surface
 *   normals live in a separate shared buffer, written by the seed pass
 *   (once, at bootstrap at t=0), by the reproject pass (every frame, before
 *   relax), and by the shading pass (every frame, after relax).
 * - `ShadingSample` — the per-point shading output layout (32 bytes):
 *   `luminance` carries the point's luminance in `x`; `clearance` carries
 *   one occlusion clearance value per quad corner (see `rayClearance`).
 *   Written by the shading pass every frame and read by the point
 *   renderer.
 * - `POINT_RADIUS_WORLD` / `tangentFrame(n)` — the canonical splat radius
 *   and tangent-basis construction shared by the shading pass (which
 *   samples occlusion clearance at the quad corners) and the point
 *   renderer (which builds the same quad corners), so corner rays and
 *   rasterized corners always coincide.
 * - `map(p, time)` — the scene signed distance field, parameterized by
 *   `time` for smooth morphing.
 * - `lightDir(time)` — the animated light direction.
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
 * - `rayClearance(ro, rd, max_dist, time)` — sphere-traces `map()` along
 *   `rd` from `ro` over `[0, max_dist]` and returns the minimum signed
 *   clearance between the ray and the SDF surface. Negative values are
 *   penetrations (the ray crossed into the surface); small positive
 *   values are grazes (the ray passed close without hitting). Combines a
 *   signed `map()` sample with a `softShadow`-style triangulated
 *   closest-approach estimate between consecutive unbounding spheres, so
 *   the minimum is tracked accurately even when the surface passes
 *   closest to the ray between sample positions. The origin's own
 *   surface is excluded by gating recording on travel distance:
 *   samples within the splat radius (`CLEARANCE_THRESHOLD`) of the
 *   origin are not recorded, since within that band the only surface is
 *   the one the point sits on.
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

struct ShadingSample {
    luminance: vec4f,
    clearance: vec4f,
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
const CLEARANCE_THRESHOLD: f32 = 1.5 * POINT_RADIUS_WORLD;

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

fn rayClearance(ro: vec3f, rd: vec3f, max_dist: f32, time: f32) -> f32 {
    if max_dist <= SURF_EPS {
        return CLEARANCE_THRESHOLD;
    }

    var t = min(PROBE_STEP, max_dist);
    var min_dist = CLEARANCE_THRESHOLD;

    // Ignore the initial positive threshold band until the ray has
    // traveled at least the desired clearance — within the splat radius
    // of the origin, the only surface is the origin's own.
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
        } else if t >= CLEARANCE_THRESHOLD {
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

// fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
//     let ab = b - a;
//     let ap = p - a;
//     let t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
//     return length(ap - ab * t) - r;
// }

fn sdSegment2D(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    return length(ap - ab * t);
}

fn sdTube(
    p: vec3f,
    a: vec3f,
    b: vec3f,
    r_a: f32,
    r_b: f32,
    w_a: f32,
    w_b: f32,
) -> f32 {
    let ab = b - a;
    let len = length(ab);
    let dir = ab / len;
    let ap = p - a;
    let t = dot(ap, dir);
    let rho = length(ap - dir * t);
    let p2d = vec2f(rho, t);
    let d_medial = sdSegment2D(p2d, vec2f(r_a, 0.0), vec2f(r_b, len));
    let w = (w_b - w_a) * t / len + w_a;
    return d_medial - w;
}

fn opTwist(p: vec3f, freq: f32, offset: f32) -> vec3f {
    let c = cos(freq * p.y + offset);
    let s = sin(freq * p.y + offset);
    return vec3f(c*p.x - s*p.z, p.y, s*p.x + c*p.z);
}

fn lightDir(time: f32) -> vec3f {
    const SWAY_AMPLITUDE: f32 = 0.3 * PI;
    const SWAY_FREQ: f32 = 0.5;
    const SWAY_OFFSET: f32 = 0.25 * PI;
    let theta = SWAY_AMPLITUDE * sin(SWAY_FREQ * time) + SWAY_OFFSET;
    return normalize(vec3f(sin(theta), 0.5, cos(theta)));
}

fn map(p_in: vec3f, time: f32) -> f32 {
    const TUBE_COUNT: f32 = 3.0;
    const R_BASE: f32 = 0.6;
    const R_TUBE: f32 = 0.75;
    const F_TWIST = 0.9;
    const F_SWAY = 1.0;
    const F_CONTRACT = F_SWAY;

    var min_dist = 1.0e20;

    let p = opTwist(p_in, 0.2 * sin(F_TWIST * time), 0.0);

    for (var i: f32 = 0.0; i < TUBE_COUNT; i += 1.0) {
        let alpha = i * TAU / TUBE_COUNT;
        let c = R_BASE * cos(alpha);
        let s = R_BASE * sin(alpha);
        let a = vec3f(0.1 * c, -2.0, 0.1 * s);
        let theta = 0.3 * i;
        let s_b = 0.85 * sin(theta + F_SWAY * time) + 1.9;
        let h_b = 0.8 * sin(theta + PI + F_CONTRACT * time);
        let b = vec3f(s_b * c, 1.0 + h_b, s_b * s);
        let r_b = (0.3 * sin(theta + F_CONTRACT * time) + 0.6) * R_TUBE;
        min_dist = smin(min_dist, sdTube(p, a, b, R_TUBE, r_b, 1.1 * R_TUBE, 0.15), 0.2);
    }
    return min_dist;
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
