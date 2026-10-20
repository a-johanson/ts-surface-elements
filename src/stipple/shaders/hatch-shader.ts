/**
 * Hatch compute shader — grows a surface-projected hatch stroke from
 * every stipple point by integrating the principal direction field,
 * plus a per-stroke-point occlusion clearance pass.
 *
 * One module, two entry points. The first is dispatched by
 * `HatchPipeline` on frames where the simulation stepped; the second
 * runs every frame (its input — the eye — changes with the camera even
 * while time is paused):
 *
 *  1. `hatch_cs` — one invocation per stipple point `i`. Grows a stroke
 *     of `STRIDE = 2 * N_SEGMENTS_HALF + 1` surface points centered on
 *     the stipple point: the center slot is the stipple point itself
 *     (written exactly), and each arm takes `N_SEGMENTS_HALF` steps —
 *     advancing by `HATCH_STEP_SIZE` along the current principal
 *     direction, then Newton-projecting onto the surface. The principal
 *     direction is the unit vector in the tangent plane orthogonal to
 *     the light direction's tangent projection — hatch lines run
 *     "across the grain" of the lighting — and it is recomputed at
 *     every step point from a fresh surface normal, so the stroke
 *     follows the direction field continuously. That is what keeps the
 *     strokes pristine where the former link-based approach wobbled:
 *     stroke vertices are field samples rather than stipple points
 *     (whose positions jitter with the relax simulation), and the
 *     evenly-spaced property is intentionally given up (line density
 *     now equals stipple density). When the direction degenerates
 *     (`|L_tan|^2 < HATCH_TAN_EPS2` — the light nearly parallel to the
 *     normal), `principalDir` returns a zero direction: the step
 *     projects the standing point in place, so the arm stops moving
 *     and its remaining slots are rewritten with that same point —
 *     every slot always holds a real surface point, and the render's
 *     zero-length sub-segment path draws the stroke as a dot. The field
 *     `cross(n, L_tan)` is a continuous function of position away from
 *     the degeneracy, so recomputing it at each step orients the march
 *     without sign-continuity bookkeeping.
 *
 *     The pass also fills a parallel slot-normals buffer, slot-aligned
 *     with the stroke buffer: the center slot gets the stipple point's
 *     own normal from the shared normals buffer, and each grown point
 *     gets the fresh surface normal the walk computed there — the
 *     clearance stage consumes those same normals instead of
 *     re-deriving them. Every slot of both buffers is rewritten each
 *     playing frame, and the stroke keeps rendering while time is
 *     paused.
 *
 *  2. `hatch_clearance_cs` — one invocation per point, dispatched every
 *     frame (view-dependent). Traces one `rayClearance` per stroke point
 *     toward the eye — origin biased along the slot's surface normal
 *     from the slot-normals buffer, back-facing points short-circuited
 *     to an explicitly occluded value — and writes the clearance into
 *     the slot's `.w`, replacing
 *     the validity placeholder (> 0 visible, < 0 occluded). Each stroke
 *     point's clearance threshold is its local stroke spacing (the
 *     longest adjacent sub-segment length, floored at `PROBE_STEP`), so
 *     the trace's origin ignore-band covers the stroke's own surface
 *     neighborhood around this sample. Because this stage runs every
 *     frame, the hatch render's occlusion culling and soft fade stay
 *     view-dependent while time is paused.
 */
import { SDF_COMMON } from "./sdf-common.js";

/**
 * Steps grown per arm from each stipple point — the single source of
 * truth for this value. Must be at least 1: the growth walk and the
 * clearance pass both assume the stroke has grown slots around the
 * stipple point.
 *
 * Tunable. It sizes the stroke buffer and the render instance count,
 * which live on the TypeScript side, so it is defined here (beside the
 * shader that consumes it) and interpolated into both the hatch compute
 * shader and the hatch render shader. Each stroke is
 * `2 * HATCH_N_SEGMENTS_HALF + 1` points — the stipple point plus
 * `HATCH_N_SEGMENTS_HALF` grown points per arm; the stroke's shape is
 * fixed by the direction field and `HATCH_STEP_SIZE`, this value only
 * sets the stroke length.
 */
export const HATCH_N_SEGMENTS_HALF = 10;

export const HATCH_SHADER = /* wgsl */ `
${SDF_COMMON}

struct HatchParams {
    eye: vec3f,
    time: f32,
    point_count: u32,
    _pad: u32,
};

/**
 * World-space step length between consecutive stroke points. Tunable —
 * together with N_SEGMENTS_HALF it sets the stroke length
 * (2 * N_SEGMENTS_HALF * HATCH_STEP_SIZE).
 */
const HATCH_STEP_SIZE: f32 = 0.02;

/**
 * Below this tangent-projection magnitude the light is considered
 * parallel to the normal: the projected direction degenerates and
 * principalDir returns a zero direction, which makes the walk stand
 * still (the stroke renders as a dot).
 */
const HATCH_TAN_EPS2: f32 = 1e-8;

/** Newton iterations per step's surface projection. */
const HATCH_NEWTON_ITERS: i32 = 1;

/** Newton step size. */
const HATCH_NEWTON_ALPHA: f32 = 1.0;

/**
 * Steps grown per arm from the stipple point — interpolated from the
 * TS constant HATCH_N_SEGMENTS_HALF (single source of truth). Each
 * stroke is 2 * N_SEGMENTS_HALF + 1 points: the stipple point at the
 * center slot plus N_SEGMENTS_HALF grown points per arm.
 */
const N_SEGMENTS_HALF: u32 = ${HATCH_N_SEGMENTS_HALF}u;

/** Stroke points per stipple point (2 * N_SEGMENTS_HALF + 1). */
const STRIDE: u32 = 2u * N_SEGMENTS_HALF + 1u;

@group(0) @binding(0) var<uniform> params: HatchParams;
@group(0) @binding(1) var<storage, read> points_in: array<Point>;
@group(0) @binding(2) var<storage, read> normals_in: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> curves_out: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> curve_normals: array<vec4f>;

/**
 * The principal direction at a surface point: the unit vector in the
 * tangent plane orthogonal to the light direction's tangent projection.
 * Returns a zero direction when the light is nearly parallel to the
 * normal (|L_tan|^2 < HATCH_TAN_EPS2): the projection degenerates, and
 * a zero direction makes the walk stand still (the stroke renders as a
 * dot) instead of normalizing a near-zero vector into NaN.
 */
fn principalDir(p: vec3f, n: vec3f, L: vec3f) -> vec3f {
    let L_tan = L - dot(L, n) * n;
    if (dot(L_tan, L_tan) < HATCH_TAN_EPS2) {
        return vec3f(0.0);
    }
    return normalize(cross(n, L_tan));
}

/**
 * Occlusion clearance of a surface point toward the eye: origin biased
 * along the surface normal, back-facing points short-circuited to an
 * explicitly occluded value, rayClearance toward the eye. Non-positive
 * results mean the surface occludes the point from the camera; positive
 * results saturate at threshold (the ray's closest unobstructed
 * approach). The caller supplies the threshold — each stroke point uses
 * its local stroke spacing.
 */
fn surfaceClearance(p: vec3f, n: vec3f, eye: vec3f, threshold: f32, time: f32) -> f32 {
    let p_biased = p + n * SHADOW_BIAS;
    let to_eye = eye - p_biased;
    let dist_eye = length(to_eye);
    let eye_dir = to_eye / dist_eye;
    if (dot(n, eye_dir) <= 0.0) {
        return -1.0;
    }
    return rayClearance(p_biased, eye_dir, dist_eye, threshold, time);
}

@compute @workgroup_size(64)
fn hatch_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let base = i * STRIDE;
    let p0 = points_in[i].pos.xyz;
    let n0 = normals_in[i].xyz;
    let L = lightDir(params.time);

    curves_out[base + N_SEGMENTS_HALF] = vec4f(p0, 1.0);
    curve_normals[base + N_SEGMENTS_HALF] = vec4f(n0, 0.0);

    var pF = p0;
    var pB = p0;
    var nF = n0;
    var nB = n0;
    for (var k: u32 = 1u; k <= N_SEGMENTS_HALF; k += 1u) {
        let dirF = principalDir(pF, nF, L);
        let dirB = -principalDir(pB, nB, L);
        pF = projectToSurface(pF + HATCH_STEP_SIZE * dirF, params.time,
                              HATCH_NEWTON_ITERS, HATCH_NEWTON_ALPHA);
        pB = projectToSurface(pB + HATCH_STEP_SIZE * dirB, params.time,
                              HATCH_NEWTON_ITERS, HATCH_NEWTON_ALPHA);
        curves_out[base + N_SEGMENTS_HALF + k] = vec4f(pF, 1.0);
        curves_out[base + N_SEGMENTS_HALF - k] = vec4f(pB, 1.0);
        nF = calcNormal(pF, params.time);
        nB = calcNormal(pB, params.time);
        curve_normals[base + N_SEGMENTS_HALF + k] = vec4f(nF, 0.0);
        curve_normals[base + N_SEGMENTS_HALF - k] = vec4f(nB, 0.0);
    }
}

@compute @workgroup_size(64)
fn hatch_clearance_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let base = i * STRIDE;
    for (var k: u32 = 0u; k < STRIDE; k = k + 1u) {
        let p = curves_out[base + k].xyz;
        let n = curve_normals[base + k].xyz;
        // Clearance threshold: the local stroke spacing — the longest of
        // the adjacent sub-segments (single neighbor at the stroke ends),
        // so the trace's origin ignore-band covers the stroke's own
        // surface neighborhood around this sample. Floored at
        // PROBE_STEP: coincident neighbors (a degenerate arm stands
        // still, so its remaining slots hold the same point) give zero
        // spacing, which would make an unobstructed trace
        // return 0.0 — indistinguishable from the pre-first-frame zero
        // buffer the renderer culls on.
        var spacing: f32;
        if (k == 0u) {
            spacing = length(curves_out[base + 1u].xyz - p);
        } else if (k == STRIDE - 1u) {
            spacing = length(curves_out[base + k - 1u].xyz - p);
        } else {
            spacing = max(length(curves_out[base + k - 1u].xyz - p),
                          length(curves_out[base + k + 1u].xyz - p));
        }
        let threshold = 1.1 * max(spacing, PROBE_STEP);
        curves_out[base + k].w = surfaceClearance(p, n, params.eye, threshold, params.time);
    }
}
`;
