/**
 * Link compute shader — best-neighbor selection along the principal
 * direction, canonicalized by target, plus per-link curve emission as
 * sampled cubic Hermite segments with C1 chain-window tangents.
 *
 * One module, four entry points. The first three are dispatched in
 * order within a single compute pass by `LinkPipeline` (WebGPU orders
 * dispatches within a pass, so each stage's writes are visible to the
 * next) on frames where the simulation stepped; the fourth runs every
 * frame (its input — the eye — changes with the camera even while
 * time is paused):
 *
 *  1. `link_owner_clear_cs` — clears the `owners` buffer to `UINT_MAX`.
 *  2. `link_cs` — one invocation per point `i`. The 27-cell neighborhood
 *     of point `i`'s grid cell is scanned exactly as in the relax shader
 *     (same cell recomputation, same `UINT_MAX` empty-cell sentinel, same
 *     thin-wall filter). For each candidate neighbor `j` within
 *     `LINK_RADIUS`:
 *
 *      - pairs closer than `COINCIDENT_EPS` are skipped (the direction
 *        `diff / dist` is undefined at `dist == 0`)
 *      - pairs whose midpoint is deep inside the surface with opposing
 *        normals are skipped (likely opposite sides of a thin wall)
 *      - the principal direction is the unit vector in the tangent plane
 *        at `p_i` orthogonal to the light direction's tangent projection
 *        — hatch lines run "across the grain" of the lighting; when the
 *        light is nearly parallel to the normal (`|L_tan| < LINK_TAN_EPS`)
 *        no principal direction exists and the point gets no link
 *      - only the forward hemisphere is considered (`alignment > 0`) and
 *        the score `alignment - DISTANCE_WEIGHT * dist` trades alignment
 *        against proximity; the highest-scoring neighbor wins
 *
 *     Writes `links[i] = best_j` (or `UINT_MAX`) and, for valid links,
 *     elects the owner via `atomicMin(&owners[best_j], i)` — after the
 *     pass, `owners[j]` is the smallest source index targeting `j`.
 *     This is the final write to `links`: the buffer stays the raw
 *     scan output, which makes the subdivision's chain-window reads
 *     race-free; every consumer evaluates the survival predicate from
 *     `links` + `owners` (decision 11 in the hatch-lines spec).
 *  3. `link_dedup_subdiv_cs` — one invocation per point. Dedup:
 *     evaluates the edge-survival predicate — `links[i]` survives when
 *     `owners[links[i]] == i`, so every point keeps at most one incoming
 *     edge (decision 9 in the hatch-lines spec) and the surviving graph
 *     is a union of paths and cycles — and additionally breaks mutual
 *     2-cycles (i→j and j→i that both survived dedup): when
 *     `owners[j] == i` and `owners[i] == j` and `i > j`, the
 *     larger-index edge is dropped. The predicate gates curve emission;
 *     `links` itself is not mutated here (stage 2 made the final
 *     write). Subdiv: for every surviving link, samples a cubic Hermite
 *     segment between the two stipple endpoints (Step 2c / decision 11
 *     in the hatch-lines spec): the tangent at each endpoint is the
 *     chain-window average over the vertex's `{previous, self, next}`
 *     stipple points — the predecessor comes from `owners`, the
 *     successor from the raw `links` slot of the target, and both
 *     candidates are validated against the dedup rules so chain ends
 *     and dropped edges fall back to one-sided chord tangents. Because
 *     the tangent at a shared chain vertex is a pure function of that
 *     vertex's window, both adjacent segments compute identical tangent
 *     vectors and the chain of curve segments is C1. Interior samples
 *     are Newton-projected onto the surface; the endpoints are the
 *     stipple points themselves and are written exactly, so adjacent
 *     segments join seamlessly. Writes `STRIDE` curve points to
 *     `curves_out` (`.xyz` = position, `.w` = validity placeholder —
 *     1.0 valid, 0.0 dropped or absent link — overwritten by the
 *     clearance stage below); dropped or absent links zero their
 *     slice — every slice is rewritten each playing frame, since the
 *     buffer persists across frames and keeps rendering while time is
 *     paused.
 *  4. `link_clearance_cs` — one invocation per point, dispatched every
 *     frame (view-dependent). Evaluates the same edge-survival
 *     predicate as stage 3 (from the finalized `links` + `owners`
 *     state) and, for surviving links, traces one `rayClearance` per
 *     curve point toward the eye — origin biased along the surface
 *     normal, back-facing points short-circuited to an explicitly
 *     occluded value and writes the clearance into `.w`, replacing the
 *     validity placeholder. Each curve point's clearance threshold is
 *     its local curve spacing (the longest adjacent sub-segment length,
 *     floored at PROBE_STEP), so the trace's origin ignore-band covers
 *     the curve's own surface neighborhood. Endpoint normals come from
 *     the shared normals buffer; interior samples get a fresh
 *     `sdfGradient` normal. Dropped or
 *     absent links reset their slice's `.w` to 0.0 — a sentinel that
 *     is both "invalid" and "fully occluded", which renders nothing.
 *     Because this stage runs every frame, the link render's
 *     occlusion culling and soft fade stay view-dependent while time
 *     is paused.
 *
 * `GridParams` is duplicated here (WGSL has no cross-module struct
 * import) — it matches the layout in `cell-index-shader.ts` and
 * `cell-ranges-shader.ts` byte-for-byte.
 */
import { SDF_COMMON } from "./sdf-common.js";

/**
 * Hermite sub-segment count per link — the single source of truth for
 * this value.
 *
 * Structural constant, not a tunable: it sizes the curve buffer and
 * the render instance count, which live on the TypeScript side, so it
 * is defined here (beside the shader that consumes it) and
 * interpolated into both the link compute shader and the link render
 * shader (decision 8 in the hatch-lines spec). Each surviving link is
 * emitted as `LINK_N_SEGMENTS + 1` curve points — the two stipple
 * endpoints plus `LINK_N_SEGMENTS - 1` interior Hermite samples. The
 * curve's shape is fixed by the chain-window tangents and
 * `LINK_TANGENT_SCALE`; this value only sets tessellation fidelity
 * (decision 11).
 */
export const LINK_N_SEGMENTS = 4;

export const LINK_SHADER = /* wgsl */ `
${SDF_COMMON}

struct LinkParams {
    eye: vec3f,
    time: f32,
    point_count: u32,
    _pad: u32,
};

struct GridParams {
    bbox_min: vec3f,
    cell_size: f32,
    bbox_max: vec3f,
    point_count: u32,
    grid_dims: vec3u,
    _pad: u32,
};

const UINT_MAX: u32 = 0xFFFFFFFFu;

const DISTANCE_WEIGHT: f32 = 0.85;
const LINK_RADIUS: f32 = 0.3;

/**
 * Below this tangent-projection magnitude the light is considered
 * parallel to the normal: the projected direction degenerates and no
 * principal direction (hence no link) is well-defined.
 */
const LINK_TAN_EPS: f32 = 1e-4;

/**
 * Below this separation the neighbor direction is numerically undefined
 * ('diff / dist' with dist == 0). Same threshold as the relax shader's
 * coincident-pair guard.
 */
const COINCIDENT_EPS: f32 = 1e-7;

/**
 * Hermite sub-segment count per link — interpolated from the TS
 * constant LINK_N_SEGMENTS (single source of truth; decision 8 in the
 * hatch-lines spec). Each surviving link is emitted as N_SEGMENTS + 1
 * curve points — the two stipple endpoints plus N_SEGMENTS - 1
 * interior Hermite samples. The curve's shape is fixed by the
 * chain-window tangents and LINK_TANGENT_SCALE and does not depend on
 * this value (decision 11 in the hatch-lines spec).
 */
const N_SEGMENTS: u32 = ${LINK_N_SEGMENTS}u;

/** Curve points per link (N_SEGMENTS + 1). */
const STRIDE: u32 = N_SEGMENTS + 1u;

/**
 * Scale of the Hermite segment endpoint tangent magnitudes, relative to
 * the mean incident chord length of the endpoint's chain window.
 * Higher values round the joints more broadly; lower values hug the
 * chords. Part of the window-pure tangent rule (decision 11 in the
 * hatch-lines spec).
 */
const LINK_TANGENT_SCALE: f32 = 1.0;

/** Newton iterations per curve-point projection. */
const LINK_NEWTON_ITERS: i32 = 2;

/** Newton step size. */
const LINK_NEWTON_ALPHA: f32 = 1.0;

@group(0) @binding(0) var<uniform> params: LinkParams;
@group(0) @binding(1) var<storage, read> points_in: array<Point>;
@group(0) @binding(2) var<storage, read> normals_in: array<vec4f>;
@group(0) @binding(3) var<uniform> grid_params: GridParams;
@group(0) @binding(4) var<storage, read> sorted_values: array<u32>;
@group(0) @binding(5) var<storage, read> cell_start: array<u32>;
@group(0) @binding(6) var<storage, read> cell_count: array<u32>;
@group(0) @binding(7) var<storage, read_write> links_out: array<u32>;
@group(0) @binding(8) var<storage, read_write> owners_elect: array<atomic<u32>>;

/**
 * Owner buffer alias for the clear entry — same GPU buffer as
 * owners_elect, bound at a slot unused by the scan. Bindings may
 * repeat across entry points: each pipeline derives its layout from its
 * own entry point's statically used resources.
 */
@group(0) @binding(1) var<storage, read_write> owners_reset: array<atomic<u32>>;

/**
 * Owner buffer alias for the dedup+subdiv entry — read-only plain view
 * of the same GPU buffer (the atomic writes of the election pass are
 * visible after the inter-dispatch ordering within the compute pass).
 */
@group(0) @binding(8) var<storage, read> owners_in: array<u32>;

/**
 * Links buffer alias for the dedup+subdiv entry — read-only view of the
 * same GPU buffer the scan writes. The scan performs the final write to
 * the buffer, so every read in this entry (the own slot and, for the
 * chain-window successor, the target's slot) is race-free.
 */
@group(0) @binding(7) var<storage, read> links_in: array<u32>;

/**
 * Subdivided curve points, written by link_dedup_subdiv_cs (positions
 * plus a validity placeholder in the .w component) and by
 * link_clearance_cs (per-curve-point occlusion clearance in .w, toward
 * the eye: > 0 visible, < 0 occluded, 0.0 dropped or absent link). The
 * other entry points do not statically use it, so it is absent from
 * their derived bind group layouts.
 */
@group(0) @binding(9) var<storage, read_write> curves_out: array<vec4f>;

@compute @workgroup_size(64)
fn link_owner_clear_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }
    atomicStore(&owners_reset[i], UINT_MAX);
}

@compute @workgroup_size(64)
fn link_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let p_i = points_in[i].pos.xyz;
    let n_i = normals_in[i].xyz;

    // Principal direction: orthogonal to the light's tangent projection.
    let L = lightDir(params.time);
    let L_tan = L - dot(L, n_i) * n_i;
    if (length(L_tan) < LINK_TAN_EPS) {
        links_out[i] = UINT_MAX;
        return;
    }
    let principal = normalize(cross(n_i, L_tan));

    // Recompute point i's cell (same formula + clamp as cell-index-shader).
    let cf = floor((p_i - grid_params.bbox_min) / grid_params.cell_size);
    let cf_clamped = clamp(cf, vec3f(0.0), vec3f(grid_params.grid_dims) - vec3f(1.0));
    let c_i = vec3u(cf_clamped);
    let dims_i = vec3i(grid_params.grid_dims);

    var best_j = UINT_MAX;
    var best_score = -1e30;

    // 27-cell neighborhood (3×3×3). The own cell is included via (0,0,0);
    // j == i is skipped in the inner loop.
    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let nc = vec3i(c_i) + vec3i(dx, dy, dz);
                if (any(nc < vec3i(0)) || any(nc >= dims_i)) {
                    continue;
                }
                let nc_u = vec3u(nc);
                let nkey = nc_u.x + grid_params.grid_dims.x * (nc_u.y + grid_params.grid_dims.y * nc_u.z);
                let start = cell_start[nkey];
                if (start == UINT_MAX) {
                    continue;
                }
                let count = cell_count[nkey];
                for (var s: u32 = 0u; s < count; s = s + 1u) {
                    let j = sorted_values[start + s];
                    if (j == i) {
                        continue;
                    }
                    let p_j = points_in[j].pos.xyz;
                    let diff = p_j - p_i;
                    let dist = length(diff);
                    if (dist > LINK_RADIUS) {
                        continue;
                    }
                    if (dist < COINCIDENT_EPS) {
                        continue;
                    }

                    let n_j = normals_in[j].xyz;
                    let p_m = 0.5 * (p_i + p_j);
                    let dist_m = map(p_m, params.time);
                    if (dist_m < -0.05 * LINK_RADIUS && dot(n_i, n_j) < -0.2) {
                        // points likely on opposite sides of a thin wall
                        continue;
                    }

                    let alignment = dot(diff / dist, principal);
                    if (alignment <= 0.0) {
                        continue;
                    }
                    let score = alignment - DISTANCE_WEIGHT * dist;
                    if (score > best_score) {
                        best_score = score;
                        best_j = j;
                    }
                }
            }
        }
    }

    links_out[i] = best_j;

    // Owner election: the smallest source index targeting best_j wins.
    // The dedup+subdiv dispatch evaluates this election to gate its
    // curve emission (decision 9 in the hatch-lines spec); links itself
    // is not written again.
    if (best_j != UINT_MAX) {
        atomicMin(&owners_elect[best_j], i);
    }
}

/**
 * Averaged chain tangent at a chain vertex, computed purely from the
 * vertex's three-point window. Both adjacent segments evaluate this
 * with the same three stipple points, so the tangent vectors match and
 * the chain of curve segments is C1 (decision 11 in the hatch-lines
 * spec). Direction: the normalized sum of the two incident unit chord
 * directions — unit chords are averaged before normalizing, so uneven
 * stipple spacing cannot overshoot. Magnitude: LINK_TANGENT_SCALE times
 * the mean incident chord length. Guards are window-pure, so C1
 * survives them: one chord degenerate → follow the other; chords
 * near-antiparallel (hairpin) → the window's through-direction; a fully
 * degenerate window → zero tangent (the Hermite reduces to a plain
 * interpolation).
 */
fn windowTangent(pred: vec3f, v: vec3f, succ: vec3f) -> vec3f {
    let d_in = v - pred;
    let d_out = succ - v;
    let l_in = length(d_in);
    let l_out = length(d_out);
    if (l_in < COINCIDENT_EPS) {
        return LINK_TANGENT_SCALE * d_out;
    }
    if (l_out < COINCIDENT_EPS) {
        return LINK_TANGENT_SCALE * d_in;
    }
    let sum = d_in / l_in + d_out / l_out;
    let sum_len = length(sum);
    let mag = LINK_TANGENT_SCALE * 0.5 * (l_in + l_out);
    if (sum_len < 1e-3) {
        let through = succ - pred;
        let l_through = length(through);
        if (l_through < COINCIDENT_EPS) {
            return vec3f(0.0);
        }
        return mag * through / l_through;
    }
    return mag * sum / sum_len;
}

/**
 * Edge-survival predicate for point i's scan output: the raw
 * links_in[i] survives when it won its target's owner election and
 * is not the larger-index edge of a mutual 2-cycle (decision 9 in the
 * hatch-lines spec). Shared by the dedup+subdiv and clearance entry
 * points; reads only the finalized links + owners state, so it is
 * race-free within either dispatch.
 */
fn linkSurvives(i: u32) -> bool {
    let j = links_in[i];
    if (j == UINT_MAX || owners_in[j] != i) {
        return false;
    }
    return !(owners_in[i] == j && i > j);
}

/**
 * Occlusion clearance of a surface point toward the eye: origin biased
 * along the surface normal, back-facing points short-circuited to an
 * explicitly occluded value, rayClearance toward the eye. Non-positive
 * results mean the surface occludes the point from the camera; positive
 * results saturate at threshold (the ray's closest unobstructed
 * approach). The caller supplies the threshold — each curve point uses
 * its local curve spacing.
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
fn link_dedup_subdiv_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    // Dedup: i's edge survives only when it won its target's owner
    // election and is not the dropped larger index of a mutual 2-cycle
    // (linkSurvives). The predicate gates curve emission; links itself
    // is not mutated (the scan made the final write), so the
    // chain-window reads below are race-free.
    let base = i * STRIDE;
    if (!linkSurvives(i)) {
        for (var k: u32 = 0u; k < STRIDE; k = k + 1u) {
            curves_out[base + k] = vec4f(0.0);
        }
        return;
    }

    let j = links_in[i];
    let p_i = points_in[i].pos.xyz;
    let p_j = points_in[j].pos.xyz;

    // Chain-window tangent at the start vertex p_i. Predecessor along
    // the chain: the elected source targeting i. Its edge was dropped
    // when it is the larger index of a mutual pair — then i is a chain
    // start and the tangent is one-sided.
    var has_pred = false;
    var pred = vec3f(0.0);
    let a = owners_in[i];
    if (a != UINT_MAX && !(owners_in[a] == i && a > i)) {
        has_pred = true;
        pred = points_in[a].pos.xyz;
    }
    // One-sided fallback tangent: the link chord itself,
    // LINK_TANGENT_SCALE-scaled. The same chord vector serves both
    // chain-end cases — the start of the first segment (pointing toward
    // the far endpoint) and the end of the last segment (pointing
    // outward past the far endpoint).
    let chord_tangent = LINK_TANGENT_SCALE * (p_j - p_i);
    let t_i = select(chord_tangent, windowTangent(pred, p_i, p_j), has_pred);

    // Chain-window tangent at the end vertex p_j. Successor along the
    // chain: j's raw scan target k. The chain continues only when that
    // edge survived dedup (owners_in[k] == j); k == i is the degenerate
    // double-back window (mutual pair or dropped back-pointer). Every
    // failing case leaves a one-sided tangent.
    var has_succ = false;
    var succ = vec3f(0.0);
    let k = links_in[j];
    if (k != UINT_MAX && k != i && owners_in[k] == j) {
        has_succ = true;
        succ = points_in[k].pos.xyz;
    }
    let t_j = select(chord_tangent, windowTangent(p_i, p_j, succ), has_succ);

    // Sample the Hermite segment; the endpoints are the stipple points
    // themselves (written exactly, so adjacent segments join
    // seamlessly) and interior samples are Newton-projected onto the
    // surface. Every slice is rewritten each frame — the buffer
    // persists across frames and keeps rendering while time is paused.
    for (var s: u32 = 0u; s <= N_SEGMENTS; s = s + 1u) {
        if (s == 0u) {
            curves_out[base + s] = vec4f(p_i, 1.0);
        } else if (s == N_SEGMENTS) {
            curves_out[base + s] = vec4f(p_j, 1.0);
        } else {
            let t = f32(s) / f32(N_SEGMENTS);
            let t2 = t * t;
            let t3 = t2 * t;
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;
            let sample = h00 * p_i + h10 * t_i + h01 * p_j + h11 * t_j;
            let projected = projectToSurface(sample, params.time, LINK_NEWTON_ITERS, LINK_NEWTON_ALPHA);
            curves_out[base + s] = vec4f(projected, 1.0);
        }
    }
}

@compute @workgroup_size(64)
fn link_clearance_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let base = i * STRIDE;
    if (!linkSurvives(i)) {
        // Dropped or absent link: the subdiv stage zeroed the whole
        // slice (positions included); keep the .w sentinel at 0.0 —
        // "invalid", which renders nothing.
        for (var k: u32 = 0u; k < STRIDE; k = k + 1u) {
            curves_out[base + k].w = 0.0;
        }
        return;
    }

    let j = links_in[i];
    for (var k: u32 = 0u; k < STRIDE; k = k + 1u) {
        let p = curves_out[base + k].xyz;
        // Endpoints are the stipple points themselves: reuse their
        // stored normals from the shared normals buffer. Interior
        // samples get a fresh surface normal.
        var n: vec3f;
        if (k == 0u) {
            n = normals_in[i].xyz;
        } else if (k == N_SEGMENTS) {
            n = normals_in[j].xyz;
        } else {
            n = normalize(sdfGradient(p, params.time));
        }
        // Clearance threshold: the local curve spacing — the longest of
        // the adjacent sub-segments (single neighbor at the chain ends),
        // so the trace's origin ignore-band covers the curve's own
        // surface neighborhood around this sample. Floored at
        // PROBE_STEP: a degenerate zero spacing would make an
        // unobstructed trace return 0.0 — the dropped/absent-link
        // sentinel the renderer culls on.
        var spacing: f32;
        if (k == 0u) {
            spacing = length(curves_out[base + 1u].xyz - p);
        } else if (k == N_SEGMENTS) {
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
