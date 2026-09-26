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
import { SDF_COMMON } from "./sdf-common.js";

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
