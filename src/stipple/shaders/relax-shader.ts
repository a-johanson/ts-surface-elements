/**
 * Relax compute shader — grid-accelerated surface repulsion.
 *
 * One invocation per point `i`. The O(n²) neighbor scan is replaced by a
 * 27-cell uniform-grid neighborhood lookup (Step 7): point `i`'s cell is
 * recomputed from its position + `GridParams` (same formula + clamp as the
 * cell-index shader), then the 3×3×3 neighborhood of that cell is scanned.
 * For each neighbor cell, `cellStart` / `cellCount` give an O(1) range into
 * `sortedValues`; empty cells (`cellStart == UINT_MAX`) are skipped. This
 * reduces the inner loop from O(n) to O(1) neighbor cells × O(points/cell).
 *
 * Pairwise force model: a Euclidean distance cutoff gates the force, and a
 * linear-decay envelope modulates its magnitude:
 *
 *  - d_E == 0 (bit-identical) → sign-based kick along a tangent basis vector
 *  - d_E > radius             → skip
 *  - decay  (1 - d_E / radius)
 *  - direction (p_i - p_j) / d_E
 *
 * Coincident pairs (d_E < COINCIDENT_EPS, including the bit-identical case
 * d_E == 0.0) receive a deterministic sign-based kick along a tangent
 * basis vector instead of the normal repulsion (diff/d_E is undefined at
 * d_E == 0). The kick is antisymmetric under i↔j swap (sign(i - j)
 * flips), so the two points separate on the tangent plane; projectToSurface
 * then re-projects, and the next substep's normal repulsion resumes.
 *
 * The accumulated force is projected onto the tangent plane at `p_i` and
 * integrated with a direct Euler position step `x* = x + dt·F_tan`, where
 * `dt` is the per-frame capped wall-clock delta threaded from the frame
 * loop (not a constant). The new position is re-projected onto the
 * surface with a few Newton steps; since per-frame drift is small,
 * `RELAX_NEWTON_ITERS = 4` with `alpha = 1` suffices.
 *
 * Grid buffer bindings: the grid is rebuilt once per frame (Step 6) from
 * whichever point buffer relax is about to read, so the sorted values and
 * cell ranges always match the current positions regardless of the
 * ping-pong direction. The grid buffers are read-only during relax and
 * are not ping-ponged — both A→B and B→A bind groups bind the same grid
 * buffer set. `sortedKeys` is not bound here: the cell ranges already
 * encode which index ranges belong to which cell, so only `sortedValues`
 * (point indices) is needed to fetch neighbors.
 *
 * `GridParams` is duplicated here (WGSL has no cross-module struct
 * import) — it matches the layout in `cell-index-shader.ts` and
 * `cell-ranges-shader.ts` byte-for-byte.
 */
import { SDF_COMMON } from "./sdf-common.js";

export const RELAX_SHADER = /* wgsl */ `
${SDF_COMMON}

struct RelaxParams {
    dt: f32,
    radius: f32,
    time: f32,
    point_count: u32,
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

@group(0) @binding(0) var<uniform> params: RelaxParams;
@group(0) @binding(1) var<storage, read> points_in: array<Point>;
@group(0) @binding(2) var<storage, read_write> points_out: array<Point>;
@group(0) @binding(3) var<storage, read> normals_in: array<vec4f>;
@group(0) @binding(4) var<uniform> grid_params: GridParams;
@group(0) @binding(5) var<storage, read> sorted_values: array<u32>;
@group(0) @binding(6) var<storage, read> cell_start: array<u32>;
@group(0) @binding(7) var<storage, read> cell_count: array<u32>;

const RELAX_NEWTON_ITERS: i32 = 4;
const RELAX_ALPHA: f32 = 1.0;

/**
 * Below this separation two points are considered bit-identical and a
 * deterministic sign-based kick is applied instead of the normal
 * repulsion (diff/d_E is undefined at d_E == 0). One substep at
 * dt ≈ 0.01 separates the pair by ~0.01, clearing this epsilon so the
 * next substep resumes normal repulsion.
 */
const COINCIDENT_EPS: f32 = 1e-7;

@compute @workgroup_size(64)
fn relax_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let p_i = points_in[i].pos.xyz;
    let n_i = normals_in[i].xyz;

    // Recompute point i's cell (same formula + clamp as cell-index-shader).
    let cf = floor((p_i - grid_params.bbox_min) / grid_params.cell_size);
    let cf_clamped = clamp(cf, vec3f(0.0), vec3f(grid_params.grid_dims) - vec3f(1.0));
    let c_i = vec3u(cf_clamped);
    let dims_i = vec3i(grid_params.grid_dims);

    var force = vec3f(0.0);
    let r = params.radius;
    let time = params.time;

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
                    let diff = p_i - p_j;
                    let d_E = length(diff);
                    if (d_E > r) {
                        continue;
                    }

                    if (d_E < COINCIDENT_EPS) {
                        // Bit-identical pair: diff/d_E is undefined. Apply
                        // a deterministic sign-based kick along a tangent
                        // basis vector. sign(i - j) is antisymmetric under
                        // i↔j swap, so the two points separate on the
                        // tangent plane; projectToSurface then re-projects.
                        let up = mix(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0),
                                     f32(abs(n_i.y) > 0.99));
                        let t1 = normalize(cross(up, n_i));
                        let s = select(-1.0, 1.0, i > j);
                        force = force + s * t1;
                        continue;
                    }

                    force = force + (1.0 - d_E / r) * diff / d_E;
                }
            }
        }
    }

    force = force - dot(force, n_i) * n_i;

    let drifted = p_i + params.dt * force;
    let new_pos = projectToSurface(drifted, time, RELAX_NEWTON_ITERS, RELAX_ALPHA);

    points_out[i].pos = vec4f(new_pos, 0.0);
}
`;
