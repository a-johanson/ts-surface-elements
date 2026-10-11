/**
 * Link compute shader — best-neighbor selection along the principal
 * direction.
 *
 * One invocation per point `i`. The 27-cell neighborhood of point `i`'s
 * grid cell is scanned exactly as in the relax shader (same cell
 * recomputation, same `UINT_MAX` empty-cell sentinel, same thin-wall
 * filter). For each candidate neighbor `j` within `LINK_RADIUS`:
 *
 *  - pairs closer than `COINCIDENT_EPS` are skipped (the direction
 *    `diff / dist` is undefined at `dist == 0`)
 *  - pairs whose midpoint is deep inside the surface with opposing
 *    normals are skipped (likely opposite sides of a thin wall)
 *  - the principal direction is the unit vector in the tangent plane at
 *    `p_i` orthogonal to the light direction's tangent projection — hatch
 *    lines run "across the grain" of the lighting; when the light is
 *    nearly parallel to the normal (`|L_tan| < LINK_TAN_EPS`) no
 *    principal direction exists and the point gets no link
 *  - only the forward hemisphere is considered (`alignment > 0`) and the
 *    score `alignment - DISTANCE_WEIGHT * dist` trades alignment against
 *    proximity; the highest-scoring neighbor wins
 *
 * The output is one `u32` per point: the best neighbor index, or
 * `UINT_MAX` when no valid neighbor was found.
 *
 * `GridParams` is duplicated here (WGSL has no cross-module struct
 * import) — it matches the layout in `cell-index-shader.ts` and
 * `cell-ranges-shader.ts` byte-for-byte.
 */
import { SDF_COMMON } from "./sdf-common.js";

export const LINK_SHADER = /* wgsl */ `
${SDF_COMMON}

struct LinkParams {
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

const DISTANCE_WEIGHT: f32 = 1.5;
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

@group(0) @binding(0) var<uniform> params: LinkParams;
@group(0) @binding(1) var<storage, read> points_in: array<Point>;
@group(0) @binding(2) var<storage, read> normals_in: array<vec4f>;
@group(0) @binding(3) var<uniform> grid_params: GridParams;
@group(0) @binding(4) var<storage, read> sorted_values: array<u32>;
@group(0) @binding(5) var<storage, read> cell_start: array<u32>;
@group(0) @binding(6) var<storage, read> cell_count: array<u32>;
@group(0) @binding(7) var<storage, read_write> links_out: array<u32>;

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
}
`;
