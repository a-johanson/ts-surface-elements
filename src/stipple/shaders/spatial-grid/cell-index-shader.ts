/**
 * Cell-index compute shader — assigns each point to a uniform-grid cell.
 *
 * One invocation per point `i`. Reads `points_in[i]`, computes the 3D cell
 * coordinate from the fixed `SCENE_BBOX` origin and `cellSize = radius`,
 * clamps each axis to `[0, gridDims-1]` (decision 6 — prevents OOB storage
 * writes if `SCENE_BBOX` is too small), packs to a linear `u32` key, and
 * writes `keys_out[i] = key`, `values_out[i] = i`.
 *
 * This is the first stage of the spatial grid build. The output `(keysA,
 * valuesA)` pair feeds the radix sort (Step 4), which produces a sorted
 * `(keys, values)` pair consumed by the cell-ranges pass (Step 6) and the
 * relax shader (Step 7).
 *
 * `SDF_COMMON` is interpolated solely for the `Point` storage struct; the
 * SDF functions it carries are unused here and stripped by the compiler.
 */
import { SDF_COMMON } from "../sdf-common.js";

export const CELL_INDEX_SHADER = /* wgsl */ `
${SDF_COMMON}

struct GridParams {
    bbox_min: vec3f,
    cell_size: f32,
    bbox_max: vec3f,
    point_count: u32,
    grid_dims: vec3u,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: GridParams;
@group(0) @binding(1) var<storage, read> points_in: array<Point>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> values_out: array<u32>;

@compute @workgroup_size(64)
fn cell_index_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let p = points_in[i].pos.xyz;
    let cf = floor((p - params.bbox_min) / params.cell_size);
    let cf_clamped = clamp(cf, vec3f(0.0), vec3f(f32(params.grid_dims) - 1.0));
    let c = vec3u(cf_clamped);
    let key = c.x + params.grid_dims.x * (c.y + params.grid_dims.y * c.z);

    keys_out[i] = key;
    values_out[i] = i;
}
`;
