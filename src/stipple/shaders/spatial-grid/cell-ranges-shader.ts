/**
 * Cell-range compute shader — builds the `cellStart` / `cellCount` lookup
 * table over the sorted keys (Step 6).
 *
 * Two entry points share one module and one set of module-scope bindings:
 *
 *   • `cell_clear_cs` — one invocation per cell. Resets `cell_start[c]` to
 *     `UINT_MAX` and `cell_count[c]` to `0`. Non-atomic writes: exactly one
 *     invocation owns each cell, so there is no race. `UINT_MAX` is the
 *     sentinel relax (Step 7) tests to skip empty cells. References only
 *     `params`, `cell_start`, `cell_count` (binding 1 — `sorted_keys` — is
 *     unused, so the clear pipeline's auto-derived layout omits it).
 *
 *   • `cell_ranges_cs` — one invocation per point `i`, run after the clear
 *     pass. Because the keys are sorted, all points sharing a cell are
 *     contiguous, so the first occurrence of a key marks the cell's start
 *     range. If `i == 0` or `sorted_keys[i] != sorted_keys[i - 1]`, this
 *     invocation writes `cell_start[sorted_keys[i]] = i` (non-atomic — only
 *     one invocation owns each cell's first index). Every invocation then
 *     `atomicAdd(&cell_count[sorted_keys[i]], 1)` to tally the cell's
 *     membership.
 *
 * `cell_clear_cs` must complete before `cell_ranges_cs` over the same
 * `cell_start` / `cell_count` storage. The driver records both dispatches in
 * a single compute pass in that order; in-pass dispatch ordering and the
 * storage-buffer coherence rules guarantee visibility.
 *
 * Reads from `sorted_keys` (the buffer pair the radix sort actually landed
 * in — A if the pass count is even, B if odd; see decision 8) rather than
 * a hardcoded `keysA`, so an odd pass count is handled correctly.
 *
 * `GridParams` is duplicated here (WGSL has no cross-module struct
 * import) — it matches the layout in `cell-index-shader.ts` byte-for-byte.
 */
export const CELL_RANGES_SHADER = /* wgsl */ `
struct GridParams {
    bbox_min: vec3f,
    cell_size: f32,
    bbox_max: vec3f,
    point_count: u32,
    grid_dims: vec3u,
    _pad: u32,
};

const UINT_MAX: u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var<uniform> params: GridParams;
@group(0) @binding(1) var<storage, read> sorted_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> cell_start: array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_count: array<u32>;

@compute @workgroup_size(64)
fn cell_clear_cs(@builtin(global_invocation_id) gid: vec3u) {
    let num_cells = params.grid_dims.x * params.grid_dims.y * params.grid_dims.z;
    let c = gid.x;
    if (c >= num_cells) {
        return;
    }
    cell_start[c] = UINT_MAX;
    cell_count[c] = 0u;
}

@compute @workgroup_size(64)
fn cell_ranges_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }
    let k = sorted_keys[i];
    if (i == 0u || sorted_keys[i - 1u] != k) {
        cell_start[k] = i;
    }
    atomicAdd(&cell_count[k], 1u);
}
`;
