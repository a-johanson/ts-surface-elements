/**
 * Workgroup size for the single-workgroup radix sort.
 *
 * Single source of truth: the TS constant is interpolated into the WGSL so
 * the dispatch math in `SpatialGridPipeline` and the `@workgroup_size`
 * attribute in the radix-split shader cannot drift. The radix sort runs as
 * one workgroup of this many invocations, each owning
 * `ceil(POINT_COUNT / RADIX_WORKGROUP_SIZE)` elements.
 */
export const RADIX_WORKGROUP_SIZE = 256;

/**
 * WGSL workgroup scan helper, reused by the radix split (Step 4).
 *
 * This module is the reusable foundation for the single-workgroup radix
 * sort's prefix-sum step. It provides `workgroupExclusiveScanU32`: a
 * workgroup-wide exclusive prefix sum that stores each invocation's value
 * in `var<workgroup>`, performs a `workgroupBarrier`, converts the array to
 * an exclusive prefix in place via a sequential scan by invocation 0,
 * barriers again, and reads the invocation's offset back.
 *
 * When interpolating into a shader, `WORKGROUP_COMMON` must appear before
 * the consuming shader's declarations. The workgroup variable and helper are
 * only valid in compute shaders.
 *
 * `local_invocation_index` can only be read in entry-point functions, so
 * `workgroupExclusiveScanU32` receives it as a plain `u32` parameter — the
 * consuming entry point passes it down.
 */
export const WORKGROUP_COMMON = /* wgsl */ `
const WORKGROUP_SIZE: u32 = ${RADIX_WORKGROUP_SIZE}u;

var<workgroup> w_thread_totals: array<u32, WORKGROUP_SIZE>;

fn workgroupExclusiveScanU32(v: u32, local_index: u32) -> u32 {
    w_thread_totals[local_index] = v;
    workgroupBarrier();

    if (local_index == 0u) {
        var running: u32 = 0u;
        for (var i: u32 = 0u; i < WORKGROUP_SIZE; i += 1u) {
            let val = w_thread_totals[i];
            w_thread_totals[i] = running;
            running = running + val;
        }
    }
    workgroupBarrier();

    return w_thread_totals[local_index];
}
`;
