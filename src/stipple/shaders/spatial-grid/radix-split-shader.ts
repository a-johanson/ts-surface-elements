/**
 * Radix-split compute shader — the core primitive of the single-workgroup
 * binary radix sort (Step 4).
 *
 * One workgroup of `SUBGROUP_WORKGROUP_SIZE` (256) invocations, each owning
 * `ELEMENTS_PER_THREAD = ceil(POINT_COUNT / 256)` contiguous elements. For a
 * single binary split at bit position `b` (passed via the `RadixBitParams`
 * uniform), the shader computes a stable partition: all elements whose key
 * has a 0 at bit `b` are moved to the front (preserving order), followed by
 * all elements whose key has a 1 (preserving order).
 *
 * Algorithm (two-level exclusive prefix sum):
 *   1. Per thread: sequentially scan `pred = (key bit == 0) ? 1 : 0` over the
 *      thread's `ELEMENTS_PER_THREAD` elements, yielding a per-thread
 *      `local_excl[e]` and a `thread_total` (inclusive sum).
 *   2. Workgroup-wide: `workgroupExclusiveScanU32(thread_total)` (Step 2
 *      helper) gives each thread's offset within the workgroup-wide count of
 *      zero-bit elements.
 *   3. Thread 255 broadcasts `totalZeros = thread_excl + thread_total` via a
 *      `var<workgroup>` scalar + one `workgroupBarrier` (three barriers total
 *      per pass: two inside the scan helper, one for the broadcast).
 *   4. Scatter: for element at global index `g`, `excl = thread_excl +
 *      local_excl[e]` is the global exclusive prefix of `pred` at `g`. A
 *      zero-bit element goes to `dest = excl`; a one-bit element goes to
 *      `dest = (g - excl) + totalZeros`. Keys and values are scattered
 *      together from the read pair into the write pair.
 *
 * Stability: zero-bit destinations `excl` are monotonic in the scan, and
 * one-bit destinations `(g - excl)` are monotonic in `g`, so both halves
 * preserve input order.
 *
 * Elements with `g >= POINT_COUNT` (only possible when `POINT_COUNT` is not a
 * multiple of 256) get `pred = 0` and skip the scatter, so any point count is
 * handled without wasted storage writes.
 *
 * Keys and values are cached in per-thread local arrays during the scan pass
 * so the scatter pass does not re-read the input storage buffers. Sort data
 * lives in storage buffers (not workgroup memory) — bounded by per-thread
 * sequential work, not the 16 KB workgroup storage limit (decision 2).
 *
 * This shader is built by {@link buildRadixSplitShader} with `POINT_COUNT` and
 * `ELEMENTS_PER_THREAD` interpolated as WGSL consts (local arrays require a
 * compile-time size). The driver (Step 5) records 32 per-bit passes,
 * alternating the A/B bind group; 32 is even so the sorted result lands back
 * in the A pair.
 */
import { SUBGROUP_COMMON, SUBGROUP_WORKGROUP_SIZE } from "../subgroup-common.js";

/**
 * Builds the radix-split compute shader for a given point count.
 *
 * @param pointCount - The total number of points (keys/values). Elements with
 *   `g >= pointCount` are no-ops, so non-multiples of 256 are handled.
 * @param elementsPerThread - Elements owned per thread =
 *   `ceil(pointCount / SUBGROUP_WORKGROUP_SIZE)`. Sized exactly so per-thread
 *   local arrays have no waste.
 * @returns The WGSL source string, with `SUBGROUP_COMMON`,
 *   `POINT_COUNT`, and `ELEMENTS_PER_THREAD` interpolated.
 */
export function buildRadixSplitShader(pointCount: number, elementsPerThread: number): string {
    return /* wgsl */ `
${SUBGROUP_COMMON}

const POINT_COUNT: u32 = ${pointCount}u;
const ELEMENTS_PER_THREAD: u32 = ${elementsPerThread}u;

struct RadixBitParams {
    bit: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: RadixBitParams;
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read> values_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> values_out: array<u32>;

var<workgroup> w_total_zeros: u32;

@compute @workgroup_size(${SUBGROUP_WORKGROUP_SIZE})
fn radix_split_cs(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(subgroup_invocation_id) sub_id: u32,
    @builtin(subgroup_size) sub_size: u32,
    @builtin(subgroup_id) sub_gid: u32,
    @builtin(num_subgroups) num_subs: u32,
) {
    var local_keys: array<u32, ELEMENTS_PER_THREAD>;
    var local_values: array<u32, ELEMENTS_PER_THREAD>;
    var local_excl: array<u32, ELEMENTS_PER_THREAD>;

    var running: u32 = 0u;
    for (var e: u32 = 0u; e < ELEMENTS_PER_THREAD; e = e + 1u) {
        let g = local_index * ELEMENTS_PER_THREAD + e;
        var k: u32 = 0u;
        var v: u32 = 0u;
        var p: u32 = 0u;
        if (g < POINT_COUNT) {
            k = keys_in[g];
            v = values_in[g];
            let bit = params.bit;
            p = select(0u, 1u, ((k >> bit) & 1u) == 0u);
        }
        local_keys[e] = k;
        local_values[e] = v;
        local_excl[e] = running;
        running = running + p;
    }
    let thread_total = running;

    let thread_excl = workgroupExclusiveScanU32(
        thread_total, sub_id, sub_size, sub_gid, num_subs, local_index,
    );

    if (local_index == WORKGROUP_SIZE - 1u) {
        w_total_zeros = thread_excl + thread_total;
    }
    workgroupBarrier();
    let total_zeros = w_total_zeros;

    for (var e2: u32 = 0u; e2 < ELEMENTS_PER_THREAD; e2 = e2 + 1u) {
        let g2 = local_index * ELEMENTS_PER_THREAD + e2;
        if (g2 >= POINT_COUNT) {
            continue;
        }
        let excl = thread_excl + local_excl[e2];
        let k2 = local_keys[e2];
        let is_zero = ((k2 >> params.bit) & 1u) == 0u;
        let dest = select(g2 - excl + total_zeros, excl, is_zero);
        keys_out[dest] = k2;
        values_out[dest] = local_values[e2];
    }
}
`;
}
