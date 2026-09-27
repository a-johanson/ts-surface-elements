/**
 * WGSL subgroup scan helpers, reused by the radix split (Step 4).
 *
 * This module is the reusable foundation for the single-workgroup radix sort's
 * prefix-sum step. It provides `workgroupExclusiveScanU32`: a workgroup-wide
 * exclusive prefix sum that runs `subgroupExclusiveAdd` per subgroup, stores
 * subgroup partials in `var<workgroup>`, performs a `workgroupBarrier`,
 * computes the cross-subgroup prefix (thread 0 sequentially scans the
 * partials array), barriers again, and combines with each lane's
 * subgroup-exclusive result.
 *
 * `enable subgroups;` is the first token of the string. When interpolating
 * into a shader, `SUBGROUP_COMMON` must appear before any other declaration
 * (WGSL requires directives to precede declarations). The workgroup variable
 * and helpers are only valid in compute shaders.
 *
 * Subgroup builtins (`subgroup_invocation_id`, `subgroup_size`,
 * `subgroup_id`, `num_subgroups`, `local_invocation_index`) can only be read
 * in entry-point functions, so `workgroupExclusiveScanU32` receives them as
 * plain `u32` parameters — the consuming entry point passes them down.
 */

export const SUBGROUP_COMMON = /* wgsl */ `
enable subgroups;

const WORKGROUP_SIZE: u32 = 256u;
const MAX_SUBGROUPS: u32 = 64u;

var<workgroup> w_subgroup_partials: array<u32, MAX_SUBGROUPS>;

fn workgroupExclusiveScanU32(
    v: u32,
    sub_id: u32,
    sub_size: u32,
    sub_gid: u32,
    num_subs: u32,
    local_index: u32,
) -> u32 {
    let sub_excl = subgroupExclusiveAdd(v);
    let sub_incl = sub_excl + v;

    if (sub_id == sub_size - 1u) {
        w_subgroup_partials[sub_gid] = sub_incl;
    }
    workgroupBarrier();

    if (local_index == 0u) {
        var running: u32 = 0u;
        for (var s: u32 = 0u; s < num_subs; s = s + 1u) {
            let val = w_subgroup_partials[s];
            w_subgroup_partials[s] = running;
            running = running + val;
        }
    }
    workgroupBarrier();

    return sub_excl + w_subgroup_partials[sub_gid];
}
`;
