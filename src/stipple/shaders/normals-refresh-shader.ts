/**
 * Normals-refresh compute shader — refreshes the shared normals buffer
 * from the post-relax positions.
 *
 * One invocation per point. Runs on frames where the simulation stepped,
 * after the relax pass, reading whichever point buffer relax most
 * recently wrote, and writes the surface normal at that (now final)
 * position into the shared normals buffer — so the normals buffer always
 * matches the buffer that the *next* reproject + relax cycle will read
 * (preserving the relax invariant that normals match the read-side
 * buffer across the ping-pong swap) and the link pass reads normals
 * matching the post-relax positions.
 *
 * The pass is view-independent — no eye, no per-point visibility
 * clearance; the link pipeline owns its own per-curve-point clearance
 * traces.
 */
import { SDF_COMMON } from "./sdf-common.js";

export const NORMALS_REFRESH_SHADER = /* wgsl */ `
${SDF_COMMON}

struct NormalsRefreshParams {
    time: f32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: NormalsRefreshParams;
@group(0) @binding(1) var<storage, read> points_in: array<Point>;
@group(0) @binding(2) var<storage, read_write> normals_out: array<vec4f>;

@compute @workgroup_size(64)
fn normals_refresh_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let time = params.time;
    let p = points_in[i].pos.xyz;
    normals_out[i] = vec4f(calcNormal(p, time), 0.0);
}
`;
