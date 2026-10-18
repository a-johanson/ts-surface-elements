/**
 * Shading compute shader — per-point visibility clearance and normal
 * refresh.
 *
 * One invocation per point. Runs every frame after the relax pass,
 * reading whichever point buffer relax most recently wrote. Computes the
 * surface normal from the (now final) position and writes it into the
 * shared normals buffer — so the normals buffer always matches the buffer
 * that the *next* reproject + relax cycle will read (preserving the relax
 * invariant that normals match the read-side buffer across the ping-pong
 * swap).
 *
 * Visibility: a single center-ray sphere-trace via `rayClearance` toward
 * the eye returns how closely the ray passes to the occluding surface.
 * Consumers (the point renderer's offscreen cull, the link render's
 * endpoint visibility check) test `clearance <= 0`. Back-facing points
 * (normal pointing away from the eye) write an explicitly occluded
 * clearance instead of being traced.
 *
 * The `luminance` field is *not* written here — it is owned by the
 * reproject pass (view-independent, computed before relax for the relax
 * kernel's density modulation). Writes are member-wise so the
 * reproject-written luminance in the same buffer is preserved.
 */
import { SDF_COMMON } from "./sdf-common.js";

export const SHADING_SHADER = /* wgsl */ `
${SDF_COMMON}

struct ShadingParams {
    eye: vec3f,
    point_count: u32,
    time: f32,
};

@group(0) @binding(0) var<uniform> params: ShadingParams;
@group(0) @binding(1) var<storage, read> points_in: array<Point>;
@group(0) @binding(2) var<storage, read_write> normals_out: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> shading_out: array<PointShading>;

@compute @workgroup_size(64)
fn shading_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let time = params.time;
    let p = points_in[i].pos.xyz;
    let n = normalize(sdfGradient(p, time));
    normals_out[i] = vec4f(n, 0.0);

    let p_biased = p + n * SHADOW_BIAS;
    let to_eye = params.eye - p_biased;
    let dist_eye = length(to_eye);
    let eye_dir = to_eye / dist_eye;

    if (dot(n, eye_dir) <= 0.0) {
        // Back-facing: explicitly occluded so the point renderer's
        // offscreen cull and the link render's visibility check cull it.
        shading_out[i].clearance = -1.0;
        return;
    }

    // Center-ray occlusion clearance; non-positive means occluded. The
    // threshold is the stipple points' splat-radius scale.
    let clearance = rayClearance(p_biased, eye_dir, dist_eye, CLEARANCE_THRESHOLD, time);
    shading_out[i].clearance = clearance;
}
`;
