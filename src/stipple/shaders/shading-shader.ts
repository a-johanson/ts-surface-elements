/**
 * Shading compute shader — per-point visibility, shadow, and normal
 * refresh.
 *
 * One invocation per point. Runs every frame after the relax pass, reading
 * whichever point buffer relax most recently wrote. Computes the surface
 * normal from the (now final) position and writes it into the shared
 * normals buffer — so the normals buffer always matches the buffer that
 * the *next* reproject + relax cycle will read (preserving the relax
 * invariant that normals match the read-side buffer across the ping-pong
 * swap).
 *
 * Visibility: sphere-traces from `p + n·bias` toward the eye. The point
 * is visible iff the march either misses (`t < 0`) or reaches the eye
 * without hitting the surface (`t >= dist_to_eye`).
 *
 * Shadow: Aaltonen soft-shadow penumbra estimation from `p + n·bias`
 * toward the light direction, returning a `[0, 1]` factor that modulates
 * the Lambert term to produce graded penumbras instead of a binary
 * shadow cut.
 *
 * Output packing (`shading_out[i]`, `vec4f`):
 *  - `x` — visibility (`1.0` visible, `0.0` occluded).
 *  - `y` — luminance (`visible * lit * lambert`, in `[0, 1]`).
 *  - `z`, `w` — unused.
 *
 * The point renderer reads this buffer to discard occluded points in the
 * vertex shader and modulate the fragment color by luminance.
 */
import { SDF_COMMON } from "./sdf-common.js";

export const SHADING_SHADER = /* wgsl */ `
${SDF_COMMON}

struct ShadingParams {
    eye: vec3f,
    point_count: u32,
    light_dir: vec3f,
    time: f32,
};

@group(0) @binding(0) var<uniform> params: ShadingParams;
@group(0) @binding(1) var<storage, read> points_in: array<Point>;
@group(0) @binding(2) var<storage, read_write> normals_out: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> shading_out: array<vec4f>;

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

    let origin = p + n * SHADOW_BIAS;

    let to_eye = params.eye - origin;
    let dist_eye = length(to_eye);
    let eye_dir = to_eye / dist_eye;

    if (dot(n, eye_dir) <= 0.0) {
        shading_out[i] = vec4f(0.0);
        return;
    }

    let t_eye = rayMarch(origin, eye_dir, time);
    if (t_eye >= 0.0 && t_eye < dist_eye) {
        shading_out[i] = vec4f(0.0);
        return;
    }

    let lit = softShadow(origin, params.light_dir, time);
    let lambert = max(dot(n, params.light_dir), 0.0);
    shading_out[i] = vec4f(1.0, lit * lambert, 0.0, 0.0);
}
`;
