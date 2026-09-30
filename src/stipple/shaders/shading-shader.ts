/**
 * Shading compute shader — per-point corner clearances, shadow, and normal
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
 * Occlusion: instead of a binary visibility test at the point center, each
 * quad corner gets an eye-ray sphere-trace via `rayClearance`, returning
 * how closely the ray passes to the occluding surface or how much it
 * penetrated into the surface. The point renderer interpolates these corner
 * values across the splat and discards fragments where the interpolated
 * clearance is negative.
 *
 * Cost control: the center ray is traced first; when its clearance already
 * clears the full splat radius, all four corners copy it and the corner
 * traces are skipped. `softShadow` runs only when at least one corner can
 * still contribute a fragment.
 *
 * Shadow: soft-shadow penumbra estimation from `p + n·bias`
 * toward the light direction, returning a `[0, 1]` factor that modulates
 * the Lambert term to produce graded penumbras instead of a binary
 * shadow cut.
 *
 * Output packing (`shading_out[i]`, `ShadingSample`):
 *  - `luminance.x` — luminance (`lit * lambert`, in `[0, 1]`; `0` when no
 *    corner is visible).
 *  - `clearance` — occlusion clearance per quad corner (in the vertex
 *    shader's corner order), in world units.
 *
 * The point renderer reads this buffer to push fully occluded points
 * offscreen in the vertex shader, clip the ring per fragment against the
 * interpolated clearance, and modulate the fragment color by luminance.
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
@group(0) @binding(3) var<storage, read_write> shading_out: array<ShadingSample>;

const CORNER_TRACE_SKIP: f32 = 1.1;

fn cornerClearance(
    origin: vec3f,
    frame: TangentFrame,
    offset: vec2f,
    eye: vec3f,
    time: f32,
) -> f32 {
    let ro = origin + (offset.x * frame.t1 + offset.y * frame.t2) * POINT_RADIUS_WORLD;
    let to_eye = eye - ro;
    let dist_eye = length(to_eye);
    let eye_dir = to_eye / dist_eye;
    return rayClearance(ro, eye_dir, dist_eye, time);
}

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
        shading_out[i] = ShadingSample(vec4f(0.0), vec4f(0.0));
        return;
    }

    let frame = tangentFrame(n);
    let clearance_center = rayClearance(origin, eye_dir, dist_eye, time);
    var clearance = vec4f(clearance_center);
    if (clearance_center < CORNER_TRACE_SKIP * POINT_RADIUS_WORLD) {
        clearance = vec4f(
            cornerClearance(origin, frame, vec2f(-1.0, -1.0), params.eye, time),
            cornerClearance(origin, frame, vec2f( 1.0, -1.0), params.eye, time),
            cornerClearance(origin, frame, vec2f(-1.0,  1.0), params.eye, time),
            cornerClearance(origin, frame, vec2f( 1.0,  1.0), params.eye, time),
        );
    }

    let clearance_min = min(min(clearance.x, clearance.y), min(clearance.z, clearance.w));
    if (clearance_min > 0.0) {
        clearance = vec4f(2.0 * CLEARANCE_THRESHOLD);
    } else {
        let clearance_max = max(max(clearance.x, clearance.y), max(clearance.z, clearance.w));
        if (clearance_max <= 0.0) {
            shading_out[i] = ShadingSample(vec4f(0.0), vec4f(0.0));
            return;
        }
    }

    let lit = softShadow(origin, params.light_dir, time);
    let lambert = max(dot(n, params.light_dir), 0.0);
    shading_out[i] = ShadingSample(vec4f(lit * lambert, 0.0, 0.0, 0.0), clearance);
}
`;
