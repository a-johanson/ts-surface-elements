/**
 * WGSL shader source strings for the n-body simulation.
 *
 * Kept separate from pipeline TypeScript so the shader code is readable
 * and syntax-highlightable as WGSL.
 */

/**
 * Compute shader for O(n²) all-pairs gravity integration.
 *
 * Each invocation owns exactly one body `i`:
 *
 * 1. Reads body `i` from the **input** buffer.
 * 2. Loops over all bodies `j ≠ i`, accumulating gravitational acceleration
 *    with Plummer softening (`1 / (r² + ε²)^(3/2)`) to avoid singularities.
 * 3. Performs semi-implicit Euler integration:
 *    `v_new = v_old + a * dt`
 *    `p_new = p_old + v_new * dt`
 * 4. Writes the updated body to the **output** buffer.
 *
 * No invocation writes to another body's slot, so there are no write-write
 * races. Input and output are distinct buffers (ping-pong), so there are
 * no read-write races either.
 *
 * Workgroup size is 64. The host dispatches `ceil(bodyCount / 64)`
 * workgroups. Invocations with `global_invocation_id.x >= bodyCount`
 * return early.
 */
export const COMPUTE_SHADER = /* wgsl */ `
struct Body {
    pos: vec4f,
    vel: vec4f,
};

struct SimParams {
    dt: f32,
    g: f32,
    softening: f32,
    body_count: u32,
};

@group(0) @binding(0) var<storage, read> input_bodies: array<Body>;
@group(0) @binding(1) var<storage, read_write> output_bodies: array<Body>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.body_count) {
        return;
    }

    let body = input_bodies[i];
    var accel = vec3f(0.0, 0.0, 0.0);
    let soft_sq = params.softening * params.softening;

    for (var j: u32 = 0u; j < params.body_count; j = j + 1u) {
        if (j == i) {
            continue;
        }
        let other = input_bodies[j];
        let diff = other.pos.xyz - body.pos.xyz;
        let dist_sq = dot(diff, diff) + soft_sq;
        let inv_dist = 1.0 / sqrt(dist_sq);
        let inv_dist3 = inv_dist * inv_dist * inv_dist;
        accel = accel + params.g * other.pos.w * diff * inv_dist3;
    }

    let new_vel = vec4f(body.vel.xyz + accel * params.dt, body.vel.w);
    let new_pos = vec4f(body.pos.xyz + new_vel.xyz * params.dt, body.pos.w);

    output_bodies[i] = Body(new_pos, new_vel);
}
`;

/**
 * Render shader for instanced billboards.
 *
 * Draws each body as a screen-space-sized quad (two triangles) facing the
 * camera. No vertex buffer is used — quad corners are derived from
 * `vertex_index` (0–5). `instance_index` selects which body to render.
 *
 * The body's world position is projected to clip space via the camera
 * uniform's view-projection matrix. Quad corners are then offset by a
 * fixed NDC delta scaled by `inv_aspect` on the x-axis to produce true
 * circles despite non-square viewports. Discs scale with depth due to
 * the perspective divide (intentional).
 *
 * The fragment shader paints a soft circular disc: fragments outside
 * radius 1.0 are discarded; the alpha falls off smoothly near the edge.
 * Disc color is tinted by the body's radial velocity relative to the
 * camera (Doppler effect): approaching bodies shift blue, receding
 * bodies shift red.
 */
export const RENDER_SHADER = /* wgsl */ `
struct Body {
    pos: vec4f,
    vel: vec4f,
};

struct CameraUniform {
    view_proj: mat4x4f,
    camera_pos: vec4f,
    inv_aspect: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(1) var<uniform> camera: CameraUniform;

const QUAD_HALF_SIZE = 0.19;
const DOPPLER_SCALE = 0.1;

struct VertexOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) radial_v: f32,
};

@vertex
fn vs(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOut {
    // Two triangles covering [-1, 1]:
    //   0:(-1,-1)  1:( 1,-1)  2:(-1, 1)
    //   3:(-1, 1)  4:( 1,-1)  5:( 1, 1)
    let corner = array<vec2f, 6>(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0),
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0),
    );

    let body = bodies[iid];
    let clip_pos = camera.view_proj * vec4f(body.pos.xyz, 1.0);

    // Offset in clip space. X is scaled by inv_aspect so the quad maps
    // to a circle in pixels despite the non-square viewport. Depth-size
    // variation is intentional (from the perspective divide on w).
    let uv = corner[vid];
    let offset = vec2f(uv.x * QUAD_HALF_SIZE * camera.inv_aspect, uv.y * QUAD_HALF_SIZE);

    // Radial velocity: component of body velocity along the line of sight
    // from camera to body. Positive = receding (redshift), negative =
    // approaching (blueshift).
    let view_dir = normalize(body.pos.xyz - camera.camera_pos.xyz);
    let radial_v = dot(body.vel.xyz, view_dir);

    var out: VertexOut;
    out.clip_pos = vec4f(clip_pos.xy + offset, clip_pos.z, clip_pos.w);
    out.uv = uv;
    out.radial_v = radial_v;
    return out;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
    let dist = length(in.uv);
    if (dist > 1.0) {
        discard;
    }
    // Soft edge: alpha falls off near radius 1.0.
    let alpha = smoothstep(1.0, 0.75, dist);

    // Doppler tint: map radial velocity to a red/blue/white color.
    // v < 0 (approaching) → blue, v > 0 (receding) → red, v ≈ 0 → white.
    let v = clamp(in.radial_v * DOPPLER_SCALE, -1.0, 1.0);
    let blue_amount = max(-v, 0.0);
    let red_amount = max(v, 0.0);
    let base = vec3f(1.0, 1.0, 0.85);
    let color = base
        + (vec3f(0.3, 0.5, 1.5) - base) * blue_amount
        + (vec3f(1.5, 0.5, 0.3) - base) * red_amount;

    return vec4f(color, alpha);
}
`;
