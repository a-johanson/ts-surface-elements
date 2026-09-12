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
