/**
 * Reproject compute shader — re-projects points onto the current animated
 * surface and refreshes the shared normals buffer.
 *
 * One invocation per point. Runs every frame *before* relax. Reads each
 * point's position (from the buffer relax is about to read), Newton-
 * projects it onto the surface at the current animation time with a
 * generous iteration budget, writes the projected position back in-place,
 * and writes the matching surface normal into the shared normals buffer.
 *
 * This ensures that relax starts with on-surface positions and matching
 * normals even when the SDF has moved since the last frame. Without this
 * pass, relax would read stale positions (off the new surface) and stale
 * normals (describing the old surface), corrupting the repulsion
 * kernel.
 *
 * In-place `read_write` on the point buffer is safe because each invocation
 * touches only index `i`. Two bind groups (A/B) cover the ping-pong
 * alternation.
 */
import { SDF_COMMON } from "./sdf-common.js";

export const REPROJECT_SHADER = /* wgsl */ `
${SDF_COMMON}

struct ReprojectParams {
    time: f32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: ReprojectParams;
@group(0) @binding(1) var<storage, read_write> points: array<Point>;
@group(0) @binding(2) var<storage, read_write> normals_out: array<vec4f>;

const REPROJECT_NEWTON_ITERS: i32 = 16;
const REPROJECT_ALPHA: f32 = 0.5;

@compute @workgroup_size(64)
fn reproject_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let p = points[i].pos.xyz;
    let projected = projectToSurface(p, params.time, REPROJECT_NEWTON_ITERS, REPROJECT_ALPHA);
    points[i].pos = vec4f(projected, 0.0);
    normals_out[i] = vec4f(normalize(sdfGradient(projected, params.time)), 0.0);
}
`;
