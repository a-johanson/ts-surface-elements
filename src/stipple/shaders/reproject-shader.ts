/**
 * Reproject compute shader — re-projects points onto the current animated
 * surface and refreshes the shared normals and luminance buffers.
 *
 * One invocation per point. Runs every frame *before* relax. Reads each
 * point's position (from the buffer relax is about to read), Newton-
 * projects it onto the surface at the current animation time with a
 * generous iteration budget, writes the projected position back in-place,
 * and writes the matching surface normal into the shared normals buffer.
 *
 * It also computes the per-point view-independent luminance
 * (`softShadow`-attenuated Lambert toward the animated light) from the
 * fresh projected position and normal and writes it into the shared
 * shading buffer's `luminance` field. Relax reads this luminance later in
 * the same frame for density modulation, so it never sees cross-frame
 * stale values. Writes are member-wise so the `clearance` field — owned
 * by the shading pass — is untouched.
 *
 * This ensures that relax starts with on-surface positions and matching
 * normals even when the SDF has moved since the last frame. Without this
 * pass, relax would read stale positions (off the new surface) and stale
 * normals (describing the old surface), corrupting the repulsion kernel.
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
@group(0) @binding(3) var<storage, read_write> shading_out: array<PointShading>;

const REPROJECT_NEWTON_ITERS: i32 = 3;
const REPROJECT_ALPHA: f32 = 0.5;

@compute @workgroup_size(64)
fn reproject_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let time = params.time;
    let p = points[i].pos.xyz;
    let projected = projectToSurface(p, time, REPROJECT_NEWTON_ITERS, REPROJECT_ALPHA);
    points[i].pos = vec4f(projected, 0.0);
    let n = normalize(sdfGradient(projected, time));
    normals_out[i] = vec4f(n, 0.0);

    // View-independent luminance (shadow-aware Lambert) for the relax
    // kernel's density modulation. Member-wise write: the clearance
    // field is owned by the shading pass.
    let light = lightDir(time);
    let lambert = max(dot(n, light), 0.0);
    var lum: f32 = 0.0;
    if (lambert > 0.0) {
        lum = softShadow(projected + n * SHADOW_BIAS, light, time) * lambert;
    }
    shading_out[i].luminance = lum;
}
`;
