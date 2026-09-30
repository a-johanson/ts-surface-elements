/**
 * Point render shader — draws stipple points as tangent-plane quads
 * oriented from the shared normals buffer and projected via a
 * view-projection matrix uniform.
 *
 * The vertex shader builds the shared tangent basis for each point and
 * offsets the quad corners in world space by `POINT_RADIUS_WORLD`, so
 * quads lie flat on the SDF surface and foreshorten with viewing angle.
 * Each corner carries its occlusion clearance from the shading buffer;
 * points whose corners are all fully occluded are pushed offscreen so no
 * fragments are rasterized for them.
 *
 * The fragment shader paints a white ring outline whose radius scales
 * linearly with per-point luminance (collapsing to a point at zero
 * luminance); the line width is configurable via `LINE_WIDTH` and the ring
 * never exceeds the quad boundary. Occlusion is resolved per fragment.
 */
import { SDF_COMMON } from "./sdf-common.js";

export const POINT_SHADER = /* wgsl */ `
${SDF_COMMON}

struct PointUniform {
    view_proj: mat4x4f,
};

@group(0) @binding(0) var<uniform> u: PointUniform;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read> shading: array<ShadingSample>;
@group(0) @binding(3) var<storage, read> normals: array<vec4f>;

struct VertexOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) luminance: f32,
    @location(2) clearance: f32,
};

@vertex
fn point_vs(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOut {
    let corner = array<vec2f, 6>(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0),
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0),
    );
    let corner_ids = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);

    let sh = shading[iid];
    var out: VertexOut;
    out.uv = corner[vid];
    out.luminance = sh.lum.x;
    out.clearance = sh.clearance[corner_ids[vid]];

    let clearance_max = max(max(sh.clearance.x, sh.clearance.y), max(sh.clearance.z, sh.clearance.w));
    if (clearance_max == 0.0) {
        out.clip_pos = vec4f(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let p = points[iid].pos.xyz;
    let n = normals[iid].xyz;
    let frame = tangentFrame(n);
    let c = corner[vid];
    let world = p + (c.x * frame.t1 + c.y * frame.t2) * POINT_RADIUS_WORLD;
    out.clip_pos = u.view_proj * vec4f(world, 1.0);
    return out;
}

@fragment
fn point_fs(in: VertexOut) -> @location(0) vec4f {
    const LINE_WIDTH: f32 = 0.25;
    const HALF_W: f32 = LINE_WIDTH * 0.5;
    const MIN_R: f32 = 0.75 * HALF_W;
    const MAX_R: f32 = 1.0 - 1.5 * HALF_W;
    const AA_WIDTH: f32 = 0.5;

    let aa_clearance = fwidth(in.clearance);
    let half_band_clearance = aa_clearance * AA_WIDTH;
    if (in.clearance < -half_band_clearance) {
        discard;
    }
    let alpha_clearance = smoothstep(-half_band_clearance, half_band_clearance, in.clearance);

    let d_center = length(in.uv);
    let aa = fwidth(d_center);
    let half_band = aa * AA_WIDTH;
    let r = (MAX_R - MIN_R) * in.luminance + MIN_R;
    let d_ring = abs(d_center - r);
    if (d_ring > HALF_W + half_band) {
        discard;
    }
    let alpha = smoothstep(HALF_W + half_band, HALF_W - half_band, d_ring);
    return vec4f(1.0, 1.0, 1.0, alpha * alpha_clearance);
}
`;
