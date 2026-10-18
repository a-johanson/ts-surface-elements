/**
 * Point render shader — draws stipple points as small fixed-size dots:
 * tangent-plane quads oriented from the shared normals buffer and
 * projected via a view-projection matrix uniform.
 *
 * The vertex shader builds the shared tangent basis for each point and
 * offsets the quad corners in world space by `POINT_RADIUS_WORLD`, so
 * quads lie flat on the SDF surface and foreshorten with viewing angle.
 * Points whose center-ray clearance (from the shading buffer) is
 * non-positive — the surface occludes them from the camera — are pushed
 * offscreen so no fragments are rasterized for them.
 *
 * The fragment shader paints a plain white anti-aliased filled disc at a
 * fixed radius, independent of luminance, so dots read as a uniform
 * overlay for the hatch lines.
 */
import { SDF_COMMON } from "./sdf-common.js";

export const POINT_SHADER = /* wgsl */ `
${SDF_COMMON}

struct PointUniform {
    view_proj: mat4x4f,
};

@group(0) @binding(0) var<uniform> u: PointUniform;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read> shading: array<PointShading>;
@group(0) @binding(3) var<storage, read> normals: array<vec4f>;

struct VertexOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) uv: vec2f,
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

    var out: VertexOut;
    out.uv = corner[vid];

    // Coarse occlusion cull: non-positive center-ray clearance means the
    // surface between the point and the camera occludes it.
    if (shading[iid].clearance <= 0.0) {
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
    // Dot radius as a fraction of the quad half-extent (fixed size,
    // irrespective of luminance).
    const DOT_RADIUS: f32 = 0.5;
    const AA_WIDTH: f32 = 0.5;

    let d_center = length(in.uv);
    let half_band = fwidth(d_center) * AA_WIDTH;
    if (d_center > DOT_RADIUS + half_band) {
        discard;
    }
    let alpha = smoothstep(DOT_RADIUS + half_band, DOT_RADIUS - half_band, d_center);
    return vec4f(0.0, 0.0, 0.0, alpha);
}
`;
