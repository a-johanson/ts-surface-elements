/**
 * Point render shader — draws stipple points as small fixed-size dots:
 * tangent-plane quads oriented from the shared normals buffer and
 * projected via a view-projection matrix uniform.
 *
 * The vertex shader builds the shared tangent basis for each point and
 * offsets the quad corners in world space by `POINT_RADIUS_WORLD`, so
 * quads lie flat on the SDF surface and foreshorten with viewing angle.
 * Back-facing points — normals pointing away from the eye — are pushed
 * offscreen (a cheap dot-product sign test; no occlusion handling, so
 * front-facing points always draw, even when another part of the
 * surface covers them).
 *
 * The fragment shader paints a plain black anti-aliased filled disc at a
 * fixed radius, independent of luminance, so dots read as a uniform
 * overlay for the hatch lines.
 */
import { SDF_COMMON } from "./sdf-common.js";

export const POINT_SHADER = /* wgsl */ `
${SDF_COMMON}

struct PointUniform {
    view_proj: mat4x4f,
    eye: vec3f,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> u: PointUniform;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read> normals: array<vec4f>;

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

    let p = points[iid].pos.xyz;
    let n = normals[iid].xyz;

    // Back-face cull: a normal pointing away from the eye (unnormalized
    // difference — only the sign matters) marks the surface's far side.
    if (dot(n, u.eye - p) <= 0.0) {
        out.clip_pos = vec4f(2.0, 2.0, 2.0, 1.0);
        return out;
    }

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
