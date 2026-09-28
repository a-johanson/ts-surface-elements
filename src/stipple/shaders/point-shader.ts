/**
 * Point render shader — draws stipple points as tangent-plane quads
 * oriented from the shared normals buffer and projected via a
 * view-projection matrix uniform.
 *
 * The vertex shader builds an orthonormal tangent basis (t1, t2) from
 * the pre-computed surface normal at each point and offsets the quad
 * corners in world space by `POINT_RADIUS_WORLD`, so quads lie flat on
 * the SDF surface and foreshorten naturally with viewing angle. Points
 * flagged as occluded in the shading buffer are pushed offscreen in the
 * vertex shader so no fragments are rasterized for them. The fragment
 * shader paints a white ring outline whose radius scales linearly with
 * per-point luminance (collapsing to a point at zero luminance); the
 * line width is configurable via `LINE_WIDTH` and the ring never
 * exceeds the quad boundary. Anti-aliasing uses `fwidth` so the soft
 * band spans ~1 pixel regardless of viewport size or camera distance.
 */
import { SDF_COMMON } from "./sdf-common.js";

export const POINT_SHADER = /* wgsl */ `
${SDF_COMMON}

struct PointUniform {
    view_proj: mat4x4f,
};

@group(0) @binding(0) var<uniform> u: PointUniform;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read> shading: array<vec4f>;
@group(0) @binding(3) var<storage, read> normals: array<vec4f>;

struct VertexOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) luminance: f32,
};

@vertex
fn point_vs(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOut {
    const POINT_RADIUS_WORLD: f32 = 0.03;

    let corner = array<vec2f, 6>(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0),
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0),
    );

    let sh = shading[iid];
    var out: VertexOut;
    out.uv = corner[vid];
    out.luminance = sh.y;

    if (sh.x < 0.5) {
        out.clip_pos = vec4f(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let p = points[iid].pos.xyz;
    let n = normals[iid].xyz;
    let up = mix(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), f32(abs(n.y) > 0.99));
    let t1 = normalize(cross(up, n));
    let t2 = cross(n, t1);
    let c = corner[vid];
    let world = p + (c.x * t1 + c.y * t2) * POINT_RADIUS_WORLD;
    out.clip_pos = u.view_proj * vec4f(world, 1.0);
    return out;
}

@fragment
fn point_fs(in: VertexOut) -> @location(0) vec4f {
    const LINE_WIDTH: f32 = 0.25;
    const HALF_W: f32 = LINE_WIDTH * 0.5;
    const MIN_R: f32 = 0.75 * HALF_W;
    const MAX_R: f32 = 1.0 - HALF_W;
    let dist = length(in.uv);
    let aa = fwidth(dist);
    let half_band = aa * 0.5;
    let r = (MAX_R - MIN_R) * in.luminance + MIN_R;
    let d = abs(dist - r);
    if (d > HALF_W + half_band) {
        discard;
    }
    let alpha = smoothstep(HALF_W + half_band, HALF_W - half_band, d);
    return vec4f(1.0, 1.0, 1.0, alpha);
}
`;
