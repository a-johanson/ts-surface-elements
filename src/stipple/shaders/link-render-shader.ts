/**
 * Link render shader — draws each point's link as a straight line
 * segment to its best-aligned neighbor (`line-list` topology, one line
 * per point instance).
 *
 * The vertex shader resolves the two endpoints from the post-relax point
 * buffer: endpoint 0 is the point itself, endpoint 1 is its linked
 * neighbor. Points with no link (`UINT_MAX`) and links whose endpoints
 * are both fully occluded (both endpoints' corner clearances in the
 * shading buffer are non-positive) are pushed offscreen so no fragments
 * are rasterized for them.
 *
 * The fragment shader paints a solid translucent color.
 */
import { SDF_COMMON } from "./sdf-common.js";

export const LINK_RENDER_SHADER = /* wgsl */ `
${SDF_COMMON}

struct LinkRenderUniform {
    view_proj: mat4x4f,
    eye: vec3f,
    _pad: f32,
};

const UINT_MAX: u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var<uniform> u: LinkRenderUniform;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read> links: array<u32>;
@group(0) @binding(3) var<storage, read> shading: array<ShadingSample>;

@vertex
fn link_vs(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> @builtin(position) vec4f {
    let j = links[iid];
    if (j == UINT_MAX) {
        return vec4f(2.0, 2.0, 2.0, 1.0);
    }

    // Coarse visibility: draw only when at least one endpoint's stipple
    // splat is visible to the camera.
    let sh_i = shading[iid];
    let sh_j = shading[j];
    let clearance_i = max(max(sh_i.clearance.x, sh_i.clearance.y), max(sh_i.clearance.z, sh_i.clearance.w));
    let clearance_j = max(max(sh_j.clearance.x, sh_j.clearance.y), max(sh_j.clearance.z, sh_j.clearance.w));
    if (clearance_i <= 0.0 && clearance_j <= 0.0) {
        return vec4f(2.0, 2.0, 2.0, 1.0);
    }

    let p_i = points[iid].pos.xyz;
    let p_j = points[j].pos.xyz;
    let p = select(p_j, p_i, vid == 0u);
    return u.view_proj * vec4f(p, 1.0);
}

@fragment
fn link_fs() -> @location(0) vec4f {
    return vec4f(0.6, 0.8, 1.0, 1.0);
}
`;
