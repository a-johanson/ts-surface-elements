/**
 * Link render shader — draws each surviving link as a chain of
 * `N_SEGMENTS` straight line segments following the subdivided,
 * surface-projected curve written by the link compute pass
 * (`line-list` topology, one sub-segment per instance).
 *
 * Each instance maps to point `i = inst / N_SEGMENTS` and sub-segment
 * `s = inst % N_SEGMENTS`; the endpoints are the adjacent curve points
 * `curves[i * STRIDE + s]` and `curves[i * STRIDE + s + 1]`.
 *
 * Two offscreen culls apply in the vertex shader:
 *
 *  - a link with no neighbor (or dropped by dedup) has an invalid
 *    curve slice — the validity flag `.w == 0.0` on either endpoint;
 *  - the interim coarse visibility cull (Step 2b in the hatch-lines
 *    spec): a link whose stipple endpoints are both fully occluded
 *    (non-positive center-ray clearance in the shading buffer) draws
 *    nothing. Per-segment occlusion replaces this in Step 4.
 *
 * The fragment shader paints a solid color.
 */
import { SDF_COMMON } from "./sdf-common.js";

export const LINK_RENDER_SHADER = /* wgsl */ `
${SDF_COMMON}

struct LinkRenderUniform {
    view_proj: mat4x4f,
    eye: vec3f,
    _pad: f32,
};

/**
 * Curve subdivision count per link — must match the N_SEGMENTS
 * constant in the link shader (the WGSL value is the source of truth;
 * this mirror resolves the instance → point/sub-segment mapping).
 */
const N_SEGMENTS: u32 = 4u;

/** Curve points per link (N_SEGMENTS + 1). */
const STRIDE: u32 = N_SEGMENTS + 1u;

@group(0) @binding(0) var<uniform> u: LinkRenderUniform;
@group(0) @binding(1) var<storage, read> curves: array<vec4f>;
@group(0) @binding(2) var<storage, read> links: array<u32>;
@group(0) @binding(3) var<storage, read> shading: array<PointShading>;

@vertex
fn link_vs(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> @builtin(position) vec4f {
    let i = iid / N_SEGMENTS;
    let s = iid % N_SEGMENTS;
    let e0 = curves[i * STRIDE + s];
    let e1 = curves[i * STRIDE + s + 1u];
    if (e0.w == 0.0 || e1.w == 0.0) {
        return vec4f(2.0, 2.0, 2.0, 1.0);
    }

    // Coarse visibility (interim until per-segment clearance arrives
    // in Step 4): draw only when at least one endpoint's stipple dot
    // is visible to the camera. The validity check above guarantees a
    // surviving link, so j is a valid point index here.
    let j = links[i];
    let clearance_i = shading[i].clearance;
    let clearance_j = shading[j].clearance;
    if (clearance_i <= 0.0 && clearance_j <= 0.0) {
        return vec4f(2.0, 2.0, 2.0, 1.0);
    }

    let p = select(e1.xyz, e0.xyz, vid == 0u);
    return u.view_proj * vec4f(p, 1.0);
}

@fragment
fn link_fs() -> @location(0) vec4f {
    return vec4f(0.6, 0.8, 1.0, 1.0);
}
`;
