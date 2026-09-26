/**
 * Debug render shader — ray-marches the time-animated SDF per fragment and
 * writes grayscale Lambert shading modulated by Aaltonen soft shadows
 * directly to the canvas color attachment.
 *
 * Background (ray miss) maps to black; hit fragments map to `[0, 1]` gray
 * via `1 - Lambert`, scaled by the soft-shadow penumbra factor.
 */
import { SDF_COMMON } from "./sdf-common.js";

export const DEBUG_RENDER_SHADER = /* wgsl */ `
${SDF_COMMON}

struct CameraRays {
    eye: vec3f,
    half_width: f32,
    forward: vec3f,
    half_height: f32,
    right: vec3f,
    _pad0: f32,
    up: vec3f,
    time: f32,
    resolution: vec2u,
    _pad2: vec2u,
};

@group(0) @binding(0) var<uniform> camera: CameraRays;

@vertex
fn blit_vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    let pos = array<vec2f, 6>(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0),
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0),
    );
    return vec4f(pos[vid], 0.0, 1.0);
}

@fragment
fn blit_fs(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let res = camera.resolution;
    let uv = frag_coord.xy / vec2f(res);
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    let rd = normalize(
        camera.forward
        + ndc.x * camera.half_width * camera.right
        + ndc.y * camera.half_height * camera.up
    );

    let t = rayMarch(camera.eye, rd, camera.time);
    if (t < 0.0) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    let p = camera.eye + rd * t;
    let n = calcNormal(p, camera.time);
    let light_dir = normalize(vec3f(0.5, 0.8, 0.6));
    let lambert = max(dot(n, light_dir), 0.0);
    let shadow = softShadow(p + n * SHADOW_BIAS, light_dir, camera.time);
    let lum = lambert * shadow;
    return vec4f(lum, lum, lum, 1.0);
}
`;
