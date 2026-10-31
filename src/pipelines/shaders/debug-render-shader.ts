/**
 * Debug render shader — ray-marches the time-animated SDF per fragment and
 * writes a 1D periodic pattern evaluated on the surface directly to the
 * canvas color attachment.
 *
 * The signal's phase is the projection of world position onto the
 * (animated) light direction, so its iso-lines are slices orthogonal to
 * the light. Its wavelength is inversely related to the Lambert luminance
 * modulated by the soft-shadow penumbra — darker fragments show denser
 * rings.
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
fn debug_vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
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
fn debug_fs(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
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
    let light_dir = lightDir(camera.time);
    let ndotl = dot(n, light_dir);
    let lambert = max(ndotl, 0.0);
    let shadow = softShadow(p + n * SHADOW_BIAS, light_dir, camera.time);
    let lum = lambert * shadow;

    const PATTERN_WAVELENGTH_MIN: f32 = 0.05;
    const PATTERN_WAVELENGTH_MAX: f32 = 0.2;
    // const PATTERN_POLE_BLEND: f32 = 1.0;

    let wavelength = mix(PATTERN_WAVELENGTH_MIN, PATTERN_WAVELENGTH_MAX, lum);
    // let theta = acos(clamp(ndotl, -1.0, 1.0));
    // let mask = smoothstep(-0.7, -1.0, ndotl);
    // let phase = TAU * (dot(p, light_dir) + PATTERN_POLE_BLEND * mask * theta) / wavelength;
    let phase = TAU * dot(p, light_dir) / wavelength;
    let shade = 0.5 + 0.5 * sin(phase);
    return vec4f(shade, shade, shade, 1.0);
}
`;
