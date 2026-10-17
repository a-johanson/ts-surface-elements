/**
 * Link render shader — draws each surviving link as a chain of
 * `N_SEGMENTS` camera-facing quads following the subdivided,
 * surface-projected curve written by the link compute pass
 * (`triangle-list` topology, one quad per sub-segment instance).
 *
 * Each instance maps to point `i = inst / N_SEGMENTS` and sub-segment
 * `s = inst % N_SEGMENTS`; the endpoints are the adjacent curve points
 * `curves[i * STRIDE + s]` and `curves[i * STRIDE + s + 1]`.
 *
 * Each quad is billboarded around the segment axis (decision 6 in the
 * hatch-lines spec): the across-axis side vector is
 * `side = normalize(cross(seg_dir, view_dir))`. The line itself keeps
 * a world-space half-width (`LINK_HALF_WIDTH`) so it foreshortens with
 * viewing angle and depth — the same philosophy as the point
 * renderer's `POINT_RADIUS_WORLD`. The quad, however, is oversized in
 * world space to always leave enough screen-pixel room for the
 * anti-alias band, even at extreme zoom-out (the quad half-extent is
 * `LINK_HALF_WIDTH + AA_PIXELS * world_per_pixel`, so the AA fade
 * never gets clipped and the line never goes sub-pixel).
 *
 * The line shape is a 2D segment distance field evaluated in the
 * fragment shader: the signed distance to the segment centerline
 * (clamped to the endpoints) minus `LINK_HALF_WIDTH` — a capsule SDF
 * whose iso-zero contour is a rectangle with semicircular rounded
 * caps. Anti-aliased via `fwidth` on the signed distance — same
 * pattern as the point shader's disc SDF, giving a ~1px soft edge
 * independent of zoom and world-space location.
 *
 * The quad extends along the segment axis by `LINK_HALF_WIDTH` plus
 * the AA corridor beyond each endpoint, so the rounded caps fit
 * entirely inside the quad (the previous axis-clipped quad chopped
 * them off). Because adjacent sub-segments share their endpoints
 * exactly (the Hermite subdivision writes identical endpoints), the
 * rounded caps of neighboring segments overlap at each joint and —
 * with `LINK_ALPHA = 1` (opaque interior, only the ~1px AA edge
 * blends) — the union renders as a rounded miter / bent-sausage join
 * with no hairline gaps. This covers Step 5 (mitering) for free: no
 * miter geometry, no SDF union of two segments, no extra varyings.
 * Chain ends terminate with a rounded cap too — no special handling.
 *
 * Occlusion (Step 4 in the hatch-lines spec) is resolved per curve
 * point and geometrically: the link compute's clearance stage stores,
 * in each curve point's `.w`, the occlusion clearance of a
 * `rayClearance` trace toward the eye (> 0 visible, < 0 occluded, 0.0
 * dropped or absent link — its whole slice is zeroed).
 */
import { LINK_N_SEGMENTS } from "./link-shader.js";

export const LINK_RENDER_SHADER = /* wgsl */ `
struct LinkRenderUniform {
    view_proj: mat4x4f,
    eye: vec3f,
    canvas_size: vec2f,
    _pad: f32,
};

/**
 * Hermite sub-segment count per link — interpolated from the TS
 * constant LINK_N_SEGMENTS (single source of truth, defined in
 * link-shader.ts; decision 8 in the hatch-lines spec). Resolves the
 * instance → point/sub-segment mapping.
 */
const N_SEGMENTS: u32 = ${LINK_N_SEGMENTS}u;

/** Curve points per link (N_SEGMENTS + 1). */
const STRIDE: u32 = N_SEGMENTS + 1u;

/**
 * World-space half-width of the line itself — sized relative to
 * POINT_RADIUS_WORLD (lines read thinner than the dots they
 * connect). Tunable. The rendered quad is larger than this to fit the
 * AA band (see AA_PIXELS).
 */
const LINK_HALF_WIDTH: f32 = 0.006;

/** Link color and base alpha (modulated by the AA edge in the fragment). */
const LINK_COLOR: vec3f = vec3f(1.0, 1.0, 1.0);
const LINK_ALPHA: f32 = 1.0;

/**
 * Anti-alias band scale — matches the point shader's AA_WIDTH so line
 * edges and dot edges have consistent softness (~1px).
 */
const AA_WIDTH: f32 = 0.5;

/**
 * Quad padding in pixels added on each side of the line to guarantee
 * the AA fade always has screen-pixel room, even at extreme zoom-out.
 * Covers the AA band plus a margin. At zoom-out where the line itself
 * goes sub-pixel, the quad settles to roughly this many pixels and the
 * line becomes its AA band — a stable minimum stroke width.
 */
const AA_PIXELS: f32 = 2.0;

/**
 * Segment length below which the endpoints are treated as coincident
 * and the segment is rendered as a disc (half_len = 0). The capsule
 * SDF collapses to a disc naturally — the fragment's clamp returns 0,
 * so closest = seg_center and dist = length(world_pos - center) — but
 * seg_dir cannot be computed from the zero vector, so a fallback
 * direction perpendicular to view_dir is picked for the billboard
 * basis. Unsquared comparison (length already took the sqrt).
 */
const DEGENERATE_EPS: f32 = 1e-4;

/**
 * Length squared of cross(seg_dir, view_dir) below which the segment
 * axis is treated as parallel to the view direction (the segment is
 * foreshortened to a point on screen). normalize(cross) would produce
 * NaN here, which the rasterizer turns into huge garbage triangles
 * spanning the framebuffer — the grazing-angle "stretched sheets" and
 * GPU freeze. Guarded with an explicit if/else fallback instead.
 */
const PARALLEL_EPS2: f32 = 1e-7;

@group(0) @binding(0) var<uniform> u: LinkRenderUniform;
@group(0) @binding(1) var<storage, read> curves: array<vec4f>;

struct VertexOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) world_pos: vec3f,
    @location(1) @interpolate(flat) seg_center: vec3f,
    @location(2) @interpolate(flat) seg_dir_v: vec3f,
    @location(3) @interpolate(flat) half_len_v: f32,
    @location(4) @interpolate(flat) clearance: f32,
};

// Two-triangle quad covering the segment. corner.x is the side
// offset (-1..+1 -> +/-quad_half across the segment axis);
// corner.y is the along-axis parameter (-1 -> e0, +1 -> e1).
const QUAD_CORNER: array<vec2f, 6> = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
);

fn offscreen() -> VertexOut {
    var out: VertexOut;
    out.clip_pos = vec4f(2.0, 2.0, 2.0, 1.0);
    out.world_pos = vec3f(0.0);
    out.seg_center = vec3f(0.0);
    out.seg_dir_v = vec3f(0.0);
    out.half_len_v = 0.0;
    out.clearance = -1.0;
    return out;
}

@vertex
fn link_vs(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOut {
    let i = iid / N_SEGMENTS;
    let s = iid % N_SEGMENTS;
    let e0 = curves[i * STRIDE + s];
    let e1 = curves[i * STRIDE + s + 1u];

    let c0 = e0.w;
    let c1 = e1.w;
    let c_hi = max(c0, c1);
    if (c_hi <= 0.0) {
        return offscreen();
    }

    let p0 = e0.xyz;
    let p1 = e1.xyz;
    let seg_vec = p1 - p0;
    let seg_len = length(seg_vec);
    let center = 0.5 * (p0 + p1);
    let view_dir = normalize(u.eye - center);

    // Camera-facing quad (decision 6): side = cross(seg_dir, view_dir).
    // For a zero-length segment the capsule SDF collapses to a disc
    // (half_len = 0 → fragment clamp returns 0 → closest = center),
    // but seg_dir cannot be computed from the zero vector. Pick any
    // unit vector perpendicular to view_dir as the fallback basis —
    // the disc faces the camera regardless of which one is chosen.
    var seg_dir: vec3f;
    var half_len: f32;
    if (seg_len < DEGENERATE_EPS) {
        let alt = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0),
                         abs(view_dir.y) > 0.9);
        seg_dir = normalize(cross(view_dir, alt));
        half_len = 0.0;
    } else {
        seg_dir = seg_vec / seg_len;
        half_len = 0.5 * seg_len;
    }

    // Clip-space center, used both for the world-units-per-pixel
    // estimate (AA padding) and for the behind-camera guard below.
    let clip_c = u.view_proj * vec4f(center, 1.0);

    // Behind-camera guard: at grazing angles some segment centers can
    // land behind the camera plane (clip.w <= 0). Quad corners extended
    // along the axis then straddle w = 0 and rasterize as slivers
    // stretching across the screen. Push offscreen instead.
    if (clip_c.w <= 0.0) {
        return offscreen();
    }

    // side = normalize(cross(seg_dir, view_dir)), but guarded: when the
    // segment axis is (near) parallel to the view direction, cross → 0
    // and normalize(0) → NaN, which the rasterizer turns into huge
    // garbage triangles and GPU freezes (the grazing-angle pathology).
    // Use an explicit if/else so normalize is never called on the zero
    // vector; in the parallel case pick any vector not parallel to
    // seg_dir and cross with it for a valid perpendicular — the
    // foreshortened segment renders as its rounded cap (a small disc).
    let cross_v = cross(seg_dir, view_dir);
    var side: vec3f;
    if (dot(cross_v, cross_v) > PARALLEL_EPS2) {
        side = normalize(cross_v);
    } else {
        let alt = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(seg_dir.y) > 0.9);
        side = normalize(cross(seg_dir, alt));
    }

    // Size the quad so the AA band always has screen-pixel room: add
    // AA_PIXELS worth of world-space padding on each side. Measure the
    // world-units-per-pixel along the side direction directly by
    // transforming side (a direction, w=0) through the view-projection
    // and linearizing at the center's clip.w.
    // Because side ⊥ view_dir by construction (both the normal and the
    // parallel-segment fallback cases), side lives in the camera's xy
    // plane, so side_clip.xy is always non-zero → wpp is always finite.
    let side_clip = u.view_proj * vec4f(side, 0.0);
    let px_delta = (side_clip.xy / clip_c.w) * (0.5 * u.canvas_size);
    let wpp = 1.0 / length(px_delta);
    let quad_half = LINK_HALF_WIDTH + AA_PIXELS * wpp;

    let c = QUAD_CORNER[vid];
    let t = (c.y + 1.0) * 0.5;          // -1 -> 0, +1 -> 1
    // Extend the quad along the segment axis by quad_half beyond each
    // endpoint so the capsule's rounded caps fit inside the quad
    // (otherwise the caps are clipped and adjacent segments show
    // hairline gaps at joints). With LINK_ALPHA = 1 the overlap of
    // neighboring segments' caps at a shared endpoint is invisible
    // and renders as a rounded miter — see the header comment.
    let p = mix(p0, p1, t) + c.y * quad_half * seg_dir;
    let world = p + c.x * side * quad_half;

    var out: VertexOut;
    out.clip_pos = u.view_proj * vec4f(world, 1.0);
    out.world_pos = world;
    out.seg_center = center;
    out.seg_dir_v = seg_dir;
    out.half_len_v = half_len;
    out.clearance = 1.0;
    return out;
}

@fragment
fn link_fs(in: VertexOut) -> @location(0) vec4f {
    let d = in.world_pos - in.seg_center;
    let t = clamp(dot(d, in.seg_dir_v), -in.half_len_v, in.half_len_v);
    let closest = in.seg_center + t * in.seg_dir_v;
    let sdist = length(in.world_pos - closest) - LINK_HALF_WIDTH;

    let half_band = fwidth(sdist) * AA_WIDTH;
    if (sdist > half_band) {
        discard;
    }
    let alpha = smoothstep(half_band, -half_band, sdist) * LINK_ALPHA;
    return vec4f(LINK_COLOR, alpha);
}
`;
