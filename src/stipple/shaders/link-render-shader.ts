/**
 * Link render shader — draws each surviving link as a chain of
 * `N_SEGMENTS` screen-space stroke quads following the subdivided,
 * surface-projected curve written by the link compute pass
 * (`triangle-list` topology, one quad per sub-segment instance).
 *
 * Each instance maps to point `i = inst / N_SEGMENTS` and sub-segment
 * `s = inst % N_SEGMENTS`; the endpoints are the adjacent curve points
 * `curves[i * STRIDE + s]` and `curves[i * STRIDE + s + 1]`.
 *
 * The stroke is built entirely in screen space. The vertex stage
 * perspective-projects the two endpoint curve points to pixel
 * coordinates and extrudes the quad as the bounding box of the 2D
 * capsule around the projected segment — the larger endpoint stroke
 * radius plus the AA margin on every side. The fragment stage evaluates
 * a 2D segment distance field (clamped closest point on the projected
 * segment) against a per-point stroke radius: the world-space
 * half-width (`LINK_HALF_WIDTH`) converted to pixels at each endpoint's
 * own depth and interpolated linearly along the projected segment —
 * the projected cross-section radius of the 3D tube is exactly
 * proportional to the inverse projective depth — and clamped to a
 * constant within each rounded cap. The stroke therefore reads with
 * the same pixel width its 3D tube would project — shrinking with
 * distance, and collapsing toward its rounded cap as the segment turns
 * parallel to the view — the same philosophy as the point renderer's
 * world-space `POINT_RADIUS_WORLD`.
 *
 * Screen-space evaluation is what keeps the chain continuous at
 * grazing angles. Each capsule's rounded cap extends the stroke radius
 * past the shared endpoint in *every* screen direction, so consecutive
 * sub-segments always overlap at their joints and — with
 * `LINK_ALPHA = 1` (opaque interior, only the ~1px AA edge blends) —
 * the union renders as a rounded, welded miter with no hairline gaps,
 * at any viewing angle. Shared curve points also share a projected
 * depth, so neighboring capsules agree on the joint's stroke radius
 * from both sides. Chain ends terminate with a rounded cap too —
 * no special handling. A world-space capsule cannot do this: its
 * along-axis reach past the shared endpoint foreshortens by the sine
 * of the angle to the view, which starves exactly those joint overlaps
 * when a segment is viewed near-parallel (and its view-compressed
 * distance field also blows the `fwidth`-derived AA band up by the
 * same factor, fading the caps out). Here the near-parallel case
 * degenerates gracefully instead: the projected segment collapses, the
 * capsule becomes a disc, and consecutive discs overlap into a
 * continuous blob along the collapsed chain.
 *
 * Each quad carries a constant clip-space `w` and depth (the segment
 * center's), so interpolation across the quad is exactly linear in
 * screen space and the interpolated stroke position tracks the
 * fragment's true screen position. The anti-aliased edge is a
 * `fwidth`-based ~1px band — independent of viewing angle and zoom,
 * the same pattern as the point shader's disc SDF.
 *
 * Occlusion is resolved per curve point and geometrically: the link
 * compute's clearance stage stores, in each curve point's `.w`, the
 * occlusion clearance of a `rayClearance` trace toward the eye
 * (> 0 visible, < 0 occluded, 0.0 dropped or absent link — its whole
 * slice is zeroed). A sub-segment renders when either of its endpoint
 * curve points is visible: fully occluded ones are pushed offscreen,
 * and partially occluded ones are trimmed at the point where the
 * linearly interpolated endpoint clearances reach zero — the estimated
 * crossing into the occluder's silhouette along the curve.
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
 * link-shader.ts). Resolves the instance → point/sub-segment mapping.
 */
const N_SEGMENTS: u32 = ${LINK_N_SEGMENTS}u;

/** Curve points per link (N_SEGMENTS + 1). */
const STRIDE: u32 = N_SEGMENTS + 1u;

/**
 * World-space half-width of the line itself — sized relative to
 * POINT_RADIUS_WORLD (lines read thinner than the dots they connect).
 * Converted to a pixel stroke radius at each endpoint's own depth.
 * Tunable.
 */
const LINK_HALF_WIDTH: f32 = 0.003;

/** Link color and base alpha (modulated by the AA edge in the fragment). */
const LINK_COLOR: vec3f = vec3f(0.0, 0.0, 0.0);
const LINK_ALPHA: f32 = 1.0;

/**
 * Anti-alias band scale — matches the point shader's AA_WIDTH so line
 * edges and dot edges have consistent softness (~1px).
 */
const AA_WIDTH: f32 = 0.5;

/**
 * Quad padding in pixels added on every side of the stroke to
 * guarantee the AA fade always has screen-pixel room, even at extreme
 * zoom-out where the stroke radius itself collapses sub-pixel. Covers
 * the AA band plus a margin; the stroke then settles to roughly this
 * many pixels and becomes its AA band — a stable minimum stroke
 * width.
 */
const AA_PIXELS: f32 = 2.0;

/**
 * Projected 2D segment length squared below which the endpoints are
 * treated as coincident on screen and the stroke renders as a disc.
 * Covers both coincident curve points and view-parallel segments whose
 * projection collapses to a point; the distance field degenerates to a
 * plain point distance instead of dividing by a zero-length axis.
 */
const DEGENERATE_PX2: f32 = 1e-6;

/**
 * Eye-to-center distance below which the view direction is numerically
 * undefined — this close, the difference of two world-position f32s at
 * scene scale is noise-dominated. The camera sits inside the stroke at
 * that range; the sub-segment is dropped rather than computing a
 * garbage basis.
 */
const EYE_EPS: f32 = 1e-4;

@group(0) @binding(0) var<uniform> u: LinkRenderUniform;
@group(0) @binding(1) var<storage, read> curves: array<vec4f>;

struct VertexOut {
    @builtin(position) clip_pos: vec4f,
    // Fragment position in stroke space (screen pixels), interpolated.
    @location(0) frag_px: vec2f,
    // Per-instance stroke geometry: projected segment endpoints and
    // per-endpoint stroke radii, in stroke pixels.
    @location(1) @interpolate(flat) p0_px: vec2f,
    @location(2) @interpolate(flat) p1_px: vec2f,
    @location(3) @interpolate(flat) r0_px: f32,
    @location(4) @interpolate(flat) r1_px: f32,
};

// Two-triangle quad covering the projected segment. corner.x is the
// across-axis offset (-1..+1 -> +/-quad_half perpendicular to the
// projected axis); corner.y is the along-axis parameter (-1 -> e0,
// +1 -> e1), extended past the endpoints by the quad padding so the
// rounded caps fit inside the quad.
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
    out.frag_px = vec2f(0.0);
    out.p0_px = vec2f(0.0);
    out.p1_px = vec2f(0.0);
    out.r0_px = 0.0;
    out.r1_px = 0.0;
    return out;
}

// NDC (y-up) to stroke space, and back. Stroke space is screen-scaled
// (one unit per canvas pixel), so distances evaluated in it are true
// screen-pixel distances; the orientation difference relative to the
// framebuffer is an isometry and does not affect them.
fn pxFromNdc(ndc: vec2f) -> vec2f {
    return (ndc * 0.5 + vec2f(0.5)) * u.canvas_size;
}

fn ndcFromPx(px: vec2f) -> vec2f {
    return (px / u.canvas_size) * 2.0 - vec2f(1.0);
}

/**
 * Unsigned distance from a stroke-space point to the projected segment
 * [a, b], in stroke pixels, plus the clamped closest-point parameter t
 * (0 at a, 1 at b).
 */
fn strokeDistance(p: vec2f, a: vec2f, b: vec2f) -> vec2f {
    let ab = b - a;
    let len2 = dot(ab, ab);
    if len2 < DEGENERATE_PX2 {
        return vec2f(length(p - a), 0.0);
    }
    let t = clamp(dot(p - a, ab) / len2, 0.0, 1.0);
    return vec2f(length(p - a - ab * t), t);
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

    // Discard segment if both clearance values are non-positive
    let c0 = e0.w;
    let c1 = e1.w;
    if c0 <= 0.0 && c1 <= 0.0 {
        return offscreen();
    }

    // Mixed clearance (one endpoint visible, one occluded): trim the
    // occluded side at the point where the linearly interpolated
    // clearance reaches zero — the estimated crossing into the
    // occluder's silhouette along the curve. The cut is made in world
    // space before projection, so the shortened segment flows through
    // the unchanged projection, radius, and distance-field stages.
    var p0 = e0.xyz;
    var p1 = e1.xyz;
    let visible0 = c0 > 0.0;
    let visible1 = c1 > 0.0;
    if visible0 != visible1 {
        let t_cut = c0 / (c0 - c1);
        if visible0 {
            p1 = mix(p0, p1, t_cut);
        } else {
            p0 = mix(p0, p1, t_cut);
        }
    }
    let center = 0.5 * (p0 + p1);

    // Perspective-project the endpoints and the center; everything
    // downstream is built from the projected geometry.
    let clip0 = u.view_proj * vec4f(p0, 1.0);
    let clip1 = u.view_proj * vec4f(p1, 1.0);
    let clip_c = u.view_proj * vec4f(center, 1.0);

    // Behind-camera guard: an endpoint with clip.w <= 0 divides by a
    // negative w and projects to a mirrored, garbage screen position.
    // Drop the sub-segment. This also covers the center, whose clip.w
    // is the affine midpoint of the endpoints'.
    if clip0.w <= 0.0 || clip1.w <= 0.0 {
        return offscreen();
    }

    // Stroke-space endpoints.
    let px0 = pxFromNdc(clip0.xy / clip0.w);
    let px1 = pxFromNdc(clip1.xy / clip1.w);

    // Stroke radius per endpoint, in pixels. The projected radius of
    // the 3D tube's cross-section is exactly proportional to the
    // inverse projective depth: the pixel displacement per world unit
    // along a view-perpendicular direction is a depth-independent
    // pixel scale divided by clip.w. The scale is measured along a
    // view-perpendicular direction because the projected segment
    // itself cannot provide it — its on-screen length collapses to
    // zero exactly in the near-parallel case this shader must survive,
    // while a view-perpendicular projection is always non-degenerate.
    // perp is view-perpendicular, hence lies in the camera's xy plane:
    // its clip-space w is 0 for the standard look-at + perspective
    // pair, so the conversion is exact rather than first-order.
    let to_eye = u.eye - center;
    let dist_eye = length(to_eye);
    if dist_eye < EYE_EPS {
        // The eye sits on the segment center: the view direction (and
        // with it the perpendicular basis) is undefined. The stroke
        // encloses the camera; drop the sub-segment.
        return offscreen();
    }
    let view_dir = to_eye / dist_eye;
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0),
                    abs(view_dir.y) > 0.9);
    let perp = normalize(cross(view_dir, up));
    let perp_clip = u.view_proj * vec4f(perp, 0.0);
    let px_scale = length(perp_clip.xy * (0.5 * u.canvas_size));
    var r0_px = LINK_HALF_WIDTH * px_scale / clip0.w;
    var r1_px = LINK_HALF_WIDTH * px_scale / clip1.w;

    // 2D basis of the projected segment. A collapsed projection
    // (coincident endpoints or a view-parallel segment) falls back to
    // a fixed axis, and both radii flatten to the larger one: seen
    // end-on, the tube's silhouette is dominated by its nearest
    // cross-section, and the distance field degenerates to that disc
    // regardless of which basis is chosen.
    let axis = px1 - px0;
    var axis_dir: vec2f;
    if dot(axis, axis) < DEGENERATE_PX2 {
        axis_dir = vec2f(1.0, 0.0);
        let r_max = max(r0_px, r1_px);
        r0_px = r_max;
        r1_px = r_max;
    } else {
        axis_dir = normalize(axis);
    }
    let perp_dir = vec2f(-axis_dir.y, axis_dir.x);

    // Quad half-extent in pixels: the larger endpoint radius plus the
    // AA margin on every side, so the tapered capsule — rounded caps
    // included — and its AA fade always fit inside the quad.
    let quad_half = max(r0_px, r1_px) + AA_PIXELS;

    // Constant clip w and depth (the segment center's) across the
    // quad: interpolation is then exactly linear in screen space, the
    // extruded quad is exactly the parallelogram built below, and the
    // interpolated stroke position matches each fragment's true screen
    // position.
    let w_c = clip_c.w;
    let z_c = clip_c.z;

    let corner = QUAD_CORNER[vid];
    let t = (corner.y + 1.0) * 0.5;          // -1 -> 0, +1 -> 1
    let frag_px = mix(px0, px1, t)
                + corner.y * quad_half * axis_dir
                + corner.x * quad_half * perp_dir;

    var out: VertexOut;
    out.clip_pos = vec4f(ndcFromPx(frag_px) * w_c, z_c, w_c);
    out.frag_px = frag_px;
    out.p0_px = px0;
    out.p1_px = px1;
    out.r0_px = r0_px;
    out.r1_px = r1_px;
    return out;
}

@fragment
fn link_fs(in: VertexOut) -> @location(0) vec4f {
    let dt = strokeDistance(in.frag_px, in.p0_px, in.p1_px);
    // The stroke radius tapers linearly with depth along the segment
    // body; the clamped closest-point parameter holds it constant
    // within each rounded cap, pinning each cap disc to its endpoint's
    // radius.
    let sdist = dt.x - mix(in.r0_px, in.r1_px, dt.y);

    let half_band = fwidth(sdist) * AA_WIDTH;
    if sdist > half_band {
        discard;
    }
    let alpha = smoothstep(half_band, -half_band, sdist) * LINK_ALPHA;
    return vec4f(LINK_COLOR, alpha);
}
`;
