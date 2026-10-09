/**
 * Bounding-box wireframe shader — projects the 12 edges of the scene
 * bounding box with a view-projection matrix for drawing as a line list
 * on top of the debug ray-march view.
 *
 * Cube corners are indexed by bits (bit 0 = x, bit 1 = y, bit 2 = z); the
 * vertex shader expands 24 vertices (12 edges × 2 endpoints) with no
 * vertex or index buffers.
 */

export const BBOX_WIREFRAME_SHADER = /* wgsl */ `
struct BBoxWireframe {
    view_proj: mat4x4f,
    bbox_min: vec3f,
    _pad0: f32,
    bbox_max: vec3f,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> bbox: BBoxWireframe;

@vertex
fn bbox_vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    let edges = array<vec2u, 12>(
        vec2u(0u, 1u), vec2u(1u, 3u), vec2u(3u, 2u), vec2u(2u, 0u),
        vec2u(4u, 5u), vec2u(5u, 7u), vec2u(7u, 6u), vec2u(6u, 4u),
        vec2u(0u, 4u), vec2u(1u, 5u), vec2u(3u, 7u), vec2u(2u, 6u),
    );
    let edge = edges[vid / 2u];
    let i = select(edge.x, edge.y, (vid & 1u) != 0u);
    let x = select(bbox.bbox_min.x, bbox.bbox_max.x, (i & 1u) != 0u);
    let y = select(bbox.bbox_min.y, bbox.bbox_max.y, (i & 2u) != 0u);
    let z = select(bbox.bbox_min.z, bbox.bbox_max.z, (i & 4u) != 0u);
    return bbox.view_proj * vec4f(x, y, z, 1.0);
}

@fragment
fn bbox_fs() -> @location(0) vec4f {
    return vec4f(0.2, 0.8, 0.4, 1.0);
}
`;
