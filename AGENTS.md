# ts-surface-elements: render textures on the surface of organic shapes using WebGPU

This repository is a WebGPU canvas app that renders elements on the surface of organic shapes, with a thin TypeScript entrypoint and a static public shell.

## Structural overview

Use this section to orient before opening files. It describes module **responsibilities and ownership boundaries**, not the specific implementation details those modules currently happen to use. Add or modify a line only when a module's responsibility itself changes, when a new module appears, or when data flow between modules changes. Do not include implementation details that might readily change and are not important to understanding the high-level structure of the implementation.

* `src/main.ts` is the application entrypoint. It resolves the canvas, initializes WebGPU, creates the pipelines and orbit camera, submits a one-shot seed dispatch, and runs the frame loop that reprojects, builds the spatial grid, relaxes, shades, and renders the points. It computes a capped wall-clock animation time that threads through all per-frame pipelines.
* `src/webgpu.ts` owns WebGPU initialization and canvas backing-store size synchronization. It exports the `GpuContext` interface bundling the device, canvas context, format, and canvas element.
* `src/orbit-controls.ts` owns the orbit camera — a mouse-driven spherical camera (drag to rotate, wheel to zoom) that exposes the eye position for view-matrix construction.
* `src/stipple` owns the surface-stippling pipeline — distributing points on the surface of a time-animated SDF scene. Data flow is one-directional: seed → reproject → spatial grid build → relax → shade → render.
  * `shaders/` owns all WGSL source, with one file per shader plus a shared common block. `index.ts` is a barrel re-exporting every shader string.
  * `shaders/sdf-common.ts` owns the shared WGSL block interpolated into every other shader that touches the SDF: the `Point` storage struct, the time-parameterized scene SDF `map(p, time)`, gradient/normal helpers, a Newton-Raphson `projectToSurface` routine, a sphere-tracing `rayMarch` helper, and a `softShadow` penumbra estimator.
  * `shaders/subgroup-common.ts` owns shared WGSL subgroup scan helpers (subgroup-level and workgroup-level exclusive prefix sum) reused by the radix split. Requires the `"subgroups"` WebGPU feature.
  * `shaders/spatial-grid/` owns the cell-index, radix-split, and cell-ranges shaders for the spatial grid build.
  * `point-buffers.ts` owns the 3D point data layout (`pos` only, 16 bytes), the ping-pong storage buffer pair, a single shared (non-ping-pong) normals buffer, and a single shared (non-ping-pong) shading buffer. The normals buffer is written once by the seed pass at bootstrap, every frame by the reproject pass (before relax, from the buffer relax is about to read), and every frame by the shading pass (after relax, from the buffer relax most recently wrote — so it always matches the buffer the next reproject + relax cycle reads). The shading buffer is written every frame by the shading pass and read by the point renderer.
  * `seed-pipeline.ts` owns the one-shot seed compute pass: rejection sampling in a bounding box followed by surface projection at t=0. Writes buffer A and the matching normals at bootstrap; view-independent.
  * `reproject-pipeline.ts` owns the per-frame reproject compute pass: Newton-projects each point onto the current animated surface and refreshes the shared normals buffer. Runs before relax; view-independent; in-place on whichever buffer relax will read (two bind groups, A/B). Owns the `ReprojectParams` uniform (time + point_count).
  * `spatial-grid-pipeline.ts` owns the per-frame spatial grid build (cell index → radix sort → cell ranges). Runs once per frame after reproject, before relax; reads the buffer relax will read; grid buffers are read-only during relax and rebuilt wholesale next frame. Uses a fixed CPU-known `SCENE_BBOX` (shared with the seed pipeline) as origin + dims. Exposes the sorted values and cell-start/count buffers via the `GridBuffers` interface for relax to bind.
  * `relax-pipeline.ts` owns the per-frame relax compute pass: grid-accelerated curvature-aware repulsion with surface re-projection. Reads the shared normals buffer (written by reproject/shading) for curvature inflation and the spatial grid buffers (`GridBuffers`) for O(1) neighbor-cell lookup; ping-pongs between the two point buffers; view-independent.
  * `shading-pipeline.ts` owns the per-frame shading compute pass: refreshes the shared normals buffer from the relaxed positions and computes per-point visibility (sphere-trace toward the eye, occluded by the SDF surface) and luminance (Aaltonen soft-shadow penumbra toward the light + Lambert) into the shared shading buffer. Runs after relax, before render; view-dependent.
  * `debug-render-pipeline.ts` owns the SDF visualization — a full-screen shader that ray-marches the scene to the canvas and shades with Lambert modulated by Aaltonen soft shadows. Owns the ray-based camera uniform and the `CameraConfig` type shared with the point renderer.
  * `point-render-pipeline.ts` owns the stipple-point renderer — tangent-plane quads oriented from the shared normals buffer and projected via a view-projection matrix uniform. Reads whichever buffer relax most recently wrote, the shared normals buffer, and the shared shading buffer; discards occluded points in the vertex shader.
* `public/index.html` is the browser shell. It defines the canvas, loads the stylesheet, and boots the bundled module.
* `public/style.css` holds the base page and canvas presentation styles.
* `package.json` owns the development workflow: `dev` serves `public/` while bundling the entrypoint, `build` emits the production bundle, `check` runs TypeScript plus Biome validation, and `lint` runs the same checks with Biome write-fixes enabled.

## General guardrails and style

* Organize the application logic in TypeScript in `src/`.
* Use modern ECMAScript patterns and TypeScript-specific type utilities (target ES2024 with ESNext module structure, avoid legacy TS).
* Prefer object-oriented design whenever it makes sense to combine data and logic into classes.
* Always generate TSDoc comments in accordance with the Google TypeScript Style Guide.
* Do not include implementation details in TSDoc comments that might readily change and are not important to understanding the functionality.
* Use American English spelling.
* After you apply edits, run `npm run lint` to check for type errors and to apply linting & formatting.

## Collaboration and design workflow

* Before implementing substantial or potentially architectural changes, provide a short design note and wait for user approval.
* Treat a change as substantial when it introduces new abstractions, changes responsibilities, or has meaningful design uncertainty (not merely because multiple files are touched).
* If a change modifies project structure, file ownership, or render-stage data flow, update the structural overview in this file as part of the same change.
* Keep design notes concise and include:
	* problem framing and constraints,
	* plausible approaches (when applicable),
	* recommended approach with tradeoffs,
	* proposed responsibilities and data flow.

## Architecture guardrails

* Favor high cohesion and low coupling.
* Prefer composition and clear interfaces over large, multi-purpose classes.
* Keep orchestration thin and place core logic in focused components.
* Use established design patterns when they provide clear value; avoid pattern-driven overengineering.

## Pre-implementation check

Before coding, explicitly confirm to yourself:
* boundaries and dependencies are clear,
* interfaces are minimal and testable,
* the chosen design is the simplest approach that satisfies the requirements,
* you do not introduce redundant logic and types but re-use existing logic and types where appropriate.
