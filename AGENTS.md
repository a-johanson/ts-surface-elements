# ts-surface-elements: render textures on the surface of organic shapes using WebGPU

This repository is a WebGPU canvas app that renders elements on the surface of organic shapes, with a thin TypeScript entrypoint and a static public shell.

## Structural overview

Use this section to orient before opening files. It describes module **responsibilities and ownership boundaries**, not the specific implementation details those modules currently happen to use. Add or modify a line only when a module's responsibility itself changes, when a new module appears, or when data flow between modules changes. Do not include implementation details that might readily change and are not important to understanding the high-level structure of the implementation.

* `src/main.ts` is the application entrypoint. It resolves the canvas, initializes WebGPU, creates the pipelines and orbit camera, submits a one-shot seed dispatch, and runs the frame loop that shades and renders the points every frame and — while time is playing — reprojects, builds the spatial grid, and relaxes, fulfilling pending frame exports after presenting. It computes a capped wall-clock animation time that threads through all per-frame pipelines.
* `src/webgpu.ts` owns WebGPU initialization and canvas sizing. It exports the `GpuContext` interface bundling the device, canvas context, format, and canvas element.
* `src/orbit-controls.ts` owns the orbit camera — a mouse-, touch-, and keyboard-driven spherical camera (drag or one-finger drag to rotate, wheel or two-finger pinch to zoom, two-finger drag to pan, WASD/Q/E to move the target) that owns the adjustable orbit-center/look-at target point and exposes both the eye position and target point for view-matrix construction.
* `src/time-controls.ts` owns animation-time playback — a window-level keyboard toggle that flips a playing flag the frame loop reads to gate the simulation passes.
* `src/frame-capture.ts` owns still-image export — a window-level keyboard trigger whose requests the frame loop fulfills after presenting, serializing the canvas to a downloadable PNG.
* `src/pipelines` owns the WebGPU pipelines.
  * `shaders/` owns all WGSL source, with one file per shader plus a shared common block.
  * `shaders/sdf-common.ts` owns the shared WGSL block interpolated into every other shader that touches the SDF: the `Point` and `ShadingSample` storage structs, the time-parameterized scene SDF `map(p, time)`, animated light direction `lightDir(time)`, gradient/normal helpers, a Newton-Raphson `projectToSurface` routine, sphere-tracing `rayMarch` and `rayClearance` helpers, a `softShadow` penumbra estimator, and the canonical splat radius plus tangent-frame construction shared by the shading pass (corner ray sampling) and the point renderer (quad corners).
  * `debug-render-pipeline.ts` owns the SDF visualization — a full-screen shader that ray-marches the scene to the canvas and shades it.
* `public/index.html` is the browser shell. It defines the canvas, loads the stylesheet, and boots the bundled module.
* `public/style.css` holds the base page and canvas presentation styles.
* `package.json` owns the development workflow: `dev` serves `public/` while bundling the entrypoint, `build` emits the production bundle, `check` runs TypeScript plus Biome validation, and `lint` runs the same checks with Biome write-fixes enabled.

## General guardrails and style

* Organize the application logic in TypeScript in `src/`.
* Use modern ECMAScript patterns and TypeScript-specific type utilities (target ES2024 with ESNext module structure, avoid legacy TS).
* Prefer object-oriented design whenever it makes sense to combine data and logic into classes.
* Always generate TSDoc comments in accordance with the Google TypeScript Style Guide.
* Do not include implementation details in TSDoc comments that might readily change and are not important to understanding the functionality.
* WGSL source lives inside backtick-delimited template literals. Never use backticks anywhere inside a shader string — including WGSL comments — because a stray backtick terminates the literal and surfaces as confusing TypeScript parse errors instead of a shader error. Refer to identifiers in shader comments as plain text, without any quoting.
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
