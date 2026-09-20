# ts-surface-elements: render textures on the surface of organic shapes using WebGPU

This repository is a WebGPU canvas app that renders elements on the surface of organic shapes, with a thin TypeScript entrypoint and a static public shell.

## Structural overview

Use this section to orient before opening files. It describes module **responsibilities and ownership boundaries**, not the specific implementation details those modules currently happen to use. Add or modify a line only when a module's responsibility itself changes, when a new module appears, or when data flow between modules changes.

* `src/main.ts` is the application entrypoint. It resolves the canvas, initializes WebGPU, creates the pipelines and orbit camera, submits a one-shot seed dispatch, and runs the frame loop that relaxes and renders the points.
* `src/webgpu.ts` owns WebGPU initialization and canvas backing-store size synchronization. It exports the `GpuContext` interface bundling the device, canvas context, format, and canvas element.
* `src/orbit-controls.ts` owns the orbit camera — a mouse-driven spherical camera (drag to rotate, wheel to zoom) that exposes the eye position for view-matrix construction.
* `src/stipple` owns the surface-stippling pipeline — distributing points on the surface of an SDF scene. Data flow is one-directional: seed → relax → render.
  * `shaders.ts` owns all WGSL source, including a shared block (the scene SDF, gradient/normal helpers, and a surface-projection routine) interpolated into the seed, relax, and debug-render shaders.
  * `point-buffers.ts` owns the 3D point data layout and the ping-pong storage buffer pair.
  * `seed-pipeline.ts` owns the one-shot seed compute pass: rejection sampling in a bounding box followed by surface projection. Writes buffer A at bootstrap; view-independent.
  * `relax-pipeline.ts` owns the per-frame relax compute pass: 3D repulsion with surface re-projection. Ping-pongs between the two point buffers; view-independent.
  * `debug-render-pipeline.ts` owns the SDF visualization — a full-screen shader that ray-marches the scene to the canvas. Owns the ray-based camera uniform and the `CameraConfig` type shared with the point renderer.
  * `point-render-pipeline.ts` owns the stipple-point renderer — billboard quads projected from world space via a view-projection matrix uniform. Reads whichever buffer relax most recently wrote.
* `public/index.html` is the browser shell. It defines the canvas, loads the stylesheet, and boots the bundled module.
* `public/style.css` holds the base page and canvas presentation styles.
* `package.json` owns the development workflow: `dev` serves `public/` while bundling the entrypoint, `build` emits the production bundle, `check` runs TypeScript plus Biome validation, and `lint` runs the same checks with Biome write-fixes enabled.

## General guardrails and style

* Organize the application logic in TypeScript in `src/`.
* Use modern ECMAScript patterns and TypeScript-specific type utilities (target ES2024 with ESNext module structure, avoid legacy TS).
* Prefer object-oriented design whenever it makes sense to combine data and logic into classes.
* Always generate TSDoc comments in accordance with the Google TypeScript Style Guide.
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
