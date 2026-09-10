# ts-surface-elements: render textures on the surface of organic shapes using WebGL

## Structural overview

Use this section to orient before opening files. Prefer the smallest relevant slice of the tree instead of reading every same-named variant. Keep this section up to date whenever a change alters module boundaries, file ownership, or the main data flow between components.

This repository is currently a small canvas app with a thin TypeScript entrypoint and a static public shell.

* `src/main.ts` is the application entrypoint consumed by esbuild. It waits for DOM readiness, resolves `canvas#outputCanvas`, initializes the canvas element, and delegates one-frame drawing to the renderer.
* `src/canvas-dimensions.ts` defines the `CanvasDimensions` value object that stores the physical canvas size in centimeters plus its derived pixel dimensions for a given DPI.
* `src/render.ts` owns the single-frame renderer.
* `src/sfc32.ts` provides the SFC32 pseudo-random number generator used by the renderer.
* `public/index.html` is the browser shell. It defines one `canvas#outputCanvas`, loads `public/style.css`, and boots the bundled module from `public/js/main.js`.
* `public/style.css` holds the base page and canvas presentation styles.
* `package.json` owns the development workflow: `dev` serves `public/` while bundling `src/main.ts`, `build` emits the production bundle to `public/js/`, `check` runs TypeScript plus Biome validation, and `lint` runs the same checks with Biome write-fixes enabled.
* `biome.json` and `tsconfig.json` define formatting, linting, and TypeScript compiler behavior for the whole project.

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
