# ts-surface-elements

Render textures on the surface of organic shapes using WebGPU.

View a live version [on GitHub Pages](https://a-johanson.github.io/ts-surface-elements/). Note that this project requires WebGPU's `subgroups` feature, which is currently only available in Chromium-based browsers.

## Method

The surface of the model is represented as a signed distance function (SDF). For initialization, seed points are drawn randomly close to the surface via rejection sampling and are then projected onto the surface via a couple of Newton-Raphson steps. In each time step, each particle is repelled by any other particle within a certain radius and is advected by the tangential component of the accumulated force exerted by all other particles within this radius. The strength of the force is inversely proportional to the Euclidean distance of each particle. After each advection step, a Newton step re-projects the points onto the zero-level set of the SDF so the particles stay on the surface.

The force kernel is accelerated by using a spatial grid to avoid quadratic complexity. To build the spatial grid on the GPU, radix sort with a naive prefix scan implementation (utilizing only a single workgroup) is used.

## Getting Started

To run the project locally, you need Node.js and npm.

1. Run `npm install`.
2. Run `npm run dev` to start the local development server.
3. Open your browser and navigate to `http://localhost:8000`.

## Development

* `npm run build` bundles the app into `public/js/main.js`.
* `npm run check` runs the TypeScript and Biome checks without rewriting files.
* `npm run lint` runs the TypeScript checks and applies Biome formatting and lint fixes.
* `npm run install-hook` installs the Git pre-commit hook.
