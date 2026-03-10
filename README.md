# Spark++

Comment by a human: What you're reading below was 100% ai generated, in fact, everything in this repo except for this comment is. I have no idea what any of it means nor if it's even a good idea, but we vibe

Spark++ is a Bun workspace for a high-end Gaussian splat engine targeting three.js WebGPU.

## Current state

Spark++ currently has prototype paths in place:

- real SPZ import into a hierarchical paged runtime asset
- a mixed-depth bootstrap scheduler inside three.js scene semantics
- optional GPU visibility readback feeding scheduler scores
- a tile-classified multi-queue sprite compositor that coexists with normal three.js meshes

This is not yet the final performance architecture. The current demo still expands active pages and builds compositor queues on the CPU, but the compositor path is already tile-classified rather than a single flat sprite pass.

## Known limitations

- Close-up coherence and frame pacing on large multi-million-splat scans are still being hardened.
- GPU visibility readback is enabled by default; invalid or late readbacks now fall back to the most recent good frame before dropping to CPU scheduling.
- Medium-complexity depth-sliced resolve and local hero resolve are not finished yet.

## Workspace

- `packages/spark`: reusable library package
- `apps/playground`: Vite + TypeScript playground app

## Getting started

```sh
bun install
bun run dev
```

## Common commands

```sh
bun run build
bun run typecheck
```

## Playground deployment

The playground is configured for GitHub Pages at:

`https://vibe-stack.github.io/sparkplusplus`

The repository includes a GitHub Actions workflow that builds with Bun and deploys `apps/playground/dist` to Pages on pushes to `main`.
