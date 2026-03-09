# Spark++

Spark++ is a Bun workspace for a high-end Gaussian splat engine targeting three.js WebGPU.

## Current state

Spark++ currently has the `M0` through `M3` prototype path in place:

- real SPZ import into a hierarchical paged runtime asset
- a mixed-depth bootstrap frontier scheduler inside three.js scene semantics
- optional GPU visibility readback feeding scheduler scores
- a baseline sprite compositor that coexists with normal three.js meshes

This is not yet the final performance architecture. The current demo still expands active pages and builds compositor queues on the CPU, and the baseline compositor is still an approximation rather than the final tiled resolve path.

## Known limitations

- Close-up coherence and frame pacing on large multi-million-splat scans are still being hardened.
- GPU visibility exists, but the playground currently defaults to the CPU scheduler path because the readback path still needs more stability work.
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
