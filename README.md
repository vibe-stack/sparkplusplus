# Spark++

Comment by a human: What you're reading below was 100% ai generated, in fact, everything in this repo except for this comment is. I have no idea what any of it means nor if it's even a good idea, but we vibe

Spark++ is a Bun workspace for a high-end Gaussian splat engine targeting three.js WebGPU.

## Current state

Spark++ currently has prototype paths in place:

- real SPZ import into a hierarchical paged runtime asset
- a mixed-depth bootstrap scheduler inside three.js scene semantics
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
