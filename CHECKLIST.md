# Spark++ Milestone Checklist

## Current status

- [x] `M0` foundation implemented
- [x] `M1` asset/page system implemented
- [ ] `M2` GPU visibility
- [ ] `M3` baseline compositor
- [ ] `M4` governor + temporal reuse polish
- [ ] `M5` animation/effects expansion
- [ ] `M6` hero compositor
- [ ] `M7` polish/demo completion

## M0 Foundation

- [x] Replaced the single-file bootstrap export with a reusable package module structure.
- [x] Added native public API scaffolding for `SplatMesh`, `SplatSkinnedMesh`, `SplatMaterial`, `SplatEffectStack`, `SplatSource`, `SplatRendererBridge`, `SplatStats`, and `SplatQualityGovernor`.
- [x] Defined initial runtime budgets, quality-governor heuristics, semantic flags, packed buffer layouts, and frame-graph pass descriptors.
- [x] Added scene descriptor packing and dirty-transform tracking for `SplatMesh` objects inside a normal `THREE.Scene`.
- [x] Wired a runnable playground demo around `THREE.WebGPURenderer`.

## M1 Asset / Page System

- [x] Added a hierarchical procedural `SplatSource` that generates cluster trees, per-cluster pages, and GPU-ready packed metadata buffers.
- [x] Implemented fixed-size page descriptors and runtime residency tracking with request, upload, and eviction behavior.
- [x] Implemented a bootstrap global visibility scheduler that scores clusters across multiple `SplatMesh` objects against page, splat, and overdraw budgets.
- [x] Added active frontier selection, page request emission, active page expansion, and debug render-proxy syncing.
- [x] Exposed runtime telemetry for frontier stability, page faults, page uploads, resident pages, and visible splat counts.

## Deferred by milestone

- [ ] `M2`: move cluster update, cull/score, frontier selection, and page expansion from the CPU bootstrap path into real GPU passes.
- [ ] `M3`: replace the debug point proxy with the staged tile/bin compositor and weighted blended resolve.
- [ ] `M4`: harden temporal reuse and make the governor drive all downgrade stages automatically from measured timing.
- [ ] `M5`: add visible-page deformation, skeletal binding execution, and richer effect graph stages.
- [ ] `M6`: add local hero resolve tiles instead of the current debug-only proxy path.
- [ ] `M7`: finish polish, stress scenes, and shippable demo tuning.

## Verification

- [x] `packages/spark` typecheck passes
- [x] `apps/playground` typecheck passes
- [x] full workspace build
