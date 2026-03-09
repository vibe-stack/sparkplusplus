# Spark++ Milestone Checklist

## Current status

- [x] `M0` foundation implemented
- [x] `M1` asset/page system implemented
- [x] `M2` GPU visibility prototype path implemented
- [x] `M3` baseline compositor prototype path implemented
- [ ] `M4` governor + temporal reuse polish
- [ ] `M5` animation/effects expansion
- [ ] `M6` hero compositor
- [ ] `M7` polish/demo completion

## Reality check

- [x] The playground imports the real `demo.spz` asset at full source resolution (`3,957,992` splats in the current file).
- [x] The imported runtime hierarchy is still a bootstrap tree with coarse internal pages plus leaf pages (`1365` total pages/clusters at `4096` splats per page).
- [x] The scheduler now maintains a mixed-depth frontier cut instead of atomically collapsing whole subtrees back to a root page.
- [ ] The current `M2`/`M3` path is still a prototype baseline, not yet the final performance target for dense close-up rendering on large scans.

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

## M2 GPU Visibility

- [x] Added a WebGPU compute visibility pipeline that updates cluster world spheres and writes per-cluster cull/score results into storage buffers.
- [x] Integrated asynchronous GPU readback into `SplatRendererBridge`, with CPU fallback retained when compute or readback is unavailable.
- [x] Switched the bootstrap scheduler to consume GPU visibility/projected-size results when ready, while preserving the existing budgeting and residency heuristics.
- [x] Exposed GPU visibility readiness, pending state, frame lag, cluster counts, and active scheduler mode in runtime stats and the demo HUD.

## M3 Baseline Compositor

- [x] Replaced the old debug `Points` proxy with a sprite-based baseline compositor that stays inside normal three.js scene semantics.
- [x] Added tile classification for weighted, depth-sliced, and hero tile modes driven from the active frontier and view state.
- [x] Shipped weighted blended and small hero sprite queues, with medium-complexity depth-sliced tiles currently classified and routed through the weighted baseline path.
- [x] Surfaced per-mesh and frame-wide compositor telemetry for queues, tile-class mix, and max tile complexity in the playground.

## Deferred by milestone

- [ ] `M4`: harden temporal reuse and make the governor drive all downgrade stages automatically from measured timing.
- [ ] `M5`: add visible-page deformation, skeletal binding execution, and richer effect graph stages.
- [ ] `M6`: add local hero resolve tiles instead of the current baseline hero sprite queue.
- [ ] `M7`: finish polish, stress scenes, and shippable demo tuning.

## Remaining gaps

- [ ] Promote close-up dense-scan coherence and frame pacing on the full `demo.spz` asset from prototype quality to milestone-quality.
- [ ] Harden the GPU visibility readback path enough to be the default demo path again without frontier instability.
- [ ] Move frontier compaction, residency request emission, and active-page expansion fully onto GPU-side list generation instead of the current GPU-readback-plus-CPU-budget path.
- [ ] Add a real depth-sliced medium-complexity resolve instead of routing those tiles through weighted blending.

## Verification

- [x] `packages/spark` typecheck passes
- [x] `apps/playground` typecheck passes
- [x] full workspace build
