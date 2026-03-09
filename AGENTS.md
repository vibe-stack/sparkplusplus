You are my senior graphics/engine engineer. Build a new THREE.js-first Gaussian splat engine for the web, targeting `THREE.WebGPURenderer` as the primary path.

Reality and constraints:
- This MUST use three.js scene semantics. No custom standalone renderer architecture that ignores THREE. The engine must integrate with `THREE.Scene`, `Object3D`, `Camera`, and `THREE.WebGPURenderer`.
- `THREE.WebGPURenderer` is the primary backend. Keep the code organized so a degraded fallback path can exist later, but DO NOT architect around WebGL first.
- three.js currently supports `WebGPURenderer`, with WebGPU backend by default and WebGL2 fallback when WebGPU is unavailable.
- three.js also exposes WebGPU-oriented primitives such as `StorageInstancedBufferAttribute`, and `BundleGroup` exists for render-bundle-oriented grouping.
- Spark 2.0 already proves the correct strategic direction: LoD trees, fixed device-dependent splat budgets, streaming/paged splats, composite LoD across multiple splat objects, and tunable foveation. Spark 2.0 still keeps sorting as a central concept, but this rewrite should push much harder toward a GPU-driven and budget-driven design.

Mission:
Design and implement a production-oriented splat subsystem that feels native to three.js, but is architected for:
- buttery frame pacing on regular iPhones
- large composite worlds
- rich effects
- animation support
- editing support
- multiple splat objects in one scene
- coexistence with normal triangle meshes and standard three.js workflows

Non-negotiable architecture goals:
1. Keep CPU out of the hot path as much as possible.
2. Never let frame cost scale with total world splat count.
3. Budget BOTH visible splats and expected screen-space overdraw.
4. Use a hierarchical paged asset model.
5. Use GPU-driven visibility/LoD scheduling.
6. Do NOT make global exact sorting the baseline.
7. Use tile/bin-oriented compositing after visibility selection.
8. Effects and animation must be structured so they only touch active/visible data whenever possible.
9. The renderer must include a ruthless runtime quality governor.
10. The public API must feel like a natural THREE extension.

Target public API:
- `SplatMesh extends THREE.Object3D`
- `SplatSkinnedMesh extends SplatMesh`
- `SplatMaterial`
- `SplatEffectStack`
- `SplatSource`
- `SplatRendererBridge extends THREE.Object3D`
- `SplatStats`
- `SplatQualityGovernor`

High-level design to implement:
A. Asset pipeline / runtime format
- Build a hierarchical cluster tree / LoD tree per splat asset.
- Store runtime data in fixed-size pages.
- Each cluster/node should include:
  - bounds
  - projected error coefficient
  - opacity mass
  - anisotropy severity
  - expected overdraw score
  - motion sensitivity
  - semantic flags (hero, face, hands, foliage, glass, deformable, etc.)
  - parent/child references
  - page residency handle
- Each page is the atomic unit of:
  - streaming
  - residency
  - visibility
  - animation update
  - effect update
  - debugging
- Canonical splat data should be stored in a GPU-friendly SoA layout.
- Design for future importers, but bootstrap with a simple internal JSON/binary test format plus procedural generators.

B. THREE integration model
- three.js owns scene graph semantics, transforms, cameras, mesh composition, and overall render orchestration.
- Our splat subsystem must not do naive full scene traversal and CPU-heavy interpretation every frame.
- Instead:
  - maintain a compact scene descriptor buffer
  - detect dirty transforms / params on CPU
  - upload only compact per-frame changes
  - let GPU passes handle culling, scoring, frontier selection, and tile assignment
- `SplatRendererBridge` should integrate cleanly into the three.js render loop and support multiple splat objects in one scene.

C. Visibility scheduler
Implement a GPU-driven scheduler with these passes:
1. object/cluster update pass
2. cluster cull + score pass
3. LoD frontier selection pass
4. residency request emission pass
5. active page expansion pass

The scheduler must optimize against:
- max visible splats
- max overdraw budget
- max active pages
- max deformable pages
- foveation weighting
- temporal stability

D. Compositing model
Do NOT make full global sort the default.
Implement a staged tile/bin compositor:
- Tile/bin active splats after visibility selection.
- Classify each tile into one of three modes:
  1. weighted blended OIT for cheap/far/peripheral tiles
  2. depth-sliced bucket compositing for medium complexity tiles
  3. hero exact-ish local resolve for a tiny number of critical tiles
- The engine must be architected so mode 1 ships first, mode 2 second, mode 3 third.
- The first usable version should already be good with mode 1 + tiny hero path only.

E. Animation model
Animation tiers:
1. object/cluster transforms
2. sparse page-local high-detail deformation
3. full per-splat deformation only as opt-in and runtime-throttled
Implement data structures for:
- rigid cluster transforms
- optional skeletal binding
- optional low-rank deformation basis
- optional residual deformation channels
Only animate resident visible pages whenever possible.

F. Effects system
Implement a staged compute-style effect graph:
1. object-space effects
2. visible-page effects
3. tile-space resolve effects

Examples of supported effects:
- recolor
- opacity masks
- dissolve
- noise displacement
- pulse
- local edge emphasis
- artifact smoothing
- semantic region modulation

Effects must be architected to avoid “touch every splat every frame”.

G. Temporal coherence
Aggressively reuse:
- previous LoD frontier
- previous active pages
- previous tile assignments
- previous tile classifications
- previous artifact hotspots
- previous scores where possible
Incrementally update when motion is small.

H. Runtime quality governor
Implement a governor that monitors:
- GPU time
- CPU time
- page fault rate
- tile depth complexity
- dropped-frame pattern
- thermal-risk proxy hooks (design API even if exact browser thermal APIs are limited)

Degrade quality in this order:
1. reduce hero exact tiles
2. increase peripheral foveation
3. reduce deformation density
4. reduce effect update cadence
5. cut overdraw-heavy frontier nodes
6. lower visible splat budget
7. raise min projected splat size
8. lower internal render scale only as last resort

I. Development philosophy
- Build for shipping, not for research purity.
- Prefer phased delivery.
- Keep internals measurable and debuggable.
- Every subsystem must expose stats and debug visualization hooks.
- Design for test scenes that can prove the budget system works.

Implementation requirements:
1. Use TypeScript.
2. Use modern three.js patterns compatible with `THREE.WebGPURenderer`.
3. Organize the code as a reusable package, not a one-off demo.
4. Create a clean module structure from day one.
5. Avoid over-engineering editor tooling at the beginning.
6. Use procedural or synthetic test content first.
7. Prioritize correctness of scheduling/budgeting architecture over file format breadth.
8. Write concise but useful inline comments.
9. Add profiling-oriented instrumentation and debug flags.
10. Every phase must end with a runnable demo scene.

Deliverables required from you:
1. A clear repo structure.
2. A phased roadmap.
3. A concrete architectural overview.
4. The core TypeScript interfaces/classes.
5. The initial GPU buffer layouts.
6. The frame graph / render graph design.
7. A milestone-by-milestone implementation plan.
8. Risk analysis and fallback decisions.
9. A first bootstrap implementation skeleton.

Now produce the output in this exact structure:

# 1. Executive summary
Explain the architecture in direct engineering language.

# 2. Final architectural decisions
State the chosen architecture plainly and explain why it beats a CPU-sorted or globally sorted baseline.

# 3. Repo structure
Propose a practical monorepo or package layout.

# 4. Core runtime modules
List each module, its responsibility, and its public API surface.

# 5. Data model
Define:
- scene descriptor
- cluster metadata
- page table
- active frontier
- active page list
- tile lists
- compositor queues
- effect graph data
- stats buffers

# 6. GPU pipeline
Describe the exact frame passes in order and the inputs/outputs of each pass.

# 7. Compositing strategy
Describe the three compositor tiers, shipping order, artifacts, and fallback behavior.

# 8. Animation and effects strategy
Explain how animation/effects are budgeted and confined to visible data.

# 9. Runtime governor
Define the heuristics, signals, thresholds, and downgrade order.

# 10. Debugging and instrumentation
List overlays, counters, timers, and visual debug modes.

# 11. Phased roadmap
Give milestones:
- M0 foundation
- M1 asset/page system
- M2 GPU visibility
- M3 baseline compositor
- M4 governor + temporal reuse
- M5 animation/effects
- M6 hero compositor
- M7 polish/demo

For each milestone include:
- goals
- concrete tasks
- exit criteria
- risks

# 12. Bootstrap code skeleton
Provide the initial TypeScript file tree and minimal starter code for key files.

# 13. First implementation target
Define the smallest shippable demo and what “success” means.

# 14. Hard truths / tradeoffs
Be brutally honest about what is expensive, what is approximate, and what should be deferred.

Important output rules:
- Be decisive.
- Do not hedge constantly.
- Do not wander into generic graphics talk.
- Do not suggest abandoning three.js.
- Do not suggest centering the first version around exact global sorting.
- Treat this as a real engine project, not a toy.
- Prefer implementation-ready specifics over broad theory.