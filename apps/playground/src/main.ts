import './style.css';
import {
  SPARK_ENGINE_DESCRIPTOR,
  SPARK_FRAME_GRAPH,
  SplatMaterial,
  SplatMesh,
  SplatQualityGovernor,
  SplatRendererBridge,
  SpzSplatSource,
  getSparkBanner,
  type SplatFrameStatsSnapshot,
} from '@sparkplusplus/spark';
import {
  ACESFilmicToneMapping,
  AmbientLight,
  Box3,
  Clock,
  Color,
  DirectionalLight,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  SRGBColorSpace,
  Vector3,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { WebGPURenderer } from 'three/webgpu';

const app = document.querySelector<HTMLDivElement>('#app');

interface PlaygroundComplexSplatPreset {
  name: string;
  importPointCap: number | null;
  pageCapacity: number;
  visibleBudget: number;
  overdrawBudget: number;
  activePages: number;
  residentPages: number;
  uploadsPerFrame: number;
  coverageBiasPx: number;
  peripheralFoveation: number;
  heroTiles: number;
  temporalStabilityBias: number;
  pointSize: number;
  opacity: number;
  useGpuVisibilityReadback: boolean;
  orbitRadiusScale: number;
  orbitHeightScale: number;
  orbitDepthScale: number;
  lookHeightScale: number;
}

const COMPLEX_SPLAT_PRESET: PlaygroundComplexSplatPreset = {
  name: 'Coverage',
  importPointCap: null,
  pageCapacity: 8_192,
  visibleBudget: 4_000_000,
  overdrawBudget: 600_000,
  activePages: 256,
  residentPages: 512,
  uploadsPerFrame: 64,
  coverageBiasPx: 36,
  peripheralFoveation: 0,
  heroTiles: 20,
  temporalStabilityBias: 1.08,
  pointSize: 6,
  opacity: 0.99,
  useGpuVisibilityReadback: true,
  orbitRadiusScale: 0.66,
  orbitHeightScale: 0.18,
  orbitDepthScale: 0.58,
  lookHeightScale: 0.28,
};

class LockedQualityGovernor extends SplatQualityGovernor {
  override observe(snapshot: SplatFrameStatsSnapshot) {
    return {
      level: this.getLevel(),
      reason: 'locked',
      budgets: this.getBudgets(),
    };
  }
}

if (!app) {
  throw new Error('Missing #app root');
}

app.innerHTML = `
  <div class="app-shell">
    <aside class="hud">
      <section class="stack-card hero-card">
        <p class="eyebrow">Spark++ milestone bootstrap</p>
        <h1>${getSparkBanner()}</h1>
        <p class="body">
          The demo imports the real <strong>demo.spz</strong> asset, runs GPU cluster visibility/scoring,
          and draws the active frontier through the baseline sprite compositor while ordinary three.js meshes
          still share the same scene.
        </p>
      </section>

      <section class="stack-card">
        <div class="card-header">
          <h2>Frame Telemetry</h2>
          <span class="pill">${SPARK_ENGINE_DESCRIPTOR.stage}</span>
        </div>
        <dl class="metric-grid">
          <div>
            <dt>CPU frame</dt>
            <dd id="metric-frame">0.0 ms</dd>
          </div>
          <div>
            <dt>GPU visibility</dt>
            <dd id="metric-gpu">warming up</dd>
          </div>
          <div>
            <dt>Scheduler</dt>
            <dd id="metric-scheduler">CPU</dd>
          </div>
          <div>
            <dt>Visible splats</dt>
            <dd id="metric-splats">0</dd>
          </div>
          <div>
            <dt>Active pages</dt>
            <dd id="metric-pages">0</dd>
          </div>
          <div>
            <dt>Resident pages</dt>
            <dd id="metric-resident">0</dd>
          </div>
          <div>
            <dt>Page uploads</dt>
            <dd id="metric-uploads">0</dd>
          </div>
          <div>
            <dt>Page fault rate</dt>
            <dd id="metric-faults">0%</dd>
          </div>
          <div>
            <dt>Frontier stability</dt>
            <dd id="metric-stability">0%</dd>
          </div>
          <div>
            <dt>Tile mix</dt>
            <dd id="metric-tiles">0 active</dd>
          </div>
          <div>
            <dt>Queues</dt>
            <dd id="metric-queues">W0 · H0</dd>
          </div>
          <div>
            <dt>Governor</dt>
            <dd id="metric-governor">L0</dd>
          </div>
        </dl>
      </section>

      <section class="stack-card">
        <div class="card-header">
          <h2>Scene Budget</h2>
          <span class="pill" id="budget-object-count">0 objects</span>
        </div>
        <div class="budget-strip" id="budget-strip">
          waiting for first frame
        </div>
        <div class="mesh-list" id="mesh-list"></div>
      </section>

      <section class="stack-card">
        <div class="card-header">
          <h2>Frame Graph</h2>
          <span class="pill">${SPARK_FRAME_GRAPH.length} passes</span>
        </div>
        <ol class="pipeline-list">
          ${SPARK_FRAME_GRAPH.map((pass) => `<li><strong>${pass.name}</strong><span>${pass.stage}</span></li>`).join('')}
        </ol>
      </section>
    </aside>

    <main class="stage-panel">
      <div class="viewport-shell">
        <div id="viewport" class="viewport"></div>
        <div class="stage-caption">
          <p class="eyebrow">three.js scene semantics intact</p>
          <h2>Composite LoD selection, page residency, and triangle mesh coexistence</h2>
        </div>
      </div>
    </main>
  </div>
`;

const viewport = app.querySelector<HTMLDivElement>('#viewport');
const metricFrame = app.querySelector<HTMLElement>('#metric-frame');
const metricGpu = app.querySelector<HTMLElement>('#metric-gpu');
const metricScheduler = app.querySelector<HTMLElement>('#metric-scheduler');
const metricSplats = app.querySelector<HTMLElement>('#metric-splats');
const metricPages = app.querySelector<HTMLElement>('#metric-pages');
const metricResident = app.querySelector<HTMLElement>('#metric-resident');
const metricUploads = app.querySelector<HTMLElement>('#metric-uploads');
const metricFaults = app.querySelector<HTMLElement>('#metric-faults');
const metricStability = app.querySelector<HTMLElement>('#metric-stability');
const metricTiles = app.querySelector<HTMLElement>('#metric-tiles');
const metricQueues = app.querySelector<HTMLElement>('#metric-queues');
const metricGovernor = app.querySelector<HTMLElement>('#metric-governor');
const budgetObjectCount = app.querySelector<HTMLElement>('#budget-object-count');
const budgetStrip = app.querySelector<HTMLElement>('#budget-strip');
const meshList = app.querySelector<HTMLDivElement>('#mesh-list');

if (
  !viewport
  || !metricFrame
  || !metricGpu
  || !metricScheduler
  || !metricSplats
  || !metricPages
  || !metricResident
  || !metricUploads
  || !metricFaults
  || !metricStability
  || !metricTiles
  || !metricQueues
  || !metricGovernor
  || !budgetObjectCount
  || !budgetStrip
  || !meshList
) {
  throw new Error('Playground UI is incomplete');
}

const ui = {
  viewport,
  metricFrame,
  metricGpu,
  metricScheduler,
  metricSplats,
  metricPages,
  metricResident,
  metricUploads,
  metricFaults,
  metricStability,
  metricTiles,
  metricQueues,
  metricGovernor,
  budgetObjectCount,
  budgetStrip,
  meshList,
};

function formatInteger(value: number): string {
  return new Intl.NumberFormat('en-US').format(Math.round(value));
}

function isEditableTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && (
    target.isContentEditable
    || target instanceof HTMLInputElement
    || target instanceof HTMLTextAreaElement
    || target instanceof HTMLSelectElement
  );
}

function updateHud(snapshot: SplatFrameStatsSnapshot): void {
  ui.metricFrame.textContent = `${snapshot.cpuFrameMs.toFixed(2)} ms`;
  ui.metricGpu.textContent = snapshot.gpuVisibilityReady
    ? `ready · ${snapshot.gpuClusterCount} clusters · ${snapshot.gpuVisibilityFrameLag}f lag`
    : snapshot.gpuVisibilityPending
      ? 'dispatching'
      : 'cpu fallback';
  ui.metricScheduler.textContent = snapshot.schedulerMode === 'gpu-readback' ? 'GPU readback' : 'CPU bootstrap';
  ui.metricSplats.textContent = formatInteger(snapshot.visibleSplats);
  ui.metricPages.textContent = `${snapshot.activePages} / ${snapshot.budgets.maxActivePages}`;
  ui.metricResident.textContent = `${snapshot.residentPages} / ${snapshot.budgets.maxResidentPages}`;
  ui.metricUploads.textContent = `${snapshot.pageUploads} / ${snapshot.budgets.maxPageUploadsPerFrame}`;
  ui.metricFaults.textContent = `${(snapshot.pageFaultRate * 100).toFixed(0)}%`;
  ui.metricStability.textContent = `${(snapshot.frontierStability * 100).toFixed(0)}%`;
  ui.metricTiles.textContent = [
    `${snapshot.compositorActiveTiles} active`,
    `${snapshot.compositorHeroTiles} hero`,
    `${snapshot.compositorDepthSlicedTiles} depth`,
  ].join(' · ');
  ui.metricQueues.textContent = [
    `W${formatInteger(snapshot.compositorWeightedInstances)}`,
    `H${formatInteger(snapshot.compositorHeroInstances)}`,
    `D${formatInteger(snapshot.compositorDepthSlicedInstances)}`,
  ].join(' · ');
  ui.metricGovernor.textContent = `L${snapshot.appliedGovernorLevel} · ${snapshot.governorReason}`;
  ui.budgetObjectCount.textContent = `${snapshot.meshCount} objects`;
  ui.budgetStrip.textContent = [
    `${formatInteger(snapshot.budgets.maxVisibleSplats)} splat budget`,
    `${snapshot.estimatedOverdraw.toFixed(0)} / ${snapshot.budgets.maxOverdrawBudget} overdraw`,
    `${snapshot.compositorWeightedTiles}/${snapshot.compositorDepthSlicedTiles}/${snapshot.compositorHeroTiles} tile classes`,
    `${(snapshot.sceneDescriptorBytes / 1024).toFixed(1)} KB scene descriptors`,
    `${((snapshot.clusterMetadataBytes + snapshot.pageDescriptorBytes + snapshot.residencyBytes) / 1024).toFixed(1)} KB runtime buffers`,
  ].join(' · ');

  ui.meshList.innerHTML = snapshot.meshStats
    .map((mesh) => `
      <article class="mesh-card">
        <div class="mesh-title">
          <strong>${mesh.meshName}</strong>
          <span>${mesh.visibleSplats} splats</span>
        </div>
        <p>
          frontier ${mesh.frontierClusters} · active ${mesh.activePages} · resident ${mesh.residentPages} ·
          requested ${mesh.requestedPages} · stability ${(mesh.frontierStability * 100).toFixed(0)}%
        </p>
        <p>
          queues W${formatInteger(mesh.weightedInstances)} · H${formatInteger(mesh.heroInstances)} ·
          D${formatInteger(mesh.depthSlicedInstances)} · tiles ${mesh.activeTiles} active /
          ${mesh.heroTiles} hero / ${mesh.depthSlicedTiles} depth
        </p>
      </article>
    `)
    .join('');
}

async function bootstrap(): Promise<void> {
  const scene = new Scene();
  scene.background = new Color(0x050814);

  ui.budgetStrip.textContent = 'Loading /demo.spz and building paged runtime asset...';

  const source = await SpzSplatSource.fromUrl('/demo.spz', {
    label: 'Demo SPZ',
    ...(COMPLEX_SPLAT_PRESET.importPointCap === null
      ? {}
      : { maxPoints: COMPLEX_SPLAT_PRESET.importPointCap }),
    pageCapacity: COMPLEX_SPLAT_PRESET.pageCapacity,
    branching: 4,
  });
  const importedAsset = source.buildAsset();
  const sampledPoints = COMPLEX_SPLAT_PRESET.importPointCap === null
    ? source.header.pointCount
    : Math.min(source.header.pointCount, COMPLEX_SPLAT_PRESET.importPointCap);
  ui.budgetStrip.textContent = [
    `${COMPLEX_SPLAT_PRESET.name} preset`,
    `Loaded ${formatInteger(source.header.pointCount)} source splats`,
    sampledPoints === source.header.pointCount
      ? 'full-resolution import'
      : `sampling ${formatInteger(sampledPoints)} into runtime pages`,
    `${COMPLEX_SPLAT_PRESET.pageCapacity}-splat pages`,
    `${COMPLEX_SPLAT_PRESET.coverageBiasPx}px coverage bias`,
  ].join(' · ');
  const assetBounds = new Box3(
    new Vector3(...importedAsset.localBoundsMin),
    new Vector3(...importedAsset.localBoundsMax),
  );
  const assetCenter = assetBounds.getCenter(new Vector3());
  const assetSize = assetBounds.getSize(new Vector3());
  const orbitRadius = Math.max(assetSize.x, assetSize.y, assetSize.z) * COMPLEX_SPLAT_PRESET.orbitRadiusScale;

  const camera = new PerspectiveCamera(42, 1, 0.1, 320);
  camera.position.set(
    orbitRadius,
    assetSize.y * COMPLEX_SPLAT_PRESET.orbitHeightScale,
    orbitRadius * COMPLEX_SPLAT_PRESET.orbitDepthScale,
  );
  const target = new Vector3(0, assetSize.y * COMPLEX_SPLAT_PRESET.lookHeightScale, 0);
  camera.lookAt(target);

  const renderer = new WebGPURenderer({
    antialias: true,
    alpha: false,
  });
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.domElement.classList.add('viewport-canvas');
  ui.viewport.append(renderer.domElement);
  await renderer.init();

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.copy(target);
  controls.minDistance = Math.max(assetSize.length() * 0.05, 0.5);
  controls.maxDistance = Math.max(assetSize.length() * 4, 12);
  controls.update();

  const resize = () => {
    const width = ui.viewport.clientWidth;
    const height = ui.viewport.clientHeight;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };

  window.addEventListener('resize', resize);
  resize();

  const pressedKeys = new Set<string>();
  const baseFlightSpeed = Math.max(assetSize.length() * 0.35, 2.5);
  const worldUp = new Vector3(0, 1, 0);
  const forward = new Vector3();
  const right = new Vector3();
  const movement = new Vector3();

  window.addEventListener('keydown', (event) => {
    if (event.repeat || isEditableTarget(event.target)) {
      return;
    }

    if ([
      'KeyW',
      'KeyA',
      'KeyS',
      'KeyD',
      'KeyQ',
      'KeyE',
      'Space',
      'ShiftLeft',
      'ShiftRight',
    ].includes(event.code)) {
      event.preventDefault();
      pressedKeys.add(event.code);
    }
  });

  window.addEventListener('keyup', (event) => {
    pressedKeys.delete(event.code);
  });

  window.addEventListener('blur', () => {
    pressedKeys.clear();
  });

  scene.add(new AmbientLight(0xffffff, 1.1));

  const sun = new DirectionalLight(0xfff4d6, 1.9);
  sun.position.set(8, 12, 6);
  scene.add(sun);

  const floor = new Mesh(
    new PlaneGeometry(Math.max(160, assetSize.x * 1.65), Math.max(120, assetSize.z * 1.65), 1, 1),
    new MeshStandardMaterial({
      color: 0x071422,
      roughness: 0.96,
      metalness: 0.08,
    }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.5;
  scene.add(floor);

  const heroSplat = new SplatMesh({
    source,
    material: new SplatMaterial({
      pointSize: COMPLEX_SPLAT_PRESET.pointSize,
      opacity: COMPLEX_SPLAT_PRESET.opacity,
      tint: 0xffffff,
      debugMode: 'albedo',
    }),
    importance: 1.5,
  });
  heroSplat.position.set(-assetCenter.x, -importedAsset.localBoundsMin[1], -assetCenter.z);
  scene.add(heroSplat);

  const bridge = new SplatRendererBridge({
    governor: new LockedQualityGovernor({
      maxVisibleSplats: COMPLEX_SPLAT_PRESET.visibleBudget,
      maxOverdrawBudget: COMPLEX_SPLAT_PRESET.overdrawBudget,
      maxActivePages: COMPLEX_SPLAT_PRESET.activePages,
      maxResidentPages: COMPLEX_SPLAT_PRESET.residentPages,
      maxPageUploadsPerFrame: COMPLEX_SPLAT_PRESET.uploadsPerFrame,
      minProjectedNodeSizePx: COMPLEX_SPLAT_PRESET.coverageBiasPx,
      peripheralFoveation: COMPLEX_SPLAT_PRESET.peripheralFoveation,
      heroTileBudget: COMPLEX_SPLAT_PRESET.heroTiles,
      temporalStabilityBias: COMPLEX_SPLAT_PRESET.temporalStabilityBias,
    }),
    useGpuVisibility: COMPLEX_SPLAT_PRESET.useGpuVisibilityReadback,
  });
  scene.add(bridge);

  const clock = new Clock();

  renderer.setAnimationLoop(() => {
    const deltaSeconds = clock.getDelta();
    movement.set(0, 0, 0);

    if (pressedKeys.size > 0) {
      forward.subVectors(controls.target, camera.position).normalize();
      right.crossVectors(forward, camera.up).normalize();

      if (pressedKeys.has('KeyW')) {
        movement.add(forward);
      }
      if (pressedKeys.has('KeyS')) {
        movement.sub(forward);
      }
      if (pressedKeys.has('KeyD')) {
        movement.add(right);
      }
      if (pressedKeys.has('KeyA')) {
        movement.sub(right);
      }
      if (pressedKeys.has('KeyE') || pressedKeys.has('Space')) {
        movement.add(worldUp);
      }
      if (pressedKeys.has('KeyQ') || pressedKeys.has('ShiftLeft') || pressedKeys.has('ShiftRight')) {
        movement.sub(worldUp);
      }

      if (movement.lengthSq() > 0) {
        movement.normalize().multiplyScalar(baseFlightSpeed * deltaSeconds);
        camera.position.add(movement);
        controls.target.add(movement);
      }
    }

    controls.update();

    const snapshot = bridge.update(scene, camera, deltaSeconds, renderer);
    updateHud(snapshot);
    renderer.render(scene, camera);
  });
}

void bootstrap().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : 'Unknown bootstrap failure';
  ui.budgetStrip.textContent = `Boot failed: ${message}`;
});
