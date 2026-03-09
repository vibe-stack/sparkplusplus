import './style.css';
import {
  SPARK_ENGINE_DESCRIPTOR,
  SPARK_FRAME_GRAPH,
  SPLAT_SEMANTIC_FLAGS,
  SplatEffectStack,
  SplatMaterial,
  SplatMesh,
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
import { WebGPURenderer } from 'three/webgpu';

const app = document.querySelector<HTMLDivElement>('#app');

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
          <strong>M0</strong> foundation and <strong>M1</strong> asset/page residency are live.
          This demo now imports the real <strong>demo.spz</strong> asset from the public folder and
          builds a paged hierarchy from the source data before the future GPU compositor stages arrive.
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
const metricSplats = app.querySelector<HTMLElement>('#metric-splats');
const metricPages = app.querySelector<HTMLElement>('#metric-pages');
const metricResident = app.querySelector<HTMLElement>('#metric-resident');
const metricUploads = app.querySelector<HTMLElement>('#metric-uploads');
const metricFaults = app.querySelector<HTMLElement>('#metric-faults');
const metricStability = app.querySelector<HTMLElement>('#metric-stability');
const metricGovernor = app.querySelector<HTMLElement>('#metric-governor');
const budgetObjectCount = app.querySelector<HTMLElement>('#budget-object-count');
const budgetStrip = app.querySelector<HTMLElement>('#budget-strip');
const meshList = app.querySelector<HTMLDivElement>('#mesh-list');

if (
  !viewport
  || !metricFrame
  || !metricSplats
  || !metricPages
  || !metricResident
  || !metricUploads
  || !metricFaults
  || !metricStability
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
  metricSplats,
  metricPages,
  metricResident,
  metricUploads,
  metricFaults,
  metricStability,
  metricGovernor,
  budgetObjectCount,
  budgetStrip,
  meshList,
};

function formatInteger(value: number): string {
  return new Intl.NumberFormat('en-US').format(Math.round(value));
}

function updateHud(snapshot: SplatFrameStatsSnapshot): void {
  ui.metricFrame.textContent = `${snapshot.cpuFrameMs.toFixed(2)} ms`;
  ui.metricSplats.textContent = formatInteger(snapshot.visibleSplats);
  ui.metricPages.textContent = `${snapshot.activePages} / ${snapshot.budgets.maxActivePages}`;
  ui.metricResident.textContent = `${snapshot.residentPages} / ${snapshot.budgets.maxResidentPages}`;
  ui.metricUploads.textContent = `${snapshot.pageUploads} / ${snapshot.budgets.maxPageUploadsPerFrame}`;
  ui.metricFaults.textContent = `${(snapshot.pageFaultRate * 100).toFixed(0)}%`;
  ui.metricStability.textContent = `${(snapshot.frontierStability * 100).toFixed(0)}%`;
  ui.metricGovernor.textContent = `L${snapshot.appliedGovernorLevel} · ${snapshot.governorReason}`;
  ui.budgetObjectCount.textContent = `${snapshot.meshCount} objects`;
  ui.budgetStrip.textContent = [
    `${formatInteger(snapshot.budgets.maxVisibleSplats)} splat budget`,
    `${snapshot.estimatedOverdraw.toFixed(0)} / ${snapshot.budgets.maxOverdrawBudget} overdraw`,
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
    maxPoints: 320_000,
    pageCapacity: 1_024,
    branching: 4,
  });
  const importedAsset = source.buildAsset();
  const assetBounds = new Box3(
    new Vector3(...importedAsset.localBoundsMin),
    new Vector3(...importedAsset.localBoundsMax),
  );
  const assetCenter = assetBounds.getCenter(new Vector3());
  const assetSize = assetBounds.getSize(new Vector3());
  const orbitRadius = Math.max(assetSize.x, assetSize.y, assetSize.z) * 0.9;

  const camera = new PerspectiveCamera(42, 1, 0.1, 320);
  camera.position.set(orbitRadius, assetSize.y * 0.32, orbitRadius * 0.72);

  const renderer = new WebGPURenderer({
    antialias: true,
    alpha: false,
  });
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.domElement.classList.add('viewport-canvas');
  ui.viewport.append(renderer.domElement);
  await renderer.init();

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
      pointSize: 6.4,
      opacity: 0.96,
      tint: 0xf7fbff,
      debugMode: 'albedo',
    }),
    effects: new SplatEffectStack([
      {
        id: 'spz-pulse',
        kind: 'pulse',
        intensity: 0.35,
        frequencyHz: 0.16,
        semanticMask: SPLAT_SEMANTIC_FLAGS.hero,
      },
      {
        id: 'spz-glow',
        kind: 'recolor',
        intensity: 0.14,
        tint: 0xe6f6ff,
      },
    ]),
    importance: 1.5,
  });
  heroSplat.position.set(-assetCenter.x, -importedAsset.localBoundsMin[1], -assetCenter.z);
  scene.add(heroSplat);

  const bridge = new SplatRendererBridge({
    budgets: {
      maxVisibleSplats: 24_000,
      maxOverdrawBudget: 1_650,
      maxActivePages: 24,
      maxResidentPages: 40,
      maxPageUploadsPerFrame: 3,
    },
  });
  scene.add(bridge);

  const clock = new Clock();
  const target = new Vector3(0, assetSize.y * 0.38, 0);

  renderer.setAnimationLoop(() => {
    const deltaSeconds = clock.getDelta();
    const elapsedSeconds = clock.getElapsedTime();
    const orbitAngle = elapsedSeconds * 0.11;
    camera.position.set(
      Math.cos(orbitAngle) * orbitRadius,
      Math.max(assetSize.y * 0.14, assetSize.y * 0.22 + Math.sin(elapsedSeconds * 0.21) * assetSize.y * 0.04),
      Math.sin(orbitAngle) * orbitRadius * 0.72,
    );
    camera.lookAt(target);

    const snapshot = bridge.update(scene, camera, deltaSeconds, renderer);
    updateHud(snapshot);
    renderer.render(scene, camera);
  });
}

void bootstrap().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : 'Unknown bootstrap failure';
  ui.budgetStrip.textContent = `Boot failed: ${message}`;
});
