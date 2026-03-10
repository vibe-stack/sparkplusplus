import './style.css';
import {
  SplatMaterial,
  SplatMesh,
  SplatQualityGovernor,
  SplatRendererBridge,
  SpzSplatSource,
} from '@sparkplusplus/spark';
import {
  AmbientLight,
  Box3,
  Clock,
  Color,
  DirectionalLight,
  Mesh,
  MeshStandardMaterial,
  NoToneMapping,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  SRGBColorSpace,
  Vector3,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { max, oneMinus, pass, vec4 } from 'three/tsl';
import { PostProcessing, WebGPURenderer } from 'three/webgpu';

const app = document.querySelector<HTMLDivElement>('#app');

interface PlaygroundComplexSplatPreset {
  importPointCap: number | null;
  pageCapacity: number;
  minLeafPoints: number;
  branching: number;
  visibleBudget: number;
  overdrawBudget: number;
  activePages: number;
  residentPages: number;
  uploadsPerFrame: number;
  minProjectedNodeSizePx: number;
  peripheralFoveation: number;
  heroTiles: number;
  temporalStabilityBias: number;
  pointSize: number;
  opacity: number;
  colorGain: number;
  targetFrameMs: number;
  minPixelRatio: number;
  maxPixelRatio: number;
  useGpuVisibilityReadback: boolean;
  orbitRadiusScale: number;
  orbitHeightScale: number;
  orbitDepthScale: number;
  lookHeightScale: number;
}

const COMPLEX_SPLAT_PRESET: PlaygroundComplexSplatPreset = {
  importPointCap: null,
  pageCapacity: 1_024,
  minLeafPoints: 1024,
  branching: 8,
  visibleBudget: 1_000_000,
  overdrawBudget: 260_000,
  activePages: 768,
  residentPages: 2_048,
  uploadsPerFrame: 320,
  minProjectedNodeSizePx: 3,
  peripheralFoveation: 2.2,
  heroTiles: 0,
  temporalStabilityBias: 1.2,
  pointSize: 15.15,
  opacity: 1,
  colorGain: 1.23,
  targetFrameMs: 32,
  minPixelRatio: 0.5,
  maxPixelRatio: 1.5,
  useGpuVisibilityReadback: false,
  orbitRadiusScale: 0.96,
  orbitHeightScale: 0.28,
  orbitDepthScale: 0.88,
  lookHeightScale: 0.38,
};

if (!app) {
  throw new Error('Missing #app root');
}

app.innerHTML = '<div id="viewport" class="viewport"></div>';

const viewportElement = app.querySelector<HTMLDivElement>('#viewport');

if (!viewportElement) {
  throw new Error('Missing viewport root');
}

const viewport: HTMLDivElement = viewportElement;

function isEditableTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && (
    target.isContentEditable
    || target instanceof HTMLInputElement
    || target instanceof HTMLTextAreaElement
    || target instanceof HTMLSelectElement
  );
}

async function bootstrap(): Promise<void> {
  const baseScene = new Scene();
  baseScene.background = new Color(0x050814);
  const splatScene = new Scene();

  const source = await SpzSplatSource.fromUrl(`${import.meta.env.BASE_URL}demo.spz`, {
    label: 'Demo SPZ',
    ...(COMPLEX_SPLAT_PRESET.importPointCap === null
      ? {}
      : { maxPoints: COMPLEX_SPLAT_PRESET.importPointCap }),
    pageCapacity: COMPLEX_SPLAT_PRESET.pageCapacity,
    minLeafPoints: COMPLEX_SPLAT_PRESET.minLeafPoints,
    branching: COMPLEX_SPLAT_PRESET.branching,
  });
  const importedAsset = source.buildAsset();
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
    antialias: false,
    alpha: false,
  });
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.toneMapping = NoToneMapping;
  renderer.domElement.classList.add('viewport-canvas');
  viewport.append(renderer.domElement);
  await renderer.init();

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.copy(target);
  controls.minDistance = Math.max(assetSize.length() * 0.05, 0.5);
  controls.maxDistance = Math.max(assetSize.length() * 4, 12);
  controls.update();

  let appliedPixelRatio = 0;
  let appliedWidth = 0;
  let appliedHeight = 0;
  let currentRenderScale = 1;

  const applyRendererScale = (renderScale: number) => {
    currentRenderScale = renderScale;
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;

    if (width === 0 || height === 0) {
      return;
    }

    const clampedRenderScale = Math.max(0.5, Math.min(1, renderScale));
    const targetPixelRatio = Math.max(
      COMPLEX_SPLAT_PRESET.minPixelRatio,
      Math.min(
        COMPLEX_SPLAT_PRESET.maxPixelRatio,
        window.devicePixelRatio * clampedRenderScale,
      ),
    );

    if (
      Math.abs(targetPixelRatio - appliedPixelRatio) > 0.04
      || width !== appliedWidth
      || height !== appliedHeight
    ) {
      renderer.setPixelRatio(targetPixelRatio);
      renderer.setSize(width, height, false);
      basePass.setSize(width, height);
      basePass.setPixelRatio(targetPixelRatio);
      splatPass.setSize(width, height);
      splatPass.setPixelRatio(targetPixelRatio);
      appliedPixelRatio = targetPixelRatio;
      appliedWidth = width;
      appliedHeight = height;
    }

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };

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

  baseScene.add(new AmbientLight(0xffffff, 1.1));

  const sun = new DirectionalLight(0xfff4d6, 1.9);
  sun.position.set(8, 12, 6);
  baseScene.add(sun);

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
  baseScene.add(floor);

  const heroSplat = new SplatMesh({
    source,
    material: new SplatMaterial({
      pointSize: COMPLEX_SPLAT_PRESET.pointSize,
      opacity: COMPLEX_SPLAT_PRESET.opacity,
      colorGain: COMPLEX_SPLAT_PRESET.colorGain,
      tint: 0xffffff,
      debugMode: 'albedo',
    }),
    importance: 1.5,
  });
  heroSplat.position.set(-assetCenter.x, -importedAsset.localBoundsMin[1], -assetCenter.z);
  splatScene.add(heroSplat);

  const bridge = new SplatRendererBridge({
    governor: new SplatQualityGovernor({
      maxVisibleSplats: COMPLEX_SPLAT_PRESET.visibleBudget,
      maxOverdrawBudget: COMPLEX_SPLAT_PRESET.overdrawBudget,
      maxActivePages: COMPLEX_SPLAT_PRESET.activePages,
      maxResidentPages: COMPLEX_SPLAT_PRESET.residentPages,
      maxPageUploadsPerFrame: COMPLEX_SPLAT_PRESET.uploadsPerFrame,
      minProjectedNodeSizePx: COMPLEX_SPLAT_PRESET.minProjectedNodeSizePx,
      peripheralFoveation: COMPLEX_SPLAT_PRESET.peripheralFoveation,
      heroTileBudget: COMPLEX_SPLAT_PRESET.heroTiles,
      temporalStabilityBias: COMPLEX_SPLAT_PRESET.temporalStabilityBias,
    }, COMPLEX_SPLAT_PRESET.targetFrameMs),
    useGpuVisibility: COMPLEX_SPLAT_PRESET.useGpuVisibilityReadback,
  });
  splatScene.add(bridge);

  const basePass = pass(baseScene, camera);
  const splatPass = pass(splatScene, camera, { depthBuffer: true });
  const baseNode = basePass.getTextureNode();
  const splatAccumulationNode = splatPass.getTextureNode();
  const accumulationWeightNode = max(splatAccumulationNode.a, 0.0001);
  const resolvedColorNode = splatAccumulationNode.rgb.div(accumulationWeightNode);
  const resolvedAlphaNode = oneMinus(max(oneMinus(splatAccumulationNode.a), 0));
  const compositeColorNode = baseNode.rgb.mul(oneMinus(resolvedAlphaNode)).add(
    resolvedColorNode.mul(resolvedAlphaNode),
  );
  const postProcessing = new PostProcessing(renderer);
  postProcessing.outputNode = vec4(compositeColorNode, 1);

  window.addEventListener('resize', () => applyRendererScale(currentRenderScale));
  applyRendererScale(1);

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
    const snapshot = bridge.update(splatScene, camera, deltaSeconds, renderer);
    applyRendererScale(snapshot.budgets.renderScale);
    postProcessing.render();
  });
}

void bootstrap().catch((error: unknown) => {
  console.error(error);
  const message = error instanceof Error ? error.message : 'Unknown bootstrap failure';
  app.innerHTML = `<div class="boot-error">Boot failed: ${message}</div>`;
});
