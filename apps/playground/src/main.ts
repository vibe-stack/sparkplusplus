import './style.css';
import {
  SplatMaterial,
  SplatMesh,
  SplatQualityGovernor,
  SplatRendererBridge,
  SpzSplatSource,
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
  override observe(_snapshot: SplatFrameStatsSnapshot) {
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
  const scene = new Scene();
  scene.background = new Color(0x050814);

  const source = await SpzSplatSource.fromUrl('/demo.spz', {
    label: 'Demo SPZ',
    ...(COMPLEX_SPLAT_PRESET.importPointCap === null
      ? {}
      : { maxPoints: COMPLEX_SPLAT_PRESET.importPointCap }),
    pageCapacity: COMPLEX_SPLAT_PRESET.pageCapacity,
    branching: 4,
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
    antialias: true,
    alpha: false,
  });
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.toneMapping = ACESFilmicToneMapping;
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

  const resize = () => {
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
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
    bridge.update(scene, camera, deltaSeconds, renderer);
    renderer.render(scene, camera);
  });
}

void bootstrap().catch((error: unknown) => {
  console.error(error);
  const message = error instanceof Error ? error.message : 'Unknown bootstrap failure';
  app.innerHTML = `<div class="boot-error">Boot failed: ${message}</div>`;
});
