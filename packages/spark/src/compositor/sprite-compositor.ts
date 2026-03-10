import {
  BufferAttribute,
  BufferGeometry,
  Color,
  DoubleSide,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  LineBasicMaterial,
  LineSegments,
  Matrix4,
  Mesh,
  OrthographicCamera,
  PerspectiveCamera,
  PlaneGeometry,
  Quaternion,
  NormalBlending,
  Sprite,
  Vector3,
  Vector4,
} from 'three';
import {
  MeshBasicNodeMaterial,
  SpriteNodeMaterial,
  StorageInstancedBufferAttribute,
  StorageTexture,
} from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  atan,
  clamp,
  cross,
  exp,
  float,
  instancedBufferAttribute,
  instanceIndex,
  min,
  max,
  modelViewMatrix,
  smoothstep,
  storage,
  texture,
  textureStore,
  sqrt,
  uint,
  uv,
  uniform,
  vec2,
  vec3,
  vec4,
  uvec2,
} from 'three/tsl';
import type { Camera } from 'three';
import { lengthSq } from 'three/tsl';
import {
  SPLAT_COMPUTE_TILE_ENTRY_FLOATS,
  SPLAT_COMPUTE_TILE_HEADER_FLOATS,
} from '../core/layouts';
import {
  SplatClusterTileBinner,
  type SplatClusterTileBinningSnapshot,
  type SplatProjectedCluster,
} from './cluster-tile-binner';
import { SPLAT_SEMANTIC_FLAGS } from '../core/semantics';
import type { SplatBudgetOptions } from '../core/budgets';
import type { SplatGpuRendererLike } from '../gpu/visibility';
import type { SplatMeshSelection } from '../scheduler/bootstrap-scheduler';
import type { SplatMesh } from '../scene/splat-mesh';

export interface SplatCompositorSnapshot {
  weightedInstances: number;
  heroInstances: number;
  depthSlicedInstances: number;
  activeTiles: number;
  weightedTiles: number;
  depthSlicedTiles: number;
  heroTiles: number;
  maxTileComplexity: number;
  visibleClusters: number;
  binnedClusters: number;
  binnedClusterReferences: number;
  overflowedTiles: number;
  overflowedClusterReferences: number;
  maxClustersPerTile: number;
  maxTileSplatEstimate: number;
  tileBufferBytes: number;
}

export interface SplatCompositorFrameContext {
  camera: Camera;
  viewportWidth: number;
  viewportHeight: number;
  budgets: SplatBudgetOptions;
  timeSeconds: number;
  frameIndex: number;
  gpuVisibilityReady: boolean;
  renderer: SplatGpuRendererLike | undefined;
}

const EMPTY_SNAPSHOT: SplatCompositorSnapshot = {
  weightedInstances: 0,
  heroInstances: 0,
  depthSlicedInstances: 0,
  activeTiles: 0,
  weightedTiles: 0,
  depthSlicedTiles: 0,
  heroTiles: 0,
  maxTileComplexity: 0,
  visibleClusters: 0,
  binnedClusters: 0,
  binnedClusterReferences: 0,
  overflowedTiles: 0,
  overflowedClusterReferences: 0,
  maxClustersPerTile: 0,
  maxTileSplatEstimate: 0,
  tileBufferBytes: 0,
};

interface PreparedPageCache {
  materialVersion: number;
  baseScales: Float32Array;
  rotations: Float32Array;
  colors: Float32Array;
  opacities: Float32Array;
}

interface ComputeTileFrame {
  totalWorkItems: number;
  maxTileWorkItems: number;
  tileHeaderBytes: number;
  tileEntryBytes: number;
}

const COMPUTE_WORKGROUP_SIZE = 64;
const COMPUTE_MAX_SAMPLING_STRIDE = 12;
const COMPUTE_ALPHA_CUTOFF = 0.992;

export class SplatSpriteCompositor {
  private readonly preparedPageCache = new Map<number, PreparedPageCache>();
  private readonly tileBinner = new SplatClusterTileBinner();
  private readonly depthSortIndexCache = new Map<number, Uint32Array>();
  private readonly depthSortValueCache = new Map<number, Float32Array>();

  private cpuSprite: Sprite | undefined;
  private cpuMaterial: SpriteNodeMaterial | undefined;
  private positionAttribute?: InstancedBufferAttribute;
  private scaleAttribute?: InstancedBufferAttribute;
  private rotationAttribute?: InstancedBufferAttribute;
  private colorAttribute?: InstancedBufferAttribute;
  private opacityAttribute?: InstancedBufferAttribute;
  private gpuSprite: Sprite | undefined;
  private gpuMaterial: SpriteNodeMaterial | undefined;
  private gpuSelectionAttribute: StorageInstancedBufferAttribute | undefined;
  private gpuPackedPositionsOpacityAttribute: StorageInstancedBufferAttribute | undefined;
  private gpuPackedScalesAttribute: StorageInstancedBufferAttribute | undefined;
  private gpuPackedRotationsAttribute: StorageInstancedBufferAttribute | undefined;
  private gpuPackedColorsAttribute: StorageInstancedBufferAttribute | undefined;
  private computeOverlay: Mesh | undefined;
  private computeOverlayMaterial: MeshBasicNodeMaterial | undefined;
  private computeOutputTexture: StorageTexture | undefined;
  private computeTileHeaderAttribute: StorageInstancedBufferAttribute | undefined;
  private computeTileEntryAttribute: StorageInstancedBufferAttribute | undefined;
  private computeNode: any;
  private clusterBoundsOverlay: LineSegments | undefined;
  private clusterBoundsGeometry: BufferGeometry | undefined;
  private clusterBoundsMaterial: LineBasicMaterial | undefined;
  private clusterBoundsPositionAttribute: BufferAttribute | undefined;
  private clusterBoundsColorAttribute: BufferAttribute | undefined;

  private positionArray = new Float32Array(0);
  private scaleArray = new Float32Array(0);
  private rotationArray = new Float32Array(0);
  private colorArray = new Float32Array(0);
  private opacityArray = new Float32Array(0);
  private gpuSelectionArray = new Float32Array(0);
  private computeTileHeaderArray = new Float32Array(0);
  private computeTileEntryArray = new Float32Array(0);
  private clusterBoundsPositionArray = new Float32Array(0);
  private clusterBoundsColorArray = new Float32Array(0);

  private readonly colorScratch = new Color();
  private readonly effectTintScratch = new Color();
  private readonly debugColorScratch = new Color();
  private readonly computePointSizeParams = new Vector4();
  private readonly computeTintParams = new Vector4();
  private readonly computeViewportParams = new Vector4();
  private readonly computeCameraParams = new Vector4();
  private readonly computeRangeParams = new Vector4();
  private readonly computeModelViewRow0 = new Vector4();
  private readonly computeModelViewRow1 = new Vector4();
  private readonly computeModelViewRow2 = new Vector4();
  private readonly computeOverlayPosition = new Vector3();
  private readonly computeOverlayScale = new Vector3(1, 1, 1);
  private readonly computeOverlayQuaternion = new Quaternion();
  private readonly computeForward = new Vector3();
  private readonly computeOwnerInverseMatrix = new Matrix4();
  private readonly depthSortModelView = new Matrix4();
  private snapshot: SplatCompositorSnapshot = EMPTY_SNAPSHOT;
  private lastBuildSignature = '';
  private lastReductionHash = 2166136261;
  private gpuMaterialVersion = -1;
  private computeTileBufferSignature = '';
  private computeTileFrame: ComputeTileFrame = {
    totalWorkItems: 0,
    maxTileWorkItems: 0,
    tileHeaderBytes: 0,
    tileEntryBytes: 0,
  };
  private readonly stableClusterSamplingStride = new Map<number, number>();

  constructor(private readonly owner: SplatMesh) {
    this.ensureCapacity(256);
  }

  getSnapshot(): SplatCompositorSnapshot {
    return this.snapshot;
  }

  sync(selection: SplatMeshSelection, context: SplatCompositorFrameContext): void {
    const asset = this.owner.getAsset();
    const canUsePreparedPages = this.canUsePreparedPages();
    // Phase 1 compatibility path:
    // 1. Project visible clusters into fixed-size screen tiles.
    // 2. Build bounded per-tile cluster lists plus depth buckets.
    // 3. Feed the existing sprite queue from tile-informed cluster order.
    const tileBinning = this.tileBinner.binVisibleClusters(
      this.owner,
      selection.activeClusters,
      context.camera,
      context.viewportWidth,
      context.viewportHeight,
      context.budgets,
    );
    const activeClusterById = new Map(
      selection.activeClusters.map((activeCluster) => [activeCluster.clusterId, activeCluster] as const),
    );
    const clusteredActiveClusters = tileBinning.clusterOrder
      .map((clusterId) => activeClusterById.get(clusterId))
      .filter((cluster): cluster is NonNullable<typeof cluster> => cluster !== undefined);
    const clusterSamplingStride = this.resolveClusterSamplingStride(tileBinning, context.budgets);
    const frameBucket = this.owner.effectStack.hasTemporalEffects()
      ? Math.floor(context.frameIndex / Math.max(1, context.budgets.effectUpdateCadence))
      : 0;
    const computeMainPath = this.shouldUseComputeMainPath(context, tileBinning);
    const alphaSortedMainPath = !computeMainPath
      && (this.owner.splatMaterial.debugMode === 'albedo' || this.owner.splatMaterial.debugMode === 'cluster-bounds');
    const renderActiveClusters = alphaSortedMainPath
      ? this.sortClustersBackToFront(clusteredActiveClusters, tileBinning)
      : clusteredActiveClusters;
    const rebuildDependsOnBinning = this.owner.splatMaterial.debugMode === 'tile-occupancy'
      || this.owner.splatMaterial.debugMode === 'tile-heatmap'
      || this.owner.splatMaterial.debugMode === 'depth-buckets';
    const buildSignature = [
      renderActiveClusters.map((cluster) => cluster.clusterId).join(','),
      this.owner.getMaterialVersion(),
      this.owner.getEffectVersion(),
      frameBucket,
      computeMainPath ? 'compute' : 'fallback',
      `${tileBinning.columns}x${tileBinning.rows}@${tileBinning.tileSizePx}`,
      rebuildDependsOnBinning || computeMainPath || alphaSortedMainPath ? tileBinning.debugHash : 'static',
      alphaSortedMainPath ? this.resolveCameraOrderingSignature(context.camera) : 'static-camera',
      this.lastReductionHash,
    ].join('::');

    this.syncClusterBoundsOverlay(tileBinning);
    this.snapshot = this.buildSnapshot(0, tileBinning);

    if (renderActiveClusters.length === 0) {
      this.lastBuildSignature = '';
      this.computeTileBufferSignature = '';
      this.computeTileFrame = {
        totalWorkItems: 0,
        maxTileWorkItems: 0,
        tileHeaderBytes: 0,
        tileEntryBytes: 0,
      };
      this.syncComputeOverlay(context.camera, context.viewportWidth, context.viewportHeight, false);
      this.syncGpuSprite(0);
      this.syncSprite(0);
      return;
    }

    const buildChanged = buildSignature !== this.lastBuildSignature;

    if (computeMainPath) {
      this.lastBuildSignature = buildSignature;
      if (buildChanged) {
        this.prepareComputeTilePath(clusteredActiveClusters, clusterSamplingStride, tileBinning, context.budgets);
      }
      this.dispatchComputeTilePath(context, tileBinning);
      this.syncGpuSprite(0);
      this.syncSprite(0);
      this.snapshot = this.buildSnapshot(this.computeTileFrame.totalWorkItems, tileBinning, {
        tileWorkPeak: this.computeTileFrame.maxTileWorkItems,
        tileBufferBytes: tileBinning.buffers.clusterScreenData.byteLength
          + tileBinning.buffers.tileHeaders.byteLength
          + tileBinning.buffers.tileEntries.byteLength
          + this.computeTileFrame.tileHeaderBytes
          + this.computeTileFrame.tileEntryBytes,
      });
      return;
    }

    this.syncComputeOverlay(context.camera, context.viewportWidth, context.viewportHeight, false);

    if (!buildChanged) {
      return;
    }

    this.lastBuildSignature = buildSignature;

    if (this.canUseGpuMainPath(context)) {
      this.syncGpuTilePath(renderActiveClusters, clusterSamplingStride, tileBinning, context.camera);
      this.syncSprite(0);
      return;
    }

    let queueEstimate = 0;

    for (const activeCluster of renderActiveClusters) {
      const page = asset.pages[activeCluster.pageId]!;
      const samplingStride = clusterSamplingStride.get(activeCluster.clusterId) ?? 1;
      queueEstimate += Math.ceil(page.splatCount / samplingStride);
    }

    this.ensureCapacity(queueEstimate);

    let instanceCount = 0;

    for (const activeCluster of renderActiveClusters) {
      const cluster = asset.clusters[activeCluster.clusterId]!;
      const page = asset.pages[activeCluster.pageId]!;
      const preparedPage = canUsePreparedPages ? this.getPreparedPageCache(page) : null;
      const projectedCluster = tileBinning.clusterProjections.get(cluster.id) ?? null;
      const samplingStride = clusterSamplingStride.get(cluster.id) ?? 1;
      const depthSortedIndices = alphaSortedMainPath && this.shouldDepthSortCluster(cluster, projectedCluster, samplingStride)
        ? this.getDepthSortedPageIndices(page, context.camera)
        : null;
      const densityCompensation = this.resolveDensityCompensation(samplingStride);
      const coverageScaleBoost = this.resolveCoverageScaleBoost(cluster, densityCompensation);

      if (preparedPage && !depthSortedIndices && samplingStride === 1) {
        this.copyPreparedPage(
          page.positions,
          preparedPage.baseScales,
          preparedPage.rotations,
          preparedPage.colors,
          preparedPage.opacities,
          instanceCount,
          coverageScaleBoost,
        );
        instanceCount += page.splatCount;
        continue;
      }

      if (preparedPage && !depthSortedIndices) {
        instanceCount += this.copyPreparedPageStrided(
          preparedPage,
          page.positions,
          instanceCount,
          samplingStride,
          coverageScaleBoost,
        );
        continue;
      }

      if (preparedPage && depthSortedIndices) {
        instanceCount += this.copyPreparedPageDepthSorted(
          preparedPage,
          page.positions,
          depthSortedIndices,
          instanceCount,
          samplingStride,
          coverageScaleBoost,
        );
        continue;
      }

      const resolvedEffects = this.owner.effectStack.evaluate(cluster.semanticMask, context.timeSeconds);
      this.effectTintScratch.copy(resolvedEffects.tint);
      const splatOrder = depthSortedIndices ?? null;
      const splatIterations = splatOrder ? splatOrder.length : page.splatCount;

      for (let orderIndex = 0; orderIndex < splatIterations; orderIndex += samplingStride) {
        const splatIndex = splatOrder ? splatOrder[orderIndex]! : orderIndex;
        const sourceOffset = splatIndex * 3;
        const rotationOffset = splatIndex * 4;

        this.colorScratch.setRGB(
          page.colors[sourceOffset + 0]!,
          page.colors[sourceOffset + 1]!,
          page.colors[sourceOffset + 2]!,
        );

        if (this.owner.splatMaterial.debugMode === 'lod') {
          this.colorScratch.setHSL(Math.min(0.85, cluster.level * 0.12), 0.75, 0.55);
        }

        if (this.owner.splatMaterial.debugMode === 'semantic') {
          if ((cluster.semanticMask & SPLAT_SEMANTIC_FLAGS.hero) !== 0) {
            this.colorScratch.set(0xffe066);
          } else if ((cluster.semanticMask & SPLAT_SEMANTIC_FLAGS.glass) !== 0) {
            this.colorScratch.set(0x93c5fd);
          } else if ((cluster.semanticMask & SPLAT_SEMANTIC_FLAGS.foliage) !== 0) {
            this.colorScratch.set(0x86efac);
          } else {
            this.colorScratch.set(0xf8fafc);
          }
        }

        if (this.owner.splatMaterial.debugMode === 'representation') {
          this.resolveRepresentationDebugColor(this.colorScratch, cluster);
        }

        if (
          this.owner.splatMaterial.debugMode === 'tile-occupancy'
          || this.owner.splatMaterial.debugMode === 'tile-heatmap'
          || this.owner.splatMaterial.debugMode === 'depth-buckets'
        ) {
          this.resolveTileDebugColor(this.colorScratch, projectedCluster, tileBinning);
        }

        this.colorScratch
          .multiply(this.owner.splatMaterial.tint)
          .multiply(this.effectTintScratch)
          .multiplyScalar(this.owner.splatMaterial.colorGain);

        this.writeInstance(
          instanceCount,
          page.positions[sourceOffset + 0]!,
          page.positions[sourceOffset + 1]!,
          page.positions[sourceOffset + 2]!,
          this.resolveBaseScale(page.scales[sourceOffset + 0]!, coverageScaleBoost * resolvedEffects.pointSizeMultiplier),
          this.resolveBaseScale(page.scales[sourceOffset + 1]!, coverageScaleBoost * resolvedEffects.pointSizeMultiplier),
          this.resolveBaseScale(page.scales[sourceOffset + 2]!, coverageScaleBoost * resolvedEffects.pointSizeMultiplier),
          page.rotations[rotationOffset + 0]!,
          page.rotations[rotationOffset + 1]!,
          page.rotations[rotationOffset + 2]!,
          page.rotations[rotationOffset + 3]!,
          this.colorScratch,
          Math.min(
            1,
            this.owner.splatMaterial.opacity * page.opacities[splatIndex]! * resolvedEffects.opacityMultiplier,
          ),
        );
        instanceCount += 1;
      }
    }

    this.flushAttributes(instanceCount);
    this.syncGpuSprite(0);
    this.syncSprite(instanceCount);
    this.snapshot = this.buildSnapshot(instanceCount, tileBinning);
  }

  private ensureCapacity(requiredCapacity: number): void {
    if (this.positionArray.length >= requiredCapacity * 3) {
      return;
    }

    const nextCapacity = this.resolveCapacity(
      Math.floor(this.positionArray.length / 3),
      requiredCapacity,
      256,
    );

    this.positionArray = new Float32Array(nextCapacity * 3);
    this.scaleArray = new Float32Array(nextCapacity * 3);
    this.rotationArray = new Float32Array(nextCapacity * 4);
    this.colorArray = new Float32Array(nextCapacity * 3);
    this.opacityArray = new Float32Array(nextCapacity);
    this.rebuildSprite();
  }

  private rebuildSprite(): void {
    if (this.cpuSprite) {
      this.owner.remove(this.cpuSprite);
    }

    this.positionAttribute = this.createAttribute(this.positionArray, 3);
    this.scaleAttribute = this.createAttribute(this.scaleArray, 3);
    this.rotationAttribute = this.createAttribute(this.rotationArray, 4);
    this.colorAttribute = this.createAttribute(this.colorArray, 3);
    this.opacityAttribute = this.createAttribute(this.opacityArray, 1);
    this.cpuMaterial = this.createMaterial(
      this.positionAttribute,
      this.scaleAttribute,
      this.rotationAttribute,
      this.colorAttribute,
      this.opacityAttribute,
    );
    this.cpuSprite = new Sprite(this.cpuMaterial);
    this.cpuSprite.count = 0;
    this.cpuSprite.frustumCulled = false;
    this.cpuSprite.renderOrder = 21;
    this.cpuSprite.name = 'SparkSpriteQueue';
    this.owner.add(this.cpuSprite);
  }

  private createMaterial(
    positionAttribute: InstancedBufferAttribute,
    scaleAttribute: InstancedBufferAttribute,
    rotationAttribute: InstancedBufferAttribute,
    colorAttribute: InstancedBufferAttribute,
    opacityAttribute: InstancedBufferAttribute,
  ): SpriteNodeMaterial {
    const material = new SpriteNodeMaterial();
    const uvNode = uv();
    const centeredUv = uvNode.sub(vec2(0.5)).mul(2.0);
    const radial = lengthSq(centeredUv);
    const feather = smoothstep(1.0, 0.5, radial);
    const gaussian = exp(radial.mul(-10.5));
    const alphaNode = instancedBufferAttribute(opacityAttribute).mul(gaussian).mul(feather);
    const scaleNode = instancedBufferAttribute(scaleAttribute);
    const rotationNode = instancedBufferAttribute(rotationAttribute);
    const quaternionXYZ = rotationNode.xyz;
    const quaternionW = rotationNode.w;
    const rotateAxis = (axisNode: any) => {
      const doubleCross = cross(quaternionXYZ, axisNode).mul(2);
      return axisNode.add(doubleCross.mul(quaternionW)).add(cross(quaternionXYZ, doubleCross));
    };
    const axisXView = modelViewMatrix.mul(vec4(rotateAxis(vec3(scaleNode.x, 0, 0)), 0)).xyz;
    const axisYView = modelViewMatrix.mul(vec4(rotateAxis(vec3(0, scaleNode.y, 0)), 0)).xyz;
    const axisZView = modelViewMatrix.mul(vec4(rotateAxis(vec3(0, 0, scaleNode.z)), 0)).xyz;
    const covarianceXX = axisXView.x.mul(axisXView.x).add(axisYView.x.mul(axisYView.x)).add(axisZView.x.mul(axisZView.x));
    const covarianceXY = axisXView.x.mul(axisXView.y).add(axisYView.x.mul(axisYView.y)).add(axisZView.x.mul(axisZView.y));
    const covarianceYY = axisXView.y.mul(axisXView.y).add(axisYView.y.mul(axisYView.y)).add(axisZView.y.mul(axisZView.y));
    const trace = covarianceXX.add(covarianceYY);
    const delta = sqrt(max(0.000001, covarianceXX.sub(covarianceYY).mul(covarianceXX.sub(covarianceYY)).add(covarianceXY.mul(covarianceXY).mul(4))));
    const majorScale = sqrt(max(0.000016, trace.add(delta).mul(0.5)));
    const minorScale = sqrt(max(0.000004, trace.sub(delta).mul(0.5)));
    const ellipseRotation = atan(covarianceXY.mul(2), covarianceXX.sub(covarianceYY)).mul(0.5);

    material.positionNode = instancedBufferAttribute(positionAttribute);
    material.scaleNode = vec2(majorScale, minorScale);
    material.rotationNode = ellipseRotation;
    material.colorNode = instancedBufferAttribute(colorAttribute).mul(alphaNode);
    material.opacityNode = alphaNode;
    material.maskNode = radial.lessThan(1.0);
    material.transparent = true;
    material.depthWrite = false;
    material.sizeAttenuation = true;
    material.side = DoubleSide;
    material.blending = NormalBlending;
    material.premultipliedAlpha = true;

    return material;
  }

  private createAttribute(array: Float32Array, itemSize: number): InstancedBufferAttribute {
    const attribute = new InstancedBufferAttribute(array, itemSize);
    attribute.setUsage(DynamicDrawUsage);
    return attribute;
  }

  private createStorageAttribute(
    array: Float32Array,
    itemSize: number,
    dynamic: boolean,
  ): StorageInstancedBufferAttribute {
    const attribute = new StorageInstancedBufferAttribute(array, itemSize);

    if (dynamic) {
      attribute.setUsage(DynamicDrawUsage);
    }

    return attribute;
  }

  private flushAttributes(instanceCount: number): void {
    this.updateAttribute(this.positionAttribute, instanceCount, 3);
    this.updateAttribute(this.scaleAttribute, instanceCount, 3);
    this.updateAttribute(this.rotationAttribute, instanceCount, 4);
    this.updateAttribute(this.colorAttribute, instanceCount, 3);
    this.updateAttribute(this.opacityAttribute, instanceCount, 1);
  }

  private updateAttribute(
    attribute: InstancedBufferAttribute | undefined,
    count: number,
    itemSize: number,
  ): void {
    if (!attribute) {
      return;
    }

    attribute.needsUpdate = true;
    attribute.clearUpdateRanges();
    attribute.addUpdateRange(0, count * itemSize);
  }

  private syncSprite(instanceCount: number): void {
    if (!this.cpuSprite) {
      return;
    }

    this.cpuSprite.count = instanceCount;
    this.cpuSprite.visible = instanceCount > 0;
  }

  private canUseComputeMainPath(context: SplatCompositorFrameContext): boolean {
    return context.renderer?.hasInitialized?.() !== false
      && typeof context.renderer?.compute === 'function'
      && this.owner.effectStack.isIdentity()
      && this.owner.splatMaterial.debugMode === 'albedo';
  }

  private shouldUseComputeMainPath(
    context: SplatCompositorFrameContext,
    tileBinning: SplatClusterTileBinningSnapshot,
  ): boolean {
    if (!this.canUseComputeMainPath(context)) {
      return false;
    }

    const maxProjectedRadiusPx = this.resolveMaxProjectedClusterRadius(tileBinning);
    const activeTileRatio = tileBinning.activeTiles / Math.max(1, tileBinning.columns * tileBinning.rows);
    const safeTileWorkCeiling = Math.max(
      48,
      Math.min(128, Math.round(context.budgets.maxSplatsPerTile * 0.18)),
    );

    // The current compute path is only viable for low-pressure distant views.
    // Near-field dense views should stay on the sprite fallback until the tile
    // compositor is upgraded to workgroup-local expansion/composition.
    return tileBinning.maxTileSplatEstimate <= safeTileWorkCeiling
      && tileBinning.overflowedTiles === 0
      && maxProjectedRadiusPx <= 72
      && activeTileRatio <= 0.35;
  }

  private canUseGpuMainPath(context: SplatCompositorFrameContext): boolean {
    return context.renderer?.hasInitialized?.() !== false
      && this.owner.effectStack.isIdentity()
      && (this.owner.splatMaterial.debugMode === 'albedo' || this.owner.splatMaterial.debugMode === 'cluster-bounds');
  }

  private prepareComputeTilePath(
    clusteredActiveClusters: ReadonlyArray<SplatMeshSelection['activeClusters'][number]>,
    clusterSamplingStride: ReadonlyMap<number, number>,
    tileBinning: SplatClusterTileBinningSnapshot,
    budgets: SplatBudgetOptions,
  ): void {
    const asset = this.owner.getAsset();
    const tileCount = tileBinning.columns * tileBinning.rows;
    let totalEntryCount = 0;

    for (const tile of tileBinning.tileBins) {
      totalEntryCount += tile.clusterIds.length;
    }

    this.ensureComputeTileCapacity(tileCount, totalEntryCount);
    const activeClusterById = new Map(
      clusteredActiveClusters.map((activeCluster) => [activeCluster.clusterId, activeCluster] as const),
    );
    const tileHeaderArray = this.computeTileHeaderArray;
    const tileEntryArray = this.computeTileEntryArray;
    tileHeaderArray.fill(0);
    tileEntryArray.fill(0);

    let tileEntryCursor = 0;
    let totalWorkItems = 0;
    let maxTileWorkItems = 0;

    // Pass layout:
    // 1. `tileBinning` already supplies bounded tile ownership and bucket-ordered cluster ids.
    // 2. The compute tile frame converts that into compact page-level expansion records.
    // 3. The per-pixel compute resolve reads only the entries owned by its tile.
    for (const tile of tileBinning.tileBins) {
      const headerOffset = tile.tileId * SPLAT_COMPUTE_TILE_HEADER_FLOATS;
      const tileEntryStart = tileEntryCursor;
      tileHeaderArray[headerOffset + 0] = tileEntryStart;
      tileHeaderArray[headerOffset + 3] = tile.overflowCount;
      const tileStrideScale = Math.max(1, Math.ceil(tile.splatEstimate / Math.max(1, budgets.maxSplatsPerTile)));

      let tileWorkItems = 0;

      for (const clusterId of tile.clusterIds) {
        const activeCluster = activeClusterById.get(clusterId);

        if (!activeCluster) {
          continue;
        }

        const cluster = asset.clusters[clusterId]!;
        const page = asset.pages[activeCluster.pageId]!;
        const baseSamplingStride = clusterSamplingStride.get(clusterId) ?? 1;
        const samplingStride = Math.max(
          1,
          Math.min(COMPUTE_MAX_SAMPLING_STRIDE, baseSamplingStride * tileStrideScale),
        );
        const densityCompensation = this.resolveDensityCompensation(samplingStride);
        const scaleMultiplier = this.resolveCoverageScaleBoost(cluster, densityCompensation);
        const pageDescriptorOffset = page.id * 8;
        const pageSplatOffset = asset.buffers.pageDescriptors[pageDescriptorOffset + 6] ?? 0;
        const entryOffset = tileEntryCursor * SPLAT_COMPUTE_TILE_ENTRY_FLOATS;
        const entryWorkItems = Math.ceil(page.splatCount / samplingStride);

        tileEntryArray[entryOffset + 0] = pageSplatOffset;
        tileEntryArray[entryOffset + 1] = page.splatCount;
        tileEntryArray[entryOffset + 2] = samplingStride;
        tileEntryArray[entryOffset + 3] = scaleMultiplier;
        tileEntryCursor += 1;
        tileWorkItems += entryWorkItems;
      }

      tileHeaderArray[headerOffset + 1] = tileEntryCursor - tileEntryStart;

      if (tileWorkItems > budgets.maxSplatsPerTile) {
        tileHeaderArray[headerOffset + 3] = Math.max(
          tileHeaderArray[headerOffset + 3] ?? 0,
          tileWorkItems - budgets.maxSplatsPerTile,
        );
      }

      tileHeaderArray[headerOffset + 2] = Math.min(tileWorkItems, budgets.maxSplatsPerTile);
      totalWorkItems += tileHeaderArray[headerOffset + 2] ?? 0;
      maxTileWorkItems = Math.max(maxTileWorkItems, tileHeaderArray[headerOffset + 2] ?? 0);
    }

    if (this.computeTileHeaderAttribute) {
      this.computeTileHeaderAttribute.needsUpdate = true;
      this.computeTileHeaderAttribute.clearUpdateRanges();
      this.computeTileHeaderAttribute.addUpdateRange(0, tileCount * SPLAT_COMPUTE_TILE_HEADER_FLOATS);
    }

    if (this.computeTileEntryAttribute) {
      this.computeTileEntryAttribute.needsUpdate = true;
      this.computeTileEntryAttribute.clearUpdateRanges();
      this.computeTileEntryAttribute.addUpdateRange(0, tileEntryCursor * SPLAT_COMPUTE_TILE_ENTRY_FLOATS);
    }

    this.computeTileFrame = {
      totalWorkItems,
      maxTileWorkItems,
      tileHeaderBytes: tileCount * SPLAT_COMPUTE_TILE_HEADER_FLOATS * Float32Array.BYTES_PER_ELEMENT,
      tileEntryBytes: tileEntryCursor * SPLAT_COMPUTE_TILE_ENTRY_FLOATS * Float32Array.BYTES_PER_ELEMENT,
    };
    this.computeTileBufferSignature = [
      tileBinning.debugHash,
      this.lastReductionHash,
      tileBinning.columns,
      tileBinning.rows,
      tileEntryCursor,
    ].join(':');
  }

  private ensureComputeTileCapacity(tileCount: number, entryCount: number): void {
    if (this.computeTileHeaderArray.length < tileCount * SPLAT_COMPUTE_TILE_HEADER_FLOATS) {
      const nextTileCapacity = this.resolveCapacity(
        Math.floor(this.computeTileHeaderArray.length / SPLAT_COMPUTE_TILE_HEADER_FLOATS),
        tileCount,
        64,
      );
      this.computeTileHeaderArray = new Float32Array(nextTileCapacity * SPLAT_COMPUTE_TILE_HEADER_FLOATS);
      this.computeTileHeaderAttribute = this.createStorageAttribute(
        this.computeTileHeaderArray,
        SPLAT_COMPUTE_TILE_HEADER_FLOATS,
        true,
      );
      this.computeNode = undefined;
    }

    if (this.computeTileEntryArray.length < entryCount * SPLAT_COMPUTE_TILE_ENTRY_FLOATS) {
      const nextEntryCapacity = this.resolveCapacity(
        Math.floor(this.computeTileEntryArray.length / SPLAT_COMPUTE_TILE_ENTRY_FLOATS),
        entryCount,
        256,
      );
      this.computeTileEntryArray = new Float32Array(nextEntryCapacity * SPLAT_COMPUTE_TILE_ENTRY_FLOATS);
      this.computeTileEntryAttribute = this.createStorageAttribute(
        this.computeTileEntryArray,
        SPLAT_COMPUTE_TILE_ENTRY_FLOATS,
        true,
      );
      this.computeNode = undefined;
    }
  }

  private dispatchComputeTilePath(
    context: SplatCompositorFrameContext,
    tileBinning: SplatClusterTileBinningSnapshot,
  ): void {
    const renderer = context.renderer;

    if (!renderer || typeof renderer.compute !== 'function') {
      return;
    }

    this.ensureComputeResources(context.viewportWidth, context.viewportHeight);
    this.updateComputeUniforms(context.camera, context.viewportWidth, context.viewportHeight, tileBinning);
    this.syncComputeOverlay(context.camera, context.viewportWidth, context.viewportHeight, true);

    if (!this.computeNode) {
      return;
    }

    renderer.compute(this.computeNode);
  }

  private ensureComputeResources(viewportWidth: number, viewportHeight: number): void {
    this.ensurePackedSplatStorageAttributes();
    const nextWidth = Math.max(1, viewportWidth);
    const nextHeight = Math.max(1, viewportHeight);
    const previousImage = this.computeOutputTexture?.image as { width?: number; height?: number } | undefined;
    const previousWidth = previousImage?.width ?? 0;
    const previousHeight = previousImage?.height ?? 0;
    const textureResized = !this.computeOutputTexture || previousWidth !== nextWidth || previousHeight !== nextHeight;

    if (textureResized) {
      this.computeOutputTexture = new StorageTexture(nextWidth, nextHeight);
      this.computeNode = undefined;
      this.rebuildComputeOverlay();
    }

    this.ensureComputeOverlay();

    if (!this.computeNode) {
      this.computeNode = this.createComputeNode();
    }
  }

  private ensureComputeOverlay(): void {
    if (this.computeOverlay || !this.computeOutputTexture) {
      return;
    }

    const material = new MeshBasicNodeMaterial();
    const outputNode = texture(this.computeOutputTexture, uv());
    material.colorNode = outputNode.rgb;
    material.opacityNode = outputNode.a;
    material.transparent = true;
    material.depthWrite = false;
    material.depthTest = false;
    material.toneMapped = false;
    this.computeOverlayMaterial = material;
    this.computeOverlay = new Mesh(new PlaneGeometry(1, 1), material);
    this.computeOverlay.frustumCulled = false;
    this.computeOverlay.renderOrder = 21;
    this.computeOverlay.matrixAutoUpdate = false;
    this.computeOverlay.visible = false;
    this.computeOverlay.name = 'SparkComputeTileResolve';
    this.owner.add(this.computeOverlay);
  }

  private rebuildComputeOverlay(): void {
    if (this.computeOverlay) {
      this.owner.remove(this.computeOverlay);
      this.computeOverlay.geometry.dispose();
      this.computeOverlay = undefined;
    }

    if (this.computeOverlayMaterial) {
      this.computeOverlayMaterial.dispose();
      this.computeOverlayMaterial = undefined;
    }
  }

  private ensurePackedSplatStorageAttributes(): void {
    if (
      this.gpuPackedPositionsOpacityAttribute
      && this.gpuPackedScalesAttribute
      && this.gpuPackedRotationsAttribute
      && this.gpuPackedColorsAttribute
    ) {
      return;
    }

    const asset = this.owner.getAsset();
    this.gpuPackedPositionsOpacityAttribute ??= this.createStorageAttribute(
      asset.buffers.packedPositionsOpacity,
      4,
      false,
    );
    this.gpuPackedScalesAttribute ??= this.createStorageAttribute(asset.buffers.packedScales, 4, false);
    this.gpuPackedRotationsAttribute ??= this.createStorageAttribute(asset.buffers.packedRotations, 4, false);
    this.gpuPackedColorsAttribute ??= this.createStorageAttribute(asset.buffers.packedColors, 4, false);
  }

  private createComputeNode(): any {
    if (
      !this.computeTileHeaderAttribute
      || !this.computeTileEntryAttribute
      || !this.gpuPackedPositionsOpacityAttribute
      || !this.gpuPackedScalesAttribute
      || !this.gpuPackedColorsAttribute
      || !this.computeOutputTexture
    ) {
      return undefined;
    }

    const tileHeadersNode = storage(
      this.computeTileHeaderAttribute,
      'vec4',
      Math.max(1, Math.floor(this.computeTileHeaderAttribute.array.length / SPLAT_COMPUTE_TILE_HEADER_FLOATS)),
    );
    const tileEntriesNode = storage(
      this.computeTileEntryAttribute,
      'vec4',
      Math.max(1, Math.floor(this.computeTileEntryAttribute.array.length / SPLAT_COMPUTE_TILE_ENTRY_FLOATS)),
    );
    const positionsNode = storage(
      this.gpuPackedPositionsOpacityAttribute,
      'vec4',
      Math.max(1, Math.floor(this.gpuPackedPositionsOpacityAttribute.array.length / 4)),
    );
    const scalesNode = storage(
      this.gpuPackedScalesAttribute,
      'vec4',
      Math.max(1, Math.floor(this.gpuPackedScalesAttribute.array.length / 4)),
    );
    const colorsNode = storage(
      this.gpuPackedColorsAttribute,
      'vec4',
      Math.max(1, Math.floor(this.gpuPackedColorsAttribute.array.length / 4)),
    );
    const viewportParams = uniform(this.computeViewportParams);
    const cameraParams = uniform(this.computeCameraParams);
    const pointSizeParams = uniform(this.computePointSizeParams);
    const tintParams = uniform(this.computeTintParams);
    const rangeParams = uniform(this.computeRangeParams);
    const modelViewRow0 = uniform(this.computeModelViewRow0);
    const modelViewRow1 = uniform(this.computeModelViewRow1);
    const modelViewRow2 = uniform(this.computeModelViewRow2);

    const outputTexture = this.computeOutputTexture;
    const outputImage = outputTexture.image as { width?: number; height?: number };
    const outputPixelCount = Math.max(1, (outputImage.width ?? 1) * (outputImage.height ?? 1));

    return Fn(() => {
      const viewportWidth = max(uint(1), uint(viewportParams.x));
      const viewportHeight = max(uint(1), uint(viewportParams.y));
      const tileSizePx = max(uint(1), uint(viewportParams.z));
      const tileColumns = max(uint(1), uint(viewportParams.w));
      const pixelIndex = uint(instanceIndex);
      const pixelX = pixelIndex.mod(viewportWidth);
      const pixelY = pixelIndex.div(viewportWidth);
      const pixelCenter = vec2(float(pixelX).add(0.5), float(pixelY).add(0.5));
      const tileX = pixelX.div(tileSizePx);
      const tileY = pixelY.div(tileSizePx);
      const tileIndex = tileY.mul(tileColumns).add(tileX);
      const tileHeader = tileHeadersNode.element(tileIndex);
      const entryOffset = uint(tileHeader.x);
      const entryCount = uint(tileHeader.y);
      const accumulatedColor = vec3(0, 0, 0).toVar();
      const accumulatedAlpha = float(0).toVar();
      const viewportWidthFloat = float(viewportParams.x);
      const viewportHeightFloat = float(viewportParams.y);
      const isPerspective = cameraParams.x.greaterThan(0.5);

      Loop(
        { start: uint(0), end: entryCount, type: 'uint', condition: '<', name: 'entryIndex' },
        (inputs: any) => {
          const entryIndex = inputs.entryIndex ?? inputs.i;

          If(accumulatedAlpha.greaterThanEqual(float(COMPUTE_ALPHA_CUTOFF)), () => {
            Break();
          });

          const tileEntry = tileEntriesNode.element(entryOffset.add(entryIndex));
          const pageSplatOffset = uint(tileEntry.x);
          const pageSplatCount = uint(tileEntry.y);
          const samplingStride = uint(tileEntry.z).toVar();
          const scaleMultiplier = tileEntry.w;

          If(samplingStride.lessThan(uint(1)), () => {
            samplingStride.assign(uint(1));
          });

          Loop(
            {
              start: uint(0),
              end: pageSplatCount,
              type: 'uint',
              condition: '<',
              name: 'splatOffset',
              update: samplingStride,
            },
            (inputs: any) => {
              const splatOffset = inputs.splatOffset ?? inputs.i;

              If(accumulatedAlpha.greaterThanEqual(float(COMPUTE_ALPHA_CUTOFF)), () => {
                Break();
              });

              const splatIndex = pageSplatOffset.add(splatOffset);
              const sourcePosition = positionsNode.element(splatIndex);
              const sourceScale = scalesNode.element(splatIndex);
              const sourceColor = colorsNode.element(splatIndex);
              const localPosition = vec4(sourcePosition.xyz, 1);
              const viewPosition = vec3(
                modelViewRow0.dot(localPosition),
                modelViewRow1.dot(localPosition),
                modelViewRow2.dot(localPosition),
              );
              const viewDepth = viewPosition.z.negate().toVar();

              If(viewDepth.greaterThan(rangeParams.x), () => {
                const ndcX = float(0).toVar();
                const ndcY = float(0).toVar();
                const projectedRadiusPx = float(0).toVar();
                const baseRadius = max(sourceScale.x, max(sourceScale.y, sourceScale.z))
                  .mul(pointSizeParams.x)
                  .mul(scaleMultiplier);

                If(isPerspective, () => {
                  ndcX.assign(viewPosition.x.div(max(viewDepth.mul(cameraParams.y).mul(cameraParams.z), float(0.001))));
                  ndcY.assign(viewPosition.y.div(max(viewDepth.mul(cameraParams.y), float(0.001))));
                  projectedRadiusPx.assign(baseRadius.mul(viewportHeightFloat).div(max(viewDepth.mul(cameraParams.y), float(0.001))));
                }).Else(() => {
                  ndcX.assign(viewPosition.x.div(max(cameraParams.y, float(0.001))));
                  ndcY.assign(viewPosition.y.div(max(cameraParams.z, float(0.001))));
                  projectedRadiusPx.assign(baseRadius.mul(viewportHeightFloat).div(max(cameraParams.z.mul(2), float(0.001))));
                });

                const screenCenter = vec2(
                  ndcX.mul(0.5).add(0.5).mul(viewportWidthFloat),
                  float(0.5).sub(ndcY.mul(0.5)).mul(viewportHeightFloat),
                );
                const screenRadius = max(projectedRadiusPx, float(0.6));
                const normalizedDelta = pixelCenter.sub(screenCenter).div(vec2(screenRadius, screenRadius));
                const radial = lengthSq(normalizedDelta);

                If(radial.lessThan(1.0), () => {
                  const feather = smoothstep(1.0, 0.5, radial);
                  const gaussian = exp(radial.mul(-10.5));
                  const contributionAlpha = clamp(
                    sourcePosition.w.mul(pointSizeParams.y).mul(gaussian).mul(feather),
                    0,
                    1,
                  );

                  If(contributionAlpha.greaterThan(0.0015), () => {
                    const remainingAlpha = float(1).sub(accumulatedAlpha);
                    const blendedAlpha = contributionAlpha.mul(remainingAlpha);
                    const shadedColor = sourceColor.xyz.mul(tintParams.xyz).mul(pointSizeParams.z);
                    accumulatedColor.assign(accumulatedColor.add(shadedColor.mul(blendedAlpha)));
                    accumulatedAlpha.assign(min(float(1), accumulatedAlpha.add(blendedAlpha)));
                  });
                });
              });
            },
          );
        },
      );

      const resolvedColor = vec3(0, 0, 0).toVar();
      If(accumulatedAlpha.greaterThan(0.0001), () => {
        resolvedColor.assign(accumulatedColor.div(max(accumulatedAlpha, float(0.0001))));
      });

      textureStore(
        outputTexture,
        uvec2(pixelX, pixelY),
        vec4(resolvedColor, clamp(accumulatedAlpha, 0, 1)),
      );
    })().compute(outputPixelCount, [
      COMPUTE_WORKGROUP_SIZE,
    ]);
  }

  private updateComputeUniforms(
    camera: Camera,
    viewportWidth: number,
    viewportHeight: number,
    tileBinning: SplatClusterTileBinningSnapshot,
  ): void {
    this.computeViewportParams.set(
      viewportWidth,
      viewportHeight,
      tileBinning.tileSizePx,
      tileBinning.columns,
    );

    if (camera instanceof PerspectiveCamera) {
      this.computeCameraParams.set(
        1,
        Math.max(1e-4, Math.tan((camera.fov * Math.PI) / 360)),
        Math.max(1e-4, camera.aspect),
        0,
      );
    } else if (camera instanceof OrthographicCamera) {
      const halfWidth = Math.abs(camera.right - camera.left) / Math.max(2 * camera.zoom, 1e-5);
      const halfHeight = Math.abs(camera.top - camera.bottom) / Math.max(2 * camera.zoom, 1e-5);
      this.computeCameraParams.set(0, Math.max(1e-4, halfWidth), Math.max(1e-4, halfHeight), 0);
    } else {
      this.computeCameraParams.set(1, 1, Math.max(1e-4, viewportWidth / Math.max(1, viewportHeight)), 0);
    }

    this.computePointSizeParams.set(
      this.owner.splatMaterial.pointSize * 0.64,
      this.owner.splatMaterial.opacity,
      this.owner.splatMaterial.colorGain,
      0,
    );
    this.computeTintParams.set(
      this.owner.splatMaterial.tint.r,
      this.owner.splatMaterial.tint.g,
      this.owner.splatMaterial.tint.b,
      0,
    );
    this.computeRangeParams.set(this.resolveCameraNear(camera), this.resolveCameraFar(camera), 0, 0);

    const modelViewMatrixWorld = camera.matrixWorldInverse.clone().multiply(this.owner.matrixWorld);
    this.computeModelViewRow0.set(
      modelViewMatrixWorld.elements[0]!,
      modelViewMatrixWorld.elements[4]!,
      modelViewMatrixWorld.elements[8]!,
      modelViewMatrixWorld.elements[12]!,
    );
    this.computeModelViewRow1.set(
      modelViewMatrixWorld.elements[1]!,
      modelViewMatrixWorld.elements[5]!,
      modelViewMatrixWorld.elements[9]!,
      modelViewMatrixWorld.elements[13]!,
    );
    this.computeModelViewRow2.set(
      modelViewMatrixWorld.elements[2]!,
      modelViewMatrixWorld.elements[6]!,
      modelViewMatrixWorld.elements[10]!,
      modelViewMatrixWorld.elements[14]!,
    );
  }

  private syncComputeOverlay(
    camera: Camera,
    viewportWidth: number,
    viewportHeight: number,
    visible: boolean,
  ): void {
    if (!this.computeOverlay) {
      return;
    }

    this.computeOverlay.visible = visible;

    if (!visible) {
      return;
    }

    const planeDistance = Math.max(0.25, this.resolveCameraNear(camera) + 0.02);
    camera.getWorldDirection(this.computeForward);
    this.computeOverlayPosition.copy(camera.position).addScaledVector(this.computeForward, planeDistance);
    this.computeOverlayQuaternion.copy(camera.quaternion);

    if (camera instanceof PerspectiveCamera) {
      const halfHeight = Math.tan((camera.fov * Math.PI) / 360) * planeDistance;
      const halfWidth = halfHeight * camera.aspect;
      this.computeOverlayScale.set(halfWidth * 2, halfHeight * 2, 1);
    } else if (camera instanceof OrthographicCamera) {
      const halfWidth = Math.abs(camera.right - camera.left) / Math.max(2 * camera.zoom, 1e-5);
      const halfHeight = Math.abs(camera.top - camera.bottom) / Math.max(2 * camera.zoom, 1e-5);
      this.computeOverlayScale.set(halfWidth * 2, halfHeight * 2, 1);
    } else {
      const aspect = viewportWidth / Math.max(1, viewportHeight);
      this.computeOverlayScale.set(aspect * 0.6, 0.6, 1);
    }

    this.computeOwnerInverseMatrix.copy(this.owner.matrixWorld).invert();
    this.computeOverlay.matrix.compose(
      this.computeOverlayPosition,
      this.computeOverlayQuaternion,
      this.computeOverlayScale,
    );
    this.computeOverlay.matrix.premultiply(this.computeOwnerInverseMatrix);
    this.computeOverlay.matrix.decompose(
      this.computeOverlay.position,
      this.computeOverlay.quaternion,
      this.computeOverlay.scale,
    );
    this.computeOverlay.matrixWorldNeedsUpdate = true;
  }

  private syncGpuTilePath(
    clusteredActiveClusters: ReadonlyArray<SplatMeshSelection['activeClusters'][number]>,
    clusterSamplingStride: ReadonlyMap<number, number>,
    tileBinning: SplatClusterTileBinningSnapshot,
    camera: Camera,
  ): void {
    const asset = this.owner.getAsset();
    let instanceCount = 0;

    for (const activeCluster of clusteredActiveClusters) {
      const page = asset.pages[activeCluster.pageId]!;
      const samplingStride = clusterSamplingStride.get(activeCluster.clusterId) ?? 1;
      instanceCount += Math.ceil(page.splatCount / samplingStride);
    }

    this.ensureGpuSelectionCapacity(instanceCount);
    this.ensureGpuSprite();

    const selectionArray = this.gpuSelectionArray;
    let cursor = 0;

    for (const activeCluster of clusteredActiveClusters) {
      const cluster = asset.clusters[activeCluster.clusterId]!;
      const page = asset.pages[activeCluster.pageId]!;
      const samplingStride = clusterSamplingStride.get(cluster.id) ?? 1;
      const densityCompensation = this.resolveDensityCompensation(samplingStride);
      const scaleMultiplier = this.resolveCoverageScaleBoost(cluster, densityCompensation);
      const pageDescriptorOffset = page.id * 8;
      const pageSplatOffset = asset.buffers.pageDescriptors[pageDescriptorOffset + 6] ?? 0;
      const projectedCluster = tileBinning.clusterProjections.get(cluster.id) ?? null;
      const depthSortedIndices = this.shouldDepthSortCluster(cluster, projectedCluster, samplingStride)
        ? this.getDepthSortedPageIndices(page, camera)
        : null;

      if (depthSortedIndices) {
        for (let orderIndex = 0; orderIndex < depthSortedIndices.length; orderIndex += samplingStride) {
          const splatIndex = depthSortedIndices[orderIndex]!;
          const offset = cursor * 4;
          selectionArray[offset + 0] = pageSplatOffset + splatIndex;
          selectionArray[offset + 1] = cluster.id;
          selectionArray[offset + 2] = scaleMultiplier;
          selectionArray[offset + 3] = 0;
          cursor += 1;
        }
        continue;
      }

      for (let splatIndex = 0; splatIndex < page.splatCount; splatIndex += samplingStride) {
        const offset = cursor * 4;
        selectionArray[offset + 0] = pageSplatOffset + splatIndex;
        selectionArray[offset + 1] = cluster.id;
        selectionArray[offset + 2] = scaleMultiplier;
        selectionArray[offset + 3] = 0;
        cursor += 1;
      }
    }

    if (this.gpuSelectionAttribute) {
      this.gpuSelectionAttribute.needsUpdate = true;
      this.gpuSelectionAttribute.clearUpdateRanges();
      this.gpuSelectionAttribute.addUpdateRange(0, cursor * 4);
    }

    this.syncGpuSprite(cursor);
    this.snapshot = this.buildSnapshot(cursor, tileBinning);
  }

  private ensureGpuSelectionCapacity(requiredCapacity: number): void {
    if (this.gpuSelectionArray.length >= requiredCapacity * 4) {
      return;
    }

    const nextCapacity = this.resolveCapacity(
      Math.floor(this.gpuSelectionArray.length / 4),
      requiredCapacity,
      256,
    );
    this.gpuSelectionArray = new Float32Array(nextCapacity * 4);
    this.gpuSelectionAttribute = this.createStorageAttribute(this.gpuSelectionArray, 4, true);

    if (this.gpuSprite) {
      this.owner.remove(this.gpuSprite);
      this.gpuSprite = undefined;
      this.gpuMaterial = undefined;
      this.gpuMaterialVersion = -1;
    }
  }

  private ensureGpuSprite(): void {
    if (
      this.gpuSprite
      && this.gpuMaterial
      && this.gpuSelectionAttribute
      && this.gpuPackedPositionsOpacityAttribute
      && this.gpuPackedScalesAttribute
      && this.gpuPackedRotationsAttribute
      && this.gpuPackedColorsAttribute
      && this.gpuMaterialVersion === this.owner.getMaterialVersion()
    ) {
      return;
    }

    if (this.gpuSprite) {
      this.owner.remove(this.gpuSprite);
      this.gpuSprite = undefined;
      this.gpuMaterial = undefined;
    }

    this.ensurePackedSplatStorageAttributes();
    this.gpuSelectionAttribute ??= this.createStorageAttribute(this.gpuSelectionArray, 4, true);
    this.gpuMaterial = this.createGpuMaterial(
      this.gpuSelectionAttribute,
      this.gpuPackedPositionsOpacityAttribute!,
      this.gpuPackedScalesAttribute!,
      this.gpuPackedRotationsAttribute!,
      this.gpuPackedColorsAttribute!,
    );
    this.gpuMaterialVersion = this.owner.getMaterialVersion();
    this.gpuSprite = new Sprite(this.gpuMaterial);
    this.gpuSprite.count = 0;
    this.gpuSprite.frustumCulled = false;
    this.gpuSprite.renderOrder = 21;
    this.gpuSprite.name = 'SparkGpuTileSpriteQueue';
    this.owner.add(this.gpuSprite);
  }

  private createGpuMaterial(
    selectionAttribute: StorageInstancedBufferAttribute,
    positionsOpacityAttribute: StorageInstancedBufferAttribute,
    scalesAttribute: StorageInstancedBufferAttribute,
    rotationsAttribute: StorageInstancedBufferAttribute,
    colorsAttribute: StorageInstancedBufferAttribute,
  ): SpriteNodeMaterial {
    const material = new SpriteNodeMaterial();
    const uvNode = uv();
    const centeredUv = uvNode.sub(vec2(0.5)).mul(2.0);
    const radial = lengthSq(centeredUv);
    const feather = smoothstep(1.0, 0.5, radial);
    const selectionNode = storage(selectionAttribute, 'vec4', Math.max(1, Math.floor(selectionAttribute.array.length / 4)));
    const positionsNode = storage(positionsOpacityAttribute, 'vec4', Math.max(1, Math.floor(positionsOpacityAttribute.array.length / 4)));
    const scalesNode = storage(scalesAttribute, 'vec4', Math.max(1, Math.floor(scalesAttribute.array.length / 4)));
    const rotationsNode = storage(rotationsAttribute, 'vec4', Math.max(1, Math.floor(rotationsAttribute.array.length / 4)));
    const colorsNode = storage(colorsAttribute, 'vec4', Math.max(1, Math.floor(colorsAttribute.array.length / 4)));
    const selectionEntry = selectionNode.element(instanceIndex);
    const splatIndex = uint(selectionEntry.x);
    const sourcePosition = positionsNode.element(splatIndex);
    const sourceScale = scalesNode.element(splatIndex);
    const sourceRotation = rotationsNode.element(splatIndex);
    const sourceColor = colorsNode.element(splatIndex);
    const alphaNode = sourcePosition.w.mul(float(this.owner.splatMaterial.opacity)).mul(exp(radial.mul(-10.5))).mul(feather);
    const scaledAxes = sourceScale.xyz.mul(selectionEntry.z).mul(float(this.owner.splatMaterial.pointSize * 0.64));
    const quaternionXYZ = sourceRotation.xyz;
    const quaternionW = sourceRotation.w;
    const rotateAxis = (axisNode: any) => {
      const doubleCross = cross(quaternionXYZ, axisNode).mul(2);
      return axisNode.add(doubleCross.mul(quaternionW)).add(cross(quaternionXYZ, doubleCross));
    };
    const axisXView = modelViewMatrix.mul(vec4(rotateAxis(vec3(scaledAxes.x, 0, 0)), 0)).xyz;
    const axisYView = modelViewMatrix.mul(vec4(rotateAxis(vec3(0, scaledAxes.y, 0)), 0)).xyz;
    const axisZView = modelViewMatrix.mul(vec4(rotateAxis(vec3(0, 0, scaledAxes.z)), 0)).xyz;
    const covarianceXX = axisXView.x.mul(axisXView.x).add(axisYView.x.mul(axisYView.x)).add(axisZView.x.mul(axisZView.x));
    const covarianceXY = axisXView.x.mul(axisXView.y).add(axisYView.x.mul(axisYView.y)).add(axisZView.x.mul(axisZView.y));
    const covarianceYY = axisXView.y.mul(axisXView.y).add(axisYView.y.mul(axisYView.y)).add(axisZView.y.mul(axisZView.y));
    const trace = covarianceXX.add(covarianceYY);
    const delta = sqrt(max(0.000001, covarianceXX.sub(covarianceYY).mul(covarianceXX.sub(covarianceYY)).add(covarianceXY.mul(covarianceXY).mul(4))));
    const majorScale = sqrt(max(0.000016, trace.add(delta).mul(0.5)));
    const minorScale = sqrt(max(0.000004, trace.sub(delta).mul(0.5)));
    const ellipseRotation = atan(covarianceXY.mul(2), covarianceXX.sub(covarianceYY)).mul(0.5);

    material.positionNode = sourcePosition.xyz;
    material.scaleNode = vec2(majorScale, minorScale);
    material.rotationNode = ellipseRotation;
    material.colorNode = sourceColor.xyz
      .mul(vec3(
        this.owner.splatMaterial.tint.r * this.owner.splatMaterial.colorGain,
        this.owner.splatMaterial.tint.g * this.owner.splatMaterial.colorGain,
        this.owner.splatMaterial.tint.b * this.owner.splatMaterial.colorGain,
      ))
      .mul(alphaNode);
    material.opacityNode = alphaNode;
    material.maskNode = radial.lessThan(1.0);
    material.transparent = true;
    material.depthWrite = false;
    material.sizeAttenuation = true;
    material.side = DoubleSide;
    material.blending = NormalBlending;
    material.premultipliedAlpha = true;

    return material;
  }

  private syncGpuSprite(instanceCount: number): void {
    if (!this.gpuSprite) {
      return;
    }

    this.gpuSprite.count = instanceCount;
    this.gpuSprite.visible = instanceCount > 0;
  }

  private buildSnapshot(
    instanceCount: number,
    tileBinning: SplatClusterTileBinningSnapshot,
    overrides: {
      tileWorkPeak?: number;
      tileBufferBytes?: number;
    } = {},
  ): SplatCompositorSnapshot {
    return {
      weightedInstances: instanceCount,
      heroInstances: 0,
      depthSlicedInstances: 0,
      activeTiles: tileBinning.activeTiles,
      weightedTiles: tileBinning.activeTiles,
      depthSlicedTiles: 0,
      heroTiles: 0,
      maxTileComplexity: overrides.tileWorkPeak ?? tileBinning.maxTileSplatEstimate,
      visibleClusters: tileBinning.visibleClusterCount,
      binnedClusters: tileBinning.binnedClusterCount,
      binnedClusterReferences: tileBinning.binnedClusterReferences,
      overflowedTiles: tileBinning.overflowedTiles,
      overflowedClusterReferences: tileBinning.overflowedClusterReferences,
      maxClustersPerTile: tileBinning.maxTileClusterCount,
      maxTileSplatEstimate: overrides.tileWorkPeak ?? tileBinning.maxTileSplatEstimate,
      tileBufferBytes: overrides.tileBufferBytes ?? (
        tileBinning.buffers.clusterScreenData.byteLength
        + tileBinning.buffers.tileHeaders.byteLength
        + tileBinning.buffers.tileEntries.byteLength
      ),
    };
  }

  private resolveClusterSamplingStride(
    tileBinning: SplatClusterTileBinningSnapshot,
    budgets: SplatBudgetOptions,
  ): Map<number, number> {
    const targetStride = new Map<number, number>();
    const clusterStride = new Map<number, number>();
    let reductionHash = 2166136261;

    for (const tile of tileBinning.tileBins) {
      const clusterPressure = tile.clusterIds.length / Math.max(1, budgets.maxClustersPerTile);
      const splatPressure = tile.splatEstimate / Math.max(1, budgets.maxSplatsPerTile);
      const overflowPressure = tile.overflowCount > 0
        ? 1 + tile.overflowCount / Math.max(1, budgets.maxClustersPerTile)
        : 1;
      const reductionFactor = Math.max(1, splatPressure, clusterPressure * 0.92, overflowPressure);

      if (reductionFactor <= 1.05) {
        continue;
      }

      for (const clusterId of tile.clusterIds) {
        const projectedCluster = tileBinning.clusterProjections.get(clusterId);

        if (projectedCluster) {
          const closeCluster = projectedCluster.screenRadius >= 18;
          const leafCluster = projectedCluster.representedSplatCount <= projectedCluster.splatCount * 1.05;

          if (closeCluster || leafCluster) {
            targetStride.set(clusterId, 1);
            continue;
          }
        }

        const nextStride = Math.max(
          targetStride.get(clusterId) ?? 1,
          Math.min(COMPUTE_MAX_SAMPLING_STRIDE, Math.ceil(reductionFactor)),
        );
        targetStride.set(clusterId, nextStride);
      }
    }

    const activeClusterIds = new Set(tileBinning.clusterOrder);

    for (const clusterId of activeClusterIds) {
      const previousStride = this.stableClusterSamplingStride.get(clusterId) ?? 1;
      const desiredStride = targetStride.get(clusterId) ?? 1;
      let resolvedStride = desiredStride;

      if (desiredStride < previousStride) {
        resolvedStride = desiredStride <= previousStride - 2
          ? previousStride - 1
          : previousStride;
      }

      resolvedStride = Math.max(1, Math.min(COMPUTE_MAX_SAMPLING_STRIDE, resolvedStride));
      clusterStride.set(clusterId, resolvedStride);
      this.stableClusterSamplingStride.set(clusterId, resolvedStride);
      reductionHash = Math.imul(reductionHash ^ clusterId, 16777619) >>> 0;
      reductionHash = Math.imul(reductionHash ^ resolvedStride, 16777619) >>> 0;
    }

    for (const clusterId of [...this.stableClusterSamplingStride.keys()]) {
      if (activeClusterIds.has(clusterId)) {
        continue;
      }

      this.stableClusterSamplingStride.delete(clusterId);
    }

    this.lastReductionHash = reductionHash;
    return clusterStride;
  }

  private resolveTileDebugColor(
    target: Color,
    projectedCluster: SplatProjectedCluster | null,
    tileBinning: SplatClusterTileBinningSnapshot,
  ): void {
    if (!projectedCluster) {
      target.set(0xf8fafc);
      return;
    }

    const headerOffset = projectedCluster.dominantTileId * 4;
    const dominantTileClusterCount = tileBinning.buffers.tileHeaders[headerOffset + 1] ?? 0;
    const dominantTileSplatEstimate = tileBinning.buffers.tileHeaders[headerOffset + 2] ?? 0;

    if (this.owner.splatMaterial.debugMode === 'tile-occupancy') {
      const occupancy = dominantTileClusterCount / Math.max(1, tileBinning.maxTileClusterCount);
      target.setHSL(0.62 - occupancy * 0.62, 0.82, 0.58);
      return;
    }

    if (this.owner.splatMaterial.debugMode === 'tile-heatmap') {
      const heat = dominantTileSplatEstimate / Math.max(1, tileBinning.maxTileSplatEstimate);
      target.setHSL(0.7 - heat * 0.7, 0.88, 0.56);
      return;
    }

    const bucketMix = projectedCluster.depthBucket / Math.max(1, tileBinning.depthBucketCount - 1);
    target.setHSL(0.75 - bucketMix * 0.55, 0.78, 0.6);
  }

  private resolveRepresentationDebugColor(
    target: Color,
    cluster: ReturnType<SplatMesh['getAsset']>['clusters'][number],
  ): void {
    const representationRatio = cluster.representedSplatCount / Math.max(1, cluster.splatCount);
    const leafCluster = cluster.childIds.length === 0;

    if (leafCluster || representationRatio <= 1.1) {
      target.set(0x93c5fd);
      return;
    }

    const proxyMix = Math.min(1, Math.log2(Math.max(1, representationRatio)) / 4);
    target.setHSL(0.56 - proxyMix * 0.48, 0.82, 0.58);
  }

  private syncClusterBoundsOverlay(tileBinning: SplatClusterTileBinningSnapshot): void {
    if (this.owner.splatMaterial.debugMode !== 'cluster-bounds' || tileBinning.visibleClusterCount === 0) {
      if (this.clusterBoundsOverlay) {
        this.clusterBoundsOverlay.visible = false;
      }
      return;
    }

    this.ensureClusterBoundsOverlay();

    const asset = this.owner.getAsset();
    const visibleClusterIds = tileBinning.clusterOrder.filter((clusterId) => tileBinning.clusterProjections.has(clusterId));
    this.ensureClusterBoundsCapacity(visibleClusterIds.length);

    let vertexCursor = 0;

    for (const clusterId of visibleClusterIds) {
      const cluster = asset.clusters[clusterId]!;
      const projection = tileBinning.clusterProjections.get(clusterId)!;
      const occupancy = projection.tileCount / Math.max(1, tileBinning.maxTileClusterCount);
      this.debugColorScratch.setHSL(
        Math.min(0.85, cluster.level * 0.12 + 0.08),
        0.72,
        Math.min(0.72, 0.42 + occupancy * 0.18),
      );
      vertexCursor = this.writeAabbOutline(
        vertexCursor,
        cluster.boundsMin,
        cluster.boundsMax,
        this.debugColorScratch,
      );
    }

    if (!this.clusterBoundsOverlay || !this.clusterBoundsGeometry) {
      return;
    }

    this.clusterBoundsOverlay.visible = true;
    this.clusterBoundsGeometry.setDrawRange(0, vertexCursor);

    if (this.clusterBoundsPositionAttribute && this.clusterBoundsColorAttribute) {
      this.clusterBoundsPositionAttribute.needsUpdate = true;
      this.clusterBoundsColorAttribute.needsUpdate = true;
    }
  }

  private ensureClusterBoundsOverlay(): void {
    if (this.clusterBoundsOverlay && this.clusterBoundsGeometry && this.clusterBoundsMaterial) {
      return;
    }

    this.clusterBoundsGeometry = new BufferGeometry();
    this.clusterBoundsPositionAttribute = new BufferAttribute(this.clusterBoundsPositionArray, 3);
    this.clusterBoundsColorAttribute = new BufferAttribute(this.clusterBoundsColorArray, 3);
    this.clusterBoundsGeometry.setAttribute('position', this.clusterBoundsPositionAttribute);
    this.clusterBoundsGeometry.setAttribute('color', this.clusterBoundsColorAttribute);
    this.clusterBoundsMaterial = new LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.82,
      depthTest: false,
      toneMapped: false,
    });
    this.clusterBoundsOverlay = new LineSegments(this.clusterBoundsGeometry, this.clusterBoundsMaterial);
    this.clusterBoundsOverlay.frustumCulled = false;
    this.clusterBoundsOverlay.renderOrder = 28;
    this.clusterBoundsOverlay.visible = false;
    this.clusterBoundsOverlay.name = 'SparkClusterBoundsOverlay';
    this.owner.add(this.clusterBoundsOverlay);
  }

  private ensureClusterBoundsCapacity(clusterCount: number): void {
    const requiredVertices = clusterCount * 24;

    if (this.clusterBoundsPositionArray.length >= requiredVertices * 3) {
      return;
    }

    const nextCapacity = this.resolveCapacity(
      Math.floor(this.clusterBoundsPositionArray.length / 3 / 24),
      clusterCount,
      16,
    ) * 24;

    this.clusterBoundsPositionArray = new Float32Array(nextCapacity * 3);
    this.clusterBoundsColorArray = new Float32Array(nextCapacity * 3);

    if (!this.clusterBoundsGeometry) {
      return;
    }

    this.clusterBoundsPositionAttribute = new BufferAttribute(this.clusterBoundsPositionArray, 3);
    this.clusterBoundsColorAttribute = new BufferAttribute(this.clusterBoundsColorArray, 3);
    this.clusterBoundsGeometry.setAttribute('position', this.clusterBoundsPositionAttribute);
    this.clusterBoundsGeometry.setAttribute('color', this.clusterBoundsColorAttribute);
  }

  private writeAabbOutline(
    vertexCursor: number,
    boundsMin: readonly [number, number, number],
    boundsMax: readonly [number, number, number],
    color: Color,
  ): number {
    const minX = boundsMin[0];
    const minY = boundsMin[1];
    const minZ = boundsMin[2];
    const maxX = boundsMax[0];
    const maxY = boundsMax[1];
    const maxZ = boundsMax[2];
    const corners: Array<[number, number, number]> = [
      [minX, minY, minZ],
      [maxX, minY, minZ],
      [maxX, maxY, minZ],
      [minX, maxY, minZ],
      [minX, minY, maxZ],
      [maxX, minY, maxZ],
      [maxX, maxY, maxZ],
      [minX, maxY, maxZ],
    ];
    const edges: Array<[number, number]> = [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 4],
      [0, 4],
      [1, 5],
      [2, 6],
      [3, 7],
    ];

    for (const [startIndex, endIndex] of edges) {
      vertexCursor = this.writeDebugVertex(vertexCursor, corners[startIndex]!, color);
      vertexCursor = this.writeDebugVertex(vertexCursor, corners[endIndex]!, color);
    }

    return vertexCursor;
  }

  private writeDebugVertex(
    vertexCursor: number,
    position: readonly [number, number, number],
    color: Color,
  ): number {
    const offset = vertexCursor * 3;
    this.clusterBoundsPositionArray[offset + 0] = position[0];
    this.clusterBoundsPositionArray[offset + 1] = position[1];
    this.clusterBoundsPositionArray[offset + 2] = position[2];
    this.clusterBoundsColorArray[offset + 0] = color.r;
    this.clusterBoundsColorArray[offset + 1] = color.g;
    this.clusterBoundsColorArray[offset + 2] = color.b;
    return vertexCursor + 1;
  }

  private writeInstance(
    index: number,
    x: number,
    y: number,
    z: number,
    scaleX: number,
    scaleY: number,
    scaleZ: number,
    rotationX: number,
    rotationY: number,
    rotationZ: number,
    rotationW: number,
    color: Color,
    opacity: number,
  ): void {
    const positionOffset = index * 3;
    const scaleOffset = index * 3;
    const rotationOffset = index * 4;
    this.positionArray[positionOffset + 0] = x;
    this.positionArray[positionOffset + 1] = y;
    this.positionArray[positionOffset + 2] = z;
    this.scaleArray[scaleOffset + 0] = scaleX;
    this.scaleArray[scaleOffset + 1] = scaleY;
    this.scaleArray[scaleOffset + 2] = scaleZ;
    this.rotationArray[rotationOffset + 0] = rotationX;
    this.rotationArray[rotationOffset + 1] = rotationY;
    this.rotationArray[rotationOffset + 2] = rotationZ;
    this.rotationArray[rotationOffset + 3] = rotationW;
    this.colorArray[positionOffset + 0] = color.r;
    this.colorArray[positionOffset + 1] = color.g;
    this.colorArray[positionOffset + 2] = color.b;
    this.opacityArray[index] = opacity;
  }

  private canUsePreparedPages(): boolean {
    return this.owner.splatMaterial.debugMode === 'albedo' && this.owner.effectStack.isIdentity();
  }

  private getPreparedPageCache(
    page: ReturnType<SplatMesh['getAsset']>['pages'][number],
  ): PreparedPageCache {
    const materialVersion = this.owner.getMaterialVersion();
    const cached = this.preparedPageCache.get(page.id);

    if (cached && cached.materialVersion === materialVersion) {
      return cached;
    }

    const baseScales = new Float32Array(page.splatCount * 3);
    const rotations = new Float32Array(page.rotations);
    const colors = new Float32Array(page.splatCount * 3);
    const opacities = new Float32Array(page.splatCount);
    const tint = this.owner.splatMaterial.tint;
    const colorGain = this.owner.splatMaterial.colorGain;
    const opacity = this.owner.splatMaterial.opacity;

    for (let splatIndex = 0; splatIndex < page.splatCount; splatIndex += 1) {
      const sourceOffset = splatIndex * 3;
      baseScales[sourceOffset + 0] = this.resolveBaseScale(page.scales[sourceOffset + 0]!);
      baseScales[sourceOffset + 1] = this.resolveBaseScale(page.scales[sourceOffset + 1]!);
      baseScales[sourceOffset + 2] = this.resolveBaseScale(page.scales[sourceOffset + 2]!);
      colors[sourceOffset + 0] = page.colors[sourceOffset + 0]! * tint.r * colorGain;
      colors[sourceOffset + 1] = page.colors[sourceOffset + 1]! * tint.g * colorGain;
      colors[sourceOffset + 2] = page.colors[sourceOffset + 2]! * tint.b * colorGain;
      opacities[splatIndex] = Math.min(1, opacity * page.opacities[splatIndex]!);
    }

    const preparedPage: PreparedPageCache = {
      materialVersion,
      baseScales,
      rotations,
      colors,
      opacities,
    };

    this.preparedPageCache.set(page.id, preparedPage);
    return preparedPage;
  }

  private copyPreparedPage(
    sourcePositions: Float32Array,
    sourceScales: Float32Array,
    sourceRotations: Float32Array,
    sourceColors: Float32Array,
    sourceOpacities: Float32Array,
    targetInstanceOffset: number,
    scaleMultiplier: number,
  ): void {
    this.positionArray.set(sourcePositions, targetInstanceOffset * 3);
    this.rotationArray.set(sourceRotations, targetInstanceOffset * 4);
    this.colorArray.set(sourceColors, targetInstanceOffset * 3);
    this.opacityArray.set(sourceOpacities, targetInstanceOffset);

    const targetScaleOffset = targetInstanceOffset * 3;

    if (Math.abs(scaleMultiplier - 1) <= 1e-4) {
      this.scaleArray.set(sourceScales, targetScaleOffset);
      return;
    }

    for (let index = 0; index < sourceScales.length; index += 1) {
      this.scaleArray[targetScaleOffset + index] = sourceScales[index]! * scaleMultiplier;
    }
  }

  private copyPreparedPageStrided(
    preparedPage: PreparedPageCache,
    sourcePositions: Float32Array,
    targetInstanceOffset: number,
    samplingStride: number,
    scaleMultiplier: number,
  ): number {
    let written = 0;

    for (let splatIndex = 0; splatIndex < preparedPage.opacities.length; splatIndex += samplingStride) {
      const sourceOffset = splatIndex * 3;
      const rotationOffset = splatIndex * 4;
      const targetIndex = targetInstanceOffset + written;

      this.writeInstance(
        targetIndex,
        sourcePositions[sourceOffset + 0]!,
        sourcePositions[sourceOffset + 1]!,
        sourcePositions[sourceOffset + 2]!,
        preparedPage.baseScales[sourceOffset + 0]! * scaleMultiplier,
        preparedPage.baseScales[sourceOffset + 1]! * scaleMultiplier,
        preparedPage.baseScales[sourceOffset + 2]! * scaleMultiplier,
        preparedPage.rotations[rotationOffset + 0]!,
        preparedPage.rotations[rotationOffset + 1]!,
        preparedPage.rotations[rotationOffset + 2]!,
        preparedPage.rotations[rotationOffset + 3]!,
        this.colorScratch.setRGB(
          preparedPage.colors[sourceOffset + 0]!,
          preparedPage.colors[sourceOffset + 1]!,
          preparedPage.colors[sourceOffset + 2]!,
        ),
        preparedPage.opacities[splatIndex]!,
      );
      written += 1;
    }

    return written;
  }

  private copyPreparedPageDepthSorted(
    preparedPage: PreparedPageCache,
    sourcePositions: Float32Array,
    sortedIndices: Uint32Array,
    targetInstanceOffset: number,
    samplingStride: number,
    scaleMultiplier: number,
  ): number {
    let written = 0;

    for (let orderIndex = 0; orderIndex < sortedIndices.length; orderIndex += samplingStride) {
      const splatIndex = sortedIndices[orderIndex]!;
      const sourceOffset = splatIndex * 3;
      const rotationOffset = splatIndex * 4;
      const targetIndex = targetInstanceOffset + written;

      this.writeInstance(
        targetIndex,
        sourcePositions[sourceOffset + 0]!,
        sourcePositions[sourceOffset + 1]!,
        sourcePositions[sourceOffset + 2]!,
        preparedPage.baseScales[sourceOffset + 0]! * scaleMultiplier,
        preparedPage.baseScales[sourceOffset + 1]! * scaleMultiplier,
        preparedPage.baseScales[sourceOffset + 2]! * scaleMultiplier,
        preparedPage.rotations[rotationOffset + 0]!,
        preparedPage.rotations[rotationOffset + 1]!,
        preparedPage.rotations[rotationOffset + 2]!,
        preparedPage.rotations[rotationOffset + 3]!,
        this.colorScratch.setRGB(
          preparedPage.colors[sourceOffset + 0]!,
          preparedPage.colors[sourceOffset + 1]!,
          preparedPage.colors[sourceOffset + 2]!,
        ),
        preparedPage.opacities[splatIndex]!,
      );
      written += 1;
    }

    return written;
  }

  private sortClustersBackToFront(
    activeClusters: ReadonlyArray<SplatMeshSelection['activeClusters'][number]>,
    tileBinning: SplatClusterTileBinningSnapshot,
  ): ReadonlyArray<SplatMeshSelection['activeClusters'][number]> {
    const orderedClusters = [...activeClusters];

    orderedClusters.sort((left, right) => {
      const leftProjection = tileBinning.clusterProjections.get(left.clusterId);
      const rightProjection = tileBinning.clusterProjections.get(right.clusterId);
      const leftDepth = leftProjection?.projectedDepth ?? -1;
      const rightDepth = rightProjection?.projectedDepth ?? -1;

      if (Math.abs(rightDepth - leftDepth) > 1e-4) {
        return rightDepth - leftDepth;
      }

      const leftRadius = leftProjection?.screenRadius ?? 0;
      const rightRadius = rightProjection?.screenRadius ?? 0;

      if (Math.abs(rightRadius - leftRadius) > 0.25) {
        return rightRadius - leftRadius;
      }

      return left.clusterId - right.clusterId;
    });

    return orderedClusters;
  }

  private resolveCameraOrderingSignature(camera: Camera): string {
    camera.getWorldDirection(this.computeForward);
    const position = camera.position;
    return [
      Math.round(position.x * 20),
      Math.round(position.y * 20),
      Math.round(position.z * 20),
      Math.round(this.computeForward.x * 200),
      Math.round(this.computeForward.y * 200),
      Math.round(this.computeForward.z * 200),
    ].join(',');
  }

  private shouldDepthSortCluster(
    cluster: ReturnType<SplatMesh['getAsset']>['clusters'][number],
    projectedCluster: SplatProjectedCluster | null,
    samplingStride: number,
  ): boolean {
    if (samplingStride > 2) {
      return false;
    }

    if (!projectedCluster) {
      return cluster.childIds.length === 0;
    }

    const isLeafCluster = cluster.childIds.length === 0;
    const representedRatio = cluster.representedSplatCount / Math.max(1, cluster.splatCount);
    const isNearLeafProxy = representedRatio <= 1.15;
    const projectedRadius = projectedCluster.screenRadius;
    return projectedRadius >= 14 || (projectedRadius >= 8 && (isLeafCluster || isNearLeafProxy));
  }

  private getDepthSortedPageIndices(
    page: ReturnType<SplatMesh['getAsset']>['pages'][number],
    camera: Camera,
  ): Uint32Array {
    let indexBuffer = this.depthSortIndexCache.get(page.id);

    if (!indexBuffer || indexBuffer.length !== page.splatCount) {
      indexBuffer = new Uint32Array(page.splatCount);
      this.depthSortIndexCache.set(page.id, indexBuffer);
    }

    let depthBuffer = this.depthSortValueCache.get(page.id);

    if (!depthBuffer || depthBuffer.length !== page.splatCount) {
      depthBuffer = new Float32Array(page.splatCount);
      this.depthSortValueCache.set(page.id, depthBuffer);
    }

    this.depthSortModelView.copy(camera.matrixWorldInverse).multiply(this.owner.matrixWorld);
    const elements = this.depthSortModelView.elements;

    for (let splatIndex = 0; splatIndex < page.splatCount; splatIndex += 1) {
      const offset = splatIndex * 3;
      const x = page.positions[offset + 0]!;
      const y = page.positions[offset + 1]!;
      const z = page.positions[offset + 2]!;
      indexBuffer[splatIndex] = splatIndex;
      depthBuffer[splatIndex] = -(elements[2]! * x + elements[6]! * y + elements[10]! * z + elements[14]!);
    }

    indexBuffer.sort((leftIndex, rightIndex) => depthBuffer[rightIndex]! - depthBuffer[leftIndex]!);
    return indexBuffer;
  }

  private resolveCapacity(
    currentCapacity: number,
    requiredCapacity: number,
    minimumCapacity: number,
  ): number {
    if (requiredCapacity <= currentCapacity) {
      return Math.max(minimumCapacity, currentCapacity);
    }

    return Math.max(
      minimumCapacity,
      requiredCapacity,
      Math.ceil(Math.max(1, currentCapacity) * 1.5),
    );
  }

  private resolveCoverageScaleBoost(
    cluster: ReturnType<SplatMesh['getAsset']>['clusters'][number],
    densityCompensation = 1,
  ): number {
    const representationGap = cluster.representedSplatCount / Math.max(1, cluster.splatCount);
    const proxyBias = cluster.childIds.length === 0
      ? 1
      : 1 + Math.min(0.9, Math.log2(Math.max(1, representationGap)) * 0.16);
    return Math.min(2.75, (1 + Math.log2(Math.max(1, representationGap)) * 0.08) * densityCompensation * proxyBias);
  }

  private resolveDensityCompensation(samplingStride: number): number {
    if (samplingStride <= 1) {
      return 1;
    }

    // Scale compensation keeps a strided cluster from turning visibly sparse
    // before the hierarchy/proxy path takes over in later phases.
    return Math.min(1.75, 1 + Math.log2(samplingStride) * 0.25);
  }

  private resolveBaseScale(
    scale: number,
    multiplier = 1,
  ): number {
    return Math.max(0.012, scale * this.owner.splatMaterial.pointSize * 0.64 * multiplier);
  }

  private resolveMaxProjectedClusterRadius(tileBinning: SplatClusterTileBinningSnapshot): number {
    let maxRadius = 0;

    for (const projection of tileBinning.clusterProjections.values()) {
      maxRadius = Math.max(maxRadius, projection.screenRadius);
    }

    return maxRadius;
  }

  private resolveCameraNear(camera: Camera): number {
    if (camera instanceof PerspectiveCamera || camera instanceof OrthographicCamera) {
      return camera.near;
    }

    return 0.01;
  }

  private resolveCameraFar(camera: Camera): number {
    if (camera instanceof PerspectiveCamera || camera instanceof OrthographicCamera) {
      return camera.far;
    }

    return 10_000;
  }

}

export { SplatSpriteCompositor as SplatTileCompositor };
