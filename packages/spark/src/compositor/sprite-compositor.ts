import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  Color,
  DoubleSide,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  LineBasicMaterial,
  LineSegments,
  Sprite,
} from 'three';
import { SpriteNodeMaterial } from 'three/webgpu';
import {
  atan,
  cross,
  exp,
  instancedBufferAttribute,
  max,
  modelViewMatrix,
  smoothstep,
  sqrt,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { Camera } from 'three';
import { lengthSq } from 'three/tsl';
import {
  SplatClusterTileBinner,
  type SplatClusterTileBinningSnapshot,
  type SplatProjectedCluster,
} from './cluster-tile-binner';
import { SPLAT_SEMANTIC_FLAGS } from '../core/semantics';
import type { SplatBudgetOptions } from '../core/budgets';
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

export class SplatSpriteCompositor {
  private readonly preparedPageCache = new Map<number, PreparedPageCache>();
  private readonly tileBinner = new SplatClusterTileBinner();

  private sprite?: Sprite;
  private material?: SpriteNodeMaterial;
  private positionAttribute?: InstancedBufferAttribute;
  private scaleAttribute?: InstancedBufferAttribute;
  private rotationAttribute?: InstancedBufferAttribute;
  private colorAttribute?: InstancedBufferAttribute;
  private opacityAttribute?: InstancedBufferAttribute;
  private clusterBoundsOverlay?: LineSegments;
  private clusterBoundsGeometry?: BufferGeometry;
  private clusterBoundsMaterial?: LineBasicMaterial;
  private clusterBoundsPositionAttribute?: BufferAttribute;
  private clusterBoundsColorAttribute?: BufferAttribute;

  private positionArray = new Float32Array(0);
  private scaleArray = new Float32Array(0);
  private rotationArray = new Float32Array(0);
  private colorArray = new Float32Array(0);
  private opacityArray = new Float32Array(0);
  private clusterBoundsPositionArray = new Float32Array(0);
  private clusterBoundsColorArray = new Float32Array(0);

  private readonly colorScratch = new Color();
  private readonly effectTintScratch = new Color();
  private readonly debugColorScratch = new Color();
  private snapshot: SplatCompositorSnapshot = EMPTY_SNAPSHOT;
  private lastBuildSignature = '';
  private lastReductionHash = 2166136261;

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
    const rebuildDependsOnBinning = this.owner.splatMaterial.debugMode === 'tile-occupancy'
      || this.owner.splatMaterial.debugMode === 'tile-heatmap'
      || this.owner.splatMaterial.debugMode === 'depth-buckets';
    const buildSignature = [
      clusteredActiveClusters.map((cluster) => cluster.clusterId).join(','),
      this.owner.getMaterialVersion(),
      this.owner.getEffectVersion(),
      frameBucket,
      rebuildDependsOnBinning ? tileBinning.debugHash : 'static',
      this.lastReductionHash,
    ].join('::');

    this.syncClusterBoundsOverlay(tileBinning);
    this.snapshot = this.buildSnapshot(0, tileBinning);

    if (clusteredActiveClusters.length === 0) {
      this.lastBuildSignature = '';
      this.syncSprite(0);
      return;
    }

    if (buildSignature === this.lastBuildSignature) {
      return;
    }

    this.lastBuildSignature = buildSignature;

    let queueEstimate = 0;

    for (const activeCluster of clusteredActiveClusters) {
      const page = asset.pages[activeCluster.pageId]!;
      const samplingStride = clusterSamplingStride.get(activeCluster.clusterId) ?? 1;
      queueEstimate += Math.ceil(page.splatCount / samplingStride);
    }

    this.ensureCapacity(queueEstimate);

    let instanceCount = 0;

    for (const activeCluster of clusteredActiveClusters) {
      const cluster = asset.clusters[activeCluster.clusterId]!;
      const page = asset.pages[activeCluster.pageId]!;
      const preparedPage = canUsePreparedPages ? this.getPreparedPageCache(page) : null;
      const projectedCluster = tileBinning.clusterProjections.get(cluster.id) ?? null;
      const samplingStride = clusterSamplingStride.get(cluster.id) ?? 1;
      const densityCompensation = this.resolveDensityCompensation(samplingStride);
      const coverageScaleBoost = this.resolveCoverageScaleBoost(cluster, densityCompensation);

      if (preparedPage && samplingStride === 1) {
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

      if (preparedPage) {
        instanceCount += this.copyPreparedPageStrided(
          preparedPage,
          page.positions,
          instanceCount,
          samplingStride,
          coverageScaleBoost,
        );
        continue;
      }

      const resolvedEffects = this.owner.effectStack.evaluate(cluster.semanticMask, context.timeSeconds);
      this.effectTintScratch.copy(resolvedEffects.tint);

      for (let splatIndex = 0; splatIndex < page.splatCount; splatIndex += samplingStride) {
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
    if (this.sprite) {
      this.owner.remove(this.sprite);
    }

    this.positionAttribute = this.createAttribute(this.positionArray, 3);
    this.scaleAttribute = this.createAttribute(this.scaleArray, 3);
    this.rotationAttribute = this.createAttribute(this.rotationArray, 4);
    this.colorAttribute = this.createAttribute(this.colorArray, 3);
    this.opacityAttribute = this.createAttribute(this.opacityArray, 1);
    this.material = this.createMaterial(
      this.positionAttribute,
      this.scaleAttribute,
      this.rotationAttribute,
      this.colorAttribute,
      this.opacityAttribute,
    );
    this.sprite = new Sprite(this.material);
    this.sprite.count = 0;
    this.sprite.frustumCulled = false;
    this.sprite.renderOrder = 21;
    this.sprite.name = 'SparkSpriteQueue';
    this.owner.add(this.sprite);
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
    material.blending = AdditiveBlending;
    material.premultipliedAlpha = false;

    return material;
  }

  private createAttribute(array: Float32Array, itemSize: number): InstancedBufferAttribute {
    const attribute = new InstancedBufferAttribute(array, itemSize);
    attribute.setUsage(DynamicDrawUsage);
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
    if (!this.sprite) {
      return;
    }

    this.sprite.count = instanceCount;
    this.sprite.visible = instanceCount > 0;
  }

  private buildSnapshot(
    instanceCount: number,
    tileBinning: SplatClusterTileBinningSnapshot,
  ): SplatCompositorSnapshot {
    return {
      weightedInstances: instanceCount,
      heroInstances: 0,
      depthSlicedInstances: 0,
      activeTiles: tileBinning.activeTiles,
      weightedTiles: tileBinning.activeTiles,
      depthSlicedTiles: 0,
      heroTiles: 0,
      maxTileComplexity: tileBinning.maxTileSplatEstimate,
      visibleClusters: tileBinning.visibleClusterCount,
      binnedClusters: tileBinning.binnedClusterCount,
      binnedClusterReferences: tileBinning.binnedClusterReferences,
      overflowedTiles: tileBinning.overflowedTiles,
      overflowedClusterReferences: tileBinning.overflowedClusterReferences,
      maxClustersPerTile: tileBinning.maxTileClusterCount,
      maxTileSplatEstimate: tileBinning.maxTileSplatEstimate,
      tileBufferBytes: tileBinning.buffers.clusterScreenData.byteLength
        + tileBinning.buffers.tileHeaders.byteLength
        + tileBinning.buffers.tileEntries.byteLength,
    };
  }

  private resolveClusterSamplingStride(
    tileBinning: SplatClusterTileBinningSnapshot,
    budgets: SplatBudgetOptions,
  ): Map<number, number> {
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
        const nextStride = Math.max(
          clusterStride.get(clusterId) ?? 1,
          Math.min(8, Math.ceil(reductionFactor)),
        );
        clusterStride.set(clusterId, nextStride);
        reductionHash = Math.imul(reductionHash ^ clusterId, 16777619) >>> 0;
        reductionHash = Math.imul(reductionHash ^ nextStride, 16777619) >>> 0;
      }
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
    return Math.min(1.85, (1 + Math.log2(Math.max(1, representationGap)) * 0.05) * densityCompensation);
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

}

export { SplatSpriteCompositor as SplatTileCompositor };
