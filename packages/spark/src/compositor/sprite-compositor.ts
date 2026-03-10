import {
  Color,
  DoubleSide,
  DynamicDrawUsage,
  Vector3,
  InstancedBufferAttribute,
  NormalBlending,
  Sprite,
} from 'three';
import { SpriteNodeMaterial } from 'three/webgpu';
import {
  exp,
  instancedBufferAttribute,
  lengthSq,
  smoothstep,
  uv,
  vec2,
} from 'three/tsl';
import type { Camera } from 'three';
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
};

interface PreparedPageCache {
  materialVersion: number;
  positions: Float32Array;
  baseScales: Float32Array;
  colors: Float32Array;
  opacities: Float32Array;
}

export class SplatSpriteCompositor {
  private readonly preparedPageCache = new Map<number, PreparedPageCache>();

  private sprite?: Sprite;
  private material?: SpriteNodeMaterial;
  private positionAttribute?: InstancedBufferAttribute;
  private scaleAttribute?: InstancedBufferAttribute;
  private colorAttribute?: InstancedBufferAttribute;
  private opacityAttribute?: InstancedBufferAttribute;

  private positionArray = new Float32Array(0);
  private scaleArray = new Float32Array(0);
  private colorArray = new Float32Array(0);
  private opacityArray = new Float32Array(0);

  private readonly colorScratch = new Color();
  private readonly effectTintScratch = new Color();
  private readonly clusterPositionScratch = new Vector3();
  private snapshot: SplatCompositorSnapshot = EMPTY_SNAPSHOT;
  private lastBuildSignature = '';

  constructor(private readonly owner: SplatMesh) {
    this.ensureCapacity(256);
  }

  getSnapshot(): SplatCompositorSnapshot {
    return this.snapshot;
  }

  sync(selection: SplatMeshSelection, context: SplatCompositorFrameContext): void {
    const asset = this.owner.getAsset();
    const canUsePreparedPages = this.canUsePreparedPages();
    const sortedActiveClusters = [...selection.activeClusters].sort((left, right) =>
      this.getClusterCameraDepth(asset.clusters[left.clusterId]!, context.camera)
      - this.getClusterCameraDepth(asset.clusters[right.clusterId]!, context.camera),
    );
    const frameBucket = this.owner.effectStack.hasTemporalEffects()
      ? Math.floor(context.frameIndex / Math.max(1, context.budgets.effectUpdateCadence))
      : 0;
    const buildSignature = [
      sortedActiveClusters.map((cluster) => cluster.clusterId).join(','),
      this.owner.getMaterialVersion(),
      this.owner.getEffectVersion(),
      frameBucket,
    ].join('::');

    if (sortedActiveClusters.length === 0) {
      this.lastBuildSignature = '';
      this.snapshot = EMPTY_SNAPSHOT;
      this.syncSprite(0);
      return;
    }

    if (buildSignature === this.lastBuildSignature) {
      return;
    }

    this.lastBuildSignature = buildSignature;

    let queueEstimate = 0;

    for (const activeCluster of sortedActiveClusters) {
      queueEstimate += asset.pages[activeCluster.pageId]!.splatCount;
    }

    this.ensureCapacity(queueEstimate);

    let instanceCount = 0;

    for (const activeCluster of sortedActiveClusters) {
      const cluster = asset.clusters[activeCluster.clusterId]!;
      const page = asset.pages[activeCluster.pageId]!;
      const preparedPage = canUsePreparedPages ? this.getPreparedPageCache(page) : null;
      const coverageScaleBoost = this.resolveCoverageScaleBoost(cluster);

      if (preparedPage) {
        this.copyPreparedPage(
          preparedPage.positions,
          preparedPage.baseScales,
          preparedPage.colors,
          preparedPage.opacities,
          this.positionArray,
          this.scaleArray,
          this.colorArray,
          this.opacityArray,
          instanceCount,
          coverageScaleBoost,
        );
        instanceCount += page.splatCount;
        continue;
      }

      const resolvedEffects = this.owner.effectStack.evaluate(cluster.semanticMask, context.timeSeconds);
      this.effectTintScratch.copy(resolvedEffects.tint);

      for (let splatIndex = 0; splatIndex < page.splatCount; splatIndex += 1) {
        const sourceOffset = splatIndex * 3;
        const baseScale = this.resolveBaseSplatScale(page.scales, sourceOffset) * coverageScaleBoost;

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

        this.colorScratch
          .multiply(this.owner.splatMaterial.tint)
          .multiply(this.effectTintScratch)
          .multiplyScalar(this.owner.splatMaterial.colorGain);

        this.writeInstance(
          instanceCount,
          page.positions[sourceOffset + 0]!,
          page.positions[sourceOffset + 1]!,
          page.positions[sourceOffset + 2]!,
          baseScale * resolvedEffects.pointSizeMultiplier,
          baseScale * resolvedEffects.pointSizeMultiplier,
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
    this.snapshot = {
      weightedInstances: instanceCount,
      heroInstances: 0,
      depthSlicedInstances: 0,
      activeTiles: 0,
      weightedTiles: 0,
      depthSlicedTiles: 0,
      heroTiles: 0,
      maxTileComplexity: 0,
    };
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
    this.scaleArray = new Float32Array(nextCapacity * 2);
    this.colorArray = new Float32Array(nextCapacity * 3);
    this.opacityArray = new Float32Array(nextCapacity);
    this.rebuildSprite();
  }

  private rebuildSprite(): void {
    if (this.sprite) {
      this.owner.remove(this.sprite);
    }

    this.positionAttribute = this.createAttribute(this.positionArray, 3);
    this.scaleAttribute = this.createAttribute(this.scaleArray, 2);
    this.colorAttribute = this.createAttribute(this.colorArray, 3);
    this.opacityAttribute = this.createAttribute(this.opacityArray, 1);
    this.material = this.createMaterial(
      this.positionAttribute,
      this.scaleAttribute,
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

    material.positionNode = instancedBufferAttribute(positionAttribute);
    material.scaleNode = instancedBufferAttribute(scaleAttribute);
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

  private flushAttributes(instanceCount: number): void {
    this.updateAttribute(this.positionAttribute, instanceCount, 3);
    this.updateAttribute(this.scaleAttribute, instanceCount, 2);
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

  private writeInstance(
    index: number,
    x: number,
    y: number,
    z: number,
    scaleX: number,
    scaleY: number,
    color: Color,
    opacity: number,
  ): void {
    const positionOffset = index * 3;
    const scaleOffset = index * 2;
    this.positionArray[positionOffset + 0] = x;
    this.positionArray[positionOffset + 1] = y;
    this.positionArray[positionOffset + 2] = z;
    this.scaleArray[scaleOffset + 0] = scaleX;
    this.scaleArray[scaleOffset + 1] = scaleY;
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

    const baseScales = new Float32Array(page.splatCount * 2);
    const colors = new Float32Array(page.splatCount * 3);
    const opacities = new Float32Array(page.splatCount);
    const tint = this.owner.splatMaterial.tint;
    const colorGain = this.owner.splatMaterial.colorGain;
    const pointSize = this.owner.splatMaterial.pointSize;
    const opacity = this.owner.splatMaterial.opacity;

    for (let splatIndex = 0; splatIndex < page.splatCount; splatIndex += 1) {
      const sourceOffset = splatIndex * 3;
      const scaleOffset = splatIndex * 2;
      const scale = this.resolveBaseSplatScale(page.scales, sourceOffset, pointSize);

      baseScales[scaleOffset + 0] = scale;
      baseScales[scaleOffset + 1] = scale;
      colors[sourceOffset + 0] = page.colors[sourceOffset + 0]! * tint.r * colorGain;
      colors[sourceOffset + 1] = page.colors[sourceOffset + 1]! * tint.g * colorGain;
      colors[sourceOffset + 2] = page.colors[sourceOffset + 2]! * tint.b * colorGain;
      opacities[splatIndex] = Math.min(1, opacity * page.opacities[splatIndex]!);
    }

    const preparedPage: PreparedPageCache = {
      materialVersion,
      positions: page.positions,
      baseScales,
      colors,
      opacities,
    };

    this.preparedPageCache.set(page.id, preparedPage);
    return preparedPage;
  }

  private copyPreparedPage(
    sourcePositions: Float32Array,
    sourceScales: Float32Array,
    sourceColors: Float32Array,
    sourceOpacities: Float32Array,
    targetPositions: Float32Array,
    targetScales: Float32Array,
    targetColors: Float32Array,
    targetOpacities: Float32Array,
    targetInstanceOffset: number,
    scaleMultiplier: number,
  ): void {
    targetPositions.set(sourcePositions, targetInstanceOffset * 3);
    targetColors.set(sourceColors, targetInstanceOffset * 3);
    targetOpacities.set(sourceOpacities, targetInstanceOffset);

    if (Math.abs(scaleMultiplier - 1) <= 1e-4) {
      targetScales.set(sourceScales, targetInstanceOffset * 2);
      return;
    }

    const targetScaleOffset = targetInstanceOffset * 2;

    for (let index = 0; index < sourceScales.length; index += 1) {
      targetScales[targetScaleOffset + index] = sourceScales[index]! * scaleMultiplier;
    }
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
  ): number {
    const representationGap = cluster.representedSplatCount / Math.max(1, cluster.splatCount);
    return Math.min(1.7, 1 + Math.log2(Math.max(1, representationGap)) * 0.12);
  }

  private resolveBaseSplatScale(
    scales: Float32Array,
    sourceOffset: number,
    pointSize = this.owner.splatMaterial.pointSize,
  ): number {
    const isotropicScale = Math.max(
      scales[sourceOffset + 0]!,
      scales[sourceOffset + 1]!,
      scales[sourceOffset + 2]!,
    );
    return Math.max(0.012, isotropicScale * pointSize * 0.64);
  }

  private getClusterCameraDepth(
    cluster: ReturnType<SplatMesh['getAsset']>['clusters'][number],
    camera: Camera,
  ): number {
    return this.clusterPositionScratch
      .set(cluster.center[0], cluster.center[1], cluster.center[2])
      .applyMatrix4(this.owner.matrixWorld)
      .applyMatrix4(camera.matrixWorldInverse)
      .z;
  }
}

export { SplatSpriteCompositor as SplatTileCompositor };
