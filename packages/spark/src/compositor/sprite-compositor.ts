import {
  Color,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  NormalBlending,
  Sprite,
  Vector3,
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
import type { SplatBudgetOptions } from '../core/budgets';
import { SPLAT_SEMANTIC_FLAGS } from '../core/semantics';
import type { SplatMeshSelection } from '../scheduler/bootstrap-scheduler';
import { SplatTileClassifier } from './tile-classifier';
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
  private readonly tileClassifier = new SplatTileClassifier();
  private readonly depthSliceCount = 2;
  private readonly preparedPageCache = new Map<number, PreparedPageCache>();

  private weightedSprite?: Sprite;
  private heroSprite?: Sprite;
  private readonly depthSprites: Sprite[] = [];
  private weightedMaterial?: SpriteNodeMaterial;
  private heroMaterial?: SpriteNodeMaterial;
  private readonly depthMaterials: SpriteNodeMaterial[] = [];

  private weightedPositionAttribute?: InstancedBufferAttribute;
  private weightedScaleAttribute?: InstancedBufferAttribute;
  private weightedColorAttribute?: InstancedBufferAttribute;
  private weightedOpacityAttribute?: InstancedBufferAttribute;
  private heroPositionAttribute?: InstancedBufferAttribute;
  private heroScaleAttribute?: InstancedBufferAttribute;
  private heroColorAttribute?: InstancedBufferAttribute;
  private heroOpacityAttribute?: InstancedBufferAttribute;
  private readonly depthPositionAttributes: InstancedBufferAttribute[] = [];
  private readonly depthScaleAttributes: InstancedBufferAttribute[] = [];
  private readonly depthColorAttributes: InstancedBufferAttribute[] = [];
  private readonly depthOpacityAttributes: InstancedBufferAttribute[] = [];

  private weightedPositionArray = new Float32Array(0);
  private weightedScaleArray = new Float32Array(0);
  private weightedColorArray = new Float32Array(0);
  private weightedOpacityArray = new Float32Array(0);
  private heroPositionArray = new Float32Array(0);
  private heroScaleArray = new Float32Array(0);
  private heroColorArray = new Float32Array(0);
  private heroOpacityArray = new Float32Array(0);
  private depthPositionArrays: Float32Array[] = [];
  private depthScaleArrays: Float32Array[] = [];
  private depthColorArrays: Float32Array[] = [];
  private depthOpacityArrays: Float32Array[] = [];

  private readonly colorScratch = new Color();
  private readonly effectTintScratch = new Color();
  private readonly worldPositionScratch = new Vector3();
  private readonly cameraPositionScratch = new Vector3();
  private readonly clusterCenterScratch = new Vector3();
  private snapshot: SplatCompositorSnapshot = EMPTY_SNAPSHOT;
  private lastBuildSignature = '';

  constructor(private readonly owner: SplatMesh) {
    this.ensureCapacity(256, 64, 128);
  }

  getSnapshot(): SplatCompositorSnapshot {
    return this.snapshot;
  }

  sync(selection: SplatMeshSelection, context: SplatCompositorFrameContext): void {
    const asset = this.owner.getAsset();
    const canUsePreparedPages = this.canUsePreparedPages();
    const tileClassification = this.tileClassifier.classify(
      this.owner,
      selection.activeClusters,
      context.camera,
      context.viewportWidth,
      context.viewportHeight,
      context.budgets,
    );
    const frameBucket = Math.floor(context.frameIndex / Math.max(1, context.budgets.effectUpdateCadence));
    const clusterModeSignature = selection.activeClusters
      .map((cluster) => `${cluster.clusterId}:${tileClassification.clusterModes.get(cluster.clusterId) ?? 'weighted'}`)
      .join('|');
    // Do NOT include a camera-position bucket in the signature.  The fine-
    // grained quantisation (×4 position, ×16 direction) caused a full
    // per-splat CPU array rebuild on every sub-0.25m camera movement, making
    // the compositor O(active_splats) every frame regardless of page changes.
    const buildSignature = [
      selection.activePageIds.join(','),
      clusterModeSignature,
      this.owner.getMaterialVersion(),
      this.owner.getEffectVersion(),
      frameBucket,
    ].join('::');

    if (selection.activeClusters.length === 0) {
      this.lastBuildSignature = '';
      this.snapshot = {
        ...EMPTY_SNAPSHOT,
        activeTiles: tileClassification.activeTiles,
        weightedTiles: tileClassification.weightedTiles,
        depthSlicedTiles: tileClassification.depthSlicedTiles,
        heroTiles: tileClassification.heroTiles,
      };
      this.syncSprites(0, 0, [0, 0]);
      return;
    }

    if (buildSignature === this.lastBuildSignature) {
      this.snapshot = {
        ...this.snapshot,
        activeTiles: tileClassification.activeTiles,
        weightedTiles: tileClassification.weightedTiles,
        depthSlicedTiles: tileClassification.depthSlicedTiles,
        heroTiles: tileClassification.heroTiles,
        maxTileComplexity: tileClassification.maxTileComplexity,
      };
      return;
    }

    this.lastBuildSignature = buildSignature;

    let weightedQueueEstimate = 0;
    let heroQueueEstimate = 0;
    let depthQueueEstimate = 0;

    for (const activeCluster of selection.activeClusters) {
      const page = asset.pages[activeCluster.pageId]!;
      const mode = tileClassification.clusterModes.get(activeCluster.clusterId) ?? 'weighted';

      if (mode === 'hero') {
        heroQueueEstimate += page.splatCount;
      } else if (mode === 'depth-sliced') {
        depthQueueEstimate += page.splatCount;
      } else {
        weightedQueueEstimate += page.splatCount;
      }
    }

    this.ensureCapacity(weightedQueueEstimate, heroQueueEstimate, depthQueueEstimate);

    let weightedCount = 0;
    let heroCount = 0;
    const depthCounts = [0, 0];

    for (const activeCluster of selection.activeClusters) {
      const cluster = asset.clusters[activeCluster.clusterId]!;
      const page = asset.pages[activeCluster.pageId]!;
      const mode = tileClassification.clusterModes.get(cluster.id) ?? 'weighted';
      const preparedPage = canUsePreparedPages ? this.getPreparedPageCache(page) : null;
      const coverageScaleBoost = this.resolveCoverageScaleBoost(cluster);

      if (preparedPage && mode === 'hero') {
        this.copyPreparedPage(
          preparedPage.positions,
          preparedPage.baseScales,
          preparedPage.colors,
          preparedPage.opacities,
          this.heroPositionArray,
          this.heroScaleArray,
          this.heroColorArray,
          this.heroOpacityArray,
          heroCount,
          coverageScaleBoost * 1.1,
        );
        heroCount += page.splatCount;
        continue;
      }

      if (preparedPage && mode === 'weighted') {
        this.copyPreparedPage(
          preparedPage.positions,
          preparedPage.baseScales,
          preparedPage.colors,
          preparedPage.opacities,
          this.weightedPositionArray,
          this.weightedScaleArray,
          this.weightedColorArray,
          this.weightedOpacityArray,
          weightedCount,
          coverageScaleBoost,
        );
        weightedCount += page.splatCount;
        continue;
      }

      const resolvedEffects = preparedPage
        ? null
        : this.owner.effectStack.evaluate(cluster.semanticMask, context.timeSeconds);
      const clusterCameraDepth = mode === 'depth-sliced'
        ? this.getCameraDepth(
            cluster.center[0],
            cluster.center[1],
            cluster.center[2],
            context.camera,
            this.clusterCenterScratch,
          )
        : 0;

      if (resolvedEffects) {
        this.effectTintScratch.copy(resolvedEffects.tint);
      }

      for (let splatIndex = 0; splatIndex < page.splatCount; splatIndex += 1) {
        const sourceOffset = splatIndex * 3;
        const scaleOffset = splatIndex * 2;
        const isotropicBaseScale = preparedPage
          ? preparedPage.baseScales[scaleOffset + 0]!
          : this.resolveBaseSplatScale(page.scales, sourceOffset);
        const baseScaleX = isotropicBaseScale * coverageScaleBoost;
        const baseScaleY = isotropicBaseScale * coverageScaleBoost;

        if (preparedPage) {
          this.colorScratch.setRGB(
            preparedPage.colors[sourceOffset + 0]!,
            preparedPage.colors[sourceOffset + 1]!,
            preparedPage.colors[sourceOffset + 2]!,
          );
        } else {
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
        }

        const opacity = preparedPage
          ? preparedPage.opacities[splatIndex]!
          : Math.min(
              1,
              this.owner.splatMaterial.opacity * page.opacities[splatIndex]! * resolvedEffects!.opacityMultiplier,
            );
        const resolvedScaleX = preparedPage ? baseScaleX : baseScaleX * resolvedEffects!.pointSizeMultiplier;
        const resolvedScaleY = preparedPage ? baseScaleY : baseScaleY * resolvedEffects!.pointSizeMultiplier;

        if (mode === 'hero') {
          this.writeInstance(
            this.heroPositionArray,
            this.heroScaleArray,
            this.heroColorArray,
            this.heroOpacityArray,
            heroCount,
            page.positions[sourceOffset + 0]!,
            page.positions[sourceOffset + 1]!,
            page.positions[sourceOffset + 2]!,
            resolvedScaleX * 1.1,
            resolvedScaleY * 1.1,
            this.colorScratch,
            opacity,
          );
          heroCount += 1;
        } else {
          if (mode === 'depth-sliced') {
            const depthSliceIndex = this.resolveDepthSlice(
              page.positions[sourceOffset + 0]!,
              page.positions[sourceOffset + 1]!,
              page.positions[sourceOffset + 2]!,
              clusterCameraDepth,
              context.camera,
            );

            this.writeInstance(
              this.depthPositionArrays[depthSliceIndex]!,
              this.depthScaleArrays[depthSliceIndex]!,
              this.depthColorArrays[depthSliceIndex]!,
              this.depthOpacityArrays[depthSliceIndex]!,
              depthCounts[depthSliceIndex]!,
              page.positions[sourceOffset + 0]!,
              page.positions[sourceOffset + 1]!,
              page.positions[sourceOffset + 2]!,
              resolvedScaleX,
              resolvedScaleY,
              this.colorScratch,
              opacity,
            );
            depthCounts[depthSliceIndex] = depthCounts[depthSliceIndex]! + 1;
            continue;
          }

          this.writeInstance(
            this.weightedPositionArray,
            this.weightedScaleArray,
            this.weightedColorArray,
            this.weightedOpacityArray,
            weightedCount,
            page.positions[sourceOffset + 0]!,
            page.positions[sourceOffset + 1]!,
            page.positions[sourceOffset + 2]!,
            resolvedScaleX,
            resolvedScaleY,
            this.colorScratch,
            opacity,
          );
          weightedCount += 1;
        }
      }
    }

    this.flushAttributes(weightedCount, heroCount, depthCounts);
    this.syncSprites(weightedCount, heroCount, depthCounts);
    this.snapshot = {
      weightedInstances: weightedCount,
      heroInstances: heroCount,
      depthSlicedInstances: depthCounts[0]! + depthCounts[1]!,
      activeTiles: tileClassification.activeTiles,
      weightedTiles: tileClassification.weightedTiles,
      depthSlicedTiles: tileClassification.depthSlicedTiles,
      heroTiles: tileClassification.heroTiles,
      maxTileComplexity: tileClassification.maxTileComplexity,
    };
  }

  private ensureCapacity(weightedCapacity: number, heroCapacity: number, depthCapacity: number): void {
    const needsWeightedResize = this.weightedPositionArray.length < weightedCapacity * 3;
    const needsHeroResize = this.heroPositionArray.length < heroCapacity * 3;
    const needsDepthResize = this.depthPositionArrays.some((array) => array.length < depthCapacity * 3);

    if (!needsWeightedResize && !needsHeroResize && !needsDepthResize) {
      return;
    }

    const nextWeightedCapacity = this.resolveCapacity(
      Math.floor(this.weightedPositionArray.length / 3),
      weightedCapacity,
      256,
    );
    const nextHeroCapacity = this.resolveCapacity(
      Math.floor(this.heroPositionArray.length / 3),
      heroCapacity,
      64,
    );
    const nextDepthCapacity = this.resolveCapacity(
      Math.floor((this.depthPositionArrays[0]?.length ?? 0) / 3),
      depthCapacity,
      128,
    );

    this.weightedPositionArray = new Float32Array(nextWeightedCapacity * 3);
    this.weightedScaleArray = new Float32Array(nextWeightedCapacity * 2);
    this.weightedColorArray = new Float32Array(nextWeightedCapacity * 3);
    this.weightedOpacityArray = new Float32Array(nextWeightedCapacity);
    this.heroPositionArray = new Float32Array(nextHeroCapacity * 3);
    this.heroScaleArray = new Float32Array(nextHeroCapacity * 2);
    this.heroColorArray = new Float32Array(nextHeroCapacity * 3);
    this.heroOpacityArray = new Float32Array(nextHeroCapacity);
    this.depthPositionArrays = Array.from({ length: this.depthSliceCount }, () => new Float32Array(nextDepthCapacity * 3));
    this.depthScaleArrays = Array.from({ length: this.depthSliceCount }, () => new Float32Array(nextDepthCapacity * 2));
    this.depthColorArrays = Array.from({ length: this.depthSliceCount }, () => new Float32Array(nextDepthCapacity * 3));
    this.depthOpacityArrays = Array.from({ length: this.depthSliceCount }, () => new Float32Array(nextDepthCapacity));

    this.rebuildSprites();
  }

  private rebuildSprites(): void {
    if (this.weightedSprite) {
      this.owner.remove(this.weightedSprite);
    }

    if (this.heroSprite) {
      this.owner.remove(this.heroSprite);
    }

    this.depthSprites.forEach((sprite) => this.owner.remove(sprite));
    this.depthSprites.length = 0;
    this.depthMaterials.length = 0;
    this.depthPositionAttributes.length = 0;
    this.depthScaleAttributes.length = 0;
    this.depthColorAttributes.length = 0;
    this.depthOpacityAttributes.length = 0;

    this.weightedPositionAttribute = this.createAttribute(this.weightedPositionArray, 3);
    this.weightedScaleAttribute = this.createAttribute(this.weightedScaleArray, 2);
    this.weightedColorAttribute = this.createAttribute(this.weightedColorArray, 3);
    this.weightedOpacityAttribute = this.createAttribute(this.weightedOpacityArray, 1);
    this.heroPositionAttribute = this.createAttribute(this.heroPositionArray, 3);
    this.heroScaleAttribute = this.createAttribute(this.heroScaleArray, 2);
    this.heroColorAttribute = this.createAttribute(this.heroColorArray, 3);
    this.heroOpacityAttribute = this.createAttribute(this.heroOpacityArray, 1);

    this.weightedMaterial = this.createMaterial(
      this.weightedPositionAttribute,
      this.weightedScaleAttribute,
      this.weightedColorAttribute,
      this.weightedOpacityAttribute,
    );
    this.heroMaterial = this.createMaterial(
      this.heroPositionAttribute,
      this.heroScaleAttribute,
      this.heroColorAttribute,
      this.heroOpacityAttribute,
    );

    this.weightedSprite = new Sprite(this.weightedMaterial);
    this.weightedSprite.count = 0;
    this.weightedSprite.frustumCulled = false;
    this.weightedSprite.renderOrder = 21;
    this.weightedSprite.name = 'SparkWeightedSpriteQueue';

    for (let sliceIndex = 0; sliceIndex < this.depthSliceCount; sliceIndex += 1) {
      const positionAttribute = this.createAttribute(this.depthPositionArrays[sliceIndex]!, 3);
      const scaleAttribute = this.createAttribute(this.depthScaleArrays[sliceIndex]!, 2);
      const colorAttribute = this.createAttribute(this.depthColorArrays[sliceIndex]!, 3);
      const opacityAttribute = this.createAttribute(this.depthOpacityArrays[sliceIndex]!, 1);
      const material = this.createMaterial(positionAttribute, scaleAttribute, colorAttribute, opacityAttribute);
      const sprite = new Sprite(material);

      sprite.count = 0;
      sprite.frustumCulled = false;
      sprite.renderOrder = 19 + sliceIndex;
      sprite.name = `SparkDepthSliceQueue${sliceIndex}`;

      this.depthPositionAttributes.push(positionAttribute);
      this.depthScaleAttributes.push(scaleAttribute);
      this.depthColorAttributes.push(colorAttribute);
      this.depthOpacityAttributes.push(opacityAttribute);
      this.depthMaterials.push(material);
      this.depthSprites.push(sprite);
      this.owner.add(sprite);
    }

    this.heroSprite = new Sprite(this.heroMaterial);
    this.heroSprite.count = 0;
    this.heroSprite.frustumCulled = false;
    this.heroSprite.renderOrder = 23;
    this.heroSprite.name = 'SparkHeroSpriteQueue';

    this.owner.add(this.weightedSprite);
    this.owner.add(this.heroSprite);
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
    const feather = smoothstep(1.0, 0.22, radial);
    const gaussian = exp(radial.mul(-6.5));
    const colorNode = instancedBufferAttribute(colorAttribute);
    const opacityNode = instancedBufferAttribute(opacityAttribute);
    const alphaNode = opacityNode.mul(gaussian).mul(feather);

    material.positionNode = instancedBufferAttribute(positionAttribute);
    material.scaleNode = instancedBufferAttribute(scaleAttribute);
    material.colorNode = colorNode.mul(alphaNode);
    material.opacityNode = alphaNode;
    material.maskNode = radial.lessThan(1.0);
    material.transparent = true;
    material.depthWrite = false;
    material.sizeAttenuation = true;

    material.blending = NormalBlending;
    material.premultipliedAlpha = true;

    return material;
  }

  private createAttribute(array: Float32Array, itemSize: number): InstancedBufferAttribute {
    const attribute = new InstancedBufferAttribute(array, itemSize);
    attribute.setUsage(DynamicDrawUsage);
    return attribute;
  }

  private flushAttributes(weightedCount: number, heroCount: number, depthCounts: readonly number[]): void {
    this.updateAttribute(this.weightedPositionAttribute, weightedCount, 3);
    this.updateAttribute(this.weightedScaleAttribute, weightedCount, 2);
    this.updateAttribute(this.weightedColorAttribute, weightedCount, 3);
    this.updateAttribute(this.weightedOpacityAttribute, weightedCount, 1);
    this.updateAttribute(this.heroPositionAttribute, heroCount, 3);
    this.updateAttribute(this.heroScaleAttribute, heroCount, 2);
    this.updateAttribute(this.heroColorAttribute, heroCount, 3);
    this.updateAttribute(this.heroOpacityAttribute, heroCount, 1);

    for (let sliceIndex = 0; sliceIndex < this.depthSliceCount; sliceIndex += 1) {
      this.updateAttribute(this.depthPositionAttributes[sliceIndex], depthCounts[sliceIndex] ?? 0, 3);
      this.updateAttribute(this.depthScaleAttributes[sliceIndex], depthCounts[sliceIndex] ?? 0, 2);
      this.updateAttribute(this.depthColorAttributes[sliceIndex], depthCounts[sliceIndex] ?? 0, 3);
      this.updateAttribute(this.depthOpacityAttributes[sliceIndex], depthCounts[sliceIndex] ?? 0, 1);
    }
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

  private syncSprites(weightedCount: number, heroCount: number, depthCounts: readonly number[]): void {
    if (this.weightedSprite) {
      this.weightedSprite.count = weightedCount;
      this.weightedSprite.visible = weightedCount > 0;
    }

    for (let sliceIndex = 0; sliceIndex < this.depthSliceCount; sliceIndex += 1) {
      const sprite = this.depthSprites[sliceIndex];
      const count = depthCounts[sliceIndex] ?? 0;

      if (sprite) {
        sprite.count = count;
        sprite.visible = count > 0;
      }
    }

    if (this.heroSprite) {
      this.heroSprite.count = heroCount;
      this.heroSprite.visible = heroCount > 0;
    }
  }

  private writeInstance(
    positionArray: Float32Array,
    scaleArray: Float32Array,
    colorArray: Float32Array,
    opacityArray: Float32Array,
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
    positionArray[positionOffset + 0] = x;
    positionArray[positionOffset + 1] = y;
    positionArray[positionOffset + 2] = z;
    scaleArray[scaleOffset + 0] = scaleX;
    scaleArray[scaleOffset + 1] = scaleY;
    colorArray[positionOffset + 0] = color.r;
    colorArray[positionOffset + 1] = color.g;
    colorArray[positionOffset + 2] = color.b;
    opacityArray[index] = opacity;
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
    return Math.min(1.85, 1 + Math.log2(Math.max(1, representationGap)) * 0.14);
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
    return Math.max(0.014, isotropicScale * pointSize * 0.78);
  }

  private resolveDepthSlice(
    x: number,
    y: number,
    z: number,
    clusterCameraDepth: number,
    camera: Camera,
  ): number {
    const splatCameraDepth = this.getCameraDepth(x, y, z, camera, this.worldPositionScratch);
    return splatCameraDepth < clusterCameraDepth ? 0 : 1;
  }

  private getCameraDepth(
    x: number,
    y: number,
    z: number,
    camera: Camera,
    target: Vector3,
  ): number {
    target.set(x, y, z).applyMatrix4(this.owner.matrixWorld).applyMatrix4(camera.matrixWorldInverse);
    this.cameraPositionScratch.copy(target);
    return this.cameraPositionScratch.z;
  }
}
