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

export class SplatSpriteCompositor {
  private readonly tileClassifier = new SplatTileClassifier();
  private readonly depthSliceCount = 2;

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
    const hasDepthSlicedClusters = selection.activeClusters.some(
      (cluster) => tileClassification.clusterModes.get(cluster.clusterId) === 'depth-sliced',
    );
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
      const resolvedEffects = this.owner.effectStack.evaluate(cluster.semanticMask, context.timeSeconds);
      const clusterCameraDepth = this.getCameraDepth(
        cluster.center[0],
        cluster.center[1],
        cluster.center[2],
        context.camera,
        this.clusterCenterScratch,
      );
      this.effectTintScratch.copy(resolvedEffects.tint);

      for (let splatIndex = 0; splatIndex < page.splatCount; splatIndex += 1) {
        const sourceOffset = splatIndex * 3;
        const baseScaleX = Math.max(0.015, page.scales[sourceOffset + 0]! * this.owner.splatMaterial.pointSize);
        const baseScaleY = Math.max(0.015, page.scales[sourceOffset + 1]! * this.owner.splatMaterial.pointSize);

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

        const opacity = Math.min(
          1,
          this.owner.splatMaterial.opacity * page.opacities[splatIndex]! * resolvedEffects.opacityMultiplier,
        );

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
            baseScaleX * resolvedEffects.pointSizeMultiplier * 1.1,
            baseScaleY * resolvedEffects.pointSizeMultiplier * 1.1,
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
              baseScaleX * resolvedEffects.pointSizeMultiplier,
              baseScaleY * resolvedEffects.pointSizeMultiplier,
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
            baseScaleX * resolvedEffects.pointSizeMultiplier,
            baseScaleY * resolvedEffects.pointSizeMultiplier,
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

    const nextWeightedCapacity = Math.max(weightedCapacity, 256);
    const nextHeroCapacity = Math.max(heroCapacity, 64);
    const nextDepthCapacity = Math.max(depthCapacity, 128);

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
    const feather = smoothstep(1.35, 0.0, radial);
    const gaussian = exp(radial.mul(-3.75));
    const colorNode = instancedBufferAttribute(colorAttribute);
    const opacityNode = instancedBufferAttribute(opacityAttribute);
    const alphaNode = opacityNode.mul(gaussian).mul(feather);

    material.positionNode = instancedBufferAttribute(positionAttribute);
    material.scaleNode = instancedBufferAttribute(scaleAttribute);
    material.colorNode = colorNode.mul(gaussian).mul(feather);
    material.opacityNode = alphaNode;
    material.maskNode = radial.lessThan(1.5);
    material.transparent = true;
    material.depthWrite = false;
    material.sizeAttenuation = true;

    material.blending = NormalBlending;
    material.premultipliedAlpha = false;

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

  private getCameraBucket(camera: Camera): string {
    const position = camera.getWorldPosition(this.worldPositionScratch);
    const forward = camera.getWorldDirection(this.cameraPositionScratch);

    return [
      Math.round(position.x * 4),
      Math.round(position.y * 4),
      Math.round(position.z * 4),
      Math.round(forward.x * 16),
      Math.round(forward.y * 16),
      Math.round(forward.z * 16),
    ].join(',');
  }
}
