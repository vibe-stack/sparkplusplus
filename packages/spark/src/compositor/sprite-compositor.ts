import {
  Color,
  DoubleSide,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  NormalBlending,
  Sprite,
  Vector3,
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
  baseScales: Float32Array;
  rotations: Float32Array;
  colors: Float32Array;
  opacities: Float32Array;
}

export class SplatSpriteCompositor {
  private readonly preparedPageCache = new Map<number, PreparedPageCache>();

  private sprite?: Sprite;
  private material?: SpriteNodeMaterial;
  private positionAttribute?: InstancedBufferAttribute;
  private scaleAttribute?: InstancedBufferAttribute;
  private rotationAttribute?: InstancedBufferAttribute;
  private colorAttribute?: InstancedBufferAttribute;
  private opacityAttribute?: InstancedBufferAttribute;

  private positionArray = new Float32Array(0);
  private scaleArray = new Float32Array(0);
  private rotationArray = new Float32Array(0);
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

      const resolvedEffects = this.owner.effectStack.evaluate(cluster.semanticMask, context.timeSeconds);
      this.effectTintScratch.copy(resolvedEffects.tint);

      for (let splatIndex = 0; splatIndex < page.splatCount; splatIndex += 1) {
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
    return Math.min(1.35, 1 + Math.log2(Math.max(1, representationGap)) * 0.05);
  }

  private resolveBaseScale(
    scale: number,
    multiplier = 1,
  ): number {
    return Math.max(0.012, scale * this.owner.splatMaterial.pointSize * 0.64 * multiplier);
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
