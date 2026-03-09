import {
  BufferAttribute,
  BufferGeometry,
  Color,
  DynamicDrawUsage,
  NormalBlending,
  Object3D,
  Points,
  PointsMaterial,
} from 'three';
import { SplatPageTable } from '../assets/page-table';
import { SPLAT_SEMANTIC_FLAGS } from '../core/semantics';
import type { SplatAsset } from '../assets/model';
import { SplatEffectStack } from './effects';
import { SplatMaterial } from './material';
import type { SplatSource } from './source';
import type { SplatMeshSelection } from '../scheduler/bootstrap-scheduler';

export interface SplatMeshOptions {
  source: SplatSource;
  material?: SplatMaterial;
  effects?: SplatEffectStack;
  importance?: number;
}

const DEFAULT_SELECTION: SplatMeshSelection = {
  frontierClusterIds: [],
  activeClusters: [],
  activePageIds: [],
  requestedPageIds: [],
  visibleSplats: 0,
  estimatedOverdraw: 0,
  frontierStability: 1,
};

export class SplatMesh extends Object3D {
  readonly isSplatMesh = true;

  source: SplatSource;
  splatMaterial: SplatMaterial;
  effectStack: SplatEffectStack;
  importance: number;

  private asset?: SplatAsset;
  private pageTable?: SplatPageTable;
  private selection: SplatMeshSelection = DEFAULT_SELECTION;
  private readonly debugGeometry = new BufferGeometry();
  private readonly debugMaterial = new PointsMaterial({
    size: 6,
    vertexColors: true,
    transparent: true,
    opacity: 0.92,
    sizeAttenuation: true,
    depthWrite: false,
    blending: NormalBlending,
    toneMapped: false,
  });
  private readonly debugProxy = new Points(this.debugGeometry, this.debugMaterial);
  private positionBuffer = new Float32Array(0);
  private colorBuffer = new Float32Array(0);
  private colorScratch = new Color();
  private effectTintScratch = new Color();

  constructor(options: SplatMeshOptions) {
    super();
    this.source = options.source;
    this.splatMaterial = options.material ?? new SplatMaterial();
    this.effectStack = options.effects ?? new SplatEffectStack();
    this.importance = options.importance ?? 1;

    this.debugProxy.name = 'SparkDebugProxy';
    this.debugProxy.frustumCulled = false;
    this.add(this.debugProxy);
    this.updateDebugMaterial();
  }

  getAsset(): SplatAsset {
    if (!this.asset) {
      this.asset = this.source.buildAsset();
      this.pageTable = new SplatPageTable(this.asset);

      if (!this.name) {
        this.name = this.asset.label;
      }
    }

    return this.asset;
  }

  getPageTable(): SplatPageTable {
    this.getAsset();
    return this.pageTable!;
  }

  getSelection(): SplatMeshSelection {
    return this.selection;
  }

  getPreviousFrontierClusterIds(): number[] {
    return this.selection.frontierClusterIds;
  }

  getMaterialVersion(): number {
    return this.splatMaterial.version;
  }

  getEffectVersion(): number {
    return this.effectStack.version;
  }

  applySelection(selection: SplatMeshSelection, timeSeconds: number): void {
    this.selection = selection;
    this.syncDebugProxy(timeSeconds);
  }

  private syncDebugProxy(timeSeconds: number): void {
    const asset = this.getAsset();
    const activeClusters = this.selection.activeClusters;
    const totalSplats = activeClusters.reduce(
      (sum, activeCluster) => sum + asset.pages[activeCluster.pageId]!.splatCount,
      0,
    );

    if (totalSplats === 0) {
      this.debugProxy.visible = false;
      this.debugGeometry.setDrawRange(0, 0);
      return;
    }

    this.debugProxy.visible = true;
    this.ensureCapacity(totalSplats);

    let writeOffset = 0;
    let pointSizeMultiplierSum = 0;
    let opacityMultiplierSum = 0;

    for (const activeCluster of activeClusters) {
      const cluster = asset.clusters[activeCluster.clusterId]!;
      const page = asset.pages[activeCluster.pageId]!;
      const resolvedEffects = this.effectStack.evaluate(cluster.semanticMask, timeSeconds);
      pointSizeMultiplierSum += resolvedEffects.pointSizeMultiplier;
      opacityMultiplierSum += resolvedEffects.opacityMultiplier;
      this.effectTintScratch.copy(resolvedEffects.tint);

      for (let i = 0; i < page.splatCount; i += 1) {
        const sourceOffset = i * 3;
        const targetOffset = writeOffset * 3;

        this.positionBuffer[targetOffset + 0] = page.positions[sourceOffset + 0]!;
        this.positionBuffer[targetOffset + 1] = page.positions[sourceOffset + 1]!;
        this.positionBuffer[targetOffset + 2] = page.positions[sourceOffset + 2]!;

        this.colorScratch.setRGB(
          page.colors[sourceOffset + 0]!,
          page.colors[sourceOffset + 1]!,
          page.colors[sourceOffset + 2]!,
        );

        if (this.splatMaterial.debugMode === 'lod') {
          this.colorScratch.setHSL(Math.min(0.85, cluster.level * 0.12), 0.75, 0.55);
        }

        if (this.splatMaterial.debugMode === 'semantic') {
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
          .multiply(this.splatMaterial.tint)
          .multiply(this.effectTintScratch)
          .multiplyScalar(this.splatMaterial.colorGain * page.opacities[i]! * resolvedEffects.opacityMultiplier);

        this.colorBuffer[targetOffset + 0] = this.colorScratch.r;
        this.colorBuffer[targetOffset + 1] = this.colorScratch.g;
        this.colorBuffer[targetOffset + 2] = this.colorScratch.b;
        writeOffset += 1;
      }
    }

    const positionAttribute = this.debugGeometry.getAttribute('position') as BufferAttribute;
    const colorAttribute = this.debugGeometry.getAttribute('color') as BufferAttribute;
    positionAttribute.needsUpdate = true;
    colorAttribute.needsUpdate = true;
    this.debugGeometry.setDrawRange(0, writeOffset);

    const averagePointMultiplier = pointSizeMultiplierSum / activeClusters.length;
    const averageOpacityMultiplier = opacityMultiplierSum / activeClusters.length;
    this.debugMaterial.size = this.splatMaterial.pointSize * averagePointMultiplier;
    this.debugMaterial.opacity = Math.min(1, this.splatMaterial.opacity * averageOpacityMultiplier);
  }

  private ensureCapacity(totalSplats: number): void {
    if (this.positionBuffer.length >= totalSplats * 3) {
      return;
    }

    const capacity = Math.max(totalSplats, 256);
    this.positionBuffer = new Float32Array(capacity * 3);
    this.colorBuffer = new Float32Array(capacity * 3);
    const positionAttribute = new BufferAttribute(this.positionBuffer, 3);
    const colorAttribute = new BufferAttribute(this.colorBuffer, 3);
    positionAttribute.setUsage(DynamicDrawUsage);
    colorAttribute.setUsage(DynamicDrawUsage);
    this.debugGeometry.setAttribute('position', positionAttribute);
    this.debugGeometry.setAttribute('color', colorAttribute);
  }

  private updateDebugMaterial(): void {
    this.debugMaterial.size = this.splatMaterial.pointSize;
    this.debugMaterial.opacity = this.splatMaterial.opacity;
    this.debugMaterial.color.copy(this.splatMaterial.tint);
  }
}
