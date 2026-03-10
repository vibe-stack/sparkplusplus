import { Object3D } from 'three';
import { SplatPageTable } from '../assets/page-table';
import type { SplatAsset } from '../assets/model';
import type { SplatCompositorFrameContext, SplatCompositorSnapshot } from '../compositor/sprite-compositor';
import { SplatTileCompositor } from '../compositor/sprite-compositor';
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
  private readonly compositor: SplatTileCompositor;

  constructor(options: SplatMeshOptions) {
    super();
    this.source = options.source;
    this.splatMaterial = options.material ?? new SplatMaterial();
    this.effectStack = options.effects ?? new SplatEffectStack();
    this.importance = options.importance ?? 1;
    this.compositor = new SplatTileCompositor(this);
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

  getCompositorSnapshot(): SplatCompositorSnapshot {
    return this.compositor.getSnapshot();
  }

  applySelection(selection: SplatMeshSelection, context: SplatCompositorFrameContext): void {
    this.selection = selection;
    this.compositor.sync(selection, context);
  }
}
