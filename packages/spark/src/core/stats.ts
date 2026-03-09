import type { SplatBudgetOptions } from './budgets';

export type SplatSchedulerMode = 'cpu-bootstrap' | 'gpu-readback';

export interface SplatMeshFrameStats {
  meshUuid: string;
  meshName: string;
  frontierClusters: number;
  activePages: number;
  residentPages: number;
  requestedPages: number;
  visibleSplats: number;
  estimatedOverdraw: number;
  frontierStability: number;
  weightedInstances: number;
  heroInstances: number;
  depthSlicedInstances: number;
  activeTiles: number;
  weightedTiles: number;
  depthSlicedTiles: number;
  heroTiles: number;
  maxTileComplexity: number;
}

export interface SplatFrameStatsSnapshot {
  frameIndex: number;
  elapsedSeconds: number;
  appliedGovernorLevel: number;
  governorReason: string;
  schedulerMode: SplatSchedulerMode;
  budgets: SplatBudgetOptions;
  meshCount: number;
  dirtyObjects: number;
  sceneDescriptorBytes: number;
  clusterMetadataBytes: number;
  pageDescriptorBytes: number;
  residencyBytes: number;
  gpuVisibilityReady: boolean;
  gpuVisibilityPending: boolean;
  gpuVisibilityFrameLag: number;
  gpuClusterCount: number;
  frontierClusters: number;
  visibleSplats: number;
  activePages: number;
  residentPages: number;
  requestedPages: number;
  pageUploads: number;
  pageFaultRate: number;
  estimatedOverdraw: number;
  frontierStability: number;
  compositorWeightedInstances: number;
  compositorHeroInstances: number;
  compositorDepthSlicedInstances: number;
  compositorActiveTiles: number;
  compositorWeightedTiles: number;
  compositorDepthSlicedTiles: number;
  compositorHeroTiles: number;
  compositorMaxTileComplexity: number;
  cpuFrameMs: number;
  sceneBuildMs: number;
  schedulerMs: number;
  meshStats: readonly SplatMeshFrameStats[];
}

export const EMPTY_SPLAT_FRAME_STATS: SplatFrameStatsSnapshot = {
  frameIndex: 0,
  elapsedSeconds: 0,
  appliedGovernorLevel: 0,
  governorReason: 'bootstrap',
  schedulerMode: 'cpu-bootstrap',
  budgets: {
    maxVisibleSplats: 0,
    maxOverdrawBudget: 0,
    maxActivePages: 0,
    maxResidentPages: 0,
    maxDeformablePages: 0,
    maxPageUploadsPerFrame: 0,
    minProjectedNodeSizePx: 0,
    peripheralFoveation: 0,
    heroTileBudget: 0,
    effectUpdateCadence: 0,
    deformationBudget: 0,
    renderScale: 1,
    temporalStabilityBias: 0,
  },
  meshCount: 0,
  dirtyObjects: 0,
  sceneDescriptorBytes: 0,
  clusterMetadataBytes: 0,
  pageDescriptorBytes: 0,
  residencyBytes: 0,
  gpuVisibilityReady: false,
  gpuVisibilityPending: false,
  gpuVisibilityFrameLag: 0,
  gpuClusterCount: 0,
  frontierClusters: 0,
  visibleSplats: 0,
  activePages: 0,
  residentPages: 0,
  requestedPages: 0,
  pageUploads: 0,
  pageFaultRate: 0,
  estimatedOverdraw: 0,
  frontierStability: 1,
  compositorWeightedInstances: 0,
  compositorHeroInstances: 0,
  compositorDepthSlicedInstances: 0,
  compositorActiveTiles: 0,
  compositorWeightedTiles: 0,
  compositorDepthSlicedTiles: 0,
  compositorHeroTiles: 0,
  compositorMaxTileComplexity: 0,
  cpuFrameMs: 0,
  sceneBuildMs: 0,
  schedulerMs: 0,
  meshStats: [],
};

export class SplatStats {
  private snapshot: SplatFrameStatsSnapshot = EMPTY_SPLAT_FRAME_STATS;

  readonly cpuFrameHistory: number[] = [];
  readonly pageFaultHistory: number[] = [];

  commit(snapshot: SplatFrameStatsSnapshot): SplatFrameStatsSnapshot {
    this.snapshot = snapshot;
    this.cpuFrameHistory.push(snapshot.cpuFrameMs);
    this.pageFaultHistory.push(snapshot.pageFaultRate);

    if (this.cpuFrameHistory.length > 120) {
      this.cpuFrameHistory.shift();
    }

    if (this.pageFaultHistory.length > 120) {
      this.pageFaultHistory.shift();
    }

    return this.snapshot;
  }

  getSnapshot(): SplatFrameStatsSnapshot {
    return this.snapshot;
  }
}
