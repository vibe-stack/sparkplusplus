import { Camera, Object3D, Scene } from 'three';
import { cloneBudgets, type SplatBudgetOptions } from '../core/budgets';
import { SplatQualityGovernor } from '../core/governor';
import { SplatSceneDescriptorBuilder } from '../core/scene-descriptor';
import { SplatStats, type SplatFrameStatsSnapshot, type SplatMeshFrameStats } from '../core/stats';
import { SplatGpuVisibilityPipeline, type SplatGpuRendererLike } from '../gpu/visibility';
import { BootstrapVisibilityScheduler } from '../scheduler/bootstrap-scheduler';

export interface SplatRendererBridgeOptions {
  budgets?: Partial<SplatBudgetOptions>;
  governor?: SplatQualityGovernor;
  useGpuVisibility?: boolean;
}

export interface SplatRendererLike extends SplatGpuRendererLike {
  domElement?: {
    clientWidth?: number;
    clientHeight?: number;
    width?: number;
    height?: number;
  };
}

export class SplatRendererBridge extends Object3D {
  readonly isSplatRendererBridge = true;

  readonly sceneDescriptorBuilder = new SplatSceneDescriptorBuilder();
  readonly scheduler = new BootstrapVisibilityScheduler();
  readonly stats = new SplatStats();
  readonly governor: SplatQualityGovernor;
  readonly gpuVisibility = new SplatGpuVisibilityPipeline();
  readonly useGpuVisibility: boolean;

  private frameIndex = 0;
  private elapsedSeconds = 0;

  constructor(options: SplatRendererBridgeOptions = {}) {
    super();
    this.governor = options.governor ?? new SplatQualityGovernor(options.budgets);
    this.useGpuVisibility = options.useGpuVisibility ?? true;
  }

  update(
    scene: Scene,
    camera: Camera,
    deltaSeconds: number,
    renderer?: SplatRendererLike,
  ): SplatFrameStatsSnapshot {
    const frameStart = performance.now();
    const appliedBudgets = this.governor.getBudgets();
    const appliedGovernorLevel = this.governor.getLevel();
    const governorReason = this.governor.getReason();
    this.elapsedSeconds += deltaSeconds;

    scene.updateMatrixWorld();
    camera.updateMatrixWorld();

    const sceneBuildStart = performance.now();
    const sceneDescriptors = this.sceneDescriptorBuilder.build(scene);
    const sceneBuildMs = performance.now() - sceneBuildStart;

    const viewportHeight = renderer?.domElement?.clientHeight
      ?? renderer?.domElement?.height
      ?? 720;
    const viewportWidth = renderer?.domElement?.clientWidth
      ?? renderer?.domElement?.width
      ?? 1280;

    if (this.useGpuVisibility) {
      this.gpuVisibility.update(
        renderer,
        sceneDescriptors.objects,
        camera,
        viewportHeight,
        appliedBudgets,
        this.frameIndex,
      );
    }
    const gpuVisibility = this.useGpuVisibility
      ? this.gpuVisibility.getLatestReadback()
      : {
          ready: false,
          pending: false,
          frameIndex: -1,
          clusterCount: 0,
          get: () => null,
          getSortedVisibleClusterIds: () => [],
        };

    const schedulerStart = performance.now();
    const sceneSelection = this.scheduler.evaluateScene({
      objects: sceneDescriptors.objects,
      camera,
      viewportHeight,
      frameIndex: this.frameIndex,
      budgets: appliedBudgets,
      gpuVisibility,
    });
    const schedulerMs = performance.now() - schedulerStart;

    const meshStats: SplatMeshFrameStats[] = [];
    let clusterMetadataBytes = 0;
    let pageDescriptorBytes = 0;
    let residencyBytes = 0;
    let compositorWeightedInstances = 0;
    let compositorHeroInstances = 0;
    let compositorDepthSlicedInstances = 0;
    let compositorActiveTiles = 0;
    let compositorWeightedTiles = 0;
    let compositorDepthSlicedTiles = 0;
    let compositorHeroTiles = 0;
    let compositorMaxTileComplexity = 0;
    let compositorVisibleClusters = 0;
    let compositorBinnedClusters = 0;
    let compositorBinnedClusterReferences = 0;
    let compositorOverflowedTiles = 0;
    let compositorOverflowedClusterReferences = 0;
    let compositorMaxClustersPerTile = 0;
    let compositorMaxTileSplatEstimate = 0;
    let compositorTileBufferBytes = 0;

    for (const descriptor of sceneDescriptors.objects) {
      const mesh = descriptor.mesh;
      const selection = sceneSelection.meshSelections.get(mesh.uuid) ?? {
        frontierClusterIds: [],
        activeClusters: [],
        activePageIds: [],
        requestedPageIds: [],
        visibleSplats: 0,
        estimatedOverdraw: 0,
        frontierStability: 1,
      };

      mesh.applySelection(selection, {
        camera,
        viewportWidth,
        viewportHeight,
        budgets: appliedBudgets,
        timeSeconds: this.elapsedSeconds,
        frameIndex: this.frameIndex,
        gpuVisibilityReady: gpuVisibility.ready,
        renderer,
      });

      const asset = mesh.getAsset();
      const compositor = mesh.getCompositorSnapshot();
      clusterMetadataBytes += asset.buffers.clusterMetadata.byteLength + asset.buffers.clusterReferences.byteLength + asset.buffers.childIndices.byteLength;
      pageDescriptorBytes += asset.buffers.pageDescriptors.byteLength;
      residencyBytes += mesh.getPageTable().residencyBuffer.byteLength;
      compositorWeightedInstances += compositor.weightedInstances;
      compositorHeroInstances += compositor.heroInstances;
      compositorDepthSlicedInstances += compositor.depthSlicedInstances;
      compositorActiveTiles += compositor.activeTiles;
      compositorWeightedTiles += compositor.weightedTiles;
      compositorDepthSlicedTiles += compositor.depthSlicedTiles;
      compositorHeroTiles += compositor.heroTiles;
      compositorMaxTileComplexity = Math.max(compositorMaxTileComplexity, compositor.maxTileComplexity);
      compositorVisibleClusters += compositor.visibleClusters;
      compositorBinnedClusters += compositor.binnedClusters;
      compositorBinnedClusterReferences += compositor.binnedClusterReferences;
      compositorOverflowedTiles += compositor.overflowedTiles;
      compositorOverflowedClusterReferences += compositor.overflowedClusterReferences;
      compositorMaxClustersPerTile = Math.max(compositorMaxClustersPerTile, compositor.maxClustersPerTile);
      compositorMaxTileSplatEstimate = Math.max(compositorMaxTileSplatEstimate, compositor.maxTileSplatEstimate);
      compositorTileBufferBytes += compositor.tileBufferBytes;

      meshStats.push({
        meshUuid: mesh.uuid,
        meshName: mesh.name || asset.label,
        frontierClusters: selection.frontierClusterIds.length,
        activePages: selection.activePageIds.length,
        residentPages: mesh.getPageTable().getResidentCount(),
        requestedPages: selection.requestedPageIds.length,
        visibleSplats: selection.visibleSplats,
        estimatedOverdraw: selection.estimatedOverdraw,
        frontierStability: selection.frontierStability,
        weightedInstances: compositor.weightedInstances,
        heroInstances: compositor.heroInstances,
        depthSlicedInstances: compositor.depthSlicedInstances,
        activeTiles: compositor.activeTiles,
        weightedTiles: compositor.weightedTiles,
        depthSlicedTiles: compositor.depthSlicedTiles,
        heroTiles: compositor.heroTiles,
        maxTileComplexity: compositor.maxTileComplexity,
        visibleClusters: compositor.visibleClusters,
        binnedClusters: compositor.binnedClusters,
        binnedClusterReferences: compositor.binnedClusterReferences,
        overflowedTiles: compositor.overflowedTiles,
        overflowedClusterReferences: compositor.overflowedClusterReferences,
        maxClustersPerTile: compositor.maxClustersPerTile,
        maxTileSplatEstimate: compositor.maxTileSplatEstimate,
        tileBufferBytes: compositor.tileBufferBytes,
      });
    }

    const gpuVisibilityFrameLag = gpuVisibility.ready
      ? Math.max(0, this.frameIndex - gpuVisibility.frameIndex)
      : 0;

    const snapshot = this.stats.commit({
      frameIndex: this.frameIndex,
      elapsedSeconds: this.elapsedSeconds,
      appliedGovernorLevel,
      governorReason,
      schedulerMode: sceneSelection.schedulerMode,
      budgets: cloneBudgets(appliedBudgets),
      meshCount: sceneDescriptors.objects.length,
      dirtyObjects: sceneDescriptors.dirtyObjectCount,
      sceneDescriptorBytes: sceneDescriptors.packedBuffer.byteLength,
      clusterMetadataBytes,
      pageDescriptorBytes,
      residencyBytes,
      gpuVisibilityReady: gpuVisibility.ready,
      gpuVisibilityPending: gpuVisibility.pending,
      gpuVisibilityFrameLag,
      gpuClusterCount: gpuVisibility.clusterCount,
      frontierClusters: sceneSelection.frontierClusters,
      visibleSplats: sceneSelection.visibleSplats,
      activePages: sceneSelection.activePages,
      residentPages: sceneSelection.residentPages,
      requestedPages: sceneSelection.requestedPages,
      pageUploads: sceneSelection.pageUploads,
      pageFaultRate: sceneSelection.frontierClusters === 0
        ? 0
        : sceneSelection.requestedPages / sceneSelection.frontierClusters,
      estimatedOverdraw: sceneSelection.estimatedOverdraw,
      frontierStability: sceneSelection.frontierStability,
      compositorWeightedInstances,
      compositorHeroInstances,
      compositorDepthSlicedInstances,
      compositorActiveTiles,
      compositorWeightedTiles,
      compositorDepthSlicedTiles,
      compositorHeroTiles,
      compositorMaxTileComplexity,
      compositorVisibleClusters,
      compositorBinnedClusters,
      compositorBinnedClusterReferences,
      compositorOverflowedTiles,
      compositorOverflowedClusterReferences,
      compositorMaxClustersPerTile,
      compositorMaxTileSplatEstimate,
      compositorTileBufferBytes,
      cpuFrameMs: performance.now() - frameStart,
      sceneBuildMs,
      schedulerMs,
      meshStats,
    });

    this.governor.observe(snapshot);
    this.frameIndex += 1;
    return snapshot;
  }
}
