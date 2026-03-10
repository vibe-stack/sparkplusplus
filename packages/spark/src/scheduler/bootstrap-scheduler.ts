import {
  Camera,
  Frustum,
  Matrix4,
  OrthographicCamera,
  PerspectiveCamera,
  Sphere,
  Vector3,
} from 'three';
import { SplatPageResidencyState } from '../assets/page-table';
import { hasSemanticFlag } from '../core/semantics';
import type { SplatBudgetOptions } from '../core/budgets';
import type { SplatSceneObjectDescriptor } from '../core/scene-descriptor';
import type { SplatSchedulerMode } from '../core/stats';
import type { SplatGpuVisibilityReadback } from '../gpu/visibility';

interface ClusterCandidate {
  descriptor: SplatSceneObjectDescriptor;
  clusterId: number;
  score: number;
  projectedSizePx: number;
  screenRadius: number;
  visibleCost: number;
  proxySplatCount: number;
  representedSplatCount: number;
}

interface FrontierBudgetState {
  reservedPageSlots: number;
  reservedVisibleSplats: number;
  reservedOverdraw: number;
}

interface FrontierRefinementOption {
  descriptor: SplatSceneObjectDescriptor;
  parent: ClusterCandidate;
  children: ClusterCandidate[];
  deltaPageSlots: number;
  deltaVisibleSplats: number;
  deltaOverdraw: number;
  priority: number;
}

export interface SplatActiveCluster {
  clusterId: number;
  pageId: number;
  level: number;
}

export interface SplatMeshSelection {
  frontierClusterIds: number[];
  activeClusters: SplatActiveCluster[];
  activePageIds: number[];
  requestedPageIds: number[];
  visibleSplats: number;
  estimatedOverdraw: number;
  frontierStability: number;
}

export interface SplatSceneSelection {
  meshSelections: Map<string, SplatMeshSelection>;
  schedulerMode: SplatSchedulerMode;
  frontierClusters: number;
  visibleSplats: number;
  activePages: number;
  residentPages: number;
  requestedPages: number;
  estimatedOverdraw: number;
  pageUploads: number;
  frontierStability: number;
}

const EMPTY_SELECTION: SplatMeshSelection = {
  frontierClusterIds: [],
  activeClusters: [],
  activePageIds: [],
  requestedPageIds: [],
  visibleSplats: 0,
  estimatedOverdraw: 0,
  frontierStability: 1,
};

export class BootstrapVisibilityScheduler {
  private readonly frustum = new Frustum();
  private readonly projectionMatrix = new Matrix4();
  private readonly worldCenter = new Vector3();
  private readonly projectedCenter = new Vector3();
  private readonly sphere = new Sphere();
  private readonly previousCameraPosition = new Vector3();
  private readonly currentCameraPosition = new Vector3();
  private readonly previousCameraForward = new Vector3();
  private readonly currentCameraForward = new Vector3();
  private hasPreviousCameraState = false;
  private cameraMotionFactor = 0;
  // Smoothed version: rises instantly on motion, decays slowly (~15 frames to
  // reach near-zero at 60 fps).  The raw per-frame factor drops to 0 in a
  // single frame when the camera stops.  That instant drop reduces
  // refinementHysteresis from ~0.92 → ~0.68 all at once, causing the entire
  // frontier to be replaced by deep children in one frame, flooding the page
  // table with simultaneous faults and triggering governor escalation.
  // Smoothing spreads the transition over ~10-15 frames so page uploads keep
  // up and the governor never sees a burst.
  private smoothedCameraMotionFactor = 0;

  evaluateScene(options: {
    objects: readonly SplatSceneObjectDescriptor[];
    camera: Camera;
    viewportHeight: number;
    frameIndex: number;
    budgets: SplatBudgetOptions;
    gpuVisibility?: SplatGpuVisibilityReadback;
  }): SplatSceneSelection {
    const { objects, camera, viewportHeight, frameIndex, budgets, gpuVisibility } = options;
    const instantMotionFactor = this.measureCameraMotion(camera);
    this.smoothedCameraMotionFactor = Math.max(
      instantMotionFactor,
      this.smoothedCameraMotionFactor * 0.82,
    );
    this.cameraMotionFactor = this.smoothedCameraMotionFactor;
    // Always accept GPU data up to 1 frame old.  GPU readback is async and is
    // never 0-lag; requiring 0-lag during motion meant the GPU path was NEVER
    // used while the camera was moving, permanently falling back to the slower
    // CPU refinement path.
    const acceptableGpuLagFrames = 2;
    const effectiveGpuVisibility = gpuVisibility && frameIndex - gpuVisibility.frameIndex <= acceptableGpuLagFrames
      ? gpuVisibility
      : undefined;
    const meshSelections = new Map<string, SplatMeshSelection>();
    const frontierCandidates = new Map<string, Map<number, ClusterCandidate>>();
    const prefetchProtectedPageIds = new Map<string, Set<number>>();
    const rootCandidates: ClusterCandidate[] = [];
    const frontierBudgetState: FrontierBudgetState = {
      reservedPageSlots: 0,
      reservedVisibleSplats: 0,
      reservedOverdraw: 0,
    };
    const schedulerMode: SplatSchedulerMode = effectiveGpuVisibility?.ready === true
      ? 'gpu-readback'
      : 'cpu-bootstrap';

    camera.updateMatrixWorld();
    this.projectionMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projectionMatrix);

    objects.forEach((descriptor) => {
      const mesh = descriptor.mesh;
      const asset = mesh.getAsset();
      const pageTable = mesh.getPageTable();
      meshSelections.set(mesh.uuid, {
        ...EMPTY_SELECTION,
        frontierClusterIds: [],
        activeClusters: [],
        activePageIds: [],
        requestedPageIds: [],
      });
      frontierCandidates.set(mesh.uuid, new Map());
      prefetchProtectedPageIds.set(mesh.uuid, new Set());

      rootCandidates.push(
        ...this.collectSeedCandidates(
          descriptor,
          camera,
          viewportHeight,
          pageTable,
          budgets,
          effectiveGpuVisibility,
        ),
      );
    });

    if (effectiveGpuVisibility?.ready === true) {
      this.buildGpuFrontier({
        objects,
        camera,
        viewportHeight,
        budgets,
        gpuVisibility: effectiveGpuVisibility,
        frontierBudgetState,
        meshSelections,
        frontierCandidates,
      });
    } else {
      rootCandidates.sort((left, right) => right.score - left.score);

      for (const rootCandidate of rootCandidates) {
        const meshSelection = meshSelections.get(rootCandidate.descriptor.mesh.uuid)!;
        const candidateSet = frontierCandidates.get(rootCandidate.descriptor.mesh.uuid)!;
        this.tryAddFrontierCandidate(
          rootCandidate,
          budgets,
          frontierBudgetState,
          meshSelection,
          candidateSet,
          meshSelection.frontierClusterIds.length === 0,
        );
      }

      this.refineFrontier({
        objects,
        camera,
        viewportHeight,
        frameIndex,
        budgets,
        gpuVisibility: effectiveGpuVisibility,
        frontierBudgetState,
        meshSelections,
        frontierCandidates,
        prefetchProtectedPageIds,
      });
    }

    meshSelections.forEach((selection) => {
      selection.frontierClusterIds.sort((left, right) => left - right);
    });

    for (const descriptor of objects) {
      this.populateSelectionResidency(descriptor, meshSelections.get(descriptor.mesh.uuid)!, frameIndex);
    }

    this.allocateResidentCapacities(objects, budgets.maxResidentPages);

    let pageUploads = 0;
    let uploadsRemaining = budgets.maxPageUploadsPerFrame;

    while (uploadsRemaining > 0) {
      let madeProgress = false;

      for (const descriptor of objects) {
        if (uploadsRemaining <= 0) {
          break;
        }

        const selection = meshSelections.get(descriptor.mesh.uuid)!;
        const protectedPageIds = new Set([
          ...selection.activePageIds,
          ...(prefetchProtectedPageIds.get(descriptor.mesh.uuid) ?? []),
        ]);
        const uploaded = descriptor.mesh
          .getPageTable()
          .serviceRequests(1, frameIndex, protectedPageIds);

        if (uploaded.length > 0) {
          pageUploads += uploaded.length;
          uploadsRemaining -= uploaded.length;
          madeProgress = true;
        }
      }

      if (!madeProgress) {
        break;
      }
    }

    for (const descriptor of objects) {
      this.populateSelectionResidency(descriptor, meshSelections.get(descriptor.mesh.uuid)!, frameIndex);
    }

    let visibleSplats = 0;
    let frontierClusters = 0;
    let activePages = 0;
    let residentPages = 0;
    let requestedPages = 0;
    let estimatedOverdraw = 0;
    let frontierStability = 0;

    objects.forEach((descriptor) => {
      const mesh = descriptor.mesh;
      const selection = meshSelections.get(mesh.uuid)!;
      const pageTable = mesh.getPageTable();
      const previousFrontier = mesh.getPreviousFrontierClusterIds();
      const union = new Set([...previousFrontier, ...selection.frontierClusterIds]);
      const stableCount = selection.frontierClusterIds.filter((clusterId) => previousFrontier.includes(clusterId)).length;

      selection.frontierStability = union.size === 0 ? 1 : stableCount / union.size;

      visibleSplats += selection.visibleSplats;
      frontierClusters += selection.frontierClusterIds.length;
      activePages += selection.activePageIds.length;
      residentPages += pageTable.getResidentCount();
      requestedPages += selection.requestedPageIds.length;
      estimatedOverdraw += selection.estimatedOverdraw;
      frontierStability += selection.frontierStability;
    });

    return {
      meshSelections,
      schedulerMode,
      frontierClusters,
      visibleSplats,
      activePages,
      residentPages,
      requestedPages,
      estimatedOverdraw,
      pageUploads,
      frontierStability: objects.length === 0 ? 1 : frontierStability / objects.length,
    };
  }

  private populateSelectionResidency(
    descriptor: SplatSceneObjectDescriptor,
    selection: SplatMeshSelection,
    frameIndex: number,
  ): void {
    const mesh = descriptor.mesh;
    const asset = mesh.getAsset();
    const pageTable = mesh.getPageTable();
    const activePageIds = new Set<number>();
    const requestedPageIds = new Set<number>();

    selection.activeClusters = [];
    selection.activePageIds = [];
    selection.requestedPageIds = [];
    selection.visibleSplats = 0;

    for (const clusterId of selection.frontierClusterIds) {
      const cluster = asset.clusters[clusterId]!;
      const page = asset.pages[cluster.pageId]!;

      if (pageTable.isResident(page.id)) {
        this.activateResidentCluster(selection, activePageIds, pageTable, page.id, clusterId, cluster.level, page.splatCount, frameIndex);
        continue;
      }

      pageTable.request(page.id, frameIndex);
      requestedPageIds.add(page.id);

      const fallbackClusterId = this.findResidentFallbackClusterId(clusterId, asset, pageTable);

      if (fallbackClusterId === null) {
        continue;
      }

      const fallbackCluster = asset.clusters[fallbackClusterId]!;
      const fallbackPage = asset.pages[fallbackCluster.pageId]!;
      this.activateResidentCluster(
        selection,
        activePageIds,
        pageTable,
        fallbackPage.id,
        fallbackClusterId,
        fallbackCluster.level,
        fallbackPage.splatCount,
        frameIndex,
      );
    }

    selection.requestedPageIds = [...requestedPageIds];

    // Keep root-cluster pages permanently warm in the LRU.
    // Root pages are the last-resort fallback for every non-resident frontier
    // cluster.  When the frontier advances to deep clusters (e.g. once the GPU
    // path takes over after the camera settles), root pages stop being touched
    // and are eventually evicted by evictColdest.  Without this guard,
    // findResidentFallbackClusterId can walk all the way to a root that is no
    // longer resident and return null, producing a tile hole with no content.
    for (const rootClusterId of asset.rootClusterIds) {
      const rootCluster = asset.clusters[rootClusterId]!;
      const rootPage = asset.pages[rootCluster.pageId]!;

      if (pageTable.isResident(rootPage.id)) {
        pageTable.markTouched(rootPage.id, frameIndex);
      } else {
        pageTable.request(rootPage.id, frameIndex);
      }
    }
  }

  private activateResidentCluster(
    selection: SplatMeshSelection,
    activePageIds: Set<number>,
    pageTable: ReturnType<SplatSceneObjectDescriptor['mesh']['getPageTable']>,
    pageId: number,
    clusterId: number,
    level: number,
    splatCount: number,
    frameIndex: number,
  ): void {
    if (activePageIds.has(pageId)) {
      return;
    }

    pageTable.markTouched(pageId, frameIndex);
    activePageIds.add(pageId);
    selection.activeClusters.push({
      clusterId,
      pageId,
      level,
    });
    selection.activePageIds.push(pageId);
    selection.visibleSplats += splatCount;
  }

  private findResidentFallbackClusterId(
    clusterId: number,
    asset: ReturnType<SplatSceneObjectDescriptor['mesh']['getAsset']>,
    pageTable: ReturnType<SplatSceneObjectDescriptor['mesh']['getPageTable']>,
  ): number | null {
    let currentCluster = asset.clusters[clusterId]!;

    while (currentCluster.parentId !== null) {
      const parentCluster = asset.clusters[currentCluster.parentId]!;

      if (pageTable.isResident(parentCluster.pageId)) {
        return parentCluster.id;
      }

      currentCluster = parentCluster;
    }

    return pageTable.isResident(currentCluster.pageId) ? currentCluster.id : null;
  }

  private buildGpuFrontier(options: {
    objects: readonly SplatSceneObjectDescriptor[];
    camera: Camera,
    viewportHeight: number,
    budgets: SplatBudgetOptions,
    gpuVisibility: SplatGpuVisibilityReadback,
    frontierBudgetState: FrontierBudgetState,
    meshSelections: Map<string, SplatMeshSelection>,
    frontierCandidates: Map<string, Map<number, ClusterCandidate>>,
  }): void {
    const {
      objects,
      camera,
      viewportHeight,
      budgets,
      gpuVisibility,
      frontierBudgetState,
      meshSelections,
      frontierCandidates,
    } = options;

    for (const descriptor of objects) {
      const mesh = descriptor.mesh;
      const asset = mesh.getAsset();
      const pageTable = mesh.getPageTable();
      const meshSelection = meshSelections.get(mesh.uuid)!;
      const meshFrontierCandidates = frontierCandidates.get(mesh.uuid)!;
      const selectedClusterIds = new Set<number>();
      const visibleClusterIds = gpuVisibility.getSortedVisibleClusterIds(mesh.uuid);
      const visibleClusterSet = new Set(visibleClusterIds);

      // Precompute visible-descendant counts once so the sort comparator is O(1)
      // instead of O(subtree) per comparison (which was O(n·log n·depth) total).
      const visibleDescendantCountCache = new Map<number, number>();
      const getVisibleDescendantCount = (clusterId: number): number => {
        const cached = visibleDescendantCountCache.get(clusterId);
        if (cached !== undefined) return cached;
        const result = this.countVisibleDescendants(clusterId, visibleClusterSet, asset);
        visibleDescendantCountCache.set(clusterId, result);
        return result;
      };

      const visibleCandidates = visibleClusterIds
        .map((clusterId) =>
          this.scoreCandidate(
            descriptor,
            clusterId,
            camera,
            viewportHeight,
            pageTable,
            budgets,
            gpuVisibility,
          ),
        )
        .filter((candidate): candidate is ClusterCandidate => candidate !== null)
        .sort((left, right) => {
          const leftCluster = asset.clusters[left.clusterId]!;
          const rightCluster = asset.clusters[right.clusterId]!;
          const leftPriority = left.score
            * (1 + leftCluster.level * 0.6)
            / Math.max(1, left.visibleCost * (1 + getVisibleDescendantCount(left.clusterId) * 0.35));
          const rightPriority = right.score
            * (1 + rightCluster.level * 0.6)
            / Math.max(1, right.visibleCost * (1 + getVisibleDescendantCount(right.clusterId) * 0.35));

          return rightPriority - leftPriority;
        });

      for (const candidate of visibleCandidates) {
        const clusterId = candidate.clusterId;

        if (selectedClusterIds.has(clusterId)) {
          continue;
        }

        // Do NOT hard-skip ancestors with visible descendants.  When leaf pages
        // are not yet resident the fallback needs the ancestor to be in the
        // frontier; the hasSelectedAncestor/Descendant checks handle mutual
        // exclusion once clusters are actually selected.
        if (
          this.hasSelectedAncestor(clusterId, selectedClusterIds, asset)
          || this.hasSelectedDescendant(clusterId, selectedClusterIds, asset)
        ) {
          continue;
        }

        if (
          this.tryAddFrontierCandidate(
            candidate,
            budgets,
            frontierBudgetState,
            meshSelection,
            meshFrontierCandidates,
            meshSelection.frontierClusterIds.length < 6,
          )
        ) {
          selectedClusterIds.add(clusterId);
        }
      }

      for (const rootClusterId of asset.rootClusterIds) {
        if (this.subtreeHasSelectedCluster(rootClusterId, selectedClusterIds, asset)) {
          continue;
        }

        const rootCandidate = this.scoreCandidate(
          descriptor,
          rootClusterId,
          camera,
          viewportHeight,
          pageTable,
          budgets,
          gpuVisibility,
        );

        if (!rootCandidate) {
          continue;
        }

        if (
          this.tryAddFrontierCandidate(
            rootCandidate,
            budgets,
            frontierBudgetState,
            meshSelection,
            meshFrontierCandidates,
            meshSelection.frontierClusterIds.length === 0,
          )
        ) {
          selectedClusterIds.add(rootClusterId);
        }
      }
    }
  }

  private collectSeedCandidates(
    descriptor: SplatSceneObjectDescriptor,
    camera: Camera,
    viewportHeight: number,
    pageTable: ReturnType<SplatSceneObjectDescriptor['mesh']['getPageTable']>,
    budgets: SplatBudgetOptions,
    gpuVisibility?: SplatGpuVisibilityReadback,
  ): ClusterCandidate[] {
    const asset = descriptor.mesh.getAsset();
    // Always reseed the CPU frontier from roots. Reusing the previous frontier
    // as the seed can strand visible sibling regions once one descendant under
    // a root survives scoring, which manifests as in-view tiles disappearing
    // exactly when the camera settles. Temporal history is still consumed later
    // via temporalWeight and pinned-descendant checks.
    return asset.rootClusterIds
      .map((clusterId) =>
        this.scoreCandidate(descriptor, clusterId, camera, viewportHeight, pageTable, budgets, gpuVisibility),
      )
      .filter((candidate): candidate is ClusterCandidate => candidate !== null);
  }

  private refineFrontier(options: {
    objects: readonly SplatSceneObjectDescriptor[];
    camera: Camera,
    viewportHeight: number,
    frameIndex: number,
    budgets: SplatBudgetOptions,
    gpuVisibility: SplatGpuVisibilityReadback | undefined,
    frontierBudgetState: FrontierBudgetState,
    meshSelections: Map<string, SplatMeshSelection>,
    frontierCandidates: Map<string, Map<number, ClusterCandidate>>;
    prefetchProtectedPageIds: Map<string, Set<number>>;
  }): void {
    const {
      objects,
      camera,
      viewportHeight,
      frameIndex,
      budgets,
      gpuVisibility,
      frontierBudgetState,
      meshSelections,
      frontierCandidates,
      prefetchProtectedPageIds,
    } = options;

    let madeProgress = true;

    while (madeProgress) {
      madeProgress = false;
      const refinementOptions: FrontierRefinementOption[] = [];

      for (const descriptor of objects) {
        const mesh = descriptor.mesh;
        const asset = mesh.getAsset();
        const pageTable = mesh.getPageTable();
        const meshFrontierCandidates = frontierCandidates.get(mesh.uuid)!;
        const previousFrontierSet = new Set(mesh.getPreviousFrontierClusterIds());
        const pinnedDescendantMemo = new Map<number, boolean>();

        for (const parent of meshFrontierCandidates.values()) {
          const cluster = asset.clusters[parent.clusterId]!;

          if (!this.shouldRefineCandidate(
            parent,
            clusterId => this.hasPinnedDescendant(
              clusterId,
              asset,
              previousFrontierSet,
              pageTable,
              pinnedDescendantMemo,
            ),
            budgets,
          )) {
            continue;
          }

          const children = cluster.childIds
            .map((childId) =>
              this.scoreCandidate(descriptor, childId, camera, viewportHeight, pageTable, budgets, gpuVisibility),
            )
            .filter((childCandidate): childCandidate is ClusterCandidate => childCandidate !== null)
            .sort((left, right) => right.score - left.score);

          if (children.length === 0) {
            continue;
          }

          this.requestRefinementChildren(
            children,
            asset,
            pageTable,
            frameIndex,
            prefetchProtectedPageIds.get(mesh.uuid)!,
          );

          if (!this.areReplacementChildrenResident(children, asset, pageTable)) {
            continue;
          }

          // Replacing a parent with only a high-scoring subset of its children
          // leaves the omitted siblings with no frontier coverage, which shows
          // up as tiles disappearing after motion settles.  Only promote a
          // parent when the full visible/scorable child set can replace it.
          let accumulatedScore = 0;
          let accumulatedVisibleCost = 0;
          let accumulatedOverdraw = 0;

          for (const child of children) {
            accumulatedScore += child.score;
            accumulatedVisibleCost += child.visibleCost;
            accumulatedOverdraw += this.getCandidateOverdraw(child);
          }

          refinementOptions.push({
            descriptor,
            parent,
            children,
            deltaPageSlots: children.length - 1,
            deltaVisibleSplats: accumulatedVisibleCost - parent.visibleCost,
            deltaOverdraw: accumulatedOverdraw - this.getCandidateOverdraw(parent),
            priority: accumulatedScore
              - parent.score
              + parent.projectedSizePx * 0.05
              + Math.log2(
                Math.max(1, parent.representedSplatCount / Math.max(1, parent.proxySplatCount)),
              ) * parent.projectedSizePx * 0.08
              + children.length * 0.12,
          });
        }
      }

      refinementOptions.sort((left, right) => right.priority - left.priority);

      for (const option of refinementOptions) {
        const mesh = option.descriptor.mesh;
        const meshSelection = meshSelections.get(mesh.uuid)!;
        const meshFrontierCandidates = frontierCandidates.get(mesh.uuid)!;

        if (!meshFrontierCandidates.has(option.parent.clusterId)) {
          continue;
        }

        if (!this.canReplaceFrontierCandidate(option, frontierBudgetState, budgets)) {
          continue;
        }

        this.replaceFrontierCandidate(
          option.parent,
          option.children,
          frontierBudgetState,
          meshSelection,
          meshFrontierCandidates,
        );
        madeProgress = true;
      }
    }
  }

  private shouldRefineCandidate(
    candidate: ClusterCandidate,
    hasPinnedDescendantSelection: (clusterId: number) => boolean,
    budgets: SplatBudgetOptions,
  ): boolean {
    if (candidate.projectedSizePx <= 0) {
      return false;
    }

    const motionDamping = 1 - Math.min(1, this.cameraMotionFactor * 1.6);
    const refinementHysteresis = hasPinnedDescendantSelection(candidate.clusterId)
      ? 0.68 + (1 - motionDamping) * 0.24
      : 0.9 + (1 - motionDamping) * 0.08;
    const representationGap = candidate.representedSplatCount / Math.max(1, candidate.proxySplatCount);
    const detailPressure = Math.min(1.8, 1 + Math.log2(Math.max(1, representationGap)) * 0.18);
    return candidate.projectedSizePx * detailPressure > budgets.minProjectedNodeSizePx * refinementHysteresis;
  }

  private requestRefinementChildren(
    children: readonly ClusterCandidate[],
    asset: ReturnType<SplatSceneObjectDescriptor['mesh']['getAsset']>,
    pageTable: ReturnType<SplatSceneObjectDescriptor['mesh']['getPageTable']>,
    frameIndex: number,
    protectedPageIds: Set<number>,
  ): void {
    for (const child of children) {
      const childPageId = asset.clusters[child.clusterId]!.pageId;
      protectedPageIds.add(childPageId);
      pageTable.request(childPageId, frameIndex);
    }
  }

  private areReplacementChildrenResident(
    children: readonly ClusterCandidate[],
    asset: ReturnType<SplatSceneObjectDescriptor['mesh']['getAsset']>,
    pageTable: ReturnType<SplatSceneObjectDescriptor['mesh']['getPageTable']>,
  ): boolean {
    return children.every((child) => pageTable.isResident(asset.clusters[child.clusterId]!.pageId));
  }

  private hasPinnedDescendant(
    clusterId: number,
    asset: ReturnType<SplatSceneObjectDescriptor['mesh']['getAsset']>,
    previousFrontierSet: ReadonlySet<number>,
    pageTable: ReturnType<SplatSceneObjectDescriptor['mesh']['getPageTable']>,
    memo: Map<number, boolean>,
  ): boolean {
    const cached = memo.get(clusterId);

    if (cached !== undefined) {
      return cached;
    }

    const cluster = asset.clusters[clusterId]!;
    let result = false;

    for (const childId of cluster.childIds) {
      const childCluster = asset.clusters[childId]!;

      if ((previousFrontierSet.has(childId) || pageTable.isResident(childCluster.pageId)) && this.cameraMotionFactor < 0.3) {
        result = true;
        break;
      }

      if (this.hasPinnedDescendant(childId, asset, previousFrontierSet, pageTable, memo)) {
        result = true;
        break;
      }
    }

    memo.set(clusterId, result);
    return result;
  }

  private tryAddFrontierCandidate(
    candidate: ClusterCandidate,
    budgets: SplatBudgetOptions,
    frontierBudgetState: FrontierBudgetState,
    meshSelection: SplatMeshSelection,
    frontierCandidates: Map<number, ClusterCandidate>,
    allowOverflowForCoverage = false,
  ): boolean {
    const cluster = candidate.descriptor.mesh.getAsset().clusters[candidate.clusterId]!;
    const projectedOverdraw = this.getCandidateOverdraw(candidate);
    const wouldOverflowPages = frontierBudgetState.reservedPageSlots + 1 > budgets.maxActivePages;
    const wouldOverflowSplats = frontierBudgetState.reservedVisibleSplats + candidate.visibleCost > budgets.maxVisibleSplats;
    const wouldOverflowOverdraw = frontierBudgetState.reservedOverdraw + projectedOverdraw > budgets.maxOverdrawBudget;

    if (!allowOverflowForCoverage && (wouldOverflowPages || wouldOverflowSplats || wouldOverflowOverdraw)) {
      return false;
    }

    if (!frontierCandidates.has(candidate.clusterId)) {
      meshSelection.frontierClusterIds.push(candidate.clusterId);
      meshSelection.estimatedOverdraw += projectedOverdraw;
      frontierBudgetState.reservedPageSlots += 1;
      frontierBudgetState.reservedVisibleSplats += candidate.visibleCost;
      frontierBudgetState.reservedOverdraw += projectedOverdraw;
      frontierCandidates.set(candidate.clusterId, candidate);
    }

    return true;
  }

  private canReplaceFrontierCandidate(
    option: FrontierRefinementOption,
    frontierBudgetState: FrontierBudgetState,
    budgets: SplatBudgetOptions,
  ): boolean {
    const closeUpFactor = Math.min(
      1.9,
      Math.max(1, option.parent.projectedSizePx / Math.max(1, budgets.minProjectedNodeSizePx * 2.2)),
    );
    const pageLimit = Math.round(budgets.maxActivePages * Math.min(1.45, 1 + (closeUpFactor - 1) * 0.28));
    const visibleLimit = Math.round(budgets.maxVisibleSplats * Math.min(1.8, 1 + (closeUpFactor - 1) * 0.7));
    const overdrawLimit = Math.round(budgets.maxOverdrawBudget * Math.min(1.6, 1 + (closeUpFactor - 1) * 0.42));

    return frontierBudgetState.reservedPageSlots + option.deltaPageSlots <= pageLimit
      && frontierBudgetState.reservedVisibleSplats + option.deltaVisibleSplats <= visibleLimit
      && frontierBudgetState.reservedOverdraw + option.deltaOverdraw <= overdrawLimit;
  }

  private replaceFrontierCandidate(
    parent: ClusterCandidate,
    children: readonly ClusterCandidate[],
    frontierBudgetState: FrontierBudgetState,
    meshSelection: SplatMeshSelection,
    frontierCandidates: Map<number, ClusterCandidate>,
  ): void {
    this.removeFrontierCandidate(parent, frontierBudgetState, meshSelection, frontierCandidates);

    for (const child of children) {
      this.addFrontierCandidateUnchecked(child, frontierBudgetState, meshSelection, frontierCandidates);
    }
  }

  private removeFrontierCandidate(
    candidate: ClusterCandidate,
    frontierBudgetState: FrontierBudgetState,
    meshSelection: SplatMeshSelection,
    frontierCandidates: Map<number, ClusterCandidate>,
  ): void {
    const overdraw = this.getCandidateOverdraw(candidate);
    const clusterIndex = meshSelection.frontierClusterIds.indexOf(candidate.clusterId);

    if (clusterIndex !== -1) {
      meshSelection.frontierClusterIds.splice(clusterIndex, 1);
    }

    meshSelection.estimatedOverdraw = Math.max(0, meshSelection.estimatedOverdraw - overdraw);
    frontierBudgetState.reservedPageSlots = Math.max(0, frontierBudgetState.reservedPageSlots - 1);
    frontierBudgetState.reservedVisibleSplats = Math.max(0, frontierBudgetState.reservedVisibleSplats - candidate.visibleCost);
    frontierBudgetState.reservedOverdraw = Math.max(0, frontierBudgetState.reservedOverdraw - overdraw);
    frontierCandidates.delete(candidate.clusterId);
  }

  private addFrontierCandidateUnchecked(
    candidate: ClusterCandidate,
    frontierBudgetState: FrontierBudgetState,
    meshSelection: SplatMeshSelection,
    frontierCandidates: Map<number, ClusterCandidate>,
  ): void {
    const overdraw = this.getCandidateOverdraw(candidate);

    if (frontierCandidates.has(candidate.clusterId)) {
      return;
    }

    meshSelection.frontierClusterIds.push(candidate.clusterId);
    meshSelection.estimatedOverdraw += overdraw;
    frontierBudgetState.reservedPageSlots += 1;
    frontierBudgetState.reservedVisibleSplats += candidate.visibleCost;
    frontierBudgetState.reservedOverdraw += overdraw;
    frontierCandidates.set(candidate.clusterId, candidate);
  }

  private getRootClusterId(
    clusterId: number,
    asset: ReturnType<SplatSceneObjectDescriptor['mesh']['getAsset']>,
  ): number {
    let currentCluster = asset.clusters[clusterId]!;

    while (currentCluster.parentId !== null) {
      currentCluster = asset.clusters[currentCluster.parentId]!;
    }

    return currentCluster.id;
  }

  private hasSelectedAncestor(
    clusterId: number,
    selectedClusterIds: ReadonlySet<number>,
    asset: ReturnType<SplatSceneObjectDescriptor['mesh']['getAsset']>,
  ): boolean {
    let currentCluster = asset.clusters[clusterId]!;

    while (currentCluster.parentId !== null) {
      if (selectedClusterIds.has(currentCluster.parentId)) {
        return true;
      }

      currentCluster = asset.clusters[currentCluster.parentId]!;
    }

    return false;
  }

  private hasSelectedDescendant(
    clusterId: number,
    selectedClusterIds: ReadonlySet<number>,
    asset: ReturnType<SplatSceneObjectDescriptor['mesh']['getAsset']>,
  ): boolean {
    const cluster = asset.clusters[clusterId]!;

    for (const childId of cluster.childIds) {
      if (selectedClusterIds.has(childId) || this.hasSelectedDescendant(childId, selectedClusterIds, asset)) {
        return true;
      }
    }

    return false;
  }

  private subtreeHasSelectedCluster(
    clusterId: number,
    selectedClusterIds: ReadonlySet<number>,
    asset: ReturnType<SplatSceneObjectDescriptor['mesh']['getAsset']>,
  ): boolean {
    if (selectedClusterIds.has(clusterId)) {
      return true;
    }

    const cluster = asset.clusters[clusterId]!;
    return cluster.childIds.some((childId) => this.subtreeHasSelectedCluster(childId, selectedClusterIds, asset));
  }

  private countVisibleDescendants(
    clusterId: number,
    visibleClusterIds: ReadonlySet<number>,
    asset: ReturnType<SplatSceneObjectDescriptor['mesh']['getAsset']>,
  ): number {
    let count = 0;
    const cluster = asset.clusters[clusterId]!;

    for (const childId of cluster.childIds) {
      if (visibleClusterIds.has(childId)) {
        count += 1;
      }

      count += this.countVisibleDescendants(childId, visibleClusterIds, asset);
    }

    return count;
  }

  private measureCameraMotion(camera: Camera): number {
    camera.getWorldPosition(this.currentCameraPosition);
    camera.getWorldDirection(this.currentCameraForward);

    if (!this.hasPreviousCameraState) {
      this.previousCameraPosition.copy(this.currentCameraPosition);
      this.previousCameraForward.copy(this.currentCameraForward);
      this.hasPreviousCameraState = true;
      return 0;
    }

    const positionDelta = this.currentCameraPosition.distanceTo(this.previousCameraPosition);
    const angularDelta = 1 - Math.max(-1, Math.min(1, this.currentCameraForward.dot(this.previousCameraForward)));
    const normalizedPositionDelta = positionDelta / Math.max(1, this.currentCameraPosition.length());
    const motionFactor = Math.min(1, normalizedPositionDelta * 8 + angularDelta * 4);

    this.previousCameraPosition.copy(this.currentCameraPosition);
    this.previousCameraForward.copy(this.currentCameraForward);

    return motionFactor;
  }

  private getCandidateOverdraw(candidate: ClusterCandidate): number {
    const cluster = candidate.descriptor.mesh.getAsset().clusters[candidate.clusterId]!;
    const closeUpPressure = Math.max(1, candidate.projectedSizePx / 96);
    const screenFill = Math.min(1.45, Math.max(0.28, candidate.screenRadius / 120));
    const leafRelief = cluster.childIds.length === 0 ? 0.42 : 1;
    const closeUpRelief = 1 / Math.min(2.4, Math.max(1, closeUpPressure * 0.75));
    return cluster.expectedOverdrawScore * screenFill * leafRelief * Math.max(0.42, closeUpRelief);
  }

  private allocateResidentCapacities(
    objects: readonly SplatSceneObjectDescriptor[],
    totalResidentPages: number,
  ): void {
    if (objects.length === 0) {
      return;
    }

    const weights = objects.map((descriptor) => Math.max(1, descriptor.mesh.importance));
    const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
    let remainingCapacity = Math.max(objects.length, totalResidentPages);

    const capacities = objects.map((descriptor, index) => {
      const weighted = Math.max(
        1,
        Math.floor((remainingCapacity * weights[index]!) / totalWeight),
      );
      return Math.min(weighted, descriptor.mesh.getAsset().pages.length);
    });

    remainingCapacity -= capacities.reduce((sum, capacity) => sum + capacity, 0);

    for (let index = 0; remainingCapacity > 0; index = (index + 1) % capacities.length) {
      const descriptor = objects[index]!;
      if (capacities[index]! < descriptor.mesh.getAsset().pages.length) {
        capacities[index] = capacities[index]! + 1;
        remainingCapacity -= 1;
      }
    }

    objects.forEach((descriptor, index) => {
      descriptor.mesh.getPageTable().setResidentCapacity(capacities[index]!);
    });
  }

  private scoreCandidate(
    descriptor: SplatSceneObjectDescriptor,
    clusterId: number,
    camera: Camera,
    viewportHeight: number,
    pageTable: ReturnType<SplatSceneObjectDescriptor['mesh']['getPageTable']>,
    budgets: SplatBudgetOptions,
    gpuVisibility?: SplatGpuVisibilityReadback,
  ): ClusterCandidate | null {
    const mesh = descriptor.mesh;
    const asset = mesh.getAsset();
    const cluster = asset.clusters[clusterId]!;
    const page = asset.pages[cluster.pageId]!;
    const proxySplatCount = Math.max(1, page.splatCount);
    const representedSplatCount = Math.max(proxySplatCount, cluster.representedSplatCount);
    const gpuResult = gpuVisibility?.get(mesh.uuid, clusterId) ?? null;
    let projectedSizePx = 0;
    let screenRadius = 0;

    if (gpuResult) {
      if (!gpuResult.visible || gpuResult.projectedSizePx <= 0.75) {
        return null;
      }

      this.worldCenter
        .set(cluster.center[0], cluster.center[1], cluster.center[2])
        .applyMatrix4(mesh.matrixWorld);
      projectedSizePx = gpuResult.projectedSizePx;
      screenRadius = gpuResult.screenRadius;
    } else {
      const matrixWorld = mesh.matrixWorld.elements;
      const maxScale = Math.max(
        Math.hypot(matrixWorld[0]!, matrixWorld[1]!, matrixWorld[2]!),
        Math.hypot(matrixWorld[4]!, matrixWorld[5]!, matrixWorld[6]!),
        Math.hypot(matrixWorld[8]!, matrixWorld[9]!, matrixWorld[10]!),
      );

      this.worldCenter
        .set(cluster.center[0], cluster.center[1], cluster.center[2])
        .applyMatrix4(mesh.matrixWorld);

      this.sphere.center.copy(this.worldCenter);
      this.sphere.radius = cluster.radius * maxScale;

      if (!this.frustum.intersectsSphere(this.sphere)) {
        return null;
      }

      projectedSizePx = this.getProjectedRadiusPx(camera, this.worldCenter, this.sphere.radius, viewportHeight);
      if (projectedSizePx <= 0.75) {
        return null;
      }

      screenRadius = projectedSizePx;
    }

    this.projectedCenter.copy(this.worldCenter).project(camera);
    const radialDistance = Math.min(1.5, Math.hypot(this.projectedCenter.x, this.projectedCenter.y));
    const foveationWeight = Math.max(0.25, 1 - radialDistance * 0.25 * budgets.peripheralFoveation);
    const temporalWeight = mesh.getPreviousFrontierClusterIds().includes(clusterId)
      ? 1 + (budgets.temporalStabilityBias - 1) * Math.max(0, 1 - this.cameraMotionFactor * 1.8)
      : 1;
    const semanticWeight = hasSemanticFlag(cluster.semanticMask, 'hero') ? 1.35 : 1;
    const residencyWeight = pageTable.getState(cluster.pageId).state === SplatPageResidencyState.Resident
      ? 1.1
      : 0.92;
    const closeUpPressure = Math.max(1, projectedSizePx / Math.max(1, budgets.minProjectedNodeSizePx));
    const detailCredit = Math.min(5, Math.pow(closeUpPressure, 0.85));
    const refinementCredit = cluster.childIds.length > 0 ? 1.15 : 1;
    const leafCredit = cluster.childIds.length === 0 ? 1.4 : 1;
    const closeUpCredit = Math.min(3.2, 1 + Math.max(0, closeUpPressure - 1) * 0.8);
    const representationGap = representedSplatCount / proxySplatCount;
    const detailDebtWeight = Math.min(2.6, 1 + Math.log2(Math.max(1, representationGap)) * 0.3);
    const visibleCost = Math.max(
      cluster.childIds.length === 0 ? 48 : 96,
      Math.round(
        proxySplatCount
        / (detailCredit * residencyWeight * refinementCredit * leafCredit * closeUpCredit),
      ),
    );
    const densityPenalty = 1 + cluster.expectedOverdrawScore / Math.max(
      1,
      budgets.maxOverdrawBudget * (cluster.childIds.length === 0 ? 1.75 : 1),
    );
    const score = (
      projectedSizePx
      * cluster.projectedErrorCoefficient
      * mesh.importance
      * foveationWeight
      * temporalWeight
      * semanticWeight
      * residencyWeight
      * detailDebtWeight
      * leafCredit
      * Math.min(1.8, 0.9 + closeUpPressure * 0.22)
    ) / densityPenalty;

    return {
      descriptor,
      clusterId,
      score,
      projectedSizePx,
      screenRadius,
      visibleCost,
      proxySplatCount,
      representedSplatCount,
    };
  }

  private getProjectedRadiusPx(
    camera: Camera,
    center: Vector3,
    radius: number,
    viewportHeight: number,
  ): number {
    if (camera instanceof PerspectiveCamera) {
      const distance = Math.max(0.001, camera.position.distanceTo(center));
      const fovRadians = (camera.fov * Math.PI) / 180;
      return (radius / (distance * Math.tan(fovRadians * 0.5))) * viewportHeight;
    }

    if (camera instanceof OrthographicCamera) {
      const span = Math.max(0.001, camera.top - camera.bottom);
      return (radius * viewportHeight * camera.zoom) / span;
    }

    return radius * 12;
  }
}
