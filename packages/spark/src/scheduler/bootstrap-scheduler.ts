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

interface ClusterCandidate {
  descriptor: SplatSceneObjectDescriptor;
  clusterId: number;
  score: number;
  projectedSizePx: number;
  screenRadius: number;
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

  evaluateScene(options: {
    objects: readonly SplatSceneObjectDescriptor[];
    camera: Camera;
    viewportHeight: number;
    frameIndex: number;
    budgets: SplatBudgetOptions;
  }): SplatSceneSelection {
    const { objects, camera, viewportHeight, frameIndex, budgets } = options;
    const meshSelections = new Map<string, SplatMeshSelection>();
    const candidates: ClusterCandidate[] = [];
    let reservedPageSlots = 0;
    let reservedVisibleSplats = 0;
    let reservedOverdraw = 0;

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

      candidates.push(
        ...asset.rootClusterIds
          .map((clusterId) =>
            this.scoreCandidate(descriptor, clusterId, camera, viewportHeight, pageTable, budgets),
          )
          .filter((candidate): candidate is ClusterCandidate => candidate !== null),
      );
    });

    while (candidates.length > 0) {
      candidates.sort((left, right) => right.score - left.score);
      const candidate = candidates.shift()!;
      const { descriptor, clusterId, projectedSizePx, screenRadius } = candidate;
      const mesh = descriptor.mesh;
      const asset = mesh.getAsset();
      const pageTable = mesh.getPageTable();
      const cluster = asset.clusters[clusterId]!;
      const meshSelection = meshSelections.get(mesh.uuid)!;
      const previouslySelected = mesh.getPreviousFrontierClusterIds().includes(clusterId);

      const shouldExpand = cluster.childIds.length > 0
        && projectedSizePx > budgets.minProjectedNodeSizePx * (previouslySelected ? 1.12 : 1);

      if (shouldExpand) {
        candidates.push(
          ...cluster.childIds
            .map((childId) =>
              this.scoreCandidate(descriptor, childId, camera, viewportHeight, pageTable, budgets),
            )
            .filter((childCandidate): childCandidate is ClusterCandidate => childCandidate !== null),
        );
        continue;
      }

      const page = asset.pages[cluster.pageId]!;
      const projectedOverdraw = cluster.expectedOverdrawScore * Math.max(0.35, screenRadius / 72);
      const wouldOverflowPages = reservedPageSlots + 1 > budgets.maxActivePages;
      const wouldOverflowSplats = reservedVisibleSplats + page.splatCount > budgets.maxVisibleSplats;
      const wouldOverflowOverdraw = reservedOverdraw + projectedOverdraw > budgets.maxOverdrawBudget;

      if (wouldOverflowPages || wouldOverflowSplats || wouldOverflowOverdraw) {
        continue;
      }

      meshSelection.frontierClusterIds.push(clusterId);
      reservedPageSlots += 1;
      reservedVisibleSplats += page.splatCount;
      reservedOverdraw += projectedOverdraw;

      if (pageTable.isResident(page.id)) {
        pageTable.markTouched(page.id, frameIndex);
        meshSelection.activeClusters.push({
          clusterId,
          pageId: page.id,
          level: cluster.level,
        });
        meshSelection.activePageIds.push(page.id);
        meshSelection.visibleSplats += page.splatCount;
      } else {
        pageTable.request(page.id, frameIndex);
        meshSelection.requestedPageIds.push(page.id);
      }

      meshSelection.estimatedOverdraw += projectedOverdraw;
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
        const uploaded = descriptor.mesh
          .getPageTable()
          .serviceRequests(1, frameIndex, new Set(selection.activePageIds));

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
  ): ClusterCandidate | null {
    const mesh = descriptor.mesh;
    const asset = mesh.getAsset();
    const cluster = asset.clusters[clusterId]!;
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

    const projectedSizePx = this.getProjectedRadiusPx(camera, this.worldCenter, this.sphere.radius, viewportHeight);
    if (projectedSizePx <= 0.75) {
      return null;
    }

    this.projectedCenter.copy(this.worldCenter).project(camera);
    const radialDistance = Math.min(1.5, Math.hypot(this.projectedCenter.x, this.projectedCenter.y));
    const foveationWeight = Math.max(0.25, 1 - radialDistance * 0.25 * budgets.peripheralFoveation);
    const temporalWeight = mesh.getPreviousFrontierClusterIds().includes(clusterId)
      ? budgets.temporalStabilityBias
      : 1;
    const semanticWeight = hasSemanticFlag(cluster.semanticMask, 'hero') ? 1.35 : 1;
    const residencyWeight = pageTable.getState(cluster.pageId).state === SplatPageResidencyState.Resident
      ? 1.1
      : 0.92;
    const densityPenalty = 1 + cluster.expectedOverdrawScore / Math.max(1, budgets.maxOverdrawBudget);
    const score = (
      projectedSizePx
      * cluster.projectedErrorCoefficient
      * mesh.importance
      * foveationWeight
      * temporalWeight
      * semanticWeight
      * residencyWeight
    ) / densityPenalty;

    return {
      descriptor,
      clusterId,
      score,
      projectedSizePx,
      screenRadius: projectedSizePx,
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
