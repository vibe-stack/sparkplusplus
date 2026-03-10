import { Camera, OrthographicCamera, PerspectiveCamera, Vector3 } from 'three';
import type { SplatBudgetOptions } from '../core/budgets';
import type { SplatMesh } from '../scene/splat-mesh';
import type { SplatActiveCluster } from '../scheduler/bootstrap-scheduler';
import { hasSemanticFlag } from '../core/semantics';

export type SplatTileMode = 'weighted' | 'depth-sliced' | 'hero';

export interface SplatTileClassification {
  clusterModes: Map<number, SplatTileMode>;
  activeTiles: number;
  weightedTiles: number;
  depthSlicedTiles: number;
  heroTiles: number;
  maxTileComplexity: number;
}

export class SplatTileClassifier {
  private readonly worldCenter = new Vector3();
  private readonly screenPosition = new Vector3();
  private previousColumns = 0;
  private previousRows = 0;
  private previousTileSizePx = 0;
  private previousTileModes: SplatTileMode[] = [];
  private previousClusterModes = new Map<number, SplatTileMode>();

  classify(
    mesh: SplatMesh,
    activeClusters: readonly SplatActiveCluster[],
    camera: Camera,
    viewportWidth: number,
    viewportHeight: number,
    budgets: SplatBudgetOptions,
    tileSizePx = 72,
  ): SplatTileClassification {
    const asset = mesh.getAsset();
    const columns = Math.max(1, Math.ceil(viewportWidth / tileSizePx));
    const rows = Math.max(1, Math.ceil(viewportHeight / tileSizePx));
    const tileCount = columns * rows;
    const reusedTileHistory = this.previousColumns === columns
      && this.previousRows === rows
      && this.previousTileSizePx === tileSizePx
      && this.previousTileModes.length === tileCount;
    const tileComplexity = new Float32Array(tileCount);
    const tileHeroScore = new Float32Array(tileCount);
    const tileModes = new Array<SplatTileMode>(tileCount).fill('weighted');
    const clusterTileIds = new Map<number, number[]>();
    const heroCandidates: Array<{ tileId: number; score: number }> = [];
    let maxTileComplexity = 0;

    for (const activeCluster of activeClusters) {
      const cluster = asset.clusters[activeCluster.clusterId]!;
      const page = asset.pages[activeCluster.pageId]!;
      const projection = this.projectCluster(
        mesh,
        cluster.center,
        cluster.radius,
        camera,
        viewportWidth,
        viewportHeight,
        tileSizePx,
      );

      if (!projection) {
        clusterTileIds.set(cluster.id, []);
        continue;
      }

      const coveredTileCount = Math.max(
        1,
        (projection.maxTileX - projection.minTileX + 1) * (projection.maxTileY - projection.minTileY + 1),
      );
      const tileIds: number[] = [];
      const contribution = Math.max(1, page.splatCount / coveredTileCount);
      const heroBoost = hasSemanticFlag(cluster.semanticMask, 'hero') ? 1.4 : 0;

      for (let tileY = projection.minTileY; tileY <= projection.maxTileY; tileY += 1) {
        for (let tileX = projection.minTileX; tileX <= projection.maxTileX; tileX += 1) {
          const tileId = tileY * columns + tileX;
          tileIds.push(tileId);
          tileComplexity[tileId]! += contribution;

          const tileCenterX = (tileX + 0.5) * tileSizePx;
          const tileCenterY = (tileY + 0.5) * tileSizePx;
          const centerBias = 1 - Math.min(
            1,
            Math.hypot(tileCenterX - viewportWidth * 0.5, tileCenterY - viewportHeight * 0.5)
              / Math.max(viewportWidth, viewportHeight),
          );

          const depthBias = 1 - Math.min(1, (projection.depth + 1) * 0.5);
          tileHeroScore[tileId]! += heroBoost + centerBias * contribution * 0.08 + depthBias * contribution * 0.04;
          maxTileComplexity = Math.max(maxTileComplexity, tileComplexity[tileId]!);
        }
      }

      clusterTileIds.set(cluster.id, tileIds);
    }

    const activeTileIds: number[] = [];

    for (let tileId = 0; tileId < tileCount; tileId += 1) {
      if (tileComplexity[tileId]! > 0) {
        activeTileIds.push(tileId);
      }
    }

    const mediumThreshold = Math.max(
      20,
      budgets.maxVisibleSplats / Math.max(1, activeTileIds.length) * 0.55,
    );
    const depthSlicedEnterThreshold = mediumThreshold * 1.08;
    const depthSlicedExitThreshold = mediumThreshold * 0.82;

    for (const tileId of activeTileIds) {
      const previousTileMode = reusedTileHistory ? this.previousTileModes[tileId] ?? 'weighted' : 'weighted';
      heroCandidates.push({
        tileId,
        score: tileHeroScore[tileId]!
          + tileComplexity[tileId]!
          + (previousTileMode === 'hero' ? mediumThreshold * 0.18 : 0),
      });

      const complexity = tileComplexity[tileId]!;
      const shouldDepthSlice = previousTileMode === 'depth-sliced'
        ? complexity > depthSlicedExitThreshold
        : complexity > depthSlicedEnterThreshold;

      if (shouldDepthSlice) {
        tileModes[tileId] = 'depth-sliced';
      }
    }

    heroCandidates.sort((left, right) => right.score - left.score);

    for (let index = 0; index < Math.min(budgets.heroTileBudget, heroCandidates.length); index += 1) {
      tileModes[heroCandidates[index]!.tileId] = 'hero';
    }

    const clusterModes = new Map<number, SplatTileMode>();

    clusterTileIds.forEach((tileIds, clusterId) => {
      const previousMode = this.previousClusterModes.get(clusterId) ?? 'weighted';
      let heroWeight = 0;
      let depthSlicedWeight = 0;
      let totalWeight = 0;

      for (const tileId of tileIds) {
        const tileMode = tileModes[tileId]!;
        const weight = tileComplexity[tileId]!;
        totalWeight += weight;

        if (tileMode === 'hero') {
          heroWeight += weight;
        } else if (tileMode === 'depth-sliced') {
          depthSlicedWeight += weight;
        }
      }

      let mode: SplatTileMode = 'weighted';
      const heroThreshold = previousMode === 'hero' ? 0.18 : 0.32;
      const depthThreshold = previousMode === 'depth-sliced' ? 0.22 : 0.44;

      if (totalWeight > 0 && heroWeight >= totalWeight * heroThreshold) {
        mode = 'hero';
      } else if (totalWeight > 0 && depthSlicedWeight >= totalWeight * depthThreshold) {
        mode = 'depth-sliced';
      }

      clusterModes.set(clusterId, mode);
    });

    let weightedTiles = 0;
    let depthSlicedTiles = 0;
    let heroTiles = 0;

    for (const tileId of activeTileIds) {
      switch (tileModes[tileId]) {
        case 'hero':
          heroTiles += 1;
          break;
        case 'depth-sliced':
          depthSlicedTiles += 1;
          break;
        default:
          weightedTiles += 1;
          break;
      }
    }

    this.previousColumns = columns;
    this.previousRows = rows;
    this.previousTileSizePx = tileSizePx;
    this.previousTileModes = tileModes;
    this.previousClusterModes = clusterModes;

    return {
      clusterModes,
      activeTiles: activeTileIds.length,
      weightedTiles,
      depthSlicedTiles,
      heroTiles,
      maxTileComplexity,
    };
  }

  private projectCluster(
    mesh: SplatMesh,
    center: readonly [number, number, number],
    radius: number,
    camera: Camera,
    viewportWidth: number,
    viewportHeight: number,
    tileSizePx: number,
  ): {
    minTileX: number;
    maxTileX: number;
    minTileY: number;
    maxTileY: number;
    depth: number;
  } | null {
    const matrixWorld = mesh.matrixWorld.elements;
    const maxScale = Math.max(
      Math.hypot(matrixWorld[0]!, matrixWorld[1]!, matrixWorld[2]!),
      Math.hypot(matrixWorld[4]!, matrixWorld[5]!, matrixWorld[6]!),
      Math.hypot(matrixWorld[8]!, matrixWorld[9]!, matrixWorld[10]!),
    );

    this.worldCenter.set(center[0], center[1], center[2]).applyMatrix4(mesh.matrixWorld);
    this.screenPosition.copy(this.worldCenter).project(camera);

    const screenX = (this.screenPosition.x * 0.5 + 0.5) * viewportWidth;
    const screenY = (this.screenPosition.y * -0.5 + 0.5) * viewportHeight;
    const screenRadius = this.getProjectedRadiusPx(
      camera,
      this.worldCenter,
      radius * maxScale,
      viewportHeight,
    );

    if (screenRadius <= 0.5) {
      return null;
    }

    // Expand coverage by one tile so clusters near a tile boundary or with
    // centers just outside clip space still keep the adjacent visible tiles.
    const tilePadding = 1;
    const minTileX = Math.max(0, Math.floor((screenX - screenRadius) / tileSizePx) - tilePadding);
    const maxTileX = Math.min(
      Math.ceil(viewportWidth / tileSizePx) - 1,
      Math.floor((screenX + screenRadius) / tileSizePx) + tilePadding,
    );
    const minTileY = Math.max(0, Math.floor((screenY - screenRadius) / tileSizePx) - tilePadding);
    const maxTileY = Math.min(
      Math.ceil(viewportHeight / tileSizePx) - 1,
      Math.floor((screenY + screenRadius) / tileSizePx) + tilePadding,
    );

    return {
      minTileX,
      maxTileX,
      minTileY,
      maxTileY,
      depth: this.screenPosition.z,
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
