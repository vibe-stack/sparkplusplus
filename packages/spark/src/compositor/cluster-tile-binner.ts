import { Camera, OrthographicCamera, PerspectiveCamera, Vector3 } from 'three';
import type { SplatBudgetOptions } from '../core/budgets';
import {
  SPLAT_CLUSTER_SCREEN_DATA_FLOATS,
  SPLAT_TILE_CLUSTER_ENTRY_UINTS,
  SPLAT_TILE_HEADER_UINTS,
} from '../core/layouts';
import type { SplatMesh } from '../scene/splat-mesh';
import type { SplatActiveCluster } from '../scheduler/bootstrap-scheduler';

export interface SplatClusterTileBinnerOptions {
  tileSizePx?: number;
  maxClustersPerTile?: number;
  depthBucketCount?: number;
  tilePadding?: number;
}

export interface SplatProjectedCluster {
  clusterId: number;
  pageId: number;
  level: number;
  screenX: number;
  screenY: number;
  screenRadius: number;
  projectedDepth: number;
  minTileX: number;
  maxTileX: number;
  minTileY: number;
  maxTileY: number;
  dominantTileId: number;
  depthBucket: number;
  tileCount: number;
  priority: number;
  splatCount: number;
  representedSplatCount: number;
}

export interface SplatTileClusterEntry {
  clusterId: number;
  pageId: number;
  level: number;
  depthBucket: number;
  priority: number;
}

export interface SplatTileBin {
  tileId: number;
  tileX: number;
  tileY: number;
  clusterIds: readonly number[];
  depthBuckets: readonly (readonly number[])[];
  splatEstimate: number;
  overflowCount: number;
}

export interface SplatTileBinningBuffers {
  clusterScreenData: Float32Array;
  tileHeaders: Uint32Array;
  tileEntries: Uint32Array;
}

export interface SplatClusterTileBinningSnapshot {
  columns: number;
  rows: number;
  tileSizePx: number;
  depthBucketCount: number;
  visibleClusterCount: number;
  binnedClusterCount: number;
  binnedClusterReferences: number;
  activeTiles: number;
  overflowedTiles: number;
  overflowedClusterReferences: number;
  maxTileClusterCount: number;
  maxTileSplatEstimate: number;
  clusterOrder: readonly number[];
  clusterProjections: ReadonlyMap<number, SplatProjectedCluster>;
  clusterTileIds: ReadonlyMap<number, readonly number[]>;
  tileBins: readonly SplatTileBin[];
  buffers: SplatTileBinningBuffers;
  debugHash: number;
}

interface MutableTileEntry extends SplatTileClusterEntry {
  splatShare: number;
}

const EMPTY_BUFFERS: SplatTileBinningBuffers = {
  clusterScreenData: new Float32Array(0),
  tileHeaders: new Uint32Array(0),
  tileEntries: new Uint32Array(0),
};

const EMPTY_SNAPSHOT: SplatClusterTileBinningSnapshot = {
  columns: 0,
  rows: 0,
  tileSizePx: 16,
  depthBucketCount: 8,
  visibleClusterCount: 0,
  binnedClusterCount: 0,
  binnedClusterReferences: 0,
  activeTiles: 0,
  overflowedTiles: 0,
  overflowedClusterReferences: 0,
  maxTileClusterCount: 0,
  maxTileSplatEstimate: 0,
  clusterOrder: [],
  clusterProjections: new Map(),
  clusterTileIds: new Map(),
  tileBins: [],
  buffers: EMPTY_BUFFERS,
  debugHash: 2166136261,
};

export class SplatClusterTileBinner {
  private readonly worldCenter = new Vector3();
  private readonly screenPosition = new Vector3();
  private readonly axisScratchX = new Vector3();
  private readonly axisScratchY = new Vector3();
  private readonly axisScratchZ = new Vector3();

  binVisibleClusters(
    mesh: SplatMesh,
    activeClusters: readonly SplatActiveCluster[],
    camera: Camera,
    viewportWidth: number,
    viewportHeight: number,
    budgets: SplatBudgetOptions,
    options: SplatClusterTileBinnerOptions = {},
  ): SplatClusterTileBinningSnapshot {
    if (activeClusters.length === 0 || viewportWidth <= 0 || viewportHeight <= 0) {
      return EMPTY_SNAPSHOT;
    }

    const tileSizePx = Math.max(8, Math.round(options.tileSizePx ?? budgets.tileSizePx));
    const maxClustersPerTile = Math.max(4, Math.round(options.maxClustersPerTile ?? budgets.maxClustersPerTile));
    const depthBucketCount = Math.max(2, Math.round(options.depthBucketCount ?? budgets.tileDepthBucketCount));
    const tilePadding = Math.max(0, Math.round(options.tilePadding ?? budgets.clusterTilePadding));
    const columns = Math.max(1, Math.ceil(viewportWidth / tileSizePx));
    const rows = Math.max(1, Math.ceil(viewportHeight / tileSizePx));
    const tileCount = columns * rows;
    const asset = mesh.getAsset();
    const tileEntries = Array.from({ length: tileCount }, () => [] as MutableTileEntry[]);
    const tileSplatEstimate = new Uint32Array(tileCount);
    const tileOverflowCounts = new Uint32Array(tileCount);
    const tileActiveMask = new Uint8Array(tileCount);
    const projectedClusters = new Map<number, SplatProjectedCluster>();
    const clusterTileIds = new Map<number, readonly number[]>();
    const activeClusterIds = activeClusters.map((cluster) => cluster.clusterId);
    const activeClusterSet = new Set(activeClusterIds);
    let overflowedClusterReferences = 0;
    let visibleClusterCount = 0;
    let debugHash = 2166136261;

    for (const activeCluster of activeClusters) {
      const cluster = asset.clusters[activeCluster.clusterId]!;
      const page = asset.pages[activeCluster.pageId]!;
      const projection = this.projectCluster(
        mesh,
        cluster.boundsMin,
        cluster.boundsMax,
        camera,
        viewportWidth,
        viewportHeight,
        tileSizePx,
        tilePadding,
      );

      if (!projection) {
        continue;
      }

      visibleClusterCount += 1;
      const tileCoverage = Math.max(
        1,
        (projection.maxTileX - projection.minTileX + 1) * (projection.maxTileY - projection.minTileY + 1),
      );
      const dominantTileX = Math.min(columns - 1, Math.max(0, Math.floor(projection.screenX / tileSizePx)));
      const dominantTileY = Math.min(rows - 1, Math.max(0, Math.floor(projection.screenY / tileSizePx)));
      const dominantTileId = dominantTileY * columns + dominantTileX;
      const depthBucket = this.resolveDepthBucket(projection.depth, depthBucketCount);
      const priority = this.resolveTilePriority(
        projection.screenRadius,
        page.splatCount,
        cluster.representedSplatCount,
        tileCoverage,
      );
      const projectedCluster: SplatProjectedCluster = {
        clusterId: cluster.id,
        pageId: page.id,
        level: cluster.level,
        screenX: projection.screenX,
        screenY: projection.screenY,
        screenRadius: projection.screenRadius,
        projectedDepth: projection.depth,
        minTileX: projection.minTileX,
        maxTileX: projection.maxTileX,
        minTileY: projection.minTileY,
        maxTileY: projection.maxTileY,
        dominantTileId,
        depthBucket,
        tileCount: tileCoverage,
        priority,
        splatCount: page.splatCount,
        representedSplatCount: cluster.representedSplatCount,
      };

      projectedClusters.set(cluster.id, projectedCluster);
      debugHash = this.hashProjection(debugHash, projectedCluster);

      const splatShare = Math.max(1, Math.ceil(page.splatCount / tileCoverage));
      const coveredTileIds: number[] = [];

      for (let tileY = projection.minTileY; tileY <= projection.maxTileY; tileY += 1) {
        for (let tileX = projection.minTileX; tileX <= projection.maxTileX; tileX += 1) {
          const tileId = tileY * columns + tileX;
          coveredTileIds.push(tileId);
          tileActiveMask[tileId] = 1;
          tileSplatEstimate[tileId] = (tileSplatEstimate[tileId] ?? 0) + splatShare;
          const inserted = this.insertTileEntry(
            tileEntries[tileId]!,
            {
              clusterId: cluster.id,
              pageId: page.id,
              level: cluster.level,
              depthBucket,
              priority,
              splatShare,
            },
            maxClustersPerTile,
          );

          if (!inserted) {
            tileOverflowCounts[tileId] = (tileOverflowCounts[tileId] ?? 0) + 1;
            overflowedClusterReferences += 1;
          }
        }
      }

      clusterTileIds.set(cluster.id, coveredTileIds);
    }

    const tileBins: SplatTileBin[] = [];
    const tileHeaders = new Uint32Array(tileCount * SPLAT_TILE_HEADER_UINTS);
    let binnedClusterReferences = 0;
    let maxTileClusterCount = 0;
    let maxTileSplatEstimate = 0;
    let activeTileCount = 0;
    let overflowedTiles = 0;

    for (let tileId = 0; tileId < tileCount; tileId += 1) {
      const entries = tileEntries[tileId]!;

      if (tileActiveMask[tileId] !== 0) {
        activeTileCount += 1;
      }

      if (tileOverflowCounts[tileId]! > 0) {
        overflowedTiles += 1;
      }

      entries.sort((left, right) => {
        if (left.depthBucket !== right.depthBucket) {
          return right.depthBucket - left.depthBucket;
        }

        return right.priority - left.priority;
      });

      const depthBuckets = Array.from({ length: depthBucketCount }, () => [] as number[]);

      for (const entry of entries) {
        depthBuckets[entry.depthBucket]!.push(entry.clusterId);
      }

      const headerOffset = tileId * SPLAT_TILE_HEADER_UINTS;
      tileHeaders[headerOffset + 0] = binnedClusterReferences;
      tileHeaders[headerOffset + 1] = entries.length;
      tileHeaders[headerOffset + 2] = tileSplatEstimate[tileId]!;
      tileHeaders[headerOffset + 3] = tileOverflowCounts[tileId]!;

      maxTileClusterCount = Math.max(maxTileClusterCount, entries.length);
      maxTileSplatEstimate = Math.max(maxTileSplatEstimate, tileSplatEstimate[tileId]!);
      binnedClusterReferences += entries.length;

      if (entries.length > 0) {
        tileBins.push({
          tileId,
          tileX: tileId % columns,
          tileY: Math.floor(tileId / columns),
          clusterIds: entries.map((entry) => entry.clusterId),
          depthBuckets,
          splatEstimate: tileSplatEstimate[tileId]!,
          overflowCount: tileOverflowCounts[tileId]!,
        });
      }
    }

    const tileEntriesBuffer = new Uint32Array(binnedClusterReferences * SPLAT_TILE_CLUSTER_ENTRY_UINTS);
    let entryCursor = 0;

    for (let tileId = 0; tileId < tileCount; tileId += 1) {
      const entries = tileEntries[tileId]!;

      for (const entry of entries) {
        const offset = entryCursor * SPLAT_TILE_CLUSTER_ENTRY_UINTS;
        tileEntriesBuffer[offset + 0] = entry.clusterId;
        tileEntriesBuffer[offset + 1] = entry.pageId;
        tileEntriesBuffer[offset + 2] = entry.depthBucket;
        tileEntriesBuffer[offset + 3] = Math.min(65535, Math.round(entry.priority * 64));
        entryCursor += 1;
      }
    }

    const clusterScreenData = new Float32Array(visibleClusterCount * SPLAT_CLUSTER_SCREEN_DATA_FLOATS);
    let clusterCursor = 0;
    let binnedClusterCount = 0;
    const binnedClusterSet = new Set<number>();

    for (const projectedCluster of projectedClusters.values()) {
      const offset = clusterCursor * SPLAT_CLUSTER_SCREEN_DATA_FLOATS;
      clusterScreenData[offset + 0] = projectedCluster.screenX;
      clusterScreenData[offset + 1] = projectedCluster.screenY;
      clusterScreenData[offset + 2] = projectedCluster.screenRadius;
      clusterScreenData[offset + 3] = projectedCluster.projectedDepth;
      clusterScreenData[offset + 4] = projectedCluster.minTileX;
      clusterScreenData[offset + 5] = projectedCluster.minTileY;
      clusterScreenData[offset + 6] = projectedCluster.maxTileX;
      clusterScreenData[offset + 7] = projectedCluster.maxTileY;
      clusterCursor += 1;
    }

    const clusterOrder: number[] = [];
    const seenClusterIds = new Set<number>();

    for (const tile of tileBins) {
      for (let bucketIndex = depthBucketCount - 1; bucketIndex >= 0; bucketIndex -= 1) {
        for (const clusterId of tile.depthBuckets[bucketIndex]!) {
          if (!activeClusterSet.has(clusterId) || seenClusterIds.has(clusterId)) {
            continue;
          }

          seenClusterIds.add(clusterId);
          clusterOrder.push(clusterId);
        }
      }

      for (const clusterId of tile.clusterIds) {
        binnedClusterSet.add(clusterId);
      }
    }

    binnedClusterCount = binnedClusterSet.size;

    for (const clusterId of activeClusterIds) {
      if (seenClusterIds.has(clusterId)) {
        continue;
      }

      seenClusterIds.add(clusterId);
      clusterOrder.push(clusterId);
    }

    return {
      columns,
      rows,
      tileSizePx,
      depthBucketCount,
      visibleClusterCount,
      binnedClusterCount,
      binnedClusterReferences,
      activeTiles: activeTileCount,
      overflowedTiles,
      overflowedClusterReferences,
      maxTileClusterCount,
      maxTileSplatEstimate,
      clusterOrder,
      clusterProjections: projectedClusters,
      clusterTileIds,
      tileBins,
      buffers: {
        clusterScreenData,
        tileHeaders,
        tileEntries: tileEntriesBuffer,
      },
      debugHash,
    };
  }

  private insertTileEntry(
    entries: MutableTileEntry[],
    nextEntry: MutableTileEntry,
    maxClustersPerTile: number,
  ): boolean {
    if (entries.some((entry) => entry.clusterId === nextEntry.clusterId)) {
      return true;
    }

    if (entries.length < maxClustersPerTile) {
      entries.push(nextEntry);
      return true;
    }

    let weakestIndex = 0;
    let weakestPriority = entries[0]!.priority;

    for (let index = 1; index < entries.length; index += 1) {
      if (entries[index]!.priority < weakestPriority) {
        weakestPriority = entries[index]!.priority;
        weakestIndex = index;
      }
    }

    if (nextEntry.priority <= weakestPriority) {
      return false;
    }

    entries[weakestIndex] = nextEntry;
    return true;
  }

  private resolveTilePriority(
    screenRadius: number,
    splatCount: number,
    representedSplatCount: number,
    tileCoverage: number,
  ): number {
    const densityBoost = Math.log2(Math.max(2, representedSplatCount)) * 0.16;
    const splatShare = splatCount / Math.max(1, tileCoverage);
    return screenRadius * (1 + densityBoost) + Math.log2(Math.max(2, splatShare)) * 3.5;
  }

  private resolveDepthBucket(depth: number, depthBucketCount: number): number {
    const normalizedDepth = Math.min(1, Math.max(0, depth * 0.5 + 0.5));
    return Math.min(depthBucketCount - 1, Math.floor(normalizedDepth * depthBucketCount));
  }

  private hashProjection(seed: number, projection: SplatProjectedCluster): number {
    let hash = seed >>> 0;
    hash = Math.imul(hash ^ projection.clusterId, 16777619);
    hash = Math.imul(hash ^ projection.dominantTileId, 16777619);
    hash = Math.imul(hash ^ projection.depthBucket, 16777619);
    hash = Math.imul(hash ^ projection.tileCount, 16777619);
    return hash >>> 0;
  }

  private projectCluster(
    mesh: SplatMesh,
    boundsMin: readonly [number, number, number],
    boundsMax: readonly [number, number, number],
    camera: Camera,
    viewportWidth: number,
    viewportHeight: number,
    tileSizePx: number,
    tilePadding: number,
  ): {
    screenX: number;
    screenY: number;
    screenRadius: number;
    minTileX: number;
    maxTileX: number;
    minTileY: number;
    maxTileY: number;
    depth: number;
  } | null {
    const centerX = (boundsMin[0] + boundsMax[0]) * 0.5;
    const centerY = (boundsMin[1] + boundsMax[1]) * 0.5;
    const centerZ = (boundsMin[2] + boundsMax[2]) * 0.5;
    const halfExtentX = Math.max(1e-5, (boundsMax[0] - boundsMin[0]) * 0.5);
    const halfExtentY = Math.max(1e-5, (boundsMax[1] - boundsMin[1]) * 0.5);
    const halfExtentZ = Math.max(1e-5, (boundsMax[2] - boundsMin[2]) * 0.5);

    this.worldCenter.set(centerX, centerY, centerZ).applyMatrix4(mesh.matrixWorld);
    this.screenPosition.copy(this.worldCenter).project(camera);

    const screenX = (this.screenPosition.x * 0.5 + 0.5) * viewportWidth;
    const screenY = (this.screenPosition.y * -0.5 + 0.5) * viewportHeight;
    const screenRadius = this.getProjectedExtentRadiusPx(
      camera,
      viewportWidth,
      viewportHeight,
      centerX,
      centerY,
      centerZ,
      halfExtentX,
      halfExtentY,
      halfExtentZ,
      mesh,
      screenX,
      screenY,
    );

    if (
      screenRadius <= 0.5
      || screenX + screenRadius < 0
      || screenY + screenRadius < 0
      || screenX - screenRadius > viewportWidth
      || screenY - screenRadius > viewportHeight
    ) {
      return null;
    }

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
      screenX,
      screenY,
      screenRadius,
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

  private getProjectedExtentRadiusPx(
    camera: Camera,
    viewportWidth: number,
    viewportHeight: number,
    centerX: number,
    centerY: number,
    centerZ: number,
    halfExtentX: number,
    halfExtentY: number,
    halfExtentZ: number,
    mesh: SplatMesh,
    screenX: number,
    screenY: number,
  ): number {
    this.axisScratchX.set(centerX + halfExtentX, centerY, centerZ).applyMatrix4(mesh.matrixWorld).project(camera);
    this.axisScratchY.set(centerX, centerY + halfExtentY, centerZ).applyMatrix4(mesh.matrixWorld).project(camera);
    this.axisScratchZ.set(centerX, centerY, centerZ + halfExtentZ).applyMatrix4(mesh.matrixWorld).project(camera);

    const projectedRadiusX = Math.hypot(
      (this.axisScratchX.x * 0.5 + 0.5) * viewportWidth - screenX,
      (this.axisScratchX.y * -0.5 + 0.5) * viewportHeight - screenY,
    );
    const projectedRadiusY = Math.hypot(
      (this.axisScratchY.x * 0.5 + 0.5) * viewportWidth - screenX,
      (this.axisScratchY.y * -0.5 + 0.5) * viewportHeight - screenY,
    );
    const projectedRadiusZ = Math.hypot(
      (this.axisScratchZ.x * 0.5 + 0.5) * viewportWidth - screenX,
      (this.axisScratchZ.y * -0.5 + 0.5) * viewportHeight - screenY,
    );

    const extentRadius = Math.max(projectedRadiusX, projectedRadiusY, projectedRadiusZ);

    if (Number.isFinite(extentRadius) && extentRadius > 0.5) {
      return extentRadius;
    }

    return this.getProjectedRadiusPx(
      camera,
      this.worldCenter,
      Math.hypot(halfExtentX, halfExtentY, halfExtentZ),
      viewportHeight,
    );
  }
}
