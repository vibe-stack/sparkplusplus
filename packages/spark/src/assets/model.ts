import {
  SPLAT_CLUSTER_METADATA_FLOATS,
  SPLAT_CLUSTER_REFERENCE_UINTS,
  SPLAT_PAGE_DESCRIPTOR_UINTS,
} from '../core/layouts';

export type SplatVec3 = [number, number, number];

export interface SplatPage {
  id: number;
  clusterId: number;
  level: number;
  splatCount: number;
  capacity: number;
  semanticMask: number;
  byteSize: number;
  positions: Float32Array;
  scales: Float32Array;
  colors: Float32Array;
  opacities: Float32Array;
}

export interface SplatClusterNode {
  id: number;
  pageId: number;
  level: number;
  splatCount: number;
  representedSplatCount: number;
  center: SplatVec3;
  radius: number;
  boundsMin: SplatVec3;
  boundsMax: SplatVec3;
  projectedErrorCoefficient: number;
  opacityMass: number;
  anisotropySeverity: number;
  expectedOverdrawScore: number;
  motionSensitivity: number;
  semanticMask: number;
  parentId: number | null;
  childIds: number[];
}

export interface SplatRuntimeBuffers {
  clusterMetadata: Float32Array;
  clusterReferences: Uint32Array;
  childIndices: Uint32Array;
  pageDescriptors: Uint32Array;
}

export interface SplatAsset {
  id: string;
  label: string;
  pageCapacity: number;
  maxDepth: number;
  totalSplatCount: number;
  rootClusterIds: number[];
  localBoundsMin: SplatVec3;
  localBoundsMax: SplatVec3;
  clusters: SplatClusterNode[];
  pages: SplatPage[];
  buffers: SplatRuntimeBuffers;
}

export interface CreateSplatAssetOptions {
  id: string;
  label: string;
  pageCapacity: number;
  maxDepth: number;
  rootClusterIds: number[];
  localBoundsMin: SplatVec3;
  localBoundsMax: SplatVec3;
  clusters: SplatClusterNode[];
  pages: SplatPage[];
}

export function createSplatAsset(options: CreateSplatAssetOptions): SplatAsset {
  const totalSplatCount = options.pages.reduce((sum, page) => sum + page.splatCount, 0);

  return {
    ...options,
    totalSplatCount,
    buffers: createSplatRuntimeBuffers(options.clusters, options.pages),
  };
}

export function createSplatRuntimeBuffers(
  clusters: readonly SplatClusterNode[],
  pages: readonly SplatPage[],
): SplatRuntimeBuffers {
  const clusterMetadata = new Float32Array(clusters.length * SPLAT_CLUSTER_METADATA_FLOATS);
  const clusterReferences = new Uint32Array(clusters.length * SPLAT_CLUSTER_REFERENCE_UINTS);
  const childIndices = new Uint32Array(
    clusters.reduce((sum, cluster) => sum + cluster.childIds.length, 0),
  );
  const pageDescriptors = new Uint32Array(pages.length * SPLAT_PAGE_DESCRIPTOR_UINTS);

  let childCursor = 0;

  clusters.forEach((cluster, clusterIndex) => {
    const metadataOffset = clusterIndex * SPLAT_CLUSTER_METADATA_FLOATS;
    clusterMetadata[metadataOffset + 0] = cluster.center[0];
    clusterMetadata[metadataOffset + 1] = cluster.center[1];
    clusterMetadata[metadataOffset + 2] = cluster.center[2];
    clusterMetadata[metadataOffset + 3] = cluster.radius;
    clusterMetadata[metadataOffset + 4] = cluster.boundsMin[0];
    clusterMetadata[metadataOffset + 5] = cluster.boundsMin[1];
    clusterMetadata[metadataOffset + 6] = cluster.boundsMin[2];
    clusterMetadata[metadataOffset + 7] = cluster.projectedErrorCoefficient;
    clusterMetadata[metadataOffset + 8] = cluster.boundsMax[0];
    clusterMetadata[metadataOffset + 9] = cluster.boundsMax[1];
    clusterMetadata[metadataOffset + 10] = cluster.boundsMax[2];
    clusterMetadata[metadataOffset + 11] = cluster.opacityMass;
    clusterMetadata[metadataOffset + 12] = cluster.anisotropySeverity;
    clusterMetadata[metadataOffset + 13] = cluster.expectedOverdrawScore;
    clusterMetadata[metadataOffset + 14] = cluster.motionSensitivity;
    clusterMetadata[metadataOffset + 15] = cluster.semanticMask;

    const referenceOffset = clusterIndex * SPLAT_CLUSTER_REFERENCE_UINTS;
    clusterReferences[referenceOffset + 0] = cluster.pageId;
    clusterReferences[referenceOffset + 1] = cluster.parentId === null ? 0 : cluster.parentId + 1;
    clusterReferences[referenceOffset + 2] = childCursor;
    clusterReferences[referenceOffset + 3] = cluster.childIds.length;

    childIndices.set(cluster.childIds, childCursor);
    childCursor += cluster.childIds.length;
  });

  pages.forEach((page, pageIndex) => {
    const offset = pageIndex * SPLAT_PAGE_DESCRIPTOR_UINTS;
    pageDescriptors[offset + 0] = page.clusterId;
    pageDescriptors[offset + 1] = page.level;
    pageDescriptors[offset + 2] = page.splatCount;
    pageDescriptors[offset + 3] = page.capacity;
    pageDescriptors[offset + 4] = page.semanticMask;
    pageDescriptors[offset + 5] = page.byteSize;
    pageDescriptors[offset + 6] = 0;
    pageDescriptors[offset + 7] = 0;
  });

  return {
    clusterMetadata,
    clusterReferences,
    childIndices,
    pageDescriptors,
  };
}
