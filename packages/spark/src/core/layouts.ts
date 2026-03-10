export const SPLAT_SCENE_DESCRIPTOR_FLOATS = 24;
export const SPLAT_CLUSTER_METADATA_FLOATS = 16;
export const SPLAT_CLUSTER_REFERENCE_UINTS = 4;
export const SPLAT_PAGE_DESCRIPTOR_UINTS = 8;
export const SPLAT_PAGE_RESIDENCY_UINTS = 5;
// Screen-space cluster data mirrors the planned per-frame GPU cull/bin stage:
// [screenX, screenY, screenRadiusPx, ndcDepth, minTileX, minTileY, maxTileX, maxTileY]
export const SPLAT_CLUSTER_SCREEN_DATA_FLOATS = 8;
// Per-tile headers map a tile to a compact slice of tileEntries:
// [entryOffset, entryCount, estimatedSplats, overflowCount]
export const SPLAT_TILE_HEADER_UINTS = 4;
// Tile entries carry the cluster/page indirection needed by a future GPU tile
// expansion pass: [clusterId, pageId, depthBucket, quantizedPriority]
export const SPLAT_TILE_CLUSTER_ENTRY_UINTS = 4;
// Compute-time tile headers mirror the compact tile graph used by the
// compositor resolve pass:
// [entryOffset, entryCount, actualTileWorkItems, overflowCount]
export const SPLAT_COMPUTE_TILE_HEADER_FLOATS = 4;
// Compute tile entries carry the per-tile splat expansion indirection:
// [pageSplatOffset, pageSplatCount, samplingStride, scaleMultiplier]
export const SPLAT_COMPUTE_TILE_ENTRY_FLOATS = 4;

export interface SplatPackedSceneBuffer {
  descriptors: Float32Array;
}
