export const SPLAT_SCENE_DESCRIPTOR_FLOATS = 24;
export const SPLAT_CLUSTER_METADATA_FLOATS = 16;
export const SPLAT_CLUSTER_REFERENCE_UINTS = 4;
export const SPLAT_PAGE_DESCRIPTOR_UINTS = 8;
export const SPLAT_PAGE_RESIDENCY_UINTS = 5;

export interface SplatPackedSceneBuffer {
  descriptors: Float32Array;
}
