export type SplatFramePassName =
  | 'scene-descriptor-upload'
  | 'object-cluster-update'
  | 'cluster-cull-score'
  | 'lod-frontier-selection'
  | 'residency-request-emission'
  | 'cluster-visible-list-compaction'
  | 'cluster-tile-binning'
  | 'tile-local-splat-expansion'
  | 'tile-local-depth-composite';

export interface SplatFramePassDescriptor {
  name: SplatFramePassName;
  stage: 'cpu-bootstrap' | 'gpu-visibility' | 'tile-compositor';
  inputs: readonly string[];
  outputs: readonly string[];
}

export const SPARK_FRAME_GRAPH: readonly SplatFramePassDescriptor[] = [
  {
    name: 'scene-descriptor-upload',
    stage: 'cpu-bootstrap',
    inputs: ['THREE.Scene', 'SplatMesh dirty transforms', 'SplatMaterial versions'],
    outputs: ['packed scene descriptor buffer'],
  },
  {
    name: 'object-cluster-update',
    stage: 'gpu-visibility',
    inputs: ['scene descriptor buffer', 'cluster metadata buffer'],
    outputs: ['world-space cluster bounds', 'object-level score seeds'],
  },
  {
    name: 'cluster-cull-score',
    stage: 'gpu-visibility',
    inputs: ['camera frustum', 'cluster bounds', 'temporal cache'],
    outputs: ['visible cluster candidates', 'priority scores'],
  },
  {
    name: 'lod-frontier-selection',
    stage: 'cpu-bootstrap',
    inputs: ['cluster candidates', 'global page/splat/overdraw budgets'],
    outputs: ['active frontier clusters', 'planned page slots'],
  },
  {
    name: 'residency-request-emission',
    stage: 'cpu-bootstrap',
    inputs: ['active frontier clusters', 'page table'],
    outputs: ['requested page queue', 'resident page set'],
  },
  {
    name: 'cluster-visible-list-compaction',
    stage: 'cpu-bootstrap',
    inputs: ['resident pages', 'active frontier clusters'],
    outputs: ['visible cluster list', 'compatible page indirection'],
  },
  {
    name: 'cluster-tile-binning',
    stage: 'tile-compositor',
    inputs: ['visible cluster list', 'camera/viewport'],
    outputs: ['cluster screen bounds', 'tile headers', 'tile cluster entries'],
  },
  {
    name: 'tile-local-splat-expansion',
    stage: 'tile-compositor',
    inputs: ['tile cluster entries', 'resident pages', 'material/effect stack'],
    outputs: ['tile-local splat work queues', 'compatibility sprite instances'],
  },
  {
    name: 'tile-local-depth-composite',
    stage: 'tile-compositor',
    inputs: ['tile-local splat work queues', 'depth buckets'],
    outputs: ['weighted/hero compositor instances', 'tile debug telemetry'],
  },
] as const;
