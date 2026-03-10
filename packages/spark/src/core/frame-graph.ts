export type SplatFramePassName =
  | 'scene-descriptor-upload'
  | 'object-cluster-update'
  | 'cluster-cull-score'
  | 'lod-frontier-selection'
  | 'residency-request-emission'
  | 'active-page-expansion'
  | 'tile-compositor-sync';

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
    name: 'active-page-expansion',
    stage: 'cpu-bootstrap',
    inputs: ['resident pages', 'active frontier clusters'],
    outputs: ['active page list', 'renderable splat count'],
  },
  {
    name: 'tile-compositor-sync',
    stage: 'tile-compositor',
    inputs: ['active page list', 'material/effect stack'],
    outputs: ['sprite tile queues', 'weighted/hero compositor instances'],
  },
] as const;
