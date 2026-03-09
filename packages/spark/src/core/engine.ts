export interface SparkEngineDescriptor {
  name: string;
  renderer: 'three-webgpu';
  stage: 'm3-baseline';
  implementedMilestones: readonly ['M0', 'M1', 'M2', 'M3'];
  scheduler: 'hybrid-gpu-visibility';
  compositor: 'sprite-tile-baseline';
}

export const SPARK_ENGINE_DESCRIPTOR = {
  name: 'Spark++',
  renderer: 'three-webgpu',
  stage: 'm3-baseline',
  implementedMilestones: ['M0', 'M1', 'M2', 'M3'],
  scheduler: 'hybrid-gpu-visibility',
  compositor: 'sprite-tile-baseline',
} as const satisfies SparkEngineDescriptor;

export function getSparkBanner(): string {
  return `${SPARK_ENGINE_DESCRIPTOR.name} (${SPARK_ENGINE_DESCRIPTOR.renderer})`;
}
