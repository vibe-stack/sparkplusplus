export interface SparkEngineDescriptor {
  name: string;
  renderer: 'three-webgpu';
  stage: 'm1-bootstrap';
  implementedMilestones: readonly ['M0', 'M1'];
  scheduler: 'cpu-bootstrap-gpu-ready';
  compositor: 'debug-points-proxy';
}

export const SPARK_ENGINE_DESCRIPTOR = {
  name: 'Spark++',
  renderer: 'three-webgpu',
  stage: 'm1-bootstrap',
  implementedMilestones: ['M0', 'M1'],
  scheduler: 'cpu-bootstrap-gpu-ready',
  compositor: 'debug-points-proxy',
} as const satisfies SparkEngineDescriptor;

export function getSparkBanner(): string {
  return `${SPARK_ENGINE_DESCRIPTOR.name} (${SPARK_ENGINE_DESCRIPTOR.renderer})`;
}
