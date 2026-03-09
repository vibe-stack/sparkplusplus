export interface SparkEngineDescriptor {
  name: string;
  renderer: 'three-webgpu';
  stage: 'bootstrap';
}

export const SPARK_ENGINE_DESCRIPTOR: SparkEngineDescriptor = {
  name: 'Spark++',
  renderer: 'three-webgpu',
  stage: 'bootstrap',
};

export function getSparkBanner(): string {
  return `${SPARK_ENGINE_DESCRIPTOR.name} (${SPARK_ENGINE_DESCRIPTOR.renderer})`;
}