import { SplatMesh, type SplatMeshOptions } from './splat-mesh';

export interface SplatSkeletonBinding {
  jointCount: number;
  label?: string;
}

export class SplatSkinnedMesh extends SplatMesh {
  skeletonBinding?: SplatSkeletonBinding;

  constructor(options: SplatMeshOptions) {
    super(options);
  }

  setSkeletonBinding(binding: SplatSkeletonBinding): this {
    this.skeletonBinding = binding;
    return this;
  }
}
