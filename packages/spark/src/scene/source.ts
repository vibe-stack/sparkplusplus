import type { SplatAsset } from '../assets/model';

export interface SplatSource {
  readonly id: string;
  readonly kind: string;
  buildAsset(): SplatAsset;
}
