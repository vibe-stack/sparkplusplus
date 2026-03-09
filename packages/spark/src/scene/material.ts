import { Color, type ColorRepresentation } from 'three';

export type SplatDebugMode = 'albedo' | 'lod' | 'semantic';

export interface SplatMaterialOptions {
  pointSize?: number;
  opacity?: number;
  colorGain?: number;
  tint?: ColorRepresentation;
  debugMode?: SplatDebugMode;
}

export class SplatMaterial {
  pointSize: number;
  opacity: number;
  colorGain: number;
  readonly tint: Color;
  debugMode: SplatDebugMode;
  version = 0;

  constructor(options: SplatMaterialOptions = {}) {
    this.pointSize = options.pointSize ?? 6;
    this.opacity = options.opacity ?? 0.92;
    this.colorGain = options.colorGain ?? 1;
    this.tint = new Color(options.tint ?? 0xffffff);
    this.debugMode = options.debugMode ?? 'albedo';
  }

  update(options: SplatMaterialOptions): this {
    if (options.pointSize !== undefined) {
      this.pointSize = options.pointSize;
    }

    if (options.opacity !== undefined) {
      this.opacity = options.opacity;
    }

    if (options.colorGain !== undefined) {
      this.colorGain = options.colorGain;
    }

    if (options.tint !== undefined) {
      this.tint.set(options.tint);
    }

    if (options.debugMode !== undefined) {
      this.debugMode = options.debugMode;
    }

    this.version += 1;
    return this;
  }
}
