import { Color, type ColorRepresentation } from 'three';

export type SplatEffectKind = 'recolor' | 'pulse' | 'dissolve';

export interface SplatEffectDescriptor {
  id: string;
  kind: SplatEffectKind;
  enabled?: boolean;
  intensity?: number;
  frequencyHz?: number;
  tint?: ColorRepresentation;
  semanticMask?: number;
}

export interface SplatResolvedEffects {
  tint: Color;
  opacityMultiplier: number;
  pointSizeMultiplier: number;
}

export class SplatEffectStack {
  private readonly effects: SplatEffectDescriptor[] = [];

  version = 0;

  constructor(effects: readonly SplatEffectDescriptor[] = []) {
    this.effects.push(...effects);
  }

  addEffect(effect: SplatEffectDescriptor): this {
    this.effects.push(effect);
    this.version += 1;
    return this;
  }

  evaluate(semanticMask: number, timeSeconds: number): SplatResolvedEffects {
    const tint = new Color(1, 1, 1);
    let opacityMultiplier = 1;
    let pointSizeMultiplier = 1;

    for (const effect of this.effects) {
      if (effect.enabled === false) {
        continue;
      }

      if (effect.semanticMask !== undefined && (effect.semanticMask & semanticMask) === 0) {
        continue;
      }

      const intensity = effect.intensity ?? 0.5;

      switch (effect.kind) {
        case 'recolor': {
          tint.multiply(new Color(effect.tint ?? 0xffffff).multiplyScalar(1 + intensity * 0.2));
          break;
        }
        case 'pulse': {
          const wave = Math.sin(timeSeconds * (effect.frequencyHz ?? 0.65) * Math.PI * 2);
          pointSizeMultiplier *= 1 + wave * intensity * 0.18;
          opacityMultiplier *= 1 + wave * intensity * 0.1;
          break;
        }
        case 'dissolve': {
          opacityMultiplier *= Math.max(0.15, 1 - intensity);
          break;
        }
      }
    }

    return {
      tint,
      opacityMultiplier,
      pointSizeMultiplier,
    };
  }
}
