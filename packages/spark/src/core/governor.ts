import { cloneBudgets, DEFAULT_SPLAT_BUDGETS, type SplatBudgetOptions } from './budgets';
import type { SplatFrameStatsSnapshot } from './stats';

export interface SplatQualityDecision {
  level: number;
  reason: string;
  budgets: SplatBudgetOptions;
}

export class SplatQualityGovernor {
  readonly targetFrameMs: number;

  private readonly baseBudgets: SplatBudgetOptions;
  private level = 0;
  private reason = 'bootstrap';
  private stableFrames = 0;

  constructor(
    baseBudgets: Partial<SplatBudgetOptions> = {},
    targetFrameMs = 16.7,
  ) {
    this.baseBudgets = cloneBudgets({
      ...DEFAULT_SPLAT_BUDGETS,
      ...baseBudgets,
    });
    this.targetFrameMs = targetFrameMs;
  }

  getLevel(): number {
    return this.level;
  }

  getReason(): string {
    return this.reason;
  }

  getBudgets(): SplatBudgetOptions {
    return this.applyLevel(this.level);
  }

  observe(snapshot: SplatFrameStatsSnapshot): SplatQualityDecision {
    const overloadedFrame = snapshot.cpuFrameMs > this.targetFrameMs * 1.15;
    const pageFaultSpike = snapshot.pageFaultRate > 0.35;
    const overdrawPressure = snapshot.estimatedOverdraw > snapshot.budgets.maxOverdrawBudget * 0.92;

    if (overloadedFrame || pageFaultSpike || overdrawPressure) {
      this.level = Math.min(this.level + 1, 7);
      this.reason = overloadedFrame
        ? 'cpu frame pressure'
        : pageFaultSpike
          ? 'page fault pressure'
          : 'overdraw pressure';
      this.stableFrames = 0;
    } else {
      this.stableFrames += 1;

      if (this.stableFrames >= 90 && this.level > 0) {
        this.level -= 1;
        this.reason = 'recovered headroom';
        this.stableFrames = 0;
      }
    }

    return {
      level: this.level,
      reason: this.reason,
      budgets: this.applyLevel(this.level),
    };
  }

  private applyLevel(level: number): SplatBudgetOptions {
    const budgets = cloneBudgets(this.baseBudgets);

    if (level >= 1) {
      budgets.heroTileBudget = Math.max(0, budgets.heroTileBudget - 1);
    }

    if (level >= 2) {
      budgets.peripheralFoveation = 1.35;
    }

    if (level >= 3) {
      budgets.deformationBudget = 0;
      budgets.maxDeformablePages = Math.max(0, budgets.maxDeformablePages - 2);
    }

    if (level >= 4) {
      budgets.effectUpdateCadence = 2;
      budgets.maxOverdrawBudget = Math.round(budgets.maxOverdrawBudget * 0.82);
    }

    if (level >= 5) {
      budgets.maxVisibleSplats = Math.round(budgets.maxVisibleSplats * 0.82);
    }

    if (level >= 6) {
      budgets.minProjectedNodeSizePx = Math.round(budgets.minProjectedNodeSizePx * 1.25);
    }

    if (level >= 7) {
      budgets.renderScale = 0.85;
    }

    return budgets;
  }
}
