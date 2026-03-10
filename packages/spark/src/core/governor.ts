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
  // Refinement and page uploads can temporarily spike CPU time for a handful
  // of frames right when the camera settles. Reacting to a single spike turns
  // a transient catch-up cost into a persistent quality collapse.
  private consecutiveOverloadedFrames = 0;
  // Page-fault spikes are expected for 1-2 frames whenever the frontier
  // transitions to a finer LOD level (e.g. when the camera settles and
  // refineFrontier pushes to deeper clusters).  Escalating on the very first
  // spike frame turns a transient loading burst into a permanent quality
  // regression.  Require 3 consecutive spike frames before escalating.
  private consecutiveFaultFrames = 0;

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
    const refinementTransition = snapshot.pageUploads > 0 || snapshot.frontierStability < 0.92;
    if (!refinementTransition && snapshot.cpuFrameMs > this.targetFrameMs * 1.25) {
      this.consecutiveOverloadedFrames += 1;
    } else {
      this.consecutiveOverloadedFrames = 0;
    }
    const overloadedFrame = this.consecutiveOverloadedFrames >= 3;
    if (!refinementTransition && snapshot.pageFaultRate > 0.35) {
      this.consecutiveFaultFrames += 1;
    } else {
      this.consecutiveFaultFrames = 0;
    }
    const pageFaultSpike = this.consecutiveFaultFrames >= 3;
    // Use 1.45× instead of 0.92× as the escalation threshold.
    // canReplaceFrontierCandidate deliberately allows estimatedOverdraw to reach
    // up to 1.6× maxOverdrawBudget at close-up range (via closeUpFactor).  With
    // the old 0.92 threshold the governor triggered overdraw escalation on
    // virtually every close-up frame, racing to level 5-6 and raising
    // minProjectedNodeSizePx / shrinking maxResidentPages, which caused the
    // very holes and detail loss the user experiences when zoomed in.
    const overdrawPressure = snapshot.estimatedOverdraw > snapshot.budgets.maxOverdrawBudget * 1.45;

    if (overloadedFrame || pageFaultSpike || overdrawPressure) {
      const nextReason = overloadedFrame
        ? 'cpu frame pressure'
        : pageFaultSpike
          ? 'page fault pressure'
          : 'overdraw pressure';
      const nextLevelCap = nextReason === 'overdraw pressure' ? 7 : 4;
      this.level = Math.min(this.level + 1, nextLevelCap);
      this.reason = nextReason;
      this.stableFrames = 0;
      this.consecutiveOverloadedFrames = 0;
      if (nextReason === 'page fault pressure') {
        this.consecutiveFaultFrames = 0;
      }
    } else {
      this.stableFrames += 1;

      // 45 frames (~0.75 s at 60 fps) instead of 90: governor transients from
      // close-up zoom-in or page-fault spikes should recover within one second
      // once the camera settles, not after a full 1.5 second window.
      if (this.stableFrames >= 45 && this.level > 0) {
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
      budgets.maxPageUploadsPerFrame = Math.max(2, Math.round(budgets.maxPageUploadsPerFrame * 0.75));
    }

    if (level >= 5) {
      budgets.maxVisibleSplats = Math.round(budgets.maxVisibleSplats * 0.82);
      budgets.maxActivePages = Math.max(12, Math.round(budgets.maxActivePages * 0.78));
    }

    if (level >= 6) {
      budgets.minProjectedNodeSizePx = Math.round(budgets.minProjectedNodeSizePx * 1.25);
      budgets.maxResidentPages = Math.max(16, Math.round(budgets.maxResidentPages * 0.75));
    }

    if (level >= 7) {
      budgets.renderScale = 0.85;
    }

    return budgets;
  }
}
