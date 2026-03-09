export interface SplatBudgetOptions {
  maxVisibleSplats: number;
  maxOverdrawBudget: number;
  maxActivePages: number;
  maxResidentPages: number;
  maxDeformablePages: number;
  maxPageUploadsPerFrame: number;
  minProjectedNodeSizePx: number;
  peripheralFoveation: number;
  heroTileBudget: number;
  effectUpdateCadence: number;
  deformationBudget: number;
  renderScale: number;
  temporalStabilityBias: number;
}

export const DEFAULT_SPLAT_BUDGETS: SplatBudgetOptions = {
  maxVisibleSplats: 7_500,
  maxOverdrawBudget: 1_200,
  maxActivePages: 18,
  maxResidentPages: 32,
  maxDeformablePages: 4,
  maxPageUploadsPerFrame: 3,
  minProjectedNodeSizePx: 64,
  peripheralFoveation: 1,
  heroTileBudget: 2,
  effectUpdateCadence: 1,
  deformationBudget: 1,
  renderScale: 1,
  temporalStabilityBias: 1.15,
};

export function cloneBudgets(
  overrides: Partial<SplatBudgetOptions> = {},
): SplatBudgetOptions {
  return {
    ...DEFAULT_SPLAT_BUDGETS,
    ...overrides,
  };
}
