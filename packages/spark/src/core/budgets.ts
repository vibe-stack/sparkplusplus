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
  tileSizePx: number;
  maxClustersPerTile: number;
  tileDepthBucketCount: number;
  clusterTilePadding: number;
  maxSplatsPerTile: number;
}

export const DEFAULT_SPLAT_BUDGETS: SplatBudgetOptions = {
  maxVisibleSplats: 24_000,
  maxOverdrawBudget: 1_800,
  maxActivePages: 36,
  maxResidentPages: 96,
  maxDeformablePages: 4,
  maxPageUploadsPerFrame: 8,
  minProjectedNodeSizePx: 44,
  peripheralFoveation: 1,
  heroTileBudget: 3,
  effectUpdateCadence: 1,
  deformationBudget: 1,
  renderScale: 1,
  temporalStabilityBias: 1.3,
  tileSizePx: 24,
  maxClustersPerTile: 32,
  tileDepthBucketCount: 6,
  clusterTilePadding: 0,
  maxSplatsPerTile: 768,
};

export function cloneBudgets(
  overrides: Partial<SplatBudgetOptions> = {},
): SplatBudgetOptions {
  return {
    ...DEFAULT_SPLAT_BUDGETS,
    ...overrides,
  };
}
