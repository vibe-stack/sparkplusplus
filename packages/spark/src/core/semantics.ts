export const SPLAT_SEMANTIC_FLAGS = {
  hero: 1 << 0,
  face: 1 << 1,
  hands: 1 << 2,
  foliage: 1 << 3,
  glass: 1 << 4,
  deformable: 1 << 5,
} as const;

export type SplatSemanticLabel = keyof typeof SPLAT_SEMANTIC_FLAGS;

export const SPLAT_SEMANTIC_LABELS = Object.keys(
  SPLAT_SEMANTIC_FLAGS,
) as SplatSemanticLabel[];

export function semanticMaskFromLabels(
  labels: readonly SplatSemanticLabel[],
): number {
  let mask = 0;

  for (const label of labels) {
    mask |= SPLAT_SEMANTIC_FLAGS[label];
  }

  return mask;
}

export function semanticLabelsFromMask(mask: number): SplatSemanticLabel[] {
  return SPLAT_SEMANTIC_LABELS.filter((label) => (mask & SPLAT_SEMANTIC_FLAGS[label]) !== 0);
}

export function hasSemanticFlag(mask: number, label: SplatSemanticLabel): boolean {
  return (mask & SPLAT_SEMANTIC_FLAGS[label]) !== 0;
}
