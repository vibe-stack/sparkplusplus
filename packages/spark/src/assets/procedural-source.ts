import { Color, MathUtils } from 'three';
import { createSplatAsset, type SplatAsset, type SplatClusterNode, type SplatPage, type SplatVec3 } from './model';
import type { SplatSource } from '../scene/source';
import {
  SPLAT_SEMANTIC_LABELS,
  SPLAT_SEMANTIC_FLAGS,
  semanticMaskFromLabels,
  type SplatSemanticLabel,
} from '../core/semantics';

export interface ProceduralSplatSourceOptions {
  id: string;
  label?: string;
  seed?: number;
  depth?: number;
  branching?: number;
  pageCapacity?: number;
  rootRadius?: number;
  center?: SplatVec3;
  palette?: readonly [string | number, string | number];
}

interface BuildClusterResult {
  clusterId: number;
  boundsMin: SplatVec3;
  boundsMax: SplatVec3;
}

class SeededRandom {
  constructor(private state: number) {}

  next(): number {
    this.state |= 0;
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  range(min: number, max: number): number {
    return min + (max - min) * this.next();
  }

  int(min: number, max: number): number {
    return Math.floor(this.range(min, max + 1));
  }

  pick<T>(items: readonly T[]): T {
    return items[Math.floor(this.next() * items.length)]!;
  }
}

export class ProceduralSplatSource implements SplatSource {
  readonly kind = 'procedural';

  private cachedAsset?: SplatAsset;

  constructor(
    readonly options: ProceduralSplatSourceOptions,
  ) {}

  get id(): string {
    return this.options.id;
  }

  buildAsset(): SplatAsset {
    if (this.cachedAsset) {
      return this.cachedAsset;
    }

    const seed = this.options.seed ?? 1337;
    const depth = Math.max(1, this.options.depth ?? 3);
    const branching = Math.max(2, Math.min(this.options.branching ?? 4, 6));
    const pageCapacity = Math.max(48, this.options.pageCapacity ?? 192);
    const center = this.options.center ?? [0, 0, 0];
    const rootRadius = this.options.rootRadius ?? 3.2;
    const palette = this.options.palette ?? [0x7cf2cf, 0xffc857];
    const random = new SeededRandom(seed);
    const clusters: SplatClusterNode[] = [];
    const pages: SplatPage[] = [];
    const colorA = new Color(palette[0]);
    const colorB = new Color(palette[1]);

    const buildCluster = (
      level: number,
      parentId: number | null,
      clusterCenter: SplatVec3,
      radius: number,
      inheritedMask: number,
      branchIndex: number,
    ): BuildClusterResult => {
      const clusterId = clusters.length;
      const semanticMask = this.getSemanticMask(level, branchIndex, inheritedMask, random);
      const pageId = pages.length;
      const page = this.createPage(
        clusterId,
        pageId,
        level,
        depth,
        pageCapacity,
        clusterCenter,
        radius,
        semanticMask,
        colorA,
        colorB,
        random,
      );

      pages.push(page);
      clusters.push({
        id: clusterId,
        pageId,
        level,
        splatCount: page.splatCount,
        center: clusterCenter,
        radius,
        boundsMin: [...clusterCenter],
        boundsMax: [...clusterCenter],
        projectedErrorCoefficient: radius * (depth - level + 1) * 120,
        opacityMass: page.opacities.reduce((sum, opacity) => sum + opacity, 0),
        anisotropySeverity: random.range(0.15, 1),
        expectedOverdrawScore: page.splatCount * radius * random.range(0.9, 1.3),
        motionSensitivity: random.range(0.05, 1),
        semanticMask,
        parentId,
        childIds: [],
      });

      const boundsMin: SplatVec3 = [...clusterCenter];
      const boundsMax: SplatVec3 = [...clusterCenter];

      for (let i = 0; i < page.splatCount; i += 1) {
        const offset = i * 3;
        boundsMin[0] = Math.min(boundsMin[0], page.positions[offset + 0]!);
        boundsMin[1] = Math.min(boundsMin[1], page.positions[offset + 1]!);
        boundsMin[2] = Math.min(boundsMin[2], page.positions[offset + 2]!);
        boundsMax[0] = Math.max(boundsMax[0], page.positions[offset + 0]!);
        boundsMax[1] = Math.max(boundsMax[1], page.positions[offset + 1]!);
        boundsMax[2] = Math.max(boundsMax[2], page.positions[offset + 2]!);
      }

      if (level < depth) {
        const childIds: number[] = [];

        for (let childIndex = 0; childIndex < branching; childIndex += 1) {
          const directionLength = random.range(0.45, 0.7) * radius;
          const theta = random.range(0, Math.PI * 2);
          const y = random.range(-0.55, 0.55);
          const horizontal = Math.sqrt(Math.max(0.05, 1 - y * y));
          const childCenter: SplatVec3 = [
            clusterCenter[0] + Math.cos(theta) * horizontal * directionLength,
            clusterCenter[1] + y * directionLength,
            clusterCenter[2] + Math.sin(theta) * horizontal * directionLength,
          ];
          const childRadius = radius * random.range(0.36, 0.5);
          const child = buildCluster(
            level + 1,
            clusterId,
            childCenter,
            childRadius,
            semanticMask,
            childIndex,
          );

          childIds.push(child.clusterId);
          boundsMin[0] = Math.min(boundsMin[0], child.boundsMin[0]);
          boundsMin[1] = Math.min(boundsMin[1], child.boundsMin[1]);
          boundsMin[2] = Math.min(boundsMin[2], child.boundsMin[2]);
          boundsMax[0] = Math.max(boundsMax[0], child.boundsMax[0]);
          boundsMax[1] = Math.max(boundsMax[1], child.boundsMax[1]);
          boundsMax[2] = Math.max(boundsMax[2], child.boundsMax[2]);
        }

        clusters[clusterId]!.childIds = childIds;
      }

      clusters[clusterId]!.boundsMin = boundsMin;
      clusters[clusterId]!.boundsMax = boundsMax;

      return {
        clusterId,
        boundsMin,
        boundsMax,
      };
    };

    const root = buildCluster(0, null, center, rootRadius, SPLAT_SEMANTIC_FLAGS.hero, 0);

    this.cachedAsset = createSplatAsset({
      id: this.options.id,
      label: this.options.label ?? this.options.id,
      pageCapacity,
      maxDepth: depth,
      rootClusterIds: [root.clusterId],
      localBoundsMin: root.boundsMin,
      localBoundsMax: root.boundsMax,
      clusters,
      pages,
    });

    return this.cachedAsset;
  }

  private getSemanticMask(
    level: number,
    branchIndex: number,
    inheritedMask: number,
    random: SeededRandom,
  ): number {
    const semanticLabels: SplatSemanticLabel[] = [];

    if ((inheritedMask & SPLAT_SEMANTIC_FLAGS.hero) !== 0) {
      semanticLabels.push('hero');
    }

    if (level > 0 && random.next() > 0.35) {
      semanticLabels.push(SPLAT_SEMANTIC_LABELS[(level + branchIndex) % SPLAT_SEMANTIC_LABELS.length]!);
    }

    if (random.next() > 0.72) {
      semanticLabels.push(random.pick(SPLAT_SEMANTIC_LABELS));
    }

    return inheritedMask | semanticMaskFromLabels(semanticLabels);
  }

  private createPage(
    clusterId: number,
    pageId: number,
    level: number,
    maxDepth: number,
    capacity: number,
    center: SplatVec3,
    radius: number,
    semanticMask: number,
    colorA: Color,
    colorB: Color,
    random: SeededRandom,
  ): SplatPage {
    const levelAlpha = maxDepth === 0 ? 1 : level / maxDepth;
    const splatCount = MathUtils.clamp(
      Math.round(capacity * (0.22 + levelAlpha * 0.58 + random.range(0.05, 0.18))),
      Math.min(48, capacity),
      capacity,
    );
    const positions = new Float32Array(splatCount * 3);
    const scales = new Float32Array(splatCount * 3);
    const colors = new Float32Array(splatCount * 3);
    const opacities = new Float32Array(splatCount);
    const tint = new Color();

    for (let i = 0; i < splatCount; i += 1) {
      const theta = random.range(0, Math.PI * 2);
      const phi = Math.acos(random.range(-1, 1));
      const radialDistance = Math.pow(random.next(), 0.65) * radius;
      const ellipse = random.range(0.42, 1);
      const offset = i * 3;
      const sinPhi = Math.sin(phi);

      positions[offset + 0] = center[0] + Math.cos(theta) * sinPhi * radialDistance * ellipse;
      positions[offset + 1] = center[1] + Math.cos(phi) * radialDistance * random.range(0.55, 1);
      positions[offset + 2] = center[2] + Math.sin(theta) * sinPhi * radialDistance;

      const scale = random.range(radius * 0.015, radius * 0.08);
      scales[offset + 0] = scale;
      scales[offset + 1] = scale * random.range(0.8, 1.4);
      scales[offset + 2] = scale * random.range(0.8, 1.4);

      tint.copy(colorA).lerp(colorB, MathUtils.clamp(levelAlpha * 0.55 + random.range(0.1, 0.5), 0, 1));

      if ((semanticMask & SPLAT_SEMANTIC_FLAGS.glass) !== 0) {
        tint.lerp(new Color(0xa6d8ff), 0.35);
      }

      if ((semanticMask & SPLAT_SEMANTIC_FLAGS.foliage) !== 0) {
        tint.lerp(new Color(0x7dd181), 0.3);
      }

      colors[offset + 0] = tint.r;
      colors[offset + 1] = tint.g;
      colors[offset + 2] = tint.b;
      opacities[i] = random.range(0.45, 0.98);
    }

    return {
      id: pageId,
      clusterId,
      level,
      splatCount,
      capacity,
      semanticMask,
      byteSize: positions.byteLength + scales.byteLength + colors.byteLength + opacities.byteLength,
      positions,
      scales,
      colors,
      opacities,
    };
  }
}
