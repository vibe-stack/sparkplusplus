import { createSplatAsset, type SplatAsset, type SplatClusterNode, type SplatPage, type SplatVec3 } from './model';
import type { SplatSource } from '../scene/source';

export interface SpzHeader {
  magic: number;
  version: number;
  pointCount: number;
  shDegree: number;
  fractionalBits: number;
  flags: number;
  reserved: number;
}

export interface SpzImportOptions {
  label?: string;
  pageCapacity?: number;
  maxPoints?: number;
  branching?: number;
  minLeafPoints?: number;
}

interface PointCloud {
  positions: Float32Array;
  scales: Float32Array;
  colors: Float32Array;
  opacities: Float32Array;
  boundsMin: SplatVec3;
  boundsMax: SplatVec3;
}

interface BuildNodeResult {
  clusterId: number;
  boundsMin: SplatVec3;
  boundsMax: SplatVec3;
}

const SPZ_MAGIC = 0x5053474e;

export class SpzSplatSource implements SplatSource {
  readonly kind = 'spz';

  constructor(
    readonly id: string,
    readonly header: SpzHeader,
    private readonly asset: SplatAsset,
  ) {}

  static async fromUrl(url: string, options: SpzImportOptions = {}): Promise<SpzSplatSource> {
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Failed to fetch SPZ asset: ${response.status} ${response.statusText}`);
    }

    const gzipBuffer = await response.arrayBuffer();
    const decompressedBuffer = await decompressGzip(gzipBuffer);
    const parsed = parseSpzBuffer(decompressedBuffer, options);

    return new SpzSplatSource(url, parsed.header, parsed.asset);
  }

  buildAsset(): SplatAsset {
    return this.asset;
  }
}

async function decompressGzip(gzipBuffer: ArrayBuffer): Promise<ArrayBuffer> {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('This browser does not support DecompressionStream for SPZ loading');
  }

  const stream = new Blob([gzipBuffer]).stream().pipeThrough(new DecompressionStream('gzip'));
  return await new Response(stream).arrayBuffer();
}

function parseSpzBuffer(buffer: ArrayBuffer, options: SpzImportOptions): {
  header: SpzHeader;
  asset: SplatAsset;
} {
  const view = new DataView(buffer);
  const header = readHeader(view);

  if (header.magic !== SPZ_MAGIC) {
    throw new Error('Invalid SPZ header');
  }

  if (header.version < 2 || header.version > 3) {
    throw new Error(`Unsupported SPZ version ${header.version}`);
  }

  const pointCloud = samplePointCloud(view, header, options.maxPoints ?? header.pointCount);
  const pageCapacity = Math.max(256, options.pageCapacity ?? 1_024);
  const branching = Math.max(2, Math.min(8, options.branching ?? 4));
  const minLeafPoints = Math.max(pageCapacity, options.minLeafPoints ?? pageCapacity);
  const asset = buildPagedAssetFromPointCloud(pointCloud, {
    id: `spz:${options.label ?? 'demo'}`,
    label: options.label ?? 'Demo SPZ',
    pageCapacity,
    branching,
    minLeafPoints,
  });

  return {
    header,
    asset,
  };
}

function readHeader(view: DataView): SpzHeader {
  return {
    magic: view.getUint32(0, true),
    version: view.getUint32(4, true),
    pointCount: view.getUint32(8, true),
    shDegree: view.getUint8(12),
    fractionalBits: view.getUint8(13),
    flags: view.getUint8(14),
    reserved: view.getUint8(15),
  };
}

function samplePointCloud(view: DataView, header: SpzHeader, maxPoints: number): PointCloud {
  const stride = Math.max(1, Math.ceil(header.pointCount / maxPoints));
  const sampledCount = Math.ceil(header.pointCount / stride);
  const positions = new Float32Array(sampledCount * 3);
  const scales = new Float32Array(sampledCount * 3);
  const colors = new Float32Array(sampledCount * 3);
  const opacities = new Float32Array(sampledCount);
  const positionScale = 1 / (1 << header.fractionalBits);

  const positionsOffset = 16;
  const alphasOffset = positionsOffset + header.pointCount * 9;
  const colorsOffset = alphasOffset + header.pointCount;
  const scalesOffset = colorsOffset + header.pointCount * 3;
  const rotationsOffset = scalesOffset + header.pointCount * 3;
  const rotationStride = header.version >= 3 ? 4 : 3;
  const shOffset = rotationsOffset + header.pointCount * rotationStride;
  const shCoefficientCount = header.shDegree === 0 ? 0 : (header.shDegree + 1) ** 2 - 1;
  const expectedLength = shOffset + header.pointCount * 3 * shCoefficientCount;

  if (expectedLength > view.byteLength) {
    throw new Error('SPZ payload is truncated');
  }

  const boundsMin: SplatVec3 = [
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  ];
  const boundsMax: SplatVec3 = [
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ];

  let sampledIndex = 0;

  for (let pointIndex = 0; pointIndex < header.pointCount; pointIndex += stride) {
    const positionByteOffset = positionsOffset + pointIndex * 9;
    const alphaByteOffset = alphasOffset + pointIndex;
    const colorByteOffset = colorsOffset + pointIndex * 3;
    const scaleByteOffset = scalesOffset + pointIndex * 3;
    const targetOffset = sampledIndex * 3;

    for (let axis = 0; axis < 3; axis += 1) {
      const quantized = readSigned24(view, positionByteOffset + axis * 3);
      const position = quantized * positionScale;
      positions[targetOffset + axis] = position;
      boundsMin[axis] = Math.min(boundsMin[axis]!, position);
      boundsMax[axis] = Math.max(boundsMax[axis]!, position);
    }

    opacities[sampledIndex] = view.getUint8(alphaByteOffset) / 255;

    for (let axis = 0; axis < 3; axis += 1) {
      colors[targetOffset + axis] = view.getUint8(colorByteOffset + axis) / 255;
      const logScale = view.getUint8(scaleByteOffset + axis) / 16 - 10;
      scales[targetOffset + axis] = Math.exp(logScale);
    }

    sampledIndex += 1;
  }

  return {
    positions,
    scales,
    colors,
    opacities,
    boundsMin,
    boundsMax,
  };
}

function readSigned24(view: DataView, byteOffset: number): number {
  const value = view.getUint8(byteOffset)
    | (view.getUint8(byteOffset + 1) << 8)
    | (view.getUint8(byteOffset + 2) << 16);

  return (value & 0x800000) !== 0 ? value | ~0xffffff : value;
}

function buildPagedAssetFromPointCloud(
  pointCloud: PointCloud,
  options: {
    id: string;
    label: string;
    pageCapacity: number;
    branching: number;
    minLeafPoints: number;
  },
): SplatAsset {
  const pointCount = pointCloud.opacities.length;
  const mortonCodes = new Uint32Array(pointCount);
  const sortedIndices = new Uint32Array(pointCount);

  for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
    mortonCodes[pointIndex] = mortonCodeForPoint(
      pointCloud.positions,
      pointIndex,
      pointCloud.boundsMin,
      pointCloud.boundsMax,
    );
    sortedIndices[pointIndex] = pointIndex;
  }

  sortedIndices.sort((left, right) => mortonCodes[left]! - mortonCodes[right]!);

  const pages: SplatPage[] = [];
  const clusters: SplatClusterNode[] = [];

  const buildNode = (
    start: number,
    end: number,
    level: number,
    parentId: number | null,
  ): BuildNodeResult => {
    const count = end - start;
    // Internal (non-leaf) nodes only need a small coverage proxy page — their
    // job is to provide fallback splats while leaf pages stream in, not to
    // consume the active-page visual budget with sparse 4096-point samples
    // spread over a huge volume.  Leaf nodes get the full page capacity.
    const isLeaf = count <= options.minLeafPoints;
    const effectiveCapacity = isLeaf ? options.pageCapacity : Math.min(256, options.pageCapacity);
    const pageId = pages.length;
    const page = createPageFromRange(
      pageId,
      sortedIndices,
      pointCloud,
      start,
      end,
      effectiveCapacity,
      level,
    );
    pages.push(page);

    const clusterId = clusters.length;
    clusters.push({
      id: clusterId,
      pageId,
      level,
      splatCount: page.splatCount,
      center: [0, 0, 0],
      radius: 0,
      boundsMin: [0, 0, 0],
      boundsMax: [0, 0, 0],
      projectedErrorCoefficient: 0,
      opacityMass: page.opacities.reduce((sum, value) => sum + value, 0),
      anisotropySeverity: 0.25,
      expectedOverdrawScore: 0,
      motionSensitivity: 0,
      semanticMask: 0,
      parentId,
      childIds: [],
    });
    page.clusterId = clusterId;

    if (count <= options.minLeafPoints) {
      finalizeClusterFromBounds(
        clusters[clusterId]!,
        page.boundsMin,
        page.boundsMax,
        page.splatCount,
        pointCloud.scales,
        sortedIndices,
        start,
        end,
      );

      return {
        clusterId,
        boundsMin: page.boundsMin,
        boundsMax: page.boundsMax,
      };
    }

    const childCount = Math.min(options.branching, Math.ceil(count / options.minLeafPoints));
    const childIds: number[] = [];
    const boundsMin: SplatVec3 = [
      Number.POSITIVE_INFINITY,
      Number.POSITIVE_INFINITY,
      Number.POSITIVE_INFINITY,
    ];
    const boundsMax: SplatVec3 = [
      Number.NEGATIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
    ];

    for (let childIndex = 0; childIndex < childCount; childIndex += 1) {
      const childStart = start + Math.floor((count * childIndex) / childCount);
      const childEnd = start + Math.floor((count * (childIndex + 1)) / childCount);

      if (childEnd <= childStart) {
        continue;
      }

      const child = buildNode(childStart, childEnd, level + 1, clusterId);
      childIds.push(child.clusterId);

      for (let axis = 0; axis < 3; axis += 1) {
        boundsMin[axis] = Math.min(boundsMin[axis]!, child.boundsMin[axis]!);
        boundsMax[axis] = Math.max(boundsMax[axis]!, child.boundsMax[axis]!);
      }
    }

    clusters[clusterId]!.childIds = childIds;
    finalizeClusterFromBounds(
      clusters[clusterId]!,
      boundsMin,
      boundsMax,
      count,
      pointCloud.scales,
      sortedIndices,
      start,
      end,
    );

    return {
      clusterId,
      boundsMin,
      boundsMax,
    };
  };

  const root = buildNode(0, pointCount, 0, null);

  return createSplatAsset({
    id: options.id,
    label: options.label,
    pageCapacity: options.pageCapacity,
    maxDepth: clusters.reduce((maxDepth, cluster) => Math.max(maxDepth, cluster.level), 0),
    rootClusterIds: [root.clusterId],
    localBoundsMin: root.boundsMin,
    localBoundsMax: root.boundsMax,
    clusters,
    pages,
  });
}

function createPageFromRange(
  pageId: number,
  sortedIndices: Uint32Array,
  pointCloud: PointCloud,
  start: number,
  end: number,
  pageCapacity: number,
  level: number,
): SplatPage & { boundsMin: SplatVec3; boundsMax: SplatVec3 } {
  const sourceCount = end - start;
  const sampleCount = Math.min(pageCapacity, sourceCount);
  const positions = new Float32Array(sampleCount * 3);
  const scales = new Float32Array(sampleCount * 3);
  const colors = new Float32Array(sampleCount * 3);
  const opacities = new Float32Array(sampleCount);
  const boundsMin: SplatVec3 = [
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  ];
  const boundsMax: SplatVec3 = [
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ];

  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    const sourceOffset = start + Math.floor((sourceCount * sampleIndex) / sampleCount);
    const pointIndex = sortedIndices[sourceOffset]!;
    const targetOffset = sampleIndex * 3;
    const inputOffset = pointIndex * 3;

    for (let axis = 0; axis < 3; axis += 1) {
      const position = pointCloud.positions[inputOffset + axis]!;
      positions[targetOffset + axis] = position;
      scales[targetOffset + axis] = pointCloud.scales[inputOffset + axis]!;
      colors[targetOffset + axis] = pointCloud.colors[inputOffset + axis]!;
      boundsMin[axis] = Math.min(boundsMin[axis]!, position);
      boundsMax[axis] = Math.max(boundsMax[axis]!, position);
    }

    opacities[sampleIndex] = pointCloud.opacities[pointIndex]!;
  }

  return {
    id: pageId,
    clusterId: -1,
    level,
    splatCount: sampleCount,
    capacity: pageCapacity,
    semanticMask: 0,
    byteSize: positions.byteLength + scales.byteLength + colors.byteLength + opacities.byteLength,
    positions,
    scales,
    colors,
    opacities,
    boundsMin,
    boundsMax,
  };
}

function finalizeClusterFromBounds(
  cluster: SplatClusterNode,
  boundsMin: SplatVec3,
  boundsMax: SplatVec3,
  representedPoints: number,
  scales: Float32Array,
  sortedIndices: Uint32Array,
  start: number,
  end: number,
): void {
  const centerX = (boundsMin[0] + boundsMax[0]) * 0.5;
  const centerY = (boundsMin[1] + boundsMax[1]) * 0.5;
  const centerZ = (boundsMin[2] + boundsMax[2]) * 0.5;
  const radius = Math.hypot(
    boundsMax[0] - boundsMin[0],
    boundsMax[1] - boundsMin[1],
    boundsMax[2] - boundsMin[2],
  ) * 0.5;
  let averageScale = 0;
  const probeCount = Math.min(16, end - start);

  for (let probeIndex = 0; probeIndex < probeCount; probeIndex += 1) {
    const sourceIndex = sortedIndices[start + Math.floor(((end - start) * probeIndex) / probeCount)]!;
    const scaleOffset = sourceIndex * 3;
    averageScale += (
      scales[scaleOffset + 0]!
      + scales[scaleOffset + 1]!
      + scales[scaleOffset + 2]!
    ) / 3;
  }

  averageScale = probeCount === 0 ? 0.01 : averageScale / probeCount;

  cluster.center = [centerX, centerY, centerZ];
  cluster.radius = radius;
  cluster.boundsMin = boundsMin;
  cluster.boundsMax = boundsMax;
  cluster.projectedErrorCoefficient = Math.max(24, radius * Math.sqrt(representedPoints));
  cluster.anisotropySeverity = averageScale;
  cluster.expectedOverdrawScore = Math.max(1, Math.sqrt(representedPoints) * averageScale * 18);
}

function mortonCodeForPoint(
  positions: Float32Array,
  pointIndex: number,
  boundsMin: SplatVec3,
  boundsMax: SplatVec3,
): number {
  const offset = pointIndex * 3;
  const x = quantize10Bit(positions[offset + 0]!, boundsMin[0], boundsMax[0]);
  const y = quantize10Bit(positions[offset + 1]!, boundsMin[1], boundsMax[1]);
  const z = quantize10Bit(positions[offset + 2]!, boundsMin[2], boundsMax[2]);
  return interleaveBits10(x, y, z);
}

function quantize10Bit(value: number, min: number, max: number): number {
  const range = Math.max(1e-5, max - min);
  return Math.max(0, Math.min(1023, Math.floor(((value - min) / range) * 1023)));
}

function interleaveBits10(x: number, y: number, z: number): number {
  let code = 0;

  for (let bit = 0; bit < 10; bit += 1) {
    code |= ((x >> bit) & 1) << (bit * 3);
    code |= ((y >> bit) & 1) << (bit * 3 + 1);
    code |= ((z >> bit) & 1) << (bit * 3 + 2);
  }

  return code >>> 0;
}
