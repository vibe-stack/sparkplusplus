import {
  BufferAttribute,
  Camera,
  Frustum,
  Matrix4,
  OrthographicCamera,
  PerspectiveCamera,
  Vector3,
  Vector4,
} from 'three';
import { StorageInstancedBufferAttribute } from 'three/webgpu';
import {
  Fn,
  If,
  dot,
  float,
  instanceIndex,
  length,
  max,
  storage,
  uint,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import { DynamicDrawUsage } from 'three';
import { SPLAT_SEMANTIC_FLAGS } from '../core/semantics';
import type { SplatBudgetOptions } from '../core/budgets';
import type { SplatSceneObjectDescriptor } from '../core/scene-descriptor';

export interface SplatGpuRendererLike {
  compute?: (...args: any[]) => Promise<unknown> | unknown;
  computeAsync?: (...args: any[]) => Promise<unknown>;
  getArrayBufferAsync?: (attribute: BufferAttribute) => Promise<ArrayBuffer>;
  hasInitialized?: () => boolean;
}

export interface SplatGpuClusterResult {
  visible: boolean;
  score: number;
  projectedSizePx: number;
  screenRadius: number;
}

export interface SplatGpuVisibilityReadback {
  ready: boolean;
  pending: boolean;
  frameIndex: number;
  clusterCount: number;
  get(meshUuid: string, clusterId: number): SplatGpuClusterResult | null;
  getSortedVisibleClusterIds(meshUuid: string): readonly number[];
}

interface ClusterMapping {
  key: string;
  meshUuid: string;
  clusterId: number;
}

class VisibilityReadbackView implements SplatGpuVisibilityReadback {
  constructor(
    readonly ready: boolean,
    readonly pending: boolean,
    readonly frameIndex: number,
    readonly clusterCount: number,
    private readonly resultMap: Map<string, SplatGpuClusterResult>,
    private readonly sortedVisibleClustersByMesh: Map<string, readonly number[]>,
  ) {}

  get(meshUuid: string, clusterId: number): SplatGpuClusterResult | null {
    return this.resultMap.get(`${meshUuid}:${clusterId}`) ?? null;
  }

  getSortedVisibleClusterIds(meshUuid: string): readonly number[] {
    return this.sortedVisibleClustersByMesh.get(meshUuid) ?? [];
  }
}

export class SplatGpuVisibilityPipeline {
  private readonly projectionMatrix = new Matrix4();
  private readonly frustum = new Frustum();
  private readonly cameraWorldPosition = new Vector3();

  private sceneSignature = '';
  private clusterMappings: ClusterMapping[] = [];
  private clusterCount = 0;
  private objectCount = 0;

  private pending = false;
  private latestFrameIndex = -1;
  private latestResultMap = new Map<string, SplatGpuClusterResult>();
  private latestSortedVisibleClustersByMesh = new Map<string, readonly number[]>();

  private objectRow0Attribute?: StorageInstancedBufferAttribute;
  private objectRow1Attribute?: StorageInstancedBufferAttribute;
  private objectRow2Attribute?: StorageInstancedBufferAttribute;
  private objectParamsAttribute?: StorageInstancedBufferAttribute;
  private clusterLocalAttribute?: StorageInstancedBufferAttribute;
  private clusterMetricAttribute?: StorageInstancedBufferAttribute;
  private clusterWorldAttribute?: StorageInstancedBufferAttribute;
  private clusterScoreAttribute?: StorageInstancedBufferAttribute;

  private objectRow0Node: any;
  private objectRow1Node: any;
  private objectRow2Node: any;
  private objectParamsNode: any;
  private clusterLocalNode: any;
  private clusterMetricNode: any;
  private clusterWorldNode: any;
  private clusterScoreNode: any;
  private worldSphereComputeNode: any;
  private cullScoreComputeNode: any;

  private readonly cameraParamsA = uniform(new Vector4());
  private readonly cameraParamsB = uniform(new Vector4());
  private readonly budgetParams = uniform(new Vector4());
  private readonly frustumPlanes = Array.from({ length: 6 }, () => uniform(new Vector4()));

  update(
    renderer: SplatGpuRendererLike | undefined,
    objects: readonly SplatSceneObjectDescriptor[],
    camera: Camera,
    viewportHeight: number,
    budgets: SplatBudgetOptions,
    frameIndex: number,
  ): void {
    const sceneSignature = objects
      .map((descriptor) => `${descriptor.mesh.uuid}:${descriptor.mesh.getAsset().clusters.length}`)
      .join('|');

    if (sceneSignature !== this.sceneSignature) {
      this.rebuildScene(objects);
      this.sceneSignature = sceneSignature;
    }

    if (this.clusterCount === 0) {
      return;
    }

    this.updateDynamicInputs(objects, camera, viewportHeight, budgets);

    if (
      !renderer
      || renderer.hasInitialized?.() === false
      || !renderer.computeAsync
      || !renderer.getArrayBufferAsync
      || this.pending
      || !this.worldSphereComputeNode
      || !this.cullScoreComputeNode
      || !this.clusterScoreAttribute
    ) {
      return;
    }

    const dispatchSignature = this.sceneSignature;
    const clusterMappings = [...this.clusterMappings];
    const clusterCount = this.clusterCount;
    const scoreAttribute = this.clusterScoreAttribute;

    this.pending = true;

    void renderer.computeAsync([this.worldSphereComputeNode, this.cullScoreComputeNode]).then(async () => {
      const arrayBuffer = await renderer.getArrayBufferAsync!(scoreAttribute);
      const raw = new Float32Array(arrayBuffer);
      const expectedValueCount = clusterCount * 4;

      if (raw.length < expectedValueCount) {
        return;
      }

      const resultMap = new Map<string, SplatGpuClusterResult>();
      const sortedVisibleClustersByMesh = new Map<string, Array<{ clusterId: number; score: number }>>();

      for (let clusterIndex = 0; clusterIndex < clusterCount; clusterIndex += 1) {
        const mapping = clusterMappings[clusterIndex]!;
        const offset = clusterIndex * 4;
        const rawScore = raw[offset + 0]!;
        const rawProjectedSizePx = raw[offset + 1]!;
        const rawScreenRadius = raw[offset + 2]!;
        const rawVisibility = raw[offset + 3]!;

        if (
          !Number.isFinite(rawScore)
          || !Number.isFinite(rawProjectedSizePx)
          || !Number.isFinite(rawScreenRadius)
          || !Number.isFinite(rawVisibility)
        ) {
          return;
        }

        const previous = this.latestResultMap.get(mapping.key);
        const visible = rawVisibility > 0.5;
        const projectedSizePx = previous && visible && previous.visible
          ? Math.max(0, rawProjectedSizePx) * 0.72 + previous.projectedSizePx * 0.28
          : Math.max(0, rawProjectedSizePx);
        const screenRadius = previous && visible && previous.visible
          ? Math.max(0, rawScreenRadius) * 0.72 + previous.screenRadius * 0.28
          : Math.max(0, rawScreenRadius);
        const score = previous && visible && previous.visible
          ? Math.max(0, rawScore) * 0.7 + previous.score * 0.3
          : Math.max(0, rawScore);

        resultMap.set(mapping.key, {
          visible,
          score,
          projectedSizePx,
          screenRadius,
        });

        if (visible) {
          const meshVisibleClusters = sortedVisibleClustersByMesh.get(mapping.meshUuid) ?? [];
          meshVisibleClusters.push({
            clusterId: mapping.clusterId,
            score,
          });
          sortedVisibleClustersByMesh.set(mapping.meshUuid, meshVisibleClusters);
        }
      }

      const finalizedVisibleClustersByMesh = new Map<string, readonly number[]>();

      sortedVisibleClustersByMesh.forEach((clusters, meshUuid) => {
        clusters.sort((left, right) => right.score - left.score);
        finalizedVisibleClustersByMesh.set(
          meshUuid,
          clusters.map((cluster) => cluster.clusterId),
        );
      });

      if (dispatchSignature === this.sceneSignature) {
        this.latestResultMap = resultMap;
        this.latestSortedVisibleClustersByMesh = finalizedVisibleClustersByMesh;
        this.latestFrameIndex = frameIndex;
      }
    }).catch(() => {
      // The CPU path remains the fallback when compute/readback is unavailable.
    }).finally(() => {
      this.pending = false;
    });
  }

  getLatestReadback(): SplatGpuVisibilityReadback {
    return new VisibilityReadbackView(
      this.latestFrameIndex >= 0,
      this.pending,
      this.latestFrameIndex,
      this.clusterCount,
      this.latestResultMap,
      this.latestSortedVisibleClustersByMesh,
    );
  }

  private rebuildScene(objects: readonly SplatSceneObjectDescriptor[]): void {
    this.objectCount = objects.length;
    this.clusterMappings = [];
    this.latestResultMap = new Map();
    this.latestSortedVisibleClustersByMesh = new Map();
    this.latestFrameIndex = -1;
    this.clusterCount = objects.reduce(
      (sum, descriptor) => sum + descriptor.mesh.getAsset().clusters.length,
      0,
    );

    if (this.objectCount === 0 || this.clusterCount === 0) {
      this.worldSphereComputeNode = null;
      this.cullScoreComputeNode = null;
      return;
    }

    this.objectRow0Attribute = this.createStorageAttribute(this.objectCount, 4);
    this.objectRow1Attribute = this.createStorageAttribute(this.objectCount, 4);
    this.objectRow2Attribute = this.createStorageAttribute(this.objectCount, 4);
    this.objectParamsAttribute = this.createStorageAttribute(this.objectCount, 4);
    this.clusterLocalAttribute = this.createStorageAttribute(this.clusterCount, 4);
    this.clusterMetricAttribute = this.createStorageAttribute(this.clusterCount, 4);
    this.clusterWorldAttribute = this.createStorageAttribute(this.clusterCount, 4);
    this.clusterScoreAttribute = this.createStorageAttribute(this.clusterCount, 4);

    this.objectRow0Node = storage(this.objectRow0Attribute, 'vec4', this.objectCount);
    this.objectRow1Node = storage(this.objectRow1Attribute, 'vec4', this.objectCount);
    this.objectRow2Node = storage(this.objectRow2Attribute, 'vec4', this.objectCount);
    this.objectParamsNode = storage(this.objectParamsAttribute, 'vec4', this.objectCount);
    this.clusterLocalNode = storage(this.clusterLocalAttribute, 'vec4', this.clusterCount);
    this.clusterMetricNode = storage(this.clusterMetricAttribute, 'vec4', this.clusterCount);
    this.clusterWorldNode = storage(this.clusterWorldAttribute, 'vec4', this.clusterCount);
    this.clusterScoreNode = storage(this.clusterScoreAttribute, 'vec4', this.clusterCount);

    const localArray = this.clusterLocalAttribute.array as Float32Array;
    const metricArray = this.clusterMetricAttribute.array as Float32Array;

    let clusterCursor = 0;

    objects.forEach((descriptor, objectIndex) => {
      const asset = descriptor.mesh.getAsset();

      asset.clusters.forEach((cluster) => {
        const localOffset = clusterCursor * 4;
        localArray[localOffset + 0] = cluster.center[0];
        localArray[localOffset + 1] = cluster.center[1];
        localArray[localOffset + 2] = cluster.center[2];
        localArray[localOffset + 3] = cluster.radius;

        metricArray[localOffset + 0] = cluster.projectedErrorCoefficient;
        metricArray[localOffset + 1] = cluster.expectedOverdrawScore;
        metricArray[localOffset + 2] = objectIndex;
        metricArray[localOffset + 3] = (cluster.semanticMask & SPLAT_SEMANTIC_FLAGS.hero) !== 0 ? 1 : 0;

        this.clusterMappings.push({
          key: `${descriptor.mesh.uuid}:${cluster.id}`,
          meshUuid: descriptor.mesh.uuid,
          clusterId: cluster.id,
        });
        clusterCursor += 1;
      });
    });

    this.clusterLocalAttribute.needsUpdate = true;
    this.clusterMetricAttribute.needsUpdate = true;
    this.createComputeNodes();
  }

  private updateDynamicInputs(
    objects: readonly SplatSceneObjectDescriptor[],
    camera: Camera,
    viewportHeight: number,
    budgets: SplatBudgetOptions,
  ): void {
    if (
      !this.objectRow0Attribute
      || !this.objectRow1Attribute
      || !this.objectRow2Attribute
      || !this.objectParamsAttribute
    ) {
      return;
    }

    const row0Array = this.objectRow0Attribute.array as Float32Array;
    const row1Array = this.objectRow1Attribute.array as Float32Array;
    const row2Array = this.objectRow2Attribute.array as Float32Array;
    const paramsArray = this.objectParamsAttribute.array as Float32Array;

    objects.forEach((descriptor, objectIndex) => {
      const elements = descriptor.mesh.matrixWorld.elements;
      const offset = objectIndex * 4;
      const maxScale = Math.max(
        Math.hypot(elements[0]!, elements[1]!, elements[2]!),
        Math.hypot(elements[4]!, elements[5]!, elements[6]!),
        Math.hypot(elements[8]!, elements[9]!, elements[10]!),
      );

      row0Array[offset + 0] = elements[0]!;
      row0Array[offset + 1] = elements[4]!;
      row0Array[offset + 2] = elements[8]!;
      row0Array[offset + 3] = elements[12]!;
      row1Array[offset + 0] = elements[1]!;
      row1Array[offset + 1] = elements[5]!;
      row1Array[offset + 2] = elements[9]!;
      row1Array[offset + 3] = elements[13]!;
      row2Array[offset + 0] = elements[2]!;
      row2Array[offset + 1] = elements[6]!;
      row2Array[offset + 2] = elements[10]!;
      row2Array[offset + 3] = elements[14]!;
      paramsArray[offset + 0] = maxScale;
      paramsArray[offset + 1] = 0;
      paramsArray[offset + 2] = 0;
      paramsArray[offset + 3] = 0;
    });

    this.objectRow0Attribute.needsUpdate = true;
    this.objectRow1Attribute.needsUpdate = true;
    this.objectRow2Attribute.needsUpdate = true;
    this.objectParamsAttribute.needsUpdate = true;

    this.projectionMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projectionMatrix);
    camera.getWorldPosition(this.cameraWorldPosition);
    this.cameraParamsA.value.set(
      this.cameraWorldPosition.x,
      this.cameraWorldPosition.y,
      this.cameraWorldPosition.z,
      viewportHeight,
    );

    if (camera instanceof PerspectiveCamera) {
      this.cameraParamsB.value.set(
        1,
        Math.tan((camera.fov * Math.PI) / 360),
        1,
        0,
      );
    } else if (camera instanceof OrthographicCamera) {
      const span = Math.max(0.001, (camera.top - camera.bottom) / Math.max(camera.zoom, 1e-5));
      this.cameraParamsB.value.set(0, 0, span, 0);
    } else {
      this.cameraParamsB.value.set(1, 1, 1, 0);
    }

    this.budgetParams.value.set(
      budgets.maxOverdrawBudget,
      budgets.peripheralFoveation,
      budgets.maxVisibleSplats,
      0,
    );

    for (let planeIndex = 0; planeIndex < 6; planeIndex += 1) {
      const plane = this.frustum.planes[planeIndex]!;
      this.frustumPlanes[planeIndex]!.value.set(
        plane.normal.x,
        plane.normal.y,
        plane.normal.z,
        plane.constant,
      );
    }
  }

  private createComputeNodes(): void {
    const workgroup = [64];

    this.worldSphereComputeNode = Fn(() => {
      const localSphere = this.clusterLocalNode.element(instanceIndex);
      const metric = this.clusterMetricNode.element(instanceIndex);
      const objectIndex = uint(metric.z);
      const row0 = this.objectRow0Node.element(objectIndex);
      const row1 = this.objectRow1Node.element(objectIndex);
      const row2 = this.objectRow2Node.element(objectIndex);
      const objectParams = this.objectParamsNode.element(objectIndex);
      const localPosition = vec4(localSphere.xyz, 1.0);
      const worldCenter = vec3(
        dot(row0, localPosition),
        dot(row1, localPosition),
        dot(row2, localPosition),
      );
      const worldRadius = localSphere.w.mul(objectParams.x);

      this.clusterWorldNode.element(instanceIndex).assign(vec4(worldCenter, worldRadius));
    })().compute(this.clusterCount, workgroup);

    this.cullScoreComputeNode = Fn(() => {
      const worldSphere = this.clusterWorldNode.element(instanceIndex);
      const metric = this.clusterMetricNode.element(instanceIndex);
      const center = worldSphere.xyz;
      const radius = worldSphere.w;
      const visibility = float(1).toVar();

      for (const plane of this.frustumPlanes) {
        const signedDistance = dot(plane.xyz, center).add(plane.w);
        If(signedDistance.lessThan(radius.negate()), () => {
          visibility.assign(0);
        });
      }

      const distanceToCamera = max(length(center.sub(this.cameraParamsA.xyz)), float(0.001));
      const projectedSizePx = float(0).toVar();

      If(this.cameraParamsB.x.greaterThan(0.5), () => {
        projectedSizePx.assign(
          radius
            .div(distanceToCamera.mul(max(this.cameraParamsB.y, float(0.001))))
            .mul(this.cameraParamsA.w),
        );
      }).Else(() => {
        projectedSizePx.assign(
          radius.mul(this.cameraParamsA.w).div(max(this.cameraParamsB.z, float(0.001))),
        );
      });

      const densityPenalty = float(1).add(metric.y.div(max(this.budgetParams.x, float(1))));
      const score = projectedSizePx.mul(metric.x).mul(visibility).div(densityPenalty);
      this.clusterScoreNode.element(instanceIndex).assign(
        vec4(score, projectedSizePx, projectedSizePx, visibility),
      );
    })().compute(this.clusterCount, workgroup);
  }

  private createStorageAttribute(count: number, itemSize: number): StorageInstancedBufferAttribute {
    const attribute = new StorageInstancedBufferAttribute(new Float32Array(count * itemSize), itemSize);
    attribute.setUsage(DynamicDrawUsage);
    return attribute;
  }
}
