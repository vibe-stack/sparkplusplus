import { Box3, Scene, Vector3 } from 'three';
import { SplatMesh } from '../scene/splat-mesh';
import { SPLAT_SCENE_DESCRIPTOR_FLOATS } from './layouts';

export interface SplatSceneObjectDescriptor {
  mesh: SplatMesh;
  matrixDirty: boolean;
  worldCenter: [number, number, number];
  worldRadius: number;
}

export interface SplatSceneDescriptorBuildResult {
  objects: SplatSceneObjectDescriptor[];
  dirtyObjectCount: number;
  packedBuffer: Float32Array;
}

export class SplatSceneDescriptorBuilder {
  private readonly matrixCache = new Map<string, Float32Array>();
  private readonly localBox = new Box3();
  private readonly worldBox = new Box3();
  private readonly center = new Vector3();
  private readonly size = new Vector3();

  build(scene: Scene): SplatSceneDescriptorBuildResult {
    const objects: SplatSceneObjectDescriptor[] = [];
    let dirtyObjectCount = 0;

    scene.traverseVisible((object) => {
      if (!(object instanceof SplatMesh)) {
        return;
      }

      const asset = object.getAsset();
      const matrixElements = object.matrixWorld.elements;
      const previousMatrix = this.matrixCache.get(object.uuid);
      let matrixDirty = false;

      if (!previousMatrix) {
        matrixDirty = true;
        this.matrixCache.set(object.uuid, new Float32Array(matrixElements));
      } else {
        for (let i = 0; i < matrixElements.length; i += 1) {
          if (Math.abs(previousMatrix[i]! - matrixElements[i]!) > 1e-5) {
            matrixDirty = true;
            previousMatrix.set(matrixElements);
            break;
          }
        }
      }

      if (matrixDirty || object.getMaterialVersion() > 0 || object.getEffectVersion() > 0) {
        dirtyObjectCount += 1;
      }

      this.localBox.min.set(...asset.localBoundsMin);
      this.localBox.max.set(...asset.localBoundsMax);
      this.worldBox.copy(this.localBox).applyMatrix4(object.matrixWorld);
      this.worldBox.getCenter(this.center);
      this.worldBox.getSize(this.size);

      objects.push({
        mesh: object,
        matrixDirty,
        worldCenter: [this.center.x, this.center.y, this.center.z],
        worldRadius: this.size.length() * 0.5,
      });
    });

    const packedBuffer = new Float32Array(objects.length * SPLAT_SCENE_DESCRIPTOR_FLOATS);

    objects.forEach((descriptor, objectIndex) => {
      const offset = objectIndex * SPLAT_SCENE_DESCRIPTOR_FLOATS;
      packedBuffer.set(descriptor.mesh.matrixWorld.elements, offset);
      packedBuffer[offset + 16] = descriptor.worldCenter[0];
      packedBuffer[offset + 17] = descriptor.worldCenter[1];
      packedBuffer[offset + 18] = descriptor.worldCenter[2];
      packedBuffer[offset + 19] = descriptor.worldRadius;
      packedBuffer[offset + 20] = descriptor.mesh.importance;
      packedBuffer[offset + 21] = descriptor.mesh.getAsset().totalSplatCount;
      packedBuffer[offset + 22] = descriptor.mesh.getAsset().rootClusterIds.length;
      packedBuffer[offset + 23] = descriptor.matrixDirty ? 1 : 0;
    });

    return {
      objects,
      dirtyObjectCount,
      packedBuffer,
    };
  }
}
