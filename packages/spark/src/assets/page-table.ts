import { SPLAT_PAGE_RESIDENCY_UINTS } from '../core/layouts';
import type { SplatAsset } from './model';

export const enum SplatPageResidencyState {
  Unloaded = 0,
  Requested = 1,
  Resident = 2,
}

interface SplatPageResidencyRecord {
  state: SplatPageResidencyState;
  requestFrame: number;
  residentFrame: number;
  lastTouchedFrame: number;
  faultCount: number;
}

export class SplatPageTable {
  private readonly records: SplatPageResidencyRecord[];
  private readonly requestQueue: number[] = [];
  private residentCapacity: number;
  private residentCount = 0;

  readonly residencyBuffer: Uint32Array;

  constructor(readonly asset: SplatAsset, residentCapacity = Math.min(32, asset.pages.length)) {
    this.residentCapacity = Math.max(1, residentCapacity);
    this.records = asset.pages.map(() => ({
      state: SplatPageResidencyState.Unloaded,
      requestFrame: 0,
      residentFrame: 0,
      lastTouchedFrame: 0,
      faultCount: 0,
    }));
    this.residencyBuffer = new Uint32Array(asset.pages.length * SPLAT_PAGE_RESIDENCY_UINTS);
  }

  getState(pageId: number): SplatPageResidencyRecord {
    return this.records[pageId]!;
  }

  setResidentCapacity(capacity: number): void {
    this.residentCapacity = Math.max(1, Math.floor(capacity));
  }

  request(pageId: number, frameIndex: number): boolean {
    const record = this.records[pageId];

    if (!record) {
      return false;
    }

    if (record.state === SplatPageResidencyState.Resident) {
      this.markTouched(pageId, frameIndex);
      return false;
    }

    if (record.state !== SplatPageResidencyState.Requested) {
      record.state = SplatPageResidencyState.Requested;
      record.requestFrame = frameIndex;
      record.faultCount += 1;
      this.requestQueue.push(pageId);
      this.syncBuffer(pageId);
      return true;
    }

    return false;
  }

  isResident(pageId: number): boolean {
    return this.records[pageId]?.state === SplatPageResidencyState.Resident;
  }

  markTouched(pageId: number, frameIndex: number): void {
    const record = this.records[pageId];

    if (!record) {
      return;
    }

    record.lastTouchedFrame = frameIndex;
    this.syncBuffer(pageId);
  }

  getResidentCount(): number {
    return this.residentCount;
  }

  getPendingCount(): number {
    return this.requestQueue.filter((pageId) => this.records[pageId]?.state === SplatPageResidencyState.Requested).length;
  }

  serviceRequests(
    maxUploads: number,
    frameIndex: number,
    protectedPageIds: ReadonlySet<number>,
  ): number[] {
    const uploadedPageIds: number[] = [];

    while (uploadedPageIds.length < maxUploads && this.requestQueue.length > 0) {
      const pageId = this.requestQueue.shift();

      if (pageId === undefined) {
        break;
      }

      const record = this.records[pageId];

      if (!record || record.state !== SplatPageResidencyState.Requested) {
        continue;
      }

      if (this.residentCount >= this.residentCapacity) {
        const evicted = this.evictColdest(protectedPageIds);

        if (evicted === null) {
          break;
        }
      }

      record.state = SplatPageResidencyState.Resident;
      record.residentFrame = frameIndex;
      record.lastTouchedFrame = frameIndex;
      this.residentCount += 1;
      uploadedPageIds.push(pageId);
      this.syncBuffer(pageId);
    }

    return uploadedPageIds;
  }

  private evictColdest(protectedPageIds: ReadonlySet<number>): number | null {
    let evictionCandidate = -1;
    let oldestFrame = Number.POSITIVE_INFINITY;

    this.records.forEach((record, pageId) => {
      if (record.state !== SplatPageResidencyState.Resident) {
        return;
      }

      if (protectedPageIds.has(pageId)) {
        return;
      }

      if (record.lastTouchedFrame < oldestFrame) {
        oldestFrame = record.lastTouchedFrame;
        evictionCandidate = pageId;
      }
    });

    if (evictionCandidate === -1) {
      return null;
    }

    const record = this.records[evictionCandidate]!;
    record.state = SplatPageResidencyState.Unloaded;
    this.residentCount -= 1;
    this.syncBuffer(evictionCandidate);
    return evictionCandidate;
  }

  private syncBuffer(pageId: number): void {
    const record = this.records[pageId]!;
    const offset = pageId * SPLAT_PAGE_RESIDENCY_UINTS;
    this.residencyBuffer[offset + 0] = record.state;
    this.residencyBuffer[offset + 1] = record.requestFrame;
    this.residencyBuffer[offset + 2] = record.residentFrame;
    this.residencyBuffer[offset + 3] = record.lastTouchedFrame;
    this.residencyBuffer[offset + 4] = record.faultCount;
  }
}
