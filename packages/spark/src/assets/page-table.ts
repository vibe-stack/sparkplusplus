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
  requestSerial: number;
  residentFrame: number;
  lastTouchedFrame: number;
  faultCount: number;
}

export class SplatPageTable {
  private readonly records: SplatPageResidencyRecord[];
  private requestSerialCounter = 0;
  private residentCapacity: number;
  private residentCount = 0;

  readonly residencyBuffer: Uint32Array;

  constructor(readonly asset: SplatAsset, residentCapacity = Math.min(32, asset.pages.length)) {
    this.residentCapacity = Math.max(1, residentCapacity);
    this.records = asset.pages.map(() => ({
      state: SplatPageResidencyState.Unloaded,
      requestFrame: 0,
      requestSerial: 0,
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
      record.requestSerial = this.nextRequestSerial();
      record.faultCount += 1;
      this.syncBuffer(pageId);
      return true;
    }

    // Refresh requested pages when they are demanded again so uploads prioritize
    // the most recently visible region instead of draining stale FIFO work from
    // an old camera view.
    record.requestFrame = frameIndex;
    record.requestSerial = this.nextRequestSerial();
    this.syncBuffer(pageId);

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
    return this.records.filter((record) => record.state === SplatPageResidencyState.Requested).length;
  }

  serviceRequests(
    maxUploads: number,
    frameIndex: number,
    protectedPageIds: ReadonlySet<number>,
    requestPriorityByPage?: ReadonlyMap<number, number>,
  ): number[] {
    const uploadedPageIds: number[] = [];

    while (uploadedPageIds.length < maxUploads) {
      const pageId = this.selectRequestedPageId(requestPriorityByPage);

      if (pageId === null) {
        break;
      }

      const record = this.records[pageId]!;

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

  private selectRequestedPageId(requestPriorityByPage?: ReadonlyMap<number, number>): number | null {
    let selectedPageId: number | null = null;
    let bestPriority = Number.NEGATIVE_INFINITY;
    let newestRequestSerial = -1;

    this.records.forEach((record, pageId) => {
      if (record.state !== SplatPageResidencyState.Requested) {
        return;
      }

      const priority = requestPriorityByPage?.get(pageId) ?? 0;

      if (
        priority > bestPriority
        || (priority === bestPriority && record.requestSerial > newestRequestSerial)
      ) {
        bestPriority = priority;
        newestRequestSerial = record.requestSerial;
        selectedPageId = pageId;
      }
    });

    return selectedPageId;
  }

  private nextRequestSerial(): number {
    this.requestSerialCounter += 1;
    return this.requestSerialCounter;
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
