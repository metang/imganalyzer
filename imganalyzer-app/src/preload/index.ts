import { contextBridge, ipcRenderer } from 'electron'
import type { XmpData } from '../main/xmp'
import type { ImageFile } from '../main/images'
import type { AnalysisProgress } from '../main/analyzer'
import type { BatchStats, BatchResult, BatchIngestProgress } from '../main/batch'
import type { SearchFilters, SearchResponse } from '../main/search'

contextBridge.exposeInMainWorld('api', {
  openFolder: (): Promise<string | null> =>
    ipcRenderer.invoke('dialog:openFolder'),

  listImages: (folderPath: string): Promise<ImageFile[]> =>
    ipcRenderer.invoke('fs:listImages', folderPath),

  getThumbnail: (imagePath: string): Promise<string> =>
    ipcRenderer.invoke('fs:getThumbnail', imagePath),

  getFullImage: (imagePath: string): Promise<string> =>
    ipcRenderer.invoke('fs:getFullImage', imagePath),

  readXmp: (imagePath: string): Promise<XmpData | null> =>
    ipcRenderer.invoke('fs:readXmp', imagePath),

  runAnalysis: (imagePath: string, aiBackend: string): Promise<{ xmp: XmpData | null; error?: string }> =>
    ipcRenderer.invoke('analyze:run', imagePath, aiBackend),

  cancelAnalysis: (imagePath: string): Promise<void> =>
    ipcRenderer.invoke('analyze:cancel', imagePath),

  runCopilotAnalysis: (imagePath: string): Promise<{ xmp: XmpData | null; error?: string }> =>
    ipcRenderer.invoke('analyze:copilot', imagePath),

  onAnalysisProgress: (cb: (p: AnalysisProgress) => void) => {
    const handler = (_evt: Electron.IpcRendererEvent, p: AnalysisProgress) => cb(p)
    ipcRenderer.on('analyze:progress', handler)
    return () => ipcRenderer.removeListener('analyze:progress', handler)
  },

  // ── Batch processing ────────────────────────────────────────────────────────

  batchIngest: (
    folder: string,
    modules: string[],
    recursive: boolean,
    noHash: boolean
  ): Promise<{ registered: number; enqueued: number; skipped: number }> =>
    ipcRenderer.invoke('batch:ingest', folder, modules, recursive, noHash),

  batchStart: (
    folder: string,
    modules: string[],
    workers: number,
    cloudProvider: string,
    recursive: boolean,
    noHash: boolean,
    cloudWorkers: number
  ): Promise<void> =>
    ipcRenderer.invoke('batch:start', folder, modules, workers, cloudProvider, recursive, noHash, cloudWorkers),

  batchPause: (): Promise<void> =>
    ipcRenderer.invoke('batch:pause'),

  batchResume: (): Promise<void> =>
    ipcRenderer.invoke('batch:resume'),

  batchStop: (folder: string): Promise<void> =>
    ipcRenderer.invoke('batch:stop', folder),

  batchCheckPending: (): Promise<{ pending: number; running: number }> =>
    ipcRenderer.invoke('batch:check-pending'),

  batchResumePending: (workers?: number, cloudProvider?: string, cloudWorkers?: number): Promise<void> =>
    ipcRenderer.invoke('batch:resume-pending', workers, cloudProvider, cloudWorkers),

  batchRetryFailed: (modules: string[]): Promise<void> =>
    ipcRenderer.invoke('batch:retry-failed', modules),

  batchQueueClearAll: (): Promise<{ deleted: number }> =>
    ipcRenderer.invoke('batch:queue-clear-all'),

  batchQueueClearDone: (): Promise<{ deleted: number }> =>
    ipcRenderer.invoke('batch:queue-clear-done'),

  onBatchTick: (cb: (stats: BatchStats) => void) => {
    const handler = (_evt: Electron.IpcRendererEvent, stats: BatchStats) => cb(stats)
    ipcRenderer.on('batch:tick', handler)
    return () => ipcRenderer.removeListener('batch:tick', handler)
  },

  onBatchResult: (cb: (result: BatchResult) => void) => {
    const handler = (_evt: Electron.IpcRendererEvent, result: BatchResult) => cb(result)
    ipcRenderer.on('batch:result', handler)
    return () => ipcRenderer.removeListener('batch:result', handler)
  },

  onBatchIngestLine: (cb: (line: string) => void) => {
    const handler = (_evt: Electron.IpcRendererEvent, line: string) => cb(line)
    ipcRenderer.on('batch:ingest-line', handler)
    return () => ipcRenderer.removeListener('batch:ingest-line', handler)
  },

  onBatchIngestProgress: (cb: (progress: BatchIngestProgress) => void) => {
    const handler = (_evt: Electron.IpcRendererEvent, progress: BatchIngestProgress) => cb(progress)
    ipcRenderer.on('batch:ingest-progress', handler)
    return () => ipcRenderer.removeListener('batch:ingest-progress', handler)
  },

  // ── Search ──────────────────────────────────────────────────────────────────

  searchImages: (filters: SearchFilters): Promise<SearchResponse> =>
    ipcRenderer.invoke('search:run', filters),

  // ── Face management ────────────────────────────────────────────────────────

  listFaces: (): Promise<{ faces: Array<{ canonical_name: string; display_name: string | null; image_count: number; identity_id: number | null }>; error?: string }> =>
    ipcRenderer.invoke('faces:list'),

  getFaceImages: (name: string, limit?: number): Promise<{ images: Array<{ image_id: number; file_path: string; face_count: number }>; error?: string }> =>
    ipcRenderer.invoke('faces:images', name, limit),

  setFaceAlias: (canonicalName: string, displayName: string, clusterId?: number | null): Promise<{ ok: boolean; error?: string }> =>
    ipcRenderer.invoke('faces:setAlias', canonicalName, displayName, clusterId ?? null),

  listFaceClusters: (): Promise<{ clusters: Array<{ cluster_id: number | null; identity_name: string; display_name: string | null; identity_id: number | null; image_count: number; face_count: number; representative_id: number | null }>; has_occurrences: boolean; error?: string }> =>
    ipcRenderer.invoke('faces:clusters'),

  getFaceClusterImages: (clusterId: number | null, identityName: string | null, limit?: number): Promise<{ occurrences: Array<{ id: number; image_id: number; file_path: string; face_idx: number; bbox_x1: number; bbox_y1: number; bbox_x2: number; bbox_y2: number; age: number | null; gender: string | null; identity_name: string }>; error?: string }> =>
    ipcRenderer.invoke('faces:clusterImages', clusterId, identityName, limit),

  getFaceCrop: (occurrenceId: number): Promise<{ data?: string; error?: string }> =>
    ipcRenderer.invoke('faces:crop', occurrenceId),

  getFaceCropBatch: (ids: number[]): Promise<{ thumbnails: Record<string, string>; error?: string }> =>
    ipcRenderer.invoke('faces:cropBatch', ids),

  runFaceClustering: (threshold?: number): Promise<{ num_clusters: number; error?: string }> =>
    ipcRenderer.invoke('faces:runClustering', threshold),

  rebuildFaces: (): Promise<{ enqueued: number; error?: string }> =>
    ipcRenderer.invoke('faces:rebuild'),

  // Person (cross-age identity grouping)
  listPersons: (): Promise<{ persons: Array<{ id: number; name: string; notes: string | null; cluster_count: number; face_count: number; image_count: number; representative_id: number | null }>; error?: string }> =>
    ipcRenderer.invoke('faces:persons'),

  createPerson: (name: string): Promise<{ id: number; error?: string }> =>
    ipcRenderer.invoke('faces:personCreate', name),

  renamePerson: (personId: number, name: string): Promise<{ ok: boolean; error?: string }> =>
    ipcRenderer.invoke('faces:personRename', personId, name),

  deletePerson: (personId: number): Promise<{ ok: boolean; error?: string }> =>
    ipcRenderer.invoke('faces:personDelete', personId),

  linkClusterToPerson: (clusterId: number, personId: number): Promise<{ ok: boolean; updated: number; error?: string }> =>
    ipcRenderer.invoke('faces:personLinkCluster', clusterId, personId),

  unlinkClusterFromPerson: (clusterId: number): Promise<{ ok: boolean; updated: number; error?: string }> =>
    ipcRenderer.invoke('faces:personUnlinkCluster', clusterId),

  getPersonClusters: (personId: number): Promise<{ clusters: Array<{ cluster_id: number; face_count: number; image_count: number; label: string; representative_id: number | null }>; error?: string }> =>
    ipcRenderer.invoke('faces:personClusters', personId),
})
