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
})
