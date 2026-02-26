// Shared type declarations for renderer — mirrors main-process interfaces
// without importing across the process boundary.

export interface XmpData {
  description?: string
  sceneType?: string
  mainSubject?: string
  lighting?: string
  mood?: string
  aestheticScore?: number
  aestheticLabel?: string
  faceCount?: number
  faceIdentities?: string[]
  faceDetails?: string[]
  detectedObjects?: string[]
  ocrText?: string
  keywords?: string[]
  sharpnessScore?: number
  sharpnessLabel?: string
  exposureEV?: number
  exposureLabel?: string
  noiseLevel?: number
  noiseLabel?: string
  snrDb?: number
  dynamicRangeStops?: number
  highlightClippingPct?: number
  shadowClippingPct?: number
  avgSaturation?: number
  warmCoolRatio?: number
  dominantColors?: string[]
  cameraMake?: string
  cameraModel?: string
  lens?: string
  fNumber?: string
  exposureTime?: string
  focalLength?: string
  iso?: string
  createDate?: string
  imageWidth?: number
  imageHeight?: number
  gpsLatitude?: string
  gpsLongitude?: string
  locationCity?: string
  locationState?: string
  locationCountry?: string
}

export interface ImageFile {
  path: string
  name: string
  ext: string
  isRaw: boolean
  xmpPath: string
  hasXmp: boolean
  size: number
  mtime: number
}

export interface AnalysisProgress {
  imagePath: string
  stage: string
  pct: number
}

// ── Batch processing types ────────────────────────────────────────────────────

export type BatchModuleKey =
  | 'metadata'
  | 'technical'
  | 'local_ai'
  | 'cloud_ai'
  | 'aesthetic'
  | 'embedding'

export type BatchStatus =
  | 'idle'
  | 'ingesting'
  | 'running'
  | 'paused'
  | 'done'
  | 'stopped'
  | 'error'

export interface BatchModuleStats {
  pending: number
  running: number
  done: number
  failed: number
  skipped: number
}

export interface BatchStats {
  status: BatchStatus
  totalImages: number
  modules: Partial<Record<string, BatchModuleStats>>
  totals: { pending: number; running: number; done: number; failed: number; skipped: number }
  avgMsPerImage: number
  imagesPerSec: number
  estimatedMs: number
  elapsedMs: number
}

export interface BatchResult {
  path: string
  module: string
  status: 'done' | 'failed' | 'skipped'
  durationMs: number
  error?: string
}

export interface BatchIngestProgress {
  scanned: number
  total: number
  registered: number
  enqueued: number
  skipped: number
  current: string
}

declare global {
  interface Window {
    api: {
      openFolder(): Promise<string | null>
      listImages(folderPath: string): Promise<ImageFile[]>
      getThumbnail(imagePath: string): Promise<string>
      getFullImage(imagePath: string): Promise<string>
      readXmp(imagePath: string): Promise<XmpData | null>
      runAnalysis(imagePath: string, aiBackend: string): Promise<{ xmp: XmpData | null; error?: string }>
      cancelAnalysis(imagePath: string): Promise<void>
      runCopilotAnalysis(imagePath: string): Promise<{ xmp: XmpData | null; error?: string }>
      onAnalysisProgress(cb: (p: AnalysisProgress) => void): () => void

      // Batch processing
      batchIngest(
        folder: string,
        modules: string[],
        recursive: boolean,
        noHash: boolean
      ): Promise<{ registered: number; enqueued: number; skipped: number }>
      batchStart(
        folder: string,
        modules: string[],
        workers: number,
        cloudProvider: string,
        recursive: boolean,
        noHash: boolean
      ): Promise<void>
      batchPause(): Promise<void>
      batchResume(): Promise<void>
      batchStop(folder: string): Promise<void>
      batchCheckPending(): Promise<{ pending: number; running: number }>
      batchResumePending(workers?: number, cloudProvider?: string): Promise<void>
      batchRetryFailed(modules: string[]): Promise<void>
      onBatchTick(cb: (stats: BatchStats) => void): () => void
      onBatchResult(cb: (result: BatchResult) => void): () => void
      onBatchIngestLine(cb: (line: string) => void): () => void
      onBatchIngestProgress(cb: (progress: BatchIngestProgress) => void): () => void
    }
  }
}
