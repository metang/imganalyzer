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

// ── Search types ──────────────────────────────────────────────────────────────

export interface SearchFilters {
  query?: string
  mode?: 'text' | 'semantic' | 'hybrid' | 'browse'
  semanticWeight?: number
  face?: string
  camera?: string
  lens?: string
  location?: string
  aestheticMin?: number
  aestheticMax?: number
  sharpnessMin?: number
  sharpnessMax?: number
  noiseMax?: number
  isoMin?: number
  isoMax?: number
  facesMin?: number
  facesMax?: number
  dateFrom?: string
  dateTo?: string
  hasPeople?: boolean
  limit?: number
  offset?: number
}

export interface SearchResult {
  image_id: number
  file_path: string
  score: number | null
  width: number | null
  height: number | null
  file_size: number | null
  camera_make: string | null
  camera_model: string | null
  lens_model: string | null
  focal_length: string | null
  f_number: string | null
  exposure_time: string | null
  iso: string | null
  date_time_original: string | null
  gps_latitude: string | null
  gps_longitude: string | null
  location_city: string | null
  location_state: string | null
  location_country: string | null
  sharpness_score: number | null
  sharpness_label: string | null
  exposure_ev: number | null
  exposure_label: string | null
  noise_level: number | null
  noise_label: string | null
  snr_db: number | null
  dynamic_range_stops: number | null
  highlight_clipping_pct: number | null
  shadow_clipping_pct: number | null
  avg_saturation: number | null
  dominant_colors: string[] | null
  description: string | null
  scene_type: string | null
  main_subject: string | null
  lighting: string | null
  mood: string | null
  keywords: string[] | null
  detected_objects: string[] | null
  face_count: number | null
  face_identities: string[] | null
  has_people: boolean | null
  ocr_text: string | null
  aesthetic_score: number | null
  aesthetic_label: string | null
  aesthetic_reason: string | null
}

export interface SearchResponse {
  results: SearchResult[]
  total: number
  error?: string
}

// ── Batch processing types ────────────────────────────────────────────────────

export type BatchModuleKey =
  | 'metadata'
  | 'technical'
  | 'local_ai'
  | 'blip2'
  | 'objects'
  | 'ocr'
  | 'faces'
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

      // Search
      searchImages(filters: SearchFilters): Promise<SearchResponse>

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
        noHash: boolean,
        cloudWorkers: number
      ): Promise<void>
      batchPause(): Promise<void>
      batchResume(): Promise<void>
      batchStop(folder: string): Promise<void>
      batchCheckPending(): Promise<{ pending: number; running: number }>
      batchResumePending(workers?: number, cloudProvider?: string, cloudWorkers?: number): Promise<void>
      batchRetryFailed(modules: string[]): Promise<void>
      batchQueueClearAll(): Promise<{ deleted: number }>
      onBatchTick(cb: (stats: BatchStats) => void): () => void
      onBatchResult(cb: (result: BatchResult) => void): () => void
      onBatchIngestLine(cb: (line: string) => void): () => void
      onBatchIngestProgress(cb: (progress: BatchIngestProgress) => void): () => void
    }
  }
}

