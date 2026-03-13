// Shared type declarations for renderer — mirrors main-process interfaces
// without importing across the process boundary.

import type {
  AppSettings,
  AppSettingsBundle,
  AppSettingsInput,
  CoordinatorStatus,
  ProcessingSettings,
  ThumbnailCacheConfig,
  ThumbnailCacheConfigInput,
  WorkerPathMapping,
  WorkerSetupInfo,
} from '../main/settings'

export interface XmpData {
  description?: string
  sceneType?: string
  mainSubject?: string
  lighting?: string
  mood?: string
  aestheticScore?: number
  aestheticLabel?: string
  perceptionIAA?: number
  perceptionIAALabel?: string
  perceptionIQA?: number
  perceptionIQALabel?: string
  perceptionISTA?: number
  perceptionISTALabel?: string
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

export type {
  AppSettings,
  AppSettingsBundle,
  AppSettingsInput,
  CoordinatorStatus,
  ProcessingSettings,
  ThumbnailCacheConfig,
  ThumbnailCacheConfigInput,
  WorkerPathMapping,
  WorkerSetupInfo,
} from '../main/settings'

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

export interface AnalysisRunResult {
  xmp: XmpData | null
  error?: string
  cancelled?: boolean
}

// ── Search types ──────────────────────────────────────────────────────────────

export interface SearchFilters {
  query?: string
  mode?: 'text' | 'semantic' | 'hybrid' | 'browse'
  semanticWeight?: number
  intent?: SearchIntent
  similarToImageId?: number
  country?: string
  recurringMonthDay?: string
  timeOfDay?: SearchTimeOfDay
  sortBy?: SearchSortBy
  expandedTerms?: string[]
  face?: string
  faces?: string[]
  faceMatch?: SearchFaceMatch
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

export type SearchIntent = 'people' | 'wildlife' | 'best-shot' | 'general'
export type SearchTimeOfDay = 'morning' | 'afternoon' | 'evening' | 'night'
export type SearchSortBy = 'relevance' | 'best' | 'aesthetic' | 'sharpness' | 'cleanest' | 'newest'
export type SearchFaceMatch = 'any' | 'all'

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
  cloud_description: string | null
  aesthetic_score: number | null
  aesthetic_label: string | null
  aesthetic_reason: string | null
  perception_iaa: number | null
  perception_iaa_label: string | null
  perception_iqa: number | null
  perception_iqa_label: string | null
  perception_ista: number | null
  perception_ista_label: string | null
}

export interface SearchResponse {
  results: SearchResult[]
  total: number | null
  hasMore: boolean
  error?: string
}

export interface SearchFaceResolution {
  face: string | null
  faces: string[]
  faceMatch: SearchFaceMatch
  remainingQuery: string
  error?: string
}

export interface SearchPlanRequest {
  prompt: string
  model?: string
  intent?: SearchIntent
}

export interface SearchPlanResponse {
  intent: SearchIntent
  filters: SearchFilters
  summary: string
  model: string
  error?: string
}

export interface GalleryFolderNode {
  path: string
  name: string
  parent_path: string | null
  depth: number
  image_count: number
  child_count: number
}

export interface GalleryChunkParams {
  folderPath?: string | null
  recursive?: boolean
  chunkSize?: number
  cursor?: string | null
}

export interface GalleryChunkResponse {
  items: SearchResult[]
  nextCursor: string | null
  hasMore: boolean
  total: number | null
  error?: string
}

// ── Batch processing types ────────────────────────────────────────────────────

export type BatchModuleKey =
  | 'metadata'
  | 'technical'
  | 'caption'
  | 'objects'
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
  imagesPerSec: number
  avgMsPerImage: number
}

export interface BatchQueueSummary {
  totalPasses: number
  activePasses: number
  completedPasses: number
  remainingPasses: number
  remainingJobs: number
}

export interface BatchActiveModule {
  module: string
  count: number
}

export interface BatchNode {
  id: string
  role: 'master' | 'worker'
  label: string
  status: string
  platform?: string
  lastHeartbeat?: string | null
  lastResultAt?: string | null
  runningJobs: number
  completedJobs: number
  doneJobs: number
  failedJobs: number
  skippedJobs: number
  imagesPerSec: number
  avgMsPerImage: number
  capabilities?: Record<string, unknown>
  activeModules: BatchActiveModule[]
}

export interface BatchStats {
  status: BatchStatus
  monitorOnly: boolean
  totalImages: number
  modules: Partial<Record<string, BatchModuleStats>>
  totals: { pending: number; running: number; done: number; failed: number; skipped: number }
  avgMsPerImage: number
  imagesPerSec: number
  estimatedMs: number
  elapsedMs: number
  queue: BatchQueueSummary
  nodes: BatchNode[]
}

export interface BatchResult {
  id: string
  jobId?: number
  path: string
  module: string
  status: 'done' | 'failed' | 'skipped'
  durationMs: number
  error?: string
  keywords?: string[]
  nodeId: string
  nodeRole: 'master' | 'worker'
  nodeLabel: string
  completedAt?: string
}

export interface BatchIngestProgress {
  scanned: number
  total: number
  registered: number
  enqueued: number
  skipped: number
  current: string
}

// ── Face management types ─────────────────────────────────────────────────────

export interface FaceSummary {
  canonical_name: string
  display_name: string | null
  image_count: number
  identity_id: number | null
}

export interface FaceImage {
  image_id: number
  file_path: string
  face_count: number
}

export interface FacePerson {
  id: number
  name: string
  notes: string | null
  cluster_count: number
  face_count: number
  image_count: number
  representative_id: number | null
}

export interface PersonCluster {
  cluster_id: number
  face_count: number
  image_count: number
  label: string
  representative_id: number | null
}

export interface FaceCluster {
  cluster_id: number | null
  identity_name: string
  display_name: string | null
  identity_id: number | null
  image_count: number
  face_count: number
  representative_id: number | null
  person_id: number | null
}

export interface FaceOccurrence {
  id: number
  image_id: number
  file_path: string
  face_idx: number
  bbox_x1: number
  bbox_y1: number
  bbox_x2: number
  bbox_y2: number
  age: number | null
  gender: string | null
  identity_name: string
}

declare global {
  interface Window {
    api: {
      openFolder(): Promise<string | null>
      listImages(folderPath: string): Promise<ImageFile[]>
      getThumbnail(imagePath: string): Promise<string>
      getFullImage(imagePath: string): Promise<string>
      openPath(filePath: string): Promise<string>
      copyText(value: string): Promise<void>
      readXmp(imagePath: string): Promise<XmpData | null>
      runAnalysis(imagePath: string, aiBackend: string): Promise<AnalysisRunResult>
      cancelAnalysis(imagePath: string): Promise<void>
      runCopilotAnalysis(imagePath: string): Promise<{ xmp: XmpData | null; error?: string }>
      onAnalysisProgress(cb: (p: AnalysisProgress) => void): () => void

      // Search
      searchImages(filters: SearchFilters): Promise<SearchResponse>
      resolveSearchFaceQuery(query: string): Promise<SearchFaceResolution>
      planSearchQuery(request: SearchPlanRequest): Promise<SearchPlanResponse>
      galleryListFolders(): Promise<{ folders: GalleryFolderNode[]; totalImages: number; error?: string }>
      galleryListImagesChunk(params: GalleryChunkParams): Promise<GalleryChunkResponse>
      getThumbnailCacheConfig(): Promise<ThumbnailCacheConfig>
      setThumbnailCacheConfig(config: ThumbnailCacheConfigInput): Promise<ThumbnailCacheConfig>
      getAppSettings(): Promise<AppSettingsBundle>
      saveAppSettings(input: AppSettingsInput): Promise<AppSettingsBundle>
      getCoordinatorStatus(): Promise<CoordinatorStatus>
      startCoordinator(): Promise<CoordinatorStatus>
      stopCoordinator(): Promise<CoordinatorStatus>

      // Face management
      listFaces(): Promise<{ faces: FaceSummary[]; error?: string }>
      getFaceImages(name: string, limit?: number): Promise<{ images: FaceImage[]; error?: string }>
      setFaceAlias(canonicalName: string, displayName: string, clusterId?: number | null): Promise<{ ok: boolean; error?: string }>
      listFaceClusters(limit?: number, offset?: number): Promise<{ clusters: FaceCluster[]; has_occurrences: boolean; total_count: number; error?: string }>
      getFaceClusterImages(clusterId: number | null, identityName: string | null, limit?: number): Promise<{ occurrences: FaceOccurrence[]; error?: string }>
      relinkFaceCluster(clusterId: number, displayName: string | null, personId?: number | null, updatePerson?: boolean): Promise<{ ok: boolean; updated: number; error?: string }>
      getFaceCrop(occurrenceId: number): Promise<{ data?: string; error?: string }>
      getFaceCropBatch(ids: number[]): Promise<{ thumbnails: Record<string, string>; error?: string }>
      runFaceClustering(threshold?: number): Promise<{ started: boolean; error?: string }>
      onClusteringDone(cb: (result: { num_clusters?: number; error?: string }) => void): () => void
      rebuildFaces(): Promise<{ enqueued: number; error?: string }>

      // Person (cross-age identity grouping)
      listPersons(): Promise<{ persons: FacePerson[]; error?: string }>
      createPerson(name: string): Promise<{ id: number; error?: string }>
      renamePerson(personId: number, name: string): Promise<{ ok: boolean; error?: string }>
      deletePerson(personId: number): Promise<{ ok: boolean; error?: string }>
      linkClusterToPerson(clusterId: number, personId: number): Promise<{ ok: boolean; updated: number; error?: string }>
      unlinkClusterFromPerson(clusterId: number): Promise<{ ok: boolean; updated: number; error?: string }>
      getPersonClusters(personId: number): Promise<{ clusters: PersonCluster[]; error?: string }>

      // Batch processing
      batchIngest(
        folder: string,
        modules: string[],
        recursive: boolean,
        noHash: boolean,
        forceReprocess?: boolean
      ): Promise<{ registered: number; enqueued: number; skipped: number }>
      batchStart(
        folder: string,
        modules: string[],
        workers: number,
        cloudProvider: string,
        recursive: boolean,
        noHash: boolean,
        cloudWorkers: number,
        profile: boolean,
        chunkSize: number,
        forceReprocess?: boolean
      ): Promise<void>
      batchPause(): Promise<void>
      batchResume(): Promise<void>
      batchStop(folder: string): Promise<void>
      batchCheckPending(): Promise<{ pending: number; running: number }>
      batchMonitorExisting(): Promise<boolean>
      batchResumePending(workers?: number, cloudProvider?: string, cloudWorkers?: number): Promise<void>
      batchRetryFailed(modules: string[]): Promise<void>
      batchRebuildModule(module: string): Promise<void>
      batchQueueClearAll(): Promise<{ deleted: number }>
      batchQueueClearDone(): Promise<{ deleted: number }>
      onBatchTick(cb: (stats: BatchStats) => void): () => void
      onBatchResult(cb: (result: BatchResult) => void): () => void
      onBatchIngestLine(cb: (line: string) => void): () => void
      onBatchIngestProgress(cb: (progress: BatchIngestProgress) => void): () => void
    }
  }
}
