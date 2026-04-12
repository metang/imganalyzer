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
  semanticProfile?: SearchSemanticProfile
  intent?: SearchIntent
  similarToImageId?: number
  country?: string
  recurringMonthDay?: string
  timeOfDay?: SearchTimeOfDay
  sortBy?: SearchSortBy
  rankPreference?: SearchRankPreference
  expandedTerms?: string[]
  mustTerms?: string[]
  shouldTerms?: string[]
  debugSearch?: boolean
  facetRequest?: boolean
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
  mapBounds?: { north: number; south: number; east: number; west: number }
  limit?: number
  offset?: number
}

export type SearchIntent = 'people' | 'wildlife' | 'best-shot' | 'general'
export type SearchTimeOfDay = 'morning' | 'afternoon' | 'evening' | 'night'
export type SearchSortBy = 'relevance' | 'best' | 'aesthetic' | 'sharpness' | 'cleanest' | 'newest'
export type SearchFaceMatch = 'any' | 'all'
export type SearchRankPreference = 'relevance' | 'quality' | 'recency' | 'aesthetic' | 'cleanest' | 'sharpest'
export type SearchSemanticProfile = 'image_dominant' | 'balanced' | 'description_dominant'

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
  face_clusters: {
    cluster_id: number
    cluster_label: string | null
    person_id: number | null
    person_name: string | null
    face_count: number
  }[] | null
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

export interface SearchProgress {
  phase: string
  message: string
  progress: number
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

export type BatchPauseMode = 'pause-drain' | 'pause-immediate'

export interface BatchControlTarget {
  scope: 'coordinator' | 'master' | 'worker'
  workerId?: string
}

export interface BatchCoordinatorStatus {
  state: 'stopped' | 'starting' | 'running' | 'error'
  pid?: number | null
  url?: string | null
  lastError?: string | null
}

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
  desiredState?: string
  stateReason?: string | null
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
  coordinator: BatchCoordinatorStatus
  totalImages: number
  modules: Partial<Record<string, BatchModuleStats>>
  totals: { pending: number; running: number; done: number; failed: number; skipped: number }
  avgMsPerImage: number
  imagesPerSec: number
  estimatedMs: number
  elapsedMs: number
  chunkAvgCompletionMs: number
  chunkElapsedMs: number
  chunkEstimatedMs: number
  queue: BatchQueueSummary
  nodes: BatchNode[]
  chunk?: { size: number; index: number; total: number; modules: Record<string, number> }
  preDecode?: { done: number; failed: number; total: number; running: boolean }
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

// ── Albums / Storyline types ──────────────────────────────────────────────────

export interface SmartAlbumSummary {
  id: string
  name: string
  description: string | null
  cover_image_id: number | null
  story_enabled: boolean
  sort_order: string
  item_count: number
  chapter_count: number
  created_at: string
  updated_at: string
}

export interface AlbumRules {
  match: 'all' | 'any'
  rules: Array<Record<string, unknown>>
}

export interface StoryChapter {
  id: string
  album_id: string
  title: string | null
  summary: string | null
  sort_order: number
  start_date: string | null
  end_date: string | null
  location: string | null
  cover_image_id: number | null
  image_count: number
  moment_count: number
}

export interface StoryMoment {
  id: string
  chapter_id: string
  title: string | null
  sort_order: number
  start_time: string | null
  end_time: string | null
  lat: number | null
  lng: number | null
  hero_image_id: number | null
  image_count: number
}

export interface MomentImage {
  image_id: number
  sort_order: number
  is_hero: number
  date_time_original: string | null
  perception_iaa: number | null
}

export interface StoryGenerateResult {
  images: number
  moments: number
  chapters: number
  generation_time_s: number
  evaluation: {
    album_id: string
    overall_pass: boolean
    criteria: Record<string, {
      name: string
      passed: boolean
      value: unknown
      threshold: unknown
      detail?: string
    }>
  }
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
  representative_thumbnail: string | null
}

export interface PersonCluster {
  cluster_id: number
  face_count: number
  image_count: number
  label: string
  representative_id: number | null
  representative_thumbnail: string | null
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

export interface FaceLinkSuggestion {
  target_type: 'person' | 'alias'
  label: string
  person_id: number | null
  cluster_id: number | null
  score: number
  representative_id: number | null
  face_count: number
  reason: string
}

export interface PersonLinkSuggestion {
  cluster_id: number
  label: string
  score: number
  representative_id: number | null
  face_count: number
  image_count: number
  reason: string
}

export interface PersonSimilarImage {
  image_id: number
  file_path: string
  similarity: number
  best_occurrence_id: number
}

export interface PersonDirectLink {
  occurrence_id: number
  image_id: number
  file_path: string
}

export interface GeoCluster {
  cell: string
  center_lat: number
  center_lng: number
  count: number
  sample_ids: number[]
}

export interface GeoNearbyImage {
  image_id: number
  gps_latitude: number
  gps_longitude: number
  file_path: string
}

export interface GeoStatsExtended {
  total_images: number
  geotagged: number
  gps_sources: Array<{ source: string; count: number }>
  countries: Array<{ country: string; count: number }>
  top_cities: Array<{ city: string; state: string; country: string; count: number }>
  monthly_activity: Array<{ month: string; count: number }>
  location_diversity: Array<{ month: string; unique_places: number }>
  camera_by_country: Array<{ country: string; camera: string; count: number }>
  top_locations: Array<{
    cell: string
    lat: number
    lng: number
    count: number
    city: string | null
    state: string | null
    country: string | null
  }>
  furthest_from_home: {
    image_id: number
    file_path: string
    lat: number
    lng: number
    distance_km: number
  } | null
  error?: string
}

export interface GapFillerPreviewItem {
  image_id: number
  file_path: string
  inferred_lat: number
  inferred_lng: number
  confidence: number
  nearest_before?: { image_id: number; gap_minutes: number }
  nearest_after?: { image_id: number; gap_minutes: number }
}

export interface GapFillerPreviewResponse {
  fillable: number
  total_missing: number
  previews: GapFillerPreviewItem[]
  error?: string
}

export interface GapFillerApplyResponse {
  filled: number
  skipped_override: number
  skipped_low_confidence: number
  error?: string
}

export interface TripDetectResult {
  start_date: string
  end_date: string
  start_location: string
  end_location: string
  image_count: number
  distance_km: number
}

export interface TripDetectResponse {
  trips: TripDetectResult[]
  error?: string
}

export interface TripStop {
  lat: number
  lng: number
  start_time: string
  end_time: string
  count: number
  cover_image_id: number
  cover_file_path: string
}

export interface TripTimelineResponse {
  stops: TripStop[]
  route_points: Array<{ lat: number; lng: number }>
  total_images: number
  error?: string
}

declare global {
  interface Window {
    api: {
      openFolder(): Promise<string | null>
      saveStoryExport(defaultPath?: string): Promise<string | null>
      listImages(folderPath: string): Promise<ImageFile[]>
      getThumbnail(imagePath: string): Promise<string>
      getThumbnailsBatch(items: Array<{ file_path?: string; image_id?: number }>): Promise<Record<string, string>>
      getFullImage(imagePath: string): Promise<string>
      getCachedImage(imagePath: string): Promise<string>
      openPath(filePath: string): Promise<string>
      copyText(value: string): Promise<void>
      readXmp(imagePath: string): Promise<XmpData | null>
      runAnalysis(imagePath: string, aiBackend: string): Promise<AnalysisRunResult>
      cancelAnalysis(imagePath: string): Promise<void>
      runCopilotAnalysis(imagePath: string): Promise<{ xmp: XmpData | null; error?: string }>
      onAnalysisProgress(cb: (p: AnalysisProgress) => void): () => void

      // Search
      searchImages(filters: SearchFilters): Promise<SearchResponse>
      onSearchProgress(cb: (progress: SearchProgress) => void): () => void
      getImageDetails(params: { image_id?: number; file_path?: string }): Promise<{ result: SearchResult | null; error?: string }>
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

      // Albums / Storyline
      albumsList(): Promise<{ albums: SmartAlbumSummary[] }>
      albumsCreate(params: { name: string; rules: AlbumRules; description?: string; story_enabled?: boolean; sort_order?: string }): Promise<{ id: string; item_count: number }>
      albumsGet(albumId: string): Promise<SmartAlbumSummary & { rules: AlbumRules }>
      albumsUpdate(params: { album_id: string; name?: string; description?: string; rules?: AlbumRules; story_enabled?: boolean; sort_order?: string }): Promise<{ id: string; item_count: number } | { error: string }>
      albumsDelete(albumId: string): Promise<{ deleted: boolean }>
      albumsRefresh(albumId: string): Promise<{ item_count: number }>
      albumsStory(albumId: string): Promise<{ chapters: StoryChapter[] }>
      albumsStoryGenerate(params: { album_id: string; time_window_minutes?: number; chapter_gap_hours?: number; chapter_distance_km?: number; force_year_breaks?: boolean }): Promise<StoryGenerateResult>
      albumsChapterMoments(chapterId: string): Promise<{ moments: StoryMoment[] }>
      albumsMomentImages(momentId: string): Promise<{ images: MomentImage[] }>
      albumsCheckNew(imageId: number): Promise<{ added_to_albums: string[] }>
      albumsGenerateNarrative(params: { album_id: string; use_ai?: boolean }): Promise<{ chapters_updated: number }>
      albumsExport(params: { album_id: string; output_path: string; include_thumbnails?: boolean; max_heroes_per_chapter?: number }): Promise<{ path: string }>
      albumsPresets(): Promise<{ presets: Record<string, { name: string; description: string; params: string[] }> }>
      albumsCreatePreset(params: { preset: string; [key: string]: unknown }): Promise<{ id: string; name: string; item_count: number } | { error: string }>

      // Face management
      listFaces(): Promise<{ faces: FaceSummary[]; error?: string }>
      getFaceImages(name: string, limit?: number): Promise<{ images: FaceImage[]; error?: string }>
      setFaceAlias(canonicalName: string, displayName: string, clusterId?: number | null): Promise<{ ok: boolean; error?: string }>
      listFaceClusters(limit?: number, offset?: number): Promise<{ clusters: FaceCluster[]; has_occurrences: boolean; total_count: number; deferred_cluster_ids: number[]; error?: string }>
      getFaceClusterImages(clusterId: number | null, identityName: string | null, limit?: number): Promise<{ occurrences: FaceOccurrence[]; error?: string }>
      relinkFaceCluster(clusterId: number, displayName: string | null, personId?: number | null, updatePerson?: boolean): Promise<{ ok: boolean; updated: number; error?: string }>
      deferFaceCluster(clusterId: number): Promise<{ ok: boolean; error?: string }>
      undeferFaceCluster(clusterId: number): Promise<{ ok: boolean; error?: string }>
      undeferAllFaceClusters(): Promise<{ ok: boolean; cleared: number; error?: string }>
      splitCluster(clusterId: number, threshold?: number): Promise<{ split_count: number; new_cluster_ids: number[]; error?: string }>
      getClusterPurity(clusterId: number): Promise<{ purity_score: number; member_count: number; error?: string }>
      getClusterLinkSuggestions(clusterId: number, limit?: number): Promise<{ suggestions: FaceLinkSuggestion[]; error?: string }>
      getFaceCrop(occurrenceId: number): Promise<{ data?: string; error?: string }>
      getFaceCropBatch(ids: number[]): Promise<{ thumbnails: Record<string, string>; error?: string }>
      runFaceClustering(threshold?: number): Promise<{ started: boolean; error?: string }>
      onClusteringDone(cb: (result: { num_clusters?: number; error?: string }) => void): () => void
      onClusteringProgress(cb: (progress: { phase: string; fraction: number; numClusters: number }) => void): () => void
      rebuildFaces(): Promise<{ enqueued: number; error?: string }>

      // Person (cross-age identity grouping)
      listPersons(): Promise<{ persons: FacePerson[]; error?: string }>
      createPerson(name: string): Promise<{ id: number; error?: string }>
      renamePerson(personId: number, name: string): Promise<{ ok: boolean; error?: string }>
      deletePerson(personId: number): Promise<{ ok: boolean; error?: string }>
      linkClusterToPerson(clusterId: number, personId: number): Promise<{ ok: boolean; updated: number; error?: string }>
      unlinkClusterFromPerson(clusterId: number): Promise<{ ok: boolean; updated: number; error?: string }>
      getPersonClusters(personId: number): Promise<{ clusters: PersonCluster[]; error?: string }>
      getPersonLinkSuggestions(personId: number, limit?: number): Promise<{ suggestions: PersonLinkSuggestion[]; error?: string }>
      getPersonSimilarImages(personId: number, limit?: number, minSimilarity?: number): Promise<{ images: PersonSimilarImage[]; error?: string }>
      linkOccurrencesToPerson(personId: number, occurrenceIds: number[]): Promise<{ ok: boolean; updated: number; error?: string }>
      unlinkOccurrenceFromPerson(occurrenceId: number): Promise<{ ok: boolean; updated: number; error?: string }>
      getPersonDirectLinks(personId: number): Promise<{ links: PersonDirectLink[]; error?: string }>

      // Geo / Map
      geoClusters(params: {
        north: number; south: number; east: number; west: number; zoom: number; limit?: number
      }): Promise<{ clusters: GeoCluster[]; total: number; error?: string }>
      geoNearby(params: {
        lat: number; lng: number; radiusKm?: number; limit?: number; excludeId?: number
      }): Promise<{ images: GeoNearbyImage[]; total: number; error?: string }>
      geoStats(): Promise<{
        total_images: number; geotagged: number
        countries: Array<{ country: string; count: number }>
        top_cities: Array<{ city: string; state: string; country: string; count: number }>
        error?: string
      }>
      geoHeatmap(params: {
        north: number; south: number; east: number; west: number; zoom: number
      }): Promise<{ points: Array<{ lat: number; lng: number; weight: number }>; error?: string }>
      geoClusterPreview(params: {
        cell: string; limit?: number
      }): Promise<{
        images: Array<{ image_id: number; file_path: string; date: string | null; aesthetic_score: number | null }>
        total: number
        error?: string
      }>
      geoStatsExtended(params?: {
        home_lat?: number; home_lng?: number
      }): Promise<GeoStatsExtended>
      geoGapFillerPreview(params?: {
        max_gap_minutes?: number; preview_limit?: number
      }): Promise<GapFillerPreviewResponse>
      geoGapFillerApply(params?: {
        max_gap_minutes?: number; min_confidence?: number
      }): Promise<GapFillerApplyResponse>
      geoTripDetect(params?: {
        min_images?: number
      }): Promise<TripDetectResponse>
      geoTripTimeline(params: {
        start_date: string; end_date: string; simplify?: boolean
      }): Promise<TripTimelineResponse>
      geoGeocode(params: { location: string }): Promise<{
        lat: number | null; lng: number | null; count: number; error?: string
      }>

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
        recursive: boolean,
        noHash: boolean,
        profile: boolean,
        chunkSize: number,
        forceReprocess?: boolean
      ): Promise<void>
      batchPause(): Promise<void>
      batchPauseTarget(target: BatchControlTarget, mode?: BatchPauseMode): Promise<void>
      batchResume(): Promise<void>
      batchResumeTarget(target: BatchControlTarget): Promise<void>
      batchRemoveWorker(workerId: string): Promise<void>
      batchStop(folder: string): Promise<void>
      batchCheckPending(): Promise<{ pending: number; running: number }>
      batchMonitorExisting(): Promise<boolean>
      batchResumePending(workers?: number): Promise<void>
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
