import { clipboard, contextBridge, ipcRenderer } from 'electron'
import type { XmpData } from '../main/xmp'
import type { ImageFile } from '../main/images'
import type { AnalysisProgress } from '../main/analyzer'
import type {
  BatchControlTarget,
  BatchIngestProgress,
  BatchPauseMode,
  BatchResult,
  BatchStats,
} from '../main/batch'
import type { SearchFilters, SearchPlanRequest, SearchPlanResponse, SearchProgress, SearchResponse, SearchResult } from '../main/search'
import type { GalleryChunkParams, GalleryChunkResponse, GalleryFoldersResponse } from '../main/gallery'
import type {
  AppSettingsBundle,
  AppSettingsInput,
  CoordinatorStatus,
  ThumbnailCacheConfig,
  ThumbnailCacheConfigInput,
} from '../main/settings'

contextBridge.exposeInMainWorld('api', {
  openFolder: (): Promise<string | null> =>
    ipcRenderer.invoke('dialog:openFolder'),

  saveStoryExport: (defaultPath?: string): Promise<string | null> =>
    ipcRenderer.invoke('dialog:saveStoryExport', defaultPath),

  listImages: (folderPath: string): Promise<ImageFile[]> =>
    ipcRenderer.invoke('fs:listImages', folderPath),

  getThumbnail: (imagePath: string): Promise<string> =>
    ipcRenderer.invoke('fs:getThumbnail', imagePath),
  getThumbnailsBatch: (items: Array<{ file_path?: string; image_id?: number }>): Promise<Record<string, string>> =>
    ipcRenderer.invoke('fs:getThumbnailsBatch', items),

  getFullImage: (imagePath: string): Promise<string> =>
    ipcRenderer.invoke('fs:getFullImage', imagePath),

  getCachedImage: (imagePath: string): Promise<string> =>
    ipcRenderer.invoke('fs:getCachedImage', imagePath),

  openPath: (filePath: string): Promise<string> =>
    ipcRenderer.invoke('shell:openPath', filePath),

  copyText: (value: string): Promise<void> =>
    Promise.resolve(clipboard.writeText(value)),

  readXmp: (imagePath: string): Promise<XmpData | null> =>
    ipcRenderer.invoke('fs:readXmp', imagePath),

  runAnalysis: (imagePath: string, aiBackend: string): Promise<{ xmp: XmpData | null; error?: string; cancelled?: boolean }> =>
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
    noHash: boolean,
    forceReprocess = false
  ): Promise<{ registered: number; enqueued: number; skipped: number }> =>
    ipcRenderer.invoke('batch:ingest', folder, modules, recursive, noHash, forceReprocess),

  batchStart: (
    folder: string,
    modules: string[],
    workers: number,
    recursive: boolean,
    noHash: boolean,
    profile: boolean,
    chunkSize: number,
    forceReprocess = false
  ): Promise<void> =>
    ipcRenderer.invoke('batch:start', folder, modules, workers, recursive, noHash, profile, chunkSize, forceReprocess),

  batchPause: (): Promise<void> =>
    ipcRenderer.invoke('batch:pause'),

  batchPauseTarget: (target: BatchControlTarget, mode: BatchPauseMode = 'pause-drain'): Promise<void> =>
    ipcRenderer.invoke('batch:pause-target', target, mode),

  batchResume: (): Promise<void> =>
    ipcRenderer.invoke('batch:resume'),

  batchResumeTarget: (target: BatchControlTarget): Promise<void> =>
    ipcRenderer.invoke('batch:resume-target', target),

  batchRemoveWorker: (workerId: string): Promise<void> =>
    ipcRenderer.invoke('batch:remove-worker', workerId),

  batchStop: (folder: string): Promise<void> =>
    ipcRenderer.invoke('batch:stop', folder),

  batchCheckPending: (): Promise<{ pending: number; running: number }> =>
    ipcRenderer.invoke('batch:check-pending'),

  batchMonitorExisting: (): Promise<boolean> =>
    ipcRenderer.invoke('batch:monitor-existing'),

  batchResumePending: (workers?: number): Promise<void> =>
    ipcRenderer.invoke('batch:resume-pending', workers),

  batchRetryFailed: (modules: string[]): Promise<void> =>
    ipcRenderer.invoke('batch:retry-failed', modules),
  batchRebuildModule: (module: string): Promise<void> =>
    ipcRenderer.invoke('batch:rebuild-module', module),

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

  onSearchProgress: (cb: (progress: SearchProgress) => void) => {
    const handler = (_evt: Electron.IpcRendererEvent, progress: SearchProgress) => cb(progress)
    ipcRenderer.on('search:progress', handler)
    return () => ipcRenderer.removeListener('search:progress', handler)
  },

  getImageDetails: (params: { image_id?: number; file_path?: string }): Promise<{ result: SearchResult | null; error?: string }> =>
    ipcRenderer.invoke('image:details', params),

  resolveSearchFaceQuery: (query: string) =>
    ipcRenderer.invoke('search:resolve-face-query', query),

  planSearchQuery: (request: SearchPlanRequest): Promise<SearchPlanResponse> =>
    ipcRenderer.invoke('search:plan', request),

  galleryListFolders: (): Promise<GalleryFoldersResponse> =>
    ipcRenderer.invoke('gallery:list-folders'),

  galleryListImagesChunk: (params: GalleryChunkParams): Promise<GalleryChunkResponse> =>
    ipcRenderer.invoke('gallery:list-images-chunk', params),

  getThumbnailCacheConfig: (): Promise<ThumbnailCacheConfig> =>
    ipcRenderer.invoke('cache:thumbnail:getConfig'),

  setThumbnailCacheConfig: (config: ThumbnailCacheConfigInput): Promise<ThumbnailCacheConfig> =>
    ipcRenderer.invoke('cache:thumbnail:setConfig', config),

  getAppSettings: (): Promise<AppSettingsBundle> =>
    ipcRenderer.invoke('settings:get'),

  saveAppSettings: (input: AppSettingsInput): Promise<AppSettingsBundle> =>
    ipcRenderer.invoke('settings:save', input),

  getCoordinatorStatus: (): Promise<CoordinatorStatus> =>
    ipcRenderer.invoke('settings:getCoordinatorStatus'),

  startCoordinator: (): Promise<CoordinatorStatus> =>
    ipcRenderer.invoke('settings:startCoordinator'),

  stopCoordinator: (): Promise<CoordinatorStatus> =>
    ipcRenderer.invoke('settings:stopCoordinator'),

  // ── Albums / Storyline ──────────────────────────────────────────────────────

  albumsList: (): Promise<{ albums: Array<{ id: string; name: string; description: string | null; cover_image_id: number | null; story_enabled: boolean; sort_order: string; item_count: number; chapter_count: number; created_at: string; updated_at: string }> }> =>
    ipcRenderer.invoke('albums:list'),

  albumsCreate: (params: { name: string; rules: { match: string; rules: Array<Record<string, unknown>> }; description?: string; story_enabled?: boolean; sort_order?: string }): Promise<{ id: string; item_count: number }> =>
    ipcRenderer.invoke('albums:create', params),

  albumsGet: (albumId: string): Promise<Record<string, unknown>> =>
    ipcRenderer.invoke('albums:get', albumId),

  albumsUpdate: (params: { album_id: string; name?: string; description?: string; rules?: Record<string, unknown>; story_enabled?: boolean; sort_order?: string }): Promise<{ id: string; item_count: number } | { error: string }> =>
    ipcRenderer.invoke('albums:update', params),

  albumsDelete: (albumId: string): Promise<{ deleted: boolean }> =>
    ipcRenderer.invoke('albums:delete', albumId),

  albumsRefresh: (albumId: string): Promise<{ item_count: number }> =>
    ipcRenderer.invoke('albums:refresh', albumId),

  albumsStory: (albumId: string): Promise<{ chapters: Array<Record<string, unknown>> }> =>
    ipcRenderer.invoke('albums:story', albumId),

  albumsStoryGenerate: (params: { album_id: string; time_window_minutes?: number; chapter_gap_hours?: number; chapter_distance_km?: number; force_year_breaks?: boolean }): Promise<Record<string, unknown>> =>
    ipcRenderer.invoke('albums:story:generate', params),

  albumsChapterMoments: (chapterId: string): Promise<{ moments: Array<Record<string, unknown>> }> =>
    ipcRenderer.invoke('albums:chapter:moments', chapterId),

  albumsMomentImages: (momentId: string): Promise<{ images: Array<Record<string, unknown>> }> =>
    ipcRenderer.invoke('albums:moment:images', momentId),

  albumsCheckNew: (imageId: number): Promise<{ added_to_albums: string[] }> =>
    ipcRenderer.invoke('albums:check-new', imageId),

  albumsGenerateNarrative: (params: { album_id: string; use_ai?: boolean }): Promise<{ chapters_updated: number }> =>
    ipcRenderer.invoke('albums:story:generate-narrative', params),

  albumsExport: (params: { album_id: string; output_path: string; include_thumbnails?: boolean; max_heroes_per_chapter?: number }): Promise<{ path: string }> =>
    ipcRenderer.invoke('albums:export', params),

  albumsPresets: (): Promise<{ presets: Record<string, { name: string; description: string; params: string[] }> }> =>
    ipcRenderer.invoke('albums:presets'),

  albumsCreatePreset: (params: { preset: string; [key: string]: unknown }): Promise<{ id: string; name: string; item_count: number } | { error: string }> =>
    ipcRenderer.invoke('albums:create-preset', params),

  // ── Face management ────────────────────────────────────────────────────────

  listFaces: (): Promise<{ faces: Array<{ canonical_name: string; display_name: string | null; image_count: number; identity_id: number | null }>; error?: string }> =>
    ipcRenderer.invoke('faces:list'),

  getFaceImages: (name: string, limit?: number): Promise<{ images: Array<{ image_id: number; file_path: string; face_count: number }>; error?: string }> =>
    ipcRenderer.invoke('faces:images', name, limit),

  setFaceAlias: (canonicalName: string, displayName: string, clusterId?: number | null): Promise<{ ok: boolean; error?: string }> =>
    ipcRenderer.invoke('faces:setAlias', canonicalName, displayName, clusterId ?? null),

  listFaceClusters: (limit?: number, offset?: number): Promise<{ clusters: Array<{ cluster_id: number | null; identity_name: string; display_name: string | null; identity_id: number | null; image_count: number; face_count: number; representative_id: number | null; person_id: number | null }>; has_occurrences: boolean; total_count: number; deferred_cluster_ids: number[]; error?: string }> =>
    ipcRenderer.invoke('faces:clusters', limit, offset),

  getFaceClusterImages: (clusterId: number | null, identityName: string | null, limit?: number): Promise<{ occurrences: Array<{ id: number; image_id: number; file_path: string; face_idx: number; bbox_x1: number; bbox_y1: number; bbox_x2: number; bbox_y2: number; age: number | null; gender: string | null; identity_name: string }>; error?: string }> =>
    ipcRenderer.invoke('faces:clusterImages', clusterId, identityName, limit),

  relinkFaceCluster: (clusterId: number, displayName: string | null, personId?: number | null, updatePerson?: boolean): Promise<{ ok: boolean; updated: number; error?: string }> =>
    ipcRenderer.invoke('faces:clusterRelink', clusterId, displayName, personId ?? null, updatePerson ?? false),

  deferFaceCluster: (clusterId: number): Promise<{ ok: boolean; error?: string }> =>
    ipcRenderer.invoke('faces:clusterDefer', clusterId),

  undeferFaceCluster: (clusterId: number): Promise<{ ok: boolean; error?: string }> =>
    ipcRenderer.invoke('faces:clusterUndefer', clusterId),

  undeferAllFaceClusters: (): Promise<{ ok: boolean; cleared: number; error?: string }> =>
    ipcRenderer.invoke('faces:clusterUndeferAll'),

  splitCluster: (
    clusterId: number,
    threshold?: number,
  ): Promise<{ split_count: number; new_cluster_ids: number[]; error?: string }> =>
    ipcRenderer.invoke('faces:splitCluster', clusterId, threshold),

  getClusterPurity: (
    clusterId: number,
  ): Promise<{ purity_score: number; member_count: number; error?: string }> =>
    ipcRenderer.invoke('faces:clusterPurity', clusterId),

  getClusterLinkSuggestions: (
    clusterId: number,
    limit?: number,
  ): Promise<{
    suggestions: Array<{
      target_type: 'person' | 'alias'
      label: string
      person_id: number | null
      cluster_id: number | null
      score: number
      representative_id: number | null
      face_count: number
      reason: string
    }>
    error?: string
  }> =>
    ipcRenderer.invoke('faces:clusterLinkSuggestions', clusterId, limit),

  getFaceCrop: (occurrenceId: number): Promise<{ data?: string; error?: string }> =>
    ipcRenderer.invoke('faces:crop', occurrenceId),

  getFaceCropBatch: (ids: number[]): Promise<{ thumbnails: Record<string, string>; error?: string }> =>
    ipcRenderer.invoke('faces:cropBatch', ids),

  runFaceClustering: (threshold?: number): Promise<{ started: boolean; error?: string }> =>
    ipcRenderer.invoke('faces:runClustering', threshold),

  onClusteringDone: (cb: (result: { num_clusters?: number; error?: string }) => void): (() => void) => {
    const handler = (_event: unknown, result: { num_clusters?: number; error?: string }): void => { cb(result) }
    ipcRenderer.on('faces:clustering-done', handler)
    return () => ipcRenderer.removeListener('faces:clustering-done', handler)
  },

  onClusteringProgress: (cb: (progress: { phase: string; fraction: number; numClusters: number }) => void): (() => void) => {
    const handler = (_event: unknown, progress: { phase: string; fraction: number; numClusters: number }): void => { cb(progress) }
    ipcRenderer.on('faces:clustering-progress', handler)
    return () => ipcRenderer.removeListener('faces:clustering-progress', handler)
  },

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

  getPersonLinkSuggestions: (personId: number, limit?: number): Promise<{ suggestions: Array<{ cluster_id: number; label: string; score: number; representative_id: number | null; face_count: number; image_count: number; reason: string }>; error?: string }> =>
    ipcRenderer.invoke('faces:personLinkSuggestions', personId, limit),

  getPersonSimilarImages: (
    personId: number,
    limit?: number,
    minSimilarity?: number,
  ): Promise<{ images: Array<{ image_id: number; file_path: string; similarity: number; best_occurrence_id: number }>; error?: string }> =>
    ipcRenderer.invoke('faces:personSimilarImages', personId, limit, minSimilarity),

  linkOccurrencesToPerson: (personId: number, occurrenceIds: number[]): Promise<{ ok: boolean; updated: number; error?: string }> =>
    ipcRenderer.invoke('faces:personLinkOccurrences', personId, occurrenceIds),

  unlinkOccurrenceFromPerson: (occurrenceId: number): Promise<{ ok: boolean; updated: number; error?: string }> =>
    ipcRenderer.invoke('faces:personUnlinkOccurrence', occurrenceId),

  getPersonDirectLinks: (personId: number): Promise<{ links: Array<{ occurrence_id: number; image_id: number; file_path: string }>; error?: string }> =>
    ipcRenderer.invoke('faces:personDirectLinks', personId),

  // ── Geo / Map ──────────────────────────────────────────────────────────────

  geoClusters: (params: {
    north: number; south: number; east: number; west: number; zoom: number; detail?: number; limit?: number
  }) => ipcRenderer.invoke('geo:clusters', params),

  geoNearby: (params: {
    lat: number; lng: number; radiusKm?: number; limit?: number; excludeId?: number
  }) => ipcRenderer.invoke('geo:nearby', params),

  geoStats: () => ipcRenderer.invoke('geo:stats'),

  geoHeatmap: (params: {
    north: number; south: number; east: number; west: number; zoom: number
  }) => ipcRenderer.invoke('geo:heatmap', params),

  geoClusterPreview: (params: { cell: string; limit?: number }) =>
    ipcRenderer.invoke('geo:cluster-preview', params),

  geoStatsExtended: (params?: { home_lat?: number; home_lng?: number }) =>
    ipcRenderer.invoke('geo:stats-extended', params ?? {}),

  geoGapFillerPreview: (params?: { max_gap_minutes?: number; preview_limit?: number }) =>
    ipcRenderer.invoke('geo:gap-filler-preview', params ?? {}),

  geoGapFillerApply: (params?: { max_gap_minutes?: number; min_confidence?: number }) =>
    ipcRenderer.invoke('geo:gap-filler-apply', params ?? {}),

  geoTripDetect: (params?: { min_images?: number }) =>
    ipcRenderer.invoke('geo:trip-detect', params ?? {}),

  geoTripTimeline: (params: { start_date: string; end_date: string; simplify?: boolean }) =>
    ipcRenderer.invoke('geo:trip-timeline', params),

  geoGeocode: (params: { location: string }) =>
    ipcRenderer.invoke('geo:geocode', params),
})
