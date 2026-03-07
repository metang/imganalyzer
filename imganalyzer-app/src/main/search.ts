/**
 * search.ts — IPC handler for the search-json CLI command.
 *
 * Delegates to the persistent Python JSON-RPC server for search queries.
 * This eliminates the 1-3s conda subprocess overhead + CLIP model load
 * per search (the CLIP model stays loaded in the persistent process).
 */

import { ipcMain } from 'electron'
import { rpc, ensureServerRunning } from './python-rpc'
import { planSearchWithCopilot } from './search-planner'

// ── Types ─────────────────────────────────────────────────────────────────────

export type SearchIntent = 'people' | 'wildlife' | 'best-shot' | 'general'
export type SearchTimeOfDay = 'morning' | 'afternoon' | 'evening' | 'night'
export type SearchSortBy = 'relevance' | 'best' | 'aesthetic' | 'sharpness' | 'cleanest' | 'newest'
export type SearchFaceMatch = 'any' | 'all'

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
  // text filters
  face?: string
  faces?: string[]
  faceMatch?: SearchFaceMatch
  camera?: string
  lens?: string
  location?: string
  // numeric ranges
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
  // pagination
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
  // metadata
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
  // technical
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
  // local AI
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
  // aesthetic
  aesthetic_score: number | null
  aesthetic_label: string | null
  aesthetic_reason: string | null
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

// ── IPC Registration ──────────────────────────────────────────────────────────

export function registerSearchHandlers(): void {
  ipcMain.handle('search:run', async (_evt, filters: SearchFilters): Promise<SearchResponse> => {
    try {
      await ensureServerRunning()

      // Map the SearchFilters to the RPC params (camelCase keys)
      const result = await rpc.call('search', {
        query: filters.query?.trim() || '',
        mode: filters.mode,
        semanticWeight: filters.semanticWeight,
        intent: filters.intent,
        similarToImageId: filters.similarToImageId,
        country: filters.country,
        recurringMonthDay: filters.recurringMonthDay,
        timeOfDay: filters.timeOfDay,
        sortBy: filters.sortBy,
        expandedTerms: filters.expandedTerms,
        face: filters.face,
        faces: filters.faces,
        faceMatch: filters.faceMatch,
        camera: filters.camera,
        lens: filters.lens,
        location: filters.location,
        aestheticMin: filters.aestheticMin,
        aestheticMax: filters.aestheticMax,
        sharpnessMin: filters.sharpnessMin,
        sharpnessMax: filters.sharpnessMax,
        noiseMax: filters.noiseMax,
        isoMin: filters.isoMin,
        isoMax: filters.isoMax,
        facesMin: filters.facesMin,
        facesMax: filters.facesMax,
        dateFrom: filters.dateFrom,
        dateTo: filters.dateTo,
        hasPeople: filters.hasPeople,
        limit: filters.limit,
        offset: filters.offset,
         }) as { results: SearchResult[]; total: number | null; hasMore: boolean }

        return { results: result.results, total: result.total, hasMore: result.hasMore }
    } catch (err) {
      return { results: [], total: 0, hasMore: false, error: String(err) }
    }
  })

  ipcMain.handle('search:plan', async (_evt, request: SearchPlanRequest): Promise<SearchPlanResponse> => {
    return planSearchWithCopilot(request)
  })

  ipcMain.handle('search:resolve-face-query', async (_evt, query: string): Promise<SearchFaceResolution> => {
    try {
      await ensureServerRunning()
      const result = await rpc.call('search/resolveFaceQuery', { query }) as SearchFaceResolution
      return result
    } catch (err) {
      return {
        face: null,
        faces: [],
        faceMatch: 'all',
        remainingQuery: query,
        error: String(err),
      }
    }
  })
}
