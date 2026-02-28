/**
 * search.ts — IPC handler for the search-json CLI command.
 *
 * Delegates to the persistent Python JSON-RPC server for search queries.
 * This eliminates the 1-3s conda subprocess overhead + CLIP model load
 * per search (the CLIP model stays loaded in the persistent process).
 */

import { ipcMain } from 'electron'
import { rpc, ensureServerRunning } from './python-rpc'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SearchFilters {
  query?: string
  mode?: 'text' | 'semantic' | 'hybrid' | 'browse'
  semanticWeight?: number
  // text filters
  face?: string
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
  // aesthetic
  aesthetic_score: number | null
  aesthetic_label: string | null
  aesthetic_reason: string | null
}

export interface SearchResponse {
  results: SearchResult[]
  total: number
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
        face: filters.face,
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
      }) as { results: SearchResult[]; total: number }

      return { results: result.results, total: result.total }
    } catch (err) {
      return { results: [], total: 0, error: String(err) }
    }
  })
}
