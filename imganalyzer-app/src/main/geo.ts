import { ipcMain } from 'electron'
import { ensureServerRunning, rpc } from './python-rpc'

export interface GeoCluster {
  cell: string
  center_lat: number
  center_lng: number
  count: number
  sample_ids: number[]
}

export interface GeoClustersResponse {
  clusters: GeoCluster[]
  total: number
  error?: string
}

export interface GeoNearbyImage {
  image_id: number
  gps_latitude: number
  gps_longitude: number
  file_path: string
}

export interface GeoNearbyResponse {
  images: GeoNearbyImage[]
  total: number
  error?: string
}

export interface GeoStatsResponse {
  total_images: number
  geotagged: number
  countries: Array<{ country: string; count: number }>
  top_cities: Array<{ city: string; state: string; country: string; count: number }>
  error?: string
}

export interface GeoHeatmapPoint {
  lat: number
  lng: number
  weight: number
}

export interface GeoHeatmapResponse {
  points: GeoHeatmapPoint[]
  error?: string
}

export interface GeoClusterPreviewImage {
  image_id: number
  file_path: string
  date: string | null
  aesthetic_score: number | null
}

export interface GeoClusterPreviewResponse {
  images: GeoClusterPreviewImage[]
  total: number
  error?: string
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

export interface GeoGeocodeResponse {
  lat: number | null
  lng: number | null
  count: number
  error?: string
}

export function registerGeoHandlers(): void {
  ipcMain.handle(
    'geo:clusters',
    async (
      _evt,
      params: { north: number; south: number; east: number; west: number; zoom: number; limit?: number },
    ): Promise<GeoClustersResponse> => {
      try {
        await ensureServerRunning()
        const result = (await rpc.call('geo/clusters', params)) as GeoClustersResponse
        return result
      } catch (err) {
        return { clusters: [], total: 0, error: String(err) }
      }
    },
  )

  ipcMain.handle(
    'geo:nearby',
    async (
      _evt,
      params: { lat: number; lng: number; radiusKm?: number; limit?: number; excludeId?: number },
    ): Promise<GeoNearbyResponse> => {
      try {
        await ensureServerRunning()
        const result = (await rpc.call('geo/nearby', params)) as GeoNearbyResponse
        return result
      } catch (err) {
        return { images: [], total: 0, error: String(err) }
      }
    },
  )

  ipcMain.handle('geo:stats', async (): Promise<GeoStatsResponse> => {
    try {
      await ensureServerRunning()
      const result = (await rpc.call('geo/stats', {})) as GeoStatsResponse
      return result
    } catch (err) {
      return { total_images: 0, geotagged: 0, countries: [], top_cities: [], error: String(err) }
    }
  })

  ipcMain.handle(
    'geo:heatmap',
    async (
      _evt,
      params: { north: number; south: number; east: number; west: number; zoom: number },
    ): Promise<GeoHeatmapResponse> => {
      try {
        await ensureServerRunning()
        const result = (await rpc.call('geo/heatmap', params)) as GeoHeatmapResponse
        return result
      } catch (err) {
        return { points: [], error: String(err) }
      }
    },
  )

  ipcMain.handle(
    'geo:cluster-preview',
    async (
      _evt,
      params: { cell: string; limit?: number },
    ): Promise<GeoClusterPreviewResponse> => {
      try {
        await ensureServerRunning()
        const result = (await rpc.call('geo/cluster-preview', params)) as GeoClusterPreviewResponse
        return result
      } catch (err) {
        return { images: [], total: 0, error: String(err) }
      }
    },
  )

  ipcMain.handle(
    'geo:stats-extended',
    async (
      _evt,
      params?: { home_lat?: number; home_lng?: number },
    ): Promise<GeoStatsExtended> => {
      try {
        await ensureServerRunning()
        const result = (await rpc.call('geo/stats-extended', params ?? {})) as GeoStatsExtended
        return result
      } catch (err) {
        return {
          total_images: 0,
          geotagged: 0,
          gps_sources: [],
          countries: [],
          top_cities: [],
          monthly_activity: [],
          location_diversity: [],
          camera_by_country: [],
          top_locations: [],
          furthest_from_home: null,
          error: String(err),
        }
      }
    },
  )

  ipcMain.handle(
    'geo:gap-filler-preview',
    async (
      _evt,
      params?: { max_gap_minutes?: number; preview_limit?: number },
    ): Promise<GapFillerPreviewResponse> => {
      try {
        await ensureServerRunning()
        const result = (await rpc.call('geo/gap-filler-preview', params ?? {})) as GapFillerPreviewResponse
        return result
      } catch (err) {
        return { fillable: 0, total_missing: 0, previews: [], error: String(err) }
      }
    },
  )

  ipcMain.handle(
    'geo:gap-filler-apply',
    async (
      _evt,
      params?: { max_gap_minutes?: number; min_confidence?: number },
    ): Promise<GapFillerApplyResponse> => {
      try {
        await ensureServerRunning()
        const result = (await rpc.call('geo/gap-filler-apply', params ?? {})) as GapFillerApplyResponse
        return result
      } catch (err) {
        return { filled: 0, skipped_override: 0, skipped_low_confidence: 0, error: String(err) }
      }
    },
  )

  ipcMain.handle(
    'geo:trip-detect',
    async (
      _evt,
      params?: { min_images?: number },
    ): Promise<TripDetectResponse> => {
      try {
        await ensureServerRunning()
        const result = (await rpc.call('geo/trip-detect', params ?? {})) as TripDetectResponse
        return result
      } catch (err) {
        return { trips: [], error: String(err) }
      }
    },
  )

  ipcMain.handle(
    'geo:trip-timeline',
    async (
      _evt,
      params: { start_date: string; end_date: string; simplify?: boolean },
    ): Promise<TripTimelineResponse> => {
      try {
        await ensureServerRunning()
        const result = (await rpc.call('geo/trip-timeline', params)) as TripTimelineResponse
        return result
      } catch (err) {
        return { stops: [], route_points: [], total_images: 0, error: String(err) }
      }
    },
  )

  ipcMain.handle(
    'geo:geocode',
    async (_evt, params: { location: string }): Promise<GeoGeocodeResponse> => {
      try {
        await ensureServerRunning()
        const result = (await rpc.call('geo/geocode', params)) as GeoGeocodeResponse
        return result
      } catch (err) {
        return { lat: null, lng: null, count: 0, error: String(err) }
      }
    },
  )
}
