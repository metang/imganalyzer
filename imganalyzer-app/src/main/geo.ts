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
}
