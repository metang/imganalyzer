import { useState, useEffect, useCallback, useRef } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMapEvents, useMap } from 'react-leaflet'
import L from 'leaflet'
import type { GeoCluster } from '../global'

// Fix Leaflet default marker icon paths in bundled Electron apps
delete (L.Icon.Default.prototype as Record<string, unknown>)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

interface GeoStats {
  total_images: number
  geotagged: number
  countries: Array<{ country: string; count: number }>
  top_cities: Array<{ city: string; state: string; country: string; count: number }>
}

function clusterIcon(count: number): L.DivIcon {
  const size = count < 100 ? 36 : count < 1000 ? 44 : count < 10000 ? 52 : 60
  const bg =
    count < 100
      ? 'bg-blue-500'
      : count < 1000
        ? 'bg-emerald-500'
        : count < 10000
          ? 'bg-amber-500'
          : 'bg-red-500'
  const label = count >= 1_000_000
    ? `${(count / 1_000_000).toFixed(1)}M`
    : count >= 1000
      ? `${(count / 1000).toFixed(1)}k`
      : String(count)

  return L.divIcon({
    className: '',
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    html: `<div class="flex items-center justify-center rounded-full ${bg} text-white font-semibold shadow-lg border-2 border-white/40"
                style="width:${size}px;height:${size}px;font-size:${size < 44 ? 11 : 13}px">${label}</div>`,
  })
}

function singleIcon(): L.DivIcon {
  return L.divIcon({
    className: '',
    iconSize: [14, 14],
    iconAnchor: [7, 7],
    html: '<div class="w-3.5 h-3.5 rounded-full bg-blue-400 border-2 border-white shadow"></div>',
  })
}

/** Listens to map move/zoom events and fires a debounced callback. */
function MapEventHandler({
  onBoundsChange,
}: {
  onBoundsChange: (bounds: { north: number; south: number; east: number; west: number; zoom: number }) => void
}) {
  const timerRef = useRef<ReturnType<typeof setTimeout>>()
  const map = useMapEvents({
    moveend: () => {
      clearTimeout(timerRef.current)
      timerRef.current = setTimeout(() => {
        const b = map.getBounds()
        onBoundsChange({
          north: b.getNorth(),
          south: b.getSouth(),
          east: b.getEast(),
          west: b.getWest(),
          zoom: map.getZoom(),
        })
      }, 300)
    },
    zoomend: () => {
      clearTimeout(timerRef.current)
      timerRef.current = setTimeout(() => {
        const b = map.getBounds()
        onBoundsChange({
          north: b.getNorth(),
          south: b.getSouth(),
          east: b.getEast(),
          west: b.getWest(),
          zoom: map.getZoom(),
        })
      }, 300)
    },
  })
  return null
}

/** Fits the map to show all clusters on initial load. */
function FitBounds({ clusters }: { clusters: GeoCluster[] }) {
  const map = useMap()
  const fitted = useRef(false)
  useEffect(() => {
    if (fitted.current || clusters.length === 0) return
    const lats = clusters.map((c) => c.center_lat)
    const lngs = clusters.map((c) => c.center_lng)
    const bounds = L.latLngBounds(
      [Math.min(...lats), Math.min(...lngs)],
      [Math.max(...lats), Math.max(...lngs)],
    )
    map.fitBounds(bounds, { padding: [40, 40], maxZoom: 14 })
    fitted.current = true
  }, [clusters, map])
  return null
}

export function MapView() {
  const [clusters, setClusters] = useState<GeoCluster[]>([])
  const [stats, setStats] = useState<GeoStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [totalInView, setTotalInView] = useState(0)

  // Load stats once on mount
  useEffect(() => {
    window.api.geoStats().then((s) => {
      if (!s.error) setStats(s)
    })
  }, [])

  // Load initial clusters (world view)
  useEffect(() => {
    setLoading(true)
    window.api
      .geoClusters({ north: 85, south: -85, east: 180, west: -180, zoom: 2 })
      .then((res) => {
        if (!res.error) {
          setClusters(res.clusters)
          setTotalInView(res.total)
        }
      })
      .finally(() => setLoading(false))
  }, [])

  const handleBoundsChange = useCallback(
    (bounds: { north: number; south: number; east: number; west: number; zoom: number }) => {
      setLoading(true)
      window.api
        .geoClusters(bounds)
        .then((res) => {
          if (!res.error) {
            setClusters(res.clusters)
            setTotalInView(res.total)
          }
        })
        .finally(() => setLoading(false))
    },
    [],
  )

  return (
    <div className="flex flex-col h-full">
      {/* Stats bar */}
      <div className="flex items-center gap-4 px-4 py-2 bg-neutral-900 border-b border-neutral-800 text-xs text-neutral-400 shrink-0">
        {stats && (
          <>
            <span>
              <span className="text-neutral-200 font-medium">{stats.geotagged.toLocaleString()}</span>
              {' '}geotagged of {stats.total_images.toLocaleString()} images
              {' '}({stats.total_images > 0 ? Math.round((stats.geotagged / stats.total_images) * 100) : 0}%)
            </span>
            {stats.countries.length > 0 && (
              <span>
                <span className="text-neutral-200 font-medium">{stats.countries.length}</span> countries
              </span>
            )}
          </>
        )}
        <span className="ml-auto">
          {loading ? (
            <span className="text-amber-400">Loading…</span>
          ) : (
            <span>
              <span className="text-neutral-200 font-medium">{totalInView.toLocaleString()}</span> photos in view
              {' '}· {clusters.length} clusters
            </span>
          )}
        </span>
      </div>

      {/* Map */}
      <div className="flex-1 min-h-0 relative">
        <MapContainer
          center={[20, 0]}
          zoom={2}
          className="h-full w-full"
          style={{ background: '#1a1a1a' }}
          zoomControl={true}
          attributionControl={true}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          <MapEventHandler onBoundsChange={handleBoundsChange} />
          <FitBounds clusters={clusters} />

          {clusters.map((cluster) => (
            <Marker
              key={cluster.cell}
              position={[cluster.center_lat, cluster.center_lng]}
              icon={cluster.count === 1 ? singleIcon() : clusterIcon(cluster.count)}
            >
              <Popup>
                <div className="text-neutral-900 text-sm min-w-[120px]">
                  <div className="font-semibold">{cluster.count.toLocaleString()} photos</div>
                  <div className="text-xs text-neutral-500 mt-0.5">
                    {cluster.center_lat.toFixed(4)}, {cluster.center_lng.toFixed(4)}
                  </div>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>
    </div>
  )
}
