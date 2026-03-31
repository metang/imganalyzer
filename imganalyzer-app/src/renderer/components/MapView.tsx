import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMapEvents, useMap } from 'react-leaflet'
import L from 'leaflet'
import type { GeoCluster } from '../global'

interface GeoStats {
  total_images: number
  geotagged: number
  countries: Array<{ country: string; count: number }>
  top_cities: Array<{ city: string; state: string; country: string; count: number }>
}

// Icon cache to avoid recreating DOM elements on every render
const _iconCache = new Map<string, L.DivIcon>()

function clusterIcon(count: number): L.DivIcon {
  const key = String(count)
  let icon = _iconCache.get(key)
  if (icon) return icon

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

  icon = L.divIcon({
    className: '',
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    html: `<div class="flex items-center justify-center rounded-full ${bg} text-white font-semibold shadow-lg border-2 border-white/40"
                style="width:${size}px;height:${size}px;font-size:${size < 44 ? 11 : 13}px">${label}</div>`,
  })
  _iconCache.set(key, icon)
  return icon
}

const _singleIcon = L.divIcon({
  className: '',
  iconSize: [14, 14],
  iconAnchor: [7, 7],
  html: '<div class="w-3.5 h-3.5 rounded-full bg-blue-400 border-2 border-white shadow"></div>',
})

/** Listens to map move/zoom events and fires a debounced callback. */
function MapEventHandler({
  onBoundsChange,
}: {
  onBoundsChange: (bounds: { north: number; south: number; east: number; west: number; zoom: number }) => void
}) {
  const timerRef = useRef<ReturnType<typeof setTimeout>>()
  const cbRef = useRef(onBoundsChange)
  cbRef.current = onBoundsChange

  const map = useMapEvents({
    moveend: () => {
      clearTimeout(timerRef.current)
      timerRef.current = setTimeout(() => {
        const b = map.getBounds()
        cbRef.current({
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

/** Fixes Leaflet tile rendering when the map is inside a hidden tab.
 *  Calls invalidateSize() whenever the container becomes visible. */
function InvalidateSize() {
  const map = useMap()
  useEffect(() => {
    const container = map.getContainer()
    const observer = new ResizeObserver(() => {
      map.invalidateSize()
    })
    observer.observe(container)
    // Also invalidateSize after a short delay for initial tab switch
    const timer = setTimeout(() => map.invalidateSize(), 100)
    return () => {
      observer.disconnect()
      clearTimeout(timer)
    }
  }, [map])
  return null
}

/** Fits the map to show all clusters on initial load. */
function FitBounds({ clusters }: { clusters: GeoCluster[] }) {
  const map = useMap()
  const fitted = useRef(false)
  useEffect(() => {
    if (fitted.current || clusters.length === 0) return
    let minLat = Infinity, maxLat = -Infinity
    let minLng = Infinity, maxLng = -Infinity
    for (const c of clusters) {
      if (c.center_lat < minLat) minLat = c.center_lat
      if (c.center_lat > maxLat) maxLat = c.center_lat
      if (c.center_lng < minLng) minLng = c.center_lng
      if (c.center_lng > maxLng) maxLng = c.center_lng
    }
    const bounds = L.latLngBounds([minLat, minLng], [maxLat, maxLng])
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
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png"
            maxZoom={20}
            tileSize={256}
            zoomOffset={0}
            detectRetina={false}
          />
          <MapEventHandler onBoundsChange={handleBoundsChange} />
          <InvalidateSize />
          <FitBounds clusters={clusters} />

          {clusters.map((cluster) => (
            <Marker
              key={cluster.cell}
              position={[cluster.center_lat, cluster.center_lng]}
              icon={cluster.count === 1 ? _singleIcon : clusterIcon(cluster.count)}
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
