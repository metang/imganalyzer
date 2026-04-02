import { useState, useEffect, useCallback, useRef } from 'react'
import { MapContainer, TileLayer, Marker, Tooltip, useMapEvents, useMap } from 'react-leaflet'
import L from 'leaflet'
import type { GeoCluster, SearchResult } from '../global'
import { SearchLightbox } from './SearchLightbox'
import { LocationStatsPanel } from './LocationStatsPanel'
import { TripTimeline } from './TripTimeline'

interface ClusterPreviewImage {
  image_id: number
  file_path: string
  date: string | null
  aesthetic_score: number | null
}

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

// ── Hover tooltip (lightweight, no thumbnails) ──────────────────────────────

function ClusterTooltipContent({ cluster }: { cluster: GeoCluster }) {
  return (
    <div className="text-neutral-900 text-sm min-w-[100px]">
      <div className="font-semibold">{cluster.count.toLocaleString()} photos</div>
      <div className="text-xs text-neutral-500 mt-0.5">Click to preview</div>
    </div>
  )
}

// ── Pinned preview panel (loads thumbnails on mount) ────────────────────────

function PinnedPreviewPanel({
  cluster,
  position,
  onClose,
  onImageClick,
}: {
  cluster: GeoCluster
  position: { x: number; y: number }
  onClose: () => void
  onImageClick: (imageId: number) => void
}) {
  const [images, setImages] = useState<ClusterPreviewImage[]>([])
  const [thumbs, setThumbs] = useState<Record<string, string>>({})
  const [total, setTotal] = useState(cluster.count)
  const [loading, setLoading] = useState(true)
  const panelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setThumbs({})

    window.api.geoClusterPreview({ cell: cluster.cell, limit: 10 }).then(async (res) => {
      if (cancelled) return
      if (res.error || !res.images.length) {
        setLoading(false)
        return
      }
      setImages(res.images)
      setTotal(res.total)

      const items = res.images.map((img) => ({
        file_path: img.file_path,
        image_id: img.image_id,
      }))
      const thumbMap = await window.api.getThumbnailsBatch(items)
      if (!cancelled) {
        setThumbs(thumbMap)
        setLoading(false)
      }
    })
    return () => { cancelled = true }
  }, [cluster.cell])

  // Dismiss on right-click anywhere
  useEffect(() => {
    const handleContextMenu = (e: MouseEvent) => {
      e.preventDefault()
      onClose()
    }
    document.addEventListener('contextmenu', handleContextMenu)
    return () => document.removeEventListener('contextmenu', handleContextMenu)
  }, [onClose])

  // Clamp position so panel stays within viewport
  const style: React.CSSProperties = {
    position: 'absolute',
    left: Math.min(position.x, window.innerWidth - 430),
    top: Math.max(position.y - 220, 10),
    zIndex: 10000,
  }

  return (
    <div ref={panelRef} style={style} className="cluster-pinned-panel">
      <div className="min-w-[200px] max-w-[400px] bg-white rounded-lg shadow-2xl p-3">
        <div className="flex items-center justify-between mb-1">
          <div className="font-semibold text-neutral-900 text-sm">
            {total.toLocaleString()} photos
          </div>
          <button
            onClick={onClose}
            className="text-neutral-400 hover:text-neutral-700 text-lg leading-none px-1"
          >
            ×
          </button>
        </div>
        <div className="text-xs text-neutral-500 mb-2">
          {cluster.center_lat.toFixed(4)}, {cluster.center_lng.toFixed(4)}
        </div>
        {loading ? (
          <div className="flex items-center justify-center h-[72px] text-xs text-neutral-400">
            Loading previews…
          </div>
        ) : images.length > 0 ? (
          <div className="flex flex-wrap gap-1">
            {images.map((img) => {
              const src = thumbs[img.file_path]
              return (
                <div
                  key={img.image_id}
                  className="w-[72px] h-[72px] rounded overflow-hidden bg-neutral-200 flex-shrink-0 cursor-pointer hover:ring-2 hover:ring-blue-400 transition-shadow"
                  onClick={() => onImageClick(img.image_id)}
                >
                  {src ? (
                    <img src={src} className="w-full h-full object-cover" />
                  ) : (
                    <div className="w-full h-full bg-neutral-300" />
                  )}
                </div>
              )
            })}
          </div>
        ) : (
          <div className="text-xs text-neutral-400">No previews available</div>
        )}
      </div>
    </div>
  )
}

export function MapView() {
  const [clusters, setClusters] = useState<GeoCluster[]>([])
  const [stats, setStats] = useState<GeoStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [totalInView, setTotalInView] = useState(0)
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')

  // Pinned preview state
  const [pinnedCluster, setPinnedCluster] = useState<GeoCluster | null>(null)
  const [pinnedPos, setPinnedPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 })

  // Lightbox state — holds full analysis data for the selected image
  const [lightboxItem, setLightboxItem] = useState<SearchResult | null>(null)
  const [lightboxItems, setLightboxItems] = useState<SearchResult[]>([])

  // Stats panel
  const [statsPanelOpen, setStatsPanelOpen] = useState(false)

  // Trip timeline mode
  const [timelineMode, setTimelineMode] = useState(false)
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

  const handleClusterClick = useCallback((cluster: GeoCluster, e: L.LeafletMouseEvent) => {
    setPinnedCluster(cluster)
    setPinnedPos({ x: e.originalEvent.clientX, y: e.originalEvent.clientY })
  }, [])

  const closePinned = useCallback(() => setPinnedCluster(null), [])

  const handleImageClick = useCallback((imageId: number) => {
    window.api.getImageDetails({ image_id: imageId }).then((res) => {
      if (res.result) {
        setLightboxItem(res.result)
        setLightboxItems([res.result])
      }
    })
  }, [])

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
        <span className="ml-auto flex items-center gap-3">
          {loading ? (
            <span className="text-amber-400">Loading…</span>
          ) : (
            <span>
              <span className="text-neutral-200 font-medium">{totalInView.toLocaleString()}</span> photos in view
              {' '}· {clusters.length} clusters
            </span>
          )}
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="px-2 py-0.5 rounded border border-neutral-700 hover:border-neutral-500 text-neutral-300 hover:text-neutral-100 transition-colors"
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} map`}
          >
            {theme === 'dark' ? '☀️ Light' : '🌙 Dark'}
          </button>
          <button
            onClick={() => setStatsPanelOpen(!statsPanelOpen)}
            className={`px-2 py-0.5 rounded border transition-colors ${
              statsPanelOpen
                ? 'border-blue-500 bg-blue-500/20 text-blue-300'
                : 'border-neutral-700 hover:border-neutral-500 text-neutral-300 hover:text-neutral-100'
            }`}
            title="Location Statistics"
          >
            📊 Stats
          </button>
          <button
            onClick={() => setTimelineMode(!timelineMode)}
            className={`px-2 py-0.5 rounded border transition-colors ${
              timelineMode
                ? 'border-emerald-500 bg-emerald-500/20 text-emerald-300'
                : 'border-neutral-700 hover:border-neutral-500 text-neutral-300 hover:text-neutral-100'
            }`}
            title="Trip Timeline"
          >
            🗺️ Trips
          </button>
        </span>
      </div>

      {/* Map */}
      <div className="flex-1 min-h-0 relative">
        <MapContainer
          center={[39.9, 116.4]}
          zoom={6}
          className="h-full w-full"
          style={{ background: theme === 'dark' ? '#1a1a1a' : '#e8e8e8' }}
          zoomControl={true}
          attributionControl={true}
        >
          <TileLayer
            key={theme}
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>'
            url={theme === 'dark'
              ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png'
              : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png'}
            maxZoom={20}
            tileSize={256}
            zoomOffset={0}
            detectRetina={false}
          />
          <MapEventHandler onBoundsChange={handleBoundsChange} />
          <InvalidateSize />
          <FitBounds clusters={clusters} />

          {/* Cluster markers (hidden in timeline mode) */}
          {!timelineMode && clusters.map((cluster) => (
            <Marker
              key={cluster.cell}
              position={[cluster.center_lat, cluster.center_lng]}
              icon={cluster.count === 1 ? _singleIcon : clusterIcon(cluster.count)}
              eventHandlers={{
                click: (e) => handleClusterClick(cluster, e),
              }}
            >
              <Tooltip
                direction="top"
                offset={[0, -10]}
                opacity={1}
                className="cluster-preview-tooltip"
              >
                <ClusterTooltipContent cluster={cluster} />
              </Tooltip>
            </Marker>
          ))}

          {/* Trip timeline overlay */}
          <TripTimeline active={timelineMode} onExit={() => setTimelineMode(false)} />
        </MapContainer>

        {/* Pinned preview panel (positioned over the map) */}
        {pinnedCluster && (
          <PinnedPreviewPanel
            cluster={pinnedCluster}
            position={pinnedPos}
            onClose={closePinned}
            onImageClick={handleImageClick}
          />
        )}

        {/* Location statistics drawer */}
        <LocationStatsPanel open={statsPanelOpen} onClose={() => setStatsPanelOpen(false)} />
      </div>

      {/* Lightbox with analysis sidebar */}
      {lightboxItem && (
        <SearchLightbox
          item={lightboxItem}
          items={lightboxItems}
          onClose={() => { setLightboxItem(null); setLightboxItems([]) }}
          onNavigate={(item) => setLightboxItem(item)}
        />
      )}
    </div>
  )
}
