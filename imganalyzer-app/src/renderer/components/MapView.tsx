import { useState, useEffect, useCallback, useRef } from 'react'
import { MapContainer, TileLayer, Marker, Tooltip, useMapEvents, useMap, CircleMarker } from 'react-leaflet'
import L from 'leaflet'
import type { GeoCluster, SearchResult, SearchFilters } from '../global'
import { SearchLightbox } from './SearchLightbox'
import { SearchBar } from './SearchBar'
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
                style="width:${size}px;height:${size}px;font-size:${size < 44 ? 11 : 13}px">${String(Number(count) || 0)}</div>`,
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

/** Captures the Leaflet Map instance into a ref passed from the parent. */
function MapRefSetter({ mapRef }: { mapRef: React.MutableRefObject<L.Map | null> }) {
  const map = useMap()
  useEffect(() => { mapRef.current = map }, [map, mapRef])
  return null
}

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

export function MapView({ pendingFilters, onClearPending, onViewAsGrid }: {
  pendingFilters?: SearchFilters | null
  onClearPending?: () => void
  onViewAsGrid?: (filters: SearchFilters, mapBounds?: { north: number; south: number; east: number; west: number }) => void
}) {
  const [clusters, setClusters] = useState<GeoCluster[]>([])
  const [stats, setStats] = useState<GeoStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [totalInView, setTotalInView] = useState(0)
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')

  // Pinned preview state
  const [pinnedCluster, setPinnedCluster] = useState<GeoCluster | null>(null)
  const [pinnedPos, setPinnedPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 })

  // Lightbox state
  const [lightboxItem, setLightboxItem] = useState<SearchResult | null>(null)
  const [lightboxItems, setLightboxItems] = useState<SearchResult[]>([])

  // Stats panel
  const [statsPanelOpen, setStatsPanelOpen] = useState(false)

  // Trip timeline mode
  const [timelineMode, setTimelineMode] = useState(false)

  // Search integration
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchFilters, setSearchFilters] = useState<SearchFilters | null>(null)
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [searchTotal, setSearchTotal] = useState<number | null>(null)
  const [searchLoading, setSearchLoading] = useState(false)
  const boundsRef = useRef<{ north: number; south: number; east: number; west: number; zoom: number } | null>(null)
  const mapRef = useRef<L.Map | null>(null)
  const searchRequestRef = useRef(0)
  const [searchContextLabel, setSearchContextLabel] = useState<string | null>(null)

  // Thumbnail cache for search result pins
  const [thumbCache, setThumbCache] = useState<Record<string, string>>({})

  // Handle pending filters from another tab
  useEffect(() => {
    if (pendingFilters) {
      setSearchFilters(pendingFilters)
      setSearchOpen(true)
      onClearPending?.()
    }
  }, [pendingFilters, onClearPending])

  // Load stats once on mount
  useEffect(() => {
    window.api.geoStats().then((s) => {
      if (!s.error) setStats(s)
    })
  }, [])

  // Load initial clusters (world view) — only when no search active
  useEffect(() => {
    if (searchFilters) return
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
  }, [searchFilters])

  // When search filters change or viewport moves, run search with mapBounds
  const runMapSearch = useCallback(async (
    filters: SearchFilters,
    bounds: { north: number; south: number; east: number; west: number } | null,
  ) => {
    const requestId = ++searchRequestRef.current
    setSearchLoading(true)

    try {
      const filtersWithBounds: SearchFilters = {
        ...filters,
        mapBounds: bounds ?? undefined,
        limit: 500,
        offset: 0,
      }
      const resp = await window.api.searchImages(filtersWithBounds)
      if (searchRequestRef.current !== requestId) return

      if (resp.error) {
        setSearchResults([])
        setSearchTotal(null)
      } else {
        setSearchResults(resp.results)
        setSearchTotal(resp.total)

        // Load thumbnails for results with GPS
        const geoResults = resp.results.filter(
          (r) => r.gps_latitude != null && r.gps_longitude != null,
        )
        if (geoResults.length > 0) {
          const items = geoResults.slice(0, 200).map((r) => ({
            file_path: r.file_path,
            image_id: r.image_id,
          }))
          const chunkSize = 40
          for (let i = 0; i < items.length; i += chunkSize) {
            const chunk = items.slice(i, i + chunkSize)
            void window.api.getThumbnailsBatch(chunk).then((thumbMap) => {
              if (searchRequestRef.current === requestId) {
                setThumbCache((prev) => ({ ...prev, ...thumbMap }))
              }
            }).catch(() => {
              // Ignore thumbnail chunk failures; markers still render without previews.
            })
          }
        }
      }
    } catch {
      if (searchRequestRef.current === requestId) {
        setSearchResults([])
        setSearchTotal(null)
      }
    } finally {
      if (searchRequestRef.current === requestId) {
        setSearchLoading(false)
      }
    }
  }, [])

  // Re-run search when viewport changes (only if search is active)
  const handleBoundsChange = useCallback(
    (bounds: { north: number; south: number; east: number; west: number; zoom: number }) => {
      boundsRef.current = bounds
      if (searchFilters) {
        runMapSearch(searchFilters, bounds)
      } else {
        // Browse-all mode: load clusters
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
      }
    },
    [searchFilters, runMapSearch],
  )

  // SearchBar callback
  const handleSearch = useCallback(async (filters: SearchFilters, contextLabel: string | null) => {
    setSearchFilters(filters)
    setSearchContextLabel(contextLabel)
    setPinnedCluster(null)

    // Auto-navigate for location filter
    if (filters.location) {
      try {
        const geo = await window.api.geoGeocode({ location: filters.location })
        if (geo.lat != null && geo.lng != null && geo.count > 0 && mapRef.current) {
          mapRef.current.flyTo([geo.lat, geo.lng], 10, { duration: 1.5 })
          // Search will re-run via handleBoundsChange when flyTo completes
          return
        }
      } catch { /* geocode failed — fall through to run search with current bounds */ }
    }

    // Run search with current bounds
    const bounds = boundsRef.current
    await runMapSearch(filters, bounds)
  }, [runMapSearch])

  const clearSearch = useCallback(() => {
    setSearchFilters(null)
    setSearchResults([])
    setSearchTotal(null)
    setSearchContextLabel(null)
    setThumbCache({})
    // Reload clusters for current bounds
    if (boundsRef.current) {
      handleBoundsChange(boundsRef.current)
    }
  }, [handleBoundsChange])

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

  // Click a search result pin → open lightbox
  const handlePinClick = useCallback((item: SearchResult) => {
    setLightboxItem(item)
    setLightboxItems(searchResults.filter(
      (r) => r.gps_latitude != null && r.gps_longitude != null,
    ))
  }, [searchResults])

  // Geotagged search results for pin rendering
  const geoResults = searchResults.filter(
    (r) => r.gps_latitude != null && r.gps_longitude != null,
  )

  const isSearchActive = searchFilters != null

  return (
    <div className="flex flex-col h-full">
      {/* Stats bar */}
      <div className="flex items-center gap-4 px-4 py-2 bg-neutral-900 border-b border-neutral-800 text-xs text-neutral-400 shrink-0">
        {stats && !isSearchActive && (
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
        {isSearchActive && (
          <>
            <span className="text-blue-300">
              {searchContextLabel ?? 'Map search'}
            </span>
            <span>
              <span className="text-neutral-200 font-medium">
                {geoResults.length}
              </span>
              {' '}pins on map
              {searchTotal != null && (
                <span className="text-neutral-500"> · {searchTotal.toLocaleString()} total matches</span>
              )}
            </span>
            <button
              onClick={clearSearch}
              className="px-2 py-0.5 rounded border border-red-700/60 hover:border-red-500 text-red-400 hover:text-red-300 transition-colors"
              title="Clear search"
            >
              ✕ Clear
            </button>
            {onViewAsGrid && (
              <button
                onClick={() => onViewAsGrid(searchFilters!, boundsRef.current ?? undefined)}
                className="px-2 py-0.5 rounded border border-neutral-600 hover:border-neutral-400 text-neutral-300 hover:text-neutral-100 transition-colors"
                title="View results as grid in Search tab"
              >
                📋 Grid View
              </button>
            )}
          </>
        )}
        <span className="ml-auto flex items-center gap-3">
          {(loading || searchLoading) ? (
            <span className="text-amber-400">Loading…</span>
          ) : !isSearchActive ? (
            <span>
              <span className="text-neutral-200 font-medium">{totalInView.toLocaleString()}</span> photos in view
              {' '}· {clusters.length} clusters
            </span>
          ) : null}
          <button
            onClick={() => setSearchOpen(!searchOpen)}
            className={`px-2 py-0.5 rounded border transition-colors ${
              searchOpen
                ? 'border-purple-500 bg-purple-500/20 text-purple-300'
                : 'border-neutral-700 hover:border-neutral-500 text-neutral-300 hover:text-neutral-100'
            }`}
            title="Search on Map"
          >
            🔍 Search
          </button>
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

      {/* Main area: optional SearchBar sidebar + map */}
      <div className="flex-1 min-h-0 flex">
        {/* SearchBar sidebar */}
        {searchOpen && (
          <div className="w-[420px] min-w-[360px] shrink-0 border-r border-neutral-800 bg-neutral-950/80 overflow-y-auto">
            <SearchBar
              onSearch={handleSearch}
              loading={searchLoading}
            />
          </div>
        )}

        {/* Map */}
        <div className="flex-1 min-h-0 relative">
          <MapContainer
            center={[39.9042, 116.4074]}
            zoom={7}
            className="h-full w-full"
            style={{ background: theme === 'dark' ? '#1a1a1a' : '#e8e8e8' }}
            zoomControl={true}
            attributionControl={true}
          >
            <MapRefSetter mapRef={mapRef} />
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
            {!isSearchActive && <FitBounds clusters={clusters} />}

            {/* Browse-all mode: cluster markers */}
            {!isSearchActive && !timelineMode && clusters.map((cluster) => (
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

            {/* Search mode: result pins */}
            {isSearchActive && geoResults.map((item) => {
              const lat = parseFloat(String(item.gps_latitude))
              const lng = parseFloat(String(item.gps_longitude))
              if (isNaN(lat) || isNaN(lng)) return null
              const thumb = thumbCache[item.file_path]
              return (
                <CircleMarker
                  key={item.image_id}
                  center={[lat, lng]}
                  radius={thumb ? 8 : 6}
                  pathOptions={{
                    color: '#3b82f6',
                    fillColor: '#60a5fa',
                    fillOpacity: 0.85,
                    weight: 2,
                  }}
                  eventHandlers={{
                    click: () => handlePinClick(item),
                  }}
                >
                  <Tooltip direction="top" offset={[0, -8]} opacity={1}>
                    <div className="text-neutral-900 text-xs min-w-[120px] max-w-[240px]">
                      {thumb && (
                        <img
                          src={thumb}
                          className="w-full h-24 object-cover rounded mb-1"
                        />
                      )}
                      <div className="font-medium truncate">
                        {item.file_path.split(/[/\\]/).pop()}
                      </div>
                      {item.location_city && (
                        <div className="text-neutral-500">
                          {item.location_city}
                          {item.location_state ? `, ${item.location_state}` : ''}
                        </div>
                      )}
                    </div>
                  </Tooltip>
                </CircleMarker>
              )
            })}

            {/* Trip timeline overlay */}
            <TripTimeline active={timelineMode} onExit={() => setTimelineMode(false)} />
          </MapContainer>

          {/* Pinned preview panel (browse-all mode only) */}
          {!isSearchActive && pinnedCluster && (
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
      </div>

      {/* Lightbox with analysis sidebar — z-[10000] to render above Leaflet map layers */}
      {lightboxItem && (
        <div className="fixed inset-0 z-[10000]">
          <SearchLightbox
            item={lightboxItem}
            items={lightboxItems}
            onClose={() => { setLightboxItem(null); setLightboxItems([]) }}
            onNavigate={(item) => setLightboxItem(item)}
          />
        </div>
      )}
    </div>
  )
}
