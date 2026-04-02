import { useState, useEffect, useCallback, useRef } from 'react'
import { Polyline, CircleMarker, Tooltip, useMap } from 'react-leaflet'
import type { TripDetectResult, TripStop, SearchResult } from '../global'
import { SearchLightbox } from './SearchLightbox'

interface Props {
  active: boolean
  onExit: () => void
}

type Phase = 'loading' | 'select' | 'viewing'

// Color gradient from blue (start) to red (end)
function routeColor(idx: number, total: number): string {
  const t = total > 1 ? idx / (total - 1) : 0
  const r = Math.round(59 + t * 196)   // 59 → 255
  const g = Math.round(130 - t * 70)   // 130 → 60
  const b = Math.round(246 - t * 180)  // 246 → 66
  return `rgb(${r},${g},${b})`
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
  } catch {
    return iso.slice(0, 10)
  }
}

function formatDateShort(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  } catch {
    return iso.slice(0, 10)
  }
}

function formatDuration(start: string, end: string): string {
  try {
    const ms = new Date(end).getTime() - new Date(start).getTime()
    const hours = Math.floor(ms / 3600000)
    const days = Math.floor(hours / 24)
    if (days > 0) return `${days}d ${hours % 24}h`
    return `${hours}h`
  } catch {
    return ''
  }
}

export function TripTimeline({ active, onExit }: Props) {
  const [phase, setPhase] = useState<Phase>('loading')
  const [trips, setTrips] = useState<TripDetectResult[]>([])
  const [selectedTrip, setSelectedTrip] = useState<TripDetectResult | null>(null)
  const [stops, setStops] = useState<TripStop[]>([])
  const [routePoints, setRoutePoints] = useState<Array<{ lat: number; lng: number }>>([])
  const [totalImages, setTotalImages] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  // Animation state
  const [playing, setPlaying] = useState(false)
  const [currentStopIdx, setCurrentStopIdx] = useState(-1)
  const animRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Lightbox state
  const [lightboxItem, setLightboxItem] = useState<SearchResult | null>(null)
  const [lightboxItems, setLightboxItems] = useState<SearchResult[]>([])

  // Detect trips on mount
  useEffect(() => {
    if (!active) return
    setPhase('loading')
    setError(null)
    window.api.geoTripDetect({ min_images: 5 })
      .then((res) => {
        if (res.error) setError(res.error)
        else setTrips(res.trips)
        setPhase('select')
      })
      .catch((err) => {
        setError(String(err))
        setPhase('select')
      })
  }, [active])

  const loadTrip = useCallback(async (trip: TripDetectResult) => {
    setSelectedTrip(trip)
    setLoading(true)
    setError(null)
    setCurrentStopIdx(-1)
    setPlaying(false)
    try {
      const res = await window.api.geoTripTimeline({
        start_date: trip.start_date,
        end_date: trip.end_date,
      })
      if (res.error) {
        setError(res.error)
        setLoading(false)
        return
      }
      setStops(res.stops)
      setRoutePoints(res.route_points)
      setTotalImages(res.total_images)
      setPhase('viewing')
    } catch (err) {
      setError(String(err))
    }
    setLoading(false)
  }, [])

  // Animation logic
  useEffect(() => {
    if (!playing || stops.length === 0) return
    const startIdx = currentStopIdx < 0 ? 0 : currentStopIdx
    let idx = startIdx

    const advance = () => {
      if (idx >= stops.length) {
        setPlaying(false)
        return
      }
      setCurrentStopIdx(idx)
      idx++
      animRef.current = setTimeout(advance, 1500)
    }
    advance()

    return () => {
      if (animRef.current) clearTimeout(animRef.current)
    }
  }, [playing, stops.length])

  const handleStopClick = useCallback(async (stop: TripStop) => {
    setCurrentStopIdx(stops.indexOf(stop))
    try {
      const res = await window.api.getImageDetails({ image_id: stop.cover_image_id })
      if (res.result) {
        setLightboxItem(res.result)
        setLightboxItems([res.result])
      }
    } catch { /* ignore */ }
  }, [stops])

  if (!active) return null

  return (
    <>
      {/* Route polyline — drawn as segments with gradient color */}
      <TripRoutePolyline points={routePoints} />

      {/* Stop markers */}
      {stops.map((stop, i) => {
        const isHighlighted = i === currentStopIdx
        const radius = Math.min(Math.max(stop.count / 5, 5), 14)
        return (
          <CircleMarker
            key={i}
            center={[stop.lat, stop.lng]}
            radius={isHighlighted ? radius + 3 : radius}
            pathOptions={{
              color: isHighlighted ? '#fff' : routeColor(i, stops.length),
              fillColor: routeColor(i, stops.length),
              fillOpacity: isHighlighted ? 1 : 0.8,
              weight: isHighlighted ? 3 : 2,
            }}
            eventHandlers={{ click: () => handleStopClick(stop) }}
          >
            <Tooltip direction="top" offset={[0, -8]} opacity={1}>
              <div className="text-xs">
                <p className="font-medium">{stop.count} photos</p>
                <p className="text-neutral-400">{formatDateShort(stop.start_time)}</p>
              </div>
            </Tooltip>
          </CircleMarker>
        )
      })}

      {/* Fit map to trip bounds */}
      {phase === 'viewing' && stops.length > 0 && <FitTripBounds stops={stops} />}

      {/* Control panel overlay */}
      <div className="absolute top-2 left-2 z-[1000] max-w-xs">
        {error && (
          <div className="text-red-400 text-xs bg-red-900/80 backdrop-blur p-2 rounded mb-2">{error}</div>
        )}

        {phase === 'loading' && (
          <div className="bg-neutral-900/90 backdrop-blur rounded-lg p-3 text-sm text-neutral-300">
            Detecting trips…
          </div>
        )}

        {phase === 'select' && (
          <div className="bg-neutral-900/90 backdrop-blur rounded-lg p-3 space-y-2 max-h-72 overflow-y-auto">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-neutral-200 uppercase tracking-wider">
                Trips ({trips.length})
              </h3>
              <button onClick={onExit} className="text-neutral-400 hover:text-neutral-200 text-xs">✕ Close</button>
            </div>
            {trips.length === 0 && (
              <p className="text-xs text-neutral-500 italic">No trips detected (need GPS photos with significant movement)</p>
            )}
            {trips.map((trip, i) => (
              <button
                key={i}
                onClick={() => loadTrip(trip)}
                disabled={loading}
                className="w-full text-left p-2 rounded hover:bg-neutral-800/80 transition-colors border border-neutral-800/50"
              >
                <div className="text-xs font-medium text-neutral-200">
                  {trip.start_location} → {trip.end_location}
                </div>
                <div className="text-[10px] text-neutral-400 mt-0.5">
                  {formatDate(trip.start_date)} – {formatDate(trip.end_date)}
                  {' · '}{trip.image_count} photos
                  {' · '}{trip.distance_km.toLocaleString()} km
                </div>
              </button>
            ))}
          </div>
        )}

        {phase === 'viewing' && selectedTrip && (
          <div className="bg-neutral-900/90 backdrop-blur rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-neutral-200">
                {selectedTrip.start_location} → {selectedTrip.end_location}
              </h3>
              <button onClick={onExit} className="text-neutral-400 hover:text-neutral-200 text-xs">✕</button>
            </div>
            <div className="text-[10px] text-neutral-400">
              {formatDate(selectedTrip.start_date)} – {formatDate(selectedTrip.end_date)}
              {' · '}{totalImages} photos · {stops.length} stops
              {' · '}{selectedTrip.distance_km.toLocaleString()} km
            </div>

            {/* Playback controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  if (playing) {
                    setPlaying(false)
                    if (animRef.current) clearTimeout(animRef.current)
                  } else {
                    if (currentStopIdx >= stops.length - 1) setCurrentStopIdx(-1)
                    setPlaying(true)
                  }
                }}
                className="px-2 py-1 text-xs bg-blue-600/80 hover:bg-blue-500/80 text-white rounded transition-colors"
              >
                {playing ? '⏸ Pause' : '▶ Play'}
              </button>
              <button
                onClick={() => { setPhase('select'); setStops([]); setRoutePoints([]); setCurrentStopIdx(-1); setPlaying(false) }}
                className="px-2 py-1 text-xs text-neutral-400 hover:text-neutral-200 transition-colors"
              >
                ← Trips
              </button>
            </div>

            {/* Timeline scrubber */}
            {stops.length > 1 && (
              <div className="space-y-1">
                <input
                  type="range"
                  min={0}
                  max={stops.length - 1}
                  value={currentStopIdx >= 0 ? currentStopIdx : 0}
                  onChange={(e) => {
                    setPlaying(false)
                    if (animRef.current) clearTimeout(animRef.current)
                    setCurrentStopIdx(Number(e.target.value))
                  }}
                  className="w-full h-1"
                />
                {currentStopIdx >= 0 && currentStopIdx < stops.length && (
                  <p className="text-[10px] text-neutral-400">
                    Stop {currentStopIdx + 1}/{stops.length} · {stops[currentStopIdx].count} photos
                    {' · '}{formatDateShort(stops[currentStopIdx].start_time)}
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Lightbox */}
      {lightboxItem && (
        <SearchLightbox
          item={lightboxItem}
          items={lightboxItems}
          onClose={() => { setLightboxItem(null); setLightboxItems([]) }}
          onNavigate={(item) => setLightboxItem(item)}
        />
      )}
    </>
  )
}

function TripRoutePolyline({ points }: { points: Array<{ lat: number; lng: number }> }) {
  if (points.length < 2) return null

  // Split into segments for gradient coloring
  const segmentSize = Math.max(1, Math.floor(points.length / 20))
  const segments: Array<[number, number][]> = []
  for (let i = 0; i < points.length - 1; i += segmentSize) {
    const end = Math.min(i + segmentSize + 1, points.length)
    segments.push(
      points.slice(i, end).map((p) => [p.lat, p.lng] as [number, number])
    )
  }

  return (
    <>
      {segments.map((seg, i) => (
        <Polyline
          key={i}
          positions={seg}
          pathOptions={{
            color: routeColor(i, segments.length),
            weight: 3,
            opacity: 0.8,
          }}
        />
      ))}
    </>
  )
}

function FitTripBounds({ stops }: { stops: TripStop[] }) {
  const map = useMap()
  const fittedRef = useRef(false)

  useEffect(() => {
    if (fittedRef.current || stops.length === 0) return
    fittedRef.current = true
    const L = (window as unknown as { L: typeof import('leaflet') }).L ?? map.options
    const lats = stops.map((s) => s.lat)
    const lngs = stops.map((s) => s.lng)
    map.fitBounds(
      [
        [Math.min(...lats) - 0.5, Math.min(...lngs) - 0.5],
        [Math.max(...lats) + 0.5, Math.max(...lngs) + 0.5],
      ],
      { padding: [40, 40], maxZoom: 14 }
    )
  }, [stops, map])

  return null
}
