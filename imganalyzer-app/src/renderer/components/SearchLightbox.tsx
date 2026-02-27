/**
 * SearchLightbox.tsx — Lightbox for images opened from search results.
 *
 * Unlike the Gallery lightbox (which reads XMP sidecars), search results
 * already contain all analysis data from the database.  This lightbox
 * renders the inline data directly in the right sidebar.
 *
 * Supports zoom/pan (same as Lightbox.tsx) and prev/next navigation
 * through the search result set.
 */
import { useEffect, useState, useCallback, useRef } from 'react'
import type { SearchResult } from '../global'

interface SearchLightboxProps {
  item: SearchResult
  items: SearchResult[]
  onClose: () => void
  onNavigate: (item: SearchResult) => void
}

const MIN_ZOOM = 0.1
const MAX_ZOOM = 10
const ZOOM_STEP_KEY = 0.25
const ZOOM_STEP_WHEEL = 0.12

function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)) }

// ── Analysis sidebar ──────────────────────────────────────────────────────────

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex gap-2 py-1 border-b border-neutral-800 last:border-0">
      <span className="text-neutral-500 text-xs w-32 shrink-0">{label}</span>
      <span className="text-neutral-200 text-xs break-words min-w-0">{value}</span>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-4">
      <h3 className="text-[11px] font-semibold uppercase tracking-widest text-neutral-500 mb-1 px-4">{title}</h3>
      <div className="px-4">{children}</div>
    </div>
  )
}

function ScoreBar({ value, max = 10 }: { value: number; max?: number }) {
  const pct = Math.min(100, (value / max) * 100)
  const color = pct >= 70 ? 'bg-green-500' : pct >= 40 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-neutral-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-neutral-300 tabular-nums w-8 text-right">{value.toFixed(1)}</span>
    </div>
  )
}

function TagList({ items }: { items: string[] }) {
  if (!items.length) return <span className="text-neutral-600 text-xs">—</span>
  return (
    <div className="flex flex-wrap gap-1">
      {items.map((item, i) => (
        <span key={i} className="px-1.5 py-0.5 bg-neutral-800 rounded text-[11px] text-neutral-300">
          {item}
        </span>
      ))}
    </div>
  )
}

function AnalysisSidebar({ item }: { item: SearchResult }) {
  const filename = item.file_path.split(/[/\\]/).pop() ?? ''

  return (
    <div className="w-80 shrink-0 flex flex-col bg-neutral-900 border-l border-neutral-800 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-neutral-800">
        <span className="text-sm font-medium truncate text-neutral-200 block" title={filename}>{filename}</span>
        {item.score !== null && (
          <span className="text-xs text-neutral-500">Match: {(item.score * 100).toFixed(0)}%</span>
        )}
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto pt-3">

        {/* Aesthetic */}
        {item.aesthetic_score !== null && (
          <Section title="Aesthetic">
            <div className="py-1 border-b border-neutral-800">
              <div className="flex justify-between mb-1">
                <span className="text-neutral-500 text-xs">Score</span>
                {item.aesthetic_label && <span className="text-xs text-neutral-400">{item.aesthetic_label}</span>}
              </div>
              <ScoreBar value={item.aesthetic_score} max={10} />
            </div>
            {item.aesthetic_reason && (
              <Row label="Reason" value={item.aesthetic_reason} />
            )}
          </Section>
        )}

        {/* AI Analysis */}
        {(item.description || item.scene_type || item.main_subject || item.lighting || item.mood ||
          item.detected_objects?.length || item.keywords?.length || item.ocr_text) && (
          <Section title="AI Analysis">
            {item.description && <Row label="Description" value={item.description} />}
            {item.scene_type && <Row label="Scene" value={item.scene_type} />}
            {item.main_subject && <Row label="Subject" value={item.main_subject} />}
            {item.lighting && <Row label="Lighting" value={item.lighting} />}
            {item.mood && <Row label="Mood" value={item.mood} />}
            {item.ocr_text && <Row label="OCR Text" value={item.ocr_text} />}
            {item.detected_objects && item.detected_objects.length > 0 && (
              <div className="py-1 border-b border-neutral-800">
                <span className="text-neutral-500 text-xs block mb-1">Objects</span>
                <TagList items={item.detected_objects} />
              </div>
            )}
            {item.keywords && item.keywords.length > 0 && (
              <div className="py-1 border-b border-neutral-800">
                <span className="text-neutral-500 text-xs block mb-1">Keywords</span>
                <TagList items={item.keywords} />
              </div>
            )}
          </Section>
        )}

        {/* Faces */}
        {(item.face_count !== null || (item.face_identities && item.face_identities.length > 0)) && (
          <Section title="Faces">
            {item.face_count !== null && <Row label="Count" value={item.face_count} />}
            {item.face_identities && item.face_identities.length > 0 && (
              <div className="py-1 border-b border-neutral-800">
                <span className="text-neutral-500 text-xs block mb-1">Identities</span>
                <TagList items={item.face_identities} />
              </div>
            )}
          </Section>
        )}

        {/* Technical */}
        {(item.sharpness_score !== null || item.exposure_ev !== null || item.noise_level !== null ||
          item.snr_db !== null || item.dynamic_range_stops !== null ||
          item.dominant_colors?.length) && (
          <Section title="Technical">
            {item.sharpness_score !== null && (
              <div className="py-1 border-b border-neutral-800">
                <div className="flex justify-between mb-1">
                  <span className="text-neutral-500 text-xs">Sharpness</span>
                  {item.sharpness_label && <span className="text-xs text-neutral-400">{item.sharpness_label}</span>}
                </div>
                <ScoreBar value={item.sharpness_score} max={100} />
              </div>
            )}
            {item.exposure_ev !== null && (
              <Row label="Exposure EV" value={`${item.exposure_ev > 0 ? '+' : ''}${item.exposure_ev.toFixed(2)}${item.exposure_label ? ` (${item.exposure_label})` : ''}`} />
            )}
            {item.noise_level !== null && (
              <Row label="Noise" value={`${item.noise_level.toFixed(3)}${item.noise_label ? ` (${item.noise_label})` : ''}`} />
            )}
            {item.snr_db !== null && <Row label="SNR" value={`${item.snr_db.toFixed(1)} dB`} />}
            {item.dynamic_range_stops !== null && (
              <Row label="Dynamic Range" value={`${item.dynamic_range_stops.toFixed(1)} stops`} />
            )}
            {item.highlight_clipping_pct !== null && (
              <Row label="Highlight Clip" value={`${item.highlight_clipping_pct.toFixed(2)}%`} />
            )}
            {item.shadow_clipping_pct !== null && (
              <Row label="Shadow Clip" value={`${item.shadow_clipping_pct.toFixed(2)}%`} />
            )}
            {item.avg_saturation !== null && (
              <Row label="Saturation" value={item.avg_saturation.toFixed(2)} />
            )}
            {item.dominant_colors && item.dominant_colors.length > 0 && (
              <div className="py-1 border-b border-neutral-800">
                <span className="text-neutral-500 text-xs block mb-1">Colors</span>
                <div className="flex gap-1 flex-wrap">
                  {item.dominant_colors.map((c, i) => (
                    <div key={i} className="flex items-center gap-1">
                      <div className="w-4 h-4 rounded border border-neutral-600" style={{ backgroundColor: c }} />
                      <span className="text-[10px] text-neutral-400">{c}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </Section>
        )}

        {/* Camera / EXIF */}
        {(item.camera_make || item.camera_model || item.f_number || item.exposure_time ||
          item.focal_length || item.iso || item.date_time_original) && (
          <Section title="Camera">
            {(item.camera_make || item.camera_model) && (
              <Row label="Camera" value={[item.camera_make, item.camera_model].filter(Boolean).join(' ')} />
            )}
            {item.lens_model && <Row label="Lens" value={item.lens_model} />}
            {item.f_number && <Row label="Aperture" value={`f/${item.f_number}`} />}
            {item.exposure_time && <Row label="Shutter" value={item.exposure_time} />}
            {item.focal_length && <Row label="Focal Length" value={`${item.focal_length} mm`} />}
            {item.iso && <Row label="ISO" value={item.iso} />}
            {item.date_time_original && <Row label="Date" value={item.date_time_original.split('T')[0]} />}
            {(item.width || item.height) && (
              <Row label="Dimensions" value={`${item.width ?? '?'} × ${item.height ?? '?'}`} />
            )}
            {(item.gps_latitude || item.gps_longitude) && (
              <Row label="GPS" value={`${item.gps_latitude}, ${item.gps_longitude}`} />
            )}
            {(item.location_city || item.location_country) && (
              <Row label="Location" value={[item.location_city, item.location_state, item.location_country].filter(Boolean).join(', ')} />
            )}
          </Section>
        )}

        {/* File info */}
        <Section title="File">
          <Row label="Path" value={<span className="break-all text-[10px]">{item.file_path}</span>} />
          {item.file_size && (
            <Row label="Size" value={`${(item.file_size / 1024 / 1024).toFixed(1)} MB`} />
          )}
        </Section>
      </div>
    </div>
  )
}

// ── Main lightbox ─────────────────────────────────────────────────────────────

export function SearchLightbox({ item, items, onClose, onNavigate }: SearchLightboxProps) {
  const [thumb, setThumb] = useState<string>('')
  const [src, setSrc] = useState<string>('')

  // Zoom/pan state
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const isPanning = useRef(false)
  const panStart = useRef({ mx: 0, my: 0, ox: 0, oy: 0 })
  const containerRef = useRef<HTMLDivElement>(null)

  // Reset zoom when image changes
  useEffect(() => {
    setZoom(1)
    setOffset({ x: 0, y: 0 })
  }, [item.file_path])

  // Load thumbnail → full-res
  useEffect(() => {
    let cancelled = false
    setThumb('')
    setSrc('')

    window.api.getThumbnail(item.file_path).then((url) => {
      if (!cancelled && url) setThumb(url)
    }).catch(() => {})

    window.api.getFullImage(item.file_path).then((url) => {
      if (!cancelled && url) setSrc(url)
    }).catch(() => {})

    return () => { cancelled = true }
  }, [item.file_path])

  // Navigation
  const currentIdx = items.findIndex((i) => i.image_id === item.image_id)
  const prev = currentIdx > 0 ? items[currentIdx - 1] : null
  const next = currentIdx < items.length - 1 ? items[currentIdx + 1] : null

  // Zoom helpers
  const zoomToward = useCallback((delta: number, cx: number, cy: number) => {
    setZoom((prevZoom) => {
      const n = clamp(prevZoom * (1 + delta), MIN_ZOOM, MAX_ZOOM)
      const scale = n / prevZoom
      setOffset((prev) => ({ x: cx + (prev.x - cx) * scale, y: cy + (prev.y - cy) * scale }))
      return n
    })
  }, [])

  const resetZoom = useCallback(() => { setZoom(1); setOffset({ x: 0, y: 0 }) }, [])

  // Keyboard
  const handleKey = useCallback((e: KeyboardEvent) => {
    if (e.ctrlKey || e.metaKey) return
    switch (e.key) {
      case 'Escape':
        if (zoom !== 1 || offset.x !== 0 || offset.y !== 0) { resetZoom() } else { onClose() }
        break
      case 'ArrowLeft':
        if (zoom === 1) { if (prev) onNavigate(prev) }
        else setOffset((o) => ({ ...o, x: o.x + 80 }))
        break
      case 'ArrowRight':
        if (zoom === 1) { if (next) onNavigate(next) }
        else setOffset((o) => ({ ...o, x: o.x - 80 }))
        break
      case 'ArrowUp':    if (zoom > 1) setOffset((o) => ({ ...o, y: o.y + 80 })); break
      case 'ArrowDown':  if (zoom > 1) setOffset((o) => ({ ...o, y: o.y - 80 })); break
      case '+': case '=': zoomToward(ZOOM_STEP_KEY, 0, 0); break
      case '-': zoomToward(-ZOOM_STEP_KEY, 0, 0); break
      case '0': resetZoom(); break
    }
  }, [zoom, offset, onClose, prev, next, onNavigate, zoomToward, resetZoom])

  useEffect(() => {
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [handleKey])

  // Mouse wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault()
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return
    const cx = e.clientX - rect.left - rect.width / 2
    const cy = e.clientY - rect.top - rect.height / 2
    const delta = -Math.sign(e.deltaY) * ZOOM_STEP_WHEEL * (Math.abs(e.deltaY) > 100 ? 1.5 : 1)
    zoomToward(delta, cx, cy)
  }, [zoomToward])

  // Drag to pan
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (zoom <= 1) return
    e.preventDefault()
    isPanning.current = true
    panStart.current = { mx: e.clientX, my: e.clientY, ox: offset.x, oy: offset.y }
  }, [zoom, offset])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning.current) return
    setOffset({ x: panStart.current.ox + (e.clientX - panStart.current.mx), y: panStart.current.oy + (e.clientY - panStart.current.my) })
  }, [])

  const handleMouseUp = useCallback(() => { isPanning.current = false }, [])

  // Double-click toggle
  const handleDblClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (zoom !== 1) {
      resetZoom()
    } else {
      const rect = containerRef.current?.getBoundingClientRect()
      if (!rect) return
      const cx = e.clientX - rect.left - rect.width / 2
      const cy = e.clientY - rect.top - rect.height / 2
      setZoom(2)
      setOffset({ x: -cx, y: -cy })
    }
  }, [zoom, resetZoom])

  const isZoomed = zoom !== 1 || offset.x !== 0 || offset.y !== 0
  const zoomPct = Math.round(zoom * 100)

  return (
    <div className="fixed inset-0 z-50 flex bg-black/90">

      {/* ── Image area ─────────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 relative">

        {/* Top bar */}
        <div className="flex items-center justify-between px-3 py-2 shrink-0">
          <button
            onClick={onClose}
            className="p-2 rounded-full bg-black/50 hover:bg-black/80 text-white transition-colors"
            title="Close (Esc)"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>

          {/* Zoom controls */}
          <div className="flex items-center gap-1">
            <button onClick={() => zoomToward(-ZOOM_STEP_KEY, 0, 0)} className="p-1.5 rounded bg-black/50 hover:bg-black/80 text-white transition-colors" title="Zoom out (−)">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11A6 6 0 115 11a6 6 0 0112 0zM8 11h6" />
              </svg>
            </button>
            <button onClick={resetZoom} className={`px-2 py-1 rounded text-xs tabular-nums transition-colors ${isZoomed ? 'bg-blue-600/70 hover:bg-blue-600 text-white' : 'bg-black/50 hover:bg-black/80 text-neutral-400'}`} title="Reset zoom (0)">
              {zoomPct}%
            </button>
            <button onClick={() => zoomToward(ZOOM_STEP_KEY, 0, 0)} className="p-1.5 rounded bg-black/50 hover:bg-black/80 text-white transition-colors" title="Zoom in (+)">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11A6 6 0 115 11a6 6 0 0112 0zM11 8v6M8 11h6" />
              </svg>
            </button>
          </div>
        </div>

        {/* Image canvas */}
        <div
          ref={containerRef}
          className={`flex-1 overflow-hidden flex items-center justify-center min-w-0 ${zoom > 1 ? 'cursor-grab active:cursor-grabbing' : 'cursor-default'}`}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onDoubleClick={handleDblClick}
        >
          {!thumb && !src && (
            <div className="w-8 h-8 border-2 border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
          )}

          {thumb && !src && (
            <img
              src={thumb}
              alt=""
              className="max-w-full max-h-full object-contain select-none"
              style={{ filter: 'blur(8px)', transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom * 1.05})`, transformOrigin: 'center center' }}
              draggable={false}
            />
          )}

          {src && (
            <img
              src={src}
              alt=""
              className="max-w-full max-h-full object-contain select-none"
              style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`, transformOrigin: 'center center', transition: isPanning.current ? 'none' : 'transform 0.1s ease-out' }}
              draggable={false}
            />
          )}

          {thumb && !src && (
            <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 text-neutral-400 text-xs">
              <div className="w-3 h-3 border border-neutral-500 border-t-neutral-300 rounded-full animate-spin" />
              Loading full resolution…
            </div>
          )}
        </div>

        {/* Bottom nav */}
        {!isZoomed && (
          <div className="shrink-0 flex items-center justify-center gap-3 pb-4">
            <button onClick={() => prev && onNavigate(prev)} disabled={!prev}
              className="p-2 rounded-full bg-black/50 hover:bg-black/80 text-white transition-colors disabled:opacity-20 disabled:cursor-default" title="Previous (←)">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <span className="text-neutral-500 text-xs tabular-nums">
              {currentIdx + 1} / {items.length}
            </span>
            <button onClick={() => next && onNavigate(next)} disabled={!next}
              className="p-2 rounded-full bg-black/50 hover:bg-black/80 text-white transition-colors disabled:opacity-20 disabled:cursor-default" title="Next (→)">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>
        )}
      </div>

      {/* ── Analysis sidebar (right) ────────────────────────────────────────── */}
      <AnalysisSidebar item={item} />
    </div>
  )
}
