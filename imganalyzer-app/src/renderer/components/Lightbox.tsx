import { useEffect, useState, useCallback, useRef } from 'react'
import type { ImageFile } from '../global'
import { Sidebar } from './Sidebar'
import { useAnalysis } from '../hooks/useAnalysis'

interface LightboxProps {
  image: ImageFile
  images: ImageFile[]
  onClose: () => void
  onNavigate: (image: ImageFile) => void
}

const MIN_ZOOM = 0.1
const MAX_ZOOM = 10
const ZOOM_STEP_KEY = 0.25   // +/- per keypress
const ZOOM_STEP_WHEEL = 0.12 // per wheel tick (multiplied by deltaY magnitude)

function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)) }

export function Lightbox({ image, images, onClose, onNavigate }: LightboxProps) {
  // thumb = low-res placeholder shown immediately while full-res loads
  const [thumb, setThumb] = useState<string>('')
  // src = full-resolution image; replaces thumb once loaded
  const [src, setSrc] = useState<string>('')
  const { state, reanalyze, cancel } = useAnalysis(image.path)

  // ── Zoom / pan state ────────────────────────────────────────────────────────
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const isPanning = useRef(false)
  const panStart = useRef({ mx: 0, my: 0, ox: 0, oy: 0 })
  const containerRef = useRef<HTMLDivElement>(null)

  // Reset zoom+pan whenever the image changes
  useEffect(() => {
    setZoom(1)
    setOffset({ x: 0, y: 0 })
  }, [image.path])

  // Load thumbnail first (fast), then replace with full-res
  useEffect(() => {
    let cancelled = false
    setThumb('')
    setSrc('')

    // Step 1: low-res thumbnail as placeholder (already cached in most cases)
    window.api.getThumbnail(image.path).then((url) => {
      if (!cancelled && url) setThumb(url)
    }).catch(() => {})

    // Step 2: full-resolution image
    window.api.getFullImage(image.path).then((url) => {
      if (!cancelled && url) {
        setSrc(url)
      }
    }).catch(() => {})

    return () => { cancelled = true }
  }, [image.path])

  // ── Navigation ──────────────────────────────────────────────────────────────
  const currentIdx = images.findIndex((i) => i.path === image.path)
  const prev = currentIdx > 0 ? images[currentIdx - 1] : null
  const next = currentIdx < images.length - 1 ? images[currentIdx + 1] : null

  // ── Zoom helpers ────────────────────────────────────────────────────────────
  // Zoom toward a focal point (cx, cy) in container-relative coords
  const zoomToward = useCallback((delta: number, cx: number, cy: number) => {
    setZoom((prevZoom) => {
      const next = clamp(prevZoom * (1 + delta), MIN_ZOOM, MAX_ZOOM)
      const scale = next / prevZoom
      setOffset((prev) => ({
        x: cx + (prev.x - cx) * scale,
        y: cy + (prev.y - cy) * scale,
      }))
      return next
    })
  }, [])

  const resetZoom = useCallback(() => {
    setZoom(1)
    setOffset({ x: 0, y: 0 })
  }, [])

  // ── Keyboard ────────────────────────────────────────────────────────────────
  const handleKey = useCallback((e: KeyboardEvent) => {
    // Don't intercept when modifier keys are held (let browser shortcuts work)
    if (e.ctrlKey || e.metaKey) return

    switch (e.key) {
      case 'Escape':
        if (zoom !== 1 || offset.x !== 0 || offset.y !== 0) {
          resetZoom()
        } else {
          onClose()
        }
        break
      case 'ArrowLeft':
        if (zoom === 1) { if (prev) onNavigate(prev) }
        else setOffset((o) => ({ ...o, x: o.x + 80 }))
        break
      case 'ArrowRight':
        if (zoom === 1) { if (next) onNavigate(next) }
        else setOffset((o) => ({ ...o, x: o.x - 80 }))
        break
      case 'ArrowUp':
        if (zoom > 1) setOffset((o) => ({ ...o, y: o.y + 80 }))
        break
      case 'ArrowDown':
        if (zoom > 1) setOffset((o) => ({ ...o, y: o.y - 80 }))
        break
      case '+':
      case '=':
        zoomToward(ZOOM_STEP_KEY, 0, 0)
        break
      case '-':
        zoomToward(-ZOOM_STEP_KEY, 0, 0)
        break
      case '0':
        resetZoom()
        break
    }
  }, [zoom, offset, onClose, prev, next, onNavigate, zoomToward, resetZoom])

  useEffect(() => {
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [handleKey])

  // ── Mouse wheel zoom ────────────────────────────────────────────────────────
  const handleWheel = useCallback((e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault()
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return
    // Focal point relative to container center
    const cx = e.clientX - rect.left - rect.width / 2
    const cy = e.clientY - rect.top - rect.height / 2
    const delta = -Math.sign(e.deltaY) * ZOOM_STEP_WHEEL * (Math.abs(e.deltaY) > 100 ? 1.5 : 1)
    zoomToward(delta, cx, cy)
  }, [zoomToward])

  // ── Drag to pan (only when zoomed in) ───────────────────────────────────────
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (zoom <= 1) return
    e.preventDefault()
    isPanning.current = true
    panStart.current = { mx: e.clientX, my: e.clientY, ox: offset.x, oy: offset.y }
  }, [zoom, offset])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning.current) return
    setOffset({
      x: panStart.current.ox + (e.clientX - panStart.current.mx),
      y: panStart.current.oy + (e.clientY - panStart.current.my),
    })
  }, [])

  const handleMouseUp = useCallback(() => {
    isPanning.current = false
  }, [])

  // Double-click: toggle fit ↔ 2×
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
      {/* Close button */}
      <button
        onClick={onClose}
        className="absolute top-4 left-4 z-10 p-2 rounded-full bg-black/50 hover:bg-black/80 text-white transition-colors"
        title="Close (Esc)"
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>

      {/* Prev button — hidden when zoomed in */}
      {prev && !isZoomed && (
        <button
          onClick={() => onNavigate(prev)}
          className="absolute left-4 top-1/2 -translate-y-1/2 z-10 p-2 rounded-full bg-black/50 hover:bg-black/80 text-white transition-colors"
          title="Previous (←)"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
          </svg>
        </button>
      )}

      {/* Next button — hidden when zoomed in */}
      {next && !isZoomed && (
        <button
          onClick={() => onNavigate(next)}
          className="absolute right-[21rem] top-1/2 -translate-y-1/2 z-10 p-2 rounded-full bg-black/50 hover:bg-black/80 text-white transition-colors"
          title="Next (→)"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
          </svg>
        </button>
      )}

      {/* Zoom controls */}
      <div className="absolute top-4 right-[21.5rem] z-10 flex items-center gap-1">
        <button
          onClick={() => zoomToward(-ZOOM_STEP_KEY, 0, 0)}
          className="p-1.5 rounded bg-black/50 hover:bg-black/80 text-white transition-colors"
          title="Zoom out (−)"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11A6 6 0 115 11a6 6 0 0112 0zM8 11h6" />
          </svg>
        </button>
        <button
          onClick={resetZoom}
          className={`px-2 py-1 rounded text-xs tabular-nums transition-colors ${isZoomed ? 'bg-blue-600/70 hover:bg-blue-600 text-white' : 'bg-black/50 hover:bg-black/80 text-neutral-400'}`}
          title="Reset zoom (0)"
        >
          {zoomPct}%
        </button>
        <button
          onClick={() => zoomToward(ZOOM_STEP_KEY, 0, 0)}
          className="p-1.5 rounded bg-black/50 hover:bg-black/80 text-white transition-colors"
          title="Zoom in (+)"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11A6 6 0 115 11a6 6 0 0112 0zM11 8v6M8 11h6" />
          </svg>
        </button>
      </div>

      {/* Image area */}
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
        {/* Show spinner only when nothing at all is available yet */}
        {!thumb && !src && (
          <div className="w-8 h-8 border-2 border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
        )}

        {/* Blurred thumbnail placeholder — visible until full-res is ready */}
        {thumb && !src && (
          <img
            src={thumb}
            alt={image.name}
            className="max-w-full max-h-full object-contain select-none"
            style={{
              filter: 'blur(8px)',
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom * 1.05})`,
              transformOrigin: 'center center',
            }}
            draggable={false}
          />
        )}

        {/* Full-resolution image */}
        {src && (
          <img
            src={src}
            alt={image.name}
            className="max-w-full max-h-full object-contain select-none"
            style={{
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
              transformOrigin: 'center center',
              transition: isPanning.current ? 'none' : 'transform 0.1s ease-out',
            }}
            draggable={false}
          />
        )}

        {/* Loading indicator while full-res is still fetching */}
        {thumb && !src && (
          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 text-neutral-400 text-xs">
            <div className="w-3 h-3 border border-neutral-500 border-t-neutral-300 rounded-full animate-spin" />
            Loading full resolution…
          </div>
        )}
      </div>

      {/* Counter */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-neutral-500 text-xs tabular-nums">
        {currentIdx + 1} / {images.length}
        {isZoomed && <span className="ml-2 text-neutral-600">· Esc to reset</span>}
      </div>

      {/* Sidebar */}
      <Sidebar
        imageName={image.name}
        state={state}
        onReanalyze={reanalyze}
        onCancel={cancel}
      />
    </div>
  )
}
