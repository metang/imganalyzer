import { useEffect, useState, useCallback, useRef } from 'react'
import type { ImageFile } from '../global'
import { Sidebar } from './Sidebar'
import { CloudSidebar } from './CloudSidebar'
import { useAnalysis } from '../hooks/useAnalysis'
import { useCloudAnalysis } from '../hooks/useCloudAnalysis'

interface LightboxProps {
  image: ImageFile
  images: ImageFile[]
  onClose: () => void
  onNavigate: (image: ImageFile) => void
}

const MIN_ZOOM = 0.1
const MAX_ZOOM = 10
const ZOOM_STEP_KEY = 0.25
const ZOOM_STEP_WHEEL = 0.12

function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)) }

/** Returns true when the local analysis result contains people. */
function hasPeople(state: ReturnType<typeof useAnalysis>['state']): boolean {
  if (state.status !== 'done' && state.status !== 'cached') return false
  const xmp = state.xmp
  if (xmp.faceCount !== undefined && xmp.faceCount > 0) return true
  if (xmp.detectedObjects?.some(obj => obj.toLowerCase().startsWith('person'))) return true
  return false
}

export function Lightbox({ image, images, onClose, onNavigate }: LightboxProps) {
  const [thumb, setThumb] = useState<string>('')
  const [src, setSrc] = useState<string>('')
  const [loadError, setLoadError] = useState(false)

  // Local model analysis (existing)
  const { state, reanalyze, cancel } = useAnalysis(image.path)
  // Cloud model analysis (new)
  const { state: cloudState, analyze: analyzeCloud } = useCloudAnalysis(image.path)

  // People-detection prompt: shown once per image when local finishes and
  // detects people, and cloud is still idle.
  const [peoplePromptDismissed, setPeoplePromptDismissed] = useState(false)

  // Reset the people prompt whenever the image changes
  useEffect(() => {
    setPeoplePromptDismissed(false)
  }, [image.path])

  const showPeoplePrompt =
    !peoplePromptDismissed &&
    cloudState.status === 'idle' &&
    hasPeople(state)

  // ── Zoom / pan state ──────────────────────────────────────────────────────
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const isPanning = useRef(false)
  const panStart = useRef({ mx: 0, my: 0, ox: 0, oy: 0 })
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setZoom(1)
    setOffset({ x: 0, y: 0 })
  }, [image.path])

  // Load thumbnail first, then replace with full-res
  useEffect(() => {
    let cancelled = false
    setThumb('')
    setSrc('')
    setLoadError(false)

    window.api.getThumbnail(image.path).then((url) => {
      if (cancelled) return
      if (url) {
        setThumb(url)
      } else {
        setLoadError(true)
      }
    }).catch(() => {
      if (!cancelled) setLoadError(true)
    })

    window.api.getFullImage(image.path).then((url) => {
      if (cancelled) return
      if (url) {
        setSrc(url)
      } else {
        setLoadError(true)
      }
    }).catch(() => {
      if (!cancelled) setLoadError(true)
    })

    return () => { cancelled = true }
  }, [image.path])

  // ── Navigation ────────────────────────────────────────────────────────────
  const currentIdx = images.findIndex((i) => i.path === image.path)
  const prev = currentIdx > 0 ? images[currentIdx - 1] : null
  const next = currentIdx < images.length - 1 ? images[currentIdx + 1] : null

  // ── Zoom helpers ──────────────────────────────────────────────────────────
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

  // ── Keyboard ──────────────────────────────────────────────────────────────
  const handleKey = useCallback((e: KeyboardEvent) => {
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

  // ── Mouse wheel zoom ──────────────────────────────────────────────────────
  const handleWheel = useCallback((e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault()
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return
    const cx = e.clientX - rect.left - rect.width / 2
    const cy = e.clientY - rect.top - rect.height / 2
    const delta = -Math.sign(e.deltaY) * ZOOM_STEP_WHEEL * (Math.abs(e.deltaY) > 100 ? 1.5 : 1)
    zoomToward(delta, cx, cy)
  }, [zoomToward])

  // ── Drag to pan ───────────────────────────────────────────────────────────
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

  // ── People prompt handlers ────────────────────────────────────────────────
  const handlePeoplePromptAnalyze = useCallback(() => {
    setPeoplePromptDismissed(true)
    analyzeCloud()
  }, [analyzeCloud])

  const handlePeoplePromptDismiss = useCallback(() => {
    setPeoplePromptDismissed(true)
  }, [])

  return (
    <div className="fixed inset-0 z-50 flex bg-black/90">

      {/* ── Cloud AI sidebar (left) ─────────────────────────────────────── */}
      <CloudSidebar
        imageName={image.name}
        state={cloudState}
        onAnalyze={analyzeCloud}
      />

      {/* ── Image area ─────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 relative">

        {/* Top bar: close button + zoom controls */}
        <div className="flex items-center justify-between px-3 py-2 shrink-0">
          {/* Close */}
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
          {!thumb && !src && !loadError && (
            <div className="w-8 h-8 border-2 border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
          )}

          {loadError && !src && !thumb && (
            <div className="flex items-center justify-center w-full h-full text-zinc-400">
              <span>Failed to load image</span>
            </div>
          )}

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

          {thumb && !src && (
            <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 text-neutral-400 text-xs">
              <div className="w-3 h-3 border border-neutral-500 border-t-neutral-300 rounded-full animate-spin" />
              Loading full resolution…
            </div>
          )}
        </div>

        {/* Bottom bar: counter + nav + people prompt */}
        <div className="shrink-0 flex flex-col items-center gap-2 pb-3">
          {/* Navigation buttons */}
          {!isZoomed && (
            <div className="flex items-center gap-3">
              <button
                onClick={() => prev && onNavigate(prev)}
                disabled={!prev}
                className="p-2 rounded-full bg-black/50 hover:bg-black/80 text-white transition-colors disabled:opacity-20 disabled:cursor-default"
                title="Previous (←)"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
                </svg>
              </button>
              <span className="text-neutral-500 text-xs tabular-nums">
                {currentIdx + 1} / {images.length}
                {isZoomed && <span className="ml-2 text-neutral-600">· Esc to reset</span>}
              </span>
              <button
                onClick={() => next && onNavigate(next)}
                disabled={!next}
                className="p-2 rounded-full bg-black/50 hover:bg-black/80 text-white transition-colors disabled:opacity-20 disabled:cursor-default"
                title="Next (→)"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          )}

          {/* People-detection prompt */}
          {showPeoplePrompt && (
            <div className="flex items-center gap-3 px-4 py-2 rounded-lg bg-purple-950/80 border border-purple-800/60 text-sm shadow-lg">
              <svg className="w-4 h-4 text-purple-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19.128a9.38 9.38 0 002.625.372 9.337 9.337 0 004.121-.952 4.125 4.125 0 00-7.533-2.493M15 19.128v-.003c0-1.113-.285-2.16-.786-3.07M15 19.128v.106A12.318 12.318 0 018.624 21c-2.331 0-4.512-.645-6.374-1.766l-.001-.109a6.375 6.375 0 0111.964-3.07M12 6.375a3.375 3.375 0 11-6.75 0 3.375 3.375 0 016.75 0zm8.25 2.25a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z" />
              </svg>
              <span className="text-purple-200">People detected — analyze with Cloud AI?</span>
              <button
                onClick={handlePeoplePromptAnalyze}
                className="text-xs px-2.5 py-1 rounded bg-purple-700 hover:bg-purple-600 text-white transition-colors shrink-0"
              >
                Analyze
              </button>
              <button
                onClick={handlePeoplePromptDismiss}
                className="text-neutral-500 hover:text-neutral-300 transition-colors shrink-0"
                title="Dismiss"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* ── Local model sidebar (right) ─────────────────────────────────── */}
      <Sidebar
        imageName={image.name}
        state={state}
        onReanalyze={reanalyze}
        onCancel={cancel}
      />
    </div>
  )
}
