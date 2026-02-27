/**
 * VirtualGrid.tsx — Virtualised image grid with progressive thumbnail loading.
 *
 * Algorithm:
 *  1. Measure container width via ResizeObserver.
 *  2. Derive column count from width and CELL_SIZE.
 *  3. Compute total virtual height = Math.ceil(items/cols) * rowHeight.
 *  4. On scroll, compute which rows are in view (plus OVERSCAN).
 *  5. Only mount <GridCell> for visible rows — invisible rows are a spacer div.
 *  6. Each <GridCell> requests its thumbnail only when it enters the viewport
 *     (IntersectionObserver), providing a second layer of progressive loading.
 */
import { useState, useEffect, useRef, useCallback, memo } from 'react'
import type { SearchResult } from '../global'

// ── Constants ─────────────────────────────────────────────────────────────────

const CELL_MIN = 160  // minimum cell width in px
const CELL_MAX = 220  // maximum cell width — drives column count
const OVERSCAN_ROWS = 3  // extra rows rendered above/below viewport

// ── Types ─────────────────────────────────────────────────────────────────────

interface VirtualGridProps {
  items: SearchResult[]
  selectedId: number | null
  onSelect: (item: SearchResult) => void
}

// ── GridCell ─────────────────────────────────────────────────────────────────

interface GridCellProps {
  item: SearchResult
  selected: boolean
  onClick: () => void
}

const GridCell = memo(function GridCell({ item, selected, onClick }: GridCellProps) {
  const [src, setSrc] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)
  const cellRef = useRef<HTMLButtonElement>(null)
  const loadedRef = useRef(false)

  // Use IntersectionObserver to defer thumbnail loading until cell is visible
  useEffect(() => {
    const el = cellRef.current
    if (!el) return

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !loadedRef.current) {
          loadedRef.current = true
          observer.disconnect()

          let cancelled = false
          setLoading(true)
          setError(false)

          window.api.getThumbnail(item.file_path).then((dataUrl) => {
            if (cancelled) return
            if (dataUrl) {
              setSrc(dataUrl)
            } else {
              setError(true)
            }
            setLoading(false)
          }).catch(() => {
            if (!cancelled) {
              setError(true)
              setLoading(false)
            }
          })

          return () => { cancelled = true }
        }
      },
      { rootMargin: '200px' }  // start loading 200px before it enters view
    )

    observer.observe(el)
    return () => observer.disconnect()
  }, [item.file_path])

  // Aesthetic score badge color
  const scoreColor = item.aesthetic_score !== null
    ? item.aesthetic_score >= 7 ? 'bg-green-500/80'
      : item.aesthetic_score >= 5 ? 'bg-yellow-500/80'
      : 'bg-red-500/80'
    : null

  const ext = item.file_path.split('.').pop()?.toUpperCase() ?? ''
  const isRaw = ['ARW','CR2','CR3','NEF','NRW','ORF','RAF','RW2','DNG','PEF',
                 'SRW','ERF','KDC','MRW','3FR','FFF','SR2','SRF','X3F','IIQ','MOS','RAW']
    .includes(ext)

  return (
    <button
      ref={cellRef}
      onClick={onClick}
      className={`
        relative group rounded overflow-hidden aspect-square bg-neutral-900
        ring-2 transition-all focus:outline-none w-full h-full
        ${selected ? 'ring-blue-500' : 'ring-transparent hover:ring-neutral-600'}
      `}
    >
      {/* Loading spinner */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-5 h-5 border-2 border-neutral-700 border-t-neutral-400 rounded-full animate-spin" />
        </div>
      )}

      {/* Error state */}
      {error && !loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 text-neutral-600">
          <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3 3l18 18" />
          </svg>
          <span className="text-[10px]">{ext}</span>
        </div>
      )}

      {/* Thumbnail */}
      {src && (
        <img
          src={src}
          alt={item.file_path.split(/[/\\]/).pop() ?? ''}
          className="w-full h-full object-cover"
          draggable={false}
        />
      )}

      {/* Aesthetic score badge */}
      {item.aesthetic_score !== null && scoreColor && (
        <div className={`absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded text-[10px] font-bold text-white ${scoreColor} shadow`}>
          {item.aesthetic_score.toFixed(1)}
        </div>
      )}

      {/* RAW badge */}
      {isRaw && (
        <div className="absolute bottom-1 left-1 px-1 py-0.5 rounded text-[10px] font-bold bg-black/70 text-yellow-400 leading-none">
          RAW
        </div>
      )}

      {/* Face count badge */}
      {item.face_count !== null && item.face_count > 0 && (
        <div className="absolute top-1.5 right-1.5 flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-purple-600/80 text-[10px] text-white font-medium">
          <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
          </svg>
          {item.face_count}
        </div>
      )}

      {/* Filename on hover */}
      <div className="absolute bottom-0 left-0 right-0 px-1.5 py-1 bg-gradient-to-t from-black/80 to-transparent
                      text-[11px] text-white truncate opacity-0 group-hover:opacity-100 transition-opacity">
        {item.file_path.split(/[/\\]/).pop()}
      </div>

      {/* Score overlay on hover (if search score available) */}
      {item.score !== null && (
        <div className="absolute top-0 left-0 right-0 flex justify-end px-1.5 py-1
                        opacity-0 group-hover:opacity-100 transition-opacity">
          <span className="text-[10px] bg-black/60 px-1 py-0.5 rounded text-neutral-300">
            {(item.score * 100).toFixed(0)}%
          </span>
        </div>
      )}
    </button>
  )
})

// ── VirtualGrid ───────────────────────────────────────────────────────────────

export function VirtualGrid({ items, selectedId, onSelect }: VirtualGridProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(800)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewportHeight, setViewportHeight] = useState(600)

  // ── Measure container ─────────────────────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width
        const h = entry.contentRect.height
        if (w > 0) setContainerWidth(w)
        if (h > 0) setViewportHeight(h)
      }
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // ── Grid layout maths ─────────────────────────────────────────────────────
  const GAP = 8  // gap in px (matches gap-2)
  const PADDING = 16  // px-4

  const availableWidth = containerWidth - PADDING * 2

  // Compute column count: fill with cells of CELL_MIN..CELL_MAX
  const cols = Math.max(1, Math.floor((availableWidth + GAP) / (CELL_MIN + GAP)))
  const cellSize = Math.min(CELL_MAX, Math.floor((availableWidth - GAP * (cols - 1)) / cols))

  const rowCount = Math.ceil(items.length / cols)
  const rowHeight = cellSize + GAP
  const totalHeight = rowCount * rowHeight + PADDING * 2

  // ── Visible row range (with overscan) ─────────────────────────────────────
  const firstVisibleRow = Math.max(0, Math.floor((scrollTop - PADDING) / rowHeight) - OVERSCAN_ROWS)
  const lastVisibleRow  = Math.min(
    rowCount - 1,
    Math.ceil((scrollTop + viewportHeight - PADDING) / rowHeight) + OVERSCAN_ROWS
  )

  // ── Scroll handler ────────────────────────────────────────────────────────
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop((e.target as HTMLDivElement).scrollTop)
  }, [])

  if (items.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-neutral-600 text-sm">
        No images found
      </div>
    )
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto"
      onScroll={handleScroll}
      style={{ position: 'relative' }}
    >
      {/* Virtual spacer — establishes scrollable height */}
      <div style={{ height: totalHeight, position: 'relative' }}>

        {/* Top padding space before first visible row */}
        {firstVisibleRow > 0 && (
          <div style={{ height: firstVisibleRow * rowHeight + PADDING }} />
        )}

        {/* Render only visible rows */}
        {Array.from({ length: lastVisibleRow - firstVisibleRow + 1 }, (_, i) => {
          const row = firstVisibleRow + i
          const startIdx = row * cols
          const rowItems = items.slice(startIdx, startIdx + cols)

          return (
            <div
              key={row}
              style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
                gap: GAP,
                paddingLeft: PADDING,
                paddingRight: PADDING,
                paddingTop: row === 0 ? PADDING : 0,
                marginBottom: GAP,
              }}
            >
              {rowItems.map((item) => (
                <div key={item.image_id} style={{ width: cellSize, height: cellSize }}>
                  <GridCell
                    item={item}
                    selected={item.image_id === selectedId}
                    onClick={() => onSelect(item)}
                  />
                </div>
              ))}
            </div>
          )
        })}
      </div>
    </div>
  )
}
