import { memo, useEffect, useRef, useState } from 'react'
import {
  getCachedFaceThumb,
  requestFaceThumbnail,
  requestImageThumbnail,
} from '../../lib/thumbnailCache'

// ── Face crop thumbnail (batch-loaded with LRU cache) ─────────────────────────

export const FaceCropThumbnail = memo(function FaceCropThumbnail({
  occurrenceId,
  size = 'md',
}: {
  occurrenceId: number
  size?: 'sm' | 'md' | 'lg' | 'fill'
}) {
  const [src, setSrc] = useState<string | null>(() => getCachedFaceThumb(occurrenceId) ?? null)
  const [failed, setFailed] = useState(false)
  const requested = useRef(false)
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const cached = getCachedFaceThumb(occurrenceId) ?? null
    setSrc(cached)
    setFailed(false)
    requested.current = cached !== null
  }, [occurrenceId])

  useEffect(() => {
    if (src || requested.current) return
    const node = containerRef.current
    if (!node) return

    let cancelled = false
    const load = () => {
      if (requested.current || cancelled) return
      requested.current = true
      requestFaceThumbnail(occurrenceId, (url) => {
        if (cancelled) return
        if (url) {
          setSrc(url)
        } else {
          setFailed(true)
        }
      })
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          observer.disconnect()
          load()
        }
      },
      { rootMargin: '200px' },
    )
    observer.observe(node)

    return () => {
      cancelled = true
      observer.disconnect()
    }
  }, [occurrenceId, src])

  const sizeClass =
    size === 'sm'
      ? 'w-12 h-12'
      : size === 'lg'
        ? 'w-24 h-24'
        : size === 'fill'
          ? 'w-full h-full'
          : 'w-16 h-16'

  const shrink = size === 'fill' ? '' : 'shrink-0'
  const wrapperClass = `${sizeClass} rounded overflow-hidden ${shrink}`

  if (failed) {
    return (
      <div
        ref={containerRef}
        className={`${wrapperClass} bg-neutral-800 flex items-center justify-center`}
      >
        <svg
          className="w-5 h-5 text-neutral-600"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"
          />
        </svg>
      </div>
    )
  }

  if (!src) {
    return (
      <div
        ref={containerRef}
        className={`${wrapperClass} bg-neutral-800 animate-pulse`}
      />
    )
  }

  return (
    <div ref={containerRef} className={wrapperClass}>
      <img
        src={src}
        alt=""
        className="w-full h-full object-cover"
        loading="lazy"
        decoding="async"
        draggable={false}
      />
    </div>
  )
})

// ── Full image thumbnail (lazy-loaded, for legacy mode) ───────────────────────

export function ImageThumbnail({ filePath, imageId }: { filePath: string; imageId?: number | null }) {
  const [src, setSrc] = useState<string | null>(null)

  useEffect(() => {
    setSrc(null)
  }, [filePath, imageId])

  useEffect(() => {
    let cancelled = false
    requestImageThumbnail(filePath, imageId, (data) => {
      if (!cancelled) setSrc(data)
    })
    return () => { cancelled = true }
  }, [filePath, imageId])

  if (!src) {
    return (
      <div className="w-24 h-24 rounded bg-neutral-800 animate-pulse shrink-0" />
    )
  }

  return (
    <img
      src={src}
      alt=""
      className="w-24 h-24 rounded object-cover shrink-0"
      loading="lazy"
      decoding="async"
      draggable={false}
    />
  )
}
