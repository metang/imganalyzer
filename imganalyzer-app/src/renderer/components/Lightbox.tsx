import { useEffect, useState, useCallback } from 'react'
import type { ImageFile } from '../global'
import { Sidebar } from './Sidebar'
import { useAnalysis } from '../hooks/useAnalysis'

interface LightboxProps {
  image: ImageFile
  images: ImageFile[]
  onClose: () => void
  onNavigate: (image: ImageFile) => void
}

export function Lightbox({ image, images, onClose, onNavigate }: LightboxProps) {
  const [src, setSrc] = useState<string>('')
  const { state, reanalyze, cancel } = useAnalysis(image.path)

  // Load full-resolution thumbnail (reuse getThumbnail — sharp/Python already resizes to fit 400px
  // but for the lightbox we want the best available, so we just display the data-url at full size)
  useEffect(() => {
    let cancelled = false
    setSrc('')
    window.api.getThumbnail(image.path).then((url) => {
      if (!cancelled) setSrc(url)
    })
    return () => { cancelled = true }
  }, [image.path])

  // Keyboard navigation
  const currentIdx = images.findIndex((i) => i.path === image.path)
  const prev = currentIdx > 0 ? images[currentIdx - 1] : null
  const next = currentIdx < images.length - 1 ? images[currentIdx + 1] : null

  const handleKey = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose()
    if (e.key === 'ArrowLeft' && prev) onNavigate(prev)
    if (e.key === 'ArrowRight' && next) onNavigate(next)
  }, [onClose, prev, next, onNavigate])

  useEffect(() => {
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [handleKey])

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

      {/* Prev button */}
      {prev && (
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

      {/* Next button */}
      {next && (
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

      {/* Image area */}
      <div className="flex-1 flex items-center justify-center min-w-0">
        {src ? (
          <img
            src={src}
            alt={image.name}
            className="max-w-full max-h-full object-contain"
            draggable={false}
          />
        ) : (
          <div className="w-8 h-8 border-2 border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
        )}
      </div>

      {/* Counter */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-neutral-500 text-xs tabular-nums">
        {currentIdx + 1} / {images.length}
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
