import { useEffect, useState } from 'react'
import type { SearchResult } from '../../global'
import { AnalysisSidebar } from '../SearchLightbox'

// ── Inline image lightbox (for viewing source image in-app) ───────────────────

export function FaceImageLightbox({
  filePath,
  imageId,
  onClose,
}: {
  filePath: string
  imageId?: number
  onClose: () => void
}) {
  const [src, setSrc] = useState<string | null>(null)
  const [metadata, setMetadata] = useState<SearchResult | null>(null)

  useEffect(() => {
    let cancelled = false
    window.api.getFullImage(filePath).then((url) => {
      if (!cancelled) setSrc(url)
    })
    return () => { cancelled = true }
  }, [filePath])

  useEffect(() => {
    let cancelled = false
    const params = imageId != null ? { image_id: imageId } : { file_path: filePath }
    window.api.getImageDetails(params).then((resp) => {
      if (!cancelled && resp.result) setMetadata(resp.result)
    })
    return () => { cancelled = true }
  }, [filePath, imageId])

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 flex bg-black/90"
      onClick={onClose}
    >
      {/* Image area */}
      <div className="flex-1 flex flex-col min-w-0 relative">
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

        {/* Image */}
        <div className="flex-1 overflow-hidden flex items-center justify-center p-4 min-w-0" onClick={(e) => e.stopPropagation()}>
          {src ? (
            <img
              src={src}
              alt=""
              className="max-w-full max-h-full object-contain rounded shadow-2xl select-none"
              draggable={false}
            />
          ) : (
            <div className="flex items-center gap-2 text-neutral-400">
              <span className="w-5 h-5 border-2 border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
              Loading...
            </div>
          )}
        </div>

        {/* File name */}
        <div className="shrink-0 flex justify-center pb-4">
          <div className="bg-black/70 px-3 py-1.5 rounded-lg">
            <p className="text-xs text-neutral-300 truncate max-w-md">
              {filePath.split(/[/\\]/).pop()}
            </p>
          </div>
        </div>
      </div>

      {/* Metadata sidebar */}
      {metadata && (
        <div className="h-full" onClick={(e) => e.stopPropagation()}>
          <AnalysisSidebar item={metadata} />
        </div>
      )}
    </div>
  )
}
