import { useState, useEffect } from 'react'
import type { ImageFile } from '../global'

interface ThumbnailProps {
  image: ImageFile
  onClick: () => void
  selected: boolean
}

export function Thumbnail({ image, onClick, selected }: ThumbnailProps) {
  const [src, setSrc] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(false)
    setSrc('')

    window.api.getThumbnail(image.path).then((dataUrl) => {
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
  }, [image.path])

  return (
    <button
      onClick={onClick}
      className={`
        relative group rounded overflow-hidden aspect-square bg-neutral-900
        ring-2 transition-all focus:outline-none
        ${selected ? 'ring-blue-500' : 'ring-transparent hover:ring-neutral-600'}
      `}
    >
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-5 h-5 border-2 border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
        </div>
      )}
      {error && !loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 text-neutral-500">
          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3 3l18 18M3.75 9A.75.75 0 014.5 8.25H6" />
          </svg>
          <span className="text-xs">{image.ext.toUpperCase()}</span>
        </div>
      )}
      {src && (
        <img
          src={src}
          alt={image.name}
          className="w-full h-full object-cover"
          draggable={false}
        />
      )}
      {/* XMP badge */}
      {image.hasXmp && (
        <div className="absolute top-1 right-1 w-2 h-2 rounded-full bg-green-500 shadow" title="Has XMP sidecar" />
      )}
      {/* RAW badge */}
      {image.isRaw && (
        <div className="absolute bottom-1 left-1 px-1 py-0.5 rounded text-[10px] font-bold bg-black/70 text-yellow-400 leading-none">
          RAW
        </div>
      )}
      {/* Filename on hover */}
      <div className="absolute bottom-0 left-0 right-0 px-1.5 py-1 bg-black/60 text-[11px] text-white truncate
                      opacity-0 group-hover:opacity-100 transition-opacity">
        {image.name}
      </div>
    </button>
  )
}
