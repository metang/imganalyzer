import { memo, useEffect, useState } from 'react'
import type { PersonDirectLink, PersonSimilarImage } from '../../global'
import { requestFaceThumbnail, requestImageThumbnail } from '../../lib/thumbnailCache'

// ── Similar image card with lazy thumbnail loading ────────────────────────────

export const SimilarImageCard = memo(function SimilarImageCard({
  image,
  selected,
  onToggleSelect,
  onOpenLightbox,
}: {
  image: PersonSimilarImage
  selected?: boolean
  onToggleSelect?: (occurrenceId: number) => void
  onOpenLightbox: (filePath: string, imageId: number) => void
}) {
  const [thumb, setThumb] = useState<string | null>(null)
  const [faceCrop, setFaceCrop] = useState<string | null>(null)

  useEffect(() => {
    setThumb(null)
    setFaceCrop(null)
  }, [image.image_id, image.file_path, image.best_occurrence_id])

  useEffect(() => {
    let cancelled = false
    requestImageThumbnail(image.file_path, image.image_id, (data) => {
      if (!cancelled) setThumb(data)
    })
    return () => { cancelled = true }
  }, [image.file_path, image.image_id])

  useEffect(() => {
    if (image.best_occurrence_id == null) return
    let cancelled = false
    requestFaceThumbnail(image.best_occurrence_id, (data) => {
      if (!cancelled) setFaceCrop(data)
    })
    return () => { cancelled = true }
  }, [image.best_occurrence_id])

  return (
    <div
      className={`relative rounded-xl border transition-colors cursor-pointer overflow-hidden ${
        selected
          ? 'border-emerald-500/70 bg-emerald-950/20'
          : 'border-amber-800/40 bg-amber-950/10 hover:bg-amber-900/15'
      }`}
      onClick={() => onOpenLightbox(image.file_path, image.image_id)}
    >
      <div className="absolute left-2 top-2 rounded px-1.5 py-0.5 text-[10px] text-amber-200/80 bg-black/60 border border-amber-700/40 z-10">
        {(image.similarity * 100).toFixed(1)}%
      </div>
      {onToggleSelect && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            onToggleSelect(image.best_occurrence_id)
          }}
          className={`absolute right-2 top-2 z-10 rounded px-1.5 py-0.5 text-[10px] border ${
            selected
              ? 'border-emerald-500/70 bg-emerald-900/70 text-emerald-100'
              : 'border-neutral-600 bg-black/55 text-neutral-200'
          }`}
        >
          {selected ? 'Selected' : 'Select'}
        </button>
      )}
      {faceCrop && !onToggleSelect && (
        <div className="absolute right-2 top-2 z-10 h-8 w-8 rounded-full border-2 border-amber-600/70 overflow-hidden bg-neutral-800">
          <img src={faceCrop} alt="" className="h-full w-full object-cover" loading="lazy" decoding="async" />
        </div>
      )}
      {faceCrop && onToggleSelect && (
        <div className="absolute right-10 top-2 z-10 h-6 w-6 rounded-full border border-amber-600/70 overflow-hidden bg-neutral-800">
          <img src={faceCrop} alt="" className="h-full w-full object-cover" loading="lazy" decoding="async" />
        </div>
      )}
      <div className="aspect-[4/3] w-full overflow-hidden rounded-t-xl bg-neutral-800">
        {thumb ? (
          <img src={thumb} alt="" className="h-full w-full object-cover" loading="lazy" decoding="async" />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <svg className="h-8 w-8 text-neutral-700 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" />
            </svg>
          </div>
        )}
      </div>
      <div className="px-3 py-2">
        <p className="truncate text-[11px] text-neutral-400">
          {image.file_path.split(/[\\/]/).pop() ?? image.file_path}
        </p>
      </div>
    </div>
  )
})

// ── Direct link card (for images linked to person without cluster) ────────────

export const DirectLinkCard = memo(function DirectLinkCard({
  link,
  onUnlink,
  onOpenLightbox,
}: {
  link: PersonDirectLink
  onUnlink: () => void
  onOpenLightbox: () => void
}) {
  const [thumb, setThumb] = useState<string | null>(null)

  useEffect(() => {
    setThumb(null)
  }, [link.occurrence_id, link.file_path])

  useEffect(() => {
    let cancelled = false
    requestImageThumbnail(link.file_path, link.image_id, (data) => {
      if (!cancelled) setThumb(data)
    })
    return () => { cancelled = true }
  }, [link.file_path, link.image_id])

  return (
    <div
      className="group relative rounded-xl border border-cyan-800/40 bg-cyan-950/10 hover:bg-cyan-900/15 transition-colors cursor-pointer overflow-hidden"
      onClick={onOpenLightbox}
    >
      <div className="aspect-[4/3] w-full overflow-hidden rounded-t-xl bg-neutral-800">
        {thumb ? (
          <img src={thumb} alt="" className="h-full w-full object-cover" loading="lazy" decoding="async" />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <svg className="h-8 w-8 text-neutral-700 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" />
            </svg>
          </div>
        )}
      </div>
      <div className="px-2 py-1.5 flex items-center justify-between gap-1">
        <p className="truncate text-[11px] text-neutral-400 flex-1">
          {link.file_path.split(/[\\/]/).pop() ?? link.file_path}
        </p>
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); onUnlink() }}
          className="shrink-0 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-neutral-400 opacity-0 group-hover:opacity-100 hover:text-red-400 transition-opacity"
        >
          Unlink
        </button>
      </div>
    </div>
  )
})
