import { useState, useEffect, useCallback, useRef, memo, useMemo } from 'react'
import type {
  FaceCluster,
  FaceOccurrence,
  FaceSummary,
  FaceImage,
  FacePerson,
  PersonCluster,
  FaceLinkSuggestion,
  PersonLinkSuggestion,
  PersonSimilarImage,
  PersonDirectLink,
  SearchResult,
} from '../global'
import { AnalysisSidebar } from './SearchLightbox'

// ── Thumbnail cache & batch fetcher ───────────────────────────────────────────

const THUMB_CACHE_MAX = 2000
const thumbCache = new Map<number, string>() // occurrence_id → base64 data URI
const pendingIds = new Set<number>()
const pendingCallbacks = new Map<number, Array<(src: string | null) => void>>()
let batchTimer: ReturnType<typeof setTimeout> | null = null

const CLUSTER_PAGE_SIZE = 200
const DEFAULT_UNLINKED_CLUSTER_TARGET = 100
const DEFAULT_SIMILARITY_THRESHOLD = 0.35

function normalizeImageSrc(value: string | null | undefined): string | null {
  if (!value) return null
  const trimmed = value.trim()
  if (!trimmed) return null
  if (
    trimmed.startsWith('data:image/')
    || trimmed.startsWith('blob:')
    || trimmed.startsWith('http://')
    || trimmed.startsWith('https://')
    || trimmed.startsWith('file://')
  ) {
    return trimmed
  }
  return `data:image/jpeg;base64,${trimmed}`
}

function countActiveUnlinkedClusters(
  clusters: FaceCluster[],
  deferredClusterIds: Set<number>,
): number {
  return clusters.reduce(
    (count, cluster) => (
      cluster.cluster_id !== null
      && !cluster.person_id
      && !deferredClusterIds.has(cluster.cluster_id)
        ? count + 1
        : count
    ),
    0,
  )
}

function clusterKey(cluster: FaceCluster): string {
  return cluster.cluster_id !== null
    ? `cluster:${cluster.cluster_id}`
    : `name:${cluster.identity_name}`
}

function appendUniqueClusters(
  existing: FaceCluster[],
  incoming: FaceCluster[],
): FaceCluster[] {
  const seen = new Set(existing.map(clusterKey))
  const appended = incoming.filter((cluster) => !seen.has(clusterKey(cluster)))
  return [...existing, ...appended]
}

function requestThumbnail(
  occurrenceId: number,
  callback: (src: string | null) => void
): void {
  // Serve from cache
  const cached = thumbCache.get(occurrenceId)
  if (cached) {
    callback(cached)
    return
  }

  // Register callback
  const cbs = pendingCallbacks.get(occurrenceId) ?? []
  cbs.push(callback)
  pendingCallbacks.set(occurrenceId, cbs)
  pendingIds.add(occurrenceId)

  // Debounce: flush batch after 16ms (one animation frame)
  if (batchTimer === null) {
    batchTimer = setTimeout(flushBatch, 16)
  }
}

/** Pre-populate the thumbnail cache from inline base64 data (e.g. from
 *  faces/persons or faces/person-clusters responses).  Avoids separate
 *  RPC roundtrips for thumbnails the server already returned. */
function primeThumbCache(entries: Array<{ id: number; base64: string }>): void {
  for (const { id, base64 } of entries) {
    if (thumbCache.has(id)) continue
    if (thumbCache.size >= THUMB_CACHE_MAX) {
      const oldest = thumbCache.keys().next().value
      if (oldest !== undefined) thumbCache.delete(oldest)
    }
    thumbCache.set(id, `data:image/jpeg;base64,${base64}`)
  }
}

async function flushBatch(): Promise<void> {
  batchTimer = null
  if (pendingIds.size === 0) return

  const ids = [...pendingIds]
  pendingIds.clear()

  // Process chunks in parallel for faster loading
  const CHUNK = 50
  const chunks: number[][] = []
  for (let i = 0; i < ids.length; i += CHUNK) {
    chunks.push(ids.slice(i, i + CHUNK))
  }

  await Promise.all(chunks.map(async (chunk) => {
    try {
      const result = await window.api.getFaceCropBatch(chunk)
      for (const id of chunk) {
        const b64 = result.thumbnails?.[String(id)]
        const dataUri = b64 ? `data:image/jpeg;base64,${b64}` : null
        if (dataUri) {
          if (thumbCache.size >= THUMB_CACHE_MAX) {
            const oldest = thumbCache.keys().next().value
            if (oldest !== undefined) thumbCache.delete(oldest)
          }
          thumbCache.set(id, dataUri)
        }
        const cbs = pendingCallbacks.get(id) ?? []
        pendingCallbacks.delete(id)
        for (const cb of cbs) cb(dataUri)
      }
    } catch {
      for (const id of chunk) {
        const cbs = pendingCallbacks.get(id) ?? []
        pendingCallbacks.delete(id)
        for (const cb of cbs) cb(null)
      }
    }
  }))
}

const IMAGE_THUMB_CACHE_MAX = 2000
const imageThumbCache = new Map<string, string>()
const pendingImageThumbs = new Map<string, { file_path: string; image_id?: number }>()
const pendingImageCallbacks = new Map<string, Array<(src: string | null) => void>>()
let imageThumbBatchTimer: ReturnType<typeof setTimeout> | null = null

function requestImageThumbnail(
  filePath: string,
  imageId: number | null | undefined,
  callback: (src: string | null) => void,
): void {
  const cached = imageThumbCache.get(filePath)
  if (cached) {
    callback(cached)
    return
  }

  const callbacks = pendingImageCallbacks.get(filePath) ?? []
  callbacks.push(callback)
  pendingImageCallbacks.set(filePath, callbacks)
  pendingImageThumbs.set(filePath, imageId != null
    ? { file_path: filePath, image_id: imageId }
    : { file_path: filePath })

  if (imageThumbBatchTimer !== null) return
  imageThumbBatchTimer = setTimeout(() => {
    void flushImageThumbnailBatch()
  }, 16)
}

async function flushImageThumbnailBatch(): Promise<void> {
  imageThumbBatchTimer = null
  if (pendingImageThumbs.size === 0) return

  const items = [...pendingImageThumbs.values()]
  pendingImageThumbs.clear()

  const CHUNK = 24
  const chunks: Array<typeof items> = []
  for (let i = 0; i < items.length; i += CHUNK) {
    chunks.push(items.slice(i, i + CHUNK))
  }

  await Promise.all(chunks.map(async (chunk) => {
    try {
      const result = await window.api.getThumbnailsBatch(chunk)
      for (const item of chunk) {
        const src = normalizeImageSrc(
          result[item.file_path] ?? (item.image_id != null ? result[String(item.image_id)] : undefined)
        )
        if (src) {
          if (imageThumbCache.size >= IMAGE_THUMB_CACHE_MAX) {
            const oldest = imageThumbCache.keys().next().value
            if (oldest !== undefined) imageThumbCache.delete(oldest)
          }
          imageThumbCache.set(item.file_path, src)
        }
        const callbacks = pendingImageCallbacks.get(item.file_path) ?? []
        pendingImageCallbacks.delete(item.file_path)
        for (const cb of callbacks) cb(src)
      }
    } catch {
      for (const item of chunk) {
        const callbacks = pendingImageCallbacks.get(item.file_path) ?? []
        pendingImageCallbacks.delete(item.file_path)
        for (const cb of callbacks) cb(null)
      }
    }
  }))
}

// ── Face crop thumbnail (batch-loaded with LRU cache) ─────────────────────────

const FaceCropThumbnail = memo(function FaceCropThumbnail({
  occurrenceId,
  size = 'md',
}: {
  occurrenceId: number
  size?: 'sm' | 'md' | 'lg' | 'fill'
}) {
  const [src, setSrc] = useState<string | null>(() => thumbCache.get(occurrenceId) ?? null)
  const [failed, setFailed] = useState(false)
  const requested = useRef(false)
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const cached = thumbCache.get(occurrenceId) ?? null
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
      requestThumbnail(occurrenceId, (dataUri) => {
        if (cancelled) return
        if (dataUri) {
          setSrc(dataUri)
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
        draggable={false}
      />
    </div>
  )
})

// ── Full image thumbnail (lazy-loaded, for legacy mode) ───────────────────────

function ImageThumbnail({ filePath, imageId }: { filePath: string; imageId?: number | null }) {
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
      draggable={false}
    />
  )
}

// ── Similar image card with lazy thumbnail loading ────────────────────────────

const SimilarImageCard = memo(function SimilarImageCard({
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
    requestThumbnail(image.best_occurrence_id, (data) => {
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
          <img src={faceCrop} alt="" className="h-full w-full object-cover" />
        </div>
      )}
      {faceCrop && onToggleSelect && (
        <div className="absolute right-10 top-2 z-10 h-6 w-6 rounded-full border border-amber-600/70 overflow-hidden bg-neutral-800">
          <img src={faceCrop} alt="" className="h-full w-full object-cover" />
        </div>
      )}
      <div className="aspect-[4/3] w-full overflow-hidden rounded-t-xl bg-neutral-800">
        {thumb ? (
          <img src={thumb} alt="" className="h-full w-full object-cover" />
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

const DirectLinkCard = memo(function DirectLinkCard({
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
          <img src={thumb} alt="" className="h-full w-full object-cover" />
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

// ── Inline image lightbox (for viewing source image in-app) ───────────────────

function FaceImageLightbox({
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

type RelinkSelection =
  | { type: 'person'; personId: number; label: string }
  | { type: 'alias'; label: string }
  | null

type AliasCandidate = {
  key: string
  label: string
  subtitle: string
  imageCount: number
  representativeId: number | null
}

function coerceText(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

type PeopleStage = 'linked' | 'suggested' | 'unlinked' | 'inspector'
type FacesDeepLinkRequest = {
  clusterId: number
  sourceImageId: number
  requestId: number
}

type FacesViewProps = {
  deepLinkRequest?: FacesDeepLinkRequest | null
  onDeepLinkHandled?: (requestId: number) => void
}

// ── Main FacesView component ──────────────────────────────────────────────────

export function FacesView({ deepLinkRequest = null, onDeepLinkHandled }: FacesViewProps) {
  // Cluster mode (Phase 2 — face_occurrences exist)
  const [clusters, setClusters] = useState<FaceCluster[]>([])
  const [hasOccurrences, setHasOccurrences] = useState(false)
  const [totalClusterCount, setTotalClusterCount] = useState(0)

  // Legacy mode (Phase 1 — no face_occurrences, fallback to identity names)
  const [legacyFaces, setLegacyFaces] = useState<FaceSummary[]>([])

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Expanded cluster/face
  const [expandedKey, setExpandedKey] = useState<string | null>(null)

  // Cached occurrences per cluster (keyed by expandedKey)
  const [clusterOccurrences, setClusterOccurrences] = useState<
    Record<string, FaceOccurrence[]>
  >({})
  // Cached legacy images per face
  const [legacyImages, setLegacyImages] = useState<
    Record<string, FaceImage[]>
  >({})
  const [loadingDetail, setLoadingDetail] = useState<string | null>(null)

  // Inline editing state
  const [editingKey, setEditingKey] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const editInputRef = useRef<HTMLInputElement>(null)

  // Clustering state
  const [clustering, setClustering] = useState(false)

  // Lightbox state (in-app image viewer)
  const [lightboxPath, setLightboxPath] = useState<string | null>(null)
  const [lightboxImageId, setLightboxImageId] = useState<number | undefined>(undefined)

  // View mode: clusters or people
  type ViewMode = 'clusters' | 'people'
  const [viewMode, setViewMode] = useState<ViewMode>('clusters')

  // Person state
  const [persons, setPersons] = useState<FacePerson[]>([])
  const [personClusters, setPersonClusters] = useState<Record<number, PersonCluster[]>>({})
  const [personClusterErrors, setPersonClusterErrors] = useState<Record<number, string>>({})
  const [personLinkSuggestions, setPersonLinkSuggestions] = useState<Record<number, PersonLinkSuggestion[]>>({})
  const [personLinkSuggestionErrors, setPersonLinkSuggestionErrors] = useState<Record<number, string>>({})
  const [loadingPersonLinkSuggestionsId, setLoadingPersonLinkSuggestionsId] = useState<number | null>(null)
  const [selectedSuggestedClusterIds, setSelectedSuggestedClusterIds] = useState<number[]>([])
  const [confirmingSuggestedLinks, setConfirmingSuggestedLinks] = useState(false)
  const [expandedPersonId, setExpandedPersonId] = useState<number | null>(null)
  const [isPeopleChooserExpanded, setIsPeopleChooserExpanded] = useState(false)
  const [peopleStage, setPeopleStage] = useState<PeopleStage>('linked')
  const [inspectorReturnStage, setInspectorReturnStage] = useState<'linked' | 'suggested' | 'unlinked'>('linked')
  const [inspectorCluster, setInspectorCluster] = useState<FaceCluster | null>(null)

  // Person editing
  const [editingPersonId, setEditingPersonId] = useState<number | null>(null)
  const [personEditValue, setPersonEditValue] = useState('')
  const personEditRef = useRef<HTMLInputElement>(null)

  // Link-to-person dropdown
  const [linkingClusterId, setLinkingClusterId] = useState<number | null>(null)
  const [showCreatePerson, setShowCreatePerson] = useState(false)
  const [newPersonName, setNewPersonName] = useState('')
  const [linkSearchFilter, setLinkSearchFilter] = useState('')
  const [peopleChooserFilter, setPeopleChooserFilter] = useState('')
  const newPersonRef = useRef<HTMLInputElement>(null)
  const linkSearchRef = useRef<HTMLInputElement>(null)

  // Delete person confirmation
  const [deletingPersonId, setDeletingPersonId] = useState<number | null>(null)

  // Cluster relink dialog
  const [relinkingCluster, setRelinkingCluster] = useState<FaceCluster | null>(null)
  const [relinkSelection, setRelinkSelection] = useState<RelinkSelection>(null)
  const [relinkSearch, setRelinkSearch] = useState('')
  const [relinkLoading, setRelinkLoading] = useState(false)
  const [relinkSubmitting, setRelinkSubmitting] = useState(false)
  const [relinkPersons, setRelinkPersons] = useState<FacePerson[]>([])
  const [relinkFaceTargets, setRelinkFaceTargets] = useState<FaceSummary[]>([])
  const [relinkClusterTargets, setRelinkClusterTargets] = useState<FaceCluster[]>([])
  const [relinkSuggestions, setRelinkSuggestions] = useState<FaceLinkSuggestion[]>([])
  const [unlinkPersonOnAliasRelink, setUnlinkPersonOnAliasRelink] = useState(false)
  const relinkSearchRef = useRef<HTMLInputElement>(null)

  // Cluster defer (park for later)
  const [deferredClusterIds, setDeferredClusterIds] = useState<Set<number>>(new Set())
  const [unlinkedSubFilter, setUnlinkedSubFilter] = useState<'active' | 'deferred'>('active')
  const [suggestedSubFilter, setSuggestedSubFilter] = useState<'active' | 'deferred'>('active')

  // Inspector similarity suggestions
  const [inspectorSuggestions, setInspectorSuggestions] = useState<FaceLinkSuggestion[]>([])
  const [inspectorSuggestionsLoading, setInspectorSuggestionsLoading] = useState(false)

  // Similar images for person (Suggested → Images sub-tab)
  const [personSimilarImages, setPersonSimilarImages] = useState<Record<number, PersonSimilarImage[]>>({})
  const [loadingPersonSimilarImagesId, setLoadingPersonSimilarImagesId] = useState<number | null>(null)
  const [suggestedInnerTab, setSuggestedInnerTab] = useState<'clusters' | 'images'>('clusters')
  const [selectedSimilarImageIds, setSelectedSimilarImageIds] = useState<number[]>([]) // best_occurrence_id[]
  const [confirmingSimilarLinks, setConfirmingSimilarLinks] = useState(false)
  const [similarityThreshold, setSimilarityThreshold] = useState<number>(DEFAULT_SIMILARITY_THRESHOLD)

  // Direct links (Linked tab — images linked to person without cluster)
  const [personDirectLinks, setPersonDirectLinks] = useState<Record<number, PersonDirectLink[]>>({})
  const [loadingDirectLinksId, setLoadingDirectLinksId] = useState<number | null>(null)
  const [unlinkedClusterTarget, setUnlinkedClusterTarget] = useState<number>(DEFAULT_UNLINKED_CLUSTER_TARGET)
  const handledDeepLinkRequestIdRef = useRef(0)

  // ── Load data ─────────────────────────────────────────────────────────────

  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      // Load first page of clusters + persons in parallel
      const [clusterResult, personsResult] = await Promise.all([
        window.api.listFaceClusters(CLUSTER_PAGE_SIZE, 0),
        window.api.listPersons(),
      ])

      if (clusterResult.error) {
        setError(clusterResult.error)
        return
      }

      if (clusterResult.has_occurrences && clusterResult.clusters.length > 0) {
        let loadedClusters = clusterResult.clusters
        let loadedTotalCount = clusterResult.total_count
        const deferredIds = new Set(clusterResult.deferred_cluster_ids ?? [])
        setDeferredClusterIds(deferredIds)
        const shouldEnsureUnlinked = !personsResult.error && personsResult.persons.length > 0

        if (shouldEnsureUnlinked) {
          let offset = loadedClusters.length
          let unlinkedCount = countActiveUnlinkedClusters(loadedClusters, deferredIds)

          while (unlinkedCount < unlinkedClusterTarget && offset < loadedTotalCount) {
            const nextPage = await window.api.listFaceClusters(CLUSTER_PAGE_SIZE, offset)
            if (nextPage.error) {
              setError(nextPage.error)
              break
            }
            if (nextPage.clusters.length === 0) {
              break
            }
            loadedClusters = [...loadedClusters, ...nextPage.clusters]
            loadedTotalCount = nextPage.total_count
            offset = loadedClusters.length
            unlinkedCount = countActiveUnlinkedClusters(loadedClusters, deferredIds)
          }
        }

        setTotalClusterCount(loadedTotalCount)
        setHasOccurrences(true)
        setClusters(loadedClusters)
        setLegacyFaces([])
      } else {
        setTotalClusterCount(clusterResult.total_count)
        // Fallback to legacy identity-name mode
        setHasOccurrences(clusterResult.has_occurrences)
        const legacyResult = await window.api.listFaces()
        if (legacyResult.error) {
          setError(legacyResult.error)
        } else {
          setLegacyFaces(legacyResult.faces)
          setClusters([])
        }
      }

      // Load persons
      if (!personsResult.error) {
        setPersons(personsResult.persons)
        // Pre-populate thumbnail cache from inline thumbnails
        const thumbEntries = personsResult.persons
          .filter((p) => p.representative_id != null && p.representative_thumbnail != null)
          .map((p) => ({ id: p.representative_id!, base64: p.representative_thumbnail! }))
        if (thumbEntries.length > 0) primeThumbCache(thumbEntries)
        // Auto-switch to people view if persons exist and no view preference yet
        if (personsResult.persons.length > 0) {
          setViewMode('people')
        }
      }
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }, [unlinkedClusterTarget])

  useEffect(() => {
    window.api.getAppSettings()
      .then((bundle) => {
        const configured = Number(bundle.settings.processing?.unlinkedClusterTarget)
        if (Number.isInteger(configured) && configured >= 1 && configured <= 1000) {
          setUnlinkedClusterTarget(configured)
        } else {
          setUnlinkedClusterTarget(DEFAULT_UNLINKED_CLUSTER_TARGET)
        }
      })
      .catch(() => {
        setUnlinkedClusterTarget(DEFAULT_UNLINKED_CLUSTER_TARGET)
      })
  }, [])

  const loadMoreClusters = useCallback(async () => {
    try {
      const result = await window.api.listFaceClusters(CLUSTER_PAGE_SIZE, clusters.length)
      if (result.error) {
        setError(result.error)
        return
      }
      setClusters((prev) => [...prev, ...result.clusters])
      setTotalClusterCount(result.total_count)
    } catch (err) {
      setError(String(err))
    }
  }, [clusters.length])

  const hasMoreClusters = clusters.length < totalClusterCount

  const ensureClusterAvailable = useCallback(async (clusterId: number): Promise<FaceCluster | null> => {
    const existing = clusters.find((cluster) => cluster.cluster_id === clusterId)
    if (existing) {
      return existing
    }

    let offset = clusters.length
    let total = totalClusterCount
    const fetched: FaceCluster[] = []

    while (offset < total) {
      const result = await window.api.listFaceClusters(CLUSTER_PAGE_SIZE, offset)
      if (result.error) {
        setError(result.error)
        return null
      }
      if (result.clusters.length === 0) {
        break
      }
      fetched.push(...result.clusters)
      total = result.total_count
      offset += result.clusters.length
      if (result.deferred_cluster_ids) {
        setDeferredClusterIds(new Set(result.deferred_cluster_ids))
      }
      const found = result.clusters.find((cluster) => cluster.cluster_id === clusterId)
      if (found) {
        setClusters((prev) => appendUniqueClusters(prev, fetched))
        setTotalClusterCount(total)
        return found
      }
    }

    if (fetched.length > 0) {
      setClusters((prev) => appendUniqueClusters(prev, fetched))
      setTotalClusterCount(total)
    }
    return null
  }, [clusters, totalClusterCount])

  useEffect(() => {
    loadData()
  }, [loadData])

  useEffect(() => {
    setSelectedSuggestedClusterIds([])
    setExpandedKey(null)
    setInspectorCluster(null)
    setPeopleStage('linked')
    setInspectorReturnStage('linked')
  }, [expandedPersonId])

  // ── Clustering ────────────────────────────────────────────────────────────

  // Listen for clustering-done notification
  useEffect(() => {
    const unsub = window.api.onClusteringDone((result) => {
      if (result.error) {
        setError(result.error)
      } else {
        void loadData()
      }
      setClustering(false)
    })
    return unsub
  }, [loadData])

  const handleCluster = useCallback(async () => {
    setClustering(true)
    try {
      const result = await window.api.runFaceClustering()
      if (result.error) {
        setError(result.error)
        setClustering(false)
      }
      // Clustering is now async — completion handled by onClusteringDone listener
    } catch (err) {
      setError(String(err))
      setClustering(false)
    }
  }, [])

  // ── Person actions ──────────────────────────────────────────────────────

  const removeSuggestedClusterEverywhere = useCallback((clusterId: number) => {
    setSelectedSuggestedClusterIds((prev) => prev.filter((id) => id !== clusterId))
    setPersonLinkSuggestions((prev) => {
      const next: Record<number, PersonLinkSuggestion[]> = {}
      for (const [personId, suggestions] of Object.entries(prev)) {
        next[Number(personId)] = suggestions.filter(
          (suggestion) => suggestion.cluster_id !== clusterId
        )
      }
      return next
    })
  }, [])

  const handleLinkCluster = useCallback(
    async (clusterId: number, personId: number) => {
      try {
        await window.api.linkClusterToPerson(clusterId, personId)
        removeSuggestedClusterEverywhere(clusterId)
        setLinkingClusterId(null)
        await loadData()
      } catch (err) {
        setError(String(err))
      }
    },
    [loadData, removeSuggestedClusterEverywhere]
  )

  const handleUnlinkCluster = useCallback(
    async (clusterId: number) => {
      try {
        await window.api.unlinkClusterFromPerson(clusterId)
        await loadData()
      } catch (err) {
        setError(String(err))
      }
    },
    [loadData]
  )

  const handleCreatePersonAndLink = useCallback(
    async (clusterId: number) => {
      const name = newPersonName.trim()
      if (!name) return
      try {
        const result = await window.api.createPerson(name)
        if (result.error) {
          setError(result.error)
          return
        }
        await window.api.linkClusterToPerson(clusterId, result.id)
        removeSuggestedClusterEverywhere(clusterId)
        setShowCreatePerson(false)
        setNewPersonName('')
        setLinkingClusterId(null)
        await loadData()
      } catch (err) {
        setError(String(err))
      }
    },
    [newPersonName, loadData, removeSuggestedClusterEverywhere]
  )

  const handleRenamePerson = useCallback(
    async (personId: number) => {
      const name = personEditValue.trim()
      if (!name) return
      try {
        await window.api.renamePerson(personId, name)
        setPersons((prev) =>
          prev.map((p) => (p.id === personId ? { ...p, name } : p))
        )
      } catch (err) {
        setError(String(err))
      }
      setEditingPersonId(null)
      setPersonEditValue('')
    },
    [personEditValue]
  )

  const handleDeletePerson = useCallback(
    async (personId: number) => {
      try {
        await window.api.deletePerson(personId)
        setDeletingPersonId(null)
        await loadData()
      } catch (err) {
        setError(String(err))
      }
    },
    [loadData]
  )

  const closeRelinkDialog = useCallback(() => {
    setRelinkingCluster(null)
    setRelinkSelection(null)
    setRelinkSearch('')
    setRelinkLoading(false)
    setRelinkSubmitting(false)
    setRelinkPersons([])
    setRelinkFaceTargets([])
    setRelinkClusterTargets([])
    setRelinkSuggestions([])
    setUnlinkPersonOnAliasRelink(false)
  }, [])

  const openRelinkDialog = useCallback(async (cluster: FaceCluster) => {
    if (cluster.cluster_id == null) {
      return
    }

    setRelinkingCluster(cluster)
    setRelinkSelection(null)
    setRelinkSearch('')
    setRelinkLoading(true)
    setRelinkSubmitting(false)
    setUnlinkPersonOnAliasRelink(false)

    try {
      const [personsResult, faceResult, clusterResult, suggestionResult] = await Promise.all([
        window.api.listPersons(),
        window.api.listFaces(),
        window.api.listFaceClusters(),
        window.api.getClusterLinkSuggestions(cluster.cluster_id, 12),
      ])
      if (personsResult.error) {
        setError(personsResult.error)
        closeRelinkDialog()
        return
      }
      if (faceResult.error) {
        setError(faceResult.error)
        closeRelinkDialog()
        return
      }
      if (clusterResult.error) {
        setError(clusterResult.error)
        closeRelinkDialog()
        return
      }

      setRelinkPersons(personsResult.persons)
      setRelinkFaceTargets(faceResult.faces)
      setRelinkClusterTargets(clusterResult.clusters)
      if (suggestionResult.error) {
        setError(suggestionResult.error)
        setRelinkSuggestions([])
      } else {
        setRelinkSuggestions(suggestionResult.suggestions)
      }
    } catch (err) {
      setError(String(err))
      closeRelinkDialog()
    } finally {
      setRelinkLoading(false)
      setTimeout(() => relinkSearchRef.current?.focus(), 0)
    }
  }, [closeRelinkDialog])

  const applyClusterRelink = useCallback(
    async (clusterId: number, displayName: string | null, personId: number | null, updatePerson: boolean) => {
      const result = await window.api.relinkFaceCluster(clusterId, displayName, personId, updatePerson)
      if (result.error || !result.ok) {
        throw new Error(result.error ?? 'Failed to relink cluster')
      }
    },
    []
  )

  const handleApplyRelinkSelection = useCallback(async () => {
    if (relinkingCluster?.cluster_id == null || relinkSelection == null) {
      return
    }

    setRelinkSubmitting(true)
    try {
      if (relinkSelection.type === 'person') {
        await applyClusterRelink(
          relinkingCluster.cluster_id,
          relinkSelection.label,
          relinkSelection.personId,
          true,
        )
        removeSuggestedClusterEverywhere(relinkingCluster.cluster_id)
      } else {
        await applyClusterRelink(
          relinkingCluster.cluster_id,
          relinkSelection.label,
          null,
          unlinkPersonOnAliasRelink,
        )
      }
      setError(null)
      closeRelinkDialog()
      await loadData()
    } catch (err) {
      setError(String(err))
    } finally {
      setRelinkSubmitting(false)
    }
  }, [applyClusterRelink, closeRelinkDialog, loadData, relinkSelection, relinkingCluster, removeSuggestedClusterEverywhere, unlinkPersonOnAliasRelink])

  const handleCreateRelinkPerson = useCallback(async () => {
    const clusterId = relinkingCluster?.cluster_id
    const name = relinkSearch.trim()
    if (clusterId == null || !name) {
      return
    }

    setRelinkSubmitting(true)
    try {
      const createResult = await window.api.createPerson(name)
      if (createResult.error) {
        throw new Error(createResult.error)
      }
      await applyClusterRelink(clusterId, name, createResult.id, true)
      removeSuggestedClusterEverywhere(clusterId)
      setError(null)
      closeRelinkDialog()
      await loadData()
    } catch (err) {
      setError(String(err))
    } finally {
      setRelinkSubmitting(false)
    }
  }, [applyClusterRelink, closeRelinkDialog, loadData, relinkSearch, relinkingCluster, removeSuggestedClusterEverywhere])

  const handleCreateRelinkAlias = useCallback(async () => {
    const clusterId = relinkingCluster?.cluster_id
    const name = relinkSearch.trim()
    if (clusterId == null || !name) {
      return
    }

    setRelinkSubmitting(true)
    try {
      await applyClusterRelink(clusterId, name, null, unlinkPersonOnAliasRelink)
      setError(null)
      closeRelinkDialog()
      await loadData()
    } catch (err) {
      setError(String(err))
    } finally {
      setRelinkSubmitting(false)
    }
  }, [applyClusterRelink, closeRelinkDialog, loadData, relinkSearch, relinkingCluster, unlinkPersonOnAliasRelink])

  const handleClearRelinkAlias = useCallback(async () => {
    const clusterId = relinkingCluster?.cluster_id
    if (clusterId == null) {
      return
    }

    setRelinkSubmitting(true)
    try {
      await applyClusterRelink(clusterId, null, null, false)
      setError(null)
      closeRelinkDialog()
      await loadData()
    } catch (err) {
      setError(String(err))
    } finally {
      setRelinkSubmitting(false)
    }
  }, [applyClusterRelink, closeRelinkDialog, loadData, relinkingCluster])

  const handleRelinkUnlinkPerson = useCallback(async () => {
    const clusterId = relinkingCluster?.cluster_id
    if (clusterId == null) {
      return
    }

    setRelinkSubmitting(true)
    try {
      await applyClusterRelink(clusterId, relinkingCluster.display_name, null, true)
      setError(null)
      closeRelinkDialog()
      await loadData()
    } catch (err) {
      setError(String(err))
    } finally {
      setRelinkSubmitting(false)
    }
  }, [applyClusterRelink, closeRelinkDialog, loadData, relinkingCluster])

  const loadPersonLinkSuggestions = useCallback(
    async (personId: number, force = false) => {
      if (!force && personLinkSuggestions[personId] !== undefined) {
        return
      }

      const startedAt = performance.now()
      setLoadingPersonLinkSuggestionsId(personId)
      setPersonLinkSuggestionErrors((prev) => {
        if (prev[personId] === undefined) {
          return prev
        }
        const next = { ...prev }
        delete next[personId]
        return next
      })
      try {
        const result = await window.api.getPersonLinkSuggestions(personId, 12)
        if (result.error) {
          setPersonLinkSuggestionErrors((prev) => ({ ...prev, [personId]: result.error }))
          setPersonLinkSuggestions((prev) => ({ ...prev, [personId]: [] }))
          return
        }
        setPersonLinkSuggestions((prev) => ({ ...prev, [personId]: result.suggestions }))
      } catch (err) {
        setPersonLinkSuggestionErrors((prev) => ({ ...prev, [personId]: String(err) }))
        setPersonLinkSuggestions((prev) => ({ ...prev, [personId]: [] }))
      } finally {
        if (import.meta.env.DEV) {
          const elapsed = Math.round(performance.now() - startedAt)
          console.debug(`[FacesView] suggestions(${personId}) loaded in ${elapsed}ms`)
        }
        setLoadingPersonLinkSuggestionsId((current) => (current === personId ? null : current))
      }
    },
    [personLinkSuggestions]
  )

  const loadPersonSimilarImages = useCallback(
    async (personId: number, force = false) => {
      if (!force && personSimilarImages[personId] !== undefined) {
        return
      }

      const startedAt = performance.now()
      setLoadingPersonSimilarImagesId(personId)
      try {
        const result = await window.api.getPersonSimilarImages(personId, 100, similarityThreshold)
        if (result.error) {
          setPersonSimilarImages((prev) => ({ ...prev, [personId]: [] }))
          return
        }
        setPersonSimilarImages((prev) => ({ ...prev, [personId]: result.images }))
      } catch {
        setPersonSimilarImages((prev) => ({ ...prev, [personId]: [] }))
      } finally {
        if (import.meta.env.DEV) {
          const elapsed = Math.round(performance.now() - startedAt)
          console.debug(`[FacesView] similarImages(${personId}) loaded in ${elapsed}ms`)
        }
        setLoadingPersonSimilarImagesId((current) => (current === personId ? null : current))
      }
    },
    [personSimilarImages, similarityThreshold]
  )

  const loadPersonDirectLinks = useCallback(
    async (personId: number, force = false) => {
      if (!force && personDirectLinks[personId] !== undefined) {
        return
      }
      setLoadingDirectLinksId(personId)
      try {
        const result = await window.api.getPersonDirectLinks(personId)
        if (result.error) {
          setPersonDirectLinks((prev) => ({ ...prev, [personId]: [] }))
          return
        }
        setPersonDirectLinks((prev) => ({ ...prev, [personId]: result.links }))
      } catch {
        setPersonDirectLinks((prev) => ({ ...prev, [personId]: [] }))
      } finally {
        setLoadingDirectLinksId((current) => (current === personId ? null : current))
      }
    },
    [personDirectLinks]
  )

  const loadPersonClusters = useCallback(
    async (personId: number, force = false) => {
      if (!force && personClusters[personId] !== undefined) {
        return
      }

      const startedAt = performance.now()
      const loadingKey = `person-clusters:${personId}`
      setLoadingDetail(loadingKey)
      setPersonClusterErrors((prev) => {
        if (prev[personId] === undefined) {
          return prev
        }
        const next = { ...prev }
        delete next[personId]
        return next
      })
      try {
        const result = await window.api.getPersonClusters(personId)
        if (result.error) {
          setPersonClusterErrors((prev) => ({ ...prev, [personId]: result.error }))
          return
        }
        // Pre-populate thumbnail cache from inline thumbnails
        const thumbEntries = result.clusters
          .filter((c) => c.representative_id != null && c.representative_thumbnail != null)
          .map((c) => ({ id: c.representative_id!, base64: c.representative_thumbnail! }))
        if (thumbEntries.length > 0) primeThumbCache(thumbEntries)
        setPersonClusters((prev) => ({ ...prev, [personId]: result.clusters }))
      } catch (err) {
        setPersonClusterErrors((prev) => ({ ...prev, [personId]: String(err) }))
      } finally {
        if (import.meta.env.DEV) {
          const elapsed = Math.round(performance.now() - startedAt)
          console.debug(`[FacesView] personClusters(${personId}) loaded in ${elapsed}ms`)
        }
        setLoadingDetail((current) => (current === loadingKey ? null : current))
      }
    },
    [personClusters]
  )

  const handleConfirmSimilarLinks = useCallback(async () => {
    if (expandedPersonId == null || selectedSimilarImageIds.length === 0) return

    setConfirmingSimilarLinks(true)
    try {
      const result = await window.api.linkOccurrencesToPerson(expandedPersonId, selectedSimilarImageIds)
      if (result.error || !result.ok) {
        throw new Error(result.error ?? 'Failed to link occurrences')
      }
      // Remove linked images from the similar images list
      const linkedSet = new Set(selectedSimilarImageIds)
      setPersonSimilarImages((prev) => ({
        ...prev,
        [expandedPersonId]: (prev[expandedPersonId] ?? []).filter(
          (img) => !linkedSet.has(img.best_occurrence_id)
        ),
      }))
      setSelectedSimilarImageIds([])
      // Refresh linked views for this person
      await Promise.all([
        loadPersonDirectLinks(expandedPersonId, true),
        loadPersonClusters(expandedPersonId, true),
      ])
      setError(null)
    } catch (err) {
      setError(String(err))
    } finally {
      setConfirmingSimilarLinks(false)
    }
  }, [expandedPersonId, selectedSimilarImageIds, loadPersonDirectLinks, loadPersonClusters])

  const handleUnlinkDirectOccurrence = useCallback(async (occurrenceId: number) => {
    if (expandedPersonId == null) return
    try {
      const result = await window.api.unlinkOccurrenceFromPerson(occurrenceId)
      if (result.error || !result.ok) {
        throw new Error(result.error ?? 'Failed to unlink occurrence')
      }
      setPersonDirectLinks((prev) => ({
        ...prev,
        [expandedPersonId]: (prev[expandedPersonId] ?? []).filter(
          (link) => link.occurrence_id !== occurrenceId
        ),
      }))
      setError(null)
    } catch (err) {
      setError(String(err))
    }
  }, [expandedPersonId])

  const togglePersonExpand = useCallback(
    async (personId: number) => {
      if (expandedPersonId === personId) {
        setExpandedPersonId(null)
        setExpandedKey(null)
        setInspectorCluster(null)
        return
      }
      setExpandedPersonId(personId)
      setIsPeopleChooserExpanded(false)
      setExpandedKey(null)
      setInspectorCluster(null)
      setPeopleStage('linked')
      setInspectorReturnStage('linked')

      // Always refresh linked views when opening a person so newly linked
      // images/clusters are visible immediately after reopen.
      await Promise.all([
        loadPersonClusters(personId, true),
        loadPersonDirectLinks(personId, true),
      ])

    },
    [expandedPersonId, loadPersonClusters, loadPersonDirectLinks]
  )

  useEffect(() => {
    // Load suggestions in the background as soon as a person is expanded,
    // not just when the "Suggested" tab is active.  This lets the tab badge
    // show an accurate count before the user clicks it.
    if (
      expandedPersonId == null
      || personLinkSuggestions[expandedPersonId] !== undefined
      || loadingPersonLinkSuggestionsId === expandedPersonId
    ) {
      return
    }
    void loadPersonLinkSuggestions(expandedPersonId)
  }, [
    expandedPersonId,
    loadPersonLinkSuggestions,
    loadingPersonLinkSuggestionsId,
    personLinkSuggestions,
  ])

  // Load similar images when the user switches to the Images sub-tab
  useEffect(() => {
    if (
      expandedPersonId == null
      || peopleStage !== 'suggested'
      || suggestedInnerTab !== 'images'
      || personSimilarImages[expandedPersonId] !== undefined
      || loadingPersonSimilarImagesId === expandedPersonId
    ) {
      return
    }
    void loadPersonSimilarImages(expandedPersonId)
  }, [
    expandedPersonId,
    peopleStage,
    suggestedInnerTab,
    loadPersonSimilarImages,
    loadingPersonSimilarImagesId,
    personSimilarImages,
  ])

  useEffect(() => {
    if (
      expandedPersonId == null
      || peopleStage !== 'suggested'
      || suggestedInnerTab !== 'images'
    ) {
      return
    }
    setPersonSimilarImages((prev) => {
      if (prev[expandedPersonId] === undefined) return prev
      const next = { ...prev }
      delete next[expandedPersonId]
      return next
    })
    setSelectedSimilarImageIds([])
  }, [
    expandedPersonId,
    peopleStage,
    suggestedInnerTab,
    similarityThreshold,
  ])

  // Load direct links when a person is expanded
  useEffect(() => {
    if (
      expandedPersonId == null
      || personDirectLinks[expandedPersonId] !== undefined
      || loadingDirectLinksId === expandedPersonId
    ) {
      return
    }
    void loadPersonDirectLinks(expandedPersonId)
  }, [
    expandedPersonId,
    loadPersonDirectLinks,
    loadingDirectLinksId,
    personDirectLinks,
  ])

  const toggleSuggestedClusterSelection= useCallback((clusterId: number) => {
    setSelectedSuggestedClusterIds((prev) =>
      prev.includes(clusterId)
        ? prev.filter((id) => id !== clusterId)
        : [...prev, clusterId]
    )
  }, [])

  const handleConfirmSuggestedLinks = useCallback(async () => {
    if (expandedPersonId == null) {
      return
    }

    const suggestions = personLinkSuggestions[expandedPersonId] ?? []
    const suggestedIds = new Set(suggestions.map((suggestion) => suggestion.cluster_id))
    const selectedIds = [...new Set(selectedSuggestedClusterIds)].filter((clusterId) =>
      suggestedIds.has(clusterId)
    )
    if (selectedIds.length === 0) {
      return
    }

    setConfirmingSuggestedLinks(true)
    try {
      for (const clusterId of selectedIds) {
        const result = await window.api.linkClusterToPerson(clusterId, expandedPersonId)
        if (result.error || !result.ok) {
          throw new Error(result.error ?? `Failed to link cluster ${clusterId}`)
        }
      }
      setSelectedSuggestedClusterIds([])
      setPersonLinkSuggestions((prev) => ({
        ...prev,
        [expandedPersonId]: (prev[expandedPersonId] ?? []).filter(
          (suggestion) => !selectedIds.includes(suggestion.cluster_id)
        ),
      }))
      setError(null)
      await loadData()

      await loadPersonClusters(expandedPersonId, true)
      await loadPersonLinkSuggestions(expandedPersonId, true)
    } catch (err) {
      setError(String(err))
    } finally {
      setConfirmingSuggestedLinks(false)
    }
  }, [expandedPersonId, loadData, loadPersonClusters, loadPersonLinkSuggestions, personLinkSuggestions, selectedSuggestedClusterIds])

  // ── Expand / collapse ─────────────────────────────────────────────────────

  const ensureExpandedDetailLoaded = useCallback(
    async (key: string, cluster: FaceCluster | null, legacyFace: FaceSummary | null) => {
      if (cluster && !clusterOccurrences[key]) {
        setLoadingDetail(key)
        try {
          const result = await window.api.getFaceClusterImages(
            cluster.cluster_id,
            cluster.cluster_id === null ? cluster.identity_name : null
          )
          if (!result.error) {
            setClusterOccurrences((prev) => ({
              ...prev,
              [key]: result.occurrences,
            }))
          }
        } catch {
          // silently ignore
        } finally {
          setLoadingDetail(null)
        }
        return
      }

      if (legacyFace && !legacyImages[key]) {
        setLoadingDetail(key)
        try {
          const result = await window.api.getFaceImages(legacyFace.canonical_name)
          if (!result.error) {
            setLegacyImages((prev) => ({
              ...prev,
              [key]: result.images,
            }))
          }
        } catch {
          // silently ignore
        } finally {
          setLoadingDetail(null)
        }
      }
    },
    [clusterOccurrences, legacyImages]
  )

  const toggleExpand = useCallback(
    async (key: string, cluster: FaceCluster | null, legacyFace: FaceSummary | null) => {
      if (expandedKey === key) {
        setExpandedKey(null)
        return
      }

      setExpandedKey(key)
      await ensureExpandedDetailLoaded(key, cluster, legacyFace)
    },
    [ensureExpandedDetailLoaded, expandedKey]
  )

  const openClusterInspector = useCallback(
    async (
      key: string,
      cluster: FaceCluster,
      returnStage: 'linked' | 'suggested' | 'unlinked'
    ) => {
      setExpandedKey(key)
      setInspectorCluster(cluster)
      setInspectorReturnStage(returnStage)
      setPeopleStage('inspector')
      // Clear stale suggestions and load fresh ones async
      setInspectorSuggestions([])
      if (cluster.cluster_id != null && !cluster.person_id) {
        setInspectorSuggestionsLoading(true)
        window.api.getClusterLinkSuggestions(cluster.cluster_id, 6).then((result) => {
          if (!result.error) {
            setInspectorSuggestions(result.suggestions.filter((s) => s.score >= 0.45))
          }
        }).catch(() => {}).finally(() => setInspectorSuggestionsLoading(false))
      }
      await ensureExpandedDetailLoaded(key, cluster, null)
    },
    [ensureExpandedDetailLoaded]
  )

  useEffect(() => {
    if (!deepLinkRequest || loading || !hasOccurrences) {
      return
    }
    if (deepLinkRequest.requestId <= handledDeepLinkRequestIdRef.current) {
      return
    }

    let cancelled = false
    void (async () => {
      const targetCluster = await ensureClusterAvailable(deepLinkRequest.clusterId)
      if (cancelled) return

      handledDeepLinkRequestIdRef.current = deepLinkRequest.requestId
      if (!targetCluster) {
        setViewMode('clusters')
        setError(
          `Cluster #${deepLinkRequest.clusterId} (from image ${deepLinkRequest.sourceImageId}) was not found.`
        )
        onDeepLinkHandled?.(deepLinkRequest.requestId)
        return
      }

      const key = clusterKey(targetCluster)
      setViewMode('clusters')
      setExpandedPersonId(null)
      setPeopleStage('linked')
      setInspectorCluster(null)
      setExpandedKey(key)
      await ensureExpandedDetailLoaded(key, targetCluster, null)
      if (cancelled) return
      void openRelinkDialog(targetCluster)
      onDeepLinkHandled?.(deepLinkRequest.requestId)
    })()

    return () => {
      cancelled = true
    }
  }, [
    deepLinkRequest,
    ensureClusterAvailable,
    ensureExpandedDetailLoaded,
    hasOccurrences,
    loading,
    onDeepLinkHandled,
    openRelinkDialog,
  ])

  // ── Inline alias editing ──────────────────────────────────────────────────

  const startEditing = useCallback(
    (key: string, currentDisplayName: string | null, _identityName: string) => {
      setEditingKey(key)
      setEditValue(currentDisplayName ?? '')
      setTimeout(() => editInputRef.current?.focus(), 0)
    },
    []
  )

  const cancelEditing = useCallback(() => {
    setEditingKey(null)
    setEditValue('')
  }, [])

  const saveAlias = useCallback(
    async (identityName: string, clusterId?: number | null) => {
      const trimmed = editValue.trim()
      try {
        await window.api.setFaceAlias(identityName, trimmed, clusterId)
        // Update local state
        if (clusters.length > 0 && clusterId != null) {
          setClusters((prev) =>
            prev.map((c) =>
              c.cluster_id === clusterId
                ? { ...c, display_name: trimmed || null }
                : c
            )
          )
        } else if (clusters.length > 0) {
          setClusters((prev) =>
            prev.map((c) =>
              c.identity_name === identityName
                ? { ...c, display_name: trimmed || null }
                : c
            )
          )
        } else {
          setLegacyFaces((prev) =>
            prev.map((f) =>
              f.canonical_name === identityName
                ? { ...f, display_name: trimmed || null }
                : f
            )
          )
        }
      } catch {
        // silently ignore
      }
      setEditingKey(null)
      setEditValue('')
    },
    [editValue, clusters.length]
  )

  const handleEditKeyDown = useCallback(
    (e: React.KeyboardEvent, identityName: string, clusterId?: number | null) => {
      if (e.key === 'Enter') {
        e.preventDefault()
        saveAlias(identityName, clusterId)
      } else if (e.key === 'Escape') {
        e.preventDefault()
        cancelEditing()
      }
    },
    [saveAlias, cancelEditing]
  )

  // ── Render helpers ────────────────────────────────────────────────────────

  const totalEntries = hasOccurrences ? clusters.length : legacyFaces.length
  const totalFaceCount = useMemo(
    () => hasOccurrences
      ? clusters.reduce((sum, c) => sum + c.face_count, 0)
      : legacyFaces.reduce((sum, f) => sum + f.image_count, 0),
    [hasOccurrences, clusters, legacyFaces],
  )

  const unlinkedClusters = useMemo(
    () => clusters.filter((c) => c.cluster_id !== null && !c.person_id),
    [clusters],
  )
  const activeUnlinkedClusters = useMemo(
    () => unlinkedClusters.filter((c) => c.cluster_id !== null && !deferredClusterIds.has(c.cluster_id!)),
    [unlinkedClusters, deferredClusterIds],
  )
  const deferredUnlinkedClusters = useMemo(
    () => unlinkedClusters.filter((c) => c.cluster_id !== null && deferredClusterIds.has(c.cluster_id!)),
    [unlinkedClusters, deferredClusterIds],
  )
  const visibleUnlinkedClusters = useMemo(
    () => (unlinkedSubFilter === 'deferred'
      ? deferredUnlinkedClusters
      : activeUnlinkedClusters.slice(0, unlinkedClusterTarget)),
    [unlinkedSubFilter, activeUnlinkedClusters, deferredUnlinkedClusters, unlinkedClusterTarget],
  )
  const activePerson = useMemo(
    () => persons.find((person) => person.id === expandedPersonId) ?? null,
    [expandedPersonId, persons],
  )
  const activePersonClustersLoaded =
    expandedPersonId != null && personClusters[expandedPersonId] !== undefined
  const activePersonClusters = useMemo(
    () => (expandedPersonId == null ? [] : (personClusters[expandedPersonId] ?? [])),
    [expandedPersonId, personClusters],
  )
  const activeLinkedCount =
    activePersonClustersLoaded ? activePersonClusters.length : (activePerson?.cluster_count ?? 0)
  const linkedClusterIds = useMemo(() => {
    const ids = new Set<number>()
    for (const cluster of clusters) {
      if (cluster.cluster_id != null && cluster.person_id != null) {
        ids.add(cluster.cluster_id)
      }
    }
    return ids
  }, [clusters])
  const allSuggestedClusters = useMemo(
    () => (
      expandedPersonId == null
        ? []
        : (personLinkSuggestions[expandedPersonId] ?? []).filter(
            (suggestion) =>
              suggestion.score >= 0.6
              && !linkedClusterIds.has(suggestion.cluster_id)
          )
    ),
    [expandedPersonId, linkedClusterIds, personLinkSuggestions],
  )
  const activeSuggestedClusters = useMemo(
    () => allSuggestedClusters.filter((s) => !deferredClusterIds.has(s.cluster_id)),
    [allSuggestedClusters, deferredClusterIds],
  )
  const deferredSuggestedClusters = useMemo(
    () => allSuggestedClusters.filter((s) => deferredClusterIds.has(s.cluster_id)),
    [allSuggestedClusters, deferredClusterIds],
  )
  const visibleSuggestedClusters = suggestedSubFilter === 'deferred'
    ? deferredSuggestedClusters
    : activeSuggestedClusters
  const activePersonLoading =
    expandedPersonId != null && loadingDetail === `person-clusters:${expandedPersonId}`
  const activePersonClusterError =
    expandedPersonId == null ? null : (personClusterErrors[expandedPersonId] ?? null)
  const activeSuggestionsLoading =
    expandedPersonId != null && loadingPersonLinkSuggestionsId === expandedPersonId
  const activeSuggestionsError =
    expandedPersonId == null ? null : (personLinkSuggestionErrors[expandedPersonId] ?? null)
  const selectedSuggestedCount = useMemo(
    () => activeSuggestedClusters.reduce(
      (count, suggestion) =>
        selectedSuggestedClusterIds.includes(suggestion.cluster_id) ? count + 1 : count,
      0,
    ),
    [activeSuggestedClusters, selectedSuggestedClusterIds],
  )
  const allSuggestedSelected =
    activeSuggestedClusters.length > 0
    && selectedSuggestedCount === activeSuggestedClusters.length

  const filteredPersons = useMemo(() => {
    const lowerFilter = linkSearchFilter.toLowerCase()
    return lowerFilter
      ? persons.filter((p) => coerceText(p.name).toLowerCase().includes(lowerFilter))
      : persons
  }, [persons, linkSearchFilter])

  const chooserPersons = useMemo(() => {
    const lf = peopleChooserFilter.toLowerCase()
    return lf
      ? persons.filter((p) => coerceText(p.name).toLowerCase().includes(lf))
      : persons
  }, [persons, peopleChooserFilter])

  const currentRelinkPerson = useMemo(
    () => relinkPersons.find((person) => person.id === relinkingCluster?.person_id) ?? null,
    [relinkPersons, relinkingCluster]
  )

  const filteredRelinkPersons = useMemo(() => {
    const lowerFilter = relinkSearch.trim().toLowerCase()
    return relinkPersons
      .filter((person) => person.id !== relinkingCluster?.person_id)
      .filter((person) =>
        !lowerFilter
        || coerceText(person.name).toLowerCase().includes(lowerFilter)
        || coerceText(person.notes).toLowerCase().includes(lowerFilter)
      )
      .sort((a, b) => b.face_count - a.face_count)
  }, [relinkPersons, relinkSearch, relinkingCluster])

  const relinkAliasCandidates = useMemo(() => {
    const lowerFilter = relinkSearch.trim().toLowerCase()
    const currentClusterId = relinkingCluster?.cluster_id ?? null
    const currentLabel = coerceText(relinkingCluster?.display_name).trim().toLowerCase()
    const byLabel = new Map<string, AliasCandidate>()

    const addCandidate = (
      label: string | null | undefined,
      subtitle: string,
      imageCount: number,
      representativeId: number | null
    ): void => {
      const trimmed = coerceText(label).trim()
      if (!trimmed) {
        return
      }
      const normalized = trimmed.toLowerCase()
      const normalizedSubtitle = coerceText(subtitle).toLowerCase()
      if (currentLabel && normalized === currentLabel) {
        return
      }
      if (lowerFilter && !normalized.includes(lowerFilter) && !normalizedSubtitle.includes(lowerFilter)) {
        return
      }

      const existing = byLabel.get(normalized)
      if (existing) {
        existing.imageCount = Math.max(existing.imageCount, imageCount)
        existing.representativeId = existing.representativeId ?? representativeId
        return
      }

      byLabel.set(normalized, {
        key: `alias:${normalized}`,
        label: trimmed,
        subtitle,
        imageCount,
        representativeId,
      })
    }

    for (const face of relinkFaceTargets) {
      const label = face.display_name ?? face.canonical_name
      const subtitle = face.display_name ? `Identity: ${face.canonical_name}` : 'Identity alias'
      addCandidate(label, subtitle, face.image_count, null)
    }

    for (const cluster of relinkClusterTargets) {
      if (cluster.cluster_id === currentClusterId || !cluster.display_name) {
        continue
      }
      addCandidate(
        cluster.display_name,
        `Cluster label · ${cluster.face_count} faces`,
        cluster.image_count,
        cluster.representative_id,
      )
    }

    return [...byLabel.values()].sort((a, b) =>
      a.label.localeCompare(b.label, undefined, { sensitivity: 'base' })
    )
  }, [relinkClusterTargets, relinkFaceTargets, relinkSearch, relinkingCluster])

  const relinkSearchTrimmed = relinkSearch.trim()

  const relinkPreviewText = useMemo(() => {
    if (!relinkingCluster || !relinkSelection) {
      return null
    }
    const currentLabel = relinkingCluster.display_name || relinkingCluster.identity_name
    if (relinkSelection.type === 'person') {
      return `Cluster ${relinkingCluster.cluster_id} will move from "${currentLabel}" to person "${relinkSelection.label}".`
    }
    if (unlinkPersonOnAliasRelink && currentRelinkPerson) {
      return `Cluster ${relinkingCluster.cluster_id} will be relabeled from "${currentLabel}" to "${relinkSelection.label}" and unlinked from ${currentRelinkPerson.name}.`
    }
    return `Cluster ${relinkingCluster.cluster_id} will be relabeled from "${currentLabel}" to "${relinkSelection.label}".`
  }, [currentRelinkPerson, relinkSelection, relinkingCluster, unlinkPersonOnAliasRelink])

  useEffect(() => {
    if (!relinkingCluster) {
      return
    }

    const handleKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape' && !relinkSubmitting) {
        closeRelinkDialog()
      }
    }

    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [closeRelinkDialog, relinkingCluster, relinkSubmitting])

  // ── Link-to-Person dropdown ────────────────────────────────────────────

  const renderLinkDropdown = (clusterId: number) => {
    if (linkingClusterId !== clusterId) {
      return (
        <button
          onClick={(e) => {
            e.stopPropagation()
            setLinkingClusterId(clusterId)
            setShowCreatePerson(false)
            setNewPersonName('')
            setLinkSearchFilter('')
            setTimeout(() => linkSearchRef.current?.focus(), 0)
          }}
          className="text-xs text-cyan-400/70 hover:text-cyan-300 transition-colors shrink-0"
          title="Link to person"
        >
          Link
        </button>
      )
    }

    return (
      <div
        className="absolute left-0 top-full mt-1 z-40 bg-neutral-900 border border-neutral-700 rounded-lg shadow-xl
                    min-w-[220px] py-1 text-sm"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search filter */}
        <div className="px-2 pb-1 pt-0.5">
          <input
            ref={linkSearchRef}
            value={linkSearchFilter}
            onChange={(e) => setLinkSearchFilter(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') { setLinkingClusterId(null); setLinkSearchFilter('') }
              if (e.key === 'Enter' && filteredPersons.length === 1) {
                handleLinkCluster(clusterId, filteredPersons[0].id)
              }
            }}
            placeholder="Search persons..."
            className="w-full px-2 py-1 text-xs rounded bg-neutral-800 border border-neutral-600
                       text-neutral-100 placeholder-neutral-500 outline-none focus:border-blue-500"
            autoFocus
          />
        </div>
        <div className="max-h-[200px] overflow-y-auto">
          {filteredPersons.map((p) => (
            <button
              key={p.id}
              onClick={() => handleLinkCluster(clusterId, p.id)}
              className="w-full text-left px-3 py-1.5 hover:bg-neutral-800 text-neutral-200 truncate"
            >
              {p.name}
              <span className="text-neutral-500 ml-1 text-xs">({p.face_count})</span>
            </button>
          ))}
          {filteredPersons.length === 0 && linkSearchFilter.trim().length > 0 && (
            <div className="px-3 py-1.5 text-xs text-neutral-500">No results</div>
          )}
        </div>
        <div className="border-t border-neutral-700 mt-1 pt-1">
          {showCreatePerson ? (
            <div className="px-3 py-1.5 flex items-center gap-1">
              <input
                ref={newPersonRef}
                value={newPersonName}
                onChange={(e) => setNewPersonName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleCreatePersonAndLink(clusterId)
                  if (e.key === 'Escape') { setShowCreatePerson(false); setLinkingClusterId(null) }
                }}
                placeholder="Person name..."
                className="flex-1 px-2 py-0.5 text-xs rounded bg-neutral-800 border border-neutral-600
                           text-neutral-100 placeholder-neutral-500 outline-none focus:border-blue-500"
                autoFocus
              />
              <button
                onClick={() => handleCreatePersonAndLink(clusterId)}
                className="text-emerald-400 hover:text-emerald-300 text-xs font-medium"
              >
                ✓
              </button>
            </div>
          ) : (
            <button
              onClick={() => {
                setShowCreatePerson(true)
                setTimeout(() => newPersonRef.current?.focus(), 0)
              }}
              className="w-full text-left px-3 py-1.5 hover:bg-neutral-800 text-cyan-400"
            >
              + New Person...
            </button>
          )}
        </div>
        <div className="border-t border-neutral-700 mt-1 pt-1">
          <button
            onClick={() => setLinkingClusterId(null)}
            className="w-full text-left px-3 py-1.5 hover:bg-neutral-800 text-neutral-500 text-xs"
          >
            Cancel
          </button>
        </div>
      </div>
    )
  }

  const renderEditingField = (_key: string, identityName: string, clusterId?: number | null) => (
    <div className="flex items-center gap-2">
      <input
        ref={editInputRef}
        type="text"
        value={editValue}
        onChange={(e) => setEditValue(e.target.value)}
        onKeyDown={(e) => handleEditKeyDown(e, identityName, clusterId)}
        placeholder="Enter alias..."
        className="flex-1 px-2 py-1 text-sm rounded bg-neutral-800 border border-neutral-600
                   text-neutral-100 placeholder-neutral-500 outline-none focus:border-blue-500
                   min-w-0"
      />
      <button
        onClick={() => saveAlias(identityName, clusterId)}
        className="text-emerald-400 hover:text-emerald-300 shrink-0"
        title="Save (Enter)"
      >
        <svg
          className="w-4 h-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M4.5 12.75l6 6 9-13.5"
          />
        </svg>
      </button>
      <button
        onClick={cancelEditing}
        className="text-neutral-500 hover:text-neutral-300 shrink-0"
        title="Cancel (Esc)"
      >
        <svg
          className="w-4 h-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      </button>
    </div>
  )

  const renderRelinkButton = (cluster: FaceCluster) => {
    if (cluster.cluster_id == null) {
      return null
    }

    return (
      <button
        onClick={(e) => {
          e.stopPropagation()
          void openRelinkDialog(cluster)
        }}
        className="rounded-lg border border-violet-800/60 bg-violet-950/20 px-3 py-1.5 text-xs text-violet-200 hover:bg-violet-900/30 transition-colors shrink-0"
        title="Relink this cluster to another alias or person"
      >
        Relink
      </button>
    )
  }

  const getRelinkClusterFromPersonCluster = useCallback(
    (cluster: PersonCluster, personId: number): FaceCluster => {
      const existing = clusters.find((item) => item.cluster_id === cluster.cluster_id)
      if (existing) {
        return existing
      }

      const trimmedLabel = cluster.label.trim()
      const inferredDisplayName =
        trimmedLabel
        && trimmedLabel.toLowerCase() !== 'unknown'
        && !/^cluster\s+\d+$/i.test(trimmedLabel)
          ? trimmedLabel
          : null

      return {
        cluster_id: cluster.cluster_id,
        identity_name: inferredDisplayName ?? `cluster-${cluster.cluster_id}`,
        display_name: inferredDisplayName,
        identity_id: null,
        image_count: cluster.image_count,
        face_count: cluster.face_count,
        representative_id: cluster.representative_id,
        person_id: personId,
      }
    },
    [clusters]
  )

  const renderEditButton = (
    key: string,
    displayName: string | null,
    identityName: string
  ) => (
    <button
       onClick={() => startEditing(key, displayName, identityName)}
       className="text-neutral-600 hover:text-neutral-300 transition-colors shrink-0"
       title="Rename label"
     >
      <svg
        className="w-4 h-4"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.5}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10"
        />
      </svg>
    </button>
  )

  const renderExpandedDetail = (key: string) => {
    const isLoading = loadingDetail === key

    if (isLoading) {
      return (
        <div className="px-5 pb-3 pt-1">
          <div className="flex items-center gap-2 text-neutral-500 text-xs py-2">
            <span className="w-3 h-3 border border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
            Loading...
          </div>
        </div>
      )
    }

    // Cluster mode: show face crop thumbnails
    const occs = clusterOccurrences[key]
    if (occs) {
      if (occs.length === 0) {
        return (
          <div className="px-5 pb-3 pt-1">
            <p className="text-xs text-neutral-600 py-2">
              No face occurrences found.
            </p>
          </div>
        )
      }
      return (
        <div className="px-5 pb-3 pt-1">
          <div className="flex flex-wrap gap-2">
            {occs.map((occ) => (
              <div
                key={occ.id}
                className="group relative cursor-pointer"
                title={`Click to open · ${occ.file_path.split(/[/\\]/).pop()}${occ.age ? ` | Age: ~${occ.age}` : ''}${occ.gender ? ` | ${occ.gender}` : ''}`}
                onClick={() => { setLightboxPath(occ.file_path); setLightboxImageId(occ.image_id) }}
              >
                <FaceCropThumbnail occurrenceId={occ.id} size="lg" />
                <div className="absolute inset-x-0 bottom-0 bg-black/70 opacity-0 group-hover:opacity-100 transition-opacity px-1 py-0.5 rounded-b">
                  <p className="text-[10px] text-neutral-300 truncate">
                    {occ.file_path.split(/[/\\]/).pop()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )
    }

    // Legacy mode: show full image thumbnails
    const imgs = legacyImages[key]
    if (imgs) {
      if (imgs.length === 0) {
        return (
          <div className="px-5 pb-3 pt-1">
            <p className="text-xs text-neutral-600 py-2">
              No images found for this identity.
            </p>
          </div>
        )
      }
      return (
        <div className="px-5 pb-3 pt-1">
          <div className="flex flex-wrap gap-2">
            {imgs.map((img) => (
              <div
                key={img.image_id}
                className="group relative"
                title={img.file_path}
              >
                <ImageThumbnail filePath={img.file_path} imageId={img.image_id} />
                <div className="absolute inset-x-0 bottom-0 bg-black/70 opacity-0 group-hover:opacity-100 transition-opacity px-1 py-0.5 rounded-b">
                  <p className="text-[10px] text-neutral-300 truncate">
                    {img.file_path.split(/[/\\]/).pop()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )
    }

    return null
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-neutral-800 shrink-0">
        <div className="flex items-center gap-3">
          <h2 className="text-base font-medium text-neutral-100">
            Face Identities
          </h2>
          {!loading && hasOccurrences && (
            <div className="flex rounded-md overflow-hidden border border-neutral-700">
              <button
                onClick={() => setViewMode('clusters')}
                className={`px-2.5 py-0.5 text-xs transition-colors ${
                  viewMode === 'clusters'
                    ? 'bg-neutral-700 text-neutral-100'
                    : 'bg-neutral-900 text-neutral-500 hover:text-neutral-300'
                }`}
              >
                Clusters
              </button>
              <button
                onClick={() => setViewMode('people')}
                className={`px-2.5 py-0.5 text-xs transition-colors ${
                  viewMode === 'people'
                    ? 'bg-neutral-700 text-neutral-100'
                    : 'bg-neutral-900 text-neutral-500 hover:text-neutral-300'
                }`}
              >
                People{persons.length > 0 && ` (${persons.length})`}
              </button>
            </div>
          )}
          {!loading && (
            <span className="text-xs text-neutral-500">
              {totalEntries}{' '}
              {hasOccurrences ? 'cluster' : 'identity'}
              {totalEntries !== 1 ? 's' : ''} &middot; {totalFaceCount}{' '}
              {hasOccurrences ? 'face' : 'image'}
              {totalFaceCount !== 1 ? 's' : ''}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Cluster button — only if we have occurrences */}
          {hasOccurrences && (
            <button
              onClick={handleCluster}
              disabled={clustering}
              className="px-3 py-1 text-xs rounded-md bg-blue-900/50 text-blue-300 hover:bg-blue-800/60
                         disabled:opacity-50 transition-colors border border-blue-700/40"
              title="Group similar faces by embedding similarity"
            >
              {clustering ? (
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 border border-blue-600 border-t-blue-300 rounded-full animate-spin" />
                  Clustering...
                </span>
              ) : (
                'Cluster Faces'
              )}
            </button>
          )}
          <button
            onClick={loadData}
            disabled={loading}
            className="px-3 py-1 text-xs rounded-md bg-neutral-800 text-neutral-300 hover:bg-neutral-700
                       disabled:opacity-50 transition-colors"
            title="Refresh"
          >
            {loading ? (
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 border border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
                Loading...
              </span>
            ) : (
              'Refresh'
            )}
          </button>
        </div>
      </div>

      {/* Cluster relink dialog */}
      {relinkingCluster && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          onClick={() => {
            if (!relinkSubmitting) {
              closeRelinkDialog()
            }
          }}
        >
          <div
            className="w-full max-w-5xl max-h-[85vh] overflow-hidden rounded-xl border border-neutral-700 bg-neutral-900 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-neutral-800 px-5 py-3">
              <div>
                <h3 className="text-sm font-semibold text-neutral-100">Relink Cluster</h3>
                <p className="text-xs text-neutral-500">
                  Move this cluster to another alias or person without relying on free-text renaming.
                </p>
              </div>
              <button
                onClick={closeRelinkDialog}
                disabled={relinkSubmitting}
                className="text-neutral-500 hover:text-neutral-300 transition-colors disabled:opacity-50"
                title="Close"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="grid md:grid-cols-[320px,minmax(0,1fr)] max-h-[calc(85vh-72px)]">
              <div className="border-r border-neutral-800 p-5 space-y-4 overflow-y-auto">
                <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4 space-y-3">
                  <div className="flex items-center gap-3">
                    {relinkingCluster.representative_id != null ? (
                      <FaceCropThumbnail occurrenceId={relinkingCluster.representative_id} size="lg" />
                    ) : (
                      <div className="w-24 h-24 rounded bg-neutral-800 flex items-center justify-center">
                        <svg className="w-8 h-8 text-neutral-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                        </svg>
                      </div>
                    )}
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-neutral-100 truncate">
                        {relinkingCluster.display_name || relinkingCluster.identity_name}
                      </p>
                      <p className="text-xs text-neutral-500 truncate">
                        {relinkingCluster.identity_name}
                      </p>
                      <p className="text-xs text-neutral-500 mt-1">
                        Cluster {relinkingCluster.cluster_id} · {relinkingCluster.face_count} faces · {relinkingCluster.image_count} images
                      </p>
                    </div>
                  </div>
                  <div className="text-xs text-neutral-400 space-y-1">
                    <p>
                      Current label:{' '}
                      <span className="text-neutral-200">
                        {relinkingCluster.display_name || 'None'}
                      </span>
                    </p>
                    <p>
                      Current person:{' '}
                      <span className="text-neutral-200">
                        {currentRelinkPerson?.name ?? 'Unlinked'}
                      </span>
                    </p>
                  </div>
                </div>

                {relinkPreviewText && (
                  <div className="rounded-lg border border-violet-800/50 bg-violet-950/30 p-3">
                    <p className="text-xs font-medium text-violet-200 mb-1">Preview</p>
                    <p className="text-xs text-violet-100/90">{relinkPreviewText}</p>
                  </div>
                )}

                {relinkSelection?.type === 'alias' && currentRelinkPerson && (
                  <label className="flex items-start gap-2 rounded-lg border border-neutral-800 bg-neutral-950/50 p-3 text-xs text-neutral-300">
                    <input
                      type="checkbox"
                      checked={unlinkPersonOnAliasRelink}
                      onChange={(e) => setUnlinkPersonOnAliasRelink(e.target.checked)}
                      className="mt-0.5 rounded border-neutral-600 bg-neutral-800 text-violet-500 focus:ring-violet-500"
                    />
                    <span>Also unlink this cluster from {currentRelinkPerson.name}</span>
                  </label>
                )}

                <div className="rounded-lg border border-neutral-800 bg-neutral-950/50 p-3 space-y-2">
                  <p className="text-[11px] uppercase tracking-wide text-neutral-500">Quick actions</p>
                  <button
                    onClick={handleClearRelinkAlias}
                    disabled={relinkSubmitting || !relinkingCluster.display_name}
                    className="w-full rounded-md border border-neutral-700 px-3 py-2 text-left text-xs text-neutral-300 hover:bg-neutral-800 disabled:opacity-40"
                  >
                    Clear current label
                  </button>
                  <button
                    onClick={handleRelinkUnlinkPerson}
                    disabled={relinkSubmitting || !currentRelinkPerson}
                    className="w-full rounded-md border border-neutral-700 px-3 py-2 text-left text-xs text-neutral-300 hover:bg-neutral-800 disabled:opacity-40"
                  >
                    Unlink current person
                  </button>
                  {relinkSearchTrimmed && (
                    <>
                      <button
                        onClick={handleCreateRelinkAlias}
                        disabled={relinkSubmitting}
                        className="w-full rounded-md border border-violet-700/50 bg-violet-950/30 px-3 py-2 text-left text-xs text-violet-200 hover:bg-violet-900/40 disabled:opacity-40"
                      >
                        Create alias "{relinkSearchTrimmed}" for this cluster
                      </button>
                      <button
                        onClick={handleCreateRelinkPerson}
                        disabled={relinkSubmitting}
                        className="w-full rounded-md border border-cyan-700/50 bg-cyan-950/20 px-3 py-2 text-left text-xs text-cyan-200 hover:bg-cyan-900/30 disabled:opacity-40"
                      >
                        Create person "{relinkSearchTrimmed}" and link this cluster
                      </button>
                    </>
                  )}
                </div>
              </div>

              <div className="min-h-0 flex flex-col">
                <div className="border-b border-neutral-800 px-5 py-4">
                  <input
                    ref={relinkSearchRef}
                    value={relinkSearch}
                    onChange={(e) => setRelinkSearch(e.target.value)}
                    placeholder="Search existing alias or person..."
                    className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-sm text-neutral-100 placeholder-neutral-500 outline-none focus:border-violet-500"
                    autoFocus
                  />
                </div>

                {relinkLoading ? (
                  <div className="flex-1 flex items-center justify-center gap-2 text-sm text-neutral-500">
                    <span className="w-4 h-4 border border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
                    Loading relink targets...
                  </div>
                ) : (
                  <div className="flex-1 overflow-y-auto">
                    <div className="px-5 pt-4 pb-2 text-[11px] uppercase tracking-wide text-neutral-500">
                      Likely matches
                    </div>
                    <div className="px-4 space-y-2">
                      {relinkSuggestions.map((suggestion) => {
                        const isPersonSuggestion = suggestion.target_type === 'person'
                        const selected = isPersonSuggestion
                          ? relinkSelection?.type === 'person'
                            && relinkSelection.personId === suggestion.person_id
                          : relinkSelection?.type === 'alias'
                            && relinkSelection.label === suggestion.label

                        return (
                          <button
                            key={`suggestion:${suggestion.target_type}:${suggestion.person_id ?? suggestion.cluster_id ?? suggestion.label}`}
                            onClick={() => {
                              if (isPersonSuggestion) {
                                if (suggestion.person_id == null) {
                                  return
                                }
                                setRelinkSelection({
                                  type: 'person',
                                  personId: suggestion.person_id,
                                  label: suggestion.label,
                                })
                                setUnlinkPersonOnAliasRelink(false)
                                return
                              }
                              setRelinkSelection({ type: 'alias', label: suggestion.label })
                            }}
                            className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                              selected
                                ? 'border-emerald-600 bg-emerald-950/25'
                                : 'border-neutral-800 bg-neutral-950/40 hover:bg-neutral-800/50'
                            }`}
                          >
                            <div className="flex items-center gap-3">
                              {suggestion.representative_id != null ? (
                                <FaceCropThumbnail occurrenceId={suggestion.representative_id} size="sm" />
                              ) : (
                                <div className="w-12 h-12 rounded bg-neutral-800" />
                              )}
                              <div className="min-w-0 flex-1">
                                <p className="text-sm text-neutral-100 truncate">{suggestion.label}</p>
                                <p className="text-xs text-neutral-500 truncate">
                                  {isPersonSuggestion ? 'Person' : 'Alias'} · similarity {suggestion.score.toFixed(3)} · {suggestion.face_count} faces
                                </p>
                                <p className="text-[11px] text-neutral-600 truncate">{suggestion.reason}</p>
                              </div>
                            </div>
                          </button>
                        )
                      })}
                      {relinkSuggestions.length === 0 && (
                        <div className="rounded-lg border border-dashed border-neutral-800 px-3 py-4 text-xs text-neutral-500">
                          No likely matches available yet.
                        </div>
                      )}
                    </div>

                    <div className="px-5 pt-4 pb-2 text-[11px] uppercase tracking-wide text-neutral-500">
                      People
                    </div>
                    <div className="px-4 space-y-2">
                      {filteredRelinkPersons.map((person) => {
                        const selected = relinkSelection?.type === 'person' && relinkSelection.personId === person.id
                        return (
                          <button
                            key={`person:${person.id}`}
                            onClick={() => {
                              setRelinkSelection({ type: 'person', personId: person.id, label: person.name })
                              setUnlinkPersonOnAliasRelink(false)
                            }}
                            className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                              selected
                                ? 'border-cyan-700 bg-cyan-950/30'
                                : 'border-neutral-800 bg-neutral-950/40 hover:bg-neutral-800/50'
                            }`}
                          >
                            <div className="flex items-center gap-3">
                              {person.representative_id != null ? (
                                <FaceCropThumbnail occurrenceId={person.representative_id} size="sm" />
                              ) : (
                                <div className="w-12 h-12 rounded bg-neutral-800" />
                              )}
                              <div className="min-w-0 flex-1">
                                <p className="text-sm text-neutral-100 truncate">{person.name}</p>
                                <p className="text-xs text-neutral-500">
                                  {person.cluster_count} clusters · {person.image_count} images
                                </p>
                              </div>
                            </div>
                          </button>
                        )
                      })}
                      {filteredRelinkPersons.length === 0 && (
                        <div className="rounded-lg border border-dashed border-neutral-800 px-3 py-4 text-xs text-neutral-500">
                          No people match this search.
                        </div>
                      )}
                    </div>

                    <div className="px-5 pt-5 pb-2 text-[11px] uppercase tracking-wide text-neutral-500">
                      Aliases
                    </div>
                    <div className="px-4 pb-4 space-y-2">
                      {relinkAliasCandidates.map((candidate) => {
                        const selected = relinkSelection?.type === 'alias' && relinkSelection.label === candidate.label
                        return (
                          <button
                            key={candidate.key}
                            onClick={() => {
                              setRelinkSelection({ type: 'alias', label: candidate.label })
                            }}
                            className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                              selected
                                ? 'border-violet-700 bg-violet-950/30'
                                : 'border-neutral-800 bg-neutral-950/40 hover:bg-neutral-800/50'
                            }`}
                          >
                            <div className="flex items-center gap-3">
                              {candidate.representativeId != null ? (
                                <FaceCropThumbnail occurrenceId={candidate.representativeId} size="sm" />
                              ) : (
                                <div className="w-12 h-12 rounded bg-neutral-800 flex items-center justify-center">
                                  <svg className="w-4 h-4 text-neutral-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 21l-7.5-4.5L4.5 21V5.25A2.25 2.25 0 016.75 3h10.5A2.25 2.25 0 0119.5 5.25V21z" />
                                  </svg>
                                </div>
                              )}
                              <div className="min-w-0 flex-1">
                                <p className="text-sm text-neutral-100 truncate">{candidate.label}</p>
                                <p className="text-xs text-neutral-500 truncate">
                                  {candidate.subtitle} · {candidate.imageCount} images
                                </p>
                              </div>
                            </div>
                          </button>
                        )
                      })}
                      {relinkAliasCandidates.length === 0 && (
                        <div className="rounded-lg border border-dashed border-neutral-800 px-3 py-4 text-xs text-neutral-500">
                          No aliases match this search.
                        </div>
                      )}
                    </div>
                  </div>
                )}

                <div className="border-t border-neutral-800 px-5 py-3 flex items-center justify-between gap-3">
                  <p className="text-xs text-neutral-500">
                    Pick a person to relink membership, or pick an alias to relabel this cluster.
                  </p>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={closeRelinkDialog}
                      disabled={relinkSubmitting}
                      className="px-3 py-1.5 text-xs rounded-md bg-neutral-800 text-neutral-300 hover:bg-neutral-700 border border-neutral-700 disabled:opacity-40"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleApplyRelinkSelection}
                      disabled={relinkSubmitting || relinkLoading || relinkSelection == null}
                      className="px-3 py-1.5 text-xs rounded-md bg-violet-700 text-white hover:bg-violet-600 disabled:opacity-40"
                    >
                      {relinkSubmitting
                        ? 'Applying...'
                        : relinkSelection?.type === 'person'
                          ? 'Relink to person'
                          : 'Apply alias'}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="px-5 py-3 bg-red-900/30 border-b border-red-800/50 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Info banner: no occurrences yet */}
      {!loading && !hasOccurrences && legacyFaces.length > 0 && (
        <div className="px-5 py-2.5 bg-amber-900/20 border-b border-amber-800/30 text-amber-300/80 text-xs">
          Face occurrences (bounding boxes + embeddings) have not been stored yet. Re-run the{' '}
          <span className="font-medium">faces</span> analysis pass to enable face crop thumbnails
          and clustering.
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && totalEntries === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center text-neutral-600 gap-3">
          <svg
            className="w-16 h-16 opacity-30"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={0.75}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"
            />
          </svg>
          <p className="text-sm">No face identities found</p>
          <p className="text-xs text-neutral-700">
            Run the face recognition pass on your images to detect faces.
          </p>
        </div>
      )}

      {/* Face list — cluster mode */}
      {hasOccurrences && clusters.length > 0 && viewMode === 'clusters' && (
        <div className="flex-1 overflow-y-auto">
          {clusters.map((cluster) => {
            const key = cluster.cluster_id !== null
              ? `cluster:${cluster.cluster_id}`
              : `name:${cluster.identity_name}`
            const isExpanded = expandedKey === key
            const isEditing = editingKey === key
            const displayLabel = cluster.display_name || cluster.identity_name

            return (
              <div key={key} className="border-b border-neutral-800/60">
                {/* Row */}
                <div className="flex items-center gap-3 px-5 py-2.5 hover:bg-neutral-800/40 transition-colors">
                  {/* Representative face crop */}
                  {cluster.representative_id != null && (
                    <FaceCropThumbnail
                      occurrenceId={cluster.representative_id}
                      size="sm"
                    />
                  )}

                  {/* Expand chevron */}
                  <button
                    onClick={() => toggleExpand(key, cluster, null)}
                    className="text-neutral-500 hover:text-neutral-300 transition-colors shrink-0"
                    title={isExpanded ? 'Collapse' : 'Expand to see faces'}
                  >
                    <svg
                      className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M8.25 4.5l7.5 7.5-7.5 7.5"
                      />
                    </svg>
                  </button>

                  {/* Name / alias */}
                  <div className="flex-1 min-w-0">
                    {isEditing ? (
                      renderEditingField(key, cluster.identity_name, cluster.cluster_id)
                    ) : (
                      <div>
                        <span className="text-sm text-neutral-100 font-medium truncate block">
                          {displayLabel}
                        </span>
                        {cluster.display_name && (
                          <span className="text-xs text-neutral-500 truncate block">
                            {cluster.identity_name}
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Counts */}
                  <span className="text-xs text-neutral-400 bg-neutral-800 px-2 py-0.5 rounded-full shrink-0">
                    {cluster.face_count} {cluster.face_count === 1 ? 'face' : 'faces'}
                    {' / '}
                    {cluster.image_count} {cluster.image_count === 1 ? 'image' : 'images'}
                  </span>

                  {/* Edit button */}
                  {!isEditing &&
                    renderRelinkButton(cluster)}

                  {!isEditing &&
                    renderEditButton(
                      key,
                      cluster.display_name,
                      cluster.identity_name
                    )}

                  {/* Link to person button */}
                  {!isEditing && cluster.cluster_id !== null && (
                    <div className="relative">
                      {renderLinkDropdown(cluster.cluster_id)}
                    </div>
                  )}
                </div>

                {/* Expanded detail */}
                {isExpanded && renderExpandedDetail(key)}
              </div>
            )
          })}

          {/* Load more button */}
          {hasMoreClusters && (
            <div className="px-5 py-3 flex justify-center">
              <button
                onClick={loadMoreClusters}
                className="px-4 py-1.5 text-xs rounded-md bg-neutral-800 text-neutral-300
                           hover:bg-neutral-700 transition-colors border border-neutral-700"
              >
                Load more ({totalClusterCount - clusters.length} remaining)
              </button>
            </div>
          )}
        </div>
      )}

      {/* Face list — people mode */}
      {hasOccurrences && viewMode === 'people' && (
        <div className="flex-1 min-h-0 overflow-hidden bg-neutral-950/50">
          <div className="flex h-full min-h-0 flex-col lg:flex-row">
            <div
              className={`shrink-0 border-b border-neutral-800 ${
                isPeopleChooserExpanded
                  ? 'flex-1 lg:border-b-0 lg:border-r-0'
                  : 'lg:w-72 lg:border-b-0 lg:border-r'
              }`}
            >
              <div className="border-b border-neutral-800/60 px-4 py-3">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-[11px] uppercase tracking-wide text-neutral-500">People chooser</p>
                  <button
                    type="button"
                    onClick={() => setIsPeopleChooserExpanded((current) => !current)}
                    className="shrink-0 rounded-lg border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-xs text-neutral-300 transition-colors hover:bg-neutral-800"
                  >
                    {isPeopleChooserExpanded ? 'Collapse chooser' : 'Expand chooser'}
                  </button>
                </div>
                <input
                  value={peopleChooserFilter}
                  onChange={(e) => setPeopleChooserFilter(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Escape') setPeopleChooserFilter('') }}
                  placeholder="Filter by name…"
                  className="mt-2 w-full px-2.5 py-1.5 text-xs rounded-lg bg-neutral-800 border border-neutral-700 text-neutral-100 placeholder-neutral-500 outline-none focus:border-cyan-600"
                />
              </div>
              <div
                className={`px-4 py-4 ${
                  isPeopleChooserExpanded
                    ? 'grid max-h-[calc(100%-76px)] grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-4 overflow-y-auto'
                    : 'flex gap-3 overflow-x-auto lg:h-[calc(100%-76px)] lg:flex-col lg:overflow-y-auto lg:overflow-x-hidden'
                }`}
              >
                {chooserPersons.map((person) => {
                  const isSelected = expandedPersonId === person.id
                  return (
                    <button
                      key={`person:${person.id}`}
                      type="button"
                      onClick={() => void togglePersonExpand(person.id)}
                      className={`rounded-xl border text-left transition-colors ${
                        isSelected
                          ? 'border-cyan-700/60 bg-cyan-950/30'
                          : 'border-neutral-800 bg-neutral-900/70 hover:border-neutral-700 hover:bg-neutral-900'
                      } ${
                        isPeopleChooserExpanded
                          ? 'flex flex-col overflow-hidden p-0'
                          : 'flex min-w-[180px] items-center gap-3 px-3 py-2 lg:min-w-0'
                      }`}
                    >
                      <div
                        className={`overflow-hidden bg-neutral-800 ${
                          isPeopleChooserExpanded
                            ? 'aspect-square w-full rounded-t-xl'
                            : 'h-14 w-14 shrink-0 rounded-lg'
                        }`}
                      >
                        {person.representative_id != null ? (
                          <FaceCropThumbnail occurrenceId={person.representative_id} size="fill" />
                        ) : (
                          <div className="flex h-full w-full items-center justify-center">
                            <svg className="h-6 w-6 text-neutral-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                            </svg>
                          </div>
                        )}
                      </div>
                      <div className={`min-w-0 ${isPeopleChooserExpanded ? 'w-full px-3 py-2' : 'flex-1'}`}>
                        <p className="truncate text-sm font-medium text-neutral-100">{person.name}</p>
                        <p className="mt-1 text-[11px] text-neutral-500">
                          {person.face_count} faces · {person.cluster_count} clusters
                        </p>
                      </div>
                    </button>
                  )
                })}
                {chooserPersons.length === 0 && (
                  <div className="rounded-xl border border-dashed border-neutral-800 px-4 py-6 text-sm text-neutral-600">
                    {peopleChooserFilter.trim() ? 'No people match your filter.' : 'No people found yet.'}
                  </div>
                )}
              </div>
            </div>

            <div className={`min-h-0 flex-1 flex-col overflow-hidden ${isPeopleChooserExpanded ? 'hidden' : 'flex'}`}>
              {activePerson ? (
                <>
                  <div className="border-b border-neutral-800 bg-neutral-950/80 px-4 py-4">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        {editingPersonId === activePerson.id ? (
                          <div className="flex items-center gap-2">
                            <input
                              ref={personEditRef}
                              value={personEditValue}
                              onChange={(e) => setPersonEditValue(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') handleRenamePerson(activePerson.id)
                                if (e.key === 'Escape') setEditingPersonId(null)
                              }}
                              className="w-64 max-w-full rounded-lg border border-neutral-600 bg-neutral-900 px-3 py-1.5 text-sm text-neutral-100 outline-none focus:border-blue-500"
                              autoFocus
                            />
                            <button
                              type="button"
                              onClick={() => setEditingPersonId(null)}
                              className="text-xs text-neutral-500 hover:text-neutral-300"
                            >
                              Cancel
                            </button>
                          </div>
                        ) : (
                          <>
                            <h3 className="text-base font-semibold text-neutral-100">{activePerson.name}</h3>
                            <p className="mt-1 text-xs text-neutral-500">
                              {activePerson.face_count} faces · {activePerson.cluster_count} clusters
                            </p>
                          </>
                        )}
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <button
                          type="button"
                          onClick={() => {
                            setEditingPersonId(activePerson.id)
                            setPersonEditValue(activePerson.name)
                            setTimeout(() => personEditRef.current?.focus(), 0)
                          }}
                          className="rounded-lg border border-neutral-700 px-3 py-1.5 text-xs text-neutral-200 hover:bg-neutral-800"
                        >
                          Rename person
                        </button>
                        <button
                          type="button"
                          onClick={() => setDeletingPersonId(activePerson.id)}
                          className="rounded-lg border border-red-800/60 bg-red-950/20 px-3 py-1.5 text-xs text-red-200 hover:bg-red-900/30"
                        >
                          Delete person
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setExpandedPersonId(null)
                            setIsPeopleChooserExpanded(false)
                          }}
                          className="text-xs text-neutral-500 hover:text-neutral-300"
                        >
                          Close workspace
                        </button>
                      </div>
                    </div>

                    <div className="mt-4 flex flex-wrap gap-2">
                      {([
                        ['linked', 'Linked', `${activeLinkedCount}`],
                        ['suggested', 'Suggested', `${activeSuggestedClusters.length}`],
                        ['unlinked', 'Unlinked', `${activeUnlinkedClusters.length > unlinkedClusterTarget ? unlinkedClusterTarget + '+' : activeUnlinkedClusters.length}`],
                        ['inspector', 'Inspector', inspectorCluster ? '1' : '0'],
                      ] as Array<[PeopleStage, string, string]>).map(([stage, label, count]) => {
                        const isDisabled = stage === 'inspector' && !inspectorCluster
                        return (
                          <button
                            key={stage}
                            type="button"
                            disabled={isDisabled}
                            onClick={() => {
                              if (stage === 'inspector') {
                                setPeopleStage('inspector')
                                return
                              }
                              setPeopleStage(stage)
                              setInspectorReturnStage(stage)
                            }}
                            className={`rounded-full border px-3 py-1.5 text-xs transition-colors ${
                              peopleStage === stage
                                ? 'border-cyan-600 bg-cyan-950/40 text-cyan-100'
                                : 'border-neutral-700 bg-neutral-900 text-neutral-300 hover:bg-neutral-800'
                            } disabled:cursor-not-allowed disabled:opacity-40`}
                          >
                            {label} <span className="ml-1 text-neutral-500">{count}</span>
                          </button>
                        )
                      })}
                    </div>
                  </div>

                  <div className="flex-1 min-h-0 overflow-y-auto px-4 py-4">
                    {peopleStage === 'linked' && (
                      <div className="space-y-4">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div>
                            <p className="text-[11px] uppercase tracking-wide text-cyan-300/90">Linked clusters</p>
                            <p className="mt-1 text-xs text-neutral-500">
                              Review confirmed matches for {activePerson.name}.
                            </p>
                          </div>
                        </div>

                        {activePersonLoading ? (
                          <div className="flex items-center gap-2 rounded-xl border border-neutral-800 bg-neutral-900/60 px-4 py-3 text-sm text-neutral-500">
                            <span className="h-4 w-4 rounded-full border border-neutral-600 border-t-neutral-300 animate-spin" />
                            Loading linked clusters...
                          </div>
                        ) : activePersonClusterError ? (
                          <div className="rounded-xl border border-amber-900/60 bg-amber-950/30 px-4 py-4 text-sm text-amber-100">
                            <p>Couldn&apos;t load linked clusters right now.</p>
                            <p className="mt-1 text-xs text-amber-200/80">{activePersonClusterError}</p>
                            <button
                              type="button"
                              onClick={() => void loadPersonClusters(activePerson.id, true)}
                              className="mt-3 rounded-lg border border-neutral-700 px-3 py-1.5 text-xs text-neutral-200 hover:bg-neutral-800"
                            >
                              Retry linked clusters
                            </button>
                          </div>
                        ) : activePersonClusters.length > 0 ? (
                          <div className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-4">
                            {activePersonClusters.map((pc) => {
                              const relinkCluster = getRelinkClusterFromPersonCluster(pc, activePerson.id)
                              const key = `cluster:${pc.cluster_id}`
                              const isSelected = expandedKey === key && peopleStage === 'inspector'
                              return (
                                <div
                                  key={pc.cluster_id}
                                  className={`group relative rounded-xl border transition-colors ${
                                    isSelected
                                      ? 'border-cyan-600/70 bg-cyan-950/20'
                                      : 'border-cyan-800/40 bg-cyan-950/10 hover:bg-cyan-900/15'
                                  }`}
                                >
                                  <button
                                    type="button"
                                    onClick={() => void openClusterInspector(key, relinkCluster, 'linked')}
                                    className="block w-full text-left"
                                  >
                                    <div className="absolute left-2 top-2 rounded px-1.5 py-0.5 text-[10px] text-cyan-100 bg-cyan-900/60 border border-cyan-700/60 z-10">
                                      Linked
                                    </div>
                                    <div className="aspect-square w-full overflow-hidden rounded-t-xl bg-neutral-800">
                                      {pc.representative_id != null ? (
                                        <FaceCropThumbnail occurrenceId={pc.representative_id} size="fill" />
                                      ) : (
                                        <div className="flex h-full w-full items-center justify-center">
                                          <svg className="h-10 w-10 text-neutral-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                                          </svg>
                                        </div>
                                      )}
                                    </div>
                                    <div className="px-3 py-2">
                                      <p className="truncate text-sm text-neutral-100">{pc.label}</p>
                                      <p className="mt-1 text-[11px] text-neutral-400">
                                        {pc.face_count} faces · {pc.image_count} images
                                      </p>
                                    </div>
                                  </button>
                                  <div className="absolute right-2 top-2 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                                    <div onClick={(e) => e.stopPropagation()}>
                                      {renderRelinkButton(relinkCluster)}
                                    </div>
                                    <button
                                      type="button"
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        void handleUnlinkCluster(pc.cluster_id)
                                      }}
                                      className="rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-neutral-300 hover:text-red-400"
                                    >
                                      Unlink
                                    </button>
                                  </div>
                                </div>
                              )
                            })}
                          </div>
                        ) : (
                          <div className="rounded-xl border border-dashed border-neutral-800 px-4 py-10 text-sm text-neutral-600">
                            No confirmed links yet.
                          </div>
                        )}

                        {/* ── Direct links (images linked without cluster) ── */}
                        {expandedPersonId != null && (personDirectLinks[expandedPersonId] ?? []).length > 0 && (
                          <div className="mt-6">
                            <p className="text-[11px] uppercase tracking-wide text-cyan-300/70 mb-3">
                              Direct links <span className="text-neutral-500">{(personDirectLinks[expandedPersonId] ?? []).length}</span>
                            </p>
                            <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-3">
                              {(personDirectLinks[expandedPersonId] ?? []).map((link) => (
                                <DirectLinkCard
                                  key={link.occurrence_id}
                                  link={link}
                                  onUnlink={() => void handleUnlinkDirectOccurrence(link.occurrence_id)}
                                  onOpenLightbox={() => {
                                    setLightboxPath(link.file_path)
                                    setLightboxImageId(link.image_id)
                                  }}
                                />
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {peopleStage === 'suggested' && (
                      <div className="space-y-4">
                        {/* Inner tab toggle: Clusters | Images */}
                        <div className="flex items-center gap-3">
                          <div className="flex shrink-0 rounded-full border border-neutral-700 bg-neutral-900 p-0.5 text-xs">
                            <button
                              type="button"
                              onClick={() => setSuggestedInnerTab('clusters')}
                              className={`rounded-full px-3 py-1 transition-colors ${
                                suggestedInnerTab === 'clusters' ? 'bg-neutral-700 text-neutral-100' : 'text-neutral-400 hover:text-neutral-200'
                              }`}
                            >
                              Clusters <span className="text-neutral-500">{activeSuggestedClusters.length}</span>
                            </button>
                            <button
                              type="button"
                              onClick={() => setSuggestedInnerTab('images')}
                              className={`rounded-full px-3 py-1 transition-colors ${
                                suggestedInnerTab === 'images' ? 'bg-neutral-700 text-neutral-100' : 'text-neutral-400 hover:text-neutral-200'
                              }`}
                            >
                              Images <span className="text-neutral-500">
                                {expandedPersonId != null
                                  ? (personSimilarImages[expandedPersonId]?.length ?? (loadingPersonSimilarImagesId === expandedPersonId ? '…' : ''))
                                  : ''}
                              </span>
                            </button>
                          </div>
                        </div>

                        {/* ── Clusters sub-tab ── */}
                        {suggestedInnerTab === 'clusters' && (
                          <>
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div>
                            <p className="text-[11px] uppercase tracking-wide text-amber-300/90">Suggested links</p>
                            <p className="mt-1 text-xs text-neutral-500">
                              {suggestedSubFilter === 'deferred'
                                ? `${deferredSuggestedClusters.length} deferred suggestion${deferredSuggestedClusters.length !== 1 ? 's' : ''} parked for later.`
                                : <>Review AI-proposed matches for {activePerson.name}.</>
                              }
                            </p>
                          </div>
                          <div className="flex flex-wrap items-center gap-2">
                            {/* Active / Deferred toggle */}
                            {(deferredSuggestedClusters.length > 0 || suggestedSubFilter === 'deferred') && (
                              <div className="flex shrink-0 rounded-full border border-neutral-700 bg-neutral-900 p-0.5 text-xs">
                                <button
                                  type="button"
                                  onClick={() => setSuggestedSubFilter('active')}
                                  className={`rounded-full px-2.5 py-1 transition-colors ${
                                    suggestedSubFilter === 'active' ? 'bg-neutral-700 text-neutral-100' : 'text-neutral-400 hover:text-neutral-200'
                                  }`}
                                >
                                  Active <span className="text-neutral-500">{activeSuggestedClusters.length}</span>
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setSuggestedSubFilter('deferred')}
                                  className={`rounded-full px-2.5 py-1 transition-colors ${
                                    suggestedSubFilter === 'deferred' ? 'bg-neutral-700 text-neutral-100' : 'text-neutral-400 hover:text-neutral-200'
                                  }`}
                                >
                                  Deferred <span className="text-neutral-500">{deferredSuggestedClusters.length}</span>
                                </button>
                              </div>
                            )}
                            {activeSuggestionsLoading && (
                              <span className="inline-flex items-center gap-1 text-[11px] text-neutral-500">
                                <span className="h-3 w-3 rounded-full border border-neutral-600 border-t-neutral-300 animate-spin" />
                                Scoring...
                              </span>
                            )}
                            {suggestedSubFilter === 'active' && activeSuggestedClusters.length > 0 && (
                              <>
                                <button
                                  type="button"
                                  onClick={() => {
                                    if (allSuggestedSelected) {
                                      setSelectedSuggestedClusterIds([])
                                      return
                                    }
                                    setSelectedSuggestedClusterIds(
                                      activeSuggestedClusters.map((suggestion) => suggestion.cluster_id)
                                    )
                                  }}
                                  disabled={confirmingSuggestedLinks}
                                  className="rounded-lg border border-neutral-700 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-800 disabled:opacity-50"
                                >
                                  {allSuggestedSelected ? 'Clear selection' : 'Select all'}
                                </button>
                                <button
                                  type="button"
                                  onClick={() => void handleConfirmSuggestedLinks()}
                                  disabled={selectedSuggestedCount === 0 || confirmingSuggestedLinks}
                                  className="rounded-lg border border-emerald-700/60 bg-emerald-950/30 px-3 py-1.5 text-xs text-emerald-200 hover:bg-emerald-900/40 disabled:opacity-40"
                                >
                                  {confirmingSuggestedLinks
                                    ? 'Linking...'
                                    : `Confirm selected (${selectedSuggestedCount})`}
                                </button>
                              </>
                            )}
                            {(activeSuggestionsError || !activeSuggestionsLoading) && (
                              <button
                                type="button"
                                onClick={() => void loadPersonLinkSuggestions(activePerson.id, true)}
                                disabled={activeSuggestionsLoading}
                                className="rounded-lg border border-neutral-700 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-800 disabled:opacity-50"
                              >
                                Refresh suggestions
                              </button>
                            )}
                          </div>
                        </div>

                        {activeSuggestionsError ? (
                          <div className="rounded-xl border border-amber-900/60 bg-amber-950/30 px-4 py-4 text-sm text-amber-100">
                            <p>Couldn&apos;t load suggested links right now.</p>
                            <p className="mt-1 text-xs text-amber-200/80">{activeSuggestionsError}</p>
                          </div>
                        ) : visibleSuggestedClusters.length > 0 ? (
                          <div className="grid grid-cols-[repeat(auto-fill,minmax(170px,1fr))] gap-4">
                            {visibleSuggestedClusters.map((suggestion) => {
                              const sourceCluster = clusters.find(
                                (cluster) => cluster.cluster_id === suggestion.cluster_id
                              ) ?? {
                                cluster_id: suggestion.cluster_id,
                                identity_name: suggestion.label || `cluster-${suggestion.cluster_id}`,
                                display_name: suggestion.label || null,
                                identity_id: null,
                                image_count: suggestion.image_count,
                                face_count: suggestion.face_count,
                                representative_id: suggestion.representative_id,
                                person_id: null,
                              }
                              const key = `cluster:${suggestion.cluster_id}`
                              const isSelected = expandedKey === key && peopleStage === 'inspector'
                              const isBatchSelected = selectedSuggestedClusterIds.includes(suggestion.cluster_id)
                              const isDeferred = deferredClusterIds.has(suggestion.cluster_id)

                              return (
                                <div
                                  key={`suggested:${suggestion.cluster_id}`}
                                  className={`relative rounded-xl border transition-colors ${
                                    isSelected
                                      ? 'border-amber-600/70 bg-amber-900/20'
                                      : 'border-amber-800/50 bg-amber-950/10 hover:bg-amber-900/15'
                                  }`}
                                >
                                  <button
                                    type="button"
                                    onClick={() => void openClusterInspector(key, sourceCluster, 'suggested')}
                                    className="block w-full text-left"
                                  >
                                    <div className="absolute left-2 top-2 rounded px-1.5 py-0.5 text-[10px] text-amber-100 bg-amber-900/60 border border-amber-700/60 z-10">
                                      {isDeferred ? 'Deferred' : 'Suggested'}
                                    </div>
                                    <div className="aspect-square w-full overflow-hidden rounded-t-xl bg-neutral-800">
                                      {suggestion.representative_id != null ? (
                                        <FaceCropThumbnail occurrenceId={suggestion.representative_id} size="fill" />
                                      ) : (
                                        <div className="flex h-full w-full items-center justify-center">
                                          <svg className="h-10 w-10 text-neutral-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                                          </svg>
                                        </div>
                                      )}
                                    </div>
                                    <div className="px-3 py-2">
                                      <p className="truncate text-sm text-neutral-100">
                                        {suggestion.label || 'Unknown'}
                                      </p>
                                      <p className="mt-1 text-[11px] text-neutral-400">
                                        {suggestion.face_count} faces · score {suggestion.score.toFixed(3)}
                                      </p>
                                    </div>
                                  </button>
                                  {!isDeferred && (
                                    <button
                                      type="button"
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        toggleSuggestedClusterSelection(suggestion.cluster_id)
                                      }}
                                      className={`absolute right-2 top-2 z-10 rounded px-1.5 py-0.5 text-[10px] border ${
                                        isBatchSelected
                                          ? 'border-emerald-500/70 bg-emerald-900/70 text-emerald-100'
                                          : 'border-neutral-600 bg-black/55 text-neutral-200'
                                      }`}
                                    >
                                      {isBatchSelected ? 'Selected' : 'Select'}
                                    </button>
                                  )}
                                </div>
                              )
                            })}
                          </div>
                        ) : !activeSuggestionsLoading ? (
                          <div className="rounded-xl border border-dashed border-neutral-800 px-4 py-10 text-sm text-neutral-600">
                            {suggestedSubFilter === 'deferred'
                              ? 'No deferred suggestions.'
                              : 'No suggested links from current embeddings.'}
                          </div>
                        ) : null}
                          </>
                        )}

                        {/* ── Images sub-tab ── */}
                        {suggestedInnerTab === 'images' && (
                          <>
                            <div className="flex flex-wrap items-center justify-between gap-3">
                              <div className="min-w-[260px] flex-1">
                                <p className="text-[11px] uppercase tracking-wide text-amber-300/90">Similar images</p>
                                <p className="mt-1 text-xs text-neutral-500">
                                  Top images containing faces similar to {activePerson.name}. Select images to link.
                                </p>
                                <div className="mt-3 max-w-sm">
                                  <div className="mb-1 flex items-center justify-between text-[11px] text-neutral-400">
                                    <span>Similarity threshold</span>
                                    <span className="font-mono text-neutral-300">{similarityThreshold.toFixed(2)}</span>
                                  </div>
                                  <input
                                    type="range"
                                    min={0.1}
                                    max={0.95}
                                    step={0.01}
                                    value={similarityThreshold}
                                    onChange={(e) => setSimilarityThreshold(Number(e.target.value))}
                                    className="w-full accent-amber-500"
                                  />
                                </div>
                              </div>
                              {(personSimilarImages[expandedPersonId!] ?? []).length > 0 && (
                                <div className="flex flex-wrap items-center gap-2">
                                  <button
                                    type="button"
                                    onClick={() => {
                                      const allIds = (personSimilarImages[expandedPersonId!] ?? []).map((img) => img.best_occurrence_id)
                                      const allSelected = allIds.length > 0 && allIds.every((id) => selectedSimilarImageIds.includes(id))
                                      setSelectedSimilarImageIds(allSelected ? [] : allIds)
                                    }}
                                    disabled={confirmingSimilarLinks}
                                    className="rounded-lg border border-neutral-700 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-800 disabled:opacity-50"
                                  >
                                    {(personSimilarImages[expandedPersonId!] ?? []).length > 0
                                      && (personSimilarImages[expandedPersonId!] ?? []).every((img) => selectedSimilarImageIds.includes(img.best_occurrence_id))
                                      ? 'Clear selection'
                                      : 'Select all'}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => void handleConfirmSimilarLinks()}
                                    disabled={selectedSimilarImageIds.length === 0 || confirmingSimilarLinks}
                                    className="rounded-lg border border-emerald-700/60 bg-emerald-950/30 px-3 py-1.5 text-xs text-emerald-200 hover:bg-emerald-900/40 disabled:opacity-40"
                                  >
                                    {confirmingSimilarLinks
                                      ? 'Linking...'
                                      : `Confirm selected (${selectedSimilarImageIds.length})`}
                                  </button>
                                </div>
                              )}
                            </div>
                            {loadingPersonSimilarImagesId === expandedPersonId ? (
                              <div className="flex items-center gap-2 py-8 justify-center text-neutral-500 text-sm">
                                <span className="h-4 w-4 rounded-full border-2 border-neutral-600 border-t-neutral-300 animate-spin" />
                                Searching similar faces…
                              </div>
                            ) : (personSimilarImages[expandedPersonId!] ?? []).length > 0 ? (
                              <div className="grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-4">
                                {(personSimilarImages[expandedPersonId!] ?? []).map((img) => (
                                  <SimilarImageCard
                                    key={img.image_id}
                                    image={img}
                                    selected={selectedSimilarImageIds.includes(img.best_occurrence_id)}
                                    onToggleSelect={(occId) => {
                                      setSelectedSimilarImageIds((prev) =>
                                        prev.includes(occId)
                                          ? prev.filter((id) => id !== occId)
                                          : [...prev, occId]
                                      )
                                    }}
                                    onOpenLightbox={(fp, iid) => {
                                      setLightboxPath(fp)
                                      setLightboxImageId(iid)
                                    }}
                                  />
                                ))}
                              </div>
                            ) : (
                              <div className="rounded-xl border border-dashed border-neutral-800 px-4 py-10 text-sm text-neutral-600">
                                No similar images found for this person.
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    )}

                    {peopleStage === 'unlinked' && (
                      <div className="space-y-4">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <p className="text-[11px] uppercase tracking-wide text-blue-300/90">Unlinked clusters</p>
                            <p className="mt-1 text-xs text-neutral-500">
                              {unlinkedSubFilter === 'deferred'
                                ? `${deferredUnlinkedClusters.length} deferred cluster${deferredUnlinkedClusters.length !== 1 ? 's' : ''} parked for later.`
                                : <>Hunt for missed matches for {activePerson?.name ?? '…'}. Showing {visibleUnlinkedClusters.length}
                                  {activeUnlinkedClusters.length > unlinkedClusterTarget ? ` of ${activeUnlinkedClusters.length}` : ''} active clusters.</>
                              }
                            </p>
                          </div>
                          {/* Active / Deferred toggle */}
                          <div className="flex shrink-0 rounded-full border border-neutral-700 bg-neutral-900 p-0.5 text-xs">
                            <button
                              type="button"
                              onClick={() => setUnlinkedSubFilter('active')}
                              className={`rounded-full px-2.5 py-1 transition-colors ${
                                unlinkedSubFilter === 'active' ? 'bg-neutral-700 text-neutral-100' : 'text-neutral-400 hover:text-neutral-200'
                              }`}
                            >
                              Active <span className="text-neutral-500">{activeUnlinkedClusters.length > unlinkedClusterTarget ? `${unlinkedClusterTarget}+` : activeUnlinkedClusters.length}</span>
                            </button>
                            <button
                              type="button"
                              onClick={() => setUnlinkedSubFilter('deferred')}
                              className={`rounded-full px-2.5 py-1 transition-colors ${
                                unlinkedSubFilter === 'deferred' ? 'bg-neutral-700 text-neutral-100' : 'text-neutral-400 hover:text-neutral-200'
                              }`}
                            >
                              Deferred <span className="text-neutral-500">{deferredUnlinkedClusters.length}</span>
                            </button>
                          </div>
                        </div>

                        {/* Bulk restore for deferred view */}
                        {unlinkedSubFilter === 'deferred' && deferredUnlinkedClusters.length > 0 && (
                          <button
                            type="button"
                            onClick={async () => {
                              await window.api.undeferAllFaceClusters()
                              setDeferredClusterIds(new Set())
                              setUnlinkedSubFilter('active')
                            }}
                            className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                          >
                            ↩ Restore all deferred clusters
                          </button>
                        )}

                        {visibleUnlinkedClusters.length > 0 ? (
                          <div className="grid grid-cols-[repeat(auto-fill,minmax(170px,1fr))] gap-4">
                            {visibleUnlinkedClusters.map((cluster) => {
                              const key = `cluster:${cluster.cluster_id}`
                              const isSelected = expandedKey === key && peopleStage === 'inspector'
                              const isDeferred = cluster.cluster_id !== null && deferredClusterIds.has(cluster.cluster_id)
                              return (
                                <div
                                  key={`unlinked:${cluster.cluster_id}`}
                                  className={`group relative rounded-xl border transition-colors ${
                                    isSelected
                                      ? 'border-blue-700/60 bg-blue-900/20'
                                      : 'border-neutral-800 bg-neutral-900/60 hover:border-neutral-700 hover:bg-neutral-800/40'
                                  }`}
                                >
                                  <button
                                    type="button"
                                    onClick={() => {
                                      if (cluster.cluster_id != null) {
                                        void openClusterInspector(key, cluster, 'unlinked')
                                      }
                                    }}
                                    className="block w-full text-left"
                                  >
                                    <div className="aspect-square w-full overflow-hidden rounded-t-xl bg-neutral-800">
                                      {cluster.representative_id != null ? (
                                        <FaceCropThumbnail occurrenceId={cluster.representative_id} size="fill" />
                                      ) : (
                                        <div className="flex h-full w-full items-center justify-center">
                                          <svg className="h-10 w-10 text-neutral-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                                          </svg>
                                        </div>
                                      )}
                                    </div>
                                    <div className="px-3 py-2">
                                      <p className="truncate text-sm text-neutral-100">Unknown</p>
                                      <p className="mt-1 text-[11px] text-neutral-500">{cluster.face_count} faces</p>
                                    </div>
                                  </button>
                                  {/* Defer / Restore + Link buttons */}
                                  {cluster.cluster_id !== null && (
                                    <div
                                      className="absolute right-2 top-2 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100"
                                      onClick={(e) => e.stopPropagation()}
                                    >
                                      {isDeferred ? (
                                        <button
                                          type="button"
                                          title="Restore cluster"
                                          onClick={async () => {
                                            const cid = cluster.cluster_id!
                                            setDeferredClusterIds((prev) => { const next = new Set(prev); next.delete(cid); return next })
                                            await window.api.undeferFaceCluster(cid)
                                          }}
                                          className="rounded-full bg-black/60 p-1.5 text-blue-400 hover:bg-black/80 hover:text-blue-300 transition-colors"
                                        >
                                          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M9 15L3 9m0 0l6-6M3 9h12a6 6 0 010 12h-3" />
                                          </svg>
                                        </button>
                                      ) : (
                                        <button
                                          type="button"
                                          title="Skip — defer for later"
                                          onClick={async () => {
                                            const cid = cluster.cluster_id!
                                            setDeferredClusterIds((prev) => new Set([...prev, cid]))
                                            await window.api.deferFaceCluster(cid)
                                          }}
                                          className="rounded-full bg-black/60 p-1.5 text-neutral-400 hover:bg-black/80 hover:text-amber-400 transition-colors"
                                        >
                                          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                          </svg>
                                        </button>
                                      )}
                                      <div className="relative">
                                        {renderLinkDropdown(cluster.cluster_id)}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )
                            })}
                          </div>
                        ) : (
                          <div className="rounded-xl border border-dashed border-neutral-800 px-4 py-10 text-sm text-neutral-600">
                            {unlinkedSubFilter === 'deferred'
                              ? 'No deferred clusters.'
                              : 'No unlinked clusters available.'}
                          </div>
                        )}
                      </div>
                    )}

                    {peopleStage === 'inspector' && (
                      <div className="space-y-4">
                        {inspectorCluster && expandedKey ? (
                          <>
                            <div className="flex flex-wrap items-center justify-between gap-3">
                              <div>
                                <p className="text-[11px] uppercase tracking-wide text-violet-300/90">Inspector</p>
                                <h4 className="mt-1 text-base font-semibold text-neutral-100">
                                  {inspectorCluster.display_name || 'Unknown'}
                                </h4>
                                <p className="mt-1 text-xs text-neutral-500">
                                  {inspectorCluster.face_count} faces · {inspectorCluster.image_count} images
                                </p>
                              </div>
                              <div className="flex flex-wrap items-center gap-3">
                                {renderRelinkButton(inspectorCluster)}
                                {inspectorCluster.cluster_id != null && (
                                  <div className="relative">
                                    {renderLinkDropdown(inspectorCluster.cluster_id)}
                                  </div>
                                )}
                                {inspectorCluster.person_id != null && inspectorCluster.cluster_id != null && (
                                  <button
                                    type="button"
                                    onClick={() => void handleUnlinkCluster(inspectorCluster.cluster_id!)}
                                    className="rounded-lg border border-red-800/60 bg-red-950/20 px-3 py-1.5 text-xs text-red-200 hover:bg-red-900/30"
                                  >
                                    Unlink
                                  </button>
                                )}
                                {inspectorCluster.cluster_id != null && !inspectorCluster.person_id && (
                                  (() => {
                                    const cid = inspectorCluster.cluster_id!
                                    const isDeferred = deferredClusterIds.has(cid)
                                    return isDeferred ? (
                                      <button
                                        type="button"
                                        onClick={async () => {
                                          setDeferredClusterIds((prev) => { const next = new Set(prev); next.delete(cid); return next })
                                          await window.api.undeferFaceCluster(cid)
                                        }}
                                        className="rounded-lg border border-blue-800/60 bg-blue-950/20 px-3 py-1.5 text-xs text-blue-200 hover:bg-blue-900/30"
                                      >
                                        Restore
                                      </button>
                                    ) : (
                                      <button
                                        type="button"
                                        onClick={async () => {
                                          setDeferredClusterIds((prev) => new Set([...prev, cid]))
                                          await window.api.deferFaceCluster(cid)
                                          setPeopleStage(inspectorReturnStage)
                                        }}
                                        className="rounded-lg border border-amber-800/60 bg-amber-950/20 px-3 py-1.5 text-xs text-amber-200 hover:bg-amber-900/30"
                                      >
                                        Defer
                                      </button>
                                    )
                                  })()
                                )}
                                {inspectorCluster.cluster_id != null && (inspectorCluster.face_count ?? 0) >= 2 && (
                                  <button
                                    type="button"
                                    onClick={async () => {
                                      const cid = inspectorCluster.cluster_id!
                                      const result = await window.api.splitCluster(cid)
                                      if (result.error) {
                                        setError(result.error)
                                        return
                                      }
                                      if (result.split_count <= 1) {
                                        setError(null)
                                        // Inform user cluster is pure
                                        return
                                      }
                                      setError(null)
                                      await loadData()
                                      if (expandedPersonId != null) {
                                        await loadPersonClusters(expandedPersonId, true)
                                        await loadPersonLinkSuggestions(expandedPersonId, true)
                                      }
                                      setPeopleStage(inspectorReturnStage)
                                    }}
                                    className="rounded-lg border border-purple-800/60 bg-purple-950/20 px-3 py-1.5 text-xs text-purple-200 hover:bg-purple-900/30"
                                    title="Split this cluster into sub-clusters using a tighter similarity threshold"
                                  >
                                    ✂ Split
                                  </button>
                                )}
                                <span className="mx-1 text-neutral-700">|</span>
                                <button
                                  type="button"
                                  onClick={() => setPeopleStage(inspectorReturnStage)}
                                  className="rounded-lg border border-neutral-700 px-3 py-1.5 text-xs text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200 transition-colors"
                                >
                                  ← Back to {inspectorReturnStage}
                                </button>
                              </div>
                            </div>

                            {/* Similarity suggestions (async-loaded) */}
                            {inspectorCluster.cluster_id != null && !inspectorCluster.person_id && (
                              inspectorSuggestionsLoading ? (
                                <div className="flex items-center gap-2 rounded-lg border border-neutral-800 bg-neutral-900/40 px-4 py-2.5 text-xs text-neutral-500">
                                  <span className="w-3.5 h-3.5 border-2 border-neutral-700 border-t-neutral-400 rounded-full animate-spin" />
                                  Checking for similar identities…
                                </div>
                              ) : inspectorSuggestions.length > 0 ? (
                                <div className="rounded-lg border border-cyan-800/40 bg-cyan-950/20 px-4 py-3">
                                  <p className="text-[11px] uppercase tracking-wide text-cyan-400/80 mb-2">Similar identities found</p>
                                  <div className="flex flex-wrap gap-2">
                                    {inspectorSuggestions.map((s) => (
                                      <button
                                        key={`${s.target_type}:${s.person_id ?? s.label}`}
                                        type="button"
                                        onClick={async () => {
                                          if (s.target_type === 'person' && s.person_id != null && inspectorCluster.cluster_id != null) {
                                            await window.api.linkClusterToPerson(inspectorCluster.cluster_id, s.person_id)
                                            setInspectorSuggestions([])
                                            await loadData()
                                            await loadPersonClusters(s.person_id, true)
                                          } else if (s.target_type === 'alias' && inspectorCluster.cluster_id != null) {
                                            void openRelinkDialog(inspectorCluster)
                                          }
                                        }}
                                        className="flex items-center gap-2 rounded-lg border border-neutral-700 bg-neutral-900/80 px-3 py-1.5 text-xs hover:border-cyan-700 hover:bg-cyan-950/30 transition-colors"
                                      >
                                        {s.representative_id != null && (
                                          <div className="w-7 h-7 shrink-0">
                                            <FaceCropThumbnail occurrenceId={s.representative_id} size="fill" />
                                          </div>
                                        )}
                                        <span className="text-neutral-200">{s.label}</span>
                                        <span className="text-neutral-500">{(s.score * 100).toFixed(0)}%</span>
                                      </button>
                                    ))}
                                  </div>
                                </div>
                              ) : null
                            )}

                            <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-4">
                              <div className="flex flex-col gap-4 xl:flex-row">
                                <div className="xl:w-72 shrink-0">
                                  <div className="aspect-square overflow-hidden rounded-xl bg-neutral-800">
                                    {inspectorCluster.representative_id != null ? (
                                      <FaceCropThumbnail occurrenceId={inspectorCluster.representative_id} size="fill" />
                                    ) : (
                                      <div className="flex h-full w-full items-center justify-center">
                                        <svg className="h-10 w-10 text-neutral-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
                                          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                                        </svg>
                                      </div>
                                    )}
                                  </div>
                                </div>
                                <div className="min-w-0 flex-1">
                                  <div className="rounded-lg border border-neutral-800 bg-neutral-950/40 px-3 py-2 text-xs text-neutral-500">
                                    Click any thumbnail below to open the source image in the lightbox.
                                  </div>
                                  <div className="mt-4 rounded-xl border border-neutral-800 bg-neutral-950/30">
                                    <div className="border-b border-neutral-800/70 px-4 py-2 text-[11px] uppercase tracking-wide text-neutral-500">
                                      Face occurrences
                                    </div>
                                    <div className="max-h-[50vh] overflow-y-auto py-3">
                                      {renderExpandedDetail(expandedKey)}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </>
                        ) : (
                          <div className="rounded-xl border border-dashed border-neutral-800 px-4 py-10 text-sm text-neutral-600">
                            Pick a linked, suggested, or unlinked cluster to inspect it here.
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="flex flex-1 items-center justify-center px-6 text-center">
                  <div className="max-w-lg rounded-2xl border border-dashed border-neutral-800 bg-neutral-900/40 px-6 py-8">
                    <p className="text-sm font-medium text-neutral-200">Choose a person to open the review workspace.</p>
                    <p className="mt-2 text-xs text-neutral-500">
                      You will land in Linked first, then move through Suggested, Unlinked, or Inspector without stacked drawers or floating review panels.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Delete person confirmation dialog */}
      {deletingPersonId !== null && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-neutral-900 border border-neutral-700 rounded-lg shadow-xl p-6 max-w-sm mx-4">
            <h3 className="text-sm font-semibold text-neutral-200 mb-2">Delete Person?</h3>
            <p className="text-xs text-neutral-400 mb-4">
              This will remove the person and unlink all their clusters. The face data itself is not deleted.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setDeletingPersonId(null)}
                className="px-3 py-1.5 text-xs rounded-md bg-neutral-800 text-neutral-300 hover:bg-neutral-700 transition-colors border border-neutral-600"
              >
                Cancel
              </button>
              <button
                onClick={() => deletingPersonId && handleDeletePerson(deletingPersonId)}
                className="px-3 py-1.5 text-xs rounded-md bg-red-700 text-white hover:bg-red-600 transition-colors"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Face list — legacy mode */}
      {!hasOccurrences && legacyFaces.length > 0 && (
        <div className="flex-1 overflow-y-auto">
          {legacyFaces.map((face) => {
            const key = `legacy:${face.canonical_name}`
            const isExpanded = expandedKey === key
            const isEditing = editingKey === key
            const displayLabel = face.display_name || face.canonical_name

            return (
              <div key={key} className="border-b border-neutral-800/60">
                <div className="flex items-center gap-3 px-5 py-2.5 hover:bg-neutral-800/40 transition-colors">
                  <button
                    onClick={() => toggleExpand(key, null, face)}
                    className="text-neutral-500 hover:text-neutral-300 transition-colors shrink-0"
                    title={isExpanded ? 'Collapse' : 'Expand to see images'}
                  >
                    <svg
                      className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M8.25 4.5l7.5 7.5-7.5 7.5"
                      />
                    </svg>
                  </button>

                  <div className="flex-1 min-w-0">
                    {isEditing ? (
                      renderEditingField(key, face.canonical_name)
                    ) : (
                      <div>
                        <span className="text-sm text-neutral-100 font-medium truncate block">
                          {displayLabel}
                        </span>
                        {face.display_name && (
                          <span className="text-xs text-neutral-500 truncate block">
                            {face.canonical_name}
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  <span className="text-xs text-neutral-400 bg-neutral-800 px-2 py-0.5 rounded-full shrink-0">
                    {face.image_count} {face.image_count === 1 ? 'image' : 'images'}
                  </span>

                  {!isEditing &&
                    renderEditButton(key, face.display_name, face.canonical_name)}
                </div>

                {isExpanded && renderExpandedDetail(key)}
              </div>
            )
          })}
        </div>
      )}

      {/* In-app lightbox */}
      {lightboxPath && (
        <FaceImageLightbox
          filePath={lightboxPath}
          imageId={lightboxImageId}
          onClose={() => { setLightboxPath(null); setLightboxImageId(undefined) }}
        />
      )}
    </div>
  )
}
