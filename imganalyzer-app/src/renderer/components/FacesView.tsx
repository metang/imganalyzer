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
} from '../global'

// ── Thumbnail cache & batch fetcher ───────────────────────────────────────────

const THUMB_CACHE_MAX = 2000
const thumbCache = new Map<number, string>() // occurrence_id → base64 data URI
const pendingIds = new Set<number>()
const pendingCallbacks = new Map<number, Array<(src: string | null) => void>>()
let batchTimer: ReturnType<typeof setTimeout> | null = null

const CLUSTER_PAGE_SIZE = 200
const UNLINKED_CLUSTER_TARGET = 60

function countUnlinkedClusters(clusters: FaceCluster[]): number {
  return clusters.reduce(
    (count, cluster) => (cluster.cluster_id !== null && !cluster.person_id ? count + 1 : count),
    0,
  )
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

async function flushBatch(): Promise<void> {
  batchTimer = null
  if (pendingIds.size === 0) return

  const ids = [...pendingIds]
  pendingIds.clear()

  // Process in chunks of 50 to avoid overly large RPC payloads
  const CHUNK = 50
  for (let i = 0; i < ids.length; i += CHUNK) {
    const chunk = ids.slice(i, i + CHUNK)
    try {
      const result = await window.api.getFaceCropBatch(chunk)
      for (const id of chunk) {
        const b64 = result.thumbnails?.[String(id)]
        const dataUri = b64 ? `data:image/jpeg;base64,${b64}` : null
        if (dataUri) {
          // Evict oldest if cache full
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
      // On error, notify all callbacks with null
      for (const id of chunk) {
        const cbs = pendingCallbacks.get(id) ?? []
        pendingCallbacks.delete(id)
        for (const cb of cbs) cb(null)
      }
    }
  }
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

  useEffect(() => {
    if (src || requested.current) return
    requested.current = true
    requestThumbnail(occurrenceId, (dataUri) => {
      if (dataUri) {
        setSrc(dataUri)
      } else {
        setFailed(true)
      }
    })
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

  if (failed) {
    return (
      <div
        className={`${sizeClass} rounded bg-neutral-800 flex items-center justify-center ${shrink}`}
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
        className={`${sizeClass} rounded bg-neutral-800 animate-pulse ${shrink}`}
      />
    )
  }

  return (
    <img
      src={src}
      alt=""
      className={`${sizeClass} rounded object-cover ${shrink}`}
      draggable={false}
    />
  )
})

// ── Full image thumbnail (lazy-loaded, for legacy mode) ───────────────────────

function ImageThumbnail({ filePath }: { filePath: string }) {
  const [src, setSrc] = useState<string | null>(null)
  const requested = useRef(false)

  useEffect(() => {
    if (requested.current) return
    requested.current = true
    window.api
      .getThumbnail(filePath)
      .then((data) => setSrc(`data:image/jpeg;base64,${data}`))
      .catch(() => setSrc(null))
  }, [filePath])

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

// ── Inline image lightbox (for viewing source image in-app) ───────────────────

function FaceImageLightbox({
  filePath,
  onClose,
}: {
  filePath: string
  onClose: () => void
}) {
  const [src, setSrc] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    window.api.getFullImage(filePath).then((url) => {
      if (!cancelled) setSrc(url)
    })
    return () => { cancelled = true }
  }, [filePath])

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/90"
      onClick={onClose}
    >
      {/* Close button */}
      <button
        onClick={onClose}
        className="absolute top-4 right-4 z-10 text-neutral-400 hover:text-white transition-colors"
        title="Close (Esc)"
      >
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>

      {/* File name */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 bg-black/70 px-3 py-1.5 rounded-lg">
        <p className="text-xs text-neutral-300 truncate max-w-md">
          {filePath.split(/[/\\]/).pop()}
        </p>
      </div>

      {/* Image */}
      {src ? (
        <img
          src={src}
          alt=""
          className="max-w-[90vw] max-h-[90vh] object-contain rounded shadow-2xl"
          onClick={(e) => e.stopPropagation()}
          draggable={false}
        />
      ) : (
        <div className="flex items-center gap-2 text-neutral-400">
          <span className="w-5 h-5 border-2 border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
          Loading...
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

// ── Main FacesView component ──────────────────────────────────────────────────

export function FacesView() {
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

  // Rebuild state
  const [rebuilding, setRebuilding] = useState(false)
  const [showRebuildConfirm, setShowRebuildConfirm] = useState(false)
  const [rebuildConfirmText, setRebuildConfirmText] = useState('')

  // Lightbox state (in-app image viewer)
  const [lightboxPath, setLightboxPath] = useState<string | null>(null)

  // View mode: clusters or people
  type ViewMode = 'clusters' | 'people'
  const [viewMode, setViewMode] = useState<ViewMode>('clusters')

  // Person state
  const [persons, setPersons] = useState<FacePerson[]>([])
  const [personClusters, setPersonClusters] = useState<Record<number, PersonCluster[]>>({})
  const [personLinkSuggestions, setPersonLinkSuggestions] = useState<Record<number, PersonLinkSuggestion[]>>({})
  const [loadingPersonLinkSuggestionsId, setLoadingPersonLinkSuggestionsId] = useState<number | null>(null)
  const [selectedSuggestedClusterIds, setSelectedSuggestedClusterIds] = useState<number[]>([])
  const [confirmingSuggestedLinks, setConfirmingSuggestedLinks] = useState(false)
  const [expandedPersonId, setExpandedPersonId] = useState<number | null>(null)

  // Person editing
  const [editingPersonId, setEditingPersonId] = useState<number | null>(null)
  const [personEditValue, setPersonEditValue] = useState('')
  const personEditRef = useRef<HTMLInputElement>(null)

  // Link-to-person dropdown
  const [linkingClusterId, setLinkingClusterId] = useState<number | null>(null)
  const [showCreatePerson, setShowCreatePerson] = useState(false)
  const [newPersonName, setNewPersonName] = useState('')
  const [linkSearchFilter, setLinkSearchFilter] = useState('')
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
        const shouldEnsureUnlinked = !personsResult.error && personsResult.persons.length > 0

        if (shouldEnsureUnlinked) {
          let offset = loadedClusters.length
          let unlinkedCount = countUnlinkedClusters(loadedClusters)

          while (unlinkedCount < UNLINKED_CLUSTER_TARGET && offset < loadedTotalCount) {
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
            unlinkedCount = countUnlinkedClusters(loadedClusters)
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

  useEffect(() => {
    loadData()
  }, [loadData])

  useEffect(() => {
    setSelectedSuggestedClusterIds([])
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

  const handleRebuild = useCallback(async () => {
    setShowRebuildConfirm(false)
    setRebuildConfirmText('')
    setRebuilding(true)
    try {
      const result = await window.api.rebuildFaces()
      if (result.error) {
        setError(result.error)
        return
      }
      if (result.enqueued === 0) {
        setError('No images found to rebuild faces for.')
        return
      }
      // Auto-start the batch run
      setError(null)
      try {
        await window.api.batchResume()
        setError(`✓ ${result.enqueued} face jobs enqueued and running. Switch to the Batch tab to monitor progress.`)
      } catch {
        setError(`✓ ${result.enqueued} face jobs enqueued. Go to the Batch tab and click Resume to start processing.`)
      }
    } catch (err) {
      setError(String(err))
    } finally {
      setRebuilding(false)
    }
  }, [])

  // ── Person actions ──────────────────────────────────────────────────────

  const handleLinkCluster = useCallback(
    async (clusterId: number, personId: number) => {
      try {
        await window.api.linkClusterToPerson(clusterId, personId)
        setLinkingClusterId(null)
        await loadData()
      } catch (err) {
        setError(String(err))
      }
    },
    [loadData]
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
        setShowCreatePerson(false)
        setNewPersonName('')
        setLinkingClusterId(null)
        await loadData()
      } catch (err) {
        setError(String(err))
      }
    },
    [newPersonName, loadData]
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
  }, [applyClusterRelink, closeRelinkDialog, loadData, relinkSelection, relinkingCluster, unlinkPersonOnAliasRelink])

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
      setError(null)
      closeRelinkDialog()
      await loadData()
    } catch (err) {
      setError(String(err))
    } finally {
      setRelinkSubmitting(false)
    }
  }, [applyClusterRelink, closeRelinkDialog, loadData, relinkSearch, relinkingCluster])

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

      setLoadingPersonLinkSuggestionsId(personId)
      try {
        const result = await window.api.getPersonLinkSuggestions(personId, 12)
        if (result.error) {
          setError(result.error)
          setPersonLinkSuggestions((prev) => ({ ...prev, [personId]: [] }))
          return
        }
        setPersonLinkSuggestions((prev) => ({ ...prev, [personId]: result.suggestions }))
      } catch (err) {
        setError(String(err))
        setPersonLinkSuggestions((prev) => ({ ...prev, [personId]: [] }))
      } finally {
        setLoadingPersonLinkSuggestionsId((current) => (current === personId ? null : current))
      }
    },
    [personLinkSuggestions]
  )

  const togglePersonExpand = useCallback(
    async (personId: number) => {
      if (expandedPersonId === personId) {
        setExpandedPersonId(null)
        return
      }
      setExpandedPersonId(personId)
      void loadPersonLinkSuggestions(personId)

      // Load clusters for this person if not cached
      const pClusters = personClusters[personId]
      if (pClusters) {
        return
      }

      const loadingKey = `person-clusters:${personId}`
      setLoadingDetail(loadingKey)
      try {
        const result = await window.api.getPersonClusters(personId)
        if (!result.error) {
          setPersonClusters((prev) => ({ ...prev, [personId]: result.clusters }))
        }
      } catch {
        // silently ignore
      } finally {
        setLoadingDetail(null)
      }
    },
    [expandedPersonId, loadPersonLinkSuggestions, personClusters]
  )

  const toggleSuggestedClusterSelection = useCallback((clusterId: number) => {
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
      setError(null)
      await loadData()

      const personClusterResult = await window.api.getPersonClusters(expandedPersonId)
      if (!personClusterResult.error) {
        setPersonClusters((prev) => ({ ...prev, [expandedPersonId]: personClusterResult.clusters }))
      }
      await loadPersonLinkSuggestions(expandedPersonId, true)
    } catch (err) {
      setError(String(err))
    } finally {
      setConfirmingSuggestedLinks(false)
    }
  }, [expandedPersonId, loadData, loadPersonLinkSuggestions, personLinkSuggestions, selectedSuggestedClusterIds])

  // ── Expand / collapse ─────────────────────────────────────────────────────

  const toggleExpand = useCallback(
    async (key: string, cluster: FaceCluster | null, legacyFace: FaceSummary | null) => {
      if (expandedKey === key) {
        setExpandedKey(null)
        return
      }

      setExpandedKey(key)

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
      } else if (legacyFace && !legacyImages[key]) {
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
    [expandedKey, clusterOccurrences, legacyImages]
  )

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
  const visibleUnlinkedClusters = useMemo(
    () => unlinkedClusters.slice(0, UNLINKED_CLUSTER_TARGET),
    [unlinkedClusters],
  )

  const filteredPersons = useMemo(() => {
    const lowerFilter = linkSearchFilter.toLowerCase()
    return lowerFilter
      ? persons.filter((p) => coerceText(p.name).toLowerCase().includes(lowerFilter))
      : persons
  }, [persons, linkSearchFilter])

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
        className="text-xs text-violet-400/80 hover:text-violet-300 transition-colors shrink-0"
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
                onClick={() => setLightboxPath(occ.file_path)}
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
                <ImageThumbnail filePath={img.file_path} />
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
            onClick={() => { setRebuildConfirmText(''); setShowRebuildConfirm(true) }}
            disabled={rebuilding}
            className="px-3 py-1 text-xs rounded-md bg-amber-900/50 text-amber-300 hover:bg-amber-800/60
                       disabled:opacity-50 transition-colors border border-amber-700/40"
            title="Re-enqueue face analysis for all images (regenerates thumbnails)"
          >
            {rebuilding ? (
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 border border-amber-600 border-t-amber-300 rounded-full animate-spin" />
                Rebuilding...
              </span>
            ) : (
              'Rebuild Faces'
            )}
          </button>
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

      {/* Rebuild confirmation dialog */}
      {showRebuildConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-neutral-900 border border-neutral-700 rounded-lg shadow-xl p-6 max-w-sm mx-4">
            <h3 className="text-sm font-semibold text-neutral-200 mb-2">Rebuild Face Analysis?</h3>
            <p className="text-xs text-neutral-400 mb-3">
              This will re-enqueue face detection jobs for <strong>all images</strong>.
              Existing face data will be replaced when the batch runs.
            </p>
            <p className="text-xs text-neutral-400 mb-2">
              Type <code className="px-1.5 py-0.5 rounded bg-neutral-800 text-amber-400 font-mono">REBUILD</code> to confirm:
            </p>
            <input
              value={rebuildConfirmText}
              onChange={(e) => setRebuildConfirmText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && rebuildConfirmText === 'REBUILD') handleRebuild()
                if (e.key === 'Escape') { setShowRebuildConfirm(false); setRebuildConfirmText('') }
              }}
              placeholder="REBUILD"
              className="w-full px-3 py-1.5 text-sm rounded-md bg-neutral-800 border border-neutral-600
                         text-neutral-100 placeholder-neutral-600 font-mono outline-none focus:border-amber-500 mb-4"
              autoFocus
            />
            <div className="flex justify-end gap-2">
              <button
                onClick={() => { setShowRebuildConfirm(false); setRebuildConfirmText('') }}
                className="px-3 py-1.5 text-xs rounded-md bg-neutral-800 text-neutral-300 hover:bg-neutral-700
                           transition-colors border border-neutral-600"
              >
                Cancel
              </button>
              <button
                onClick={handleRebuild}
                disabled={rebuildConfirmText !== 'REBUILD'}
                className="px-3 py-1.5 text-xs rounded-md bg-amber-700 text-white hover:bg-amber-600
                           transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Rebuild All Faces
              </button>
            </div>
          </div>
        </div>
      )}

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
        <div className="flex-1 flex flex-col min-h-0">
        <div className="flex-1 overflow-y-auto">
          {/* Persons grid */}
          {persons.length > 0 && (
            <div className="p-4">
              <div className="grid grid-cols-[repeat(auto-fill,minmax(120px,1fr))] gap-3">
                {persons.map((person) => {
                  const isExpPerson = expandedPersonId === person.id

                  return (
                    <div
                      key={`person:${person.id}`}
                      className={`group relative rounded-lg border transition-colors cursor-pointer ${
                        isExpPerson
                          ? 'border-cyan-700/60 bg-cyan-900/20'
                          : 'border-neutral-800 hover:border-neutral-700 bg-neutral-900/50 hover:bg-neutral-800/40'
                      }`}
                      onClick={() => togglePersonExpand(person.id)}
                    >
                      {/* Large representative thumbnail */}
                      <div className="aspect-square w-full overflow-hidden rounded-t-lg bg-neutral-800">
                        {person.representative_id != null ? (
                          <FaceCropThumbnail occurrenceId={person.representative_id} size="fill" />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            <svg className="w-10 h-10 text-neutral-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                            </svg>
                          </div>
                        )}
                      </div>

                      {/* Name + stats */}
                      <div className="px-2 py-1.5">
                        {editingPersonId === person.id ? (
                          <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                            <input
                              ref={personEditRef}
                              value={personEditValue}
                              onChange={(e) => setPersonEditValue(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') handleRenamePerson(person.id)
                                if (e.key === 'Escape') setEditingPersonId(null)
                              }}
                              className="w-full px-1.5 py-0.5 text-xs rounded bg-neutral-800 border border-neutral-600
                                         text-neutral-100 outline-none focus:border-blue-500"
                              autoFocus
                            />
                          </div>
                        ) : (
                          <p className="text-xs text-neutral-100 font-medium truncate">{person.name}</p>
                        )}
                        <p className="text-[10px] text-neutral-500 mt-0.5">
                          {person.face_count} faces · {person.cluster_count} cl.
                        </p>
                      </div>

                      {/* Action buttons (hover) */}
                      <div className="absolute top-1 right-1 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={(e) => e.stopPropagation()}>
                        <button
                          onClick={() => {
                            setEditingPersonId(person.id)
                            setPersonEditValue(person.name)
                            setTimeout(() => personEditRef.current?.focus(), 0)
                          }}
                          className="p-1 rounded bg-black/60 text-neutral-400 hover:text-white transition-colors"
                          title="Rename"
                        >
                          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931z" />
                          </svg>
                        </button>
                        <button
                          onClick={() => setDeletingPersonId(person.id)}
                          className="p-1 rounded bg-black/60 text-neutral-400 hover:text-red-400 transition-colors"
                          title="Delete"
                        >
                          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}



          {/* Unlinked clusters */}
          {visibleUnlinkedClusters.length > 0 && (
            <>
              <div className="px-5 py-2 text-xs text-neutral-500 font-medium uppercase tracking-wide bg-neutral-900/80 border-b border-neutral-800/60 sticky top-0">
                Unlinked Clusters ({visibleUnlinkedClusters.length}{unlinkedClusters.length > visibleUnlinkedClusters.length ? ` of ${unlinkedClusters.length}` : ''})
              </div>
              <div className="p-4">
                <div className="grid grid-cols-[repeat(auto-fill,minmax(120px,1fr))] gap-3">
                  {visibleUnlinkedClusters.map((cluster) => {
                    const displayLabel = 'Unknown'
                    const isSelected = expandedKey === `cluster:${cluster.cluster_id}`

                    return (
                      <div
                        key={`unlinked:${cluster.cluster_id}`}
                        className={`group relative rounded-lg border transition-colors cursor-pointer ${
                          isSelected
                            ? 'border-blue-700/60 bg-blue-900/20'
                            : 'border-neutral-800 hover:border-neutral-700 bg-neutral-900/50 hover:bg-neutral-800/40'
                        }`}
                        onClick={() => {
                          if (cluster.cluster_id !== null) {
                            const key = `cluster:${cluster.cluster_id}`
                            toggleExpand(key, cluster, null)
                          }
                        }}
                      >
                        <div className="aspect-square w-full overflow-hidden rounded-t-lg bg-neutral-800">
                          {cluster.representative_id != null ? (
                            <FaceCropThumbnail occurrenceId={cluster.representative_id} size="fill" />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center">
                              <svg className="w-10 h-10 text-neutral-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                              </svg>
                            </div>
                          )}
                        </div>
                        <div className="px-2 py-1.5">
                          <p className="text-xs text-neutral-100 truncate">{displayLabel}</p>
                          <p className="text-[10px] text-neutral-500 mt-0.5">{cluster.face_count} faces</p>
                        </div>
                        {/* Link button (hover) */}
                        {cluster.cluster_id !== null && (
                          <div className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity"
                            onClick={(e) => e.stopPropagation()}>
                            <div className="relative">
                              {renderLinkDropdown(cluster.cluster_id)}
                            </div>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            </>
          )}
        </div>

        {/* Bottom drawer for expanded person or cluster details */}
        {expandedPersonId !== null && (() => {
          const pClusters = personClusters[expandedPersonId] ?? []
          const suggestedClusters = personLinkSuggestions[expandedPersonId] ?? []
          const person = persons.find((p) => p.id === expandedPersonId)
          const isLoading = loadingDetail === `person-clusters:${expandedPersonId}`
          const isLoadingSuggestions = loadingPersonLinkSuggestionsId === expandedPersonId
          const selectedSuggestedCount = suggestedClusters.reduce(
            (count, suggestion) =>
              selectedSuggestedClusterIds.includes(suggestion.cluster_id) ? count + 1 : count,
            0,
          )
          const allSuggestedSelected =
            suggestedClusters.length > 0
            && selectedSuggestedCount === suggestedClusters.length

          return (
            <div className="shrink-0 border-t border-cyan-800/50 bg-neutral-900 max-h-[40vh] overflow-y-auto">
              <div className="px-4 py-2 border-b border-neutral-800/60 flex items-center justify-between sticky top-0 bg-neutral-900 z-10">
                <span className="text-xs text-cyan-300 font-medium">
                  {person?.name} — {pClusters.length} linked
                  {suggestedClusters.length > 0 && ` · ${suggestedClusters.length} suggested`}
                </span>
                <div className="flex items-center gap-2">
                  <span className="hidden sm:inline-flex px-1.5 py-0.5 rounded border border-cyan-700/50 bg-cyan-950/25 text-[10px] text-cyan-200">
                    Linked
                  </span>
                  <span className="hidden sm:inline-flex px-1.5 py-0.5 rounded border border-amber-700/50 bg-amber-950/20 text-[10px] text-amber-200">
                    Suggested
                  </span>
                  <button onClick={() => setExpandedPersonId(null)} className="text-neutral-500 hover:text-neutral-300 text-xs">Close</button>
                </div>
              </div>

              {isLoading && (
                <div className="px-4 py-3 flex items-center gap-2 text-xs text-neutral-500">
                  <span className="w-3 h-3 border border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
                  Loading clusters...
                </div>
              )}

              <div className="p-4 space-y-4">
                <div>
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <p className="text-[11px] uppercase tracking-wide text-amber-300/90">
                      Suggested links
                    </p>
                    <div className="flex items-center gap-2">
                      {isLoadingSuggestions && (
                        <span className="inline-flex items-center gap-1 text-[11px] text-neutral-500">
                          <span className="w-3 h-3 border border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
                          Scoring...
                        </span>
                      )}
                      {suggestedClusters.length > 0 && (
                        <>
                          <button
                            onClick={() => {
                              if (allSuggestedSelected) {
                                setSelectedSuggestedClusterIds([])
                                return
                              }
                              setSelectedSuggestedClusterIds(
                                suggestedClusters.map((suggestion) => suggestion.cluster_id)
                              )
                            }}
                            disabled={confirmingSuggestedLinks}
                            className="px-2 py-1 text-[11px] rounded border border-neutral-700 text-neutral-300 hover:bg-neutral-800 disabled:opacity-50"
                          >
                            {allSuggestedSelected ? 'Clear' : 'Select all'}
                          </button>
                          <button
                            onClick={() => void handleConfirmSuggestedLinks()}
                            disabled={selectedSuggestedCount === 0 || confirmingSuggestedLinks}
                            className="px-2 py-1 text-[11px] rounded border border-emerald-700/60 bg-emerald-950/30 text-emerald-200 hover:bg-emerald-900/40 disabled:opacity-40"
                          >
                            {confirmingSuggestedLinks
                              ? 'Linking...'
                              : `Confirm selected (${selectedSuggestedCount})`}
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                  {suggestedClusters.length > 0 ? (
                    <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-3">
                      {suggestedClusters.map((suggestion) => {
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
                        const isSelected = expandedKey === key
                        const isBatchSelected = selectedSuggestedClusterIds.includes(
                          suggestion.cluster_id
                        )

                        return (
                          <div
                            key={`suggested:${suggestion.cluster_id}`}
                            className={`relative rounded-lg border transition-colors cursor-pointer ${
                              isSelected
                                ? 'border-amber-600/70 bg-amber-900/20'
                                : 'border-amber-800/50 bg-amber-950/10 hover:bg-amber-900/15'
                            }`}
                            onClick={() => toggleExpand(key, sourceCluster, null)}
                          >
                            <div className="absolute top-1 left-1 rounded px-1.5 py-0.5 text-[10px] text-amber-100 bg-amber-900/60 border border-amber-700/60 z-10">
                              Suggested
                            </div>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                toggleSuggestedClusterSelection(suggestion.cluster_id)
                              }}
                              className={`absolute top-1 right-1 z-10 rounded px-1.5 py-0.5 text-[10px] border ${
                                isBatchSelected
                                  ? 'border-emerald-500/70 bg-emerald-900/70 text-emerald-100'
                                  : 'border-neutral-600 bg-black/55 text-neutral-200'
                              }`}
                            >
                              {isBatchSelected ? 'Selected' : 'Select'}
                            </button>
                            <div className="aspect-square w-full overflow-hidden rounded-t-lg bg-neutral-800">
                              {suggestion.representative_id != null ? (
                                <FaceCropThumbnail occurrenceId={suggestion.representative_id} size="fill" />
                              ) : (
                                <div className="w-full h-full flex items-center justify-center">
                                  <svg className="w-10 h-10 text-neutral-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                                  </svg>
                                </div>
                              )}
                            </div>
                            <div className="px-2 py-1.5">
                              <p className="text-xs text-neutral-100 truncate">
                                {suggestion.label || 'Unknown'}
                              </p>
                              <p className="text-[10px] text-neutral-400 mt-0.5">
                                {suggestion.face_count} faces · score {suggestion.score.toFixed(3)}
                              </p>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ) : !isLoadingSuggestions ? (
                    <p className="text-xs text-neutral-600">
                      No suggested links from current embeddings.
                    </p>
                  ) : null}
                </div>

                <div className={suggestedClusters.length > 0 ? 'pt-4 border-t border-neutral-800/70' : ''}>
                  <p className="mb-2 text-[11px] uppercase tracking-wide text-cyan-300/90">
                    Confirmed links
                  </p>
                  {pClusters.length > 0 ? (
                    <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-3">
                      {pClusters.map((pc) => {
                        const relinkCluster = getRelinkClusterFromPersonCluster(pc, expandedPersonId)
                        const key = `cluster:${pc.cluster_id}`
                        const isSelected = expandedKey === key
                        return (
                          <div
                            key={pc.cluster_id}
                            className={`group relative rounded-lg border transition-colors cursor-pointer ${
                              isSelected
                                ? 'border-cyan-600/70 bg-cyan-900/20'
                                : 'border-cyan-800/50 bg-cyan-950/10 hover:bg-cyan-900/15'
                            }`}
                            onClick={() => toggleExpand(key, relinkCluster, null)}
                          >
                            <div className="absolute top-1 left-1 rounded px-1.5 py-0.5 text-[10px] text-cyan-100 bg-cyan-900/60 border border-cyan-700/60 z-10">
                              Linked
                            </div>
                            <div className="aspect-square w-full overflow-hidden rounded-t-lg bg-neutral-800">
                              {pc.representative_id != null ? (
                                <FaceCropThumbnail occurrenceId={pc.representative_id} size="fill" />
                              ) : (
                                <div className="w-full h-full flex items-center justify-center">
                                  <svg className="w-10 h-10 text-neutral-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                                  </svg>
                                </div>
                              )}
                            </div>
                            <div className="px-2 py-1.5">
                              <p className="text-xs text-neutral-100 truncate">{pc.label}</p>
                              <p className="text-[10px] text-neutral-400 mt-0.5">
                                {pc.face_count} faces · {pc.image_count} images
                              </p>
                            </div>
                            <div
                              className="absolute top-1 right-1 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity"
                              onClick={(e) => e.stopPropagation()}
                            >
                              {renderRelinkButton(relinkCluster)}
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  void handleUnlinkCluster(pc.cluster_id)
                                }}
                                className="px-1.5 py-0.5 rounded bg-black/60 text-[10px] text-neutral-300 hover:text-red-400 transition-colors"
                                title="Unlink from person"
                              >
                                Unlink
                              </button>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ) : !isLoading ? (
                    <p className="text-xs text-neutral-600">No confirmed links yet.</p>
                  ) : null}
                </div>
              </div>
            </div>
          )
        })()}

        {expandedKey?.startsWith('cluster:') && (() => {
          const cId = parseInt(expandedKey.replace('cluster:', ''), 10)
          if (Number.isNaN(cId)) {
            return null
          }

          const suggested = Object.values(personLinkSuggestions)
            .flat()
            .find((suggestion) => suggestion.cluster_id === cId)
          const cl = clusters.find((cluster) => cluster.cluster_id === cId) ?? (
            suggested
              ? {
                  cluster_id: suggested.cluster_id,
                  identity_name: suggested.label || `cluster-${suggested.cluster_id}`,
                  display_name: suggested.label || null,
                  identity_id: null,
                  image_count: suggested.image_count,
                  face_count: suggested.face_count,
                  representative_id: suggested.representative_id,
                  person_id: null,
                }
              : null
          )
          const displayLabel = cl?.display_name || 'Unknown'
          return (
            <div className="shrink-0 border-t border-neutral-700 bg-neutral-900 max-h-[40vh] overflow-y-auto">
              <div className="px-4 py-2 border-b border-neutral-800/60 flex items-center justify-between sticky top-0 bg-neutral-900 z-10">
                <span className="text-xs text-neutral-300 font-medium">{displayLabel}</span>
                <div className="flex items-center gap-3">
                  {cl && renderRelinkButton(cl)}
                  <div className="relative">
                    {renderLinkDropdown(cId)}
                  </div>
                  <button onClick={() => setExpandedKey(null)} className="text-neutral-500 hover:text-neutral-300 text-xs">Close</button>
                </div>
              </div>
              {renderExpandedDetail(expandedKey)}
            </div>
          )
        })()}
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
          onClose={() => setLightboxPath(null)}
        />
      )}
    </div>
  )
}
