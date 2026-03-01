import { useState, useEffect, useCallback, useRef, memo } from 'react'
import type { FaceCluster, FaceOccurrence, FaceSummary, FaceImage, FacePerson, PersonCluster } from '../global'

// ── Thumbnail cache & batch fetcher ───────────────────────────────────────────

const THUMB_CACHE_MAX = 2000
const thumbCache = new Map<number, string>() // occurrence_id → base64 data URI
const pendingIds = new Set<number>()
const pendingCallbacks = new Map<number, Array<(src: string | null) => void>>()
let batchTimer: ReturnType<typeof setTimeout> | null = null

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

// ── Main FacesView component ──────────────────────────────────────────────────

export function FacesView() {
  // Cluster mode (Phase 2 — face_occurrences exist)
  const [clusters, setClusters] = useState<FaceCluster[]>([])
  const [hasOccurrences, setHasOccurrences] = useState(false)

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

  // Lightbox state (in-app image viewer)
  const [lightboxPath, setLightboxPath] = useState<string | null>(null)

  // View mode: clusters or people
  type ViewMode = 'clusters' | 'people'
  const [viewMode, setViewMode] = useState<ViewMode>('clusters')

  // Person state
  const [persons, setPersons] = useState<FacePerson[]>([])
  const [personClusters, setPersonClusters] = useState<Record<number, PersonCluster[]>>({})
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

  // ── Load data ─────────────────────────────────────────────────────────────

  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      // Load clusters + persons in parallel
      const [clusterResult, personsResult] = await Promise.all([
        window.api.listFaceClusters(),
        window.api.listPersons(),
      ])

      if (clusterResult.error) {
        setError(clusterResult.error)
        return
      }

      if (clusterResult.has_occurrences && clusterResult.clusters.length > 0) {
        setHasOccurrences(true)
        setClusters(clusterResult.clusters)
        setLegacyFaces([])
      } else {
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

  useEffect(() => {
    loadData()
  }, [loadData])

  // ── Clustering ────────────────────────────────────────────────────────────

  const handleCluster = useCallback(async () => {
    setClustering(true)
    try {
      const result = await window.api.runFaceClustering()
      if (result.error) {
        setError(result.error)
      } else {
        // Reload data after clustering
        await loadData()
      }
    } catch (err) {
      setError(String(err))
    } finally {
      setClustering(false)
    }
  }, [loadData])

  const handleRebuild = useCallback(async () => {
    setShowRebuildConfirm(false)
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

  const togglePersonExpand = useCallback(
    async (personId: number) => {
      if (expandedPersonId === personId) {
        setExpandedPersonId(null)
        return
      }
      setExpandedPersonId(personId)

      // Load clusters for this person if not cached
      let pClusters = personClusters[personId]
      if (!pClusters) {
        const result = await window.api.getPersonClusters(personId)
        if (!result.error) {
          pClusters = result.clusters
          setPersonClusters((prev) => ({ ...prev, [personId]: pClusters! }))
        }
      }

      // Load all face occurrences for this person (all clusters merged)
      const personKey = `person:${personId}`
      if (pClusters && !clusterOccurrences[personKey]) {
        setLoadingDetail(personKey)
        try {
          const allOccs: FaceOccurrence[] = []
          for (const pc of pClusters) {
            const r = await window.api.getFaceClusterImages(pc.cluster_id, null)
            if (!r.error) allOccs.push(...r.occurrences)
          }
          setClusterOccurrences((prev) => ({ ...prev, [personKey]: allOccs }))
        } catch {
          // silently ignore
        } finally {
          setLoadingDetail(null)
        }
      }
    },
    [expandedPersonId, personClusters, clusterOccurrences]
  )

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
  const totalFaceCount = hasOccurrences
    ? clusters.reduce((sum, c) => sum + c.face_count, 0)
    : legacyFaces.reduce((sum, f) => sum + f.image_count, 0)

  // For people view: clusters not assigned to any person
  const unlinkedClusters = clusters.filter(
    (c) => c.cluster_id !== null && !c.person_id
  )

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

    const lowerFilter = linkSearchFilter.toLowerCase()
    const filteredPersons = lowerFilter
      ? persons.filter((p) => p.name.toLowerCase().includes(lowerFilter))
      : persons

    return (
      <div
        className="absolute right-4 top-full mt-1 z-40 bg-neutral-900 border border-neutral-700 rounded-lg shadow-xl
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
          {filteredPersons.length === 0 && lowerFilter && (
            <div className="px-3 py-1.5 text-xs text-neutral-500">No match</div>
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

  const renderEditButton = (
    key: string,
    displayName: string | null,
    identityName: string
  ) => (
    <button
      onClick={() => startEditing(key, displayName, identityName)}
      className="text-neutral-600 hover:text-neutral-300 transition-colors shrink-0"
      title="Set alias"
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
            onClick={() => setShowRebuildConfirm(true)}
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
            <p className="text-xs text-neutral-400 mb-4">
              This will re-enqueue face detection jobs for <strong>all images</strong>.
              Existing face data will be replaced when the batch runs.
              You&apos;ll need to start a batch run afterwards to process the queue.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowRebuildConfirm(false)}
                className="px-3 py-1.5 text-xs rounded-md bg-neutral-800 text-neutral-300 hover:bg-neutral-700
                           transition-colors border border-neutral-600"
              >
                Cancel
              </button>
              <button
                onClick={handleRebuild}
                className="px-3 py-1.5 text-xs rounded-md bg-amber-700 text-white hover:bg-amber-600
                           transition-colors"
              >
                Rebuild All Faces
              </button>
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
          {unlinkedClusters.length > 0 && (
            <>
              <div className="px-5 py-2 text-xs text-neutral-500 font-medium uppercase tracking-wide bg-neutral-900/80 border-b border-neutral-800/60 sticky top-0">
                Unlinked Clusters ({unlinkedClusters.length})
              </div>
              <div className="p-4">
                <div className="grid grid-cols-[repeat(auto-fill,minmax(120px,1fr))] gap-3">
                  {unlinkedClusters.map((cluster) => {
                    const displayLabel = cluster.display_name || cluster.identity_name
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
          const person = persons.find((p) => p.id === expandedPersonId)
          const personKey = `person:${expandedPersonId}`
          const allOccs = clusterOccurrences[personKey]
          const isLoading = loadingDetail === personKey

          return (
            <div className="shrink-0 border-t border-cyan-800/50 bg-neutral-900 max-h-[40vh] overflow-y-auto">
              <div className="px-4 py-2 border-b border-neutral-800/60 flex items-center justify-between sticky top-0 bg-neutral-900 z-10">
                <span className="text-xs text-cyan-300 font-medium">
                  {person?.name} — {pClusters.length} cluster{pClusters.length !== 1 ? 's' : ''}
                  {allOccs ? ` · ${allOccs.length} faces` : ''}
                </span>
                <button onClick={() => setExpandedPersonId(null)} className="text-neutral-500 hover:text-neutral-300 text-xs">Close</button>
              </div>

              {isLoading && (
                <div className="px-4 py-3 flex items-center gap-2 text-xs text-neutral-500">
                  <span className="w-3 h-3 border border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
                  Loading faces...
                </div>
              )}
              {allOccs && allOccs.length > 0 && (
                <div className="px-4 py-3">
                  <div className="flex flex-wrap gap-2">
                    {allOccs.map((occ) => (
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
              )}
              {allOccs && allOccs.length === 0 && (
                <div className="px-4 py-3 text-xs text-neutral-600">No face occurrences found.</div>
              )}

              {pClusters.length > 0 && (
                <div className="border-t border-neutral-800/60">
                  <div className="px-4 py-1.5 text-[10px] text-neutral-500 uppercase tracking-wide">
                    Clusters
                  </div>
                  {pClusters.map((pc) => (
                    <div key={pc.cluster_id} className="flex items-center gap-2 px-4 py-1 text-xs hover:bg-neutral-800/30">
                      {pc.representative_id != null && (
                        <FaceCropThumbnail occurrenceId={pc.representative_id} size="sm" />
                      )}
                      <span className="flex-1 text-neutral-300 truncate">{pc.label}</span>
                      <span className="text-neutral-500">{pc.face_count} faces</span>
                      <button
                        onClick={() => handleUnlinkCluster(pc.cluster_id)}
                        className="text-neutral-600 hover:text-red-400 transition-colors"
                        title="Unlink from person"
                      >
                        Unlink
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )
        })()}

        {expandedKey && unlinkedClusters.some((c) => `cluster:${c.cluster_id}` === expandedKey) && (() => {
          const cId = parseInt(expandedKey.replace('cluster:', ''), 10)
          const cl = unlinkedClusters.find((c) => c.cluster_id === cId)
          const displayLabel = cl?.display_name || cl?.identity_name || `Cluster #${cId}`
          return (
          <div className="shrink-0 border-t border-neutral-700 bg-neutral-900 max-h-[40vh] overflow-y-auto">
            <div className="px-4 py-2 border-b border-neutral-800/60 flex items-center justify-between sticky top-0 bg-neutral-900 z-10">
              <span className="text-xs text-neutral-300 font-medium">{displayLabel}</span>
              <div className="flex items-center gap-3">
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
