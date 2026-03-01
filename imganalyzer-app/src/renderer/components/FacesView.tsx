import { useState, useEffect, useCallback, useRef, memo } from 'react'
import type { FaceCluster, FaceOccurrence, FaceSummary, FaceImage } from '../global'

// ── Face crop thumbnail (lazy-loaded) ─────────────────────────────────────────

const FaceCropThumbnail = memo(function FaceCropThumbnail({
  occurrenceId,
  size = 'md',
}: {
  occurrenceId: number
  size?: 'sm' | 'md' | 'lg'
}) {
  const [src, setSrc] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  const requested = useRef(false)

  useEffect(() => {
    if (requested.current) return
    requested.current = true
    window.api
      .getFaceCrop(occurrenceId)
      .then((result) => {
        if (result.data) {
          setSrc(`data:image/jpeg;base64,${result.data}`)
        } else {
          setFailed(true)
        }
      })
      .catch(() => setFailed(true))
  }, [occurrenceId])

  const sizeClass =
    size === 'sm'
      ? 'w-12 h-12'
      : size === 'lg'
        ? 'w-24 h-24'
        : 'w-16 h-16'

  if (failed) {
    return (
      <div
        className={`${sizeClass} rounded bg-neutral-800 flex items-center justify-center shrink-0`}
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
        className={`${sizeClass} rounded bg-neutral-800 animate-pulse shrink-0`}
      />
    )
  }

  return (
    <img
      src={src}
      alt=""
      className={`${sizeClass} rounded object-cover shrink-0`}
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

  // ── Load data ─────────────────────────────────────────────────────────────

  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      // Try cluster mode first
      const clusterResult = await window.api.listFaceClusters()
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
    async (identityName: string) => {
      const trimmed = editValue.trim()
      try {
        await window.api.setFaceAlias(identityName, trimmed)
        // Update local state
        if (clusters.length > 0) {
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
    (e: React.KeyboardEvent, identityName: string) => {
      if (e.key === 'Enter') {
        e.preventDefault()
        saveAlias(identityName)
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

  const renderEditingField = (_key: string, identityName: string) => (
    <div className="flex items-center gap-2">
      <input
        ref={editInputRef}
        type="text"
        value={editValue}
        onChange={(e) => setEditValue(e.target.value)}
        onKeyDown={(e) => handleEditKeyDown(e, identityName)}
        placeholder="Enter alias..."
        className="flex-1 px-2 py-1 text-sm rounded bg-neutral-800 border border-neutral-600
                   text-neutral-100 placeholder-neutral-500 outline-none focus:border-blue-500
                   min-w-0"
      />
      <button
        onClick={() => saveAlias(identityName)}
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
                className="group relative"
                title={`${occ.file_path.split(/[/\\]/).pop()}${occ.age ? ` | Age: ~${occ.age}` : ''}${occ.gender ? ` | ${occ.gender}` : ''}`}
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
      {hasOccurrences && clusters.length > 0 && (
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
                      renderEditingField(key, cluster.identity_name)
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
                </div>

                {/* Expanded detail */}
                {isExpanded && renderExpandedDetail(key)}
              </div>
            )
          })}
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
    </div>
  )
}
