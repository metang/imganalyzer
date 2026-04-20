import { useState, useEffect, useRef, useCallback, Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import type { SearchFilters } from './global'
import { DbGalleryView } from './components/DbGalleryView'
import { BatchConfigView, BatchRunView } from './components/BatchView'
import { SearchView } from './components/SearchView'
import { FacesView } from './components/FacesView'
import { SettingsView } from './components/SettingsView'
import { MapView } from './components/MapView'
import { AlbumsView } from './components/AlbumsView'
import { useBatchProcess } from './hooks/useBatchProcess'

type Tab = 'gallery' | 'batch' | 'running' | 'search' | 'albums' | 'faces' | 'map' | 'settings'
type FaceClusterDeepLink = {
  clusterId: number
  sourceImageId: number
  requestId: number
}

// ── Error Boundary ────────────────────────────────────────────────────────────

class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary]', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="h-full flex flex-col items-center justify-center gap-4 p-8 text-center">
          <p className="text-red-400 text-sm font-semibold">Something went wrong</p>
          <pre className="text-xs text-neutral-500 max-w-lg whitespace-pre-wrap break-words">
            {this.state.error.message}
          </pre>
          <button
            onClick={() => this.setState({ error: null })}
            className="px-4 py-1.5 rounded-lg text-sm bg-neutral-700 text-neutral-200 hover:bg-neutral-600 transition-colors"
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

export default function App() {
  const [tab, setTab] = useState<Tab>('gallery')
  const [galleryFolderContext, setGalleryFolderContext] = useState('')
  const [facesDeepLink, setFacesDeepLink] = useState<FaceClusterDeepLink | null>(null)
  const facesDeepLinkRequestId = useRef(0)

  // Batch state — lifted here so it stays mounted across tab switches
  const batch = useBatchProcess()
  const [resumeBanner, setResumeBanner] = useState<string | null>(null)

  // Auto-resume any pending/running jobs from a previous session (runs once on mount)
  // without hijacking the user's current tab.
  const didCheckRef = useRef(false)
  useEffect(() => {
    if (didCheckRef.current) return
    didCheckRef.current = true
    ;(async () => {
      const resumed = await batch.resumePending()
      if (resumed) {
        setResumeBanner('Resuming unfinished jobs from queue…')
        return
      }
      const monitoring = await batch.monitorExisting()
      if (monitoring) {
        setResumeBanner('Monitoring jobs already being processed by a distributed worker…')
      }
    })()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (batch.stats.status !== 'idle') return
    let cancelled = false
    let inFlight = false

    const poll = async () => {
      if (cancelled || document.hidden || inFlight) return
      inFlight = true
      try {
        const resumed = await batch.resumePending()
        if (cancelled) return
        if (resumed) {
          setResumeBanner('Resuming unfinished jobs from queue…')
          return
        }
        const monitoring = await batch.monitorExisting()
        if (cancelled || !monitoring) return
        setResumeBanner('Monitoring jobs already being processed by a distributed worker…')
      } finally {
        inFlight = false
      }
    }

    const timer = window.setInterval(() => {
      void poll()
    }, 15000)
    const handleVisibility = () => {
      if (!document.hidden) void poll()
    }
    document.addEventListener('visibilitychange', handleVisibility)
    return () => {
      cancelled = true
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', handleVisibility)
    }
  }, [batch.monitorExisting, batch.resumePending, batch.stats.status])

  // Whether there is an active / paused / errored batch in progress
  const batchIsActive =
    batch.stats.status === 'running' ||
    batch.stats.status === 'paused' ||
    batch.stats.status === 'ingesting' ||
    batch.stats.status === 'error'

  const handleOpenFaceCluster = useCallback((clusterId: number, sourceImageId: number) => {
    facesDeepLinkRequestId.current += 1
    setFacesDeepLink({
      clusterId,
      sourceImageId,
      requestId: facesDeepLinkRequestId.current,
    })
    setTab('faces')
  }, [])

  const handleFacesDeepLinkHandled = useCallback((requestId: number) => {
    setFacesDeepLink((current) => (
      current && current.requestId === requestId
        ? null
        : current
    ))
  }, [])

  // Cross-tab: Search ↔ Map
  const [pendingMapFilters, setPendingMapFilters] = useState<SearchFilters | null>(null)
  const [pendingGridSearch, setPendingGridSearch] = useState<{
    filters: SearchFilters
    mapBounds?: { north: number; south: number; east: number; west: number }
  } | null>(null)

  const handleViewOnMap = useCallback((filters: SearchFilters) => {
    setPendingMapFilters(filters)
    setTab('map')
  }, [])

  const handleViewAsGrid = useCallback((
    filters: SearchFilters,
    mapBounds?: { north: number; south: number; east: number; west: number },
  ) => {
    setPendingGridSearch({ filters: { ...filters, mapBounds }, mapBounds })
    setTab('search')
  }, [])

  return (
    <ErrorBoundary>
    <div className="h-full flex flex-col">

      {/* ── Tab bar ──────────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-1.5 px-3 py-2 border-b border-neutral-800/80 bg-neutral-950/30 shrink-0 backdrop-blur-sm">
        <div className="mr-2 rounded-full border border-cyan-500/20 bg-cyan-500/5 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-cyan-200">
          imganalyzer
        </div>

        {/* Gallery */}
        <TabButton active={tab === 'gallery'} onClick={() => setTab('gallery')}>
          Gallery
        </TabButton>

        {/* Batch */}
        <TabButton active={tab === 'batch'} onClick={() => setTab('batch')}>
          Batch
        </TabButton>

        {/* Running — shows an animated dot when a batch is in progress */}
        <TabButton active={tab === 'running'} onClick={() => setTab('running')}>
          <span className="flex items-center gap-1.5">
            Running
            {batchIsActive && (
              <span className={`
                inline-block w-2 h-2 rounded-full shrink-0
                ${batch.stats.status === 'error'
                  ? 'bg-red-500'
                  : batch.stats.status === 'paused'
                  ? 'bg-yellow-500'
                  : 'bg-emerald-500 animate-pulse'}
              `} />
            )}
          </span>
        </TabButton>

        {/* Search */}
        <TabButton active={tab === 'search'} onClick={() => setTab('search')}>
          Search
        </TabButton>

        {/* Albums */}
        <TabButton active={tab === 'albums'} onClick={() => setTab('albums')}>
          Albums
        </TabButton>

        {/* Faces */}
        <TabButton active={tab === 'faces'} onClick={() => setTab('faces')}>
          Faces
        </TabButton>

        {/* Map */}
        <TabButton active={tab === 'map'} onClick={() => setTab('map')}>
          Map
        </TabButton>

        <div className="ml-auto flex items-center">
          <IconButton
            active={tab === 'settings'}
            title="Settings"
            onClick={() => setTab('settings')}
          >
            <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317a1 1 0 0 1 1.35-.936l.83.356a1 1 0 0 0 .76 0l.83-.356a1 1 0 0 1 1.35.936l.115.895a1 1 0 0 0 .474.724l.765.458a1 1 0 0 1 .35 1.357l-.45.776a1 1 0 0 0-.11.735l.195.868a1 1 0 0 1-.824 1.19l-.886.14a1 1 0 0 0-.644.383l-.57.692a1 1 0 0 1-1.401.172l-.702-.548a1 1 0 0 0-.748-.2l-.88.16a1 1 0 0 1-1.154-.873l-.093-.902a1 1 0 0 0-.4-.695l-.721-.543a1 1 0 0 1-.23-1.381l.53-.735a1 1 0 0 0 .157-.724l-.097-.9a1 1 0 0 1 .962-1.094l.906-.024a1 1 0 0 0 .703-.327l.61-.645Z" />
              <circle cx="12" cy="12" r="2.75" />
            </svg>
          </IconButton>
        </div>
      </div>

      {/* ── Gallery tab ───────────────────────────────────────────────────────── */}
      <div className={`flex-1 min-h-0 overflow-hidden flex flex-col${tab === 'gallery' ? '' : ' hidden'}`}>
        <DbGalleryView onFolderContextChange={setGalleryFolderContext} />
      </div>

      {/* ── Batch tab (config + ingest phases) ───────────────────────────────── */}
      <div className={`flex-1 min-h-0 overflow-hidden flex flex-col${tab === 'batch' ? '' : ' hidden'}`}>
        <BatchConfigView
          batch={batch}
          initialFolder={galleryFolderContext}
        />
      </div>

      {/* ── Running tab (active/paused/done/error phases) ─────────────────────── */}
      <div className={`flex-1 min-h-0 overflow-hidden${tab === 'running' ? '' : ' hidden'}`}>
        <BatchRunView
          batch={batch}
          initialFolder={galleryFolderContext}
          resumeBanner={resumeBanner}
          onDismissBanner={() => setResumeBanner(null)}
        />
      </div>

      {/* ── Search tab — always mounted so search state survives tab switches ── */}
      <div className={`flex-1 min-h-0 overflow-hidden flex flex-col${tab === 'search' ? '' : ' hidden'}`}>
        <SearchView
          onOpenFaceCluster={handleOpenFaceCluster}
          onViewOnMap={handleViewOnMap}
          pendingSearch={pendingGridSearch}
          onClearPendingSearch={() => setPendingGridSearch(null)}
        />
      </div>

      {/* ── Albums tab ─────────────────────────────────────────────────────── */}
      <div className={`flex-1 min-h-0 overflow-hidden flex flex-col${tab === 'albums' ? '' : ' hidden'}`}>
        <AlbumsView />
      </div>

      {/* ── Faces tab ────────────────────────────────────────────────────────── */}
      <div className={`flex-1 min-h-0 overflow-hidden flex flex-col${tab === 'faces' ? '' : ' hidden'}`}>
        <FacesView
          deepLinkRequest={facesDeepLink}
          onDeepLinkHandled={handleFacesDeepLinkHandled}
        />
      </div>

      {/* ── Map tab ──────────────────────────────────────────────────────────── */}
      <div className={`flex-1 min-h-0 overflow-hidden flex flex-col${tab === 'map' ? '' : ' hidden'}`}>
        <MapView
          pendingFilters={pendingMapFilters}
          onClearPending={() => setPendingMapFilters(null)}
          onViewAsGrid={handleViewAsGrid}
        />
      </div>

      {/* ── Settings tab ─────────────────────────────────────────────────────── */}
      <div className={`flex-1 min-h-0 overflow-hidden flex flex-col${tab === 'settings' ? '' : ' hidden'}`}>
        <SettingsView active={tab === 'settings'} />
      </div>
    </div>
    </ErrorBoundary>
  )
}

// ── TabButton ─────────────────────────────────────────────────────────────────

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick(): void
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className={`
        px-4 py-1.5 text-sm rounded-xl border transition-colors
        ${active
          ? 'bg-cyan-500/10 text-cyan-100 border-cyan-500/30'
          : 'text-neutral-400 border-transparent hover:text-neutral-200 hover:border-neutral-700 hover:bg-neutral-900/70'}
      `}
    >
      {children}
    </button>
  )
}

function IconButton({
  active,
  onClick,
  title,
  children,
}: {
  active: boolean
  onClick(): void
  title: string
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      onClick={onClick}
      className={`
        p-2 rounded-xl transition-colors border
        ${active
          ? 'bg-cyan-500/10 text-cyan-100 border-cyan-500/30'
          : 'text-neutral-400 border-transparent hover:text-neutral-200 hover:bg-neutral-900 hover:border-neutral-700'}
      `}
    >
      {children}
    </button>
  )
}
