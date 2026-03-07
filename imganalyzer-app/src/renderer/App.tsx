import { useState, useEffect, useRef, Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { DbGalleryView } from './components/DbGalleryView'
import { BatchConfigView, BatchRunView } from './components/BatchView'
import { SearchView } from './components/SearchView'
import { FacesView } from './components/FacesView'
import { SettingsView } from './components/SettingsView'
import { useBatchProcess } from './hooks/useBatchProcess'

type Tab = 'gallery' | 'batch' | 'running' | 'search' | 'faces' | 'settings'

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

  // Batch state — lifted here so it stays mounted across tab switches
  const batch = useBatchProcess()
  const [resumeBanner, setResumeBanner] = useState<string | null>(null)

  // Auto-resume any pending/running jobs from a previous session (runs once on mount)
  const didCheckRef = useRef(false)
  useEffect(() => {
    if (didCheckRef.current) return
    didCheckRef.current = true
    ;(async () => {
      const monitoring = await batch.monitorExisting()
      if (monitoring) {
        setResumeBanner('Monitoring jobs already being processed by a distributed worker…')
        setTab('running')
        return
      }
      const resumed = await batch.resumePending()
      if (resumed) {
        setResumeBanner('Resuming unfinished jobs from a previous session…')
        setTab('running')
      }
    })()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (batch.stats.status !== 'idle') return
    const timer = window.setInterval(() => {
      void batch.monitorExisting().then((monitoring) => {
        if (!monitoring) return
        setResumeBanner('Monitoring jobs already being processed by a distributed worker…')
        setTab('running')
      })
    }, 5000)
    return () => window.clearInterval(timer)
  }, [batch.monitorExisting, batch.stats.status])

  // Auto-switch to Running tab whenever the batch transitions to running/paused/done/error
  const prevStatusRef = useRef(batch.stats.status)
  useEffect(() => {
    const prev = prevStatusRef.current
    const curr = batch.stats.status
    prevStatusRef.current = curr
    if (prev !== curr && (curr === 'running' || curr === 'paused' || curr === 'done' || curr === 'error')) {
      setTab('running')
    }
  }, [batch.stats.status])

  // Whether there is an active / paused / errored batch in progress
  const batchIsActive =
    batch.stats.status === 'running' ||
    batch.stats.status === 'paused' ||
    batch.stats.status === 'ingesting' ||
    batch.stats.status === 'error'

  return (
    <ErrorBoundary>
    <div className="h-full flex flex-col">

      {/* ── Tab bar ──────────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-1 px-3 pt-2 pb-0 border-b border-neutral-800 shrink-0">

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

        {/* Faces */}
        <TabButton active={tab === 'faces'} onClick={() => setTab('faces')}>
          Faces
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
      {tab === 'gallery' && (
        <DbGalleryView onFolderContextChange={setGalleryFolderContext} />
      )}

      {/* ── Batch tab (config + ingest phases) ───────────────────────────────── */}
      {tab === 'batch' && (
        <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
          <BatchConfigView
            batch={batch}
            initialFolder={galleryFolderContext}
            onBatchStarted={() => setTab('running')}
          />
        </div>
      )}

      {/* ── Running tab (active/paused/done/error phases) ─────────────────────── */}
      {tab === 'running' && (
        <div className="flex-1 min-h-0 overflow-hidden">
          <BatchRunView
            batch={batch}
            initialFolder={galleryFolderContext}
            resumeBanner={resumeBanner}
            onDismissBanner={() => setResumeBanner(null)}
          />
        </div>
      )}

      {/* ── Search tab — always mounted so search state survives tab switches ── */}
      <div className={`flex-1 min-h-0 overflow-hidden flex flex-col${tab === 'search' ? '' : ' hidden'}`}>
        <SearchView />
      </div>

      {/* ── Faces tab ────────────────────────────────────────────────────────── */}
      {tab === 'faces' && (
        <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
          <FacesView />
        </div>
      )}

      {tab === 'settings' && (
        <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
          <SettingsView />
        </div>
      )}
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
        px-4 py-1.5 text-sm rounded-t-md transition-colors
        ${active
          ? 'bg-neutral-800 text-neutral-100 border border-b-transparent border-neutral-700'
          : 'text-neutral-500 hover:text-neutral-300'}
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
        p-2 rounded-md transition-colors border
        ${active
          ? 'bg-neutral-800 text-neutral-100 border-neutral-700'
          : 'text-neutral-500 border-transparent hover:text-neutral-300 hover:bg-neutral-900'}
      `}
    >
      {children}
    </button>
  )
}
