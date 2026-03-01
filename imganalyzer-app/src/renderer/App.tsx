import { useState, useCallback, useEffect, useRef, Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import type { ImageFile } from './global'
import { FolderPicker } from './components/FolderPicker'
import { Gallery } from './components/Gallery'
import { Lightbox } from './components/Lightbox'
import { BatchConfigView, BatchRunView } from './components/BatchView'
import { SearchView } from './components/SearchView'
import { FacesView } from './components/FacesView'
import { useBatchProcess } from './hooks/useBatchProcess'

type Tab = 'gallery' | 'batch' | 'running' | 'search' | 'faces'

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

  // Gallery state
  const [folderPath, setFolderPath] = useState<string | null>(null)
  const [images, setImages] = useState<ImageFile[]>([])
  const [lightboxImage, setLightboxImage] = useState<ImageFile | null>(null)
  const [loading, setLoading] = useState(false)

  // Batch state — lifted here so it stays mounted across tab switches
  const batch = useBatchProcess()
  const [resumeBanner, setResumeBanner] = useState<string | null>(null)

  // Auto-resume any pending/running jobs from a previous session (runs once on mount)
  const didCheckRef = useRef(false)
  useEffect(() => {
    if (didCheckRef.current) return
    didCheckRef.current = true
    ;(async () => {
      const resumed = await batch.resumePending()
      if (resumed) {
        setResumeBanner('Resuming unfinished jobs from a previous session…')
        setTab('running')
      }
    })()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

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

  const handleFolderChange = useCallback(async (path: string) => {
    setFolderPath(path)
    setLightboxImage(null)
    setLoading(true)
    try {
      const imgs = await window.api.listImages(path)
      setImages(imgs)
    } catch (err) {
      console.error('Failed to list images:', err)
      setImages([])
    } finally {
      setLoading(false)
    }
  }, [])

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
      </div>

      {/* ── Gallery tab ───────────────────────────────────────────────────────── */}
      {tab === 'gallery' && (
        <>
          <FolderPicker folderPath={folderPath} onFolderChange={handleFolderChange} />

          {loading && (
            <div className="flex-1 flex items-center justify-center text-neutral-600 text-sm gap-2">
              <div className="w-4 h-4 border-2 border-neutral-700 border-t-neutral-400 rounded-full animate-spin" />
              Loading images…
            </div>
          )}

          {!loading && !folderPath && (
            <div className="flex-1 flex flex-col items-center justify-center text-neutral-600 gap-3">
              <svg className="w-16 h-16 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3 3l18 18M3.75 9A.75.75 0 014.5 8.25H6" />
              </svg>
              <p className="text-sm">Open a folder to get started</p>
            </div>
          )}

          {!loading && folderPath && (
            <Gallery
              images={images}
              selectedPath={lightboxImage?.path ?? null}
              onSelect={setLightboxImage}
            />
          )}

          {lightboxImage && (
            <Lightbox
              image={lightboxImage}
              images={images}
              onClose={() => setLightboxImage(null)}
              onNavigate={setLightboxImage}
            />
          )}
        </>
      )}

      {/* ── Batch tab (config + ingest phases) ───────────────────────────────── */}
      {tab === 'batch' && (
        <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
          <BatchConfigView
            batch={batch}
            initialFolder={folderPath ?? ''}
            onBatchStarted={() => setTab('running')}
          />
        </div>
      )}

      {/* ── Running tab (active/paused/done/error phases) ─────────────────────── */}
      {tab === 'running' && (
        <div className="flex-1 min-h-0 overflow-hidden">
          <BatchRunView
            batch={batch}
            initialFolder={folderPath ?? ''}
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
