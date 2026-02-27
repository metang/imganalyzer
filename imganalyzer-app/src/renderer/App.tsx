import { useState, useCallback } from 'react'
import type { ImageFile } from './global'
import { FolderPicker } from './components/FolderPicker'
import { Gallery } from './components/Gallery'
import { Lightbox } from './components/Lightbox'
import { BatchView } from './components/BatchView'
import { SearchView } from './components/SearchView'

type Tab = 'gallery' | 'batch' | 'search'

export default function App() {
  const [tab, setTab] = useState<Tab>('gallery')

  // Gallery state
  const [folderPath, setFolderPath] = useState<string | null>(null)
  const [images, setImages] = useState<ImageFile[]>([])
  const [lightboxImage, setLightboxImage] = useState<ImageFile | null>(null)
  const [loading, setLoading] = useState(false)

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

  return (
    <div className="h-full flex flex-col">

      {/* ── Tab bar ──────────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-1 px-3 pt-2 pb-0 border-b border-neutral-800 shrink-0">
        {(['gallery', 'batch', 'search'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`
              px-4 py-1.5 text-sm rounded-t-md transition-colors capitalize
              ${tab === t
                ? 'bg-neutral-800 text-neutral-100 border border-b-transparent border-neutral-700'
                : 'text-neutral-500 hover:text-neutral-300'}
            `}
          >
            {t === 'gallery' ? 'Gallery' : t === 'batch' ? 'Batch' : 'Search'}
          </button>
        ))}
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

      {/* ── Batch tab ─────────────────────────────────────────────────────────── */}
      {tab === 'batch' && (
        <div className="flex-1 min-h-0 overflow-hidden">
          <BatchView initialFolder={folderPath ?? ''} />
        </div>
      )}

      {/* ── Search tab ────────────────────────────────────────────────────────── */}
      {tab === 'search' && (
        <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
          <SearchView />
        </div>
      )}
    </div>
  )
}
