import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type {
  GalleryFolderNode,
  SearchResult,
} from '../global'
import { SearchLightbox } from './SearchLightbox'
import { VirtualGrid } from './VirtualGrid'

const CHUNK_SIZE = 300

interface FolderSidebarProps {
  folders: GalleryFolderNode[]
  selectedFolderPath: string | null
  expanded: Set<string>
  totalImages: number | null
  recursive: boolean
  onToggleRecursive: (next: boolean) => void
  onToggleExpand: (path: string) => void
  onSelectFolder: (path: string | null) => void
}

function FolderSidebar({
  folders,
  selectedFolderPath,
  expanded,
  totalImages,
  recursive,
  onToggleRecursive,
  onToggleExpand,
  onSelectFolder,
}: FolderSidebarProps) {
  const childrenByParent = useMemo(() => {
    const map = new Map<string | null, GalleryFolderNode[]>()
    for (const folder of folders) {
      const key = folder.parent_path
      const list = map.get(key) ?? []
      list.push(folder)
      map.set(key, list)
    }
    for (const list of map.values()) {
      list.sort((a, b) => a.name.localeCompare(b.name))
    }
    return map
  }, [folders])

  const visibleFolders = useMemo(() => {
    const out: GalleryFolderNode[] = []
    const walk = (parent: string | null): void => {
      const children = childrenByParent.get(parent) ?? []
      for (const node of children) {
        out.push(node)
        if (expanded.has(node.path)) walk(node.path)
      }
    }
    walk(null)
    return out
  }, [childrenByParent, expanded])

  return (
    <div className="h-full flex flex-col bg-neutral-950">
      <div className="px-3 py-2 border-b border-neutral-800 shrink-0">
        <h2 className="text-sm font-medium text-neutral-200">Folders</h2>
        <button
          className={`mt-2 w-full text-left px-2 py-1.5 rounded text-sm transition-colors ${
            selectedFolderPath === null
              ? 'bg-blue-600/30 text-blue-200'
              : 'text-neutral-300 hover:bg-neutral-800'
          }`}
          onClick={() => onSelectFolder(null)}
        >
          All processed images
          {typeof totalImages === 'number' && (
            <span className="text-xs text-neutral-500 ml-2">({totalImages})</span>
          )}
        </button>

        <label className="mt-2 flex items-center gap-2 text-xs text-neutral-400">
          <input
            type="checkbox"
            checked={recursive}
            onChange={(e) => onToggleRecursive(e.target.checked)}
            className="accent-blue-500"
          />
          Include subfolders
        </label>
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-2 min-h-0">
        {visibleFolders.length === 0 && (
          <div className="text-xs text-neutral-600 px-2 py-3">No folders found.</div>
        )}
        {visibleFolders.map((folder) => {
          const hasChildren = folder.child_count > 0
          const isExpanded = expanded.has(folder.path)
          const isSelected = folder.path === selectedFolderPath
          return (
            <div key={folder.path} className="flex items-center gap-1 py-0.5">
              <button
                type="button"
                onClick={() => hasChildren && onToggleExpand(folder.path)}
                className={`w-5 h-5 text-[10px] rounded ${
                  hasChildren ? 'text-neutral-500 hover:bg-neutral-800' : 'text-transparent'
                }`}
                title={hasChildren ? (isExpanded ? 'Collapse' : 'Expand') : undefined}
              >
                {hasChildren ? (isExpanded ? 'v' : '>') : '.'}
              </button>
              <button
                type="button"
                onClick={() => onSelectFolder(folder.path)}
                className={`flex-1 min-w-0 text-left px-2 py-1 rounded text-sm transition-colors ${
                  isSelected
                    ? 'bg-blue-600/30 text-blue-200'
                    : 'text-neutral-300 hover:bg-neutral-800'
                }`}
                style={{ paddingLeft: `${folder.depth * 12 + 8}px` }}
                title={folder.path}
              >
                <span className="truncate block">{folder.name}</span>
              </button>
              <span className="text-[10px] text-neutral-600 pr-1">{folder.image_count}</span>
            </div>
          )
        })}
      </div>

      <div className="shrink-0 border-t border-neutral-800 px-3 py-2 flex flex-col gap-2">
        <p className="text-[10px] text-neutral-500">
          Thumbnail cache settings moved to the Settings page.
        </p>
      </div>
    </div>
  )
}

interface DbGalleryViewProps {
  onFolderContextChange?: (folderPath: string) => void
}

export function DbGalleryView({ onFolderContextChange }: DbGalleryViewProps = {}) {
  const [folders, setFolders] = useState<GalleryFolderNode[]>([])
  const [totalImages, setTotalImages] = useState<number | null>(null)
  const [selectedFolderPath, setSelectedFolderPath] = useState<string | null>(null)
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set())
  const [recursive, setRecursive] = useState(true)

  const [items, setItems] = useState<SearchResult[]>([])
  const [selectedItem, setSelectedItem] = useState<SearchResult | null>(null)
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(false)
  const [loadingInitial, setLoadingInitial] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const chunkRequestTokenRef = useRef(0)

  const loadFolders = useCallback(async () => {
    const resp = await window.api.galleryListFolders()
    if (resp.error) throw new Error(resp.error)
    setFolders(resp.folders)
    setTotalImages(resp.totalImages)

    const roots = resp.folders
      .filter((f) => f.parent_path === null)
      .map((f) => f.path)
    setExpandedFolders(new Set(roots))
  }, [])

  const loadInitialChunk = useCallback(async () => {
    const token = ++chunkRequestTokenRef.current
    setLoading(true)
    setLoadingInitial(true)
    setError(null)
    setSelectedItem(null)
    try {
      const resp = await window.api.galleryListImagesChunk({
        folderPath: selectedFolderPath,
        recursive,
        chunkSize: CHUNK_SIZE,
        cursor: null,
      })
      if (resp.error) throw new Error(resp.error)
      if (token !== chunkRequestTokenRef.current) return
      setItems(resp.items)
      setNextCursor(resp.nextCursor)
      setHasMore(resp.hasMore)
      if (typeof resp.total === 'number') setTotalImages(resp.total)
    } catch (err) {
      if (token !== chunkRequestTokenRef.current) return
      const msg = err instanceof Error ? err.message : String(err)
      setError(msg)
      setItems([])
      setNextCursor(null)
      setHasMore(false)
    } finally {
      if (token !== chunkRequestTokenRef.current) return
      setLoading(false)
      setLoadingInitial(false)
    }
  }, [recursive, selectedFolderPath])

  const loadMore = useCallback(async () => {
    if (loading || !hasMore || !nextCursor) return
    const token = chunkRequestTokenRef.current
    setLoading(true)
    try {
      const resp = await window.api.galleryListImagesChunk({
        folderPath: selectedFolderPath,
        recursive,
        chunkSize: CHUNK_SIZE,
        cursor: nextCursor,
      })
      if (resp.error) throw new Error(resp.error)
      if (token !== chunkRequestTokenRef.current) return
      setItems((prev) => {
        const seen = new Set(prev.map((item) => item.image_id))
        const next = resp.items.filter((item) => !seen.has(item.image_id))
        return [...prev, ...next]
      })
      setNextCursor(resp.nextCursor)
      setHasMore(resp.hasMore)
    } catch (err) {
      if (token !== chunkRequestTokenRef.current) return
      const msg = err instanceof Error ? err.message : String(err)
      setError(msg)
      setHasMore(false)
    } finally {
      if (token !== chunkRequestTokenRef.current) return
      setLoading(false)
    }
  }, [hasMore, loading, nextCursor, recursive, selectedFolderPath])

  useEffect(() => {
    void loadFolders().catch((err) => {
      const msg = err instanceof Error ? err.message : String(err)
      setError(msg)
    })
  }, [loadFolders])

  useEffect(() => {
    void loadInitialChunk()
  }, [loadInitialChunk])

  useEffect(() => {
    onFolderContextChange?.(selectedFolderPath ?? '')
  }, [onFolderContextChange, selectedFolderPath])

  const toggleExpanded = useCallback((path: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }, [])

  const handleSelectFolder = useCallback((path: string | null) => {
    setSelectedFolderPath(path)
    setMobileSidebarOpen(false)
  }, [])

  const selectedFolderLabel = selectedFolderPath ?? 'All processed images'

  return (
    <div className="flex-1 min-h-0 flex overflow-hidden">
      <aside className="hidden lg:flex w-80 shrink-0 border-r border-neutral-800">
        <FolderSidebar
          folders={folders}
          selectedFolderPath={selectedFolderPath}
          expanded={expandedFolders}
          totalImages={totalImages}
          recursive={recursive}
          onToggleRecursive={setRecursive}
          onToggleExpand={toggleExpanded}
          onSelectFolder={handleSelectFolder}
        />
      </aside>

      {mobileSidebarOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <button
            type="button"
            className="absolute inset-0 bg-black/60"
            onClick={() => setMobileSidebarOpen(false)}
            aria-label="Close folder sidebar"
          />
          <div className="absolute left-0 top-0 bottom-0 w-80 max-w-[88vw] border-r border-neutral-800">
            <FolderSidebar
              folders={folders}
              selectedFolderPath={selectedFolderPath}
              expanded={expandedFolders}
              totalImages={totalImages}
              recursive={recursive}
              onToggleRecursive={setRecursive}
              onToggleExpand={toggleExpanded}
              onSelectFolder={handleSelectFolder}
            />
          </div>
        </div>
      )}

      <div className="flex-1 min-h-0 flex flex-col">
        <div className="px-3 py-2 border-b border-neutral-800 flex items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={() => setMobileSidebarOpen(true)}
            className="lg:hidden px-3 py-1.5 rounded bg-neutral-800 text-neutral-200 text-sm hover:bg-neutral-700"
          >
            Folders
          </button>
          <div className="min-w-0">
            <p className="text-sm text-neutral-200 truncate" title={selectedFolderLabel}>
              {selectedFolderLabel}
            </p>
            <p className="text-xs text-neutral-500">
              {items.length} loaded
              {typeof totalImages === 'number' ? ` of ${totalImages}` : ''}
              {hasMore ? ' (progressive loading)' : ''}
            </p>
          </div>
        </div>

        <div className="flex-1 min-h-0 relative overflow-hidden flex flex-col">
          {loadingInitial && (
            <div className="absolute inset-0 flex items-center justify-center text-neutral-600 text-sm gap-2">
              <div className="w-4 h-4 border-2 border-neutral-700 border-t-neutral-400 rounded-full animate-spin" />
              Loading processed images...
            </div>
          )}

          {!loadingInitial && error && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-red-400 text-sm px-6 text-center gap-2">
              <p>Failed to load gallery</p>
              <p className="text-xs text-neutral-500">{error}</p>
            </div>
          )}

          {!loadingInitial && !error && items.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center text-neutral-600 text-sm">
              No processed images found for this folder.
            </div>
          )}

          {!loadingInitial && !error && items.length > 0 && (
            <>
              <VirtualGrid
                items={items}
                selectedId={selectedItem?.image_id ?? null}
                onSelect={setSelectedItem}
                onEndReached={hasMore ? loadMore : undefined}
              />
              {loading && (
                <div className="absolute bottom-3 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded-full bg-black/60 text-neutral-300 text-xs flex items-center gap-2">
                  <div className="w-3 h-3 border border-neutral-500 border-t-neutral-300 rounded-full animate-spin" />
                  Loading more...
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {selectedItem && (
        <SearchLightbox
          item={selectedItem}
          items={items}
          onClose={() => setSelectedItem(null)}
          onNavigate={setSelectedItem}
        />
      )}
    </div>
  )
}
