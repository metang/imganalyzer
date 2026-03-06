/**
 * SearchView.tsx — The "Search" tab: orchestrates SearchBar → VirtualGrid → SearchLightbox.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { SearchFilters, SearchResult } from '../global'
import { SearchBar } from './SearchBar'
import { VirtualGrid } from './VirtualGrid'
import { SearchLightbox } from './SearchLightbox'

const SEARCH_PAGE_SIZE = 200

function appendUniqueResults(
  existing: SearchResult[],
  incoming: SearchResult[],
): SearchResult[] {
  const seen = new Set(existing.map((item) => item.image_id))
  const appended = incoming.filter((item) => !seen.has(item.image_id))
  return [...existing, ...appended]
}

export function SearchView() {
  const [results, setResults] = useState<SearchResult[]>([])
  const [total, setTotal] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [hasMore, setHasMore] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedItem, setSelectedItem] = useState<SearchResult | null>(null)
  const [searchContextLabel, setSearchContextLabel] = useState<string | null>(null)
  const activeFiltersRef = useRef<SearchFilters | null>(null)
  const requestIdRef = useRef(0)
  const resultsRef = useRef<SearchResult[]>([])

  useEffect(() => {
    resultsRef.current = results
  }, [results])

  const resultSummary = useMemo(() => {
    if (!hasSearched) return null
    if (total !== null) {
      return `${total} result${total !== 1 ? 's' : ''}`
    }
    const loaded = results.length
    if (hasMore) {
      return `${loaded}+ results loaded`
    }
    return `${loaded} result${loaded !== 1 ? 's' : ''}`
  }, [hasMore, hasSearched, results.length, total])

  const runSearch = useCallback(async (
    filters: SearchFilters,
    contextLabel: string | null = null,
  ) => {
    const requestId = requestIdRef.current + 1
    requestIdRef.current = requestId
    activeFiltersRef.current = { ...filters }

    setHasSearched(true)
    setLoading(true)
    setLoadingMore(false)
    setHasMore(false)
    setResults([])
    setTotal(null)
    setError(null)
    setSelectedItem(null)
    setSearchContextLabel(contextLabel)

    try {
      const resp = await window.api.searchImages({
        ...filters,
        limit: SEARCH_PAGE_SIZE,
        offset: 0,
      })
      if (requestIdRef.current !== requestId) return
      if (resp.error) {
        setError(resp.error)
        setResults([])
        setTotal(null)
        setHasMore(false)
      } else {
        setResults(resp.results)
        setTotal(resp.total)
        setHasMore(resp.hasMore)
      }
    } catch (err) {
      if (requestIdRef.current !== requestId) return
      const msg = err instanceof Error ? err.message : String(err)
      setError(msg)
      setResults([])
      setTotal(null)
    } finally {
      if (requestIdRef.current !== requestId) return
      setLoading(false)
    }
  }, [])

  const handleSearch = useCallback(async (filters: SearchFilters, contextLabel: string | null) => {
    await runSearch(filters, contextLabel)
  }, [runSearch])

  const handleFindSimilar = useCallback(async (item: SearchResult) => {
    const filename = item.file_path.split(/[/\\]/).pop() ?? `image ${item.image_id}`
    await runSearch(
      { similarToImageId: item.image_id, mode: 'semantic' },
      `Similar to ${filename}`,
    )
  }, [runSearch])

  const loadMore = useCallback(async () => {
    const filters = activeFiltersRef.current
    if (!filters || loading || loadingMore || !hasMore) return

    const requestId = requestIdRef.current
    const offset = resultsRef.current.length
    setLoadingMore(true)

    try {
      const resp = await window.api.searchImages({
        ...filters,
        limit: SEARCH_PAGE_SIZE,
        offset,
      })
      if (requestIdRef.current !== requestId) return
      if (resp.error) {
        setError(resp.error)
        setHasMore(false)
        return
      }
      setError(null)
      setResults((prev) => appendUniqueResults(prev, resp.results))
      setTotal(resp.total)
      setHasMore(resp.hasMore)
    } catch (err) {
      if (requestIdRef.current !== requestId) return
      const msg = err instanceof Error ? err.message : String(err)
      setError(msg)
      setHasMore(false)
    } finally {
      if (requestIdRef.current !== requestId) return
      setLoadingMore(false)
    }
  }, [hasMore, loading, loadingMore])

  return (
    <div className="flex-1 flex min-h-0 flex-col overflow-hidden">
      <SearchBar
        onSearch={handleSearch}
        loading={loading}
        resultSummary={resultSummary}
      />

      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">

        {loading && (
          <div className="flex-1 flex items-center justify-center text-neutral-600 text-sm gap-2">
            <div className="w-4 h-4 border-2 border-neutral-700 border-t-neutral-400 rounded-full animate-spin" />
            Searching…
          </div>
        )}

        {!loading && error && results.length === 0 && (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 px-8">
            <svg className="w-10 h-10 text-red-500/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
            </svg>
            <p className="text-sm text-red-400 text-center max-w-lg">{error}</p>
          </div>
        )}

        {!loading && !error && !hasSearched && (
          <div className="flex-1 flex flex-col items-center justify-center text-neutral-600 gap-3">
            <svg className="w-16 h-16 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M21 21l-4.35-4.35M17 11A6 6 0 115 11a6 6 0 0112 0z" />
            </svg>
            <p className="text-sm">Start with a prompt or pick an intent above to search your image library.</p>
            <p className="text-xs text-neutral-700">
              Try: <code className="text-neutral-600">Alice in the US every Feb 1 morning</code>,{' '}
              <code className="text-neutral-600">duck on water</code>,{' '}
              <code className="text-neutral-600">best photo of the sunset scene</code>
            </p>
          </div>
        )}

        {!loading && !error && hasSearched && results.length === 0 && (
          <div className="flex-1 flex flex-col items-center justify-center text-neutral-600 gap-3 px-8">
            <svg className="w-16 h-16 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 15l6 6m-11-4a7 7 0 110-14 7 7 0 010 14z" />
            </svg>
            <p className="text-sm">No images matched this search.</p>
            <p className="text-xs text-neutral-700">Try broadening the query or removing some filters.</p>
          </div>
        )}

        {results.length > 0 && (
          <div className="flex-1 min-h-0 relative overflow-hidden flex flex-col">
            {searchContextLabel && (
              <div className="shrink-0 px-4 py-2 border-b border-neutral-800 text-xs text-neutral-400">
                {searchContextLabel}
              </div>
            )}
            {error && (
              <div className="shrink-0 px-4 py-2 border-b border-red-900/40 bg-red-950/40 text-xs text-red-300">
                {error}
              </div>
            )}
            <VirtualGrid
              items={results}
              selectedId={selectedItem?.image_id ?? null}
              onSelect={setSelectedItem}
              onEndReached={hasMore ? loadMore : undefined}
            />
            {loadingMore && (
              <div className="absolute bottom-3 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded-full bg-black/60 text-neutral-300 text-xs flex items-center gap-2">
                <div className="w-3 h-3 border border-neutral-500 border-t-neutral-300 rounded-full animate-spin" />
                Loading more...
              </div>
            )}
          </div>
        )}

      </div>

      {/* ── Lightbox ─────────────────────────────────────────────────────────── */}
      {selectedItem && (
        <SearchLightbox
          item={selectedItem}
          items={results}
          onClose={() => setSelectedItem(null)}
          onFindSimilar={handleFindSimilar}
          onNavigate={setSelectedItem}
        />
      )}
    </div>
  )
}
