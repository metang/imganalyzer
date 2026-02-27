/**
 * SearchView.tsx — The "Search" tab: orchestrates SearchBar → VirtualGrid → SearchLightbox.
 */
import { useState, useCallback } from 'react'
import type { SearchFilters, SearchResult } from '../global'
import { SearchBar } from './SearchBar'
import { VirtualGrid } from './VirtualGrid'
import { SearchLightbox } from './SearchLightbox'

export function SearchView() {
  const [results, setResults] = useState<SearchResult[]>([])
  const [total, setTotal] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedItem, setSelectedItem] = useState<SearchResult | null>(null)

  const handleSearch = useCallback(async (filters: SearchFilters) => {
    setLoading(true)
    setError(null)
    setSelectedItem(null)

    try {
      const resp = await window.api.searchImages(filters)
      if (resp.error) {
        setError(resp.error)
        setResults([])
        setTotal(null)
      } else {
        setResults(resp.results)
        setTotal(resp.total)
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setError(msg)
      setResults([])
      setTotal(null)
    } finally {
      setLoading(false)
    }
  }, [])

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">

      {/* ── Search bar ───────────────────────────────────────────────────────── */}
      <SearchBar
        onSearch={handleSearch}
        loading={loading}
        resultCount={total}
      />

      {/* ── Content area ─────────────────────────────────────────────────────── */}
      {loading && (
        <div className="flex-1 flex items-center justify-center text-neutral-600 text-sm gap-2">
          <div className="w-4 h-4 border-2 border-neutral-700 border-t-neutral-400 rounded-full animate-spin" />
          Searching…
        </div>
      )}

      {!loading && error && (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 px-8">
          <svg className="w-10 h-10 text-red-500/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
          </svg>
          <p className="text-sm text-red-400 text-center max-w-lg">{error}</p>
        </div>
      )}

      {!loading && !error && total === null && (
        <div className="flex-1 flex flex-col items-center justify-center text-neutral-600 gap-3">
          <svg className="w-16 h-16 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.75}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M21 21l-4.35-4.35M17 11A6 6 0 115 11a6 6 0 0112 0z" />
          </svg>
          <p className="text-sm">Enter a query above to search your image library</p>
          <p className="text-xs text-neutral-700">
            Try: <code className="text-neutral-600">sunset</code>,{' '}
            <code className="text-neutral-600">portrait score&gt;7</code>,{' '}
            <code className="text-neutral-600">has:faces camera:Sony</code>
          </p>
        </div>
      )}

      {!loading && !error && total !== null && (
        <VirtualGrid
          items={results}
          selectedId={selectedItem?.image_id ?? null}
          onSelect={setSelectedItem}
        />
      )}

      {/* ── Lightbox ─────────────────────────────────────────────────────────── */}
      {selectedItem && (
        <SearchLightbox
          item={selectedItem}
          items={results}
          onClose={() => setSelectedItem(null)}
          onNavigate={setSelectedItem}
        />
      )}
    </div>
  )
}
