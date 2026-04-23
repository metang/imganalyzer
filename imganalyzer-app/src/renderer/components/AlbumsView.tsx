/**
 * AlbumsView — Smart albums list + story chapter timeline.
 *
 * Layout:
 *   Left panel  — album list + create/preset dialogs
 *   Right panel — story timeline (chapters → moments → images)
 */
import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import type {
  AlbumRules,
  FacePerson,
  SmartAlbumSummary,
  StoryChapter,
  StoryMoment,
  MomentImage,
  StoryGenerateResult,
  SearchResult,
} from '../global'
import { SearchLightbox } from './SearchLightbox'

type PresetDefinition = {
  name: string
  description: string
  params: string[]
}

type StatusBanner = {
  tone: 'success' | 'error'
  text: string
}

const FALLBACK_PRESETS: Record<string, PresetDefinition> = {
  year_in_review: {
    name: 'Year in Review',
    description: 'All photos from a specific year',
    params: ['year'],
  },
  on_this_day: {
    name: 'On This Day',
    description: 'Photos from the same date across all years',
    params: ['month', 'day'],
  },
  person_timeline: {
    name: 'Person Timeline',
    description: 'All photos of a specific person',
    params: ['person_id', 'person_name'],
  },
  growth_story: {
    name: 'Growth Story',
    description: 'A year-by-year story for a specific person',
    params: ['person_id', 'person_name'],
  },
  together: {
    name: 'Together',
    description: 'Co-occurrence album for multiple people',
    params: ['person_ids', 'person_names'],
  },
  location: {
    name: 'Location Story',
    description: 'All photos from a location',
    params: ['country', 'city'],
  },
}

function sanitizeFileName(value: string): string {
  const cleaned = value.replace(/[<>:"/\\|?*\u0000-\u001F]/g, ' ').trim().replace(/^\.+|\.+$/g, '')
  return cleaned.length > 0 ? cleaned.replace(/\s+/g, ' ') : 'story-export'
}

function mergeThumbnailMap(thumbs: Record<string, string>): Record<number, string> {
  const next: Record<number, string> = {}
  for (const [key, value] of Object.entries(thumbs)) {
    const imageId = Number(key)
    if (Number.isFinite(imageId)) {
      next[imageId] = value
    }
  }
  return next
}

// ── Rule Editor (shared between create and edit) ─────────────────────────────

type RuleEntry =
  | { type: 'person'; person_ids: number[]; mode: 'any' | 'all' }
  | { type: 'date_range'; start?: string; end?: string }
  | { type: 'location'; country: string; city?: string }
  | { type: 'keyword'; values: string[] }

function parseRulesFromAlbum(albumRules: AlbumRules): { match: 'all' | 'any'; entries: RuleEntry[] } {
  const entries: RuleEntry[] = []
  for (const raw of albumRules.rules) {
    const t = raw.type as string
    if (t === 'person') {
      entries.push({
        type: 'person',
        person_ids: (raw.person_ids as number[]) ?? [],
        mode: (raw.mode as 'any' | 'all') ?? 'any',
      })
    } else if (t === 'date_range') {
      entries.push({
        type: 'date_range',
        start: (raw.start as string) ?? undefined,
        end: (raw.end as string) ?? undefined,
      })
    } else if (t === 'location') {
      entries.push({
        type: 'location',
        country: (raw.country as string) ?? '',
        city: (raw.city as string) ?? undefined,
      })
    } else if (t === 'keyword') {
      entries.push({
        type: 'keyword',
        values: (raw.values as string[]) ?? [],
      })
    }
  }
  return { match: albumRules.match, entries }
}

function rulesToAlbumRules(match: 'all' | 'any', entries: RuleEntry[]): AlbumRules {
  const rules: Array<Record<string, unknown>> = []
  for (const e of entries) {
    if (e.type === 'person' && e.person_ids.length > 0) {
      rules.push({ type: 'person', person_ids: e.person_ids, mode: e.mode })
    } else if (e.type === 'date_range' && (e.start || e.end)) {
      rules.push({
        type: 'date_range',
        ...(e.start && { start: e.start }),
        ...(e.end && { end: e.end }),
      })
    } else if (e.type === 'location' && e.country.trim()) {
      rules.push({
        type: 'location',
        country: e.country.trim(),
        ...(e.city?.trim() && { city: e.city.trim() }),
      })
    } else if (e.type === 'keyword' && e.values.length > 0) {
      rules.push({ type: 'keyword', values: e.values })
    }
  }
  return { match, rules }
}

const RULE_TYPE_LABELS: Record<RuleEntry['type'], string> = {
  person: 'Person',
  date_range: 'Date Range',
  location: 'Location',
  keyword: 'Keyword',
}

// ── Searchable Person Picker (multi-select with search + selected-first) ─────

function PersonPickerMulti({
  persons,
  selectedIds,
  onChange,
}: {
  persons: Array<{ id: number; name: string }>
  selectedIds: number[]
  onChange: (ids: number[]) => void
}) {
  const [search, setSearch] = useState('')
  const query = search.toLowerCase().trim()

  // Selected first, then alphabetical within each group
  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds])
  const sorted = [...persons].sort((a, b) => {
    const aSelected = selectedSet.has(a.id) ? 0 : 1
    const bSelected = selectedSet.has(b.id) ? 0 : 1
    if (aSelected !== bSelected) return aSelected - bSelected
    return a.name.localeCompare(b.name)
  })

  const filtered = query
    ? sorted.filter((p) => p.name.toLowerCase().includes(query))
    : sorted

  return (
    <div>
      {/* Search input */}
      <div className="relative mb-1.5">
        <svg className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11A6 6 0 115 11a6 6 0 0112 0z" />
        </svg>
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={`Search ${persons.length} people…`}
          className="w-full pl-7 pr-2 py-1.5 rounded bg-neutral-700 text-white text-xs border border-neutral-600 placeholder:text-neutral-500 focus:border-blue-500 focus:outline-none"
        />
        {selectedIds.length > 0 && (
          <span className="absolute right-2 top-1/2 -translate-y-1/2 px-1.5 py-0.5 rounded-full bg-blue-600 text-[10px] text-white font-medium tabular-nums">
            {selectedIds.length}
          </span>
        )}
      </div>

      {/* Scrollable list */}
      <div className="max-h-40 overflow-y-auto bg-neutral-800 rounded p-1.5">
        {persons.length === 0 && (
          <span className="text-xs text-neutral-500 px-1">No persons found. Analyze faces first.</span>
        )}
        {filtered.length === 0 && persons.length > 0 && (
          <span className="text-xs text-neutral-500 px-1">No match for &ldquo;{search}&rdquo;</span>
        )}
        {filtered.map((p) => {
          const checked = selectedIds.includes(p.id)
          return (
            <label
              key={p.id}
              className={`flex items-center gap-2 px-1.5 py-1 rounded text-sm cursor-pointer transition-colors ${
                checked
                  ? 'text-white bg-blue-600/20'
                  : 'text-neutral-300 hover:bg-neutral-700/60'
              }`}
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={(ev) => {
                  const next = ev.target.checked
                    ? [...selectedIds, p.id]
                    : selectedIds.filter((x) => x !== p.id)
                  onChange(next)
                }}
                className="accent-blue-500"
              />
              <span className="truncate">{p.name}</span>
            </label>
          )
        })}
      </div>
    </div>
  )
}

// ── Searchable Person Picker (single-select with search) ─────────────────────

function PersonPickerSingle({
  persons,
  selectedId,
  onChange,
  placeholder,
}: {
  persons: Array<{ id: number; name: string }>
  selectedId: number | ''
  onChange: (id: number | '') => void
  placeholder?: string
}) {
  const [search, setSearch] = useState('')
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const query = search.toLowerCase().trim()

  const selected = persons.find((p) => p.id === selectedId)

  const filtered = query
    ? persons.filter((p) => p.name.toLowerCase().includes(query)).sort((a, b) => a.name.localeCompare(b.name))
    : [...persons].sort((a, b) => a.name.localeCompare(b.name))

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handleClick = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
        setSearch('')
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open])

  return (
    <div ref={containerRef} className="relative">
      {/* Display / trigger */}
      <div
        className="flex items-center gap-1.5 w-full px-3 py-1.5 rounded bg-neutral-700 text-sm border border-neutral-600 cursor-pointer hover:border-neutral-500 transition-colors"
        onClick={() => { setOpen(!open); setTimeout(() => inputRef.current?.focus(), 0) }}
      >
        {open ? (
          <input
            ref={inputRef}
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={`Search ${persons.length} people…`}
            className="flex-1 bg-transparent text-white placeholder:text-neutral-500 outline-none text-sm min-w-0"
            onClick={(e) => e.stopPropagation()}
          />
        ) : (
          <span className={`flex-1 truncate ${selected ? 'text-white' : 'text-neutral-500'}`}>
            {selected ? selected.name : (placeholder ?? 'Select a person…')}
          </span>
        )}
        <svg className={`w-3.5 h-3.5 text-neutral-500 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </div>

      {/* Dropdown */}
      {open && (
        <div className="absolute z-20 left-0 right-0 mt-1 max-h-48 overflow-y-auto bg-neutral-800 rounded border border-neutral-600 shadow-xl">
          {filtered.length === 0 && (
            <div className="px-3 py-2 text-xs text-neutral-500">
              {query ? `No match for "${search}"` : 'No persons found'}
            </div>
          )}
          {filtered.map((p) => (
            <button
              key={p.id}
              className={`w-full text-left px-3 py-1.5 text-sm transition-colors ${
                p.id === selectedId
                  ? 'text-white bg-blue-600/30'
                  : 'text-neutral-300 hover:bg-neutral-700'
              }`}
              onClick={() => {
                onChange(p.id)
                setOpen(false)
                setSearch('')
              }}
            >
              {p.name}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function RuleEditor({
  entry,
  index,
  persons,
  onChange,
  onRemove,
}: {
  entry: RuleEntry
  index: number
  persons: Array<{ id: number; name: string }>
  onChange: (i: number, e: RuleEntry) => void
  onRemove: (i: number) => void
}) {
  return (
    <div className="bg-neutral-900 rounded p-3 relative group">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-blue-400 uppercase tracking-wide">
          {RULE_TYPE_LABELS[entry.type]}
        </span>
        <button
          onClick={() => onRemove(index)}
          className="text-neutral-600 hover:text-red-400 text-xs px-1"
          title="Remove rule"
        >
          ✕
        </button>
      </div>

      {entry.type === 'person' && (
        <>
          <PersonPickerMulti
            persons={persons}
            selectedIds={entry.person_ids}
            onChange={(next) => onChange(index, { ...entry, person_ids: next })}
          />
          <select
            value={entry.mode}
            onChange={(e) => onChange(index, { ...entry, mode: e.target.value as 'any' | 'all' })}
            className="px-2 py-0.5 rounded bg-neutral-700 text-white text-xs border border-neutral-600"
          >
            <option value="any">Any of the selected</option>
            <option value="all">All together (co-occurrence)</option>
          </select>
        </>
      )}

      {entry.type === 'date_range' && (
        <div className="flex gap-2">
          <div className="flex-1">
            <label className="block text-xs text-neutral-500 mb-0.5">Start</label>
            <input
              type="date"
              value={entry.start ?? ''}
              onChange={(e) => onChange(index, { ...entry, start: e.target.value || undefined })}
              className="w-full px-2 py-1 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
            />
          </div>
          <div className="flex-1">
            <label className="block text-xs text-neutral-500 mb-0.5">End</label>
            <input
              type="date"
              value={entry.end ?? ''}
              onChange={(e) => onChange(index, { ...entry, end: e.target.value || undefined })}
              className="w-full px-2 py-1 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
            />
          </div>
        </div>
      )}

      {entry.type === 'location' && (
        <div className="flex gap-2">
          <div className="flex-1">
            <label className="block text-xs text-neutral-500 mb-0.5">Country code</label>
            <input
              value={entry.country}
              onChange={(e) => onChange(index, { ...entry, country: e.target.value })}
              className="w-full px-2 py-1 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
              placeholder="US"
            />
          </div>
          <div className="flex-1">
            <label className="block text-xs text-neutral-500 mb-0.5">City (optional)</label>
            <input
              value={entry.city ?? ''}
              onChange={(e) => onChange(index, { ...entry, city: e.target.value || undefined })}
              className="w-full px-2 py-1 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
              placeholder="Tokyo"
            />
          </div>
        </div>
      )}

      {entry.type === 'keyword' && (
        <div>
          <label className="block text-xs text-neutral-500 mb-0.5">Keywords (comma separated)</label>
          <input
            value={entry.values.join(', ')}
            onChange={(e) => {
              const vals = e.target.value
                .split(',')
                .map((v) => v.trim())
                .filter(Boolean)
              onChange(index, { ...entry, values: vals })
            }}
            className="w-full px-2 py-1 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
            placeholder="beach, sunset"
          />
        </div>
      )}
    </div>
  )
}

// ── Edit Album Dialog ─────────────────────────────────────────────────────────

function EditAlbumDialog({
  album,
  onClose,
  onSaved,
  persons,
}: {
  album: SmartAlbumSummary
  onClose: () => void
  onSaved: () => void
  persons: Array<{ id: number; name: string }>
}) {
  const [name, setName] = useState(album.name)
  const [description, setDescription] = useState(album.description ?? '')
  const [matchMode, setMatchMode] = useState<'all' | 'any'>('all')
  const [ruleEntries, setRuleEntries] = useState<RuleEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [rulesChanged, setRulesChanged] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const originalRulesRef = useRef<string>('')

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const full = await window.api.albumsGet(album.id)
        if (cancelled) return
        if (full.rules) {
          const parsed = parseRulesFromAlbum(full.rules)
          setMatchMode(parsed.match)
          setRuleEntries(parsed.entries)
          originalRulesRef.current = JSON.stringify(full.rules)
        }
      } catch (err) {
        console.error('Failed to load album rules:', err)
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => { cancelled = true }
  }, [album.id])

  const handleRuleChange = (i: number, entry: RuleEntry) => {
    const next = [...ruleEntries]
    next[i] = entry
    setRuleEntries(next)
    setRulesChanged(true)
  }

  const handleRuleRemove = (i: number) => {
    setRuleEntries(ruleEntries.filter((_, idx) => idx !== i))
    setRulesChanged(true)
  }

  const handleAddRule = (type: RuleEntry['type']) => {
    const defaults: Record<RuleEntry['type'], RuleEntry> = {
      person: { type: 'person', person_ids: [], mode: 'any' },
      date_range: { type: 'date_range' },
      location: { type: 'location', country: '' },
      keyword: { type: 'keyword', values: [] },
    }
    setRuleEntries([...ruleEntries, defaults[type]])
    setRulesChanged(true)
  }

  const handleSave = async () => {
    if (!name.trim()) return
    setSaving(true)
    setError(null)
    try {
      const params: Parameters<typeof window.api.albumsUpdate>[0] = {
        album_id: album.id,
        name: name.trim(),
        description: description.trim() || undefined,
      }
      if (rulesChanged) {
        params.rules = rulesToAlbumRules(matchMode, ruleEntries)
      }
      const result = await window.api.albumsUpdate(params)
      if ('error' in result) throw new Error(result.error)
      onSaved()
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save album.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" role="dialog" aria-modal="true" onClick={onClose}>
      <div
        className="bg-neutral-800 rounded-lg shadow-xl w-[500px] max-h-[85vh] overflow-y-auto p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-semibold text-white mb-4">Edit Album</h2>

        <label className="block text-sm text-neutral-400 mb-1">Name</label>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600 focus:border-blue-500 outline-none mb-3"
          autoFocus
        />

        <label className="block text-sm text-neutral-400 mb-1">Description</label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={2}
          className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600 focus:border-blue-500 outline-none mb-4 resize-none"
        />

        {/* ── Rules Section ── */}
        <div className="border-t border-neutral-700 pt-3 mb-3">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-neutral-300">Search Criteria</h3>
            <select
              value={matchMode}
              onChange={(e) => { setMatchMode(e.target.value as 'all' | 'any'); setRulesChanged(true) }}
              className="px-2 py-0.5 rounded bg-neutral-700 text-white text-xs border border-neutral-600"
            >
              <option value="all">Match ALL rules</option>
              <option value="any">Match ANY rule</option>
            </select>
          </div>

          {loading ? (
            <div className="text-xs text-neutral-500 py-4 text-center">Loading rules…</div>
          ) : (
            <div className="space-y-2 mb-3">
              {ruleEntries.map((entry, i) => (
                <RuleEditor
                  key={`${entry.type}-${i}`}
                  entry={entry}
                  index={i}
                  persons={persons}
                  onChange={handleRuleChange}
                  onRemove={handleRuleRemove}
                />
              ))}

              {ruleEntries.length === 0 && (
                <div className="text-xs text-neutral-500 text-center py-2">
                  No rules defined. Add criteria below.
                </div>
              )}
            </div>
          )}

          {/* Add rule buttons */}
          <div className="flex flex-wrap gap-1.5">
            <span className="text-xs text-neutral-500 leading-6">Add:</span>
            {(['person', 'date_range', 'location', 'keyword'] as const).map((type) => (
              <button
                key={type}
                onClick={() => handleAddRule(type)}
                className="px-2 py-0.5 text-xs rounded bg-neutral-700 text-neutral-300 hover:bg-neutral-600 border border-neutral-600"
              >
                + {RULE_TYPE_LABELS[type]}
              </button>
            ))}
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-4 pt-3 border-t border-neutral-700">
          {error && (
            <p className="flex-1 text-xs text-red-400 self-center">{error}</p>
          )}
          <button onClick={onClose} className="px-3 py-1.5 text-sm rounded bg-neutral-700 text-neutral-300 hover:bg-neutral-600">
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving || !name.trim()}
            className="px-3 py-1.5 text-sm rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Delete Confirm Dialog ────────────────────────────────────────────────────

function DeleteConfirmDialog({
  albumName,
  onConfirm,
  onCancel,
}: {
  albumName: string
  onConfirm: () => void
  onCancel: () => void
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" role="dialog" aria-modal="true" onClick={onCancel}>
      <div className="bg-neutral-800 rounded-lg shadow-xl w-[360px] p-5" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-lg font-semibold text-white mb-2">Delete Album</h2>
        <p className="text-sm text-neutral-400 mb-4">
          Are you sure you want to delete <span className="text-white font-medium">"{albumName}"</span>?
          This will remove all chapters, moments, and story data. Your original photos are not affected.
        </p>
        <div className="flex justify-end gap-2">
          <button onClick={onCancel} className="px-3 py-1.5 text-sm rounded bg-neutral-700 text-neutral-300 hover:bg-neutral-600">
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="px-3 py-1.5 text-sm rounded bg-red-600 text-white hover:bg-red-500"
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Album List Panel ────────────────────────────────────────────────────────

function AlbumListPanel({
  albums,
  selectedId,
  onSelect,
  onRefresh,
  onCreate,
  onCreatePreset,
  onEdit,
  onDelete,
}: {
  albums: SmartAlbumSummary[]
  selectedId: string | null
  onSelect: (id: string) => void
  onRefresh: () => void
  onCreate: () => void
  onCreatePreset: () => void
  onEdit: (album: SmartAlbumSummary) => void
  onDelete: (album: SmartAlbumSummary) => void
}) {
  return (
    <div className="flex flex-col h-full border-r border-neutral-700">
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-700">
        <span className="text-sm font-medium text-neutral-300">Albums</span>
        <div className="flex gap-1">
          <button
            onClick={onRefresh}
            className="px-2 py-0.5 text-xs rounded bg-neutral-700 text-neutral-300 hover:bg-neutral-600"
            title="Refresh list"
          >
            ↻
          </button>
          <button
            onClick={onCreatePreset}
            className="px-2 py-0.5 text-xs rounded bg-violet-600 text-white hover:bg-violet-500"
            title="Create preset album"
          >
            Preset
          </button>
          <button
            onClick={onCreate}
            className="px-2 py-0.5 text-xs rounded bg-blue-600 text-white hover:bg-blue-500"
          >
            + New
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto">
        {albums.length === 0 && (
          <div className="p-4 text-sm text-neutral-500 text-center">
            No albums yet. Create one manually or from a preset.
          </div>
        )}
        {albums.map((a) => (
          <div
            key={a.id}
            className={`group relative w-full text-left px-3 py-2.5 border-b border-neutral-800 transition-colors cursor-pointer ${
              selectedId === a.id
                ? 'bg-neutral-700/60 text-white'
                : 'text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200'
            }`}
            onClick={() => onSelect(a.id)}
          >
            <div className="text-sm font-medium truncate pr-12">{a.name}</div>
            <div className="text-xs text-neutral-500 mt-0.5">
              {a.item_count} images · {a.chapter_count} chapters
            </div>
            {a.description && (
              <div className="text-xs text-neutral-500 mt-0.5 truncate">{a.description}</div>
            )}
            {/* Edit / Delete buttons — visible on hover */}
            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={(e) => { e.stopPropagation(); onEdit(a) }}
                className="p-1 rounded text-neutral-400 hover:text-blue-400 hover:bg-neutral-700"
                title="Edit album"
              >
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M11.5 1.5l3 3L5 14H2v-3L11.5 1.5z" />
                </svg>
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); onDelete(a) }}
                className="p-1 rounded text-neutral-400 hover:text-red-400 hover:bg-neutral-700"
                title="Delete album"
              >
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M2 4h12M5 4V2h6v2M6 7v5M10 7v5M3 4l1 10h8l1-10" />
                </svg>
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Create Album Dialog ──────────────────────────────────────────────────────

function CreateAlbumDialog({
  open,
  onClose,
  onCreated,
  persons,
}: {
  open: boolean
  onClose: () => void
  onCreated: (id: string) => void
  persons: Array<{ id: number; name: string }>
}) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [matchMode, setMatchMode] = useState<'all' | 'any'>('all')
  const [ruleEntries, setRuleEntries] = useState<RuleEntry[]>([])
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  if (!open) return null

  const handleRuleChange = (i: number, entry: RuleEntry) => {
    const next = [...ruleEntries]
    next[i] = entry
    setRuleEntries(next)
  }

  const handleRuleRemove = (i: number) => {
    setRuleEntries(ruleEntries.filter((_, idx) => idx !== i))
  }

  const handleAddRule = (type: RuleEntry['type']) => {
    const defaults: Record<RuleEntry['type'], RuleEntry> = {
      person: { type: 'person', person_ids: [], mode: 'any' },
      date_range: { type: 'date_range' },
      location: { type: 'location', country: '' },
      keyword: { type: 'keyword', values: [] },
    }
    setRuleEntries([...ruleEntries, defaults[type]])
  }

  const handleCreate = async () => {
    if (!name.trim()) return
    setCreating(true)
    setError(null)
    try {
      const albumRules = rulesToAlbumRules(matchMode, ruleEntries)
      const result = await window.api.albumsCreate({
        name: name.trim(),
        rules: albumRules,
        description: description.trim() || undefined,
      })
      onCreated(result.id)
      setName('')
      setDescription('')
      setRuleEntries([])
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create album.')
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" role="dialog" aria-modal="true">
      <div className="bg-neutral-800 rounded-lg shadow-xl w-[500px] max-h-[80vh] overflow-y-auto p-5">
        <h2 className="text-lg font-semibold text-white mb-4">Create Smart Album</h2>

        <label className="block text-sm text-neutral-400 mb-1">Name</label>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600 focus:border-blue-500 outline-none mb-3"
          placeholder="e.g. Alice & Bob Together"
        />

        <label className="block text-sm text-neutral-400 mb-1">Description (optional)</label>
        <input
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600 focus:border-blue-500 outline-none mb-3"
        />

        {/* ── Rules Section ── */}
        <div className="border-t border-neutral-700 pt-3 mb-3">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-neutral-300">Search Criteria</h3>
            <select
              value={matchMode}
              onChange={(e) => setMatchMode(e.target.value as 'all' | 'any')}
              className="px-2 py-0.5 rounded bg-neutral-700 text-white text-xs border border-neutral-600"
            >
              <option value="all">Match ALL rules</option>
              <option value="any">Match ANY rule</option>
            </select>
          </div>

          <div className="space-y-2 mb-3">
            {ruleEntries.map((entry, i) => (
              <RuleEditor
                key={`${entry.type}-${i}`}
                entry={entry}
                index={i}
                persons={persons}
                onChange={handleRuleChange}
                onRemove={handleRuleRemove}
              />
            ))}

            {ruleEntries.length === 0 && (
              <div className="text-xs text-neutral-500 text-center py-2">
                No rules defined. Add criteria below.
              </div>
            )}
          </div>

          {/* Add rule buttons */}
          <div className="flex flex-wrap gap-1.5">
            <span className="text-xs text-neutral-500 leading-6">Add:</span>
            {(['person', 'date_range', 'location', 'keyword'] as const).map((type) => (
              <button
                key={type}
                onClick={() => handleAddRule(type)}
                className="px-2 py-0.5 text-xs rounded bg-neutral-700 text-neutral-300 hover:bg-neutral-600 border border-neutral-600"
              >
                + {RULE_TYPE_LABELS[type]}
              </button>
            ))}
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-4">
          {error && (
            <p className="flex-1 text-xs text-red-400 self-center">{error}</p>
          )}
          <button onClick={onClose} className="px-3 py-1.5 text-sm rounded bg-neutral-700 text-neutral-300 hover:bg-neutral-600">
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={creating || !name.trim()}
            className="px-3 py-1.5 text-sm rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
          >
            {creating ? 'Creating...' : 'Create Album'}
          </button>
        </div>
      </div>
    </div>
  )
}

function PresetAlbumDialog({
  open,
  onClose,
  onCreated,
  persons,
  presets,
}: {
  open: boolean
  onClose: () => void
  onCreated: (id: string) => void
  persons: FacePerson[]
  presets: Record<string, PresetDefinition>
}) {
  const presetKeys = Object.keys(presets)
  const [preset, setPreset] = useState<string>(presetKeys[0] ?? 'year_in_review')
  const [year, setYear] = useState(new Date().getFullYear() - 1)
  const [month, setMonth] = useState(new Date().getMonth() + 1)
  const [day, setDay] = useState(new Date().getDate())
  const [personId, setPersonId] = useState<number | ''>('')
  const [secondPersonId, setSecondPersonId] = useState<number | ''>('')
  const [country, setCountry] = useState('')
  const [city, setCity] = useState('')
  const [creating, setCreating] = useState(false)

  useEffect(() => {
    if (!open) return
    const firstPerson = persons[0]?.id ?? ''
    const secondPerson = persons[1]?.id ?? persons[0]?.id ?? ''
    setPreset(Object.keys(presets)[0] ?? 'year_in_review')
    setPersonId(firstPerson)
    setSecondPersonId(secondPerson)
  }, [open, persons, presets])

  if (!open) return null

  const currentPreset = presets[preset] ?? FALLBACK_PRESETS[preset]

  const canCreate = (() => {
    if (preset === 'year_in_review' || preset === 'on_this_day') return true
    if (preset === 'person_timeline' || preset === 'growth_story') return personId !== ''
    if (preset === 'together') return personId !== '' && secondPersonId !== '' && personId !== secondPersonId
    if (preset === 'location') return country.trim().length > 0
    return false
  })()

  const handleCreate = async () => {
    if (!canCreate) return
    setCreating(true)
    try {
      const params: Record<string, unknown> = { preset }
      if (preset === 'year_in_review') {
        params.year = year
      } else if (preset === 'on_this_day') {
        params.month = month
        params.day = day
      } else if (preset === 'person_timeline' || preset === 'growth_story') {
        const person = persons.find((p) => p.id === personId)
        params.person_id = personId
        params.person_name = person?.name
      } else if (preset === 'together') {
        const first = persons.find((p) => p.id === personId)
        const second = persons.find((p) => p.id === secondPersonId)
        params.person_ids = [personId, secondPersonId]
        params.person_names = [first?.name ?? `Person ${personId}`, second?.name ?? `Person ${secondPersonId}`]
      } else if (preset === 'location') {
        params.country = country.trim().toUpperCase()
        if (city.trim()) params.city = city.trim()
      }

      const result = await window.api.albumsCreatePreset(params)
      if ('error' in result) throw new Error(result.error)
      onCreated(result.id)
      onClose()
    } catch (err) {
      console.error('Failed to create preset album:', err)
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" role="dialog" aria-modal="true">
      <div className="bg-neutral-800 rounded-lg shadow-xl w-[460px] max-h-[80vh] overflow-y-auto p-5">
        <h2 className="text-lg font-semibold text-white mb-4">Create Preset Album</h2>

        <label className="block text-sm text-neutral-400 mb-1">Preset</label>
        <select
          value={preset}
          onChange={(e) => setPreset(e.target.value)}
          className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600 mb-2"
        >
          {presetKeys.map((key) => (
            <option key={key} value={key}>
              {presets[key]?.name ?? key}
            </option>
          ))}
        </select>

        {currentPreset && (
          <p className="text-xs text-neutral-500 mb-3">{currentPreset.description}</p>
        )}

        {preset === 'year_in_review' && (
          <div className="mb-3">
            <label className="block text-sm text-neutral-400 mb-1">Year</label>
            <input
              type="number"
              value={year}
              onChange={(e) => setYear(Number(e.target.value) || new Date().getFullYear() - 1)}
              className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
            />
          </div>
        )}

        {preset === 'on_this_day' && (
          <div className="grid grid-cols-2 gap-2 mb-3">
            <div>
              <label className="block text-sm text-neutral-400 mb-1">Month</label>
              <input
                type="number"
                min={1}
                max={12}
                value={month}
                onChange={(e) => setMonth(Math.min(12, Math.max(1, Number(e.target.value) || 1)))}
                className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
              />
            </div>
            <div>
              <label className="block text-sm text-neutral-400 mb-1">Day</label>
              <input
                type="number"
                min={1}
                max={31}
                value={day}
                onChange={(e) => setDay(Math.min(31, Math.max(1, Number(e.target.value) || 1)))}
                className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
              />
            </div>
          </div>
        )}

        {(preset === 'person_timeline' || preset === 'growth_story') && (
          <div className="mb-3">
            <label className="block text-sm text-neutral-400 mb-1">Person</label>
            <PersonPickerSingle
              persons={persons.map((p) => ({ id: p.id, name: p.name }))}
              selectedId={personId}
              onChange={(id) => setPersonId(id)}
            />
          </div>
        )}

        {preset === 'together' && (
          <div className="grid grid-cols-2 gap-2 mb-3">
            <div>
              <label className="block text-sm text-neutral-400 mb-1">First person</label>
              <PersonPickerSingle
                persons={persons.map((p) => ({ id: p.id, name: p.name }))}
                selectedId={personId}
                onChange={(id) => setPersonId(id)}
                placeholder="Select first…"
              />
            </div>
            <div>
              <label className="block text-sm text-neutral-400 mb-1">Second person</label>
              <PersonPickerSingle
                persons={persons.map((p) => ({ id: p.id, name: p.name }))}
                selectedId={secondPersonId}
                onChange={(id) => setSecondPersonId(id)}
                placeholder="Select second…"
              />
            </div>
          </div>
        )}

        {preset === 'location' && (
          <>
            <div className="mb-3">
              <label className="block text-sm text-neutral-400 mb-1">Country code</label>
              <input
                value={country}
                onChange={(e) => setCountry(e.target.value)}
                className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
                placeholder="US"
              />
            </div>
            <div className="mb-3">
              <label className="block text-sm text-neutral-400 mb-1">City (optional)</label>
              <input
                value={city}
                onChange={(e) => setCity(e.target.value)}
                className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
                placeholder="Tokyo"
              />
            </div>
          </>
        )}

        <div className="flex justify-end gap-2 mt-4">
          <button onClick={onClose} className="px-3 py-1.5 text-sm rounded bg-neutral-700 text-neutral-300 hover:bg-neutral-600">
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={creating || !canCreate}
            className="px-3 py-1.5 text-sm rounded bg-violet-600 text-white hover:bg-violet-500 disabled:opacity-50"
          >
            {creating ? 'Creating...' : 'Create Preset'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Story Chapter Timeline ──────────────────────────────────────────────────

/** Group chapters by year for visual sectioning. */
function groupChaptersByYear(chapters: StoryChapter[]): Record<string, StoryChapter[]> {
  const groups: Record<string, StoryChapter[]> = {}
  for (const ch of chapters) {
    let year = 'Unknown'
    if (ch.start_date) {
      try {
        const d = new Date(ch.start_date)
        if (!isNaN(d.getTime())) year = String(d.getFullYear())
      } catch {
        /* ignore */
      }
    }
    if (!groups[year]) groups[year] = []
    groups[year].push(ch)
  }
  return groups
}

function formatChapterDateRange(start: string | null, end: string | null): string {
  if (!start) return ''
  try {
    const s = new Date(start)
    if (isNaN(s.getTime())) return ''
    const opts: Intl.DateTimeFormatOptions = { month: 'short', day: 'numeric' }
    if (!end) return s.toLocaleDateString(undefined, opts)
    const e = new Date(end)
    if (isNaN(e.getTime())) return s.toLocaleDateString(undefined, opts)
    if (s.toDateString() === e.toDateString()) return s.toLocaleDateString(undefined, opts)
    return `${s.toLocaleDateString(undefined, opts)} – ${e.toLocaleDateString(undefined, opts)}`
  } catch {
    return ''
  }
}

/**
 * Collage cell — renders a thumbnail or placeholder at the given grid position.
 * Uses CSS grid placement so each cell can span different rows/columns.
 */
function CollageCell({
  img,
  thumb,
  style,
  onClick,
}: {
  img: MomentImage
  thumb: string | undefined
  style: React.CSSProperties
  onClick?: (img: MomentImage) => void
}) {
  return (
    <div
      className={`overflow-hidden relative${onClick ? ' cursor-pointer hover:brightness-110 transition-[filter]' : ''}`}
      style={style}
      onClick={onClick ? (e) => { e.stopPropagation(); onClick(img) } : undefined}
    >
      {thumb ? (
        <img
          src={thumb}
          alt=""
          className="absolute inset-0 w-full h-full object-cover object-top"
          loading="lazy"
          decoding="async"
        />
      ) : (
        <div className="absolute inset-0 bg-neutral-800" />
      )}
    </div>
  )
}

/**
 * Layout definition for collage — each slot specifies CSS grid placement.
 * gridColumn/gridRow use CSS shorthand like "1 / 3" meaning "start at 1, end at 3".
 */
type CollageLayout = {
  columns: string          // grid-template-columns
  rows: string             // grid-template-rows
  slots: React.CSSProperties[]
}

const MAX_VISIBLE_MOMENT_IMAGES = 20
const EXPANDED_MOMENT_MIN_WIDTH = 400

function getLayout(count: number): CollageLayout {
  const columns =
    count <= 1 ? 1
      : count === 2 ? 2
        : count <= 4 ? 2
          : count <= 8 ? 3
            : count <= 14 ? 4
              : 5
  // Use aspect-ratio instead of fixed pixel heights so cells scale
  // proportionally with container width and never become overly wide/short.
  const cellAspect =
    count === 1 ? '3/2'
      : count === 2 ? '3/2'
        : count <= 4 ? '4/3'
          : '1/1'
  const rowCount = Math.ceil(count / columns)

  return {
    columns: `repeat(${columns}, minmax(0, 1fr))`,
    rows: `repeat(${rowCount}, auto)`,
    slots: Array.from({ length: count }, (_, index) => ({
      gridColumn: `${(index % columns) + 1}`,
      gridRow: `${Math.floor(index / columns) + 1}`,
      aspectRatio: cellAspect,
    })),
  }
}

function MomentCollage({
  images,
  thumbs,
  onImageClick,
}: {
  images: MomentImage[]
  thumbs: Record<number, string>
  onImageClick?: (img: MomentImage, siblings: MomentImage[]) => void
}) {
  if (images.length === 0) return null

  const hero = images.find((img) => img.is_hero) ?? images[0]
  // Put hero first, then the rest in original order
  const ordered = [hero, ...images.filter((img) => img.image_id !== hero.image_id)]

  const displayCount = Math.min(ordered.length, MAX_VISIBLE_MOMENT_IMAGES)
  const overflow = ordered.length - displayCount
  const layout = getLayout(displayCount)
  const shown = ordered.slice(0, displayCount)

  // Wrap click to include all images as siblings for lightbox navigation
  const handleClick = onImageClick
    ? (img: MomentImage) => onImageClick(img, images)
    : undefined

  return (
    <div
      className="rounded-xl overflow-hidden border border-neutral-800/70 bg-neutral-950/40 p-1.5"
      style={{
        display: 'grid',
        gridTemplateColumns: layout.columns,
        gridTemplateRows: layout.rows,
        gap: '6px',
      }}
    >
      {shown.map((img, i) => {
        const isLast = i === displayCount - 1 && overflow > 0
        const slot = layout.slots[i]
        if (isLast) {
          // Last cell with +N overflow badge
          return (
            <div
              key={img.image_id}
              className={`overflow-hidden relative${handleClick ? ' cursor-pointer hover:brightness-110 transition-[filter]' : ''}`}
              style={slot}
              onClick={handleClick ? (e) => { e.stopPropagation(); handleClick(img) } : undefined}
            >
              {thumbs[img.image_id] ? (
                <img
                  src={thumbs[img.image_id]}
                  alt=""
                  className="absolute inset-0 w-full h-full object-cover object-top"
                  loading="lazy"
                  decoding="async"
                />
              ) : (
                <div className="absolute inset-0 bg-neutral-800" />
              )}
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                <span className="text-white text-lg font-semibold">+{overflow}</span>
              </div>
            </div>
          )
        }
        return (
          <CollageCell
            key={img.image_id}
            img={img}
            thumb={thumbs[img.image_id]}
            style={slot}
            onClick={handleClick}
          />
        )
      })}
    </div>
  )
}

type ViewMode = 'quilted' | 'zigzag'

/** Shared expanded-chapter detail panel (used by both view modes). */
function ExpandedChapterDetail({
  chapter,
  dateRange,
  moments,
  momentImages,
  heroThumbs,
  onImageClick,
  onCollapse,
  colWidth = 350,
}: {
  chapter: StoryChapter
  dateRange: string
  moments: StoryMoment[]
  momentImages: Record<string, MomentImage[]>
  heroThumbs: Record<number, string>
  onImageClick?: (img: MomentImage, siblings: MomentImage[]) => void
  onCollapse?: () => void
  colWidth?: number
}){
  const detailMinWidth = Math.max(EXPANDED_MOMENT_MIN_WIDTH, Math.min(520, colWidth + 80))
  const detailMaxWidth = Math.max(detailMinWidth, Math.min(860, detailMinWidth + 220))
  const isSingleMoment = moments.length === 1
  const singleMomentImageCount = isSingleMoment ? moments[0]?.image_count ?? 0 : 0
  const isSparseSingleMoment = isSingleMoment && singleMomentImageCount <= 3
  const singleMomentWidth = isSparseSingleMoment
    ? Math.min(760, detailMaxWidth + 80)
    : Math.max(920, Math.min(1180, colWidth * 4 + 120))
  const detailColumns = isSingleMoment
    ? `minmax(0, min(${singleMomentWidth}px, 100%))`
    : `repeat(auto-fit, minmax(min(${detailMinWidth}px, 100%), min(${detailMaxWidth}px, 100%)))`

  return (
    <div className="p-5" onClick={(e) => e.stopPropagation()}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-semibold text-white leading-tight">
            {chapter.title || 'Untitled Chapter'}
          </h3>
          <div className="flex items-center gap-2 mt-1 text-xs text-neutral-500">
            {chapter.location && <span className="text-blue-400">📍 {chapter.location}</span>}
            {dateRange && <span>{dateRange}</span>}
            <span>{chapter.image_count} photos · {chapter.moment_count} moments</span>
          </div>
          {chapter.summary && (
            <p className="text-sm text-neutral-400 italic leading-relaxed mt-2 max-w-2xl">
              &ldquo;{chapter.summary}&rdquo;
            </p>
          )}
        </div>
        <button
          className="text-xs text-neutral-500 hover:text-neutral-300 ml-3 shrink-0 transition-colors"
          onClick={(e) => { e.stopPropagation(); onCollapse?.() }}
        >
          ▾ collapse
        </button>
      </div>

      {moments.length === 0 ? (
        <div className="text-xs text-neutral-500 py-4">Loading moments…</div>
      ) : (
        <div
          className="mt-2"
          style={{
            display: 'grid',
            gridTemplateColumns: detailColumns,
            gap: '16px',
            alignItems: 'start',
            justifyContent: isSparseSingleMoment ? 'center' : 'start',
          }}
        >
          {moments.map((moment) => (
            <div key={moment.id} className="min-w-0 w-full rounded-xl border border-neutral-800/70 bg-neutral-950/35 p-3">
              <div className="flex items-baseline gap-2 mb-1.5">
                <span className="text-xs font-medium text-neutral-300">
                  {moment.title || formatMomentTime(moment.start_time)}
                </span>
                <span className="text-[10px] text-neutral-600">
                  {moment.image_count} photo{moment.image_count !== 1 ? 's' : ''}
                </span>
              </div>
              {momentImages[moment.id] ? (
                <MomentCollage
                  images={momentImages[moment.id]}
                  thumbs={heroThumbs}
                  onImageClick={onImageClick}
                />
              ) : moment.hero_image_id && heroThumbs[moment.hero_image_id] ? (
                <div className="rounded-lg overflow-hidden">
                  <img src={heroThumbs[moment.hero_image_id]} alt="" className="w-full aspect-[3/2] object-cover object-top" loading="lazy" decoding="async" />
                </div>
              ) : (
                <div className="w-full h-32 rounded-lg bg-neutral-800 flex items-center justify-center">
                  <span className="text-neutral-600 text-sm">Loading…</span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════
 *  VIEW MODE 1 — Quilted / Patchwork Grid (Google Photos style)
 * ═══════════════════════════════════════════════════════════════════ */

function QuiltedGrid({
  yearGroups,
  years,
  expandedChapter,
  moments,
  momentImages,
  heroThumbs,
  toggleChapter,
  colWidth,
  onImageClick,
  coverAspects,
  maxImageSize,
}: {
  yearGroups: Record<string, StoryChapter[]>
  years: string[]
  expandedChapter: string | null
  moments: StoryMoment[]
  momentImages: Record<string, MomentImage[]>
  heroThumbs: Record<number, string>
  toggleChapter: (id: string) => void
  colWidth: number
  onImageClick?: (img: MomentImage, siblings: MomentImage[]) => void
  coverAspects: Record<number, number>
  maxImageSize: number
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(900)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      setContainerWidth(entries[0]?.contentRect.width ?? 900)
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const colCount = Math.max(2, Math.floor(containerWidth / colWidth))
  const GAP = 8

  function renderCard(chapter: StoryChapter) {
    const coverThumb = chapter.cover_image_id ? heroThumbs[chapter.cover_image_id] : undefined
    const dateRange = formatChapterDateRange(chapter.start_date, chapter.end_date)
    const DEFAULT_AR = 4 / 3
    const rawAR = chapter.cover_image_id != null
      ? (coverAspects[chapter.cover_image_id] ?? DEFAULT_AR)
      : DEFAULT_AR
    // Clamp to avoid wide strips (max 2:1) or overly tall cards (min 1:2)
    const imgAR = Math.max(0.5, Math.min(2.0, rawAR))

    return (
      <div
        key={chapter.id}
        style={{ breakInside: 'avoid', marginBottom: `${GAP}px`, maxWidth: `${maxImageSize}px` }}
      >
        <div
          className="group rounded-xl overflow-hidden cursor-pointer bg-neutral-900/70 hover:bg-neutral-800/60 hover:ring-1 hover:ring-neutral-700/50 transition-all duration-200"
          onClick={() => toggleChapter(chapter.id)}
        >
          <div
            className="relative w-full overflow-hidden"
            style={{ aspectRatio: `${imgAR}`, maxHeight: `${maxImageSize}px` }}
          >
            {coverThumb ? (
              <>
                <img
                  src={coverThumb}
                  alt=""
                  className="absolute inset-0 w-full h-full object-cover object-top transition-transform duration-300 group-hover:scale-105"
                  loading="lazy"
                  decoding="async"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/10 to-transparent" />
              </>
            ) : (
              <div className="absolute inset-0 bg-neutral-800/60" />
            )}
            <div className="absolute bottom-0 left-0 right-0 p-3">
              <h3 className="text-sm font-semibold text-white drop-shadow-lg leading-tight truncate">
                {chapter.title || 'Untitled Chapter'}
              </h3>
            </div>
          </div>
          <div className="px-3 py-2">
            <div className="flex items-center gap-1.5 text-[11px] text-neutral-400">
              {chapter.location && <span className="text-blue-400">📍 {chapter.location}</span>}
              {dateRange && <span>{dateRange}</span>}
              <span className="opacity-60">{chapter.image_count} photos</span>
            </div>
            {chapter.summary && (
              <p className="text-xs text-neutral-500 italic line-clamp-2 leading-relaxed mt-1">
                {chapter.summary}
              </p>
            )}
          </div>
        </div>
      </div>
    )
  }

  function renderMasonry(chapters: StoryChapter[]) {
    return (
      <div
        style={{
          columnCount: Math.min(colCount, chapters.length),
          columnGap: `${GAP}px`,
        }}
      >
        {chapters.map((ch) => renderCard(ch))}
      </div>
    )
  }

  return (
    <div ref={containerRef} className="py-6 px-5">
      {years.map((year) => {
        const yearChapters = yearGroups[year]
        const expandedIdx = yearChapters.findIndex((ch) => ch.id === expandedChapter)
        const before = expandedIdx >= 0 ? yearChapters.slice(0, expandedIdx) : yearChapters
        const expandedCh = expandedIdx >= 0 ? yearChapters[expandedIdx] : null
        const after = expandedIdx >= 0 ? yearChapters.slice(expandedIdx + 1) : []

        return (
          <div key={year} className="mb-8">
            {/* Year divider */}
            <div className="flex items-center gap-4 mb-5 px-1">
              <span className="text-2xl font-black text-white tracking-tight">{year}</span>
              <div className="flex-1 h-px bg-gradient-to-r from-neutral-700 to-transparent" />
              <span className="text-xs text-neutral-600 tabular-nums">
                {yearChapters.length} chapter{yearChapters.length !== 1 ? 's' : ''}
              </span>
            </div>

            {before.length > 0 && renderMasonry(before)}

            {expandedCh && (() => {
              const dateRange = formatChapterDateRange(expandedCh.start_date, expandedCh.end_date)
              return (
                <div className="mb-2">
                  <div className="bg-neutral-850 ring-1 ring-blue-500/20 rounded-xl overflow-hidden transition-all duration-200">
                    <ExpandedChapterDetail
                      chapter={expandedCh}
                      dateRange={dateRange}
                      moments={moments}
                      momentImages={momentImages}
                      heroThumbs={heroThumbs}
                      onImageClick={onImageClick}
                      onCollapse={() => toggleChapter(expandedCh.id)}
                      colWidth={colWidth}
                    />
                  </div>
                </div>
              )
            })()}

            {after.length > 0 && renderMasonry(after)}
          </div>
        )
      })}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════
 *  VIEW MODE 2 — Alternating Zigzag Timeline (classic story timeline)
 * ═══════════════════════════════════════════════════════════════════ */

function ZigzagTimeline({
  yearGroups,
  years,
  expandedChapter,
  moments,
  momentImages,
  heroThumbs,
  toggleChapter,
  onImageClick,
  maxImageSize,
}: {
  yearGroups: Record<string, StoryChapter[]>
  years: string[]
  expandedChapter: string | null
  moments: StoryMoment[]
  momentImages: Record<string, MomentImage[]>
  heroThumbs: Record<number, string>
  toggleChapter: (id: string) => void
  onImageClick?: (img: MomentImage, siblings: MomentImage[]) => void
  coverAspects?: Record<number, number>
  maxImageSize: number
}) {
  let globalIdx = 0

  return (
    <div className="py-8 px-6">
      {years.map((year) => {
        const yearChapters = yearGroups[year]
        return (
          <div key={year} className="mb-12">
            {/* Year badge — centered on the timeline spine */}
            <div className="flex justify-center mb-8">
              <span className="px-5 py-1.5 rounded-full bg-blue-600 text-white text-sm font-bold tracking-wide shadow-lg shadow-blue-600/20">
                {year}
              </span>
            </div>

            {/* Chapters along the spine */}
            <div className="relative">
              {/* Central timeline line */}
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gradient-to-b from-blue-500/40 via-neutral-700/60 to-transparent -translate-x-px" />

              {yearChapters.map((chapter) => {
                const isExpanded = expandedChapter === chapter.id
                const coverThumb = chapter.cover_image_id ? heroThumbs[chapter.cover_image_id] : undefined
                const dateRange = formatChapterDateRange(chapter.start_date, chapter.end_date)
                const side = globalIdx % 2 === 0 ? 'left' : 'right'
                globalIdx++

                if (isExpanded) {
                  return (
                    <div key={chapter.id} className="relative mb-8">
                      {/* Dot on spine */}
                      <div className="absolute left-1/2 top-6 w-3 h-3 rounded-full bg-blue-500 ring-4 ring-neutral-900 -translate-x-1.5 z-10" />
                      {/* Full-width expanded card */}
                      <div className="mx-auto w-full max-w-6xl 2xl:max-w-[96rem]">
                        <div
                          className="bg-neutral-850 ring-1 ring-blue-500/30 rounded-xl overflow-hidden transition-all duration-200"
                        >
                          <ExpandedChapterDetail
                            chapter={chapter}
                            dateRange={dateRange}
                            moments={moments}
                            momentImages={momentImages}
                            heroThumbs={heroThumbs}
                            onImageClick={onImageClick}
                            onCollapse={() => toggleChapter(chapter.id)}
                          />
                        </div>
                      </div>
                    </div>
                  )
                }

                return (
                  <div key={chapter.id} className="relative mb-8">
                    {/* Dot on spine */}
                    <div className="absolute left-1/2 top-6 w-3 h-3 rounded-full bg-neutral-600 ring-4 ring-neutral-900 -translate-x-1.5 z-10 group-hover:bg-blue-400 transition-colors" />

                    {/* Card — alternates left / right of the spine */}
                    <div className={`grid grid-cols-2 gap-8 ${side === 'right' ? '' : ''}`}>
                      {/* Left side */}
                      <div className={side === 'left' ? '' : 'flex items-center justify-end'}>
                        {side === 'left' ? (
                          <div
                            className="group rounded-xl overflow-hidden cursor-pointer bg-neutral-900/70 hover:bg-neutral-800/60 ring-1 ring-neutral-800/50 hover:ring-neutral-700/50 transition-all duration-200 ml-auto w-full"
                            style={{ maxWidth: `${maxImageSize}px` }}
                            onClick={() => toggleChapter(chapter.id)}
                          >
                            {coverThumb ? (
                              <div className="relative aspect-[3/2] overflow-hidden">
                                <img
                                  src={coverThumb}
                                  alt=""
                                  className="w-full h-full object-cover object-top transition-transform duration-300 group-hover:scale-105"
                                  loading="lazy"
                                  decoding="async"
                                />
                                <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent" />
                                <div className="absolute bottom-0 left-0 right-0 p-3">
                                  <h3 className="text-base font-semibold text-white drop-shadow-lg leading-tight">
                                    {chapter.title || 'Untitled Chapter'}
                                  </h3>
                                </div>
                              </div>
                            ) : (
                              <div className="h-32 bg-neutral-800/50 flex items-end p-3">
                                <h3 className="text-base font-semibold text-neutral-200 leading-tight">
                                  {chapter.title || 'Untitled Chapter'}
                                </h3>
                              </div>
                            )}
                            <div className="px-3 py-2.5">
                              <div className="flex items-center gap-1.5 text-[11px] text-neutral-400">
                                {chapter.location && <span className="text-blue-400">📍 {chapter.location}</span>}
                                {dateRange && <span>{dateRange}</span>}
                                <span className="opacity-60">{chapter.image_count} photos</span>
                              </div>
                              {chapter.summary && (
                                <p className="text-xs text-neutral-500 italic line-clamp-2 leading-relaxed mt-1">
                                  {chapter.summary}
                                </p>
                              )}
                            </div>
                          </div>
                        ) : (
                          /* Date label on the opposite side */
                          <div className="text-right pr-4 pt-6">
                            <div className="text-sm font-medium text-neutral-400">{dateRange}</div>
                            <div className="text-[11px] text-neutral-600 mt-0.5">
                              {chapter.image_count} photos
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Right side */}
                      <div className={side === 'right' ? '' : 'flex items-center'}>
                        {side === 'right' ? (
                          <div
                            className="group rounded-xl overflow-hidden cursor-pointer bg-neutral-900/70 hover:bg-neutral-800/60 ring-1 ring-neutral-800/50 hover:ring-neutral-700/50 transition-all duration-200 mr-auto w-full"
                            style={{ maxWidth: `${maxImageSize}px` }}
                            onClick={() => toggleChapter(chapter.id)}
                          >
                            {coverThumb ? (
                              <div className="relative aspect-[3/2] overflow-hidden">
                                <img
                                  src={coverThumb}
                                  alt=""
                                  className="w-full h-full object-cover object-top transition-transform duration-300 group-hover:scale-105"
                                  loading="lazy"
                                  decoding="async"
                                />
                                <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent" />
                                <div className="absolute bottom-0 left-0 right-0 p-3">
                                  <h3 className="text-base font-semibold text-white drop-shadow-lg leading-tight">
                                    {chapter.title || 'Untitled Chapter'}
                                  </h3>
                                </div>
                              </div>
                            ) : (
                              <div className="h-32 bg-neutral-800/50 flex items-end p-3">
                                <h3 className="text-base font-semibold text-neutral-200 leading-tight">
                                  {chapter.title || 'Untitled Chapter'}
                                </h3>
                              </div>
                            )}
                            <div className="px-3 py-2.5">
                              <div className="flex items-center gap-1.5 text-[11px] text-neutral-400">
                                {chapter.location && <span className="text-blue-400">📍 {chapter.location}</span>}
                                {dateRange && <span>{dateRange}</span>}
                                <span className="opacity-60">{chapter.image_count} photos</span>
                              </div>
                              {chapter.summary && (
                                <p className="text-xs text-neutral-500 italic line-clamp-2 leading-relaxed mt-1">
                                  {chapter.summary}
                                </p>
                              )}
                            </div>
                          </div>
                        ) : (
                          /* Date label on the opposite side */
                          <div className="pl-4 pt-6">
                            <div className="text-sm font-medium text-neutral-400">{dateRange}</div>
                            <div className="text-[11px] text-neutral-600 mt-0.5">
                              {chapter.image_count} photos
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )
      })}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════
 *  StoryTimeline — top-level wrapper with view-mode switch
 * ═══════════════════════════════════════════════════════════════════ */

function StoryTimeline({
  album,
  chapters,
  onGenerateStory,
  onGenerateNarrative,
  onExportStory,
  onRefresh,
  generating,
  narrating,
  exporting,
  refreshing,
  onImageClick,
}: {
  album: SmartAlbumSummary
  chapters: StoryChapter[]
  onGenerateStory: () => void
  onGenerateNarrative: () => void
  onExportStory: () => void
  onRefresh: () => void
  generating: boolean
  narrating: boolean
  exporting: boolean
  refreshing: boolean
  onImageClick?: (img: MomentImage, siblings: MomentImage[]) => void
}) {
  const [viewMode, setViewMode] = useState<ViewMode>('quilted')
  const [colWidth, setColWidth] = useState(240)  // px per column — slider controls this
  // Map slider range [160..450] → max image size [256..768]
  const maxImageSize = Math.round(256 + (colWidth - 160) / (450 - 160) * (768 - 256))
  const [expandedChapter, setExpandedChapter] = useState<string | null>(null)
  const [moments, setMoments] = useState<StoryMoment[]>([])
  const [momentImages, setMomentImages] = useState<Record<string, MomentImage[]>>({})
  const [heroThumbs, setHeroThumbs] = useState<Record<number, string>>({})
  const requestedThumbsRef = useRef<Set<number>>(new Set())
  const loadedMomentIdsRef = useRef<Set<string>>(new Set())
  const expandedChapterRef = useRef<string | null>(null)
  const [coverAspects, setCoverAspects] = useState<Record<number, number>>({})
  const measuredAspectsRef = useRef<Set<number>>(new Set())

  // Preserve only cover thumbnails, clear moment-specific state
  const coverIdsRef = useRef<Set<number>>(new Set())
  useEffect(() => {
    const ids = new Set(
      chapters.map((ch) => ch.cover_image_id).filter((id): id is number => id != null)
    )
    coverIdsRef.current = ids
  }, [chapters])

  useEffect(() => {
    requestedThumbsRef.current.clear()
    loadedMomentIdsRef.current.clear()
    measuredAspectsRef.current.clear()
    setMomentImages({})
    setHeroThumbs({})
    setCoverAspects({})
  }, [album.id])

  const loadThumbnailIds = useCallback(async (imageIds: number[]) => {
    const unique = [...new Set(imageIds)].filter((id) => !requestedThumbsRef.current.has(id))
    if (unique.length === 0) return
    unique.forEach((id) => requestedThumbsRef.current.add(id))

    const BATCH_SIZE = 50
    const chunks: number[][] = []
    for (let i = 0; i < unique.length; i += BATCH_SIZE) {
      chunks.push(unique.slice(i, i + BATCH_SIZE))
    }

    await Promise.all(chunks.map(async (chunk) => {
      try {
        const thumbs = await window.api.getThumbnailsBatch(chunk.map((imageId) => ({ image_id: imageId })))
        const mapped = mergeThumbnailMap(thumbs)
        setHeroThumbs((prev) => ({ ...prev, ...mapped }))
      } catch {
        chunk.forEach((id) => requestedThumbsRef.current.delete(id))
      }
    }))
  }, [])

  useEffect(() => {
    const coverIds = chapters
      .map((ch) => ch.cover_image_id)
      .filter((id): id is number => id != null)
    void loadThumbnailIds(coverIds)
  }, [chapters, loadThumbnailIds])

  // Measure cover image aspect ratios from loaded thumbnails
  useEffect(() => {
    for (const [idStr, dataUrl] of Object.entries(heroThumbs)) {
      const id = Number(idStr)
      if (!coverIdsRef.current.has(id) || measuredAspectsRef.current.has(id)) continue
      measuredAspectsRef.current.add(id)
      const img = new Image()
      img.onload = () => {
        if (img.naturalWidth > 0 && img.naturalHeight > 0) {
          setCoverAspects(prev => ({ ...prev, [id]: img.naturalWidth / img.naturalHeight }))
        }
      }
      img.src = dataUrl
    }
  }, [heroThumbs])

  const clearMomentState = useCallback(() => {
    setMoments([])
    setMomentImages({})
    loadedMomentIdsRef.current.clear()
    // Evict non-cover thumbnails to bound memory
    const covers = coverIdsRef.current
    setHeroThumbs((prev) => {
      const next: Record<number, string> = {}
      for (const [k, v] of Object.entries(prev)) {
        const id = Number(k)
        if (covers.has(id)) next[id] = v
      }
      return next
    })
    requestedThumbsRef.current = new Set(covers)
  }, [])

  const loadMoments = useCallback(async (chapterId: string) => {
    try {
      const { moments: loaded } = await window.api.albumsChapterMoments(chapterId)
      if (expandedChapterRef.current !== chapterId) return
      const nextMoments = loaded as StoryMoment[]
      setMoments(nextMoments)

      // Collect all image IDs from all moments in parallel, then batch-load thumbs once
      const allImageIds: number[] = nextMoments
        .map((m) => m.hero_image_id)
        .filter((id): id is number => id != null)

      await Promise.all(nextMoments.map(async (m) => {
        if (loadedMomentIdsRef.current.has(m.id)) return
        loadedMomentIdsRef.current.add(m.id)
        try {
          const { images } = await window.api.albumsMomentImages(m.id)
          if (expandedChapterRef.current !== chapterId) return
          const imgs = images as MomentImage[]
          setMomentImages((prev) => ({ ...prev, [m.id]: imgs }))
          allImageIds.push(...imgs.map((img) => img.image_id))
        } catch {
          loadedMomentIdsRef.current.delete(m.id)
        }
      }))

      if (allImageIds.length > 0) {
        void loadThumbnailIds(allImageIds)
      }
    } catch { /* chapter may have been collapsed */ }
  }, [loadThumbnailIds])

  const toggleChapter = useCallback((chapterId: string) => {
    if (expandedChapterRef.current === chapterId) {
      expandedChapterRef.current = null
      setExpandedChapter(null)
      clearMomentState()
    } else {
      expandedChapterRef.current = chapterId
      setExpandedChapter(chapterId)
      clearMomentState()
      void loadMoments(chapterId)
    }
  }, [loadMoments, clearMomentState])

  const yearGroups = useMemo(() => groupChaptersByYear(chapters), [chapters])
  const years = useMemo(() => Object.keys(yearGroups).sort(), [yearGroups])

  const sharedProps = {
    yearGroups,
    years,
    expandedChapter,
    moments,
    momentImages,
    heroThumbs,
    toggleChapter,
    onImageClick,
    coverAspects,
    maxImageSize,
  }

  return (
    <div className="flex flex-col h-full">
      {/* ── Header bar ── */}
      <div className="shrink-0 border-b border-neutral-800 bg-neutral-900 px-5 py-4 xl:px-6 2xl:px-8">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">{album.name}</h2>
            <p className="text-xs text-neutral-500 mt-0.5">
              {album.item_count} photos · {chapters.length} chapters
              {years.length > 1 && ` · ${years[0]}–${years[years.length - 1]}`}
            </p>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            {/* View mode toggle */}
            <div className="flex items-center bg-neutral-800 rounded-full p-0.5 mr-1">
              <button
                onClick={() => setViewMode('quilted')}
                className={`px-2.5 py-1 text-xs font-medium rounded-full transition-colors ${
                  viewMode === 'quilted'
                    ? 'bg-neutral-600 text-white'
                    : 'text-neutral-400 hover:text-neutral-200'
                }`}
                title="Quilted grid view"
              >
                ▦ Grid
              </button>
              <button
                onClick={() => setViewMode('zigzag')}
                className={`px-2.5 py-1 text-xs font-medium rounded-full transition-colors ${
                  viewMode === 'zigzag'
                    ? 'bg-neutral-600 text-white'
                    : 'text-neutral-400 hover:text-neutral-200'
                }`}
                title="Zigzag timeline view"
              >
                ⟡ Timeline
              </button>
            </div>

            {/* Column / image size slider */}
            <div className="flex items-center gap-1.5 mr-1" title="Card size">
              <span className="text-[10px] text-neutral-500">▪</span>
              <input
                type="range"
                min={160}
                max={450}
                step={10}
                value={colWidth}
                onChange={(e) => setColWidth(Number(e.target.value))}
                className="w-20 h-1 accent-neutral-500 cursor-pointer"
              />
              <span className="text-[10px] text-neutral-500">▮</span>
            </div>

            <button
              onClick={onRefresh}
              disabled={refreshing || generating}
              className="px-3 py-1.5 text-xs font-medium rounded-full bg-neutral-800 text-neutral-300 hover:bg-neutral-700 disabled:opacity-40 transition-colors"
              title="Refresh album membership and story with newly processed images"
            >
              {refreshing ? (
                <span className="flex items-center gap-1.5">
                  <span className="inline-block w-3 h-3 border border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
                  Refreshing…
                </span>
              ) : '↻ Refresh'}
            </button>
            <button
              onClick={onGenerateNarrative}
              disabled={narrating || chapters.length === 0}
              className="px-3 py-1.5 text-xs font-medium rounded-full bg-neutral-800 text-neutral-300 hover:bg-neutral-700 disabled:opacity-40 transition-colors"
            >
              {narrating ? 'Summarizing…' : '✨ Summaries'}
            </button>
            <button
              onClick={onExportStory}
              disabled={exporting || chapters.length === 0}
              className="px-3 py-1.5 text-xs font-medium rounded-full bg-neutral-800 text-neutral-300 hover:bg-neutral-700 disabled:opacity-40 transition-colors"
            >
              {exporting ? 'Exporting…' : '↗ Export'}
            </button>
            <button
              onClick={onGenerateStory}
              disabled={generating || refreshing}
              className="px-3 py-1.5 text-xs font-medium rounded-full bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40 transition-colors"
            >
              {generating ? 'Generating…' : chapters.length > 0 ? '↻ Regenerate' : '▶ Generate Story'}
            </button>
          </div>
        </div>
      </div>

      {/* ── Timeline body ── */}
      <div className="flex-1 overflow-y-auto">
        {chapters.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-neutral-500 gap-3 px-8">
            <div className="text-4xl opacity-30">📖</div>
            <p className="text-sm">No story generated yet.</p>
            <p className="text-xs text-neutral-600 text-center max-w-xs">
              Generate a story to organise your photos into an automatic timeline
              with chapters, moments, and hero images.
            </p>
          </div>
        ) : viewMode === 'quilted' ? (
          <QuiltedGrid {...sharedProps} colWidth={colWidth} />
        ) : (
          <ZigzagTimeline {...sharedProps} />
        )}
      </div>
    </div>
  )
}

function formatMomentTime(iso: string | null): string {
  if (!iso) return 'Unknown time'
  try {
    // Append Z if no timezone indicator to ensure consistent UTC interpretation
    const normalized = /[Zz+\-]/.test(iso.slice(10)) ? iso : iso + 'Z'
    const d = new Date(normalized)
    if (isNaN(d.getTime())) return iso
    return d.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return iso
  }
}

// ── Main AlbumsView ─────────────────────────────────────────────────────────

export function AlbumsView() {
  const [albums, setAlbums] = useState<SmartAlbumSummary[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [chapters, setChapters] = useState<StoryChapter[]>([])
  const [creating, setCreating] = useState(false)
  const [creatingPreset, setCreatingPreset] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [narrating, setNarrating] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [persons, setPersons] = useState<FacePerson[]>([])
  const [evalReport, setEvalReport] = useState<StoryGenerateResult['evaluation'] | null>(null)
  const [status, setStatus] = useState<StatusBanner | null>(null)
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const showStatus = useCallback((banner: StatusBanner) => {
    setStatus(banner)
    if (statusTimerRef.current) clearTimeout(statusTimerRef.current)
    statusTimerRef.current = setTimeout(() => setStatus(null), banner.tone === 'error' ? 8000 : 5000)
  }, [])
  const [presetDefinitions, setPresetDefinitions] = useState<Record<string, PresetDefinition>>(FALLBACK_PRESETS)
  const [editingAlbum, setEditingAlbum] = useState<SmartAlbumSummary | null>(null)
  const [deletingAlbum, setDeletingAlbum] = useState<SmartAlbumSummary | null>(null)

  // Lightbox state
  const [lightboxItem, setLightboxItem] = useState<SearchResult | null>(null)
  const [lightboxItems, setLightboxItems] = useState<SearchResult[]>([])

  const handleImageClick = useCallback(async (img: MomentImage, siblings: MomentImage[]) => {
    try {
      const { result } = await window.api.getImageDetails({ image_id: img.image_id })
      if (!result) return
      setLightboxItem(result)

      // Fetch sibling details in parallel for lightbox navigation
      const details = await Promise.all(
        siblings.map(async (s) => {
          try {
            const { result: r } = await window.api.getImageDetails({ image_id: s.image_id })
            return r
          } catch { return null }
        })
      )
      const validItems = details.filter((r): r is SearchResult => r != null)
      setLightboxItems(validItems.length > 0 ? validItems : [result])
    } catch { /* ignore */ }
  }, [])

  const handleCloseLightbox = useCallback(() => {
    setLightboxItem(null)
    setLightboxItems([])
  }, [])

  const handleNavigateLightbox = useCallback((item: SearchResult) => {
    setLightboxItem(item)
  }, [])

  const loadAlbums = useCallback(async () => {
    try {
      const { albums: list } = await window.api.albumsList()
      setAlbums(list)
    } catch (err) {
      console.error('Failed to load albums:', err)
    }
  }, [])

  const loadPersons = useCallback(async () => {
    try {
      const { persons: list } = await window.api.listPersons()
      setPersons(list)
    } catch {
      // Persons may not exist yet
    }
  }, [])

  const loadPresets = useCallback(async () => {
    try {
      const { presets } = await window.api.albumsPresets()
      setPresetDefinitions(Object.keys(presets).length > 0 ? presets : FALLBACK_PRESETS)
    } catch {
      setPresetDefinitions(FALLBACK_PRESETS)
    }
  }, [])

  useEffect(() => {
    void loadAlbums()
    void loadPersons()
    void loadPresets()
  }, [loadAlbums, loadPersons, loadPresets])

  const loadStory = useCallback(async (albumId: string) => {
    try {
      const { chapters: loadedChapters } = await window.api.albumsStory(albumId)
      setChapters(loadedChapters)
    } catch (err) {
      console.error('Failed to load story:', err)
      setChapters([])
    }
  }, [])

  const handleSelect = useCallback((albumId: string) => {
    setSelectedId(albumId)
    setEvalReport(null)
    setStatus(null)
    void loadStory(albumId)
  }, [loadStory])

  const handleRefresh = useCallback(async () => {
    if (!selectedId) return
    const currentAlbum = albums.find((album) => album.id === selectedId)
    const previousCount = currentAlbum?.item_count ?? 0
    setRefreshing(true)
    setStatus(null)
    try {
      const { item_count } = await window.api.albumsRefresh(selectedId)
      const generation = await window.api.albumsStoryGenerate({ album_id: selectedId }) as StoryGenerateResult
      setEvalReport(generation.evaluation)
      await loadAlbums()
      await loadStory(selectedId)
      const added = item_count - previousCount
      showStatus({
        tone: 'success',
        text:
          added > 0
            ? `Album refreshed. Added ${added} new photo${added === 1 ? '' : 's'}.`
            : item_count === 0
              ? 'Album refreshed. No matching photos right now.'
              : 'Album refreshed. No new matching photos.',
      })
    } catch (err) {
      console.error('Failed to refresh album:', err)
      showStatus({ tone: 'error', text: 'Failed to refresh album data.' })
    } finally {
      setRefreshing(false)
    }
  }, [selectedId, albums, loadAlbums, loadStory, showStatus])

  const handleGenerate = useCallback(async () => {
    if (!selectedId) return
    setGenerating(true)
    setStatus(null)
    try {
      const result = await window.api.albumsStoryGenerate({ album_id: selectedId })
      const generation = result as unknown as StoryGenerateResult
      setEvalReport(generation.evaluation)
      await loadStory(selectedId)
      await loadAlbums()
      showStatus({
        tone: generation.evaluation.overall_pass ? 'success' : 'error',
        text: `Story generated in ${generation.generation_time_s}s.`,
      })
    } catch (err) {
      console.error('Failed to generate story:', err)
      showStatus({ tone: 'error', text: 'Failed to generate story.' })
    } finally {
      setGenerating(false)
    }
  }, [selectedId, loadStory, loadAlbums])

  const handleGenerateNarrative = useCallback(async () => {
    if (!selectedId) return
    setNarrating(true)
    setStatus(null)
    try {
      const result = await window.api.albumsGenerateNarrative({ album_id: selectedId, use_ai: true })
      await loadStory(selectedId)
      showStatus({
        tone: 'success',
        text: `Updated ${result.chapters_updated} chapter summaries.`,
      })
    } catch (err) {
      console.error('Failed to generate narratives:', err)
      showStatus({ tone: 'error', text: 'Failed to generate chapter summaries.' })
    } finally {
      setNarrating(false)
    }
  }, [selectedId, loadStory])

  const handleExportStory = useCallback(async () => {
    if (!selectedId) return
    const album = albums.find((item) => item.id === selectedId)
    const suggested = `${sanitizeFileName(album?.name ?? 'story-export')}.html`
    const outputPath = await window.api.saveStoryExport(suggested)
    if (!outputPath) return

    setExporting(true)
    setStatus(null)
    try {
      const result = await window.api.albumsExport({
        album_id: selectedId,
        output_path: outputPath,
        include_thumbnails: true,
      })
      showStatus({ tone: 'success', text: `Exported story to ${result.path}` })
      await window.api.openPath(result.path)
    } catch (err) {
      console.error('Failed to export story:', err)
      showStatus({ tone: 'error', text: 'Failed to export story.' })
    } finally {
      setExporting(false)
    }
  }, [albums, selectedId])

  const handleCreated = useCallback(async (albumId: string) => {
    await loadAlbums()
    handleSelect(albumId)
  }, [loadAlbums, handleSelect])

  const handleDeleteAlbum = useCallback(async () => {
    const target = deletingAlbum
    if (!target) return
    setDeletingAlbum(null)
    try {
      await window.api.albumsDelete(target.id)
      if (selectedId === target.id) {
        setSelectedId(null)
        setChapters([])
        setEvalReport(null)
      }
      showStatus({ tone: 'success', text: `"${target.name}" deleted.` })
      await loadAlbums()
    } catch (err) {
      console.error('Failed to delete album:', err)
      showStatus({ tone: 'error', text: 'Failed to delete album.' })
    }
  }, [deletingAlbum, selectedId, loadAlbums])

  const handleEditSaved = useCallback(async () => {
    await loadAlbums()
    showStatus({ tone: 'success', text: 'Album updated.' })
  }, [loadAlbums])

  const selectedAlbum = albums.find((album) => album.id === selectedId)

  return (
    <div className="flex h-full min-w-0">
      <div className="w-56 shrink-0 xl:w-60 2xl:w-64">
        <AlbumListPanel
          albums={albums}
          selectedId={selectedId}
          onSelect={handleSelect}
          onRefresh={() => void loadAlbums()}
          onCreate={() => setCreating(true)}
          onCreatePreset={() => setCreatingPreset(true)}
          onEdit={(a) => setEditingAlbum(a)}
          onDelete={(a) => setDeletingAlbum(a)}
        />
      </div>

      <div className="flex-1 min-w-0 flex flex-col">
        {selectedAlbum ? (
          <>
            <StoryTimeline
              key={selectedAlbum.id}
              album={selectedAlbum}
              chapters={chapters}
              onGenerateStory={() => void handleGenerate()}
              onGenerateNarrative={() => void handleGenerateNarrative()}
              onExportStory={() => void handleExportStory()}
              onRefresh={() => void handleRefresh()}
              generating={generating}
              narrating={narrating}
              exporting={exporting}
              refreshing={refreshing}
              onImageClick={handleImageClick}
            />
            {evalReport && (
              <div className={`px-4 py-2 border-t text-xs ${
                evalReport.overall_pass
                  ? 'border-emerald-800 bg-emerald-900/30 text-emerald-400'
                  : 'border-amber-800 bg-amber-900/30 text-amber-400'
              }`}>
                <span className="font-medium">
                  Evaluation: {evalReport.overall_pass ? 'PASS' : 'FAIL'}
                </span>
                {' — '}
                {Object.values(evalReport.criteria).filter((criterion) => criterion.passed).length}/
                {Object.keys(evalReport.criteria).length} criteria passed
              </div>
            )}
            {status && (
              <div className={`px-4 py-2 border-t text-xs ${
                status.tone === 'success'
                  ? 'border-blue-800 bg-blue-900/20 text-blue-300'
                  : 'border-red-900 bg-red-950/40 text-red-300'
              }`}>
                {status.text}
              </div>
            )}
            <div className="px-4 py-2 border-t border-neutral-800 flex justify-end gap-2">
              <button
                onClick={() => setEditingAlbum(selectedAlbum)}
                className="px-2 py-1 text-xs rounded bg-neutral-700 text-neutral-300 hover:bg-neutral-600"
              >
                ✏ Edit
              </button>
              <button
                onClick={() => setDeletingAlbum(selectedAlbum)}
                className="px-2 py-1 text-xs rounded bg-red-900/50 text-red-400 hover:bg-red-800/50"
              >
                🗑 Delete
              </button>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-neutral-500">
            <div className="text-center">
              <p className="text-sm">Select an album or create a new one</p>
              <p className="text-xs mt-1 text-neutral-600">
                Smart albums auto-collect images by rules or presets.
              </p>
            </div>
          </div>
        )}
      </div>

      <CreateAlbumDialog
        open={creating}
        onClose={() => setCreating(false)}
        onCreated={handleCreated}
        persons={persons.map((person) => ({ id: person.id, name: person.name }))}
      />

      <PresetAlbumDialog
        open={creatingPreset}
        onClose={() => setCreatingPreset(false)}
        onCreated={handleCreated}
        persons={persons}
        presets={presetDefinitions}
      />

      {editingAlbum && (
        <EditAlbumDialog
          album={editingAlbum}
          onClose={() => setEditingAlbum(null)}
          onSaved={handleEditSaved}
          persons={persons.map((p) => ({ id: p.id, name: p.name }))}
        />
      )}

      {deletingAlbum && (
        <DeleteConfirmDialog
          albumName={deletingAlbum.name}
          onConfirm={() => void handleDeleteAlbum()}
          onCancel={() => setDeletingAlbum(null)}
        />
      )}

      {/* Full lightbox opened from image click */}
      {lightboxItem && (
        <SearchLightbox
          item={lightboxItem}
          items={lightboxItems}
          onClose={handleCloseLightbox}
          onNavigate={handleNavigateLightbox}
        />
      )}
    </div>
  )
}
