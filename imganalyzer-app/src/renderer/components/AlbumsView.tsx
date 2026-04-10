/**
 * AlbumsView — Smart albums list + story chapter timeline.
 *
 * Layout:
 *   Left panel  — album list + create/preset dialogs
 *   Right panel — story timeline (chapters → moments → images)
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import type {
  FacePerson,
  SmartAlbumSummary,
  StoryChapter,
  StoryMoment,
  MomentImage,
  StoryGenerateResult,
} from '../global'

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
  const cleaned = value.replace(/[<>:"/\\|?*\u0000-\u001F]/g, ' ').trim()
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

// ── Album List Panel ────────────────────────────────────────────────────────

function AlbumListPanel({
  albums,
  selectedId,
  onSelect,
  onRefresh,
  onCreate,
  onCreatePreset,
}: {
  albums: SmartAlbumSummary[]
  selectedId: string | null
  onSelect: (id: string) => void
  onRefresh: () => void
  onCreate: () => void
  onCreatePreset: () => void
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
          <button
            key={a.id}
            onClick={() => onSelect(a.id)}
            className={`w-full text-left px-3 py-2.5 border-b border-neutral-800 transition-colors ${
              selectedId === a.id
                ? 'bg-neutral-700/60 text-white'
                : 'text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200'
            }`}
          >
            <div className="text-sm font-medium truncate">{a.name}</div>
            <div className="text-xs text-neutral-500 mt-0.5">
              {a.item_count} images · {a.chapter_count} chapters
            </div>
            {a.description && (
              <div className="text-xs text-neutral-500 mt-0.5 truncate">{a.description}</div>
            )}
          </button>
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
  const [ruleType, setRuleType] = useState<'person' | 'date_range' | 'location'>('person')
  const [selectedPersons, setSelectedPersons] = useState<number[]>([])
  const [personMode, setPersonMode] = useState<'any' | 'all'>('any')
  const [dateStart, setDateStart] = useState('')
  const [dateEnd, setDateEnd] = useState('')
  const [country, setCountry] = useState('')
  const [creating, setCreating] = useState(false)

  if (!open) return null

  const handleCreate = async () => {
    if (!name.trim()) return
    setCreating(true)
    try {
      const rules: Array<Record<string, unknown>> = []
      if (ruleType === 'person' && selectedPersons.length > 0) {
        rules.push({
          type: 'person',
          person_ids: selectedPersons,
          mode: personMode,
        })
      }
      if (ruleType === 'date_range' && (dateStart || dateEnd)) {
        rules.push({
          type: 'date_range',
          ...(dateStart && { start: dateStart }),
          ...(dateEnd && { end: dateEnd }),
        })
      }
      if (ruleType === 'location' && country.trim()) {
        rules.push({ type: 'location', country: country.trim() })
      }

      const result = await window.api.albumsCreate({
        name: name.trim(),
        rules: { match: 'all', rules },
        description: description.trim() || undefined,
      })
      onCreated(result.id)
      setName('')
      setDescription('')
      setSelectedPersons([])
      onClose()
    } catch (err) {
      console.error('Failed to create album:', err)
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-neutral-800 rounded-lg shadow-xl w-[460px] max-h-[80vh] overflow-y-auto p-5">
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

        <label className="block text-sm text-neutral-400 mb-1">Rule Type</label>
        <select
          value={ruleType}
          onChange={(e) => setRuleType(e.target.value as typeof ruleType)}
          className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600 mb-3"
        >
          <option value="person">Person</option>
          <option value="date_range">Date Range</option>
          <option value="location">Location (Country)</option>
        </select>

        {ruleType === 'person' && (
          <div className="mb-3">
            <label className="block text-sm text-neutral-400 mb-1">People</label>
            <div className="max-h-32 overflow-y-auto bg-neutral-900 rounded p-2 mb-2">
              {persons.length === 0 && (
                <span className="text-xs text-neutral-500">No persons found. Analyze faces first.</span>
              )}
              {persons.map((p) => (
                <label key={p.id} className="flex items-center gap-2 py-0.5 text-sm text-neutral-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedPersons.includes(p.id)}
                    onChange={(e) => {
                      if (e.target.checked) setSelectedPersons([...selectedPersons, p.id])
                      else setSelectedPersons(selectedPersons.filter((x) => x !== p.id))
                    }}
                  />
                  {p.name}
                </label>
              ))}
            </div>
            <label className="flex items-center gap-2 text-sm text-neutral-400">
              <select
                value={personMode}
                onChange={(e) => setPersonMode(e.target.value as 'any' | 'all')}
                className="px-2 py-0.5 rounded bg-neutral-700 text-white text-xs border border-neutral-600"
              >
                <option value="any">Any of the selected</option>
                <option value="all">All of the selected (co-occurrence)</option>
              </select>
            </label>
          </div>
        )}

        {ruleType === 'date_range' && (
          <div className="flex gap-2 mb-3">
            <div className="flex-1">
              <label className="block text-xs text-neutral-500 mb-0.5">Start</label>
              <input
                type="date"
                value={dateStart}
                onChange={(e) => setDateStart(e.target.value)}
                className="w-full px-2 py-1 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
              />
            </div>
            <div className="flex-1">
              <label className="block text-xs text-neutral-500 mb-0.5">End</label>
              <input
                type="date"
                value={dateEnd}
                onChange={(e) => setDateEnd(e.target.value)}
                className="w-full px-2 py-1 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
              />
            </div>
          </div>
        )}

        {ruleType === 'location' && (
          <div className="mb-3">
            <label className="block text-sm text-neutral-400 mb-1">Country code (e.g. US, FR, JP)</label>
            <input
              value={country}
              onChange={(e) => setCountry(e.target.value)}
              className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
              placeholder="US"
            />
          </div>
        )}

        <div className="flex justify-end gap-2 mt-4">
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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
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
            <select
              value={personId}
              onChange={(e) => setPersonId(Number(e.target.value) || '')}
              className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
            >
              <option value="">Select a person...</option>
              {persons.map((person) => (
                <option key={person.id} value={person.id}>
                  {person.name}
                </option>
              ))}
            </select>
          </div>
        )}

        {preset === 'together' && (
          <div className="grid grid-cols-2 gap-2 mb-3">
            <div>
              <label className="block text-sm text-neutral-400 mb-1">First person</label>
              <select
                value={personId}
                onChange={(e) => setPersonId(Number(e.target.value) || '')}
                className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
              >
                <option value="">Select...</option>
                {persons.map((person) => (
                  <option key={person.id} value={person.id}>
                    {person.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-neutral-400 mb-1">Second person</label>
              <select
                value={secondPersonId}
                onChange={(e) => setSecondPersonId(Number(e.target.value) || '')}
                className="w-full px-3 py-1.5 rounded bg-neutral-700 text-white text-sm border border-neutral-600"
              >
                <option value="">Select...</option>
                {persons.map((person) => (
                  <option key={person.id} value={person.id}>
                    {person.name}
                  </option>
                ))}
              </select>
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

/** Collage layout for moment images — 1 hero large + rest arranged around it. */
/**
 * Collage cell — renders a thumbnail or placeholder at the given grid position.
 * Uses CSS grid placement so each cell can span different rows/columns.
 */
function CollageCell({
  img,
  thumb,
  style,
}: {
  img: MomentImage
  thumb: string | undefined
  style: React.CSSProperties
}) {
  return (
    <div className="overflow-hidden relative" style={style}>
      {thumb ? (
        <img
          src={thumb}
          alt=""
          className="absolute inset-0 w-full h-full object-cover object-top"
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

/** Pick a deterministic layout variant from moment id to avoid re-shuffling on re-render. */
function pickVariant(momentId: number, count: number): number {
  return Math.abs(momentId) % count
}

function getLayout(count: number, momentId: number): CollageLayout {
  if (count === 1) {
    return {
      columns: '1fr',
      rows: '200px',
      slots: [{ gridColumn: '1', gridRow: '1' }],
    }
  }

  if (count === 2) {
    const variant = pickVariant(momentId, 2)
    if (variant === 0) {
      // Wide left, narrow right
      return {
        columns: '3fr 2fr',
        rows: '180px',
        slots: [
          { gridColumn: '1', gridRow: '1' },
          { gridColumn: '2', gridRow: '1' },
        ],
      }
    }
    // Narrow left, wide right
    return {
      columns: '2fr 3fr',
      rows: '180px',
      slots: [
        { gridColumn: '1', gridRow: '1' },
        { gridColumn: '2', gridRow: '1' },
      ],
    }
  }

  if (count === 3) {
    const variant = pickVariant(momentId, 3)
    if (variant === 0) {
      // Big left, 2 stacked right
      return {
        columns: '3fr 2fr',
        rows: '110px 110px',
        slots: [
          { gridColumn: '1', gridRow: '1 / 3' },
          { gridColumn: '2', gridRow: '1' },
          { gridColumn: '2', gridRow: '2' },
        ],
      }
    }
    if (variant === 1) {
      // 2 stacked left, big right
      return {
        columns: '2fr 3fr',
        rows: '110px 110px',
        slots: [
          { gridColumn: '1', gridRow: '1' },
          { gridColumn: '1', gridRow: '2' },
          { gridColumn: '2', gridRow: '1 / 3' },
        ],
      }
    }
    // Top wide, 2 bottom
    return {
      columns: '1fr 1fr',
      rows: '140px 100px',
      slots: [
        { gridColumn: '1 / 3', gridRow: '1' },
        { gridColumn: '1', gridRow: '2' },
        { gridColumn: '2', gridRow: '2' },
      ],
    }
  }

  if (count === 4) {
    const variant = pickVariant(momentId, 3)
    if (variant === 0) {
      // Big hero top-left spanning 2 rows, 2 small right, 1 bottom-left
      return {
        columns: '3fr 2fr',
        rows: '100px 100px 80px',
        slots: [
          { gridColumn: '1', gridRow: '1 / 3' },
          { gridColumn: '2', gridRow: '1' },
          { gridColumn: '2', gridRow: '2' },
          { gridColumn: '1 / 3', gridRow: '3' },
        ],
      }
    }
    if (variant === 1) {
      // L-shape: hero top spanning full width, 3 across bottom
      return {
        columns: '1fr 1fr 1fr',
        rows: '150px 90px',
        slots: [
          { gridColumn: '1 / 4', gridRow: '1' },
          { gridColumn: '1', gridRow: '2' },
          { gridColumn: '2', gridRow: '2' },
          { gridColumn: '3', gridRow: '2' },
        ],
      }
    }
    // 2×2 with uneven sizes
    return {
      columns: '2fr 3fr',
      rows: '130px 110px',
      slots: [
        { gridColumn: '1', gridRow: '1' },
        { gridColumn: '2', gridRow: '1' },
        { gridColumn: '1', gridRow: '2' },
        { gridColumn: '2', gridRow: '2' },
      ],
    }
  }

  if (count === 5) {
    const variant = pickVariant(momentId, 2)
    if (variant === 0) {
      // Hero tall left, 4 in 2×2 grid right
      return {
        columns: '3fr 2fr 2fr',
        rows: '110px 110px',
        slots: [
          { gridColumn: '1', gridRow: '1 / 3' },
          { gridColumn: '2', gridRow: '1' },
          { gridColumn: '3', gridRow: '1' },
          { gridColumn: '2', gridRow: '2' },
          { gridColumn: '3', gridRow: '2' },
        ],
      }
    }
    // Top row: 2 images, bottom row: 3 images
    return {
      columns: '1fr 1fr 1fr',
      rows: '130px 100px',
      slots: [
        { gridColumn: '1 / 3', gridRow: '1' },
        { gridColumn: '3', gridRow: '1' },
        { gridColumn: '1', gridRow: '2' },
        { gridColumn: '2', gridRow: '2' },
        { gridColumn: '3', gridRow: '2' },
      ],
    }
  }

  if (count === 6) {
    const variant = pickVariant(momentId, 2)
    if (variant === 0) {
      // Hero left 2 rows, 2×2 right, 2 across bottom
      return {
        columns: '3fr 2fr 2fr',
        rows: '95px 95px 80px',
        slots: [
          { gridColumn: '1', gridRow: '1 / 3' },
          { gridColumn: '2', gridRow: '1' },
          { gridColumn: '3', gridRow: '1' },
          { gridColumn: '2', gridRow: '2' },
          { gridColumn: '3', gridRow: '2' },
          { gridColumn: '1 / 4', gridRow: '3' },
        ],
      }
    }
    // 3 columns, top 2 with big left, bottom row 3 equal
    return {
      columns: '2fr 1fr 1fr',
      rows: '130px 100px',
      slots: [
        { gridColumn: '1', gridRow: '1 / 3' },
        { gridColumn: '2', gridRow: '1' },
        { gridColumn: '3', gridRow: '1' },
        { gridColumn: '1', gridRow: '2' },    // under hero — hero col only
        { gridColumn: '2', gridRow: '2' },
        { gridColumn: '3', gridRow: '2' },
      ],
    }
  }

  // 7+ images: show first 7 in a varied layout, +N badge on last cell
  return {
    columns: '2fr 1fr 1fr 1fr',
    rows: '110px 90px 90px',
    slots: [
      { gridColumn: '1', gridRow: '1 / 3' },    // hero tall left
      { gridColumn: '2', gridRow: '1' },
      { gridColumn: '3 / 5', gridRow: '1' },     // wide top-right
      { gridColumn: '2', gridRow: '2' },
      { gridColumn: '3', gridRow: '2' },
      { gridColumn: '4', gridRow: '2' },
      { gridColumn: '1 / 5', gridRow: '3' },     // panoramic bottom strip
    ],
  }
}

function MomentCollage({
  images,
  thumbs,
  momentId,
}: {
  images: MomentImage[]
  thumbs: Record<number, string>
  momentId: number
}) {
  if (images.length === 0) return null

  const hero = images.find((img) => img.is_hero) ?? images[0]
  // Put hero first, then the rest in original order
  const ordered = [hero, ...images.filter((img) => img.image_id !== hero.image_id)]

  const displayCount = Math.min(ordered.length, 7)
  const overflow = ordered.length - displayCount
  const layout = getLayout(displayCount, momentId)
  const shown = ordered.slice(0, displayCount)

  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{
        display: 'grid',
        gridTemplateColumns: layout.columns,
        gridTemplateRows: layout.rows,
        gap: '3px',
      }}
    >
      {shown.map((img, i) => {
        const isLast = i === displayCount - 1 && overflow > 0
        const slot = layout.slots[i]
        if (isLast) {
          // Last cell with +N overflow badge
          return (
            <div key={img.image_id} className="overflow-hidden relative" style={slot}>
              {thumbs[img.image_id] ? (
                <img
                  src={thumbs[img.image_id]}
                  alt=""
                  className="absolute inset-0 w-full h-full object-cover object-top"
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
}: {
  chapter: StoryChapter
  dateRange: string
  moments: StoryMoment[]
  momentImages: Record<string, MomentImage[]>
  heroThumbs: Record<number, string>
}) {
  return (
    <div className="p-5">
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
        <span className="text-xs text-neutral-600 ml-3 shrink-0">▾ collapse</span>
      </div>

      {moments.length === 0 ? (
        <div className="text-xs text-neutral-500 py-4">Loading moments…</div>
      ) : (
        <div className="space-y-4 mt-2">
          {moments.map((moment) => (
            <div key={moment.id}>
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
                  momentId={moment.id}
                />
              ) : moment.hero_image_id && heroThumbs[moment.hero_image_id] ? (
                <div className="rounded-lg overflow-hidden">
                  <img src={heroThumbs[moment.hero_image_id]} alt="" className="w-full h-44 object-cover object-top" />
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

function QuildedGrid({
  yearGroups,
  years,
  expandedChapter,
  moments,
  momentImages,
  heroThumbs,
  toggleChapter,
}: {
  yearGroups: Record<string, StoryChapter[]>
  years: string[]
  expandedChapter: string | null
  moments: StoryMoment[]
  momentImages: Record<string, MomentImage[]>
  heroThumbs: Record<number, string>
  toggleChapter: (id: string) => void
}) {
  // Measure container width to compute column count dynamically
  const containerRef = useRef<HTMLDivElement>(null)
  const [colCount, setColCount] = useState(3)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 900
      // ~280px per column, minimum 2, no hard max
      setColCount(Math.max(2, Math.floor(w / 280)))
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return (
    <div ref={containerRef} className="py-6 px-5">
      {years.map((year) => {
        const yearChapters = yearGroups[year]
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

            <div
              className="grid gap-2"
              style={{
                gridTemplateColumns: `repeat(${colCount}, 1fr)`,
                gridAutoRows: '100px',
              }}
            >
              {yearChapters.map((chapter, idx) => {
                const isExpanded = expandedChapter === chapter.id
                const coverThumb = chapter.cover_image_id ? heroThumbs[chapter.cover_image_id] : undefined
                const dateRange = formatChapterDateRange(chapter.start_date, chapter.end_date)

                /* Sizing pattern adapts to available columns.
                 * Feature cards span 2 cols (capped to colCount).
                 * Cycle length scales with colCount for variety. */
                let colSpan = 1
                let rowSpan = 2
                if (isExpanded) {
                  colSpan = colCount; rowSpan = 0
                } else {
                  const cycle = colCount + 4  // e.g. 3→7, 5→9, 7→11
                  const pos = idx % cycle
                  if (pos === 0) { colSpan = 2; rowSpan = 3 }           // feature
                  else if (pos === 2) { colSpan = 1; rowSpan = 3 }      // tall
                  else if (pos === Math.floor(cycle / 2)) { colSpan = 2; rowSpan = 2 }  // wide
                  // Additional tall card for wider grids
                  else if (colCount >= 5 && pos === colCount) { colSpan = 1; rowSpan = 3 }
                  // rest → standard 1×2
                  colSpan = Math.min(colSpan, colCount)
                }

                const gridStyle: React.CSSProperties = isExpanded
                  ? { gridColumn: '1 / -1' }
                  : { gridColumn: `span ${colSpan}`, gridRow: `span ${rowSpan}` }

                return (
                  <div key={chapter.id} style={gridStyle}>
                    <div
                      className={`group rounded-xl overflow-hidden transition-all duration-200 cursor-pointer h-full ${
                        isExpanded
                          ? 'bg-neutral-850 ring-1 ring-blue-500/20'
                          : 'bg-neutral-900/70 hover:bg-neutral-800/60 hover:ring-1 hover:ring-neutral-700/50'
                      }`}
                      onClick={() => toggleChapter(chapter.id)}
                    >
                      {isExpanded ? (
                        <ExpandedChapterDetail
                          chapter={chapter}
                          dateRange={dateRange}
                          moments={moments}
                          momentImages={momentImages}
                          heroThumbs={heroThumbs}
                        />
                      ) : (
                        <div className="relative w-full h-full overflow-hidden">
                          {coverThumb ? (
                            <>
                              <img
                                src={coverThumb}
                                alt=""
                                className="absolute inset-0 w-full h-full object-cover object-top transition-transform duration-300 group-hover:scale-105"
                              />
                              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/10 to-transparent" />
                            </>
                          ) : (
                            <div className="absolute inset-0 bg-neutral-800/60" />
                          )}
                          <div className="absolute bottom-0 left-0 right-0 p-3">
                            <h3 className={`font-semibold text-white drop-shadow-lg leading-tight ${
                              colSpan >= 2 ? 'text-base' : 'text-sm truncate'
                            }`}>
                              {chapter.title || 'Untitled Chapter'}
                            </h3>
                            <div className="flex items-center gap-1.5 mt-0.5 text-[11px] text-neutral-300/80">
                              {chapter.location && <span className="text-blue-400/90">📍 {chapter.location}</span>}
                              {dateRange && <span>{dateRange}</span>}
                              <span className="opacity-60">{chapter.image_count} photos</span>
                            </div>
                            {chapter.summary && colSpan >= 2 && (
                              <p className="text-xs text-neutral-400/80 italic line-clamp-2 leading-relaxed mt-1">
                                {chapter.summary}
                              </p>
                            )}
                          </div>
                        </div>
                      )}
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
}: {
  yearGroups: Record<string, StoryChapter[]>
  years: string[]
  expandedChapter: string | null
  moments: StoryMoment[]
  momentImages: Record<string, MomentImage[]>
  heroThumbs: Record<number, string>
  toggleChapter: (id: string) => void
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
                      <div className="mx-auto max-w-4xl">
                        <div
                          className="bg-neutral-850 ring-1 ring-blue-500/30 rounded-xl overflow-hidden cursor-pointer transition-all duration-200"
                          onClick={() => toggleChapter(chapter.id)}
                        >
                          <ExpandedChapterDetail
                            chapter={chapter}
                            dateRange={dateRange}
                            moments={moments}
                            momentImages={momentImages}
                            heroThumbs={heroThumbs}
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
                            className="group rounded-xl overflow-hidden cursor-pointer bg-neutral-900/70 hover:bg-neutral-800/60 ring-1 ring-neutral-800/50 hover:ring-neutral-700/50 transition-all duration-200 ml-auto max-w-md"
                            onClick={() => toggleChapter(chapter.id)}
                          >
                            {coverThumb ? (
                              <div className="relative h-48 overflow-hidden">
                                <img
                                  src={coverThumb}
                                  alt=""
                                  className="w-full h-full object-cover object-top transition-transform duration-300 group-hover:scale-105"
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
                            className="group rounded-xl overflow-hidden cursor-pointer bg-neutral-900/70 hover:bg-neutral-800/60 ring-1 ring-neutral-800/50 hover:ring-neutral-700/50 transition-all duration-200 mr-auto max-w-md"
                            onClick={() => toggleChapter(chapter.id)}
                          >
                            {coverThumb ? (
                              <div className="relative h-48 overflow-hidden">
                                <img
                                  src={coverThumb}
                                  alt=""
                                  className="w-full h-full object-cover object-top transition-transform duration-300 group-hover:scale-105"
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
  generating,
  narrating,
  exporting,
}: {
  album: SmartAlbumSummary
  chapters: StoryChapter[]
  onGenerateStory: () => void
  onGenerateNarrative: () => void
  onExportStory: () => void
  generating: boolean
  narrating: boolean
  exporting: boolean
}) {
  const [viewMode, setViewMode] = useState<ViewMode>('quilted')
  const [expandedChapter, setExpandedChapter] = useState<string | null>(null)
  const [moments, setMoments] = useState<StoryMoment[]>([])
  const [momentImages, setMomentImages] = useState<Record<string, MomentImage[]>>({})
  const [heroThumbs, setHeroThumbs] = useState<Record<number, string>>({})
  const loadingRef = useRef(false)
  const requestedThumbsRef = useRef<Set<number>>(new Set())
  const loadedMomentIdsRef = useRef<Set<string>>(new Set())
  const expandedChapterRef = useRef<string | null>(null)

  useEffect(() => {
    requestedThumbsRef.current.clear()
    loadedMomentIdsRef.current.clear()
    setMomentImages({})
    setHeroThumbs({})
  }, [album.id])

  const loadThumbnailIds = useCallback(async (imageIds: number[]) => {
    const unique = [...new Set(imageIds)].filter((id) => !requestedThumbsRef.current.has(id))
    if (unique.length === 0) return
    unique.forEach((id) => requestedThumbsRef.current.add(id))

    // Server caps at 50 per batch — chunk to avoid silent truncation
    const BATCH_SIZE = 50
    for (let i = 0; i < unique.length; i += BATCH_SIZE) {
      const chunk = unique.slice(i, i + BATCH_SIZE)
      try {
        const thumbs = await window.api.getThumbnailsBatch(chunk.map((imageId) => ({ image_id: imageId })))
        const mapped = mergeThumbnailMap(thumbs)
        setHeroThumbs((prev) => ({ ...prev, ...mapped }))
      } catch {
        chunk.forEach((id) => requestedThumbsRef.current.delete(id))
      }
    }
  }, [])

  useEffect(() => {
    const coverIds = chapters
      .map((ch) => ch.cover_image_id)
      .filter((id): id is number => id != null)
    void loadThumbnailIds(coverIds)
  }, [chapters, loadThumbnailIds])

  const loadMoments = useCallback(async (chapterId: string) => {
    if (loadingRef.current) return
    loadingRef.current = true
    try {
      const { moments: loaded } = await window.api.albumsChapterMoments(chapterId)
      if (expandedChapterRef.current !== chapterId) return
      const nextMoments = loaded as StoryMoment[]
      setMoments(nextMoments)
      const heroIds = nextMoments
        .map((m) => m.hero_image_id)
        .filter((id): id is number => id != null)
      void loadThumbnailIds(heroIds)
      for (const m of nextMoments) {
        void loadMomentImagesInternal(m.id)
      }
    } finally {
      loadingRef.current = false
    }
  }, [loadThumbnailIds]) // eslint-disable-line react-hooks/exhaustive-deps

  const loadMomentImagesInternal = useCallback(async (momentId: string) => {
    if (loadedMomentIdsRef.current.has(momentId)) return
    loadedMomentIdsRef.current.add(momentId)
    try {
      const { images } = await window.api.albumsMomentImages(momentId)
      const nextImages = images as MomentImage[]
      setMomentImages((prev) => ({ ...prev, [momentId]: nextImages }))
      void loadThumbnailIds(nextImages.map((img) => img.image_id))
    } catch {
      loadedMomentIdsRef.current.delete(momentId)
    }
  }, [loadThumbnailIds])

  const toggleChapter = useCallback((chapterId: string) => {
    if (expandedChapterRef.current === chapterId) {
      expandedChapterRef.current = null
      setExpandedChapter(null)
      setMoments([])
    } else {
      expandedChapterRef.current = chapterId
      setExpandedChapter(chapterId)
      void loadMoments(chapterId)
    }
  }, [loadMoments])

  const yearGroups = groupChaptersByYear(chapters)
  const years = Object.keys(yearGroups).sort()

  const sharedProps = {
    yearGroups,
    years,
    expandedChapter,
    moments,
    momentImages,
    heroThumbs,
    toggleChapter,
  }

  return (
    <div className="flex flex-col h-full">
      {/* ── Header bar ── */}
      <div className="shrink-0 px-6 py-4 border-b border-neutral-800 bg-neutral-900/60 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">{album.name}</h2>
            <p className="text-xs text-neutral-500 mt-0.5">
              {album.item_count} photos · {chapters.length} chapters
              {years.length > 1 && ` · ${years[0]}–${years[years.length - 1]}`}
            </p>
          </div>
          <div className="flex items-center gap-2">
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
              disabled={generating}
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
          <QuildedGrid {...sharedProps} />
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
    const d = new Date(iso)
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
  const [persons, setPersons] = useState<FacePerson[]>([])
  const [evalReport, setEvalReport] = useState<StoryGenerateResult['evaluation'] | null>(null)
  const [status, setStatus] = useState<StatusBanner | null>(null)
  const [presetDefinitions, setPresetDefinitions] = useState<Record<string, PresetDefinition>>(FALLBACK_PRESETS)

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
      setStatus({
        tone: generation.evaluation.overall_pass ? 'success' : 'error',
        text: `Story generated in ${generation.generation_time_s}s.`,
      })
    } catch (err) {
      console.error('Failed to generate story:', err)
      setStatus({ tone: 'error', text: 'Failed to generate story.' })
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
      setStatus({
        tone: 'success',
        text: `Updated ${result.chapters_updated} chapter summaries.`,
      })
    } catch (err) {
      console.error('Failed to generate narratives:', err)
      setStatus({ tone: 'error', text: 'Failed to generate chapter summaries.' })
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
      setStatus({ tone: 'success', text: `Exported story to ${result.path}` })
      await window.api.openPath(result.path)
    } catch (err) {
      console.error('Failed to export story:', err)
      setStatus({ tone: 'error', text: 'Failed to export story.' })
    } finally {
      setExporting(false)
    }
  }, [albums, selectedId])

  const handleCreated = useCallback(async (albumId: string) => {
    await loadAlbums()
    handleSelect(albumId)
  }, [loadAlbums, handleSelect])

  const handleDeleteAlbum = useCallback(async () => {
    if (!selectedId) return
    try {
      await window.api.albumsDelete(selectedId)
      setSelectedId(null)
      setChapters([])
      setEvalReport(null)
      setStatus({ tone: 'success', text: 'Album deleted.' })
      await loadAlbums()
    } catch (err) {
      console.error('Failed to delete album:', err)
      setStatus({ tone: 'error', text: 'Failed to delete album.' })
    }
  }, [selectedId, loadAlbums])

  const selectedAlbum = albums.find((album) => album.id === selectedId)

  return (
    <div className="flex h-full">
      <div className="w-64 shrink-0">
        <AlbumListPanel
          albums={albums}
          selectedId={selectedId}
          onSelect={handleSelect}
          onRefresh={() => void loadAlbums()}
          onCreate={() => setCreating(true)}
          onCreatePreset={() => setCreatingPreset(true)}
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
              generating={generating}
              narrating={narrating}
              exporting={exporting}
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
            <div className="px-4 py-2 border-t border-neutral-800 flex justify-end">
              <button
                onClick={() => void handleDeleteAlbum()}
                className="px-2 py-1 text-xs rounded bg-red-900/50 text-red-400 hover:bg-red-800/50"
              >
                Delete Album
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
    </div>
  )
}
