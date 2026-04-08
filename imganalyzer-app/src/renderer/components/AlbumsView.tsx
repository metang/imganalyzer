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
  const [expandedChapter, setExpandedChapter] = useState<string | null>(null)
  const [moments, setMoments] = useState<StoryMoment[]>([])
  const [momentImages, setMomentImages] = useState<Record<string, MomentImage[]>>({})
  const [heroThumbs, setHeroThumbs] = useState<Record<number, string>>({})
  const loadingRef = useRef(false)
  const requestedThumbsRef = useRef<Set<number>>(new Set())
  const loadedMomentIdsRef = useRef<Set<string>>(new Set())
  const expandedChapterRef = useRef<string | null>(null)

  // Reset state when album changes
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
    try {
      const thumbs = await window.api.getThumbnailsBatch(unique.map((imageId) => ({ image_id: imageId })))
      const mapped = mergeThumbnailMap(thumbs)
      setHeroThumbs((prev) => ({ ...prev, ...mapped }))
    } catch {
      unique.forEach((id) => requestedThumbsRef.current.delete(id))
    }
  }, [])

  useEffect(() => {
    const coverIds = chapters
      .map((chapter) => chapter.cover_image_id)
      .filter((imageId): imageId is number => imageId != null)
    void loadThumbnailIds(coverIds)
  }, [chapters, loadThumbnailIds])

  const loadMoments = useCallback(async (chapterId: string) => {
    if (loadingRef.current) return
    loadingRef.current = true
    try {
      const { moments: loadedMoments } = await window.api.albumsChapterMoments(chapterId)
      // Guard against stale responses from a previously expanded chapter
      if (expandedChapterRef.current !== chapterId) return
      const nextMoments = loadedMoments as StoryMoment[]
      setMoments(nextMoments)

      const heroIds = nextMoments
        .map((moment) => moment.hero_image_id)
        .filter((imageId): imageId is number => imageId != null)
      void loadThumbnailIds(heroIds)
    } finally {
      loadingRef.current = false
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

  const loadMomentImages = useCallback(async (momentId: string) => {
    if (loadedMomentIdsRef.current.has(momentId)) return
    loadedMomentIdsRef.current.add(momentId)
    try {
      const { images } = await window.api.albumsMomentImages(momentId)
      const nextImages = images as MomentImage[]
      setMomentImages((prev) => ({ ...prev, [momentId]: nextImages }))
      void loadThumbnailIds(nextImages.map((image) => image.image_id))
    } catch {
      loadedMomentIdsRef.current.delete(momentId)
    }
  }, [loadThumbnailIds])

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-neutral-700">
        <div>
          <h2 className="text-lg font-semibold text-white">{album.name}</h2>
          <p className="text-xs text-neutral-500">
            {album.item_count} images · {chapters.length} chapters
            {album.description && ` · ${album.description}`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onGenerateNarrative}
            disabled={narrating || chapters.length === 0}
            className="px-3 py-1.5 text-sm rounded bg-violet-700 text-white hover:bg-violet-600 disabled:opacity-50"
          >
            {narrating ? 'Summarizing...' : 'Generate Summaries'}
          </button>
          <button
            onClick={onExportStory}
            disabled={exporting || chapters.length === 0}
            className="px-3 py-1.5 text-sm rounded bg-neutral-700 text-white hover:bg-neutral-600 disabled:opacity-50"
          >
            {exporting ? 'Exporting...' : 'Export HTML'}
          </button>
          <button
            onClick={onGenerateStory}
            disabled={generating}
            className="px-3 py-1.5 text-sm rounded bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-50"
          >
            {generating ? 'Generating...' : chapters.length > 0 ? 'Regenerate Story' : 'Generate Story'}
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {chapters.length === 0 && (
          <div className="p-8 text-center text-neutral-500">
            <p className="text-sm">No story generated yet.</p>
            <p className="text-xs mt-1">Generate the story, then create summaries or export it.</p>
          </div>
        )}

        {chapters.map((chapter, idx) => (
          <div key={chapter.id} className="border-b border-neutral-800">
            <button
              onClick={() => toggleChapter(chapter.id)}
              className="w-full text-left px-4 py-3 hover:bg-neutral-800/50 transition-colors"
            >
              <div className="flex items-center gap-3">
                <span className={`text-neutral-500 transition-transform ${expandedChapter === chapter.id ? 'rotate-90' : ''}`}>
                  ▸
                </span>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-neutral-200 truncate">
                    {chapter.title || `Chapter ${idx + 1}`}
                  </div>
                  <div className="text-xs text-neutral-500 mt-0.5">
                    {chapter.image_count} images · {chapter.moment_count} moments
                    {chapter.location && ` · ${chapter.location}`}
                  </div>
                  {chapter.summary && (
                    <div className="text-xs text-neutral-400 mt-1 line-clamp-2">
                      {chapter.summary}
                    </div>
                  )}
                </div>
                {chapter.cover_image_id && heroThumbs[chapter.cover_image_id] && (
                  <img
                    src={heroThumbs[chapter.cover_image_id]}
                    className="w-10 h-10 rounded object-cover shrink-0"
                    alt=""
                  />
                )}
              </div>
            </button>

            {expandedChapter === chapter.id && (
              <div className="bg-neutral-900/50 px-4 pb-3">
                {moments.length === 0 && (
                  <div className="text-xs text-neutral-500 py-2">Loading moments...</div>
                )}
                <div className="grid grid-cols-3 gap-2">
                  {moments.map((moment) => (
                    <button
                      key={moment.id}
                      onClick={() => void loadMomentImages(moment.id)}
                      className="text-left bg-neutral-800 rounded-lg overflow-hidden hover:ring-1 hover:ring-blue-500 transition-all"
                    >
                      {moment.hero_image_id && heroThumbs[moment.hero_image_id] ? (
                        <img
                          src={heroThumbs[moment.hero_image_id]}
                          className="w-full h-24 object-cover"
                          alt=""
                        />
                      ) : (
                        <div className="w-full h-24 bg-neutral-700 flex items-center justify-center text-neutral-500 text-xs">
                          No preview
                        </div>
                      )}
                      <div className="px-2 py-1.5">
                        <div className="text-xs text-neutral-400 truncate">
                          {moment.title || formatMomentTime(moment.start_time)}
                        </div>
                        <div className="text-[10px] text-neutral-500">
                          {moment.image_count} images
                        </div>
                      </div>
                    </button>
                  ))}
                </div>

                {moments.map((moment) =>
                  momentImages[moment.id] ? (
                    <div key={`imgs-${moment.id}`} className="mt-2">
                      <div className="text-xs text-neutral-500 mb-1">
                        {moment.title || formatMomentTime(moment.start_time)} — {moment.image_count} images
                      </div>
                      <div className="flex gap-1 flex-wrap">
                        {momentImages[moment.id].map((image) => (
                          <div
                            key={image.image_id}
                            className={`w-16 h-16 rounded overflow-hidden ${
                              image.is_hero ? 'ring-2 ring-yellow-500' : ''
                            }`}
                          >
                            {heroThumbs[image.image_id] ? (
                              <img src={heroThumbs[image.image_id]} className="w-full h-full object-cover" alt="" />
                            ) : (
                              <div className="w-full h-full bg-neutral-700" />
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null,
                )}
              </div>
            )}
          </div>
        ))}
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
