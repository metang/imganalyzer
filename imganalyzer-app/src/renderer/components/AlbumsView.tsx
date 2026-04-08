/**
 * AlbumsView — Smart albums list + story chapter timeline.
 *
 * Layout:
 *   Left panel  — album list + create dialog
 *   Right panel — story timeline (chapters → moments → images)
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import type {
  SmartAlbumSummary,
  StoryChapter,
  StoryMoment,
  MomentImage,
  StoryGenerateResult,
} from '../global'

// ── Album List Panel ────────────────────────────────────────────────────────

function AlbumListPanel({
  albums,
  selectedId,
  onSelect,
  onRefresh,
  onCreate,
}: {
  albums: SmartAlbumSummary[]
  selectedId: string | null
  onSelect: (id: string) => void
  onRefresh: () => void
  onCreate: () => void
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
            No albums yet. Click "+ New" to create one.
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
            {creating ? 'Creating…' : 'Create Album'}
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
  generating,
}: {
  album: SmartAlbumSummary
  chapters: StoryChapter[]
  onGenerateStory: () => void
  generating: boolean
}) {
  const [expandedChapter, setExpandedChapter] = useState<string | null>(null)
  const [moments, setMoments] = useState<StoryMoment[]>([])
  const [momentImages, setMomentImages] = useState<Record<string, MomentImage[]>>({})
  const [heroThumbs, setHeroThumbs] = useState<Record<number, string>>({})
  const loadingRef = useRef(false)

  const loadMoments = useCallback(async (chapterId: string) => {
    if (loadingRef.current) return
    loadingRef.current = true
    try {
      const { moments: ms } = await window.api.albumsChapterMoments(chapterId)
      setMoments(ms as StoryMoment[])

      // Load hero thumbnails
      const heroIds = (ms as StoryMoment[])
        .map((m) => m.hero_image_id)
        .filter((id): id is number => id != null)
      if (heroIds.length > 0) {
        const items = heroIds.map((id) => ({ file_path: '', image_id: id }))
        try {
          const thumbs = await window.api.getThumbnailsBatch(items)
          setHeroThumbs((prev) => ({ ...prev, ...Object.fromEntries(
            Object.entries(thumbs).map(([k, v]) => [Number(k) || k, v])
          ) }))
        } catch {
          // Thumbnails are best-effort
        }
      }
    } finally {
      loadingRef.current = false
    }
  }, [])

  const toggleChapter = useCallback((chapterId: string) => {
    if (expandedChapter === chapterId) {
      setExpandedChapter(null)
      setMoments([])
    } else {
      setExpandedChapter(chapterId)
      loadMoments(chapterId)
    }
  }, [expandedChapter, loadMoments])

  const loadMomentImages = useCallback(async (momentId: string) => {
    if (momentImages[momentId]) return
    const { images } = await window.api.albumsMomentImages(momentId)
    setMomentImages((prev) => ({ ...prev, [momentId]: images as MomentImage[] }))
  }, [momentImages])

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-neutral-700">
        <div>
          <h2 className="text-lg font-semibold text-white">{album.name}</h2>
          <p className="text-xs text-neutral-500">
            {album.item_count} images · {chapters.length} chapters
            {album.description && ` · ${album.description}`}
          </p>
        </div>
        <button
          onClick={onGenerateStory}
          disabled={generating}
          className="px-3 py-1.5 text-sm rounded bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-50"
        >
          {generating ? 'Generating…' : chapters.length > 0 ? '↻ Regenerate Story' : '▶ Generate Story'}
        </button>
      </div>

      {/* Chapters */}
      <div className="flex-1 overflow-y-auto">
        {chapters.length === 0 && (
          <div className="p-8 text-center text-neutral-500">
            <p className="text-sm">No story generated yet.</p>
            <p className="text-xs mt-1">Click "Generate Story" to auto-create chapters and moments.</p>
          </div>
        )}

        {chapters.map((ch, idx) => (
          <div key={ch.id} className="border-b border-neutral-800">
            {/* Chapter header */}
            <button
              onClick={() => toggleChapter(ch.id)}
              className="w-full text-left px-4 py-3 hover:bg-neutral-800/50 transition-colors"
            >
              <div className="flex items-center gap-3">
                <span className={`text-neutral-500 transition-transform ${expandedChapter === ch.id ? 'rotate-90' : ''}`}>
                  ▸
                </span>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-neutral-200 truncate">
                    {ch.title || `Chapter ${idx + 1}`}
                  </div>
                  <div className="text-xs text-neutral-500 mt-0.5">
                    {ch.image_count} images · {ch.moment_count} moments
                    {ch.location && ` · ${ch.location}`}
                  </div>
                </div>
                {ch.cover_image_id && heroThumbs[ch.cover_image_id] && (
                  <img
                    src={heroThumbs[ch.cover_image_id]}
                    className="w-10 h-10 rounded object-cover shrink-0"
                    alt=""
                  />
                )}
              </div>
            </button>

            {/* Expanded: moments */}
            {expandedChapter === ch.id && (
              <div className="bg-neutral-900/50 px-4 pb-3">
                {moments.length === 0 && (
                  <div className="text-xs text-neutral-500 py-2">Loading moments…</div>
                )}
                <div className="grid grid-cols-3 gap-2">
                  {moments.map((m) => (
                    <button
                      key={m.id}
                      onClick={() => loadMomentImages(m.id)}
                      className="text-left bg-neutral-800 rounded-lg overflow-hidden hover:ring-1 hover:ring-blue-500 transition-all"
                    >
                      {m.hero_image_id && heroThumbs[m.hero_image_id] ? (
                        <img
                          src={heroThumbs[m.hero_image_id]}
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
                          {m.title || formatMomentTime(m.start_time)}
                        </div>
                        <div className="text-[10px] text-neutral-500">
                          {m.image_count} images
                        </div>
                      </div>
                    </button>
                  ))}
                </div>

                {/* Expanded moment images */}
                {moments.map((m) =>
                  momentImages[m.id] ? (
                    <div key={`imgs-${m.id}`} className="mt-2">
                      <div className="text-xs text-neutral-500 mb-1">
                        {m.title || formatMomentTime(m.start_time)} — {m.image_count} images
                      </div>
                      <div className="flex gap-1 flex-wrap">
                        {momentImages[m.id].map((img) => (
                          <div
                            key={img.image_id}
                            className={`w-16 h-16 rounded overflow-hidden ${
                              img.is_hero ? 'ring-2 ring-yellow-500' : ''
                            }`}
                          >
                            {heroThumbs[img.image_id] ? (
                              <img src={heroThumbs[img.image_id]} className="w-full h-full object-cover" alt="" />
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
  const [generating, setGenerating] = useState(false)
  const [persons, setPersons] = useState<Array<{ id: number; name: string }>>([])
  const [evalReport, setEvalReport] = useState<StoryGenerateResult['evaluation'] | null>(null)

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
      setPersons(list.map((p: { id: number; name: string }) => ({ id: p.id, name: p.name })))
    } catch {
      // Persons may not exist yet
    }
  }, [])

  useEffect(() => {
    loadAlbums()
    loadPersons()
  }, [loadAlbums, loadPersons])

  const loadStory = useCallback(async (albumId: string) => {
    try {
      const { chapters: chs } = await window.api.albumsStory(albumId)
      setChapters(chs)
    } catch (err) {
      console.error('Failed to load story:', err)
      setChapters([])
    }
  }, [])

  const handleSelect = useCallback((albumId: string) => {
    setSelectedId(albumId)
    setEvalReport(null)
    loadStory(albumId)
  }, [loadStory])

  const handleGenerate = useCallback(async () => {
    if (!selectedId) return
    setGenerating(true)
    try {
      const result = await window.api.albumsStoryGenerate({ album_id: selectedId })
      const genResult = result as unknown as StoryGenerateResult
      setEvalReport(genResult.evaluation)
      await loadStory(selectedId)
      await loadAlbums() // refresh counts
    } catch (err) {
      console.error('Failed to generate story:', err)
    } finally {
      setGenerating(false)
    }
  }, [selectedId, loadStory, loadAlbums])

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
      await loadAlbums()
    } catch (err) {
      console.error('Failed to delete album:', err)
    }
  }, [selectedId, loadAlbums])

  const selectedAlbum = albums.find((a) => a.id === selectedId)

  return (
    <div className="flex h-full">
      {/* Left: album list */}
      <div className="w-64 shrink-0">
        <AlbumListPanel
          albums={albums}
          selectedId={selectedId}
          onSelect={handleSelect}
          onRefresh={loadAlbums}
          onCreate={() => setCreating(true)}
        />
      </div>

      {/* Right: story timeline */}
      <div className="flex-1 min-w-0 flex flex-col">
        {selectedAlbum ? (
          <>
            <StoryTimeline
              album={selectedAlbum}
              chapters={chapters}
              onGenerateStory={handleGenerate}
              generating={generating}
            />
            {/* Evaluation report banner */}
            {evalReport && (
              <div className={`px-4 py-2 border-t text-xs ${
                evalReport.overall_pass
                  ? 'border-emerald-800 bg-emerald-900/30 text-emerald-400'
                  : 'border-amber-800 bg-amber-900/30 text-amber-400'
              }`}>
                <span className="font-medium">
                  Evaluation: {evalReport.overall_pass ? '✓ PASS' : '✗ FAIL'}
                </span>
                {' — '}
                {Object.values(evalReport.criteria).filter((c) => c.passed).length}/
                {Object.keys(evalReport.criteria).length} criteria passed
              </div>
            )}
            {/* Delete button */}
            <div className="px-4 py-2 border-t border-neutral-800 flex justify-end">
              <button
                onClick={handleDeleteAlbum}
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
                Smart albums auto-collect images by rules (person, date, location)
              </p>
            </div>
          </div>
        )}
      </div>

      <CreateAlbumDialog
        open={creating}
        onClose={() => setCreating(false)}
        onCreated={handleCreated}
        persons={persons}
      />
    </div>
  )
}
