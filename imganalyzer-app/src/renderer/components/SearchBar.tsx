/**
 * SearchBar.tsx — Quick search bar with collapsible advanced filters panel.
 *
 * The quick bar parses natural-language shortcuts inline:
 *   "landscape score>7"  → query="landscape" + aestheticMin=7
 *   "cat has:faces"      → query="cat" + facesMin=1
 *   "camera:Sony"        → camera="Sony"
 * Full structured filters are available via the "Filters" toggle.
 */
import { useState, useCallback, useRef, useEffect } from 'react'
import type { SearchFilters } from '../global'

interface SearchBarProps {
  onSearch: (filters: SearchFilters) => void
  loading: boolean
  resultCount: number | null
}

// ── Inline query parser ────────────────────────────────────────────────────────

interface ParsedQuery {
  textQuery: string
  patches: Partial<SearchFilters>
}

/**
 * Parse shorthand tokens out of a freeform query string.
 * Tokens supported:
 *   score>N  score>=N  score<N  score<=N
 *   sharpness>N  noise<N
 *   iso>N  iso<N
 *   faces>N  has:faces  no:faces  has:people  no:people
 *   camera:X  lens:X  location:X  face:X
 *   mode:text  mode:semantic  mode:hybrid
 */
function parseQuery(raw: string): ParsedQuery {
  const patches: Partial<SearchFilters> = {}
  let text = raw

  // Helper to strip a token and update `text`
  const strip = (re: RegExp) => {
    const m = text.match(re)
    if (m) { text = text.replace(m[0], '').trim() }
    return m
  }

  // score comparisons
  const scoreGte = strip(/\bscore\s*>=?\s*(\d+(?:\.\d+)?)/i)
  if (scoreGte) patches.aestheticMin = parseFloat(scoreGte[1])
  const scoreLte = strip(/\bscore\s*<=?\s*(\d+(?:\.\d+)?)/i)
  if (scoreLte) patches.aestheticMax = parseFloat(scoreLte[1])

  // sharpness comparisons
  const sharpGte = strip(/\bsharpness\s*>=?\s*(\d+(?:\.\d+)?)/i)
  if (sharpGte) patches.sharpnessMin = parseFloat(sharpGte[1])
  const sharpLte = strip(/\bsharpness\s*<=?\s*(\d+(?:\.\d+)?)/i)
  if (sharpLte) patches.sharpnessMax = parseFloat(sharpLte[1])

  // noise comparisons
  const noiseLte = strip(/\bnoise\s*<=?\s*(\d+(?:\.\d+)?)/i)
  if (noiseLte) patches.noiseMax = parseFloat(noiseLte[1])

  // iso comparisons
  const isoGte = strip(/\biso\s*>=?\s*(\d+)/i)
  if (isoGte) patches.isoMin = parseInt(isoGte[1])
  const isoLte = strip(/\biso\s*<=?\s*(\d+)/i)
  if (isoLte) patches.isoMax = parseInt(isoLte[1])

  // faces
  const facesGte = strip(/\bfaces?\s*>=?\s*(\d+)/i)
  if (facesGte) patches.facesMin = parseInt(facesGte[1])
  strip(/\bhas:faces?\b/i) && (patches.facesMin = patches.facesMin ?? 1)
  strip(/\bno:faces?\b/i) && (patches.facesMax = 0)

  // people
  const hasPeople = strip(/\bhas:people\b/i)
  if (hasPeople) patches.hasPeople = true
  const noPeople = strip(/\bno:people\b/i)
  if (noPeople) patches.hasPeople = false

  // camera:X
  const camM = strip(/\bcamera:(\S+)/i)
  if (camM) patches.camera = camM[1].replace(/^["']|["']$/g, '')

  // lens:X
  const lensM = strip(/\blens:(\S+)/i)
  if (lensM) patches.lens = lensM[1].replace(/^["']|["']$/g, '')

  // location:X
  const locM = strip(/\blocation:(\S+)/i)
  if (locM) patches.location = locM[1].replace(/^["']|["']$/g, '')

  // face:X
  const faceM = strip(/\bface:(\S+)/i)
  if (faceM) patches.face = faceM[1].replace(/^["']|["']$/g, '')

  // mode:X
  const modeM = strip(/\bmode:(text|semantic|hybrid|browse)\b/i)
  if (modeM) patches.mode = modeM[1].toLowerCase() as SearchFilters['mode']

  // Remove repeated spaces
  text = text.replace(/\s+/g, ' ').trim()

  return { textQuery: text, patches }
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Label({ children }: { children: React.ReactNode }) {
  return (
    <label className="text-[11px] font-semibold uppercase tracking-wider text-neutral-500 mb-0.5 block">
      {children}
    </label>
  )
}

function RangeInput({
  label,
  minVal, maxVal,
  minPlaceholder, maxPlaceholder,
  onMin, onMax,
  step = 0.1,
  min = 0,
  max = 10,
}: {
  label: string
  minVal: string; maxVal: string
  minPlaceholder: string; maxPlaceholder: string
  onMin: (v: string) => void; onMax: (v: string) => void
  step?: number; min?: number; max?: number
}) {
  return (
    <div>
      <Label>{label}</Label>
      <div className="flex gap-1.5">
        <input
          type="number"
          value={minVal}
          onChange={(e) => onMin(e.target.value)}
          placeholder={minPlaceholder}
          step={step} min={min} max={max}
          className="w-full px-2 py-1 text-xs bg-neutral-800 border border-neutral-700 rounded text-neutral-200 placeholder-neutral-600 focus:outline-none focus:border-blue-500"
        />
        <input
          type="number"
          value={maxVal}
          onChange={(e) => onMax(e.target.value)}
          placeholder={maxPlaceholder}
          step={step} min={min} max={max}
          className="w-full px-2 py-1 text-xs bg-neutral-800 border border-neutral-700 rounded text-neutral-200 placeholder-neutral-600 focus:outline-none focus:border-blue-500"
        />
      </div>
    </div>
  )
}

function TextInput({
  label, value, placeholder, onChange,
}: {
  label: string; value: string; placeholder: string; onChange: (v: string) => void
}) {
  return (
    <div>
      <Label>{label}</Label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full px-2 py-1 text-xs bg-neutral-800 border border-neutral-700 rounded text-neutral-200 placeholder-neutral-600 focus:outline-none focus:border-blue-500"
      />
    </div>
  )
}

// ── State type for advanced filter form fields (all strings for input binding) ─

interface AdvancedState {
  mode: 'text' | 'semantic' | 'hybrid'
  semanticWeight: string
  camera: string
  lens: string
  location: string
  face: string
  dateFrom: string
  dateTo: string
  aestheticMin: string
  aestheticMax: string
  sharpnessMin: string
  sharpnessMax: string
  noiseMax: string
  isoMin: string
  isoMax: string
  facesMin: string
  facesMax: string
  hasPeople: 'any' | 'yes' | 'no'
}

function defaultAdvanced(): AdvancedState {
  return {
    mode: 'hybrid',
    semanticWeight: '0.5',
    camera: '',
    lens: '',
    location: '',
    face: '',
    dateFrom: '',
    dateTo: '',
    aestheticMin: '',
    aestheticMax: '',
    sharpnessMin: '',
    sharpnessMax: '',
    noiseMax: '',
    isoMin: '',
    isoMax: '',
    facesMin: '',
    facesMax: '',
    hasPeople: 'any',
  }
}

function advancedToFilters(adv: AdvancedState): Partial<SearchFilters> {
  const f: Partial<SearchFilters> = {
    mode: adv.mode,
    semanticWeight: parseFloat(adv.semanticWeight) || 0.5,
  }
  if (adv.camera.trim())     f.camera = adv.camera.trim()
  if (adv.lens.trim())       f.lens = adv.lens.trim()
  if (adv.location.trim())   f.location = adv.location.trim()
  if (adv.face.trim())       f.face = adv.face.trim()
  if (adv.dateFrom.trim())   f.dateFrom = adv.dateFrom.trim()
  if (adv.dateTo.trim())     f.dateTo = adv.dateTo.trim()
  if (adv.aestheticMin !== '') f.aestheticMin = parseFloat(adv.aestheticMin)
  if (adv.aestheticMax !== '') f.aestheticMax = parseFloat(adv.aestheticMax)
  if (adv.sharpnessMin !== '') f.sharpnessMin = parseFloat(adv.sharpnessMin)
  if (adv.sharpnessMax !== '') f.sharpnessMax = parseFloat(adv.sharpnessMax)
  if (adv.noiseMax !== '')    f.noiseMax = parseFloat(adv.noiseMax)
  if (adv.isoMin !== '')      f.isoMin = parseInt(adv.isoMin)
  if (adv.isoMax !== '')      f.isoMax = parseInt(adv.isoMax)
  if (adv.facesMin !== '')    f.facesMin = parseInt(adv.facesMin)
  if (adv.facesMax !== '')    f.facesMax = parseInt(adv.facesMax)
  if (adv.hasPeople === 'yes') f.hasPeople = true
  if (adv.hasPeople === 'no')  f.hasPeople = false
  return f
}

/** Count how many advanced filters are non-default */
function countActiveFilters(adv: AdvancedState): number {
  let n = 0
  if (adv.camera)       n++
  if (adv.lens)         n++
  if (adv.location)     n++
  if (adv.face)         n++
  if (adv.dateFrom)     n++
  if (adv.dateTo)       n++
  if (adv.aestheticMin) n++
  if (adv.aestheticMax) n++
  if (adv.sharpnessMin) n++
  if (adv.sharpnessMax) n++
  if (adv.noiseMax)     n++
  if (adv.isoMin)       n++
  if (adv.isoMax)       n++
  if (adv.facesMin)     n++
  if (adv.facesMax)     n++
  if (adv.hasPeople !== 'any') n++
  if (adv.mode !== 'hybrid')   n++
  return n
}

// ── Main component ─────────────────────────────────────────────────────────────

export function SearchBar({ onSearch, loading, resultCount }: SearchBarProps) {
  const [rawQuery, setRawQuery] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  const [adv, setAdv] = useState<AdvancedState>(defaultAdvanced())
  const inputRef = useRef<HTMLInputElement>(null)

  const setAdv1 = useCallback(<K extends keyof AdvancedState>(key: K, val: AdvancedState[K]) => {
    setAdv((prev) => ({ ...prev, [key]: val }))
  }, [])

  const handleSearch = useCallback(() => {
    const { textQuery, patches } = parseQuery(rawQuery)
    const advFilters = advancedToFilters(adv)

    const filters: SearchFilters = {
      ...advFilters,
      ...patches, // inline tokens override advanced fields
      query: textQuery || undefined,
      mode: patches.mode ?? advFilters.mode ?? 'hybrid',
      limit: 500,
    }

    // If no text query and mode would invoke CLIP, fall back to browse
    if (!filters.query && !filters.face) {
      filters.mode = 'browse'
    }

    onSearch(filters)
  }, [rawQuery, adv, onSearch])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch()
  }, [handleSearch])

  const handleClear = useCallback(() => {
    setRawQuery('')
    setAdv(defaultAdvanced())
  }, [])

  // Auto-focus on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const activeFilterCount = countActiveFilters(adv)

  return (
    <div className="shrink-0 border-b border-neutral-800 bg-neutral-950">
      {/* ── Quick search bar ───────────────────────────────────────────────── */}
      <div className="flex items-center gap-2 px-4 py-3">
        {/* Search icon */}
        <svg className="w-4 h-4 text-neutral-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11A6 6 0 115 11a6 6 0 0112 0z" />
        </svg>

        {/* Text input */}
        <input
          ref={inputRef}
          type="text"
          value={rawQuery}
          onChange={(e) => setRawQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder='Search images… e.g. "sunset mountain score>7 has:people"'
          className="flex-1 bg-transparent text-sm text-neutral-200 placeholder-neutral-600 focus:outline-none"
        />

        {/* Clear */}
        {(rawQuery || activeFilterCount > 0) && (
          <button
            onClick={handleClear}
            className="text-neutral-600 hover:text-neutral-400 transition-colors shrink-0"
            title="Clear search"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}

        {/* Filters toggle */}
        <button
          onClick={() => setShowFilters((v) => !v)}
          className={`
            flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs transition-colors shrink-0
            ${showFilters || activeFilterCount > 0
              ? 'bg-blue-600/30 text-blue-300 border border-blue-600/50'
              : 'bg-neutral-800 text-neutral-400 hover:text-neutral-200 border border-neutral-700'}
          `}
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 4h18M7 8h10M11 12h2M13 16h-2" />
          </svg>
          Filters
          {activeFilterCount > 0 && (
            <span className="bg-blue-500 text-white text-[10px] rounded-full w-4 h-4 flex items-center justify-center font-bold">
              {activeFilterCount}
            </span>
          )}
        </button>

        {/* Search button */}
        <button
          onClick={handleSearch}
          disabled={loading}
          className="px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-colors shrink-0 disabled:opacity-50"
        >
          {loading ? (
            <div className="w-3.5 h-3.5 border border-white/50 border-t-white rounded-full animate-spin" />
          ) : 'Search'}
        </button>
      </div>

      {/* ── Hint line ──────────────────────────────────────────────────────── */}
      <div className="px-4 pb-2 flex items-center gap-3">
        <span className="text-[11px] text-neutral-700">
          Shortcuts: <code className="text-neutral-600">score&gt;7</code>&nbsp;
          <code className="text-neutral-600">has:faces</code>&nbsp;
          <code className="text-neutral-600">camera:Sony</code>&nbsp;
          <code className="text-neutral-600">iso&lt;800</code>&nbsp;
          <code className="text-neutral-600">mode:semantic</code>
        </span>
        {resultCount !== null && (
          <span className="text-[11px] text-neutral-600 ml-auto">
            {resultCount} result{resultCount !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* ── Advanced filters panel ─────────────────────────────────────────── */}
      {showFilters && (
        <div className="border-t border-neutral-800 px-4 py-4 grid grid-cols-2 gap-x-6 gap-y-4 bg-neutral-900/50">

          {/* Search mode */}
          <div>
            <Label>Search mode</Label>
            <div className="flex gap-1">
              {(['hybrid', 'semantic', 'text'] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setAdv1('mode', m)}
                  className={`px-2 py-1 rounded text-xs capitalize transition-colors ${
                    adv.mode === m
                      ? 'bg-blue-600 text-white'
                      : 'bg-neutral-800 text-neutral-400 hover:text-neutral-200'
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>

          {/* Semantic weight */}
          {adv.mode === 'hybrid' && (
            <div>
              <Label>Semantic weight ({adv.semanticWeight})</Label>
              <input
                type="range"
                min="0" max="1" step="0.05"
                value={adv.semanticWeight}
                onChange={(e) => setAdv1('semanticWeight', e.target.value)}
                className="w-full accent-blue-500"
              />
            </div>
          )}

          {/* Aesthetic score range */}
          <RangeInput
            label="Aesthetic score (0–10)"
            minVal={adv.aestheticMin} maxVal={adv.aestheticMax}
            minPlaceholder="min" maxPlaceholder="max"
            onMin={(v) => setAdv1('aestheticMin', v)}
            onMax={(v) => setAdv1('aestheticMax', v)}
            step={0.5} min={0} max={10}
          />

          {/* Sharpness range */}
          <RangeInput
            label="Sharpness (0–100)"
            minVal={adv.sharpnessMin} maxVal={adv.sharpnessMax}
            minPlaceholder="min" maxPlaceholder="max"
            onMin={(v) => setAdv1('sharpnessMin', v)}
            onMax={(v) => setAdv1('sharpnessMax', v)}
            step={1} min={0} max={100}
          />

          {/* ISO range */}
          <RangeInput
            label="ISO range"
            minVal={adv.isoMin} maxVal={adv.isoMax}
            minPlaceholder="min" maxPlaceholder="max"
            onMin={(v) => setAdv1('isoMin', v)}
            onMax={(v) => setAdv1('isoMax', v)}
            step={100} min={50} max={204800}
          />

          {/* Face count range */}
          <RangeInput
            label="Face count"
            minVal={adv.facesMin} maxVal={adv.facesMax}
            minPlaceholder="min" maxPlaceholder="max"
            onMin={(v) => setAdv1('facesMin', v)}
            onMax={(v) => setAdv1('facesMax', v)}
            step={1} min={0} max={100}
          />

          {/* Camera */}
          <TextInput label="Camera" value={adv.camera} placeholder="Sony, Canon R5…" onChange={(v) => setAdv1('camera', v)} />

          {/* Lens */}
          <TextInput label="Lens" value={adv.lens} placeholder="85mm, Sigma…" onChange={(v) => setAdv1('lens', v)} />

          {/* Location */}
          <TextInput label="Location" value={adv.location} placeholder="Paris, Tokyo…" onChange={(v) => setAdv1('location', v)} />

          {/* Face identity */}
          <TextInput label="Face identity" value={adv.face} placeholder="Alice, Bob…" onChange={(v) => setAdv1('face', v)} />

          {/* Date range */}
          <div>
            <Label>Date from</Label>
            <input
              type="date"
              value={adv.dateFrom}
              onChange={(e) => setAdv1('dateFrom', e.target.value)}
              className="w-full px-2 py-1 text-xs bg-neutral-800 border border-neutral-700 rounded text-neutral-200 focus:outline-none focus:border-blue-500"
            />
          </div>
          <div>
            <Label>Date to</Label>
            <input
              type="date"
              value={adv.dateTo}
              onChange={(e) => setAdv1('dateTo', e.target.value)}
              className="w-full px-2 py-1 text-xs bg-neutral-800 border border-neutral-700 rounded text-neutral-200 focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Max noise */}
          <div>
            <Label>Max noise level</Label>
            <input
              type="number"
              value={adv.noiseMax}
              onChange={(e) => setAdv1('noiseMax', e.target.value)}
              placeholder="e.g. 0.05"
              step={0.01} min={0}
              className="w-full px-2 py-1 text-xs bg-neutral-800 border border-neutral-700 rounded text-neutral-200 placeholder-neutral-600 focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Has people */}
          <div>
            <Label>People</Label>
            <div className="flex gap-1">
              {(['any', 'yes', 'no'] as const).map((v) => (
                <button
                  key={v}
                  onClick={() => setAdv1('hasPeople', v)}
                  className={`px-2 py-1 rounded text-xs capitalize transition-colors ${
                    adv.hasPeople === v
                      ? 'bg-blue-600 text-white'
                      : 'bg-neutral-800 text-neutral-400 hover:text-neutral-200'
                  }`}
                >
                  {v}
                </button>
              ))}
            </div>
          </div>

          {/* Apply + Reset buttons */}
          <div className="col-span-2 flex justify-end gap-2 pt-2">
            <button
              onClick={handleClear}
              className="px-3 py-1.5 rounded text-xs bg-neutral-800 text-neutral-400 hover:text-neutral-200 transition-colors"
            >
              Reset all
            </button>
            <button
              onClick={() => { setShowFilters(false); handleSearch() }}
              className="px-3 py-1.5 rounded text-xs bg-blue-600 hover:bg-blue-500 text-white font-medium transition-colors"
            >
              Apply filters
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
