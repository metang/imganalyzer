/**
 * SearchBar.tsx — compact search controls for the left search sidebar.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type {
  SearchFaceMatch,
  SearchFilters,
  SearchIntent,
  SearchSemanticProfile,
  SearchSortBy,
  SearchTimeOfDay,
} from '../global'

interface SearchBarProps {
  onSearch: (filters: SearchFilters, contextLabel: string | null) => void
  loading: boolean
}

interface ParsedQuery {
  textQuery: string
  patches: Partial<SearchFilters>
}

interface SearchDraft {
  intent: SearchIntent
  prompt: string
  activity: string
  species: string
  faces: string[]
  faceMatch: SearchFaceMatch
  country: string
  location: string
  recurringDate: string
  timeOfDay: SearchTimeOfDay | 'any'
  sortBy: SearchSortBy
  mode: 'text' | 'semantic' | 'hybrid'
  semanticWeight: string
  semanticProfile: SearchSemanticProfile
  camera: string
  lens: string
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
  includeRelatedSpecies: boolean
  expandedTerms: string[]
  mustTerms: string[]
  shouldTerms: string[]
}

interface ChipDescriptor {
  id: string
  label: string
  tone?: 'accent' | 'neutral'
}

const SORT_LABELS: Record<SearchSortBy, string> = {
  relevance: 'Relevance',
  best: 'Best overall',
  aesthetic: 'Most aesthetic',
  sharpness: 'Sharpest',
  cleanest: 'Cleanest',
  newest: 'Newest',
}

const TIME_LABELS: Record<SearchTimeOfDay, string> = {
  morning: 'Morning',
  afternoon: 'Afternoon',
  evening: 'Evening',
  night: 'Night',
}

const SEMANTIC_PROFILE_LABELS: Record<SearchSemanticProfile, string> = {
  image_dominant: 'Image CLIP dominant',
  balanced: 'Balanced',
  description_dominant: 'Description CLIP dominant',
}

const SEMANTIC_PROFILE_VALUE_TO_KEY: Record<'0' | '1' | '2', SearchSemanticProfile> = {
  '0': 'image_dominant',
  '1': 'balanced',
  '2': 'description_dominant',
}

const SEMANTIC_PROFILE_KEY_TO_VALUE: Record<SearchSemanticProfile, '0' | '1' | '2'> = {
  image_dominant: '0',
  balanced: '1',
  description_dominant: '2',
}

const INTENT_COPY: Record<SearchIntent, { title: string; placeholder: string }> = {
  general: {
    title: 'General',
    placeholder: 'Search anything: golden gate sunset, portrait with backlight, snowy owl...',
  },
  people: {
    title: 'People',
    placeholder: 'wyy in the US every Feb 1 morning playing basketball',
  },
  wildlife: {
    title: 'Wildlife',
    placeholder: 'ducks in flight over water',
  },
  'best-shot': {
    title: 'Best Shot',
    placeholder: 'best photo of the sunset scene from Yosemite',
  },
}

const WILDLIFE_EXPANSIONS: Record<string, string[]> = {
  bird: ['birds in flight', 'flying birds', 'flock of birds'],
  duck: ['mallard', 'teal', 'pintail', 'wigeon', 'gadwall', 'shoveler', 'wood duck', 'merganser'],
  goose: ['canada goose', 'snow goose', 'greylag goose', 'barnacle goose'],
  owl: ['barn owl', 'snowy owl', 'great horned owl', 'eagle owl'],
  hawk: ['red-tailed hawk', 'sparrowhawk', 'goshawk', 'kestrel'],
  eagle: ['bald eagle', 'golden eagle', 'white-tailed eagle'],
  heron: ['grey heron', 'great blue heron', 'egret', 'bittern'],
  gull: ['herring gull', 'tern', 'kittiwake', 'black-headed gull'],
  flock: ['flock of birds', 'birds in flight', 'flying birds'],
}

function dedupeStrings(values: string[]): string[] {
  const seen = new Set<string>()
  const results: string[] = []
  for (const value of values) {
    const clean = value.trim()
    const lowered = clean.toLowerCase()
    if (!clean || seen.has(lowered)) continue
    seen.add(lowered)
    results.push(clean)
  }
  return results
}

function splitFaceInput(value: string): string[] {
  return dedupeStrings(
    value
      .replace(/\r/g, '\n')
      .split(/\s*(?:,|;|\n|\band\b|&)\s*/i)
  )
}

function splitTermInput(value: string): string[] {
  return dedupeStrings(
    value
      .replace(/\r/g, '\n')
      .split(/\s*(?:,|;|\n)\s*/i)
  )
}

function parseQuery(raw: string): ParsedQuery {
  const patches: Partial<SearchFilters> = {}
  let text = raw

  const strip = (re: RegExp) => {
    const match = text.match(re)
    if (match) text = text.replace(match[0], '').trim()
    return match
  }

  const explicitFaceMatches = [...text.matchAll(/\bface:(?:"([^"]+)"|'([^']+)'|([^\s,]+))/gi)]
  if (explicitFaceMatches.length > 0) {
    const faces = dedupeStrings(
      explicitFaceMatches.map((match) => match[1] || match[2] || match[3] || '')
    )
    if (faces.length > 0) {
      patches.faces = faces
      patches.face = faces[0]
      if (faces.length > 1) patches.faceMatch = 'all'
    }
    text = text.replace(/\bface:(?:"[^"]+"|'[^']+'|[^\s,]+)/gi, '').trim()
  }

  const explicitMustMatches = [...text.matchAll(/\bmust:(?:"([^"]+)"|'([^']+)'|([^\s,]+))/gi)]
  if (explicitMustMatches.length > 0) {
    const mustTerms = dedupeStrings(
      explicitMustMatches.map((match) => match[1] || match[2] || match[3] || '')
    )
    if (mustTerms.length > 0) patches.mustTerms = mustTerms
    text = text.replace(/\bmust:(?:"[^"]+"|'[^']+'|[^\s,]+)/gi, '').trim()
  }

  const explicitShouldMatches = [...text.matchAll(/\bshould:(?:"([^"]+)"|'([^']+)'|([^\s,]+))/gi)]
  if (explicitShouldMatches.length > 0) {
    const shouldTerms = dedupeStrings(
      explicitShouldMatches.map((match) => match[1] || match[2] || match[3] || '')
    )
    if (shouldTerms.length > 0) patches.shouldTerms = shouldTerms
    text = text.replace(/\bshould:(?:"[^"]+"|'[^']+'|[^\s,]+)/gi, '').trim()
  }

  const aestheticRankingPhrase = strip(
    /^\s*(?:show|find)\s+(?:me\s+)?(?:the\s+)?(?:most\s+)?(?:beautiful|gorgeous|stunning|aesthetic|prettiest)\s+(?:photos?|pictures?|shots?|images?)\s+of\b/i
  ) ?? strip(
    /^\s*(?:most\s+)?(?:beautiful|gorgeous|stunning|aesthetic|prettiest)\s+(?:photos?|pictures?|shots?|images?)\s+of\b/i
  )
  if (aestheticRankingPhrase) patches.sortBy = patches.sortBy ?? 'aesthetic'

  const bestRankingPhrase = strip(
    /^\s*(?:show|find)\s+(?:me\s+)?(?:the\s+)?(?:best|top)\s+(?:photos?|pictures?|shots?|images?)\s+of\b/i
  ) ?? strip(
    /^\s*(?:best|top)\s+(?:photos?|pictures?|shots?|images?)\s+of\b/i
  )
  if (bestRankingPhrase) patches.sortBy = patches.sortBy ?? 'best'

  const scoreGte = strip(/\bscore\s*>=?\s*(\d+(?:\.\d+)?)/i)
  if (scoreGte) patches.aestheticMin = parseFloat(scoreGte[1])
  const scoreLte = strip(/\bscore\s*<=?\s*(\d+(?:\.\d+)?)/i)
  if (scoreLte) patches.aestheticMax = parseFloat(scoreLte[1])

  const sharpGte = strip(/\bsharpness\s*>=?\s*(\d+(?:\.\d+)?)/i)
  if (sharpGte) patches.sharpnessMin = parseFloat(sharpGte[1])
  const sharpLte = strip(/\bsharpness\s*<=?\s*(\d+(?:\.\d+)?)/i)
  if (sharpLte) patches.sharpnessMax = parseFloat(sharpLte[1])

  const noiseLte = strip(/\bnoise\s*<=?\s*(\d+(?:\.\d+)?)/i)
  if (noiseLte) patches.noiseMax = parseFloat(noiseLte[1])

  const isoGte = strip(/\biso\s*>=?\s*(\d+)/i)
  if (isoGte) patches.isoMin = parseInt(isoGte[1])
  const isoLte = strip(/\biso\s*<=?\s*(\d+)/i)
  if (isoLte) patches.isoMax = parseInt(isoLte[1])

  const facesGte = strip(/\bfaces?\s*>=?\s*(\d+)/i)
  if (facesGte) patches.facesMin = parseInt(facesGte[1])
  strip(/\bhas:faces?\b/i) && (patches.facesMin = patches.facesMin ?? 1)
  strip(/\bno:faces?\b/i) && (patches.facesMax = 0)
  strip(/\bhas:people\b/i) && (patches.hasPeople = true)
  strip(/\bno:people\b/i) && (patches.hasPeople = false)

  const applyPeopleCountPatch = (match: RegExpMatchArray | null, apply: (count: number) => void) => {
    if (!match) return
    const count = parseInt(match[1], 10)
    if (Number.isNaN(count)) return
    apply(count)
    patches.hasPeople = true
  }

  applyPeopleCountPatch(
    strip(/\b(?:with\s+)?(\d+)\s+or\s+more\s+people\b/i),
    (count) => { patches.facesMin = count },
  )
  applyPeopleCountPatch(
    strip(/\b(?:with\s+)?more than\s+(\d+)\s+people\b/i),
    (count) => { patches.facesMin = count + 1 },
  )
  applyPeopleCountPatch(
    strip(/\b(?:with\s+)?(?:at least|minimum of|no fewer than)\s+(\d+)\s+people\b/i),
    (count) => { patches.facesMin = count },
  )
  applyPeopleCountPatch(
    strip(/\b(?:with\s+)?less than\s+(\d+)\s+people\b/i),
    (count) => { patches.facesMax = Math.max(0, count - 1) },
  )
  applyPeopleCountPatch(
    strip(/\b(?:with\s+)?(?:at most|no more than)\s+(\d+)\s+people\b/i),
    (count) => { patches.facesMax = count },
  )
  applyPeopleCountPatch(
    strip(/\b(?:with\s+)?exactly\s+(\d+)\s+people\b/i),
    (count) => {
      patches.facesMin = count
      patches.facesMax = count
    },
  )

  if (/\b(group photo|crowd|crowded|large group|many people)\b/i.test(text)) {
    patches.hasPeople = true
  }
  text = text.replace(/\b(?:is|are)\s+in\s+the\s+(?:picture|photo)\b/gi, ' ')

  const camM = strip(/\bcamera:(\S+)/i)
  if (camM) patches.camera = camM[1].replace(/^["']|["']$/g, '')

  const lensM = strip(/\blens:(\S+)/i)
  if (lensM) patches.lens = lensM[1].replace(/^["']|["']$/g, '')

  const locM = strip(/\blocation:(\S+)/i)
  if (locM) patches.location = locM[1].replace(/^["']|["']$/g, '')

  const countryM = strip(/\bcountry:(\S+)/i)
  if (countryM) patches.country = countryM[1].replace(/^["']|["']$/g, '')

  const dayM = strip(/\bday:(\d{2}-\d{2})\b/i)
  if (dayM) patches.recurringMonthDay = dayM[1]

  const timeM = strip(/\btime:(morning|afternoon|evening|night)\b/i)
  if (timeM) patches.timeOfDay = timeM[1].toLowerCase() as SearchTimeOfDay

  const sortM = strip(/\bsort:(relevance|best|aesthetic|sharpness|cleanest|newest)\b/i)
  if (sortM) patches.sortBy = sortM[1].toLowerCase() as SearchSortBy

  const modeM = strip(/\bmode:(text|semantic|hybrid|browse)\b/i)
  if (modeM) patches.mode = modeM[1].toLowerCase() as SearchFilters['mode']

  text = text.replace(/\s+/g, ' ').trim()
  return { textQuery: text, patches }
}

function joinUniqueParts(parts: Array<string | undefined>): string | undefined {
  const seen = new Set<string>()
  const merged: string[] = []
  for (const part of parts) {
    const clean = part?.trim()
    if (!clean) continue
    const lowered = clean.toLowerCase()
    if (seen.has(lowered)) continue
    seen.add(lowered)
    merged.push(clean)
  }
  return merged.length > 0 ? merged.join(' ') : undefined
}

function toRecurringMonthDay(value: string): string | undefined {
  if (!value) return undefined
  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) return value.slice(5, 10)
  if (/^\d{2}-\d{2}$/.test(value)) return value
  return undefined
}

function formatRecurringLabel(value: string): string {
  const monthDay = toRecurringMonthDay(value)
  if (!monthDay) return value
  const [month, day] = monthDay.split('-').map((piece) => parseInt(piece, 10))
  return new Date(2000, month - 1, day).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  })
}

function expandWildlifeTerms(value: string): string[] {
  const lowered = value.toLowerCase()
  const expanded: string[] = []
  for (const [group, terms] of Object.entries(WILDLIFE_EXPANSIONS)) {
    if (!lowered.includes(group)) continue
    for (const term of terms) {
      if (!expanded.some((existing) => existing.toLowerCase() === term.toLowerCase())) {
        expanded.push(term)
      }
    }
  }
  return expanded
}

function defaultDraft(): SearchDraft {
  return {
    intent: 'general',
    prompt: '',
    activity: '',
    species: '',
    faces: [],
    faceMatch: 'all',
    country: '',
    location: '',
    recurringDate: '',
    timeOfDay: 'any',
    sortBy: 'relevance',
    mode: 'hybrid',
    semanticWeight: '0.5',
    semanticProfile: 'balanced',
    camera: '',
    lens: '',
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
    includeRelatedSpecies: true,
    expandedTerms: [],
    mustTerms: [],
    shouldTerms: [],
  }
}

function buildFilters(draft: SearchDraft): SearchFilters {
  const { textQuery, patches } = parseQuery(draft.prompt)
  const faces = dedupeStrings([
    ...(patches.faces ?? (patches.face ? [patches.face] : [])),
    ...draft.faces,
  ])
  const query = joinUniqueParts([
    textQuery,
    draft.intent === 'people' ? draft.activity : undefined,
    draft.intent === 'wildlife' ? draft.species : undefined,
  ])
  const mustTerms = dedupeStrings([...(patches.mustTerms ?? []), ...draft.mustTerms])
  const shouldTerms = dedupeStrings([...(patches.shouldTerms ?? []), ...draft.shouldTerms])

  const filters: SearchFilters = {
    ...patches,
    intent: draft.intent,
    query,
    mode: draft.mode,
    semanticWeight: parseFloat(draft.semanticWeight) || 0.5,
    semanticProfile: draft.semanticProfile,
  }

  if (faces.length > 0) {
    filters.faces = faces
    filters.face = faces[0]
    if (faces.length > 1) {
      filters.faceMatch = patches.faceMatch ?? (draft.faces.length > 1 ? draft.faceMatch : 'all')
    }
  }
  if (draft.country.trim()) filters.country = draft.country.trim()
  if (draft.location.trim()) filters.location = draft.location.trim()
  if (draft.camera.trim()) filters.camera = draft.camera.trim()
  if (draft.lens.trim()) filters.lens = draft.lens.trim()
  if (draft.dateFrom.trim()) filters.dateFrom = draft.dateFrom.trim()
  if (draft.dateTo.trim()) filters.dateTo = draft.dateTo.trim()

  const recurringMonthDay = toRecurringMonthDay(draft.recurringDate) ?? patches.recurringMonthDay
  if (recurringMonthDay) filters.recurringMonthDay = recurringMonthDay
  if (draft.timeOfDay !== 'any') filters.timeOfDay = draft.timeOfDay

  if (draft.aestheticMin !== '') filters.aestheticMin = parseFloat(draft.aestheticMin)
  if (draft.aestheticMax !== '') filters.aestheticMax = parseFloat(draft.aestheticMax)
  if (draft.sharpnessMin !== '') filters.sharpnessMin = parseFloat(draft.sharpnessMin)
  if (draft.sharpnessMax !== '') filters.sharpnessMax = parseFloat(draft.sharpnessMax)
  if (draft.noiseMax !== '') filters.noiseMax = parseFloat(draft.noiseMax)
  if (draft.isoMin !== '') filters.isoMin = parseInt(draft.isoMin)
  if (draft.isoMax !== '') filters.isoMax = parseInt(draft.isoMax)
  if (draft.facesMin !== '') filters.facesMin = parseInt(draft.facesMin)
  if (draft.facesMax !== '') filters.facesMax = parseInt(draft.facesMax)

  if (draft.hasPeople === 'yes') filters.hasPeople = true
  if (draft.hasPeople === 'no') filters.hasPeople = false
  if (draft.intent === 'people' && filters.hasPeople === undefined && (faces.length > 0 || draft.activity.trim())) {
    filters.hasPeople = true
  }

  if (draft.sortBy !== 'relevance') filters.sortBy = draft.sortBy
  if (draft.intent === 'best-shot' && !filters.sortBy) filters.sortBy = 'best'

  const expandedTerms = [
    ...draft.expandedTerms,
    ...(draft.includeRelatedSpecies ? expandWildlifeTerms(joinUniqueParts([textQuery, draft.species]) ?? '') : []),
  ]
  const seen = new Set<string>()
  const mergedExpanded = expandedTerms.filter((term) => {
    const lowered = term.toLowerCase()
    if (!term || seen.has(lowered)) return false
    seen.add(lowered)
    return true
  })
  if (mergedExpanded.length > 0) filters.expandedTerms = mergedExpanded
  if (mustTerms.length > 0) filters.mustTerms = mustTerms
  if (shouldTerms.length > 0) filters.shouldTerms = shouldTerms

  const hasMeaningfulFilter = Boolean(
    filters.query ||
    filters.faces?.length ||
    filters.country ||
    filters.location ||
    filters.camera ||
    filters.lens ||
    filters.dateFrom ||
    filters.dateTo ||
    filters.recurringMonthDay ||
    filters.timeOfDay ||
    filters.aestheticMin !== undefined ||
    filters.aestheticMax !== undefined ||
    filters.sharpnessMin !== undefined ||
    filters.sharpnessMax !== undefined ||
    filters.noiseMax !== undefined ||
    filters.isoMin !== undefined ||
    filters.isoMax !== undefined ||
    filters.facesMin !== undefined ||
    filters.facesMax !== undefined ||
    filters.hasPeople !== undefined ||
    filters.expandedTerms?.length ||
    filters.mustTerms?.length ||
    filters.shouldTerms?.length ||
    filters.sortBy
  )

  if (!hasMeaningfulFilter) {
    filters.mode = 'browse'
  }

  return filters
}

function buildContextLabel(intent: SearchIntent, filters: SearchFilters): string {
  const title = INTENT_COPY[intent].title
  const parts: string[] = []
  if (filters.faces && filters.faces.length > 0) {
    parts.push(filters.faces.join(', '))
    if (filters.faces.length > 1) {
      parts.push(filters.faceMatch === 'any' ? 'any selected person' : 'all selected people')
    }
  } else if (filters.face) {
    parts.push(filters.face)
  }
  if (filters.query) parts.push(filters.query)
  if (filters.country) parts.push(filters.country)
  if (filters.location) parts.push(filters.location)
  if (filters.recurringMonthDay) parts.push(`every ${filters.recurringMonthDay}`)
  if (filters.timeOfDay) parts.push(TIME_LABELS[filters.timeOfDay])
  if (filters.sortBy && filters.sortBy !== 'relevance') parts.push(SORT_LABELS[filters.sortBy])
  return parts.length > 0 ? `${title}: ${parts.join(' · ')}` : title
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return <label className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.18em] text-neutral-500">{children}</label>
}

function TextField({
  label,
  value,
  placeholder,
  onChange,
  type = 'text',
}: {
  label: string
  value: string
  placeholder: string
  onChange: (value: string) => void
  type?: 'text' | 'date'
}) {
  return (
    <div>
      <FieldLabel>{label}</FieldLabel>
      <input
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        className="w-full rounded-xl border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 placeholder-neutral-500 focus:border-blue-500 focus:outline-none"
      />
    </div>
  )
}

function RangeInput({
  label,
  minVal,
  maxVal,
  onMin,
  onMax,
  minPlaceholder,
  maxPlaceholder,
  step = 0.1,
  min = 0,
  max = 10,
}: {
  label: string
  minVal: string
  maxVal: string
  onMin: (value: string) => void
  onMax: (value: string) => void
  minPlaceholder: string
  maxPlaceholder: string
  step?: number
  min?: number
  max?: number
}) {
  return (
    <div>
      <FieldLabel>{label}</FieldLabel>
      <div className="grid grid-cols-2 gap-2">
        <input
          type="number"
          value={minVal}
          onChange={(event) => onMin(event.target.value)}
          placeholder={minPlaceholder}
          step={step}
          min={min}
          max={max}
          className="w-full rounded-xl border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 placeholder-neutral-500 focus:border-blue-500 focus:outline-none"
        />
        <input
          type="number"
          value={maxVal}
          onChange={(event) => onMax(event.target.value)}
          placeholder={maxPlaceholder}
          step={step}
          min={min}
          max={max}
          className="w-full rounded-xl border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 placeholder-neutral-500 focus:border-blue-500 focus:outline-none"
        />
      </div>
    </div>
  )
}

function ChoicePill({
  active,
  label,
  onClick,
}: {
  active: boolean
  label: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-full px-3 py-1.5 text-sm transition-colors ${
        active
          ? 'bg-blue-600 text-white'
          : 'bg-neutral-900 text-neutral-300 hover:bg-neutral-800'
      }`}
    >
      {label}
    </button>
  )
}

function SearchChip({ chip, onRemove }: { chip: ChipDescriptor; onRemove: (chipId: string) => void }) {
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs ${chip.tone === 'accent' ? 'border-blue-500/50 bg-blue-500/10 text-blue-100' : 'border-neutral-700 bg-neutral-900 text-neutral-200'}`}>
      {chip.label}
      <button type="button" onClick={() => onRemove(chip.id)} className="text-neutral-400 transition-colors hover:text-white">
        ×
      </button>
    </span>
  )
}

export function SearchBar({ onSearch, loading }: SearchBarProps) {
  const [draft, setDraft] = useState<SearchDraft>(defaultDraft())
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [resolvingFace, setResolvingFace] = useState(false)
  const promptRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    promptRef.current?.focus()
  }, [])

  const setDraftValue = useCallback(<K extends keyof SearchDraft>(key: K, value: SearchDraft[K]) => {
    setDraft((prev) => ({ ...prev, [key]: value }))
  }, [])

  const derivedFilters = useMemo(() => buildFilters(draft), [draft])

  const chips = useMemo<ChipDescriptor[]>(() => {
    const items: ChipDescriptor[] = []
    if (draft.prompt.trim()) items.push({ id: 'prompt', label: `Prompt: ${draft.prompt.trim()}`, tone: 'accent' })
    if (draft.activity.trim()) items.push({ id: 'activity', label: `Activity: ${draft.activity.trim()}` })
    if (draft.species.trim()) items.push({ id: 'species', label: `Species: ${draft.species.trim()}` })
    draft.faces.forEach((face, index) => {
      items.push({ id: `face:${index}`, label: `Person: ${face}` })
    })
    if (draft.faces.length > 1) {
      items.push({
        id: 'faceMatch',
        label: draft.faceMatch === 'any' ? 'Match any selected person' : 'Match all selected people',
      })
    }
    if (draft.country.trim()) items.push({ id: 'country', label: `Country: ${draft.country.trim()}` })
    if (draft.location.trim()) items.push({ id: 'location', label: `Location: ${draft.location.trim()}` })
    if (draft.recurringDate) items.push({ id: 'recurringDate', label: `Every ${formatRecurringLabel(draft.recurringDate)}` })
    if (draft.timeOfDay !== 'any') items.push({ id: 'timeOfDay', label: TIME_LABELS[draft.timeOfDay] })
    if (derivedFilters.sortBy && derivedFilters.sortBy !== 'relevance') items.push({ id: 'sortBy', label: `Sort: ${SORT_LABELS[derivedFilters.sortBy]}` })
    if (draft.mode !== 'text') {
      items.push({ id: 'semanticProfile', label: `Semantic profile: ${SEMANTIC_PROFILE_LABELS[draft.semanticProfile]}` })
    }
    if (draft.includeRelatedSpecies && derivedFilters.expandedTerms && derivedFilters.expandedTerms.length > 0) {
      const preview = derivedFilters.expandedTerms.slice(0, 4).join(', ')
      const suffix = derivedFilters.expandedTerms.length > 4 ? '…' : ''
      items.push({ id: 'expandedTerms', label: `Related species: ${preview}${suffix}` })
    }
    if (draft.camera.trim()) items.push({ id: 'camera', label: `Camera: ${draft.camera.trim()}` })
    if (draft.lens.trim()) items.push({ id: 'lens', label: `Lens: ${draft.lens.trim()}` })
    if (draft.dateFrom.trim()) items.push({ id: 'dateFrom', label: `From: ${draft.dateFrom.trim()}` })
    if (draft.dateTo.trim()) items.push({ id: 'dateTo', label: `To: ${draft.dateTo.trim()}` })
    if (draft.aestheticMin !== '' || draft.aestheticMax !== '') {
      items.push({ id: 'aestheticRange', label: `Aesthetic: ${draft.aestheticMin || '0'}–${draft.aestheticMax || '10'}` })
    }
    if (draft.sharpnessMin !== '' || draft.sharpnessMax !== '') {
      items.push({ id: 'sharpnessRange', label: `Sharpness: ${draft.sharpnessMin || '0'}–${draft.sharpnessMax || '100'}` })
    }
    if (draft.noiseMax !== '') items.push({ id: 'noiseMax', label: `Noise ≤ ${draft.noiseMax}` })
    if (draft.hasPeople !== 'any') items.push({ id: 'hasPeople', label: draft.hasPeople === 'yes' ? 'People only' : 'No people' })
    ;(derivedFilters.mustTerms ?? []).forEach((term, index) => {
      items.push({ id: `mustTerm:${index}`, label: `Must: ${term}` })
    })
    ;(derivedFilters.shouldTerms ?? []).forEach((term, index) => {
      items.push({ id: `shouldTerm:${index}`, label: `Should: ${term}` })
    })
    return items
  }, [draft, derivedFilters])

  const executeSearch = useCallback((nextDraft: SearchDraft) => {
    const filters = buildFilters(nextDraft)
    const contextLabel = buildContextLabel(nextDraft.intent, filters)
    onSearch(filters, contextLabel)
  }, [onSearch])

  const resolvePromptFace = useCallback(async (sourceDraft: SearchDraft): Promise<SearchDraft> => {
    if (sourceDraft.faces.length > 0 || !sourceDraft.prompt.trim()) {
      return sourceDraft
    }

    setResolvingFace(true)
    try {
      const resolution = await window.api.resolveSearchFaceQuery(sourceDraft.prompt)
      const resolvedFaces = resolution.faces.length > 0
        ? resolution.faces
        : resolution.face
          ? [resolution.face]
          : []
      if (resolution.error || resolvedFaces.length === 0) {
        return sourceDraft
      }

      const nextDraft: SearchDraft = {
        ...sourceDraft,
        faces: resolvedFaces,
        faceMatch: resolution.faceMatch ?? (resolvedFaces.length > 1 ? 'all' : sourceDraft.faceMatch),
        prompt: resolution.remainingQuery,
      }
      setDraft(nextDraft)
      return nextDraft
    } finally {
      setResolvingFace(false)
    }
  }, [])

  const handleSearch = useCallback(async () => {
    const nextDraft = await resolvePromptFace(draft)
    executeSearch(nextDraft)
  }, [draft, executeSearch, resolvePromptFace])

  const handlePromptKeyDown = useCallback((event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void handleSearch()
    }
  }, [handleSearch])

  const handleClear = useCallback(() => {
    setDraft(defaultDraft())
    setShowAdvanced(false)
  }, [])

  const handleRemoveChip = useCallback((chipId: string) => {
    setDraft((prev) => {
      switch (chipId) {
        case 'prompt':
          return { ...prev, prompt: '' }
        case 'activity':
          return { ...prev, activity: '' }
        case 'species':
          return { ...prev, species: '' }
        case 'faceMatch':
          return { ...prev, faceMatch: 'all' }
        case 'country':
          return { ...prev, country: '' }
        case 'location':
          return { ...prev, location: '' }
        case 'recurringDate':
          return { ...prev, recurringDate: '' }
        case 'timeOfDay':
          return { ...prev, timeOfDay: 'any' }
        case 'sortBy':
          return { ...prev, sortBy: 'relevance' }
        case 'semanticProfile':
          return { ...prev, semanticProfile: 'balanced' }
        case 'expandedTerms':
          return { ...prev, includeRelatedSpecies: false, expandedTerms: [] }
        case 'camera':
          return { ...prev, camera: '' }
        case 'lens':
          return { ...prev, lens: '' }
        case 'dateFrom':
          return { ...prev, dateFrom: '' }
        case 'dateTo':
          return { ...prev, dateTo: '' }
        case 'aestheticRange':
          return { ...prev, aestheticMin: '', aestheticMax: '' }
        case 'sharpnessRange':
          return { ...prev, sharpnessMin: '', sharpnessMax: '' }
        case 'noiseMax':
          return { ...prev, noiseMax: '' }
        case 'hasPeople':
          return { ...prev, hasPeople: 'any' }
        default:
          if (chipId.startsWith('mustTerm:')) {
            const termIndex = parseInt(chipId.split(':')[1] ?? '', 10)
            if (Number.isNaN(termIndex)) return prev
            const nextMust = prev.mustTerms.filter((_, index) => index !== termIndex)
            return { ...prev, mustTerms: nextMust }
          }
          if (chipId.startsWith('shouldTerm:')) {
            const termIndex = parseInt(chipId.split(':')[1] ?? '', 10)
            if (Number.isNaN(termIndex)) return prev
            const nextShould = prev.shouldTerms.filter((_, index) => index !== termIndex)
            return { ...prev, shouldTerms: nextShould }
          }
          if (chipId.startsWith('face:')) {
            const faceIndex = parseInt(chipId.split(':')[1] ?? '', 10)
            if (Number.isNaN(faceIndex)) return prev
            const nextFaces = prev.faces.filter((_, index) => index !== faceIndex)
            return {
              ...prev,
              faces: nextFaces,
              faceMatch: nextFaces.length > 1 ? prev.faceMatch : 'all',
            }
          }
          return prev
      }
    })
  }, [])

  return (
    <div className="flex h-full min-h-0 flex-col bg-neutral-950">
      <div className="border-b border-neutral-800 px-4 py-4">
        <div className="rounded-2xl border border-neutral-800 bg-neutral-950/70 p-4">
          <FieldLabel>Search type</FieldLabel>
          <div className="flex flex-wrap gap-2">
            {(Object.keys(INTENT_COPY) as SearchIntent[]).map((intent) => (
              <ChoicePill
                key={intent}
                active={draft.intent === intent}
                label={INTENT_COPY[intent].title}
                onClick={() => {
                  setDraft((prev) => ({
                    ...prev,
                    intent,
                    sortBy: intent === 'best-shot' ? 'best' : prev.sortBy,
                  }))
                }}
              />
            ))}
          </div>

          <div className="mt-4">
            <FieldLabel>Search</FieldLabel>
            <div className="rounded-2xl border border-neutral-800 bg-neutral-900 px-3 py-3">
              <textarea
                ref={promptRef}
                value={draft.prompt}
                onChange={(event) => setDraftValue('prompt', event.target.value)}
                onKeyDown={handlePromptKeyDown}
                placeholder={INTENT_COPY[draft.intent].placeholder}
                rows={4}
                className="min-h-28 w-full resize-y bg-transparent text-base text-neutral-100 placeholder-neutral-500 focus:outline-none"
              />
            </div>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={handleSearch}
              disabled={loading || resolvingFace}
              className="inline-flex items-center gap-2 rounded-full bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-500 disabled:opacity-60"
            >
              {loading || resolvingFace ? 'Searching…' : 'Search'}
            </button>
            <button
              type="button"
              onClick={handleClear}
              className="rounded-full border border-neutral-700 px-4 py-2 text-sm text-neutral-300 transition-colors hover:border-neutral-500 hover:text-white"
            >
              Reset
            </button>
          </div>

          {chips.length > 0 && (
            <div className="mt-4 flex flex-wrap gap-2">
              {chips.map((chip) => (
                <SearchChip key={chip.id} chip={chip} onRemove={handleRemoveChip} />
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <div className="space-y-4">
          {draft.intent === 'people' && (
            <section className="rounded-2xl border border-neutral-800 bg-neutral-950/70 p-4">
              <p className="text-sm font-medium text-white">People filters</p>
              <div className="mt-4 grid gap-4">
                <TextField
                  label="People or aliases"
                  value={draft.faces.join(', ')}
                  placeholder="Alice, cxc, Bob…"
                  onChange={(value) => {
                    const faces = splitFaceInput(value)
                    setDraft((prev) => ({
                      ...prev,
                      faces,
                      faceMatch: 'all',
                    }))
                  }}
                />
                {draft.faces.length > 1 && (
                  <div>
                    <FieldLabel>People matching</FieldLabel>
                    <div className="flex flex-wrap gap-2">
                      <ChoicePill
                        active={draft.faceMatch === 'all'}
                        label="All selected people"
                        onClick={() => setDraftValue('faceMatch', 'all')}
                      />
                      <ChoicePill
                        active={draft.faceMatch === 'any'}
                        label="Any selected person"
                        onClick={() => setDraftValue('faceMatch', 'any')}
                      />
                    </div>
                  </div>
                )}
                <TextField
                  label="Activity / scene"
                  value={draft.activity}
                  placeholder="playing basketball, walking downtown…"
                  onChange={(value) => setDraftValue('activity', value)}
                />
                <TextField
                  label="Country"
                  value={draft.country}
                  placeholder="US, Japan, France…"
                  onChange={(value) => setDraftValue('country', value)}
                />
                <div>
                  <TextField
                    label="Every year on"
                    value={draft.recurringDate}
                    type="date"
                    placeholder=""
                    onChange={(value) => setDraftValue('recurringDate', value)}
                  />
                  <p className="mt-1 text-xs text-neutral-500">The year is ignored; only the month and day are used.</p>
                </div>
                <div>
                  <FieldLabel>Time of day</FieldLabel>
                  <div className="flex flex-wrap gap-2">
                    <ChoicePill active={draft.timeOfDay === 'any'} label="Any time" onClick={() => setDraftValue('timeOfDay', 'any')} />
                    {(Object.keys(TIME_LABELS) as SearchTimeOfDay[]).map((bucket) => (
                      <ChoicePill
                        key={bucket}
                        active={draft.timeOfDay === bucket}
                        label={TIME_LABELS[bucket]}
                        onClick={() => setDraftValue('timeOfDay', bucket)}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </section>
          )}

          {draft.intent === 'wildlife' && (
            <section className="rounded-2xl border border-neutral-800 bg-neutral-950/70 p-4">
              <p className="text-sm font-medium text-white">Wildlife filters</p>
              <div className="mt-4 grid gap-4">
                <TextField
                  label="Species or group"
                  value={draft.species}
                  placeholder="duck, mallard, snowy owl…"
                  onChange={(value) => setDraftValue('species', value)}
                />
                <div className="rounded-2xl border border-neutral-800 bg-neutral-900 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-medium text-white">Related species</p>
                      <p className="text-sm text-neutral-400">Expand broad bird terms like duck, goose, owl, or hawk.</p>
                    </div>
                    <button
                      type="button"
                      onClick={() => setDraftValue('includeRelatedSpecies', !draft.includeRelatedSpecies)}
                      className={`rounded-full px-3 py-1.5 text-sm transition-colors ${draft.includeRelatedSpecies ? 'bg-blue-600 text-white' : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'}`}
                    >
                      {draft.includeRelatedSpecies ? 'On' : 'Off'}
                    </button>
                  </div>
                  {draft.includeRelatedSpecies && derivedFilters.expandedTerms && derivedFilters.expandedTerms.length > 0 && (
                    <p className="mt-3 text-sm text-neutral-300">
                      Expanding to: <span className="text-white">{derivedFilters.expandedTerms.join(', ')}</span>
                    </p>
                  )}
                </div>
              </div>
            </section>
          )}

          {draft.intent === 'best-shot' && (
            <section className="rounded-2xl border border-neutral-800 bg-neutral-950/70 p-4">
              <p className="text-sm font-medium text-white">Ranking</p>
              <div className="mt-4 flex flex-wrap gap-2">
                {(Object.keys(SORT_LABELS) as SearchSortBy[]).map((sortBy) => (
                  <ChoicePill
                    key={sortBy}
                    active={draft.sortBy === sortBy || (sortBy === 'best' && draft.sortBy === 'relevance')}
                    label={SORT_LABELS[sortBy]}
                    onClick={() => setDraftValue('sortBy', sortBy)}
                  />
                ))}
              </div>
              <p className="mt-3 text-sm text-neutral-400">
                Best overall combines aesthetic score, sharpness, and noise to surface the strongest frame after the scene matches.
              </p>
            </section>
          )}

          <section className="rounded-2xl border border-neutral-800 bg-neutral-950/70 p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-sm font-medium text-white">Advanced filters</p>
                <p className="text-sm text-neutral-400">Use these only when you need to narrow the result set.</p>
              </div>
              <button
                type="button"
                onClick={() => setShowAdvanced((prev) => !prev)}
                className="rounded-full border border-neutral-700 px-3 py-1.5 text-sm text-neutral-300 transition-colors hover:border-neutral-500 hover:text-white"
              >
                {showAdvanced ? 'Hide' : 'Show'}
              </button>
            </div>

            {showAdvanced && (
              <div className="mt-4 grid gap-4 border-t border-neutral-800 pt-4">
                <div>
                  <FieldLabel>Search mode</FieldLabel>
                  <div className="flex flex-wrap gap-2">
                    {(['hybrid', 'semantic', 'text'] as const).map((mode) => (
                      <ChoicePill
                        key={mode}
                        active={draft.mode === mode}
                        label={mode}
                        onClick={() => setDraftValue('mode', mode)}
                      />
                    ))}
                  </div>
                  {draft.mode === 'hybrid' && (
                    <div className="mt-3">
                      <FieldLabel>Semantic weight ({draft.semanticWeight})</FieldLabel>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={draft.semanticWeight}
                        onChange={(event) => setDraftValue('semanticWeight', event.target.value)}
                        className="w-full accent-blue-500"
                      />
                    </div>
                  )}
                  {draft.mode !== 'text' && (
                    <div className="mt-3 rounded-xl border border-neutral-800 bg-neutral-900 p-3">
                      <FieldLabel>Image vs description semantic balance</FieldLabel>
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="1"
                        value={SEMANTIC_PROFILE_KEY_TO_VALUE[draft.semanticProfile]}
                        onChange={(event) => {
                          const value = event.target.value as '0' | '1' | '2'
                          setDraftValue('semanticProfile', SEMANTIC_PROFILE_VALUE_TO_KEY[value] ?? 'balanced')
                        }}
                        className="w-full accent-blue-500"
                      />
                      <div className="mt-2 flex justify-between text-xs text-neutral-400">
                        <span>{SEMANTIC_PROFILE_LABELS.image_dominant}</span>
                        <span>{SEMANTIC_PROFILE_LABELS.balanced}</span>
                        <span>{SEMANTIC_PROFILE_LABELS.description_dominant}</span>
                      </div>
                    </div>
                  )}
                </div>

                <TextField label="Camera" value={draft.camera} placeholder="Sony, Canon R5…" onChange={(value) => setDraftValue('camera', value)} />
                <TextField label="Lens" value={draft.lens} placeholder="85mm, Sigma…" onChange={(value) => setDraftValue('lens', value)} />
                <TextField label="Location" value={draft.location} placeholder="Yosemite, Paris…" onChange={(value) => setDraftValue('location', value)} />
                <TextField label="Date from" value={draft.dateFrom} type="date" placeholder="" onChange={(value) => setDraftValue('dateFrom', value)} />
                <TextField label="Date to" value={draft.dateTo} type="date" placeholder="" onChange={(value) => setDraftValue('dateTo', value)} />

                <RangeInput
                  label="Aesthetic score"
                  minVal={draft.aestheticMin}
                  maxVal={draft.aestheticMax}
                  onMin={(value) => setDraftValue('aestheticMin', value)}
                  onMax={(value) => setDraftValue('aestheticMax', value)}
                  minPlaceholder="min"
                  maxPlaceholder="max"
                  step={0.5}
                  min={0}
                  max={10}
                />
                <RangeInput
                  label="Sharpness"
                  minVal={draft.sharpnessMin}
                  maxVal={draft.sharpnessMax}
                  onMin={(value) => setDraftValue('sharpnessMin', value)}
                  onMax={(value) => setDraftValue('sharpnessMax', value)}
                  minPlaceholder="min"
                  maxPlaceholder="max"
                  step={1}
                  min={0}
                  max={100}
                />
                <RangeInput
                  label="ISO range"
                  minVal={draft.isoMin}
                  maxVal={draft.isoMax}
                  onMin={(value) => setDraftValue('isoMin', value)}
                  onMax={(value) => setDraftValue('isoMax', value)}
                  minPlaceholder="min"
                  maxPlaceholder="max"
                  step={100}
                  min={50}
                  max={204800}
                />
                <RangeInput
                  label="Face count"
                  minVal={draft.facesMin}
                  maxVal={draft.facesMax}
                  onMin={(value) => setDraftValue('facesMin', value)}
                  onMax={(value) => setDraftValue('facesMax', value)}
                  minPlaceholder="min"
                  maxPlaceholder="max"
                  step={1}
                  min={0}
                  max={100}
                />

                <div>
                  <FieldLabel>Max noise level</FieldLabel>
                  <input
                    type="number"
                    value={draft.noiseMax}
                    onChange={(event) => setDraftValue('noiseMax', event.target.value)}
                    placeholder="e.g. 0.05"
                    step="0.01"
                    min="0"
                    className="w-full rounded-xl border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 placeholder-neutral-500 focus:border-blue-500 focus:outline-none"
                  />
                </div>

                <div>
                  <FieldLabel>People filter</FieldLabel>
                  <div className="flex flex-wrap gap-2">
                    {(['any', 'yes', 'no'] as const).map((value) => (
                      <ChoicePill
                        key={value}
                        active={draft.hasPeople === value}
                        label={value === 'any' ? 'Any people state' : value === 'yes' ? 'Only with people' : 'Only without people'}
                        onClick={() => setDraftValue('hasPeople', value)}
                      />
                    ))}
                  </div>
                </div>

                <TextField
                  label="Must include terms"
                  value={draft.mustTerms.join(', ')}
                  placeholder="basketball, sunset..."
                  onChange={(value) => setDraftValue('mustTerms', splitTermInput(value))}
                />
                <TextField
                  label="Should include terms"
                  value={draft.shouldTerms.join(', ')}
                  placeholder="court, hoop..."
                  onChange={(value) => setDraftValue('shouldTerms', splitTermInput(value))}
                />
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  )
}
