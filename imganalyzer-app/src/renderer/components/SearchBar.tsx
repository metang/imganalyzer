/**
 * SearchBar.tsx — compact search controls for the left search sidebar.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { SearchFilters, SearchIntent, SearchSortBy, SearchTimeOfDay } from '../global'

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
  aiModel: string
  activity: string
  species: string
  face: string
  country: string
  location: string
  recurringDate: string
  timeOfDay: SearchTimeOfDay | 'any'
  sortBy: SearchSortBy
  mode: 'text' | 'semantic' | 'hybrid'
  semanticWeight: string
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
}

interface ChipDescriptor {
  id: string
  label: string
  tone?: 'accent' | 'neutral'
}

const AI_MODELS = [
  'gpt-5.4',
  'claude-opus-4.5',
  'claude-sonnet-4.6',
  'gpt-5-mini',
] as const

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
  duck: ['mallard', 'teal', 'pintail', 'wigeon', 'gadwall', 'shoveler', 'wood duck', 'merganser'],
  goose: ['canada goose', 'snow goose', 'greylag goose', 'barnacle goose'],
  owl: ['barn owl', 'snowy owl', 'great horned owl', 'eagle owl'],
  hawk: ['red-tailed hawk', 'sparrowhawk', 'goshawk', 'kestrel'],
  eagle: ['bald eagle', 'golden eagle', 'white-tailed eagle'],
  heron: ['grey heron', 'great blue heron', 'egret', 'bittern'],
  gull: ['herring gull', 'tern', 'kittiwake', 'black-headed gull'],
}

function parseQuery(raw: string): ParsedQuery {
  const patches: Partial<SearchFilters> = {}
  let text = raw

  const strip = (re: RegExp) => {
    const match = text.match(re)
    if (match) text = text.replace(match[0], '').trim()
    return match
  }

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

  const camM = strip(/\bcamera:(\S+)/i)
  if (camM) patches.camera = camM[1].replace(/^["']|["']$/g, '')

  const lensM = strip(/\blens:(\S+)/i)
  if (lensM) patches.lens = lensM[1].replace(/^["']|["']$/g, '')

  const locM = strip(/\blocation:(\S+)/i)
  if (locM) patches.location = locM[1].replace(/^["']|["']$/g, '')

  const countryM = strip(/\bcountry:(\S+)/i)
  if (countryM) patches.country = countryM[1].replace(/^["']|["']$/g, '')

  const faceM = strip(/\bface:(\S+)/i)
  if (faceM) patches.face = faceM[1].replace(/^["']|["']$/g, '')

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

function recurringDateForInput(value: string | undefined): string {
  return value ? `2000-${value}` : ''
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
    aiModel: AI_MODELS[0],
    activity: '',
    species: '',
    face: '',
    country: '',
    location: '',
    recurringDate: '',
    timeOfDay: 'any',
    sortBy: 'relevance',
    mode: 'hybrid',
    semanticWeight: '0.5',
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
  }
}

function buildFilters(draft: SearchDraft): SearchFilters {
  const { textQuery, patches } = parseQuery(draft.prompt)
  const query = joinUniqueParts([
    textQuery,
    draft.intent === 'people' ? draft.activity : undefined,
    draft.intent === 'wildlife' ? draft.species : undefined,
  ])

  const filters: SearchFilters = {
    ...patches,
    intent: draft.intent,
    query,
    mode: draft.mode,
    semanticWeight: parseFloat(draft.semanticWeight) || 0.5,
  }

  if (draft.face.trim()) filters.face = draft.face.trim()
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
  if (draft.intent === 'people' && filters.hasPeople === undefined && (filters.face || draft.activity.trim())) {
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

  const hasMeaningfulFilter = Boolean(
    filters.query ||
    filters.face ||
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
    filters.sortBy
  )

  if (!hasMeaningfulFilter || (!filters.query && !filters.face && !filters.expandedTerms?.length)) {
    filters.mode = 'browse'
  }

  return filters
}

function buildContextLabel(intent: SearchIntent, filters: SearchFilters, plannerSummary: string | null): string {
  if (plannerSummary) return plannerSummary
  const title = INTENT_COPY[intent].title
  const parts: string[] = []
  if (filters.face) parts.push(filters.face)
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
  const [planning, setPlanning] = useState(false)
  const [resolvingFace, setResolvingFace] = useState(false)
  const [plannerSummary, setPlannerSummary] = useState<string | null>(null)
  const [plannerError, setPlannerError] = useState<string | null>(null)
  const promptRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    promptRef.current?.focus()
  }, [])

  const setDraftValue = useCallback(<K extends keyof SearchDraft>(key: K, value: SearchDraft[K]) => {
    setDraft((prev) => ({ ...prev, [key]: value }))
    setPlannerSummary(null)
    setPlannerError(null)
  }, [])

  const derivedFilters = useMemo(() => buildFilters(draft), [draft])

  const chips = useMemo<ChipDescriptor[]>(() => {
    const items: ChipDescriptor[] = []
    if (draft.prompt.trim()) items.push({ id: 'prompt', label: `Prompt: ${draft.prompt.trim()}`, tone: 'accent' })
    if (draft.activity.trim()) items.push({ id: 'activity', label: `Activity: ${draft.activity.trim()}` })
    if (draft.species.trim()) items.push({ id: 'species', label: `Species: ${draft.species.trim()}` })
    if (draft.face.trim()) items.push({ id: 'face', label: `Person: ${draft.face.trim()}` })
    if (draft.country.trim()) items.push({ id: 'country', label: `Country: ${draft.country.trim()}` })
    if (draft.location.trim()) items.push({ id: 'location', label: `Location: ${draft.location.trim()}` })
    if (draft.recurringDate) items.push({ id: 'recurringDate', label: `Every ${formatRecurringLabel(draft.recurringDate)}` })
    if (draft.timeOfDay !== 'any') items.push({ id: 'timeOfDay', label: TIME_LABELS[draft.timeOfDay] })
    if (derivedFilters.sortBy && derivedFilters.sortBy !== 'relevance') items.push({ id: 'sortBy', label: `Sort: ${SORT_LABELS[derivedFilters.sortBy]}` })
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
    return items
  }, [draft, derivedFilters])

  const executeSearch = useCallback((nextDraft: SearchDraft, summaryOverride: string | null = null) => {
    const filters = buildFilters(nextDraft)
    const contextLabel = buildContextLabel(nextDraft.intent, filters, summaryOverride)
    onSearch(filters, contextLabel)
  }, [onSearch])

  const resolvePromptFace = useCallback(async (sourceDraft: SearchDraft): Promise<SearchDraft> => {
    if (sourceDraft.face.trim() || !sourceDraft.prompt.trim()) {
      return sourceDraft
    }

    setResolvingFace(true)
    try {
      const resolution = await window.api.resolveSearchFaceQuery(sourceDraft.prompt)
      if (resolution.error || !resolution.face) {
        return sourceDraft
      }

      const nextDraft: SearchDraft = {
        ...sourceDraft,
        face: resolution.face,
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
    executeSearch(nextDraft, plannerSummary)
  }, [draft, executeSearch, plannerSummary, resolvePromptFace])

  const handlePromptKeyDown = useCallback((event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void handleSearch()
    }
  }, [handleSearch])

  const handleClear = useCallback(() => {
    setDraft(defaultDraft())
    setPlannerSummary(null)
    setPlannerError(null)
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
        case 'face':
          return { ...prev, face: '' }
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
          return prev
      }
    })
    setPlannerSummary(null)
  }, [])

  const applyPlannerResult = useCallback((response: { intent: SearchIntent; filters: SearchFilters; summary: string }) => {
    const nextDraft: SearchDraft = {
      ...draft,
      intent: response.intent,
      prompt: response.filters.query ?? draft.prompt,
      face: response.filters.face ?? draft.face,
      country: response.filters.country ?? draft.country,
      location: response.filters.location ?? draft.location,
      recurringDate: recurringDateForInput(response.filters.recurringMonthDay) || draft.recurringDate,
      timeOfDay: response.filters.timeOfDay ?? draft.timeOfDay,
      sortBy: response.filters.sortBy ?? (response.intent === 'best-shot' ? 'best' : draft.sortBy),
      mode: response.filters.mode === 'browse' ? draft.mode : (response.filters.mode ?? draft.mode),
      hasPeople: response.filters.hasPeople === undefined
        ? draft.hasPeople
        : response.filters.hasPeople ? 'yes' : 'no',
      includeRelatedSpecies: Boolean(response.filters.expandedTerms?.length) || draft.includeRelatedSpecies,
      expandedTerms: response.filters.expandedTerms ?? draft.expandedTerms,
    }
    setDraft(nextDraft)
    setPlannerSummary(response.summary)
    setPlannerError(null)
    executeSearch(nextDraft, response.summary)
  }, [draft, executeSearch])

  const handlePlanWithAI = useCallback(async () => {
    const plannerPrompt = joinUniqueParts([
      draft.prompt,
      draft.intent === 'people' ? draft.activity : undefined,
      draft.intent === 'wildlife' ? draft.species : undefined,
      draft.face ? `person ${draft.face}` : undefined,
      draft.country ? `in ${draft.country}` : undefined,
      draft.recurringDate ? `every ${formatRecurringLabel(draft.recurringDate)}` : undefined,
      draft.timeOfDay !== 'any' ? draft.timeOfDay : undefined,
      draft.intent === 'best-shot' ? 'find the best shot' : undefined,
    ])

    if (!plannerPrompt) {
      setPlannerError('Enter a prompt to interpret first.')
      return
    }

    setPlanning(true)
    setPlannerError(null)
    try {
      const response = await window.api.planSearchQuery({
        prompt: plannerPrompt,
        model: draft.aiModel,
        intent: draft.intent,
      })
      if (response.error) {
        setPlannerError(response.error)
        return
      }
      applyPlannerResult(response)
    } catch (error) {
      setPlannerError(error instanceof Error ? error.message : String(error))
    } finally {
      setPlanning(false)
    }
  }, [applyPlannerResult, draft])

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
                  setPlannerSummary(null)
                  setPlannerError(null)
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
              onClick={handlePlanWithAI}
              disabled={planning}
              className="inline-flex items-center gap-2 rounded-full border border-blue-500/40 bg-blue-500/10 px-4 py-2 text-sm font-medium text-blue-100 transition-colors hover:bg-blue-500/20 disabled:opacity-60"
            >
              {planning ? 'Interpreting…' : 'Interpret with AI'}
            </button>
            <button
              type="button"
              onClick={handleClear}
              className="rounded-full border border-neutral-700 px-4 py-2 text-sm text-neutral-300 transition-colors hover:border-neutral-500 hover:text-white"
            >
              Reset
            </button>
          </div>

          {(plannerSummary || plannerError) && (
            <div className={`mt-4 rounded-2xl border px-4 py-3 text-sm ${plannerError ? 'border-red-900/60 bg-red-950/40 text-red-200' : 'border-blue-900/40 bg-blue-950/30 text-blue-100'}`}>
              {plannerError ?? plannerSummary}
            </div>
          )}

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
                  label="Person or alias"
                  value={draft.face}
                  placeholder="Alice, cxc, Bob…"
                  onChange={(value) => setDraftValue('face', value)}
                />
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
                </div>

                <div>
                  <FieldLabel>AI model</FieldLabel>
                  <select
                    value={draft.aiModel}
                    onChange={(event) => setDraftValue('aiModel', event.target.value as SearchDraft['aiModel'])}
                    className="w-full rounded-xl border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 focus:border-blue-500 focus:outline-none"
                  >
                    {AI_MODELS.map((model) => (
                      <option key={model} value={model}>{model}</option>
                    ))}
                  </select>
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
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  )
}
