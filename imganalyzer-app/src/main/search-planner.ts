import { spawn, execSync } from 'child_process'
import { existsSync } from 'fs'
import { dirname, join } from 'path'
import type {
  SearchFaceMatch,
  SearchFilters,
  SearchIntent,
  SearchPlanRequest,
  SearchPlanResponse,
  SearchRankPreference,
  SearchSemanticProfile,
  SearchSortBy,
  SearchTimeOfDay,
} from './search'

const DEFAULT_SEARCH_MODEL = 'gpt-5.4'
const VALID_INTENTS = new Set<SearchIntent>(['people', 'wildlife', 'best-shot', 'general'])
const VALID_TIME_OF_DAY = new Set<SearchTimeOfDay>(['morning', 'afternoon', 'evening', 'night'])
const VALID_SORT_BY = new Set<SearchSortBy>([
  'relevance',
  'best',
  'aesthetic',
  'sharpness',
  'cleanest',
  'newest',
])
const VALID_RANK_PREFERENCE = new Set<SearchRankPreference>([
  'relevance',
  'quality',
  'recency',
  'aesthetic',
  'cleanest',
  'sharpest',
])
const VALID_MODES = new Set<NonNullable<SearchFilters['mode']>>(['text', 'semantic', 'hybrid', 'browse'])
const VALID_SEMANTIC_PROFILE = new Set<SearchSemanticProfile>([
  'image_dominant',
  'balanced',
  'description_dominant',
])

function findSystemNode(): string {
  try {
    const result = execSync('where node', { encoding: 'utf8' }).trim()
    const first = result.split('\n')[0].trim()
    if (first && existsSync(first)) return first
  } catch {
    // ignore
  }
  return process.execPath
}

const SYSTEM_NODE = findSystemNode()

const SEARCH_PLANNER_PROMPT = `You are a search planner for a photo library application.
Convert the user's natural-language request into a JSON object with this exact shape:
{
  "intent": "people" | "wildlife" | "best-shot" | "general",
    "summary": "short human-readable summary",
    "filters": {
      "query": string | null,
      "face": string | null,
      "faces": string[] | null,
      "faceMatch": "any" | "all" | null,
      "location": string | null,
      "country": string | null,
      "dateFrom": string | null,
      "dateTo": string | null,
      "recurringMonthDay": "MM-DD" | null,
      "timeOfDay": "morning" | "afternoon" | "evening" | "night" | null,
      "mode": "text" | "semantic" | "hybrid" | "browse" | null,
      "semanticProfile": "image_dominant" | "balanced" | "description_dominant" | null,
      "sortBy": "relevance" | "best" | "aesthetic" | "sharpness" | "cleanest" | "newest" | null,
      "rankPreference": "relevance" | "quality" | "recency" | "aesthetic" | "cleanest" | "sharpest" | null,
      "expandedTerms": string[] | null,
      "mustTerms": string[] | null,
      "shouldTerms": string[] | null,
      "debugSearch": boolean | null,
      "facetRequest": boolean | null,
      "hasPeople": boolean | null,
      "facesMin": number | null,
      "facesMax": number | null
    }
}

Rules:
- Only use supported fields; never invent new keys.
- "Best photo", "best shot", or "best picture" should usually set intent="best-shot" and sortBy="best".
- Named people, display names, and aliases should go into faces when the request clearly identifies one or more people. Use face only for backward-compatible single-person output if needed.
- When the request clearly wants multiple people in the same image (for example "A and B together"), use faceMatch="all".
- Requests like "more than 10 people" should use facesMin / facesMax and hasPeople instead of leaving the count in query text.
- Geographic constraints like "in the US" should use country when possible; broader place details can go into location.
- Requests like "every Feb 1" should use recurringMonthDay="02-01".
- Time buckets should be one of morning / afternoon / evening / night.
- Wildlife group terms can populate expandedTerms with likely species or sub-species (keep the list concise and useful).
- Keep query focused on the visual/activity concept that should still be matched textually or semantically.
- Return JSON only, with no markdown fences and no extra prose.`

function parseJsonResponse(text: string): Record<string, unknown> {
  let cleaned = text.trim()
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.split('\n').slice(1).join('\n')
    cleaned = cleaned.split('```')[0]
  }
  return JSON.parse(cleaned) as Record<string, unknown>
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function asInteger(value: unknown): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined
  return Math.trunc(value)
}

function asIntent(value: unknown, fallback: SearchIntent): SearchIntent {
  return typeof value === 'string' && VALID_INTENTS.has(value as SearchIntent)
    ? value as SearchIntent
    : fallback
}

function buildFallbackSummary(intent: SearchIntent, filters: SearchFilters, prompt: string): string {
  const parts: string[] = []
  const base = filters.query ?? prompt.trim()
  if (base) parts.push(base)
  const faces = filters.faces && filters.faces.length > 0
    ? filters.faces
    : filters.face
      ? [filters.face]
      : []
  if (faces.length > 0) parts.push(`people: ${faces.join(', ')}`)
  if (filters.country) parts.push(filters.country)
  if (filters.recurringMonthDay) parts.push(`every ${filters.recurringMonthDay}`)
  if (filters.timeOfDay) parts.push(filters.timeOfDay)
  if (filters.sortBy && filters.sortBy !== 'relevance') parts.push(`sort: ${filters.sortBy}`)
  const prefix = intent === 'general' ? 'Search' : intent === 'best-shot' ? 'Best Shot' : intent[0].toUpperCase() + intent.slice(1)
  return parts.length > 0 ? `${prefix}: ${parts.join(' · ')}` : prefix
}

function sanitizeFilters(rawFilters: unknown): SearchFilters {
  const source = typeof rawFilters === 'object' && rawFilters !== null
    ? rawFilters as Record<string, unknown>
    : {}
  const filters: SearchFilters = {}

  const query = asString(source.query)
  if (query) filters.query = query

  const face = asString(source.face)
  if (face) filters.face = face

  if (Array.isArray(source.faces)) {
    const seen = new Set<string>()
    const faces = source.faces
      .filter((value): value is string => typeof value === 'string')
      .map((value) => value.trim())
      .filter((value) => {
        const lowered = value.toLowerCase()
        if (!value || seen.has(lowered)) return false
        seen.add(lowered)
        return true
      })
    if (faces.length > 0) {
      filters.faces = faces
      if (!filters.face) {
        filters.face = faces[0]
      }
    }
  } else if (face) {
    filters.faces = [face]
  }

  const faceMatch = source.faceMatch
  if (faceMatch === 'any' || faceMatch === 'all') {
    filters.faceMatch = faceMatch as SearchFaceMatch
  }

  const location = asString(source.location)
  if (location) filters.location = location

  const country = asString(source.country)
  if (country) filters.country = country

  const dateFrom = asString(source.dateFrom)
  if (dateFrom) filters.dateFrom = dateFrom

  const dateTo = asString(source.dateTo)
  if (dateTo) filters.dateTo = dateTo

  const recurringMonthDay = asString(source.recurringMonthDay)
  if (recurringMonthDay && /^\d{2}-\d{2}$/.test(recurringMonthDay)) {
    filters.recurringMonthDay = recurringMonthDay
  }

  const timeOfDay = source.timeOfDay
  if (typeof timeOfDay === 'string' && VALID_TIME_OF_DAY.has(timeOfDay as SearchTimeOfDay)) {
    filters.timeOfDay = timeOfDay as SearchTimeOfDay
  }

  const mode = source.mode
  if (typeof mode === 'string' && VALID_MODES.has(mode as NonNullable<SearchFilters['mode']>)) {
    filters.mode = mode as NonNullable<SearchFilters['mode']>
  }
  const semanticProfile = source.semanticProfile
  if (typeof semanticProfile === 'string' && VALID_SEMANTIC_PROFILE.has(semanticProfile as SearchSemanticProfile)) {
    filters.semanticProfile = semanticProfile as SearchSemanticProfile
  }

  const sortBy = source.sortBy
  if (typeof sortBy === 'string' && VALID_SORT_BY.has(sortBy as SearchSortBy)) {
    filters.sortBy = sortBy as SearchSortBy
  }
  const rankPreference = source.rankPreference
  if (typeof rankPreference === 'string' && VALID_RANK_PREFERENCE.has(rankPreference as SearchRankPreference)) {
    filters.rankPreference = rankPreference as SearchRankPreference
  }

  if (typeof source.debugSearch === 'boolean') {
    filters.debugSearch = source.debugSearch
  }
  if (typeof source.facetRequest === 'boolean') {
    filters.facetRequest = source.facetRequest
  }

  const toTerms = (value: unknown): string[] | undefined => {
    if (!Array.isArray(value)) return undefined
    const seen = new Set<string>()
    const terms = value
      .filter((entry): entry is string => typeof entry === 'string')
      .map((entry) => entry.trim())
      .filter((entry) => {
        const lowered = entry.toLowerCase()
        if (!entry || seen.has(lowered)) return false
        seen.add(lowered)
        return true
      })
    return terms.length > 0 ? terms : undefined
  }
  const mustTerms = toTerms(source.mustTerms)
  if (mustTerms) filters.mustTerms = mustTerms
  const shouldTerms = toTerms(source.shouldTerms)
  if (shouldTerms) filters.shouldTerms = shouldTerms

  if (typeof source.hasPeople === 'boolean') {
    filters.hasPeople = source.hasPeople
  }

  const facesMin = asInteger(source.facesMin)
  if (facesMin !== undefined && facesMin >= 0) {
    filters.facesMin = facesMin
  }

  const facesMax = asInteger(source.facesMax)
  if (facesMax !== undefined && facesMax >= 0) {
    filters.facesMax = facesMax
  }

  if (Array.isArray(source.expandedTerms)) {
    const seen = new Set<string>()
    const expandedTerms = source.expandedTerms
      .filter((term): term is string => typeof term === 'string')
      .map((term) => term.trim())
      .filter((term) => {
        const lowered = term.toLowerCase()
        if (!term || seen.has(lowered)) return false
        seen.add(lowered)
        return true
      })
    if (expandedTerms.length > 0) {
      filters.expandedTerms = expandedTerms
    }
  }

  return filters
}

export async function planSearchWithCopilot(request: SearchPlanRequest): Promise<SearchPlanResponse> {
  const prompt = request.prompt.trim()
  const requestedIntent = request.intent ?? 'general'
  const model = request.model?.trim() || DEFAULT_SEARCH_MODEL

  if (!prompt) {
    return {
      intent: requestedIntent,
      filters: {},
      summary: '',
      model,
      error: 'Enter a prompt to interpret.',
    }
  }

  // eslint-disable-next-line @typescript-eslint/consistent-type-imports
  type SDK = typeof import('@github/copilot-sdk')
  const { CopilotClient } = await import('@github/copilot-sdk') as SDK

  let resolvedCliPath: string | undefined
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const sdkPath: string = require.resolve('@github/copilot/sdk')
    resolvedCliPath = join(dirname(dirname(sdkPath)), 'index.js')
  } catch {
    // fall back to SDK resolution
  }

  let client: InstanceType<SDK['CopilotClient']> | null = null

  try {
    client = resolvedCliPath
      ? new CopilotClient({ cliPath: resolvedCliPath })
      : new CopilotClient()

    const anyClient = client as Record<string, unknown>
    const origStartCLI = anyClient['startCLIServer'] as () => Promise<void>
    anyClient['startCLIServer'] = async function patchedStartCLI(this: Record<string, unknown>) {
      const opts = this['options'] as {
        cliPath: string
        cliArgs: string[]
        cwd: string
        port: number
        useStdio: boolean
        logLevel: string
        githubToken?: string
        useLoggedInUser?: boolean
        env?: Record<string, string>
      }

      if (!opts.cliPath.endsWith('.js')) {
        return origStartCLI.call(this)
      }

      return new Promise<void>((resolve, reject) => {
        const args = [
          opts.cliPath,
          ...opts.cliArgs,
          '--headless',
          '--no-auto-update',
          '--log-level',
          opts.logLevel,
        ]
        if (opts.useStdio) args.push('--stdio')
        else if (opts.port > 0) args.push('--port', String(opts.port))
        if (opts.githubToken) args.push('--auth-token-env', 'COPILOT_SDK_AUTH_TOKEN')
        if (!opts.useLoggedInUser) args.push('--no-auto-login')

        const env = { ...process.env, ...(opts.env ?? {}) }
        delete env.NODE_DEBUG
        if (opts.githubToken) env.COPILOT_SDK_AUTH_TOKEN = opts.githubToken

        const stdioConfig = opts.useStdio
          ? (['pipe', 'pipe', 'pipe'] as const)
          : (['ignore', 'pipe', 'pipe'] as const)

        const proc = spawn(SYSTEM_NODE, args, {
          stdio: stdioConfig,
          cwd: opts.cwd,
          env,
          windowsHide: true,
        })

        this['cliProcess'] = proc

        if (opts.useStdio) {
          resolve()
          return
        }

        let stdout = ''
        let resolved = false
        proc.stdout?.on('data', (chunk: Buffer) => {
          stdout += chunk.toString()
          const match = stdout.match(/Listening on port (\d+)/)
          if (match && !resolved) {
            resolved = true
            this['actualPort'] = parseInt(match[1], 10)
            resolve()
          }
        })
        proc.stderr?.on('data', (chunk: Buffer) => {
          this['stderrBuffer'] = String(this['stderrBuffer'] ?? '') + chunk.toString()
        })
        proc.on('error', reject)
        proc.on('exit', (code) => {
          if (!resolved) reject(new Error(`CLI exited with code ${code}`))
        })
      })
    }

    await client.start()
    const session = await client.createSession({ model })

    let responseText = ''
    const result = await new Promise<SearchPlanResponse>((resolve) => {
      let settled = false

      const finish = (value: SearchPlanResponse) => {
        if (settled) return
        settled = true
        resolve(value)
      }

      session.on('assistant.message', (event) => {
        const content = (event as { data?: { content?: string } }).data?.content
        if (typeof content === 'string') responseText += content
      })

      session.on('session.error', (event) => {
        const message = (event as { data?: { message?: string } }).data?.message
        finish({
          intent: requestedIntent,
          filters: {},
          summary: '',
          model,
          error: message ? String(message) : 'Unknown search planner error',
        })
      })

      session.on('session.idle', () => {
        if (!responseText) {
          finish({
            intent: requestedIntent,
            filters: {},
            summary: '',
            model,
            error: 'Search planner returned an empty response',
          })
          return
        }

        try {
          const parsed = parseJsonResponse(responseText)
          const intent = asIntent(parsed.intent, requestedIntent)
          const filters = sanitizeFilters(parsed.filters)
          const summary = asString(parsed.summary) ?? buildFallbackSummary(intent, filters, prompt)
          finish({ intent, filters, summary, model })
        } catch (error) {
          finish({
            intent: requestedIntent,
            filters: {},
            summary: '',
            model,
            error: `Failed to parse search planner response: ${String(error)}`,
          })
        }
      })

      session.send({
        prompt: `${SEARCH_PLANNER_PROMPT}\n\nRequested intent: ${requestedIntent}\n\nUser request:\n${prompt}`,
      }).catch((error: unknown) => {
        finish({
          intent: requestedIntent,
          filters: {},
          summary: '',
          model,
          error: String(error),
        })
      })
    })

    await session.destroy().catch(() => {})
    return result
  } catch (error: unknown) {
    return {
      intent: requestedIntent,
      filters: {},
      summary: '',
      model,
      error: String(error),
    }
  } finally {
    if (client) {
      await (client as { stop(): Promise<void> }).stop().catch(() => {})
    }
  }
}
