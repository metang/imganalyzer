/**
 * search.ts — IPC handler for the search-json CLI command.
 *
 * Delegates to `imganalyzer search-json` via conda subprocess and parses
 * the JSON response into typed SearchResult records for the renderer.
 */

import { ipcMain } from 'electron'
import { spawn } from 'child_process'

const PKG_ROOT = process.env.IMGANALYZER_PKG_ROOT || 'D:\\Code\\imganalyzer'
const CONDA_ENV = 'imganalyzer'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SearchFilters {
  query?: string
  mode?: 'text' | 'semantic' | 'hybrid' | 'browse'
  semanticWeight?: number
  // text filters
  face?: string
  camera?: string
  lens?: string
  location?: string
  // numeric ranges
  aestheticMin?: number
  aestheticMax?: number
  sharpnessMin?: number
  sharpnessMax?: number
  noiseMax?: number
  isoMin?: number
  isoMax?: number
  facesMin?: number
  facesMax?: number
  dateFrom?: string
  dateTo?: string
  hasPeople?: boolean
  // pagination
  limit?: number
  offset?: number
}

export interface SearchResult {
  image_id: number
  file_path: string
  score: number | null
  width: number | null
  height: number | null
  file_size: number | null
  // metadata
  camera_make: string | null
  camera_model: string | null
  lens_model: string | null
  focal_length: string | null
  f_number: string | null
  exposure_time: string | null
  iso: string | null
  date_time_original: string | null
  gps_latitude: string | null
  gps_longitude: string | null
  location_city: string | null
  location_state: string | null
  location_country: string | null
  // technical
  sharpness_score: number | null
  sharpness_label: string | null
  exposure_ev: number | null
  exposure_label: string | null
  noise_level: number | null
  noise_label: string | null
  snr_db: number | null
  dynamic_range_stops: number | null
  highlight_clipping_pct: number | null
  shadow_clipping_pct: number | null
  avg_saturation: number | null
  dominant_colors: string[] | null
  // local AI
  description: string | null
  scene_type: string | null
  main_subject: string | null
  lighting: string | null
  mood: string | null
  keywords: string[] | null
  detected_objects: string[] | null
  face_count: number | null
  face_identities: string[] | null
  has_people: boolean | null
  ocr_text: string | null
  // aesthetic
  aesthetic_score: number | null
  aesthetic_label: string | null
  aesthetic_reason: string | null
}

export interface SearchResponse {
  results: SearchResult[]
  total: number
  error?: string
}

// ── CLI execution ─────────────────────────────────────────────────────────────

function buildArgs(filters: SearchFilters): string[] {
  const args: string[] = ['search-json']

  // Query (positional argument)
  args.push(filters.query?.trim() || '')

  if (filters.mode)           args.push('--mode', filters.mode)
  if (filters.semanticWeight !== undefined) args.push('--semantic-weight', String(filters.semanticWeight))
  if (filters.face)           args.push('--face', filters.face)
  if (filters.camera)         args.push('--camera', filters.camera)
  if (filters.lens)           args.push('--lens', filters.lens)
  if (filters.location)       args.push('--location', filters.location)
  if (filters.aestheticMin !== undefined) args.push('--aesthetic-min', String(filters.aestheticMin))
  if (filters.aestheticMax !== undefined) args.push('--aesthetic-max', String(filters.aestheticMax))
  if (filters.sharpnessMin !== undefined) args.push('--sharpness-min', String(filters.sharpnessMin))
  if (filters.sharpnessMax !== undefined) args.push('--sharpness-max', String(filters.sharpnessMax))
  if (filters.noiseMax !== undefined)     args.push('--noise-max', String(filters.noiseMax))
  if (filters.isoMin !== undefined)       args.push('--iso-min', String(filters.isoMin))
  if (filters.isoMax !== undefined)       args.push('--iso-max', String(filters.isoMax))
  if (filters.facesMin !== undefined)     args.push('--faces-min', String(filters.facesMin))
  if (filters.facesMax !== undefined)     args.push('--faces-max', String(filters.facesMax))
  if (filters.dateFrom)       args.push('--date-from', filters.dateFrom)
  if (filters.dateTo)         args.push('--date-to', filters.dateTo)
  if (filters.hasPeople === true)  args.push('--has-people')
  if (filters.hasPeople === false) args.push('--no-people')
  if (filters.limit !== undefined) args.push('--limit', String(filters.limit))
  if (filters.offset !== undefined) args.push('--offset', String(filters.offset))

  return args
}

function runSearchCli(filters: SearchFilters): Promise<SearchResponse> {
  return new Promise((resolve) => {
    const args = buildArgs(filters)

    const proc = spawn(
      'conda',
      [
        'run', '-n', CONDA_ENV, '--no-capture-output',
        'python', '-m', 'imganalyzer.cli',
        ...args,
      ],
      {
        cwd: PKG_ROOT,
        env: {
          ...process.env,
          HF_HUB_DISABLE_SYMLINKS_WARNING: '1',
          PYTHONIOENCODING: 'utf-8',
          PYTHONUTF8: '1',
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      }
    )

    let stdout = ''
    let stderr = ''
    proc.stdout?.on('data', (chunk: Buffer) => { stdout += chunk.toString('utf8') })
    proc.stderr?.on('data', (chunk: Buffer) => { stderr += chunk.toString('utf8') })

    const timer = setTimeout(() => {
      try { proc.kill() } catch { /* ignore */ }
      resolve({ results: [], total: 0, error: 'Search timed out after 60 s' })
    }, 60_000)

    proc.on('close', (code) => {
      clearTimeout(timer)
      // Find the JSON line in stdout
      const jsonLine = stdout.split('\n').find((l) => l.trim().startsWith('{'))
      if (!jsonLine) {
        const msg = code !== 0
          ? `CLI exited with code ${code}: ${stderr.slice(0, 500)}`
          : 'No JSON output from search command'
        resolve({ results: [], total: 0, error: msg })
        return
      }
      try {
        const data = JSON.parse(jsonLine) as { results: SearchResult[]; total: number }
        resolve({ results: data.results, total: data.total })
      } catch (e) {
        resolve({ results: [], total: 0, error: `JSON parse error: ${String(e)}` })
      }
    })

    proc.on('error', (err) => {
      clearTimeout(timer)
      resolve({ results: [], total: 0, error: String(err) })
    })
  })
}

// ── IPC Registration ──────────────────────────────────────────────────────────

export function registerSearchHandlers(): void {
  ipcMain.handle('search:run', async (_evt, filters: SearchFilters): Promise<SearchResponse> => {
    return runSearchCli(filters)
  })
}
