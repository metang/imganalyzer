/**
 * Cloud AI analysis via the GitHub Copilot SDK (model: gpt-4.1).
 *
 * @github/copilot-sdk is a pure ESM package that uses import.meta.resolve()
 * internally to locate the bundled Copilot CLI. It cannot be statically
 * imported into a CJS Electron main bundle — doing so causes Vite to inline
 * and mangle import.meta.resolve into (void 0), breaking CLI discovery.
 *
 * Solution: use a dynamic import() at call-time. Electron's Node runtime
 * supports top-level await-free dynamic import() in CJS modules, and the SDK
 * is marked as external in electron.vite.config.ts so it is never bundled.
 *
 * ELECTRON NODE VERSION MISMATCH:
 * The @github/copilot CLI requires node:sqlite (Node ≥22.5). Electron 31
 * bundles Node 20, so spawning the CLI via process.execPath (the SDK default)
 * fails. Fix: we monkey-patch startCLIServer on the CopilotClient instance to
 * spawn the CLI using the system node executable instead.
 */

import { spawn, execSync, execFileSync } from 'child_process'
import { app } from 'electron'
import { existsSync, unlinkSync, writeFileSync } from 'fs'
import { dirname, join, extname } from 'path'
import { tmpdir } from 'os'
import type { XmpData } from './xmp'

// Resolve the system node binary (not Electron's embedded node).
// Falls back to process.execPath if not found, which may fail on old Electron.
function findSystemNode(): string {
  try {
    const result = execSync('where node', { encoding: 'utf8' }).trim()
    // 'where' may return multiple lines — take the first real node.exe
    const first = result.split('\n')[0].trim()
    if (first && existsSync(first)) return first
  } catch {
    // ignore
  }
  return process.execPath
}

const SYSTEM_NODE = findSystemNode()

// ─── RAW preprocessing ────────────────────────────────────────────────────────

// Extensions that cannot be sent directly to the Copilot SDK vision model.
const RAW_EXTENSIONS = new Set([
  '.arw', '.cr2', '.cr3', '.crw', '.dng', '.nef', '.nrw', '.orf', '.pef',
  '.raf', '.raw', '.rw2', '.rwl', '.sr2', '.srf', '.srw', '.x3f', '.iiq',
  '.3fr', '.fff', '.mef', '.mos', '.mrw', '.rwz', '.erf',
])

// Python script: decode RAW with rawpy, resize to ≤1568px, write temp JPEG.
// Mirrors the approach used in cloud.py (_encode_image) and images.ts thumbnails.
const RAW_CONVERT_SCRIPT = `
import sys, rawpy, io
from PIL import Image
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

with rawpy.imread(str(src)) as raw:
    rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=True)

img = Image.fromarray(rgb)
w, h = img.size
max_dim = 1568
if max(w, h) > max_dim:
    ratio = max_dim / max(w, h)
    img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

img.save(str(dst), format='JPEG', quality=85)
`

/**
 * Convert a RAW file to a temporary JPEG using the conda imganalyzer environment.
 * Returns the temp JPEG path (caller must delete it when done).
 */
function convertRawToJpeg(rawPath: string): string {
  const scriptPath = join(tmpdir(), 'copilot_raw_convert.py')
  const jpegPath   = join(tmpdir(), `copilot_raw_${Date.now()}.jpg`)

  writeFileSync(scriptPath, RAW_CONVERT_SCRIPT, 'utf-8')

  const pkgRoot = process.env.IMGANALYZER_PKG_ROOT || dirname(app.getAppPath())

  execFileSync(
    'conda',
    ['run', '-n', 'imganalyzer', '--no-capture-output', 'python', scriptPath, rawPath, jpegPath],
    { cwd: pkgRoot, timeout: 60_000 }
  )

  if (!existsSync(jpegPath)) {
    throw new Error(`RAW conversion produced no output for: ${rawPath}`)
  }
  return jpegPath
}



const SYSTEM_PROMPT = `You are an expert photography analyst. Analyze the provided image and return a JSON object with exactly these fields:
- description: (string) A detailed 2-3 sentence description of the image content
- scene_type: (string) e.g. "landscape", "portrait", "street", "architecture", "macro", "wildlife", "abstract", "product"
- main_subject: (string) Primary subject(s) in the image
- lighting: (string) Lighting conditions e.g. "golden hour", "overcast", "harsh midday", "studio softbox"
- mood: (string) Emotional tone e.g. "serene", "dramatic", "intimate", "moody"
- keywords: (array of strings) 10-15 descriptive keywords suitable as photo tags
- technical_notes: (string) Notable photographic or technical observations
- aesthetic_score: (number) Overall aesthetic quality score from 0.0 to 10.0. Consider composition, lighting, subject interest, technical quality, and emotional impact. Be critical and realistic: 0-3 = poor, 4-5 = average, 6-7 = good, 8-9 = excellent, 10 = exceptional/masterpiece
- aesthetic_label: (string) One-word label matching the score: "Poor" (0-3), "Average" (4-5), "Good" (6-7), "Excellent" (8-9), "Masterpiece" (10)

Return ONLY valid JSON with no extra text, no markdown fences.`

// ─── JSON parser (tolerant of markdown fences) ────────────────────────────

function parseJsonResponse(text: string): Record<string, unknown> {
  let t = text.trim()
  if (t.startsWith('```')) {
    t = t.split('\n').slice(1).join('\n')
    t = t.split('```')[0]
  }
  try {
    return JSON.parse(t) as Record<string, unknown>
  } catch {
    return { description: text, keywords: [] }
  }
}

// ─── Field mapper ─────────────────────────────────────────────────────────

function mapToXmpData(raw: Record<string, unknown>): XmpData {
  const xmp: XmpData = {}

  const str = (key: string): string | undefined => {
    const v = raw[key]
    return typeof v === 'string' && v ? v : undefined
  }

  const strArr = (key: string): string[] => {
    const v = raw[key]
    if (Array.isArray(v)) return v.filter((x): x is string => typeof x === 'string')
    return []
  }

  const desc = str('description')
  const notes = str('technical_notes')
  if (desc && notes) {
    xmp.description = `${desc}\n\n${notes}`
  } else if (desc) {
    xmp.description = desc
  } else if (notes) {
    xmp.description = notes
  }

  xmp.sceneType   = str('scene_type')
  xmp.mainSubject = str('main_subject')
  xmp.lighting    = str('lighting')
  xmp.mood        = str('mood')

  const kws = strArr('keywords')
  if (kws.length > 0) xmp.keywords = kws

  const aScore = raw['aesthetic_score']
  if (typeof aScore === 'number' && isFinite(aScore)) {
    xmp.aestheticScore = Math.max(0, Math.min(10, aScore))
  }
  xmp.aestheticLabel = str('aesthetic_label')

  return xmp
}

// ─── Main export ──────────────────────────────────────────────────────────

export async function runCopilotAnalysis(
  imagePath: string
): Promise<{ xmp: XmpData | null; error?: string }> {
  // Preprocess RAW files: convert to a temp JPEG before sending to cloud AI.
  let analysisPath = imagePath
  let tempJpeg: string | null = null

  if (RAW_EXTENSIONS.has(extname(imagePath).toLowerCase())) {
    try {
      tempJpeg = convertRawToJpeg(imagePath)
      analysisPath = tempJpeg
    } catch (err) {
      return { xmp: null, error: `RAW conversion failed: ${String(err)}` }
    }
  }

  // Dynamic import keeps the ESM package out of the CJS bundle entirely.
  // eslint-disable-next-line @typescript-eslint/consistent-type-imports
  type SDK = typeof import('@github/copilot-sdk')
  const { CopilotClient } = await import('@github/copilot-sdk') as SDK

  // Resolve CLI path without import.meta.resolve (unavailable in CJS Electron).
  // @github/copilot/sdk is the entry point exported by the CLI package; the
  // CLI's main index.js lives two directories up from it.
  let resolvedCliPath: string | undefined
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const sdkPath: string = require.resolve('@github/copilot/sdk')
    resolvedCliPath = join(dirname(dirname(sdkPath)), 'index.js')
  } catch {
    // fall back: let SDK call getBundledCliPath() via import.meta.resolve
  }

  let client: InstanceType<SDK['CopilotClient']> | null = null

  try {
    client = resolvedCliPath
      ? new CopilotClient({ cliPath: resolvedCliPath })
      : new CopilotClient()

    // Patch startCLIServer to use the system node binary instead of
    // process.execPath (which points to the Electron binary on Electron).
    // The SDK's default spawn is: spawn(process.execPath, [cliPath, ...args])
    // We replace it with:         spawn(SYSTEM_NODE, [cliPath, ...args])
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

      // If cliPath is not a .js file, fall through to original
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
        delete env['NODE_DEBUG']
        if (opts.githubToken) env['COPILOT_SDK_AUTH_TOKEN'] = opts.githubToken

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

        // TCP mode: wait for "Listening on port NNNN"
        let stdout = ''
        let resolved = false
        proc.stdout?.on('data', (chunk: Buffer) => {
          stdout += chunk.toString()
          const m = stdout.match(/Listening on port (\d+)/)
          if (m && !resolved) {
            resolved = true
            ; (this as Record<string, unknown>)['actualPort'] = parseInt(m[1], 10)
            resolve()
          }
        })
        proc.stderr?.on('data', (chunk: Buffer) => {
          ; (this as Record<string, unknown>)['stderrBuffer'] =
            ((this as Record<string, unknown>)['stderrBuffer'] as string) + chunk.toString()
        })
        proc.on('error', reject)
        proc.on('exit', (code) => {
          if (!resolved) reject(new Error(`CLI exited with code ${code}`))
        })
      })
    }

    await client.start()

    const session = await client.createSession({ model: 'gpt-4.1' })

    let responseText = ''

    const result = await new Promise<{ xmp: XmpData | null; error?: string }>(
      (resolve) => {
        let settled = false

        const finish = (val: { xmp: XmpData | null; error?: string }) => {
          if (settled) return
          settled = true
          resolve(val)
        }

        session.on('assistant.message', (event) => {
          const content = (event as { data?: { content?: string } }).data?.content
          if (typeof content === 'string') responseText += content
        })

        session.on('session.error', (event) => {
          const msg = (event as { data?: { message?: string } }).data?.message
          finish({ xmp: null, error: msg ? String(msg) : 'Unknown session error' })
        })

        session.on('session.idle', () => {
          if (!responseText) {
            finish({ xmp: null, error: 'Cloud AI returned an empty response' })
            return
          }
          try {
            const raw = parseJsonResponse(responseText)
            finish({ xmp: mapToXmpData(raw) })
          } catch (e) {
            finish({ xmp: null, error: `Failed to parse response: ${String(e)}` })
          }
        })

        session
          .send({
            prompt: 'Analyze this image.\n\n' + SYSTEM_PROMPT,
            attachments: [{ type: 'file', path: analysisPath }],
          })
          .catch((err: unknown) => {
            finish({ xmp: null, error: String(err) })
          })
      }
    )

    await session.destroy().catch(() => {})
    return result
  } catch (err: unknown) {
    return { xmp: null, error: String(err) }
  } finally {
    if (client) {
      await (client as { stop(): Promise<void> }).stop().catch(() => {})
    }
    if (tempJpeg) {
      try { unlinkSync(tempJpeg) } catch { /* ignore cleanup errors */ }
    }
  }
}
