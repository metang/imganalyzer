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

import { spawn, execSync } from 'child_process'
import { existsSync } from 'fs'
import { dirname, join } from 'path'
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

// ─── Prompt ──────────────────────────────────────────────────────────────────

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
            attachments: [{ type: 'file', path: imagePath }],
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
  }
}
