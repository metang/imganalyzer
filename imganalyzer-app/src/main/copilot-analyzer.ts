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
 */

import type { XmpData } from './xmp'

// ─── Prompt ──────────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are an expert photography analyst. Analyze the provided image and return a JSON object with exactly these fields:
- description: (string) A detailed 2-3 sentence description of the image content
- scene_type: (string) e.g. "landscape", "portrait", "street", "architecture", "macro", "wildlife", "abstract", "product"
- main_subject: (string) Primary subject(s) in the image
- lighting: (string) Lighting conditions e.g. "golden hour", "overcast", "harsh midday", "studio softbox"
- mood: (string) Emotional tone e.g. "serene", "dramatic", "intimate", "moody"
- keywords: (array of strings) 10-15 descriptive keywords suitable as photo tags
- technical_notes: (string) Notable photographic or technical observations

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

  let client: InstanceType<SDK['CopilotClient']> | null = null

  try {
    client = new CopilotClient()
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
