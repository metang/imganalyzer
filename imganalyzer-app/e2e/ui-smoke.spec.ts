import { _electron as electron, expect, test, type ElectronApplication, type Page } from '@playwright/test'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdir, rm } from 'node:fs/promises'
import path from 'node:path'

declare global {
  interface Window {
    api: {
      batchCheckPending(): Promise<{ pending: number; running: number }>
      searchImages(filters: Record<string, unknown>): Promise<{
        results: Array<{ description: string | null; keywords: string[] | null }>
        total: number | null
        hasMore: boolean
        error?: string
      }>
    }
  }
}

const appRoot = path.resolve(__dirname, '..')
const electronEntry = path.join(appRoot, 'out', 'main', 'index.js')
const fakeBackend = path.join(appRoot, 'e2e', 'fake-json-rpc-server.cjs')
const e2eCacheRoot = path.join(appRoot, 'node_modules', '.cache', 'imganalyzer-e2e')
const userDataDir = path.join(e2eCacheRoot, 'user-data')
const thumbnailCacheDir = path.join(e2eCacheRoot, 'thumbs')
const januaryFolder = 'E:/Pic/2013/01'
const currentFolder = 'E:/Pic/2013/02-current'
const slowStaleFolder = 'E:/Pic/2013/03-slow-stale'
const slowFailureFolder = 'E:/Pic/2013/04-slow-failure'

interface JsonRpcReply {
  jsonrpc: '2.0'
  id?: number | null
  result?: unknown
  error?: { code: number; message: string }
  method?: string
  params?: unknown
}

interface FakeRpcClient {
  callRaw(method: string, params?: Record<string, unknown>): Promise<JsonRpcReply>
  close(): Promise<void>
}

async function cleanE2eCache(): Promise<void> {
  await rm(e2eCacheRoot, { recursive: true, force: true })
  await mkdir(userDataDir, { recursive: true })
  await mkdir(thumbnailCacheDir, { recursive: true })
}

async function removeE2eCache(): Promise<void> {
  await rm(e2eCacheRoot, { recursive: true, force: true })
}

async function startFakeRpc(): Promise<FakeRpcClient> {
  const proc = spawn(process.execPath, [fakeBackend], {
    cwd: appRoot,
    env: process.env,
    stdio: 'pipe',
  }) as ChildProcessWithoutNullStreams

  let nextId = 1
  let stdoutBuf = ''
  let ready = false
  const pending = new Map<number, {
    resolve: (reply: JsonRpcReply) => void
    reject: (err: Error) => void
    timer: ReturnType<typeof setTimeout>
  }>()

  const rejectPending = (err: Error): void => {
    for (const [id, call] of pending) {
      clearTimeout(call.timer)
      call.reject(err)
      pending.delete(id)
    }
  }

  const client: FakeRpcClient = {
    callRaw(method: string, params: Record<string, unknown> = {}) {
      const id = nextId++
      return new Promise<JsonRpcReply>((resolve, reject) => {
        const timer = setTimeout(() => {
          pending.delete(id)
          reject(new Error(`fake RPC '${method}' timed out`))
        }, 5_000)

        pending.set(id, { resolve, reject, timer })
        proc.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
      })
    },
    async close() {
      if (proc.exitCode !== null || proc.killed) return
      proc.kill()
      await new Promise<void>((resolve) => {
        const timer = setTimeout(resolve, 1_000)
        proc.once('exit', () => {
          clearTimeout(timer)
          resolve()
        })
      })
    },
  }

  return new Promise<FakeRpcClient>((resolve, reject) => {
    const readyTimer = setTimeout(() => {
      reject(new Error('fake RPC server did not become ready'))
      proc.kill()
    }, 5_000)

    proc.stdout.on('data', (chunk: Buffer) => {
      stdoutBuf += chunk.toString('utf8')
      const lines = stdoutBuf.split('\n')
      stdoutBuf = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.trim()) continue
        const msg = JSON.parse(line) as JsonRpcReply
        if (msg.method === 'server/ready' && msg.id === undefined) {
          ready = true
          clearTimeout(readyTimer)
          resolve(client)
          continue
        }
        if (typeof msg.id === 'number') {
          const call = pending.get(msg.id)
          if (!call) continue
          pending.delete(msg.id)
          clearTimeout(call.timer)
          call.resolve(msg)
        }
      }
    })

    proc.once('exit', (code) => {
      const err = new Error(`fake RPC server exited with code ${code}`)
      if (!ready) {
        clearTimeout(readyTimer)
        reject(err)
      }
      rejectPending(err)
    })
  })
}

async function launchApp(): Promise<{ app: ElectronApplication; page: Page }> {
  expect(existsSync(electronEntry), 'Run npm run build before npm run test:e2e').toBeTruthy()

  const app = await electron.launch({
    args: [`--user-data-dir=${userDataDir}`, appRoot],
    env: {
      ...process.env,
      ELECTRON_DISABLE_SECURITY_WARNINGS: 'true',
      IMGANALYZER_PYTHON_MODULE_COMMAND: JSON.stringify([process.execPath, fakeBackend]),
      IMGANALYZER_THUMB_CACHE_DIR: thumbnailCacheDir,
      IMGANALYZER_THUMB_CACHE_MAX_GB: '1',
    },
  })

  const page = await app.firstWindow()
  page.on('console', (message) => {
    if (message.type() === 'error') {
      console.error(`[renderer:${message.type()}] ${message.text()}`)
    }
  })
  page.on('pageerror', (error) => {
    console.error(`[renderer:pageerror] ${error.stack ?? error.message}`)
  })
  await page.waitForLoadState('domcontentloaded')
  return { app, page }
}

test.describe('Electron UI smoke', () => {
  let electronApp: ElectronApplication | null = null

  test.beforeEach(async () => {
    await cleanE2eCache()
  })

  test.afterEach(async () => {
    if (electronApp) {
      await electronApp.close()
      electronApp = null
    }
    await removeE2eCache()
  })

  test('launches the shell and reaches the fake JSON-RPC backend through preload IPC', async () => {
    const launched = await launchApp()
    electronApp = launched.app
    const { page } = launched

    await expect(page.getByText('imganalyzer')).toBeVisible()
    await expect(page.getByRole('button', { name: 'Gallery' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Batch' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Search' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Settings' })).toBeVisible()

    const pending = await page.evaluate(() => window.api.batchCheckPending())
    expect(pending).toEqual({ pending: 0, running: 0 })

    const search = await page.evaluate(() =>
      window.api.searchImages({ query: 'hermetic smoke', mode: 'text', limit: 1 }),
    )
    expect(search).toMatchObject({ total: 1, hasMore: false })
    expect(search.results[0]?.description).toContain('fake JSON-RPC backend')
    expect(search.results[0]?.keywords).toContain('e2e')
  })

  test('fake backend returns real JSON-RPC errors instead of success-shaped unknown methods', async () => {
    const rpc = await startFakeRpc()
    try {
      const known = await rpc.callRaw('gallery/listImagesChunk', {
        folderPath: januaryFolder,
        recursive: true,
        chunkSize: 150,
        cursor: null,
      })
      expect(known.error).toBeUndefined()
      expect(known.result).toMatchObject({
        items: expect.any(Array),
        nextCursor: null,
        hasMore: false,
        total: 863,
      })

      const unknown = await rpc.callRaw('fake/not-a-real-method')
      expect(unknown.result).toBeUndefined()
      expect(unknown.error).toMatchObject({
        code: -32601,
        message: expect.stringContaining('Method not found'),
      })
    } finally {
      await rpc.close()
    }
  })

  test('loads a large gallery folder without showing the timeout error', async () => {
    const launched = await launchApp()
    electronApp = launched.app
    const { page } = launched

    await expect(page.getByRole('button', { name: 'Gallery' })).toBeVisible()
    await expect(page.getByText('1 loaded of 474186')).toBeVisible()

    const renderedFolderButtons = page.locator('aside button[title^="E:/Pic/"]')
    await expect(renderedFolderButtons.first()).toBeVisible()
    expect(await renderedFolderButtons.count()).toBeLessThan(80)

    await page.locator(`button[title="${januaryFolder}"]`).click()

    await expect(page.getByText(januaryFolder)).toBeVisible()
    await expect(page.getByText('1 loaded of 863')).toBeVisible()
    await expect(page.getByText('Failed to load gallery')).toHaveCount(0)
    await expect(page.getByText(/gallery\/listImagesChunk timed out/i)).toHaveCount(0)
  })

  test('ignores obsolete slow gallery success responses after switching folders', async () => {
    const launched = await launchApp()
    electronApp = launched.app
    const { page } = launched

    await expect(page.getByText('1 loaded of 474186')).toBeVisible()
    await page.locator(`button[title="${slowStaleFolder}"]`).click()
    await expect(page.getByText(slowStaleFolder)).toBeVisible()
    await page.waitForTimeout(50)

    await page.locator(`button[title="${currentFolder}"]`).click()
    await expect(page.getByText(currentFolder)).toBeVisible()
    await expect(page.getByText('1 loaded of 2')).toBeVisible()

    await page.waitForTimeout(650)
    await expect(page.getByText(currentFolder)).toBeVisible()
    await expect(page.getByText('1 loaded of 2')).toBeVisible()
    await expect(page.getByText('1 loaded of 99')).toHaveCount(0)
  })

  test('ignores obsolete slow gallery failures after switching folders', async () => {
    const launched = await launchApp()
    electronApp = launched.app
    const { page } = launched

    await expect(page.getByText('1 loaded of 474186')).toBeVisible()
    await page.locator(`button[title="${slowFailureFolder}"]`).click()
    await expect(page.getByText(slowFailureFolder)).toBeVisible()
    await page.waitForTimeout(50)

    await page.locator(`button[title="${currentFolder}"]`).click()
    await expect(page.getByText(currentFolder)).toBeVisible()
    await expect(page.getByText('1 loaded of 2')).toBeVisible()

    await page.waitForTimeout(650)
    await expect(page.getByText(currentFolder)).toBeVisible()
    await expect(page.getByText('1 loaded of 2')).toBeVisible()
    await expect(page.getByText('Failed to load gallery')).toHaveCount(0)
    await expect(page.getByText(/gallery\/listImagesChunk timed out/i)).toHaveCount(0)
  })
})
