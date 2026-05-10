import { _electron as electron, expect, test, type ElectronApplication, type Page } from '@playwright/test'
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

async function cleanE2eCache(): Promise<void> {
  await rm(e2eCacheRoot, { recursive: true, force: true })
  await mkdir(userDataDir, { recursive: true })
  await mkdir(thumbnailCacheDir, { recursive: true })
}

async function removeE2eCache(): Promise<void> {
  await rm(e2eCacheRoot, { recursive: true, force: true })
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
})
