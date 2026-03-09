import { mkdir, readFile, writeFile } from 'fs/promises'
import { randomBytes } from 'crypto'
import { app } from 'electron'
import { dirname, join, resolve } from 'path'
import { homedir, hostname, networkInterfaces } from 'os'

export interface ThumbnailCacheConfigInput {
  directory?: string
  maxGB?: number
}

export interface ThumbnailCacheConfig {
  directory: string
  maxGB: number
  source: {
    directory: 'default' | 'env' | 'settings'
    maxGB: 'default' | 'env' | 'settings'
  }
}

export interface DistributedCoordinatorSettings {
  enabled: boolean
  autostart: boolean
  bindHost: string
  port: number
  publicHost: string
  authToken: string
  workerPathMappings: WorkerPathMapping[]
}

export interface WorkerPathMapping {
  sourcePrefix: string
  targetPrefix: string
}

export interface AppSettings {
  thumbnailCache: ThumbnailCacheConfig
  distributed: DistributedCoordinatorSettings
}

export interface AppSettingsInput {
  thumbnailCache?: ThumbnailCacheConfigInput
  distributed?: Partial<DistributedCoordinatorSettings>
}

export interface WorkerSetupInfo {
  coordinatorUrl: string
  command: string
  notes: string[]
}

export interface AppSettingsBundle {
  settings: AppSettings
  workerSetup: WorkerSetupInfo
}

export interface CoordinatorStatus {
  state: 'stopped' | 'starting' | 'running' | 'error'
  pid: number | null
  url: string | null
  lastError: string | null
}

interface StoredAppSettings {
  thumbnailCache?: {
    directory?: string
    maxGB?: number
  }
  distributed?: Partial<DistributedCoordinatorSettings>
}

const SETTINGS_FILE = 'app-settings.json'
const LEGACY_THUMB_CACHE_SETTINGS_FILE = 'thumbnail-cache-settings.json'
const THUMB_CACHE_DIR_ENV = 'IMGANALYZER_THUMB_CACHE_DIR'
const THUMB_CACHE_MAX_GB_ENV = 'IMGANALYZER_THUMB_CACHE_MAX_GB'
const DEFAULT_THUMB_CACHE_GB = 300
const DEFAULT_THUMB_CACHE_DIR = resolve(join(homedir(), '.cache', 'imganalyzer', 'thumbs'))
const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '::1', '[::1]'])
const DEFAULT_COORDINATOR_PORT = 8765

let cachedBundle: AppSettingsBundle | null = null

function isErrnoCode(err: unknown, code: string): boolean {
  return typeof err === 'object' && err !== null && 'code' in err && (err as { code?: string }).code === code
}

function parsePositiveNumber(raw: string | undefined): number | null {
  if (!raw) return null
  const parsed = Number(raw)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function expandHomePath(value: string): string {
  if (!value.startsWith('~')) return value
  if (value === '~') return homedir()
  if (value.startsWith('~/') || value.startsWith('~\\')) {
    return join(homedir(), value.slice(2))
  }
  return value
}

function isPrivateIpv4(ip: string): boolean {
  if (ip.startsWith('10.')) return true
  if (ip.startsWith('192.168.')) return true
  const match = /^172\.(\d+)\./.exec(ip)
  if (!match) return false
  const octet = Number(match[1])
  return octet >= 16 && octet <= 31
}

function detectLanIpv4(): string | null {
  const interfaces = networkInterfaces()
  let fallback: string | null = null
  for (const entries of Object.values(interfaces)) {
    for (const entry of entries ?? []) {
      if (!entry || entry.internal || entry.family !== 'IPv4') continue
      if (entry.address.startsWith('169.254.')) continue
      if (isPrivateIpv4(entry.address)) return entry.address
      fallback = fallback ?? entry.address
    }
  }
  return fallback
}

function isLoopbackHost(value: string): boolean {
  return LOOPBACK_HOSTS.has(value.trim())
}

function generateCoordinatorAuthToken(): string {
  return randomBytes(24).toString('hex')
}

function getSettingsPath(): string {
  return join(app.getPath('userData'), SETTINGS_FILE)
}

function getLegacyThumbnailSettingsPath(): string {
  return join(app.getPath('userData'), LEGACY_THUMB_CACHE_SETTINGS_FILE)
}

async function readJsonFile<T>(path: string): Promise<T | null> {
  try {
    const raw = await readFile(path, 'utf-8')
    return JSON.parse(raw) as T
  } catch (err) {
    if (isErrnoCode(err, 'ENOENT')) return null
    throw err
  }
}

async function readStoredSettings(): Promise<StoredAppSettings> {
  const stored = await readJsonFile<StoredAppSettings>(getSettingsPath()) ?? {}
  if (stored.thumbnailCache?.directory || typeof stored.thumbnailCache?.maxGB === 'number') {
    return stored
  }

  const legacyThumb = await readJsonFile<{ directory?: string; maxGB?: number }>(getLegacyThumbnailSettingsPath())
  if (!legacyThumb) return stored

  return {
    ...stored,
    thumbnailCache: {
      ...stored.thumbnailCache,
      ...(typeof legacyThumb.directory === 'string' || typeof legacyThumb.maxGB === 'number' ? legacyThumb : {}),
    },
  }
}

async function writeStoredSettings(settings: StoredAppSettings): Promise<void> {
  const path = getSettingsPath()
  await mkdir(dirname(path), { recursive: true })
  await writeFile(path, JSON.stringify(settings, null, 2), 'utf-8')
}

function resolveThumbnailCacheConfig(stored?: StoredAppSettings['thumbnailCache']): ThumbnailCacheConfig {
  const envDir = process.env[THUMB_CACHE_DIR_ENV]?.trim()
  const envMaxGB = parsePositiveNumber(process.env[THUMB_CACHE_MAX_GB_ENV])

  let directorySource: ThumbnailCacheConfig['source']['directory'] = 'default'
  let directory = DEFAULT_THUMB_CACHE_DIR
  if (envDir) {
    directory = resolve(expandHomePath(envDir))
    directorySource = 'env'
  } else if (typeof stored?.directory === 'string' && stored.directory.trim()) {
    directory = resolve(expandHomePath(stored.directory.trim()))
    directorySource = 'settings'
  }

  let maxGBSource: ThumbnailCacheConfig['source']['maxGB'] = 'default'
  let maxGB = DEFAULT_THUMB_CACHE_GB
  if (envMaxGB !== null) {
    maxGB = envMaxGB
    maxGBSource = 'env'
  } else if (typeof stored?.maxGB === 'number' && Number.isFinite(stored.maxGB) && stored.maxGB > 0) {
    maxGB = stored.maxGB
    maxGBSource = 'settings'
  }

  return {
    directory,
    maxGB,
    source: {
      directory: directorySource,
      maxGB: maxGBSource,
    },
  }
}

function normalizeDistributedSettings(stored?: Partial<DistributedCoordinatorSettings>): DistributedCoordinatorSettings {
  const lanIp = detectLanIpv4()
  const enabled = stored?.enabled === true
  const autostart = enabled && stored?.autostart === true
  const rawBindHost = typeof stored?.bindHost === 'string' ? stored.bindHost.trim() : ''
  let bindHost = rawBindHost || lanIp || '127.0.0.1'
  if (rawBindHost && isLoopbackHost(rawBindHost) && lanIp) {
    bindHost = lanIp
  }
  const port = typeof stored?.port === 'number' && Number.isInteger(stored.port) && stored.port > 0 && stored.port <= 65535
    ? stored.port
    : DEFAULT_COORDINATOR_PORT
  let publicHost = typeof stored?.publicHost === 'string' ? stored.publicHost.trim() : ''
  if (!publicHost && rawBindHost && isLoopbackHost(rawBindHost) && lanIp) {
    publicHost = lanIp
  }
  const authToken = typeof stored?.authToken === 'string' ? stored.authToken.trim() : ''
  const workerPathMappings = Array.isArray(stored?.workerPathMappings)
    ? stored.workerPathMappings
        .filter((item): item is WorkerPathMapping =>
          typeof item?.sourcePrefix === 'string' && typeof item?.targetPrefix === 'string')
        .map((item) => ({
          sourcePrefix: item.sourcePrefix.trim(),
          targetPrefix: item.targetPrefix.trim(),
        }))
        .filter((item) => item.sourcePrefix && item.targetPrefix)
    : []

  return {
    enabled,
    autostart,
    bindHost,
    port,
    publicHost,
    authToken,
    workerPathMappings,
  }
}

function maybePopulateCoordinatorAuthToken(stored: StoredAppSettings): StoredAppSettings | null {
  const normalized = normalizeDistributedSettings(stored.distributed)
  if (!normalized.enabled || isLoopbackHost(normalized.bindHost) || normalized.authToken) {
    return null
  }

  return {
    ...stored,
    distributed: {
      ...(stored.distributed ?? {}),
      authToken: generateCoordinatorAuthToken(),
    },
  }
}

function getAdvertisedHost(settings: DistributedCoordinatorSettings): string {
  if (settings.publicHost.trim()) return settings.publicHost.trim()
  const bindHost = settings.bindHost.trim()
  if (bindHost && bindHost !== '0.0.0.0' && bindHost !== '::' && bindHost !== '[::]') {
    return bindHost
  }
  const lanIp = detectLanIpv4()
  if (lanIp) return lanIp
  const name = hostname().trim()
  return name || '127.0.0.1'
}

export function getDistributedCoordinatorUrl(settings: DistributedCoordinatorSettings, forWorkers = false): string {
  const host = forWorkers ? getAdvertisedHost(settings) : settings.bindHost.trim()
  return `http://${host}:${settings.port}/jsonrpc`
}

function buildWorkerSetupInfo(settings: AppSettings): WorkerSetupInfo {
  const coordinatorUrl = getDistributedCoordinatorUrl(settings.distributed, true)
  const tokenSegment = settings.distributed.authToken
    ? ` --auth-token ${settings.distributed.authToken}`
    : ''
  const mappingSegments = settings.distributed.workerPathMappings.map(
    (mapping) => `  --path-mapping "${mapping.sourcePrefix}=${mapping.targetPrefix}"`)

  return {
    coordinatorUrl,
    command: [
      'imganalyzer run-distributed-worker',
      `  --coordinator ${coordinatorUrl}`,
      '  --worker-id worker-01',
      '  --cloud copilot',
      '  --auto-update',
      ...mappingSegments,
      tokenSegment ? ` ${tokenSegment.trimStart()}` : '',
    ].filter(Boolean).join(' \\\n'),
    notes: [
      'Workers only need coordinator HTTP access plus read-only access to the shared image files; they no longer open the coordinator SQLite database directly.',
      'Workers must either read the stored image paths directly or remap them with --path-mapping when the NAS mount root differs.',
      'Add one --path-mapping SOURCE_PREFIX=LOCAL_PREFIX flag per differing NAS mount root.',
      'Analysis results are sent back to the coordinator, which remains the only database writer.',
      'Set --cloud to the same provider used by the batch session so workers can process cloud_ai and aesthetic jobs consistently.',
      '--auto-update makes the worker check git for new commits every 60s and automatically pull + restart when updates are found.',
      settings.distributed.authToken
        ? 'Pass the configured auth token to each worker with --auth-token.'
        : 'Auth is currently disabled for the job server; enable a token before exposing it beyond localhost.',
    ],
  }
}

async function resolveBundle(force = false): Promise<AppSettingsBundle> {
  if (cachedBundle && !force) return cachedBundle

  let stored = await readStoredSettings()
  const hydratedStored = maybePopulateCoordinatorAuthToken(stored)
  if (hydratedStored) {
    await writeStoredSettings(hydratedStored)
    stored = hydratedStored
  }
  const settings: AppSettings = {
    thumbnailCache: resolveThumbnailCacheConfig(stored.thumbnailCache),
    distributed: normalizeDistributedSettings(stored.distributed),
  }

  cachedBundle = {
    settings,
    workerSetup: buildWorkerSetupInfo(settings),
  }
  return cachedBundle
}

export async function getAppSettingsBundle(force = false): Promise<AppSettingsBundle> {
  return resolveBundle(force)
}

export async function getAppSettings(): Promise<AppSettings> {
  return (await resolveBundle()).settings
}

export async function updateAppSettings(input: AppSettingsInput): Promise<AppSettingsBundle> {
  const stored = await readStoredSettings()
  const next: StoredAppSettings = {
    thumbnailCache: { ...(stored.thumbnailCache ?? {}) },
    distributed: { ...(stored.distributed ?? {}) },
  }

  if (input.thumbnailCache) {
    if (input.thumbnailCache.directory !== undefined) {
      const trimmed = input.thumbnailCache.directory.trim()
      if (!trimmed) throw new Error('Thumbnail cache directory cannot be empty')
      next.thumbnailCache!.directory = trimmed
    }
    if (input.thumbnailCache.maxGB !== undefined) {
      if (!Number.isFinite(input.thumbnailCache.maxGB) || input.thumbnailCache.maxGB <= 0) {
        throw new Error('Thumbnail cache maxGB must be a positive number')
      }
      next.thumbnailCache!.maxGB = Number(input.thumbnailCache.maxGB)
    }
  }

  if (input.distributed) {
    if (input.distributed.enabled !== undefined) {
      next.distributed!.enabled = input.distributed.enabled === true
    }
    if (input.distributed.autostart !== undefined) {
      next.distributed!.autostart = input.distributed.autostart === true
    }
    if (input.distributed.bindHost !== undefined) {
      const trimmed = input.distributed.bindHost.trim()
      if (!trimmed) throw new Error('Job server bind host cannot be empty')
      next.distributed!.bindHost = trimmed
    }
    if (input.distributed.port !== undefined) {
      if (!Number.isInteger(input.distributed.port) || input.distributed.port <= 0 || input.distributed.port > 65535) {
        throw new Error('Job server port must be an integer between 1 and 65535')
      }
      next.distributed!.port = input.distributed.port
    }
    if (input.distributed.publicHost !== undefined) {
      next.distributed!.publicHost = input.distributed.publicHost.trim()
    }
    if (input.distributed.authToken !== undefined) {
      next.distributed!.authToken = input.distributed.authToken.trim()
    }
    if (input.distributed.workerPathMappings !== undefined) {
      next.distributed!.workerPathMappings = input.distributed.workerPathMappings.map((item) => {
        const sourcePrefix = item.sourcePrefix.trim()
        const targetPrefix = item.targetPrefix.trim()
        if (!sourcePrefix || !targetPrefix) {
          throw new Error('Each worker path mapping requires both a source prefix and a local prefix')
        }
        return { sourcePrefix, targetPrefix }
      })
    }
  }

  const finalSettings = maybePopulateCoordinatorAuthToken(next) ?? next
  await writeStoredSettings(finalSettings)
  cachedBundle = null
  return resolveBundle(true)
}
