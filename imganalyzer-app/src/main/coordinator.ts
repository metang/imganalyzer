import type { ChildProcess } from 'child_process'
import { execSync } from 'child_process'
import { spawnPythonModule, killProcessTree } from './python-runtime'
import type { CoordinatorStatus, DistributedCoordinatorSettings } from './settings'
import { getDistributedCoordinatorUrl } from './settings'

const STARTUP_TIMEOUT_MS = 60_000
const READY_PATTERN = /\[server\.http\] listening on http:/i
const STARTUP_DIAGNOSTIC_LINE_LIMIT = 25
const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '::1', '[::1]'])

let coordinatorProcess: ChildProcess | null = null
let startPromise: Promise<void> | null = null
let expectedStopPid: number | null = null
let runningSignature: string | null = null
let startingSignature: string | null = null
let activePort: number | null = null
let status: CoordinatorStatus = {
  state: 'stopped',
  pid: null,
  url: null,
  lastError: null,
}

function getSignature(settings: DistributedCoordinatorSettings): string {
  return JSON.stringify({
    bindHost: settings.bindHost,
    port: settings.port,
    authToken: settings.authToken,
  })
}

function setStatus(next: CoordinatorStatus): void {
  status = next
}

function setCoordinatorError(settings: DistributedCoordinatorSettings, message: string): void {
  setStatus({
    state: 'error',
    pid: null,
    url: getDistributedCoordinatorUrl(settings),
    lastError: message,
  })
}

function getCoordinatorConfigError(settings: DistributedCoordinatorSettings): string | null {
  const bindHost = settings.bindHost.trim().toLowerCase()
  if (!bindHost || LOOPBACK_HOSTS.has(bindHost) || settings.authToken.trim()) {
    return null
  }

  return 'Distributed job server requires an auth token when Bind host is not localhost. Add an auth token or change Bind host to 127.0.0.1.'
}

function handleCoordinatorLine(
  line: string,
  settings: DistributedCoordinatorSettings,
  resolveStart: () => void,
): void {
  if (!line.trim()) return
  console.error('[coordinator]', line)

  if (READY_PATTERN.test(line)) {
    runningSignature = getSignature(settings)
    startingSignature = null
    setStatus({
      state: 'running',
      pid: coordinatorProcess?.pid ?? null,
      url: getDistributedCoordinatorUrl(settings),
      lastError: null,
    })
    resolveStart()
  }
}

function recordStartupLine(lines: string[], line: string): void {
  if (!line.trim()) return
  lines.push(line)
  if (lines.length > STARTUP_DIAGNOSTIC_LINE_LIMIT) {
    lines.splice(0, lines.length - STARTUP_DIAGNOSTIC_LINE_LIMIT)
  }
}

function selectStartupDiagnostics(lines: string[]): string[] {
  const filtered = lines.filter((line) => !READY_PATTERN.test(line))
  const prioritized = filtered.filter((line) =>
    /traceback|error|exception|failed|required|usage:|no module named|address already in use|permission denied|winerror/i.test(line))
  if (prioritized.length > 0) return prioritized.slice(-6)
  return filtered.slice(-6)
}

function formatStartupFailure(message: string, stderrLines: string[], stdoutLines: string[]): string {
  const stderrDiagnostics = selectStartupDiagnostics(stderrLines)
  if (stderrDiagnostics.length > 0) {
    return `${message}. Startup stderr: ${stderrDiagnostics.join(' | ')}`
  }

  const stdoutDiagnostics = selectStartupDiagnostics(stdoutLines)
  if (stdoutDiagnostics.length > 0) {
    return `${message}. Startup stdout: ${stdoutDiagnostics.join(' | ')}`
  }

  return message
}

function attachLineReader(
  stream: NodeJS.ReadableStream | null | undefined,
  onLine: (line: string) => void,
): () => void {
  let buffer = ''
  const flush = () => {
    const trailing = buffer.trim()
    buffer = ''
    if (trailing) onLine(trailing)
  }
  stream?.on('data', (chunk: Buffer | string) => {
    buffer += chunk.toString()
    const lines = buffer.split(/\r?\n/)
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      onLine(line.trim())
    }
  })
  stream?.on('end', flush)
  stream?.on('close', flush)
  return flush
}

/**
 * Kill any orphaned coordinator processes listening on the given port.
 * On Windows, uses netstat + taskkill; on Unix, uses lsof + kill.
 * Ignores errors silently — this is a best-effort cleanup.
 */
function killOrphanedCoordinators(port: number, ownPid: number | undefined): void {
  if (!Number.isInteger(port) || port < 1 || port > 65535) return
  try {
    if (process.platform === 'win32') {
      // Find PIDs listening on the target port
      const out = execSync(
        `netstat -ano | findstr LISTENING | findstr :${port}`,
        { encoding: 'utf-8', timeout: 5000, stdio: ['pipe', 'pipe', 'pipe'] },
      )
      const pids = new Set<number>()
      for (const line of out.split(/\r?\n/)) {
        const match = line.trim().match(/\s(\d+)$/)
        if (match) pids.add(Number(match[1]))
      }
      for (const pid of pids) {
        if (pid === ownPid || pid === 0) continue
        console.error(`[coordinator] killing orphaned process PID ${pid} on port ${port}`)
        try {
          execSync(`taskkill /T /F /PID ${pid}`, { stdio: 'ignore', timeout: 5000 })
        } catch { /* already gone */ }
      }
    } else {
      // macOS / Linux: lsof to find listeners
      const out = execSync(
        `lsof -iTCP:${port} -sTCP:LISTEN -t 2>/dev/null || true`,
        { encoding: 'utf-8', timeout: 5000, stdio: ['pipe', 'pipe', 'pipe'] },
      )
      for (const line of out.split(/\r?\n/)) {
        const pid = Number(line.trim())
        if (!pid || pid === ownPid) continue
        console.error(`[coordinator] killing orphaned process PID ${pid} on port ${port}`)
        try {
          process.kill(pid, 'SIGKILL')
        } catch { /* already gone */ }
      }
    }
  } catch {
    // Best-effort — netstat/lsof might not find anything
  }
}

export function getCoordinatorStatus(): CoordinatorStatus {
  return { ...status }
}

export async function stopCoordinator(): Promise<void> {
  const proc = coordinatorProcess
  if (!proc) {
    // Even without a tracked process, kill anything on the port
    if (activePort) killOrphanedCoordinators(activePort, undefined)
    activePort = null
    setStatus({ state: 'stopped', pid: null, url: null, lastError: null })
    return
  }

  expectedStopPid = proc.pid ?? null
  try {
    killProcessTree(proc.pid)
  } catch {
    // The process may already be gone.
  }

  await new Promise<void>((resolve) => {
    proc.once('close', () => resolve())
    setTimeout(resolve, 3000)
  })

  // Kill any orphaned Python processes that survived the conda wrapper kill
  if (activePort) killOrphanedCoordinators(activePort, undefined)

  coordinatorProcess = null
  startPromise = null
  startingSignature = null
  runningSignature = null
  activePort = null
  setStatus({ state: 'stopped', pid: null, url: null, lastError: null })
}

export async function startCoordinator(settings: DistributedCoordinatorSettings): Promise<void> {
  const signature = getSignature(settings)
  if (coordinatorProcess && status.state === 'running' && runningSignature === signature) return
  if (startPromise && startingSignature === signature) return startPromise
  const configError = getCoordinatorConfigError(settings)
  if (configError) {
    if (!coordinatorProcess) {
      startingSignature = null
      runningSignature = null
      setCoordinatorError(settings, configError)
    }
    return Promise.reject(new Error(configError))
  }
  if (coordinatorProcess) await stopCoordinator()

  // Kill any orphaned coordinator processes left over from a previous session
  killOrphanedCoordinators(settings.port, undefined)
  activePort = settings.port

  const args = [
    '--transport', 'http',
    '--host', settings.bindHost,
    '--port', String(settings.port),
  ]
  if (settings.authToken) {
    args.push('--auth-token', settings.authToken)
  }

  const proc = spawnPythonModule('imganalyzer.server', args)
  coordinatorProcess = proc
  startingSignature = signature
  runningSignature = null
  setStatus({
    state: 'starting',
    pid: proc.pid ?? null,
    url: getDistributedCoordinatorUrl(settings),
    lastError: null,
  })

  startPromise = new Promise<void>((resolve, reject) => {
    let settled = false
    const startupStdoutLines: string[] = []
    const startupStderrLines: string[] = []
    const finishResolve = () => {
      if (settled) return
      settled = true
      startPromise = null
      resolve()
    }
    const finishReject = (message: string) => {
      if (settled) return
      settled = true
      coordinatorProcess = null
      startPromise = null
      startingSignature = null
      runningSignature = null
      setCoordinatorError(settings, message)
      reject(new Error(message))
    }

    const timer = setTimeout(() => {
      if (!settled) {
        try {
          killProcessTree(proc.pid)
        } catch {
          // Ignore shutdown races.
        }
        finishReject(formatStartupFailure('Distributed job server startup timed out', startupStderrLines, startupStdoutLines))
      }
    }, STARTUP_TIMEOUT_MS)

    const flushStdout = attachLineReader(proc.stdout, (line) => {
      recordStartupLine(startupStdoutLines, line)
      handleCoordinatorLine(line, settings, () => {
        clearTimeout(timer)
        finishResolve()
      })
    })
    const flushStderr = attachLineReader(proc.stderr, (line) => {
      recordStartupLine(startupStderrLines, line)
      handleCoordinatorLine(line, settings, () => {
        clearTimeout(timer)
        finishResolve()
      })
    })

    proc.on('error', (err) => {
      clearTimeout(timer)
      flushStdout()
      flushStderr()
      finishReject(
        formatStartupFailure(
          `Distributed job server failed to start: ${err.message}`,
          startupStderrLines,
          startupStdoutLines,
        ),
      )
    })

    proc.on('close', (code) => {
      clearTimeout(timer)
      flushStdout()
      flushStderr()
      const wasExpected = expectedStopPid !== null && expectedStopPid === proc.pid
      if (wasExpected) {
        expectedStopPid = null
        coordinatorProcess = null
        startPromise = null
        startingSignature = null
        runningSignature = null
        setStatus({ state: 'stopped', pid: null, url: null, lastError: null })
        if (!settled) {
          settled = true
          resolve()
        }
        return
      }

      const message = formatStartupFailure(
        `Distributed job server exited with code ${code ?? 'unknown'}`,
        startupStderrLines,
        startupStdoutLines,
      )
      if (status.state === 'running') {
        coordinatorProcess = null
        startPromise = null
        startingSignature = null
        runningSignature = null
        setCoordinatorError(settings, message)
        return
      }
      finishReject(message)
    })
  })

  return startPromise
}

export async function applyCoordinatorSettings(settings: DistributedCoordinatorSettings): Promise<void> {
  if (!settings.enabled) {
    await stopCoordinator()
    return
  }
  await startCoordinator(settings)
}

export async function startCoordinatorOnLaunch(settings: DistributedCoordinatorSettings): Promise<void> {
  if (!settings.enabled || !settings.autostart) return
  const configError = getCoordinatorConfigError(settings)
  if (configError) {
    if (!coordinatorProcess) {
      startingSignature = null
      runningSignature = null
      setCoordinatorError(settings, `Auto-start skipped: ${configError}`)
    }
    return
  }
  await startCoordinator(settings)
}
