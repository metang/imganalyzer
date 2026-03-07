import type { ChildProcess } from 'child_process'
import { spawnPythonModule, killProcessTree } from './python-runtime'
import type { CoordinatorStatus, DistributedCoordinatorSettings } from './settings'
import { getDistributedCoordinatorUrl } from './settings'

const STARTUP_TIMEOUT_MS = 30_000
const READY_PATTERN = /\[server\.http\] listening on http:/i
const STARTUP_DIAGNOSTIC_LINE_LIMIT = 25
const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '::1', '[::1]'])

let coordinatorProcess: ChildProcess | null = null
let startPromise: Promise<void> | null = null
let expectedStopPid: number | null = null
let runningSignature: string | null = null
let startingSignature: string | null = null
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

export function getCoordinatorStatus(): CoordinatorStatus {
  return { ...status }
}

export async function stopCoordinator(): Promise<void> {
  const proc = coordinatorProcess
  if (!proc) {
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

  coordinatorProcess = null
  startPromise = null
  startingSignature = null
  runningSignature = null
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
