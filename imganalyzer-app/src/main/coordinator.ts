import type { ChildProcess } from 'child_process'
import { spawnPythonModule, killProcessTree } from './python-runtime'
import type { CoordinatorStatus, DistributedCoordinatorSettings } from './settings'
import { getDistributedCoordinatorUrl } from './settings'

const STARTUP_TIMEOUT_MS = 30_000
const READY_PATTERN = /\[server\.http\] listening on http:/i

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

function attachLineReader(
  stream: NodeJS.ReadableStream | null | undefined,
  onLine: (line: string) => void,
): void {
  let buffer = ''
  stream?.on('data', (chunk: Buffer | string) => {
    buffer += chunk.toString()
    const lines = buffer.split(/\r?\n/)
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      onLine(line.trim())
    }
  })
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
      setStatus({
        state: 'error',
        pid: null,
        url: getDistributedCoordinatorUrl(settings),
        lastError: message,
      })
      reject(new Error(message))
    }

    const timer = setTimeout(() => {
      if (!settled) {
        try {
          killProcessTree(proc.pid)
        } catch {
          // Ignore shutdown races.
        }
        finishReject('Distributed job server startup timed out')
      }
    }, STARTUP_TIMEOUT_MS)

    attachLineReader(proc.stdout, (line) => handleCoordinatorLine(line, settings, () => {
      clearTimeout(timer)
      finishResolve()
    }))
    attachLineReader(proc.stderr, (line) => handleCoordinatorLine(line, settings, () => {
      clearTimeout(timer)
      finishResolve()
    }))

    proc.on('error', (err) => {
      clearTimeout(timer)
      finishReject(`Distributed job server failed to start: ${err.message}`)
    })

    proc.on('close', (code) => {
      clearTimeout(timer)
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

      const message = `Distributed job server exited with code ${code ?? 'unknown'}`
      if (status.state === 'running') {
        coordinatorProcess = null
        startPromise = null
        startingSignature = null
        runningSignature = null
        setStatus({
          state: 'error',
          pid: null,
          url: getDistributedCoordinatorUrl(settings),
          lastError: message,
        })
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
  await startCoordinator(settings)
}
