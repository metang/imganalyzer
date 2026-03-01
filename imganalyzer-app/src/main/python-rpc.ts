/**
 * python-rpc.ts — JSON-RPC client for the persistent Python process.
 *
 * Manages a single long-lived Python process communicating via
 * line-delimited JSON-RPC 2.0 over stdin/stdout.  Eliminates the
 * 1-3s conda subprocess startup overhead per operation.
 *
 * Usage:
 *   import { rpc, ensureServerRunning, shutdownServer } from './python-rpc'
 *
 *   // One-shot call:
 *   const stats = await rpc.call('status', {})
 *
 *   // Streaming call (notifications arrive via callback):
 *   await rpc.call('ingest', { folders: ['/photos'] }, (notif) => {
 *     if (notif.method === 'ingest/progress') { ... }
 *   })
 */

import { spawn, ChildProcess } from 'child_process'
import { app } from 'electron'
import { dirname } from 'path'

// ── Constants ────────────────────────────────────────────────────────────────

// app.getAppPath() returns imganalyzer-app/ in dev; the package root is its parent.
const PKG_ROOT = process.env.IMGANALYZER_PKG_ROOT || dirname(app.getAppPath())
const CONDA_ENV = 'imganalyzer'
const STARTUP_TIMEOUT_MS = 30_000
const CALL_TIMEOUT_MS = 120_000

// ── Types ─────────────────────────────────────────────────────────────────────

interface JsonRpcRequest {
  jsonrpc: '2.0'
  id: number
  method: string
  params: Record<string, unknown>
}

interface JsonRpcResponse {
  jsonrpc: '2.0'
  id?: number
  result?: unknown
  error?: { code: number; message: string }
  method?: string   // notification
  params?: unknown  // notification
}

type NotificationCallback = (notification: { method: string; params: unknown }) => void

interface PendingCall {
  resolve: (value: unknown) => void
  reject: (error: Error) => void
  notificationCb?: NotificationCallback
  timer: ReturnType<typeof setTimeout>
}

// ── Server process management ────────────────────────────────────────────────

let serverProcess: ChildProcess | null = null
let nextId = 1
const pendingCalls = new Map<number, PendingCall>()

// Global notification listeners (for streaming results during run/ingest)
let globalNotificationCb: NotificationCallback | null = null

let stdoutBuf = ''
let isReady = false
let readyResolve: (() => void) | null = null
let readyReject: ((err: Error) => void) | null = null

function handleLine(line: string): void {
  const trimmed = line.trim()
  if (!trimmed) return

  let msg: JsonRpcResponse
  try {
    msg = JSON.parse(trimmed)
  } catch {
    // Not JSON — ignore (could be Python stderr leak)
    return
  }

  // ── Notification (no id) ────────────────────────────────────────────
  if (msg.method && msg.id === undefined) {
    // server/ready is handled specially
    if (msg.method === 'server/ready') {
      isReady = true
      readyResolve?.()
      return
    }

    // Route to global notification listener
    if (globalNotificationCb) {
      globalNotificationCb({ method: msg.method, params: msg.params })
    }

    // Also route to any pending call's notification callback
    // (streaming methods like ingest/run/analyze emit notifications
    // while the call is still pending)
    for (const pending of pendingCalls.values()) {
      if (pending.notificationCb) {
        pending.notificationCb({ method: msg.method, params: msg.params })
      }
    }
    return
  }

  // ── Response (has id) ──────────────────────────────────────────────
  if (msg.id !== undefined) {
    const pending = pendingCalls.get(msg.id)
    if (!pending) return
    pendingCalls.delete(msg.id)
    clearTimeout(pending.timer)

    if (msg.error) {
      pending.reject(new Error(msg.error.message))
    } else {
      pending.resolve(msg.result)
    }
  }
}

function attachStdoutParser(proc: ChildProcess): void {
  proc.stdout?.on('data', (chunk: Buffer) => {
    stdoutBuf += chunk.toString('utf8')
    const lines = stdoutBuf.split('\n')
    stdoutBuf = lines.pop() ?? ''
    for (const line of lines) {
      handleLine(line)
    }
  })
}

/**
 * Start the persistent Python JSON-RPC server if not already running.
 * Resolves when the server emits its `server/ready` notification.
 */
export function ensureServerRunning(): Promise<void> {
  if (serverProcess && !serverProcess.killed && isReady) {
    return Promise.resolve()
  }

  // If already starting, wait for the same ready promise
  if (serverProcess && !serverProcess.killed && readyResolve) {
    return new Promise((resolve, reject) => {
      const prevResolve = readyResolve!
      const prevReject = readyReject!
      readyResolve = () => { prevResolve(); resolve() }
      readyReject = (err) => { prevReject(err); reject(err) }
    })
  }

  return new Promise((resolve, reject) => {
    isReady = false
    stdoutBuf = ''
    readyResolve = resolve
    readyReject = reject

    const proc = spawn(
      'conda',
      [
        'run', '-n', CONDA_ENV, '--no-capture-output',
        'python', '-m', 'imganalyzer.server',
      ],
      {
        cwd: PKG_ROOT,
        env: {
          ...process.env,
          HF_HUB_DISABLE_SYMLINKS_WARNING: '1',
          PYTHONIOENCODING: 'utf-8',
          PYTHONUTF8: '1',
        },
        stdio: ['pipe', 'pipe', 'pipe'],
      }
    )

    serverProcess = proc
    attachStdoutParser(proc)

    // Log stderr for debugging but don't parse it
    proc.stderr?.on('data', (chunk: Buffer) => {
      const text = chunk.toString('utf8').trim()
      if (text) console.error('[python-rpc stderr]', text)
    })

    proc.on('error', (err) => {
      serverProcess = null
      isReady = false
      readyReject?.(err)
      readyResolve = null
      readyReject = null
      // Reject all pending calls
      for (const [id, pending] of pendingCalls) {
        clearTimeout(pending.timer)
        pending.reject(new Error('Python server process error'))
        pendingCalls.delete(id)
      }
    })

    proc.on('close', (code) => {
      const wasReady = isReady
      serverProcess = null
      isReady = false
      if (!wasReady) {
        readyReject?.(new Error(`Python server exited with code ${code} during startup`))
      }
      readyResolve = null
      readyReject = null
      // Reject all pending calls
      for (const [id, pending] of pendingCalls) {
        clearTimeout(pending.timer)
        pending.reject(new Error(`Python server exited with code ${code}`))
        pendingCalls.delete(id)
      }
    })

    // Startup timeout
    const timer = setTimeout(() => {
      if (!isReady) {
        try { proc.kill() } catch { /* ignore */ }
        reject(new Error('Python server startup timed out'))
      }
    }, STARTUP_TIMEOUT_MS)

    // Clear timeout once ready
    const origResolve = readyResolve
    readyResolve = () => {
      clearTimeout(timer)
      origResolve?.()
    }
  })
}

/**
 * Gracefully shut down the Python server.
 */
export async function shutdownServer(): Promise<void> {
  if (!serverProcess || serverProcess.killed) return

  try {
    await rpc.call('shutdown', {})
  } catch {
    // If call fails, force kill
  }

  // Give it a moment to exit cleanly, then force kill
  await new Promise<void>((resolve) => {
    const timer = setTimeout(() => {
      try { serverProcess?.kill() } catch { /* ignore */ }
      resolve()
    }, 3000)

    serverProcess?.on('close', () => {
      clearTimeout(timer)
      resolve()
    })
  })

  serverProcess = null
  isReady = false
}

/**
 * Set a global notification listener for streaming results.
 * Used by batch.ts to receive run/result and ingest/progress notifications.
 */
export function setNotificationListener(cb: NotificationCallback | null): void {
  globalNotificationCb = cb
}

/**
 * Check if the server is currently running and ready.
 */
export function isServerReady(): boolean {
  return isReady && serverProcess !== null && !serverProcess.killed
}

// ── RPC call interface ───────────────────────────────────────────────────────

export const rpc = {
  /**
   * Send a JSON-RPC call to the Python server.
   *
   * @param method - The RPC method name
   * @param params - Parameters object
   * @param notificationCb - Optional callback for notifications received while this call is pending
   * @param timeoutMs - Override the default timeout
   * @returns The result from the server
   */
  async call(
    method: string,
    params: Record<string, unknown>,
    notificationCb?: NotificationCallback,
    timeoutMs: number = CALL_TIMEOUT_MS,
  ): Promise<unknown> {
    await ensureServerRunning()

    const id = nextId++
    const request: JsonRpcRequest = {
      jsonrpc: '2.0',
      id,
      method,
      params,
    }

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pendingCalls.delete(id)
        reject(new Error(`RPC call '${method}' timed out after ${timeoutMs}ms`))
      }, timeoutMs)

      pendingCalls.set(id, { resolve, reject, notificationCb, timer })

      const line = JSON.stringify(request) + '\n'
      try {
        serverProcess?.stdin?.write(line)
      } catch (err) {
        pendingCalls.delete(id)
        clearTimeout(timer)
        reject(new Error(`Failed to write to Python server: ${err}`))
      }
    })
  },
}
