import { useCallback, useEffect, useMemo, useState } from 'react'
import type { HTMLAttributes } from 'react'
import type {
  AppSettings,
  AppSettingsBundle,
  AppSettingsInput,
  CoordinatorStatus,
  WorkerPathMapping,
  WorkerSetupInfo,
} from '../global'

function statusLabel(status: CoordinatorStatus | null): string {
  if (!status) return 'Loading…'
  switch (status.state) {
    case 'running':
      return 'Running'
    case 'starting':
      return 'Starting…'
    case 'error':
      return 'Error'
    default:
      return 'Stopped'
  }
}

function statusDotClass(status: CoordinatorStatus | null): string {
  if (!status) return 'bg-neutral-500'
  switch (status.state) {
    case 'running':
      return 'bg-emerald-500'
    case 'starting':
      return 'bg-yellow-500 animate-pulse'
    case 'error':
      return 'bg-red-500'
    default:
      return 'bg-neutral-500'
  }
}

export function SettingsView() {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [workerSetup, setWorkerSetup] = useState<WorkerSetupInfo | null>(null)
  const [coordinatorStatus, setCoordinatorStatus] = useState<CoordinatorStatus | null>(null)
  const [saving, setSaving] = useState(false)
  const [busyAction, setBusyAction] = useState<'start' | 'stop' | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [copiedField, setCopiedField] = useState<'coordinator-url' | 'worker-command' | null>(null)

  const loadSettings = useCallback(async () => {
    const bundle = await window.api.getAppSettings()
    setSettings(bundle.settings)
    setWorkerSetup(bundle.workerSetup)
  }, [])

  const refreshCoordinatorStatus = useCallback(async () => {
    const status = await window.api.getCoordinatorStatus()
    setCoordinatorStatus(status)
  }, [])

  useEffect(() => {
    void loadSettings().catch((err) => {
      const msg = err instanceof Error ? err.message : String(err)
      setMessage(`Failed to load settings: ${msg}`)
    })
    void refreshCoordinatorStatus().catch((err) => {
      const msg = err instanceof Error ? err.message : String(err)
      setMessage(`Failed to load job server status: ${msg}`)
    })
  }, [loadSettings, refreshCoordinatorStatus])

  useEffect(() => {
    const timer = setInterval(() => {
      void refreshCoordinatorStatus().catch(() => {
        // Keep the UI responsive even if a status poll fails once.
      })
    }, 2000)
    return () => clearInterval(timer)
  }, [refreshCoordinatorStatus])

  useEffect(() => {
    if (!copiedField) return
    const timer = window.setTimeout(() => setCopiedField(null), 1500)
    return () => window.clearTimeout(timer)
  }, [copiedField])

  const applyBundle = useCallback((bundle: AppSettingsBundle) => {
    setSettings(bundle.settings)
    setWorkerSetup(bundle.workerSetup)
  }, [])

  const updateSettings = useCallback((updater: (current: AppSettings) => AppSettings) => {
    setSettings((current) => (current ? updater(current) : current))
  }, [])

  const browseCacheDirectory = useCallback(async () => {
    const directory = await window.api.openFolder()
    if (!directory) return
    updateSettings((current) => ({
      ...current,
      thumbnailCache: {
        ...current.thumbnailCache,
        directory,
      },
    }))
  }, [updateSettings])

  const updatePathMapping = useCallback((index: number, field: keyof WorkerPathMapping, value: string) => {
    updateSettings((current) => ({
      ...current,
      distributed: {
        ...current.distributed,
        workerPathMappings: current.distributed.workerPathMappings.map((mapping, mappingIndex) =>
          mappingIndex === index ? { ...mapping, [field]: value } : mapping),
      },
    }))
  }, [updateSettings])

  const addPathMapping = useCallback(() => {
    updateSettings((current) => ({
      ...current,
      distributed: {
        ...current.distributed,
        workerPathMappings: [
          ...current.distributed.workerPathMappings,
          { sourcePrefix: '', targetPrefix: '' },
        ],
      },
    }))
  }, [updateSettings])

  const removePathMapping = useCallback((index: number) => {
    updateSettings((current) => ({
      ...current,
      distributed: {
        ...current.distributed,
        workerPathMappings: current.distributed.workerPathMappings.filter((_, mappingIndex) => mappingIndex !== index),
      },
    }))
  }, [updateSettings])

  const saveSettings = useCallback(async () => {
    if (!settings) return
    setSaving(true)
    setMessage(null)
    try {
      const input: AppSettingsInput = {
        thumbnailCache: {
          directory: settings.thumbnailCache.directory,
          maxGB: Number(settings.thumbnailCache.maxGB),
        },
        distributed: {
          enabled: settings.distributed.enabled,
          autostart: settings.distributed.autostart,
          bindHost: settings.distributed.bindHost,
          port: Number(settings.distributed.port),
          publicHost: settings.distributed.publicHost,
          authToken: settings.distributed.authToken,
          workerPathMappings: settings.distributed.workerPathMappings
            .filter((mapping) => mapping.sourcePrefix.trim() || mapping.targetPrefix.trim())
            .map((mapping) => ({
              sourcePrefix: mapping.sourcePrefix,
              targetPrefix: mapping.targetPrefix,
            })),
        },
        processing: {
          chunkSize: Number(settings.processing.chunkSize),
        },
      }
      const bundle = await window.api.saveAppSettings(input)
      applyBundle(bundle)
      await refreshCoordinatorStatus()
      setMessage('Settings saved.')
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setMessage(`Failed to save settings: ${msg}`)
    } finally {
      setSaving(false)
    }
  }, [applyBundle, refreshCoordinatorStatus, settings])

  const startCoordinator = useCallback(async () => {
    if (!settings?.distributed.enabled) {
      setMessage('Enable the distributed job server first, then save the settings.')
      return
    }
    setBusyAction('start')
    setMessage(null)
    try {
      const status = await window.api.startCoordinator()
      setCoordinatorStatus(status)
      setMessage('Distributed job server started.')
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setMessage(`Failed to start job server: ${msg}`)
    } finally {
      setBusyAction(null)
    }
  }, [settings])

  const stopCoordinator = useCallback(async () => {
    setBusyAction('stop')
    setMessage(null)
    try {
      const status = await window.api.stopCoordinator()
      setCoordinatorStatus(status)
      setMessage('Distributed job server stopped.')
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setMessage(`Failed to stop job server: ${msg}`)
    } finally {
      setBusyAction(null)
    }
  }, [])

  const generatedWorkerNotes = useMemo(() => workerSetup?.notes ?? [], [workerSetup])

  const copyWorkerText = useCallback(async (
    field: 'coordinator-url' | 'worker-command',
    value: string | null | undefined,
  ) => {
    if (!value) {
      setMessage('Nothing to copy yet.')
      return
    }
    try {
      await window.api.copyText(value)
      setCopiedField(field)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setMessage(`Failed to copy worker setup: ${msg}`)
    }
  }, [])

  if (!settings) {
    return (
      <div className="flex-1 min-h-0 flex items-center justify-center text-sm text-neutral-500">
        Loading settings…
      </div>
    )
  }

  return (
    <div className="flex-1 min-h-0 overflow-y-auto bg-neutral-950">
      <div className="max-w-5xl mx-auto px-6 py-6 flex flex-col gap-6">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h1 className="text-xl font-semibold text-neutral-100">Settings</h1>
            <p className="text-sm text-neutral-400">
              Centralized runtime settings for cache management and distributed job processing.
            </p>
          </div>
          <button
            type="button"
            onClick={saveSettings}
            disabled={saving}
            className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-60"
          >
            {saving ? 'Saving…' : 'Save settings'}
          </button>
        </div>

        {message && (
          <div className="rounded-lg border border-neutral-800 bg-neutral-900 px-4 py-3 text-sm text-neutral-300">
            {message}
          </div>
        )}

        <section className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 flex flex-col gap-4">
          <div>
            <h2 className="text-base font-semibold text-neutral-100">Thumbnail cache</h2>
            <p className="text-sm text-neutral-400">
              Move the gallery cache controls here so storage settings are managed from one place.
            </p>
          </div>

          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_180px]">
            <div className="flex flex-col gap-2">
              <label className="text-xs font-medium uppercase tracking-wide text-neutral-500">
                Cache directory
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={settings.thumbnailCache.directory}
                  onChange={(e) => updateSettings((current) => ({
                    ...current,
                    thumbnailCache: {
                      ...current.thumbnailCache,
                      directory: e.target.value,
                    },
                  }))}
                  className="flex-1 px-3 py-2 rounded-lg bg-neutral-950 border border-neutral-700 text-sm text-neutral-100"
                />
                <button
                  type="button"
                  onClick={() => void browseCacheDirectory().catch((err) => {
                    const msg = err instanceof Error ? err.message : String(err)
                    setMessage(`Failed to choose cache directory: ${msg}`)
                  })}
                  className="px-3 py-2 rounded-lg bg-neutral-800 text-sm text-neutral-200 hover:bg-neutral-700"
                >
                  Browse
                </button>
              </div>
            </div>

            <div className="flex flex-col gap-2">
              <label className="text-xs font-medium uppercase tracking-wide text-neutral-500">
                Max size (GB)
              </label>
              <input
                type="number"
                min={1}
                value={settings.thumbnailCache.maxGB}
                onChange={(e) => updateSettings((current) => ({
                  ...current,
                  thumbnailCache: {
                    ...current.thumbnailCache,
                    maxGB: Number(e.target.value || 0),
                  },
                }))}
                className="px-3 py-2 rounded-lg bg-neutral-950 border border-neutral-700 text-sm text-neutral-100"
              />
            </div>
          </div>

          <p className="text-xs text-neutral-500">
            Source: directory={settings.thumbnailCache.source.directory}, size={settings.thumbnailCache.source.maxGB}
          </p>
        </section>

        {/* ── Processing ──────────────────────────────────────── */}
        <section className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 flex flex-col gap-4">
          <div>
            <h2 className="text-base font-semibold text-neutral-100">Processing</h2>
            <p className="text-sm text-neutral-400">
              Configure batch processing behaviour.
            </p>
          </div>

          <div className="flex flex-col gap-2" style={{ maxWidth: 260 }}>
            <label className="text-xs font-medium uppercase tracking-wide text-neutral-500">
              Chunk size
            </label>
            <input
              type="number"
              min={0}
              max={10000}
              step={100}
              value={settings.processing.chunkSize}
              onChange={(e) => updateSettings((current) => ({
                ...current,
                processing: {
                  ...current.processing,
                  chunkSize: Math.max(0, Math.min(10000, Number(e.target.value) || 0)),
                },
              }))}
              className="px-3 py-2 rounded-lg bg-neutral-950 border border-neutral-700 text-sm text-neutral-100 w-full"
            />
            <p className="text-xs text-neutral-500">
              Process images in chunks of this size so each chunk is fully analyzed before the next.
              Set to 0 to process all images at once (no chunking). Default: 500.
            </p>
          </div>
        </section>

        <section className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 flex flex-col gap-5">
          <div className="flex flex-col gap-2">
            <h2 className="text-base font-semibold text-neutral-100">Distributed job server</h2>
            <p className="text-sm text-neutral-400">
              Configure the HTTP coordinator that remote workers connect to for leased jobs.
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <label className="flex items-start gap-3 rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3">
              <input
                type="checkbox"
                checked={settings.distributed.enabled}
                onChange={(e) => updateSettings((current) => ({
                  ...current,
                  distributed: {
                    ...current.distributed,
                    enabled: e.target.checked,
                    autostart: e.target.checked ? current.distributed.autostart : false,
                  },
                }))}
                className="mt-0.5 accent-blue-500"
              />
              <span className="flex flex-col">
                <span className="text-sm font-medium text-neutral-100">Enable distributed job server</span>
                <span className="text-xs text-neutral-500">Allow this machine to host the coordinator for remote workers.</span>
              </span>
            </label>

            <label className="flex items-start gap-3 rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3">
              <input
                type="checkbox"
                checked={settings.distributed.autostart}
                disabled={!settings.distributed.enabled}
                onChange={(e) => updateSettings((current) => ({
                  ...current,
                  distributed: {
                    ...current.distributed,
                    autostart: e.target.checked,
                  },
                }))}
                className="mt-0.5 accent-blue-500"
              />
              <span className="flex flex-col">
                <span className="text-sm font-medium text-neutral-100">Start with imganalyzer-app</span>
                <span className="text-xs text-neutral-500">Launch the coordinator automatically whenever the desktop app starts.</span>
              </span>
            </label>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Field
              label="Bind host"
              value={settings.distributed.bindHost}
              onChange={(value) => updateSettings((current) => ({
                ...current,
                distributed: {
                  ...current.distributed,
                  bindHost: value,
                },
              }))}
              help="Use 127.0.0.1 for local-only access. Set an auth token before using a LAN IP here."
            />
            <Field
              label="Port"
              value={String(settings.distributed.port)}
              onChange={(value) => updateSettings((current) => ({
                ...current,
                distributed: {
                  ...current.distributed,
                  port: Number(value || 0),
                },
              }))}
              help="Remote workers will connect to this port."
              inputMode="numeric"
            />
            <Field
              label="Worker-visible host"
              value={settings.distributed.publicHost}
              onChange={(value) => updateSettings((current) => ({
                ...current,
                distributed: {
                  ...current.distributed,
                  publicHost: value,
                },
              }))}
              help="Optional hostname or IP shown in worker setup instructions. Leave blank to auto-derive it."
            />
            <Field
              label="Auth token"
              value={settings.distributed.authToken}
              onChange={(value) => updateSettings((current) => ({
                ...current,
                distributed: {
                  ...current.distributed,
                  authToken: value,
                },
              }))}
              help="Required when Bind host is not localhost or 127.0.0.1. If left blank for a LAN bind host, imganalyzer will generate one."
            />
          </div>

          <div className="rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-4 flex flex-col gap-3">
            <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
              <div className="flex items-center gap-2">
                <span className={`inline-block w-2.5 h-2.5 rounded-full ${statusDotClass(coordinatorStatus)}`} />
                <span className="text-sm font-medium text-neutral-100">
                  Status: {statusLabel(coordinatorStatus)}
                </span>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => void startCoordinator()}
                  disabled={saving || busyAction === 'start' || coordinatorStatus?.state === 'starting'}
                  className="px-3 py-2 rounded-lg bg-emerald-600 text-white text-sm hover:bg-emerald-500 disabled:opacity-60"
                >
                  {busyAction === 'start' ? 'Starting…' : 'Start now'}
                </button>
                <button
                  type="button"
                  onClick={() => void stopCoordinator()}
                  disabled={saving || busyAction === 'stop' || coordinatorStatus?.state === 'stopped'}
                  className="px-3 py-2 rounded-lg bg-neutral-800 text-neutral-200 text-sm hover:bg-neutral-700 disabled:opacity-60"
                >
                  {busyAction === 'stop' ? 'Stopping…' : 'Stop'}
                </button>
              </div>
            </div>
            <p className="text-xs text-neutral-500">
              Listen URL: {coordinatorStatus?.url ?? 'Not running'}
            </p>
            {coordinatorStatus?.lastError && (
              <p className="text-xs text-red-400">{coordinatorStatus.lastError}</p>
            )}
          </div>
        </section>

        <section className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 flex flex-col gap-4">
          <div>
            <h2 className="text-base font-semibold text-neutral-100">Worker setup</h2>
            <p className="text-sm text-neutral-400">
              Use this on worker machines to connect back to the coordinator hosted by this app.
            </p>
          </div>

          <div className="rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-4 flex flex-col gap-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Path remapping</p>
                <p className="mt-1 text-sm text-neutral-400">
                  Use these rules when a worker mounts the same NAS share at a different root path.
                </p>
              </div>
              <button
                type="button"
                onClick={addPathMapping}
                className="px-3 py-2 rounded-lg bg-neutral-800 text-sm text-neutral-200 hover:bg-neutral-700"
              >
                Add mapping
              </button>
            </div>

            {settings.distributed.workerPathMappings.length === 0 && (
              <p className="text-sm text-neutral-500">
                No mappings configured. Example: `Z:\photos` on the coordinator to `/Volumes/photos` on macOS.
              </p>
            )}

            <div className="flex flex-col gap-3">
              {settings.distributed.workerPathMappings.map((mapping, index) => (
                <div key={index} className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] items-end">
                  <Field
                    label="Stored path prefix"
                    value={mapping.sourcePrefix}
                    onChange={(value) => updatePathMapping(index, 'sourcePrefix', value)}
                    help="Prefix currently stored in the shared database, such as Z:\\photos."
                  />
                  <Field
                    label="Worker-local prefix"
                    value={mapping.targetPrefix}
                    onChange={(value) => updatePathMapping(index, 'targetPrefix', value)}
                    help="Equivalent path on this worker, such as /Volumes/photos."
                  />
                  <button
                    type="button"
                    onClick={() => removePathMapping(index)}
                    className="px-3 py-2 rounded-lg bg-neutral-800 text-sm text-neutral-200 hover:bg-neutral-700"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3">
            <div className="flex items-start justify-between gap-3">
              <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Coordinator URL</p>
              <button
                type="button"
                onClick={() => void copyWorkerText('coordinator-url', workerSetup?.coordinatorUrl)}
                className="px-3 py-1.5 rounded-lg bg-neutral-800 text-xs text-neutral-200 hover:bg-neutral-700"
              >
                {copiedField === 'coordinator-url' ? 'Copied' : 'Copy'}
              </button>
            </div>
            <p className="mt-1 text-sm text-neutral-200">{workerSetup?.coordinatorUrl ?? 'Unavailable'}</p>
          </div>

          <div className="rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3">
            <div className="flex items-start justify-between gap-3">
              <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Worker command</p>
              <button
                type="button"
                onClick={() => void copyWorkerText('worker-command', workerSetup?.command)}
                className="px-3 py-1.5 rounded-lg bg-neutral-800 text-xs text-neutral-200 hover:bg-neutral-700"
              >
                {copiedField === 'worker-command' ? 'Copied' : 'Copy'}
              </button>
            </div>
            <pre className="mt-2 whitespace-pre-wrap break-words text-sm text-neutral-100">{workerSetup?.command ?? 'Unavailable'}</pre>
          </div>

          <ul className="space-y-2 text-sm text-neutral-300">
            {generatedWorkerNotes.map((note) => (
              <li key={note} className="flex gap-2">
                <span className="text-neutral-500">•</span>
                <span>{note}</span>
              </li>
            ))}
          </ul>
        </section>

        {/* ── Maintenance ──────────────────────────────────────── */}
        <section className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold text-neutral-100">Maintenance</h2>
          </div>
          <RebuildFacesButton />
          <RebuildPerceptionButton />
        </section>
      </div>
    </div>
  )
}

function RebuildFacesButton() {
  const [showConfirm, setShowConfirm] = useState(false)
  const [confirmText, setConfirmText] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  const doRebuild = async () => {
    setShowConfirm(false)
    setConfirmText('')
    setBusy(true)
    setMessage(null)
    try {
      const result = await window.api.rebuildFaces()
      if (result.error) {
        setMessage(result.error)
        return
      }
      if (result.enqueued === 0) {
        setMessage('No images found to rebuild faces for.')
        return
      }
      try {
        await window.api.batchResume()
        setMessage(`${result.enqueued} face-analysis jobs queued and running. Check the Running tab for progress.`)
      } catch {
        setMessage(`${result.enqueued} face-analysis jobs queued. Open the Running tab and resume processing when ready.`)
      }
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="rounded-lg border border-amber-800/40 bg-amber-950/10 px-4 py-4">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-sm font-medium text-amber-200">Rebuild face analysis</p>
          <p className="mt-1 text-xs text-neutral-400">
            Re-enqueue face detection for all images. This is a maintenance action, so it lives here instead of the Faces page.
          </p>
        </div>
        {!showConfirm && (
          <button
            type="button"
            onClick={() => setShowConfirm(true)}
            disabled={busy}
            className="rounded-lg border border-amber-700/50 bg-amber-900/30 px-4 py-1.5 text-sm text-amber-300 transition-colors hover:border-amber-600 hover:bg-amber-800/40 disabled:opacity-50"
          >
            {busy ? 'Queueing…' : 'Rebuild face analysis'}
          </button>
        )}
      </div>

      {showConfirm && (
        <div className="mt-4 flex flex-col gap-3 rounded-lg border border-amber-800/40 bg-neutral-950/60 p-3">
          <p className="text-sm text-amber-200">
            Type <strong>REBUILD</strong> to re-enqueue face analysis on all images:
          </p>
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
            <input
              autoFocus
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && confirmText === 'REBUILD') void doRebuild()
                else if (e.key === 'Escape') { setShowConfirm(false); setConfirmText('') }
              }}
              className="w-full rounded border border-amber-700/60 bg-neutral-900 px-3 py-2 text-sm text-neutral-200 outline-none focus:border-amber-500 sm:w-40"
              placeholder="REBUILD"
            />
            <button
              type="button"
              disabled={confirmText !== 'REBUILD' || busy}
              onClick={() => void doRebuild()}
              className="rounded bg-amber-700 px-3 py-2 text-sm text-white transition-colors enabled:hover:bg-amber-600 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Confirm
            </button>
            <button
              type="button"
              onClick={() => { setShowConfirm(false); setConfirmText('') }}
              className="text-sm text-neutral-500 hover:text-neutral-300"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {message && (
        <p className="mt-3 text-xs text-neutral-300">{message}</p>
      )}
    </div>
  )
}

function RebuildPerceptionButton() {
  const [showConfirm, setShowConfirm] = useState(false)
  const [confirmText, setConfirmText] = useState('')
  const [triggered, setTriggered] = useState(false)

  const doRebuild = async () => {
    setShowConfirm(false)
    setConfirmText('')
    setTriggered(true)
    try {
      await window.api.batchRebuildModule('perception')
    } catch {
      /* handled by batch hook */
    }
  }

  if (triggered) {
    return (
      <p className="text-sm text-emerald-400">
        Rebuild queued — switch to the <strong>Running</strong> tab to see progress.
      </p>
    )
  }

  if (showConfirm) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-sm text-purple-300">
          Type <strong>Confirm</strong> to rebuild perception on all images:
        </span>
        <input
          autoFocus
          value={confirmText}
          onChange={(e) => setConfirmText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && confirmText === 'Confirm') doRebuild()
            else if (e.key === 'Escape') { setShowConfirm(false); setConfirmText('') }
          }}
          className="w-28 rounded border border-purple-700/60 bg-neutral-900 px-2 py-1 text-sm text-neutral-200 outline-none focus:border-purple-500"
          placeholder="Confirm"
        />
        <button
          disabled={confirmText !== 'Confirm'}
          onClick={doRebuild}
          className="rounded bg-purple-700 px-3 py-1 text-sm text-white transition-colors enabled:hover:bg-purple-600 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Go
        </button>
        <button
          onClick={() => { setShowConfirm(false); setConfirmText('') }}
          className="text-sm text-neutral-500 hover:text-neutral-300"
        >
          Cancel
        </button>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-4">
      <button
        onClick={() => setShowConfirm(true)}
        className="rounded-lg border border-purple-700/50 bg-purple-900/30 px-4 py-1.5 text-sm text-purple-300 transition-colors hover:border-purple-600 hover:bg-purple-800/40"
      >
        Rebuild Perception
      </button>
      <span className="text-xs text-neutral-500">
        Re-run IAA / IQA / ISTA analysis on all images
      </span>
    </div>
  )
}

function Field({
  label,
  value,
  onChange,
  help,
  inputMode,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  help: string
  inputMode?: HTMLAttributes<HTMLInputElement>['inputMode']
}) {
  return (
    <label className="flex flex-col gap-2">
      <span className="text-xs font-medium uppercase tracking-wide text-neutral-500">{label}</span>
      <input
        type="text"
        value={value}
        inputMode={inputMode}
        onChange={(e) => onChange(e.target.value)}
        className="px-3 py-2 rounded-lg bg-neutral-950 border border-neutral-700 text-sm text-neutral-100"
      />
      <span className="text-xs text-neutral-500">{help}</span>
    </label>
  )
}
