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
              help="Defaults to the detected LAN IP so other devices on the same network can reach the coordinator."
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
              help="Recommended before exposing the coordinator beyond localhost."
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
            <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Coordinator URL</p>
            <p className="mt-1 text-sm text-neutral-200">{workerSetup?.coordinatorUrl ?? 'Unavailable'}</p>
          </div>

          <div className="rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3">
            <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Worker command</p>
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
      </div>
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
