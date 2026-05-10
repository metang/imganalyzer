import {
  DEFAULT_PASS_SELECTOR_KEYS,
  PASS_SELECTOR_MODULES,
  WORKER_THREAD_POOL_MODULE_KEYS,
  formatModuleResultLabelList,
} from '../../shared/moduleMetadata'
import type { ModuleKey } from '../../shared/moduleMetadata'

// ── Types ─────────────────────────────────────────────────────────────────────

/**
 * selectedKeys holds unique UI keys (one per row).
 * Call `resolveModuleKeys(selectedKeys)` to get the CLI module keys.
 */
export interface PassSelectorValue {
  selectedKeys: Set<ModuleKey>
  workers: number
  recursive: boolean
  noHash: boolean
  forceReprocess: boolean
}

/**
 * Convert the UI-level selectedKeys set to the CLI module key array.
 * Each UI pass maps to a distinct module key since the split-pass refactor.
 */
export function resolveModuleKeys(selectedKeys: ReadonlySet<ModuleKey>): ModuleKey[] {
  const result = new Set<ModuleKey>()
  for (const pass of PASS_SELECTOR_MODULES) {
    if (selectedKeys.has(pass.uiKey)) {
      result.add(pass.moduleKey)
    }
  }
  return [...result]
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  value: PassSelectorValue
  onChange(value: PassSelectorValue): void
  disabled?: boolean
}

export function PassSelector({ value, onChange, disabled }: Props) {
  const { selectedKeys, workers, recursive, noHash, forceReprocess } = value
  const workerThreadPoolLabel = formatModuleResultLabelList(WORKER_THREAD_POOL_MODULE_KEYS)

  const toggleKey = (uiKey: ModuleKey) => {
    const next = new Set(selectedKeys)
    if (next.has(uiKey)) next.delete(uiKey)
    else next.add(uiKey)
    onChange({ ...value, selectedKeys: next })
  }

  return (
    <div className="flex flex-col gap-5">

      {/* ── Pass checkboxes ─────────────────────────────────────────────────── */}
      <fieldset disabled={disabled} className="flex flex-col gap-1.5">
        <legend className="text-xs font-semibold text-neutral-400 uppercase tracking-wider mb-2">
          Analysis passes
        </legend>
        {PASS_SELECTOR_MODULES.map((pass) => {
          const checked = selectedKeys.has(pass.uiKey)
          return (
            <label
              key={pass.uiKey}
              className="flex items-center gap-2.5 cursor-pointer group select-none"
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => toggleKey(pass.uiKey)}
                className="
                  w-4 h-4 rounded accent-blue-500 cursor-pointer
                  disabled:cursor-not-allowed disabled:opacity-50
                "
              />
              <span className="text-sm text-neutral-300 group-hover:text-neutral-100 transition-colors">
                {pass.label}
              </span>
              {pass.note && (
                <span className="text-xs text-neutral-600 italic">({pass.note})</span>
              )}
            </label>
          )
        })}
      </fieldset>

      {/* ── Options ─────────────────────────────────────────────────────────── */}
      <div className="flex flex-col gap-3">
        <p className="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Options</p>

        {/* Workers slider — local thread pool modules */}
        <div className="flex items-center gap-3">
          <label className="text-sm text-neutral-300 w-36 shrink-0">
            Workers
            <span className="block text-xs text-neutral-600 font-normal">{workerThreadPoolLabel}</span>
          </label>
          <input
            type="range"
            min={1}
            max={16}
            step={1}
            value={workers}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, workers: Number(e.target.value) })}
            className="flex-1 accent-blue-500 disabled:opacity-50"
          />
          <span className="text-sm text-neutral-300 w-5 text-right">{workers}</span>
        </div>

        {/* Recursive toggle */}
        <label className="flex items-center gap-2.5 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={recursive}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, recursive: e.target.checked })}
            className="w-4 h-4 rounded accent-blue-500 disabled:opacity-50"
          />
          <span className="text-sm text-neutral-300">Recursive (include subfolders)</span>
        </label>

        {/* Skip hash toggle */}
        <label className="flex items-center gap-2.5 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={noHash}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, noHash: e.target.checked })}
            className="w-4 h-4 rounded accent-blue-500 disabled:opacity-50"
          />
          <span className="text-sm text-neutral-300">Skip hash computation (faster ingest)</span>
        </label>

        <label className="flex items-center gap-2.5 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={forceReprocess}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, forceReprocess: e.target.checked })}
            className="w-4 h-4 rounded accent-blue-500 disabled:opacity-50"
          />
          <span className="text-sm text-neutral-300">
            Force reprocess selected passes (ignore existing analysis)
          </span>
        </label>
      </div>
    </div>
  )
}

/** Default value — all passes enabled, 2 workers. */
export function defaultPassSelectorValue(): PassSelectorValue {
  return {
    selectedKeys: new Set<ModuleKey>([
      ...DEFAULT_PASS_SELECTOR_KEYS,
    ]),
    workers: 2,
    recursive: true,
    noHash: false,
    forceReprocess: false,
  }
}
