import { useState } from 'react'

interface Props {
  onConfirm(): void
  onCancel(): void
}

/**
 * Modal dialog requiring the user to type the exact word "STOP" (uppercase)
 * before the confirm button becomes enabled.
 */
export function ConfirmStopDialog({ onConfirm, onCancel }: Props) {
  const [value, setValue] = useState('')
  const confirmed = value === 'STOP'

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="bg-neutral-900 border border-neutral-700 rounded-xl shadow-2xl w-96 p-6 flex flex-col gap-4">
        <h2 className="text-base font-semibold text-neutral-100">Stop batch processing?</h2>
        <p className="text-sm text-neutral-400 leading-relaxed">
          This will kill the active worker and clear all pending and running jobs for this folder
          from the queue. Completed jobs are unaffected.
        </p>
        <p className="text-sm text-neutral-400">
          Type <span className="font-mono font-bold text-red-400">STOP</span> to confirm:
        </p>
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="STOP"
          autoFocus
          className="
            w-full px-3 py-2 rounded-lg bg-neutral-800 border border-neutral-700
            text-sm text-neutral-100 placeholder-neutral-600
            focus:outline-none focus:border-red-500
            font-mono
          "
        />
        <div className="flex gap-3 justify-end mt-1">
          <button
            onClick={onCancel}
            className="
              px-4 py-2 rounded-lg text-sm text-neutral-300
              hover:bg-neutral-800 transition-colors
            "
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={!confirmed}
            className="
              px-4 py-2 rounded-lg text-sm font-medium transition-colors
              bg-red-700 text-white
              disabled:opacity-30 disabled:cursor-not-allowed
              enabled:hover:bg-red-600
            "
          >
            Stop
          </button>
        </div>
      </div>
    </div>
  )
}
