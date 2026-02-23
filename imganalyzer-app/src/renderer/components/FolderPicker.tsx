

interface FolderPickerProps {
  folderPath: string | null
  onFolderChange: (path: string) => void
}

export function FolderPicker({ folderPath, onFolderChange }: FolderPickerProps) {
  async function handleClick() {
    const result = await window.api.openFolder()
    if (result) onFolderChange(result)
  }

  return (
    <div className="flex items-center gap-3 px-4 py-3 bg-neutral-900 border-b border-neutral-800">
      <button
        onClick={handleClick}
        className="flex items-center gap-2 px-3 py-1.5 bg-neutral-800 hover:bg-neutral-700 rounded text-sm font-medium transition-colors"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round"
            d="M2.25 12.75V12A2.25 2.25 0 014.5 9.75h15A2.25 2.25 0 0121.75 12v.75m-8.69-6.44l-2.12-2.12a1.5 1.5 0 00-1.061-.44H4.5A2.25 2.25 0 002.25 6v12a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9a2.25 2.25 0 00-2.25-2.25h-5.379a1.5 1.5 0 01-1.06-.44z" />
        </svg>
        Open Folder
      </button>
      {folderPath && (
        <span className="text-neutral-400 text-sm truncate max-w-lg" title={folderPath}>
          {folderPath}
        </span>
      )}
      {!folderPath && (
        <span className="text-neutral-600 text-sm">No folder selected</span>
      )}
    </div>
  )
}
