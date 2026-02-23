import { contextBridge, ipcRenderer } from 'electron'
import type { XmpData } from '../main/xmp'
import type { ImageFile } from '../main/images'
import type { AnalysisProgress } from '../main/analyzer'

contextBridge.exposeInMainWorld('api', {
  openFolder: (): Promise<string | null> =>
    ipcRenderer.invoke('dialog:openFolder'),

  listImages: (folderPath: string): Promise<ImageFile[]> =>
    ipcRenderer.invoke('fs:listImages', folderPath),

  getThumbnail: (imagePath: string): Promise<string> =>
    ipcRenderer.invoke('fs:getThumbnail', imagePath),

  readXmp: (imagePath: string): Promise<XmpData | null> =>
    ipcRenderer.invoke('fs:readXmp', imagePath),

  runAnalysis: (imagePath: string, aiBackend: string): Promise<{ xmp: XmpData | null; error?: string }> =>
    ipcRenderer.invoke('analyze:run', imagePath, aiBackend),

  cancelAnalysis: (imagePath: string): Promise<void> =>
    ipcRenderer.invoke('analyze:cancel', imagePath),

  onAnalysisProgress: (cb: (p: AnalysisProgress) => void) => {
    const handler = (_evt: Electron.IpcRendererEvent, p: AnalysisProgress) => cb(p)
    ipcRenderer.on('analyze:progress', handler)
    return () => ipcRenderer.removeListener('analyze:progress', handler)
  }
})
