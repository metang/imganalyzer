import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { readFile } from 'fs/promises'
import { existsSync } from 'fs'
import { listImages, getThumbnail, getFullImage } from './images'
import { parseXmp } from './xmp'
import { runAnalysis, cancelAnalysis } from './analyzer'
import { runCopilotAnalysis } from './copilot-analyzer'

function createWindow(): void {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    backgroundColor: '#111111',
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false // allow loading local file:// images in renderer
    },
    show: false
  })

  win.once('ready-to-show', () => win.show())

  if (process.env.ELECTRON_RENDERER_URL) {
    win.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    win.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

app.whenReady().then(() => {
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

// ─── IPC: Folder dialog ──────────────────────────────────────────────────────
ipcMain.handle('dialog:openFolder', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'],
    title: 'Select image folder'
  })
  return result.canceled ? null : result.filePaths[0]
})

// ─── IPC: List images in folder ───────────────────────────────────────────────
ipcMain.handle('fs:listImages', async (_evt, folderPath: string) => {
  return listImages(folderPath)
})

// ─── IPC: Get thumbnail ───────────────────────────────────────────────────────
ipcMain.handle('fs:getThumbnail', async (_evt, imagePath: string) => {
  return getThumbnail(imagePath)
})

// ─── IPC: Get full-resolution image for lightbox ──────────────────────────────
ipcMain.handle('fs:getFullImage', async (_evt, imagePath: string) => {
  return getFullImage(imagePath)
})

// ─── IPC: Read XMP sidecar ───────────────────────────────────────────────────
ipcMain.handle('fs:readXmp', async (_evt, imagePath: string) => {
  const xmpPath = imagePath.replace(/\.[^.]+$/, '.xmp')
  if (!existsSync(xmpPath)) return null
  try {
    const xml = await readFile(xmpPath, 'utf-8')
    return parseXmp(xml)
  } catch {
    return null
  }
})

// ─── IPC: Run analysis ───────────────────────────────────────────────────────
ipcMain.handle('analyze:run', async (evt, imagePath: string, aiBackend: string) => {
  return runAnalysis(imagePath, aiBackend, (progress) => {
    evt.sender.send('analyze:progress', progress)
  })
})

ipcMain.handle('analyze:cancel', async (_evt, imagePath: string) => {
  cancelAnalysis(imagePath)
})

// ─── IPC: Cloud analysis via GitHub Copilot SDK ───────────────────────────
ipcMain.handle('analyze:copilot', async (_evt, imagePath: string) => {
  return runCopilotAnalysis(imagePath)
})
