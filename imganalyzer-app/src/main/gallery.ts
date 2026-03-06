import { ipcMain } from 'electron'
import { ensureServerRunning, rpc } from './python-rpc'
import type { SearchResult } from './search'

export interface GalleryFolderNode {
  path: string
  name: string
  parent_path: string | null
  depth: number
  image_count: number
  child_count: number
}

export interface GalleryFoldersResponse {
  folders: GalleryFolderNode[]
  totalImages: number
  error?: string
}

export interface GalleryChunkParams {
  folderPath?: string | null
  recursive?: boolean
  chunkSize?: number
  cursor?: string | null
}

export interface GalleryChunkResponse {
  items: SearchResult[]
  nextCursor: string | null
  hasMore: boolean
  total: number | null
  error?: string
}

export function registerGalleryHandlers(): void {
  ipcMain.handle('gallery:list-folders', async (): Promise<GalleryFoldersResponse> => {
    try {
      await ensureServerRunning()
      const result = await rpc.call('gallery/listFolders', {}) as {
        folders: GalleryFolderNode[]
        totalImages: number
      }
      return {
        folders: result.folders,
        totalImages: result.totalImages,
      }
    } catch (err) {
      return { folders: [], totalImages: 0, error: String(err) }
    }
  })

  ipcMain.handle(
    'gallery:list-images-chunk',
    async (_evt, params: GalleryChunkParams): Promise<GalleryChunkResponse> => {
      try {
        await ensureServerRunning()
        const result = await rpc.call('gallery/listImagesChunk', {
          folderPath: params.folderPath ?? null,
          recursive: params.recursive ?? true,
          chunkSize: params.chunkSize ?? 300,
          cursor: params.cursor ?? null,
        }) as {
          items: SearchResult[]
          nextCursor: string | null
          hasMore: boolean
          total: number | null
        }

        return {
          items: result.items,
          nextCursor: result.nextCursor,
          hasMore: result.hasMore,
          total: result.total,
        }
      } catch (err) {
        return { items: [], nextCursor: null, hasMore: false, total: null, error: String(err) }
      }
    }
  )
}
