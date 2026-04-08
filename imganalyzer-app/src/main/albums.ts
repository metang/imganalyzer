/**
 * albums.ts — IPC handlers for smart albums / storyline feature.
 *
 * Delegates to the persistent Python JSON-RPC server for album CRUD
 * and story generation.
 */

import { ipcMain } from 'electron'
import { rpc, ensureServerRunning } from './python-rpc'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface AlbumRule {
  type: string
  [key: string]: unknown
}

export interface AlbumRules {
  match: 'all' | 'any'
  rules: AlbumRule[]
}

export interface SmartAlbum {
  id: string
  name: string
  description: string | null
  cover_image_id: number | null
  rules: AlbumRules
  story_enabled: boolean
  sort_order: string
  item_count: number
  chapter_count: number
  created_at: string
  updated_at: string
}

export interface StoryChapter {
  id: string
  album_id: string
  title: string | null
  summary: string | null
  sort_order: number
  start_date: string | null
  end_date: string | null
  location: string | null
  cover_image_id: number | null
  image_count: number
  moment_count: number
}

export interface StoryMoment {
  id: string
  chapter_id: string
  title: string | null
  sort_order: number
  start_time: string | null
  end_time: string | null
  lat: number | null
  lng: number | null
  hero_image_id: number | null
  image_count: number
}

export interface MomentImage {
  image_id: number
  sort_order: number
  is_hero: number
  date_time_original: string | null
  perception_iaa: number | null
}

export interface StoryGenerateResult {
  images: number
  moments: number
  chapters: number
  generation_time_s: number
  evaluation: {
    album_id: string
    overall_pass: boolean
    criteria: Record<string, {
      name: string
      passed: boolean
      value: unknown
      threshold: unknown
      detail?: string
    }>
  }
}

// ── IPC Registration ──────────────────────────────────────────────────────────

export function registerAlbumHandlers(): void {
  ipcMain.handle('albums:list', async (): Promise<{ albums: SmartAlbum[] }> => {
    await ensureServerRunning()
    return await rpc.call('albums/list', {})
  })

  ipcMain.handle('albums:create', async (_evt, params: {
    name: string
    rules: AlbumRules
    description?: string
    story_enabled?: boolean
    sort_order?: string
  }): Promise<{ id: string; item_count: number }> => {
    await ensureServerRunning()
    return await rpc.call('albums/create', params)
  })

  ipcMain.handle('albums:get', async (_evt, albumId: string): Promise<SmartAlbum | { error: string }> => {
    await ensureServerRunning()
    return await rpc.call('albums/get', { album_id: albumId })
  })

  ipcMain.handle('albums:update', async (_evt, params: {
    album_id: string
    name?: string
    description?: string
    rules?: AlbumRules
    story_enabled?: boolean
    sort_order?: string
  }): Promise<{ id: string; item_count: number } | { error: string }> => {
    await ensureServerRunning()
    return await rpc.call('albums/update', params)
  })

  ipcMain.handle('albums:delete', async (_evt, albumId: string): Promise<{ deleted: boolean }> => {
    await ensureServerRunning()
    return await rpc.call('albums/delete', { album_id: albumId })
  })

  ipcMain.handle('albums:refresh', async (_evt, albumId: string): Promise<{ item_count: number }> => {
    await ensureServerRunning()
    return await rpc.call('albums/refresh', { album_id: albumId })
  })

  ipcMain.handle('albums:story', async (_evt, albumId: string): Promise<{ chapters: StoryChapter[] }> => {
    await ensureServerRunning()
    return await rpc.call('albums/story', { album_id: albumId })
  })

  ipcMain.handle('albums:story:generate', async (_evt, params: {
    album_id: string
    time_window_minutes?: number
    chapter_gap_hours?: number
    chapter_distance_km?: number
    force_year_breaks?: boolean
  }): Promise<StoryGenerateResult> => {
    await ensureServerRunning()
    return await rpc.call('albums/story/generate', params)
  })

  ipcMain.handle('albums:chapter:moments', async (_evt, chapterId: string): Promise<{ moments: StoryMoment[] }> => {
    await ensureServerRunning()
    return await rpc.call('albums/chapter/moments', { chapter_id: chapterId })
  })

  ipcMain.handle('albums:moment:images', async (_evt, momentId: string): Promise<{ images: MomentImage[] }> => {
    await ensureServerRunning()
    return await rpc.call('albums/moment/images', { moment_id: momentId })
  })
}
