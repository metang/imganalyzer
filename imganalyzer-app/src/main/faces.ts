/**
 * faces.ts — IPC handlers for face identity management.
 *
 * Delegates to the persistent Python JSON-RPC server for listing face
 * identities/clusters, fetching images per face, setting display names,
 * cropping face thumbnails, and running clustering.
 */

import { ipcMain } from 'electron'
import { rpc, ensureServerRunning } from './python-rpc'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface FaceSummary {
  canonical_name: string
  display_name: string | null
  image_count: number
  identity_id: number | null
}

export interface FaceImage {
  image_id: number
  file_path: string
  face_count: number
}

export interface FaceCluster {
  cluster_id: number | null
  identity_name: string
  display_name: string | null
  identity_id: number | null
  image_count: number
  face_count: number
  representative_id: number | null
  person_id: number | null
}

export interface FaceOccurrence {
  id: number
  image_id: number
  file_path: string
  face_idx: number
  bbox_x1: number
  bbox_y1: number
  bbox_x2: number
  bbox_y2: number
  age: number | null
  gender: string | null
  identity_name: string
}

export interface FacePerson {
  id: number
  name: string
  notes: string | null
  cluster_count: number
  face_count: number
  image_count: number
  representative_id: number | null
}

export interface PersonCluster {
  cluster_id: number
  face_count: number
  image_count: number
  label: string
  representative_id: number | null
}

// ── IPC Registration ──────────────────────────────────────────────────────────

export function registerFaceHandlers(): void {
  ipcMain.handle('faces:list', async (): Promise<{ faces: FaceSummary[]; error?: string }> => {
    try {
      await ensureServerRunning()
      const result = await rpc.call('faces/list', {}) as { faces: FaceSummary[] }
      return { faces: result.faces }
    } catch (err) {
      return { faces: [], error: String(err) }
    }
  })

  ipcMain.handle(
    'faces:images',
    async (_evt, name: string, limit?: number): Promise<{ images: FaceImage[]; error?: string }> => {
      try {
        await ensureServerRunning()
        const result = await rpc.call('faces/images', { name, limit: limit ?? 100 }) as {
          images: FaceImage[]
        }
        return { images: result.images }
      } catch (err) {
        return { images: [], error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'faces:setAlias',
    async (_evt, canonicalName: string, displayName: string, clusterId?: number | null): Promise<{ ok: boolean; error?: string }> => {
      try {
        await ensureServerRunning()
        const rpcParams: Record<string, unknown> = {
          canonical_name: canonicalName,
          display_name: displayName,
        }
        if (clusterId != null) {
          rpcParams.cluster_id = clusterId
        }
        await rpc.call('faces/set-alias', rpcParams)
        return { ok: true }
      } catch (err) {
        return { ok: false, error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'faces:clusters',
    async (): Promise<{ clusters: FaceCluster[]; has_occurrences: boolean; error?: string }> => {
      try {
        await ensureServerRunning()
        const result = await rpc.call('faces/clusters', {}) as {
          clusters: FaceCluster[]
          has_occurrences: boolean
        }
        return { clusters: result.clusters, has_occurrences: result.has_occurrences }
      } catch (err) {
        return { clusters: [], has_occurrences: false, error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'faces:clusterImages',
    async (
      _evt,
      clusterId: number | null,
      identityName: string | null,
      limit?: number
    ): Promise<{ occurrences: FaceOccurrence[]; error?: string }> => {
      try {
        await ensureServerRunning()
        const result = await rpc.call('faces/cluster-images', {
          cluster_id: clusterId,
          identity_name: identityName,
          limit: limit ?? 50,
        }) as { occurrences: FaceOccurrence[] }
        return { occurrences: result.occurrences }
      } catch (err) {
        return { occurrences: [], error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'faces:crop',
    async (_evt, occurrenceId: number): Promise<{ data?: string; error?: string }> => {
      try {
        await ensureServerRunning()
        const result = await rpc.call('faces/crop', {
          occurrence_id: occurrenceId,
        }) as { data?: string; error?: string }
        if (result.error) {
          return { error: result.error }
        }
        return { data: result.data }
      } catch (err) {
        return { error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'faces:cropBatch',
    async (_evt, ids: number[]): Promise<{ thumbnails: Record<string, string>; error?: string }> => {
      try {
        await ensureServerRunning()
        const result = await rpc.call('faces/crop-batch', { ids }) as {
          thumbnails: Record<string, string>
        }
        return { thumbnails: result.thumbnails }
      } catch (err) {
        return { thumbnails: {}, error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'faces:runClustering',
    async (_evt, threshold?: number): Promise<{ num_clusters: number; error?: string }> => {
      try {
        await ensureServerRunning()
        const result = await rpc.call('faces/run-clustering', {
          threshold: threshold ?? 0.55,
        }, undefined, 300_000) as { num_clusters: number }
        return { num_clusters: result.num_clusters }
      } catch (err) {
        return { num_clusters: 0, error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'faces:rebuild',
    async (): Promise<{ enqueued: number; error?: string }> => {
      try {
        await ensureServerRunning()
        const result = await rpc.call('rebuild', {
          module: 'faces',
          force: true,
        }) as { enqueued: number }
        return { enqueued: result.enqueued }
      } catch (err) {
        return { enqueued: 0, error: String(err) }
      }
    }
  )

  // ── Person (cross-age identity grouping) ────────────────────────────────

  ipcMain.handle(
    'faces:persons',
    async (): Promise<{ persons: FacePerson[]; error?: string }> => {
      try {
        await ensureServerRunning()
        const result = await rpc.call('faces/persons', {}) as { persons: FacePerson[] }
        return { persons: result.persons }
      } catch (err) {
        return { persons: [], error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'faces:personCreate',
    async (_evt, name: string): Promise<{ id: number; error?: string }> => {
      try {
        await ensureServerRunning()
        const result = await rpc.call('faces/person-create', { name }) as { id: number }
        return { id: result.id }
      } catch (err) {
        return { id: 0, error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'faces:personRename',
    async (_evt, personId: number, name: string): Promise<{ ok: boolean; error?: string }> => {
      try {
        await ensureServerRunning()
        await rpc.call('faces/person-rename', { person_id: personId, name })
        return { ok: true }
      } catch (err) {
        return { ok: false, error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'faces:personDelete',
    async (_evt, personId: number): Promise<{ ok: boolean; error?: string }> => {
      try {
        await ensureServerRunning()
        await rpc.call('faces/person-delete', { person_id: personId })
        return { ok: true }
      } catch (err) {
        return { ok: false, error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'faces:personLinkCluster',
    async (_evt, clusterId: number, personId: number): Promise<{ ok: boolean; updated: number; error?: string }> => {
      try {
        await ensureServerRunning()
        const result = await rpc.call('faces/person-link-cluster', {
          cluster_id: clusterId,
          person_id: personId,
        }) as { ok: boolean; updated: number }
        return { ok: true, updated: result.updated }
      } catch (err) {
        return { ok: false, updated: 0, error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'faces:personUnlinkCluster',
    async (_evt, clusterId: number): Promise<{ ok: boolean; updated: number; error?: string }> => {
      try {
        await ensureServerRunning()
        const result = await rpc.call('faces/person-unlink-cluster', {
          cluster_id: clusterId,
        }) as { ok: boolean; updated: number }
        return { ok: true, updated: result.updated }
      } catch (err) {
        return { ok: false, updated: 0, error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'faces:personClusters',
    async (_evt, personId: number): Promise<{ clusters: PersonCluster[]; error?: string }> => {
      try {
        await ensureServerRunning()
        const result = await rpc.call('faces/person-clusters', {
          person_id: personId,
        }) as { clusters: PersonCluster[] }
        return { clusters: result.clusters }
      } catch (err) {
        return { clusters: [], error: String(err) }
      }
    }
  )
}
