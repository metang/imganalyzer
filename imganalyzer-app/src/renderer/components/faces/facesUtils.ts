import type { FaceCluster } from '../../global'

export function countActiveUnlinkedClusters(
  clusters: FaceCluster[],
  deferredClusterIds: Set<number>,
): number {
  return clusters.reduce(
    (count, cluster) => (
      cluster.cluster_id !== null
      && !cluster.person_id
      && !deferredClusterIds.has(cluster.cluster_id)
        ? count + 1
        : count
    ),
    0,
  )
}

export function clusterKey(cluster: FaceCluster): string {
  return cluster.cluster_id !== null
    ? `cluster:${cluster.cluster_id}`
    : `name:${cluster.identity_name}`
}

export function appendUniqueClusters(
  existing: FaceCluster[],
  incoming: FaceCluster[],
): FaceCluster[] {
  const seen = new Set(existing.map(clusterKey))
  const appended = incoming.filter((cluster) => !seen.has(clusterKey(cluster)))
  return [...existing, ...appended]
}

export function coerceText(value: unknown): string {
  return typeof value === 'string' ? value : ''
}
