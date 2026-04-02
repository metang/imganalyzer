import { useState, useEffect } from 'react'
import type { GeoStatsExtended } from '../global'
import { GpsGapFiller } from './GpsGapFiller'

interface Props {
  open: boolean
  onClose: () => void
}

function BarChart({ items, maxCount }: { items: Array<{ label: string; count: number }>; maxCount: number }) {
  if (!items.length) return <p className="text-neutral-500 text-xs italic">No data</p>
  return (
    <div className="space-y-1">
      {items.map((item) => (
        <div key={item.label} className="flex items-center gap-2 text-xs">
          <span className="w-28 truncate text-neutral-300 shrink-0" title={item.label}>{item.label}</span>
          <div className="flex-1 h-4 bg-neutral-800 rounded overflow-hidden">
            <div
              className="h-full bg-blue-500/70 rounded"
              style={{ width: `${Math.max((item.count / maxCount) * 100, 2)}%` }}
            />
          </div>
          <span className="w-12 text-right text-neutral-400 tabular-nums shrink-0">{item.count.toLocaleString()}</span>
        </div>
      ))}
    </div>
  )
}

function MiniBarChart({ data, labelKey, valueKey }: {
  data: Array<Record<string, unknown>>
  labelKey: string
  valueKey: string
}) {
  if (!data.length) return <p className="text-neutral-500 text-xs italic">No data</p>
  const maxVal = Math.max(...data.map((d) => d[valueKey] as number))
  return (
    <div className="flex items-end gap-[2px] h-16">
      {data.map((d, i) => {
        const val = d[valueKey] as number
        const label = d[labelKey] as string
        const pct = maxVal > 0 ? (val / maxVal) * 100 : 0
        return (
          <div
            key={i}
            className="flex-1 bg-blue-500/60 rounded-t hover:bg-blue-400/80 transition-colors min-w-[2px]"
            style={{ height: `${Math.max(pct, 3)}%` }}
            title={`${label}: ${val.toLocaleString()}`}
          />
        )
      })}
    </div>
  )
}

function Section({ title, children, defaultOpen = true }: {
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border-b border-neutral-800 pb-3">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 w-full text-left text-xs font-semibold text-neutral-200 uppercase tracking-wider mb-2 hover:text-neutral-100"
      >
        <span className="transition-transform" style={{ transform: open ? 'rotate(90deg)' : 'rotate(0)' }}>▶</span>
        {title}
      </button>
      {open && children}
    </div>
  )
}

export function LocationStatsPanel({ open, onClose }: Props) {
  const [stats, setStats] = useState<GeoStatsExtended | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [gapFillerOpen, setGapFillerOpen] = useState(false)

  const loadStats = () => {
    setLoading(true)
    setError(null)
    window.api.geoStatsExtended()
      .then((result) => {
        if (result.error) setError(result.error)
        else setStats(result)
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    if (!open) return
    loadStats()
  }, [open])

  if (!open) return null

  return (
    <div className="absolute right-0 top-0 bottom-0 w-80 bg-neutral-900/95 backdrop-blur-sm border-l border-neutral-700 z-[1000] flex flex-col shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-neutral-800 shrink-0">
        <h2 className="text-sm font-semibold text-neutral-100">📊 Location Statistics</h2>
        <button
          onClick={onClose}
          className="text-neutral-400 hover:text-neutral-200 text-lg leading-none"
          title="Close"
        >×</button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
        {loading && (
          <div className="flex items-center justify-center py-12">
            <span className="text-neutral-400 text-sm">Loading statistics…</span>
          </div>
        )}
        {error && (
          <div className="text-red-400 text-xs bg-red-900/20 p-2 rounded">{error}</div>
        )}
        {stats && !loading && (
          <>
            {/* GPS Coverage */}
            <Section title="GPS Coverage">
              <div className="flex items-center gap-3">
                <CoverageRing geotagged={stats.geotagged} total={stats.total_images} />
                <div className="text-xs space-y-0.5">
                  <p className="text-neutral-200">
                    <span className="font-semibold text-blue-400">{stats.geotagged.toLocaleString()}</span> geotagged
                  </p>
                  <p className="text-neutral-400">
                    of {stats.total_images.toLocaleString()} total
                  </p>
                  <p className="text-neutral-400">
                    {stats.total_images > 0
                      ? `${Math.round((stats.geotagged / stats.total_images) * 100)}%`
                      : '0%'} coverage
                  </p>
                </div>
              </div>
              {stats.gps_sources.length > 1 && (
                <div className="mt-2 text-xs space-y-0.5">
                  {stats.gps_sources.map((s) => (
                    <div key={s.source} className="flex justify-between text-neutral-400">
                      <span className="capitalize">{s.source}</span>
                      <span className="tabular-nums">{s.count.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              )}
              {stats.total_images > stats.geotagged && (
                <button
                  onClick={() => setGapFillerOpen(true)}
                  className="mt-2 w-full px-3 py-1.5 text-xs bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 border border-blue-600/30 rounded transition-colors"
                >
                  🛰️ Fill GPS gaps ({(stats.total_images - stats.geotagged).toLocaleString()} missing)
                </button>
              )}
            </Section>

            {/* Countries */}
            <Section title={`Countries (${stats.countries.length})`}>
              <BarChart
                items={stats.countries.slice(0, 15).map((c) => ({ label: c.country, count: c.count }))}
                maxCount={stats.countries[0]?.count ?? 1}
              />
            </Section>

            {/* Top Cities */}
            <Section title="Top Cities">
              <BarChart
                items={stats.top_cities.map((c) => ({
                  label: `${c.city}${c.state ? `, ${c.state}` : ''}`,
                  count: c.count,
                }))}
                maxCount={stats.top_cities[0]?.count ?? 1}
              />
            </Section>

            {/* Top Locations */}
            <Section title="Top 10 Locations">
              <div className="space-y-1">
                {stats.top_locations.map((loc, i) => (
                  <div key={loc.cell} className="flex items-center justify-between text-xs">
                    <span className="text-neutral-300 truncate">
                      {i + 1}. {loc.city ?? loc.state ?? loc.country ?? `${loc.lat.toFixed(2)}, ${loc.lng.toFixed(2)}`}
                    </span>
                    <span className="text-neutral-400 tabular-nums ml-2 shrink-0">{loc.count.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </Section>

            {/* Monthly Activity */}
            <Section title="Monthly Activity">
              <MiniBarChart data={stats.monthly_activity} labelKey="month" valueKey="count" />
              {stats.monthly_activity.length > 0 && (
                <div className="flex justify-between text-[10px] text-neutral-500 mt-1">
                  <span>{stats.monthly_activity[0].month}</span>
                  <span>{stats.monthly_activity[stats.monthly_activity.length - 1].month}</span>
                </div>
              )}
            </Section>

            {/* Location Diversity */}
            <Section title="Location Diversity">
              <MiniBarChart data={stats.location_diversity} labelKey="month" valueKey="unique_places" />
              <p className="text-[10px] text-neutral-500 mt-1">Unique ~10km areas visited per month</p>
            </Section>

            {/* Camera × Country */}
            {stats.camera_by_country.length > 0 && (
              <Section title="Cameras by Country" defaultOpen={false}>
                <CameraByCountryTable data={stats.camera_by_country} />
              </Section>
            )}

            {/* Furthest from Home */}
            {stats.furthest_from_home && (
              <Section title="Furthest from Home">
                <div className="text-xs space-y-1">
                  <p className="text-neutral-200 font-medium">
                    {stats.furthest_from_home.distance_km.toLocaleString()} km away
                  </p>
                  <p className="text-neutral-400 truncate" title={stats.furthest_from_home.file_path}>
                    {stats.furthest_from_home.file_path.split(/[\\/]/).pop()}
                  </p>
                </div>
              </Section>
            )}
          </>
        )}
      </div>

      {/* GPS Gap Filler modal */}
      <GpsGapFiller
        open={gapFillerOpen}
        onClose={() => setGapFillerOpen(false)}
        onApplied={loadStats}
      />
    </div>
  )
}

function CoverageRing({ geotagged, total }: { geotagged: number; total: number }) {
  const pct = total > 0 ? (geotagged / total) * 100 : 0
  const radius = 28
  const circ = 2 * Math.PI * radius
  const offset = circ - (pct / 100) * circ
  return (
    <svg width="72" height="72" viewBox="0 0 72 72" className="shrink-0">
      <circle cx="36" cy="36" r={radius} fill="none" stroke="#333" strokeWidth="6" />
      <circle
        cx="36" cy="36" r={radius}
        fill="none" stroke="#3b82f6" strokeWidth="6"
        strokeLinecap="round"
        strokeDasharray={circ}
        strokeDashoffset={offset}
        transform="rotate(-90 36 36)"
      />
      <text x="36" y="36" textAnchor="middle" dominantBaseline="central"
        className="fill-neutral-200 text-xs font-semibold"
      >
        {Math.round(pct)}%
      </text>
    </svg>
  )
}

function CameraByCountryTable({ data }: { data: Array<{ country: string; camera: string; count: number }> }) {
  // Group by country
  const grouped = new Map<string, Array<{ camera: string; count: number }>>()
  for (const row of data) {
    const arr = grouped.get(row.country) ?? []
    arr.push({ camera: row.camera, count: row.count })
    grouped.set(row.country, arr)
  }
  return (
    <div className="space-y-2">
      {Array.from(grouped.entries()).slice(0, 10).map(([country, cameras]) => (
        <div key={country}>
          <p className="text-xs font-medium text-neutral-300 mb-0.5">{country}</p>
          {cameras.slice(0, 5).map((c) => (
            <div key={c.camera} className="flex justify-between text-[11px] text-neutral-400 pl-2">
              <span className="truncate">{c.camera}</span>
              <span className="tabular-nums ml-2 shrink-0">{c.count.toLocaleString()}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}
