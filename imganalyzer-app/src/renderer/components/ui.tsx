import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from 'react'

function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ')
}

type Tone = 'default' | 'subtle' | 'accent' | 'danger'

export function SurfaceCard({
  children,
  className,
  tone = 'default',
  ...props
}: HTMLAttributes<HTMLElement> & {
  children: ReactNode
  tone?: Tone
}) {
  const toneClass =
    tone === 'accent'
      ? 'border-cyan-500/25 bg-slate-950/80 shadow-[0_8px_30px_rgba(8,145,178,0.12)]'
      : tone === 'danger'
        ? 'border-red-800/70 bg-red-950/15'
        : tone === 'subtle'
          ? 'border-neutral-800/80 bg-neutral-900/35'
          : 'border-neutral-800 bg-neutral-950/70'

  return (
    <section
      className={cx('rounded-2xl border p-4 backdrop-blur-sm', toneClass, className)}
      {...props}
    >
      {children}
    </section>
  )
}

export function SectionHeading({
  eyebrow,
  title,
  description,
  actions,
  className,
}: {
  eyebrow?: string
  title: string
  description?: string
  actions?: ReactNode
  className?: string
}) {
  return (
    <div className={cx('flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between', className)}>
      <div className="min-w-0">
        {eyebrow && (
          <p className="mb-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-cyan-300/80">
            {eyebrow}
          </p>
        )}
        <h2 className="text-base font-semibold text-white">{title}</h2>
        {description && <p className="mt-1 text-sm text-neutral-400">{description}</p>}
      </div>
      {actions ? <div className="shrink-0">{actions}</div> : null}
    </div>
  )
}

export function StatusBadge({
  children,
  tone = 'default',
  className,
}: {
  children: ReactNode
  tone?: 'default' | 'info' | 'success' | 'warning' | 'danger'
  className?: string
}) {
  const toneClass =
    tone === 'info'
      ? 'border-cyan-500/30 bg-cyan-500/10 text-cyan-100'
      : tone === 'success'
        ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-100'
        : tone === 'warning'
          ? 'border-amber-500/30 bg-amber-500/10 text-amber-100'
          : tone === 'danger'
            ? 'border-red-500/30 bg-red-500/10 text-red-100'
            : 'border-neutral-700 bg-neutral-900 text-neutral-200'

  return (
    <span className={cx('inline-flex items-center rounded-full border px-2.5 py-1 text-[11px]', toneClass, className)}>
      {children}
    </span>
  )
}

export function UiButton({
  className,
  variant = 'secondary',
  size = 'md',
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'sm' | 'md'
}) {
  const variantClass =
    variant === 'primary'
      ? 'border border-cyan-500/60 bg-cyan-500 text-slate-950 hover:bg-cyan-400'
      : variant === 'danger'
        ? 'border border-red-700/80 bg-red-950/40 text-red-200 hover:bg-red-900/60'
        : variant === 'ghost'
          ? 'border border-neutral-700 bg-transparent text-neutral-300 hover:border-neutral-500 hover:text-white'
          : 'border border-neutral-700 bg-neutral-900 text-neutral-200 hover:border-neutral-500 hover:bg-neutral-800'

  const sizeClass = size === 'sm' ? 'px-3 py-1.5 text-xs' : 'px-4 py-2 text-sm'

  return (
    <button
      className={cx(
        'rounded-xl font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50',
        variantClass,
        sizeClass,
        className,
      )}
      {...props}
    />
  )
}

export function MetricCard({
  label,
  value,
  hint,
  tone = 'default',
}: {
  label: string
  value: string
  hint?: string
  tone?: Tone
}) {
  const toneClass =
    tone === 'accent'
      ? 'border-cyan-500/20 bg-cyan-500/5'
      : tone === 'danger'
        ? 'border-red-800/70 bg-red-950/20'
        : tone === 'subtle'
          ? 'border-neutral-800/70 bg-black/15'
          : 'border-neutral-800 bg-neutral-900/50'

  return (
    <div className={cx('rounded-2xl border px-3 py-3', toneClass)}>
      <div className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">{label}</div>
      <div className="mt-1 font-mono text-base text-neutral-100">{value}</div>
      {hint && <div className="mt-1 text-xs text-neutral-500">{hint}</div>}
    </div>
  )
}

export { cx }
