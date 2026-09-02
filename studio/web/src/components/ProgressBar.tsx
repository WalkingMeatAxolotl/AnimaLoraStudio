import type { HTMLAttributes } from 'react'

export type ProgressBarSize = 'xs' | 'sm' | 'md'
export type ProgressBarTone = 'accent' | 'success'

export interface ProgressBarProps extends Omit<HTMLAttributes<HTMLDivElement>, 'children' | 'role'> {
  /** Accessible name for the operation whose progress is shown. */
  label: string
  /** Omit or pass null when the amount of work is unknown. */
  value?: number | null
  max?: number
  valueText?: string
  size?: ProgressBarSize
  tone?: ProgressBarTone
}

const SIZE_CLASS: Record<ProgressBarSize, string> = {
  xs: 'ui-progress-xs',
  sm: 'ui-progress-sm',
  md: 'ui-progress-md',
}

const TONE_CLASS: Record<ProgressBarTone, string> = {
  accent: 'ui-progress-accent',
  success: 'ui-progress-success',
}

export default function ProgressBar({
  label,
  value,
  max = 100,
  valueText,
  size = 'sm',
  tone = 'accent',
  className = '',
  ...rest
}: ProgressBarProps) {
  const safeMax = Number.isFinite(max) && max > 0 ? max : 100
  const determinate = value != null && Number.isFinite(value)
  const safeValue = determinate
    ? Math.min(safeMax, Math.max(0, value))
    : undefined
  const percentage = safeValue == null ? undefined : (safeValue / safeMax) * 100

  return (
    <div
      {...rest}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={safeMax}
      aria-valuenow={safeValue}
      aria-valuetext={valueText}
      data-state={determinate ? 'determinate' : 'indeterminate'}
      className={[
        'ui-progress',
        SIZE_CLASS[size],
        TONE_CLASS[tone],
        className,
      ].filter(Boolean).join(' ')}
    >
      <div
        className="ui-progress-fill"
        style={percentage == null ? undefined : { transform: `scaleX(${percentage / 100})` }}
        aria-hidden="true"
      />
    </div>
  )
}
