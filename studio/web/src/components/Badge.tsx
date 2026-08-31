import type { HTMLAttributes, ReactNode } from 'react'

export type BadgeTone = 'neutral' | 'accent' | 'info' | 'success' | 'warning' | 'danger'
export type BadgeSize = 'md' | 'sm'

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone
  size?: BadgeSize
  /** Active work receives the shared pulsing status indicator. */
  active?: boolean
  children: ReactNode
}

const TONE_CLASS: Record<BadgeTone, string> = {
  neutral: 'badge-neutral',
  accent: 'badge-accent',
  info: 'badge-info',
  success: 'badge-ok',
  warning: 'badge-warn',
  danger: 'badge-err',
}

export default function Badge({
  tone = 'neutral',
  size = 'md',
  active = false,
  className = '',
  children,
  ...rest
}: BadgeProps) {
  const classes = [
    'badge',
    TONE_CLASS[tone],
    size === 'sm' && 'badge-sm',
    className,
  ].filter(Boolean).join(' ')

  return (
    <span {...rest} className={classes}>
      {active && <span className="dot dot-running" aria-hidden="true" />}
      {children}
    </span>
  )
}
