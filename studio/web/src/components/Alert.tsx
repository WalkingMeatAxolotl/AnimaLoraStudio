import type { HTMLAttributes, ReactNode } from 'react'

export type AlertTone = 'info' | 'success' | 'warning' | 'danger'
export type AlertSize = 'md' | 'sm'

export interface AlertProps extends Omit<HTMLAttributes<HTMLDivElement>, 'children' | 'title'> {
  tone?: AlertTone
  size?: AlertSize
  title?: ReactNode
  action?: ReactNode
  icon?: ReactNode | false
  children: ReactNode
}

const TONE_CLASS: Record<AlertTone, string> = {
  info: 'alert-info',
  success: 'alert-success',
  warning: 'alert-warning',
  danger: 'alert-danger',
}

const SIZE_CLASS: Record<AlertSize, string> = {
  md: '',
  sm: 'alert-sm',
}

export function alertClassName({
  tone = 'info',
  size = 'md',
  className = '',
}: {
  tone?: AlertTone
  size?: AlertSize
  className?: string
} = {}): string {
  return [
    'alert',
    TONE_CLASS[tone],
    SIZE_CLASS[size],
    className,
  ].filter(Boolean).join(' ')
}

function AlertIcon({ tone }: { tone: AlertTone }) {
  if (tone === 'success') {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="9" />
        <path d="m8.5 12 2.2 2.2 4.8-5" />
      </svg>
    )
  }

  if (tone === 'warning') {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10.3 4.2 2.7 17.4A1.8 1.8 0 0 0 4.3 20h15.4a1.8 1.8 0 0 0 1.6-2.6L13.7 4.2a2 2 0 0 0-3.4 0Z" />
        <path d="M12 9v4" />
        <path d="M12 17h.01" />
      </svg>
    )
  }

  if (tone === 'danger') {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="9" />
        <path d="M12 8v5" />
        <path d="M12 17h.01" />
      </svg>
    )
  }

  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5" />
      <path d="M12 8h.01" />
    </svg>
  )
}

export default function Alert({
  tone = 'info',
  size = 'md',
  title,
  action,
  icon,
  className,
  children,
  ...rest
}: AlertProps) {
  const resolvedIcon = icon === false ? null : (icon ?? <AlertIcon tone={tone} />)

  return (
    <div {...rest} className={alertClassName({ tone, size, className })}>
      {resolvedIcon && (
        <span className="alert-icon" aria-hidden="true">
          {resolvedIcon}
        </span>
      )}
      <div className="alert-content">
        {title && <div className="alert-title">{title}</div>}
        <div className="alert-message">{children}</div>
      </div>
      {action && <div className="alert-action">{action}</div>}
    </div>
  )
}
