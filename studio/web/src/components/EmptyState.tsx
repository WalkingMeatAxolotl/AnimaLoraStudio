import type { HTMLAttributes, ReactNode } from 'react'
import Card from './Card'

export type EmptyStateSize = 'md' | 'sm'

export interface EmptyStateProps extends Omit<HTMLAttributes<HTMLElement>, 'children' | 'title'> {
  title?: ReactNode
  description: ReactNode
  action?: ReactNode
  size?: EmptyStateSize
}

export default function EmptyState({
  title,
  description,
  action,
  size = 'md',
  className = '',
  ...rest
}: EmptyStateProps) {
  const classes = [
    'empty-state',
    size === 'sm' && 'empty-state-sm',
    className,
  ].filter(Boolean).join(' ')

  return (
    <Card {...rest} className={classes}>
      {title && <p className="empty-state-title">{title}</p>}
      <p className="empty-state-description">{description}</p>
      {action && <div className="empty-state-action">{action}</div>}
    </Card>
  )
}
