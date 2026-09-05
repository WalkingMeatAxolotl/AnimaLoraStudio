import type { HTMLAttributes, ReactNode } from 'react'
import Card from './Card'

export type EmptyStateSize = 'md' | 'sm'

export interface EmptyStateProps extends Omit<HTMLAttributes<HTMLElement>, 'children' | 'title'> {
  title?: ReactNode
  description: ReactNode
  action?: ReactNode
  size?: EmptyStateSize
  /** Use inside an existing panel without adding a nested card surface. */
  embedded?: boolean
}

export default function EmptyState({
  title,
  description,
  action,
  size = 'md',
  embedded = false,
  className = '',
  ...rest
}: EmptyStateProps) {
  const classes = [
    'empty-state',
    size === 'sm' && 'empty-state-sm',
    className,
  ].filter(Boolean).join(' ')

  const Container = embedded ? 'div' : Card
  return (
    <Container {...rest} className={classes}>
      {title && <p className="empty-state-title">{title}</p>}
      <p className="empty-state-description">{description}</p>
      {action && <div className="empty-state-action">{action}</div>}
    </Container>
  )
}
