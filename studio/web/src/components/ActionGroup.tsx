import type { HTMLAttributes, ReactNode } from 'react'

export interface ActionGroupProps extends Omit<HTMLAttributes<HTMLDivElement>, 'children'> {
  /** Save state or other non-interactive context. Rendered before every action. */
  status?: ReactNode
  /** Lower-emphasis, recovery, or destructive actions. */
  secondary?: ReactNode
  /** The single primary action for this scope. Always rendered last. */
  primary?: ReactNode
}

/**
 * Pattern-layer scaffold for a related set of save or submit actions.
 *
 * Slot order is intentional: status → secondary → primary. Callers keep all
 * business state and handlers; this component only owns ordering, wrapping,
 * and spacing.
 */
export default function ActionGroup({
  status,
  secondary,
  primary,
  className = '',
  role,
  'aria-label': ariaLabel,
  'aria-labelledby': ariaLabelledBy,
  ...rest
}: ActionGroupProps) {
  return (
    <div
      {...rest}
      role={role ?? (ariaLabel || ariaLabelledBy ? 'group' : undefined)}
      aria-label={ariaLabel}
      aria-labelledby={ariaLabelledBy}
      className={[
        'flex min-w-0 flex-wrap items-center justify-end gap-related',
        className,
      ].filter(Boolean).join(' ')}
    >
      {status && (
        <div data-action-slot="status" className="min-w-0 shrink-0">
          {status}
        </div>
      )}
      {secondary && (
        <div data-action-slot="secondary" className="flex flex-wrap items-center justify-end gap-related">
          {secondary}
        </div>
      )}
      {primary && (
        <div data-action-slot="primary" className="flex items-center justify-end">
          {primary}
        </div>
      )}
    </div>
  )
}
