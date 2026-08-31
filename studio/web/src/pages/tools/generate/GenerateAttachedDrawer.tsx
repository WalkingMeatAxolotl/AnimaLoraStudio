import type { ReactNode } from 'react'

export default function GenerateAttachedDrawer({
  id,
  ariaLabel,
  testId,
  open = true,
  children,
}: {
  id: string
  ariaLabel: string
  testId: string
  /** Keep mounted when closed so heavy lists retain DOM, scroll position, and decoded images. */
  open?: boolean
  children: ReactNode
}) {
  return (
    <aside
      id={id}
      aria-label={ariaLabel}
      aria-hidden={!open || undefined}
      hidden={!open}
      style={open ? undefined : { display: 'none' }}
      className="generate-attached-drawer absolute z-20 flex flex-col bg-surface border border-subtle border-l-0 shadow-xl"
      data-testid={testId}
    >
      {children}
    </aside>
  )
}
