import type { ReactNode } from 'react'

export default function GenerateAttachedDrawer({
  id,
  ariaLabel,
  testId,
  children,
}: {
  id: string
  ariaLabel: string
  testId: string
  children: ReactNode
}) {
  return (
    <aside
      id={id}
      aria-label={ariaLabel}
      className="generate-attached-drawer absolute z-20 flex flex-col bg-surface border border-subtle border-l-0 shadow-xl"
      data-testid={testId}
    >
      {children}
    </aside>
  )
}
