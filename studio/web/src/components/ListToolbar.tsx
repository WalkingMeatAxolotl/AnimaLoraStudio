import type { HTMLAttributes, ReactNode } from 'react'

export interface ListToolbarProps
  extends Omit<HTMLAttributes<HTMLDivElement>, 'aria-label' | 'children' | 'role'> {
  /** Visible or localized name for this filter region. */
  ariaLabel: string
  /** Dominant list query control. It owns the flexible primary track. */
  search: ReactNode
  /** Facets that narrow the result set. Business state remains page-owned. */
  filters?: ReactNode
  /** Ordering control, rendered after filters in DOM and keyboard order. */
  sort?: ReactNode
}

/**
 * Pattern-layer scaffold for an ordinary list search / filter row.
 *
 * Slot order is intentional: search → filters → sort. The Pattern owns the
 * named region, page-aligned shell, density-aware spacing, and narrow-desktop
 * wrapping. Query state, debounce, persistence, API parameters, and disclosure
 * state stay with the product page.
 */
export default function ListToolbar({
  ariaLabel,
  search,
  filters,
  sort,
  className = '',
  ...rest
}: ListToolbarProps) {
  return (
    <div
      {...rest}
      role="region"
      aria-label={ariaLabel}
      className={['list-toolbar', className].filter(Boolean).join(' ')}
    >
      <div data-list-toolbar-slot="search" className="list-toolbar-search">
        {search}
      </div>
      {(filters || sort) && (
        <div className="list-toolbar-controls">
          {filters && (
            <div data-list-toolbar-slot="filters" className="list-toolbar-filters">
              {filters}
            </div>
          )}
          {sort && (
            <div data-list-toolbar-slot="sort" className="list-toolbar-sort">
              {sort}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
