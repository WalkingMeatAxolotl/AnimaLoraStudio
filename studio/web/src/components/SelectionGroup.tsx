import { useRef, type KeyboardEvent, type ReactNode } from 'react'

export type SelectionSize = 'md' | 'sm'
export type TabsAppearance = 'underline' | 'segmented'

export interface SelectionItem<T extends string> {
  value: T
  label: string
  icon?: ReactNode
  controls?: string
  disabled?: boolean
}

export type TabItem<T extends string> = SelectionItem<T> & {
  controls: string
}

type SharedSelectionProps<T extends string, I extends SelectionItem<T> = SelectionItem<T>> = {
  items: readonly I[]
  value: T
  onChange: (value: T) => void
  ariaLabel: string
  idPrefix: string
  size?: SelectionSize
  className?: string
}

export type TabsProps<T extends string> = SharedSelectionProps<T, TabItem<T>> & {
  appearance?: TabsAppearance
}

export type SegmentedControlProps<T extends string> = SharedSelectionProps<T>

export function selectionItemId(idPrefix: string, value: string): string {
  const suffix = value
    .trim()
    .toLocaleLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return `${idPrefix}-${suffix || 'item'}`
}

function SelectionGroup<T extends string>({
  items,
  value,
  onChange,
  ariaLabel,
  idPrefix,
  size = 'md',
  className = '',
  semantics,
  appearance,
}: SharedSelectionProps<T> & {
  semantics: 'tabs' | 'radio'
  appearance: TabsAppearance
}) {
  const itemRefs = useRef(new Map<T, HTMLButtonElement>())
  const activeIndex = items.findIndex((item) => item.value === value && !item.disabled)
  const fallbackIndex = items.findIndex((item) => !item.disabled)
  const tabbableIndex = activeIndex >= 0 ? activeIndex : fallbackIndex

  const move = (currentIndex: number, direction: 1 | -1) => {
    if (items.length === 0) return
    let nextIndex = currentIndex
    for (let step = 0; step < items.length; step += 1) {
      nextIndex = (nextIndex + direction + items.length) % items.length
      if (!items[nextIndex].disabled) {
        const next = items[nextIndex]
        onChange(next.value)
        itemRefs.current.get(next.value)?.focus()
        return
      }
    }
  }

  const moveToEdge = (fromEnd: boolean) => {
    const ordered = fromEnd ? [...items].reverse() : items
    const next = ordered.find((item) => !item.disabled)
    if (!next) return
    onChange(next.value)
    itemRefs.current.get(next.value)?.focus()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      event.preventDefault()
      move(index, 1)
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      event.preventDefault()
      move(index, -1)
    } else if (event.key === 'Home') {
      event.preventDefault()
      moveToEdge(false)
    } else if (event.key === 'End') {
      event.preventDefault()
      moveToEdge(true)
    }
  }

  return (
    <div
      role={semantics === 'tabs' ? 'tablist' : 'radiogroup'}
      aria-label={ariaLabel}
      className={[
        'ui-selection-group',
        `ui-selection-${appearance}`,
        `ui-selection-${size}`,
        className,
      ].filter(Boolean).join(' ')}
    >
      {items.map((item, index) => {
        const active = item.value === value
        return (
          <button
            key={item.value}
            ref={(node) => {
              if (node) itemRefs.current.set(item.value, node)
              else itemRefs.current.delete(item.value)
            }}
            id={selectionItemId(idPrefix, item.value)}
            type="button"
            role={semantics === 'tabs' ? 'tab' : 'radio'}
            aria-selected={semantics === 'tabs' ? active : undefined}
            aria-checked={semantics === 'radio' ? active : undefined}
            aria-controls={item.controls}
            disabled={item.disabled}
            tabIndex={index === tabbableIndex ? 0 : -1}
            title={item.label}
            data-state={active ? 'active' : 'inactive'}
            onClick={() => {
              if (!item.disabled && item.value !== value) onChange(item.value)
            }}
            onKeyDown={(event) => handleKeyDown(event, index)}
            className="ui-selection-item"
          >
            {item.icon && <span className="ui-selection-icon" aria-hidden="true">{item.icon}</span>}
            <span className="ui-selection-label">{item.label}</span>
          </button>
        )
      })}
    </div>
  )
}

/** Content navigation for switching associated tab panels. */
export function Tabs<T extends string>({ appearance = 'underline', ...props }: TabsProps<T>) {
  return <SelectionGroup {...props} semantics="tabs" appearance={appearance} />
}

/** Mutually exclusive value selection; not content navigation. */
export function SegmentedControl<T extends string>(props: SegmentedControlProps<T>) {
  return <SelectionGroup {...props} semantics="radio" appearance="segmented" />
}
