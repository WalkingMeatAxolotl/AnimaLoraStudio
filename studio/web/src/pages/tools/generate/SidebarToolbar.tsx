import type { ButtonHTMLAttributes, ReactNode } from 'react'

export type SidebarToolIconName =
  | 'image'
  | 'grid'
  | 'axes'
  | 'layers'
  | 'text'
  | 'sliders'
  | 'swap'
  | 'edit'
  | 'plus'
  | 'dataset'
  | 'collapse'

export function SidebarToolIcon({ name, size = 15 }: { name: SidebarToolIconName; size?: number }) {
  const paths: Record<SidebarToolIconName, ReactNode> = {
    image: <><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="9" cy="9" r="1.5" /><path d="m21 15-5-5L5 21" /></>,
    grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
    axes: <><path d="M5 19V5M5 19h14" /><path d="m3 7 2-2 2 2M17 17l2 2-2 2" /><circle cx="11" cy="13" r="1.5" /></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5" /><path d="m3 16 9 5 9-5" /></>,
    text: <><path d="M4 6V4h16v2M9 20h6M12 4v16" /></>,
    sliders: <><path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h7M15 18h5" /><circle cx="16" cy="6" r="2" /><circle cx="8" cy="12" r="2" /><circle cx="13" cy="18" r="2" /></>,
    swap: <><path d="m17 3 4 4-4 4" /><path d="M21 7H8" /><path d="m7 21-4-4 4-4" /><path d="M3 17h13" /></>,
    edit: <><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4Z" /></>,
    plus: <><path d="M12 5v14M5 12h14" /></>,
    dataset: <><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5" /><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" /></>,
    collapse: <><path d="m9 18 6-6-6-6" /></>,
  }

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="shrink-0"
    >
      {paths[name]}
    </svg>
  )
}

type ToolbarActionProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'type' | 'children'> & {
  label: string
  icon: ReactNode
  iconOnly?: boolean
}

export function ToolbarAction({ label, icon, iconOnly = false, className = '', title, ...props }: ToolbarActionProps) {
  return (
    <button
      {...props}
      type="button"
      title={title ?? label}
      aria-label={props['aria-label'] ?? label}
      className={[
        'inline-flex h-8 shrink-0 items-center justify-center gap-1.5 rounded-md border border-transparent bg-transparent text-xs font-medium text-fg-secondary transition-colors',
        'hover:border-subtle hover:bg-overlay hover:text-fg-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent',
        iconOnly ? 'w-8 p-0' : 'px-2.5',
        className,
      ].join(' ')}
    >
      {icon}
      {!iconOnly && <span>{label}</span>}
    </button>
  )
}
