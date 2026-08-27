import { useTranslation } from 'react-i18next'
import type { ViewMode } from './ViewModeTabs'

export type SidebarTab = 'lora' | 'xy' | 'prompts' | 'config'

export default function SidebarSectionTabs({
  tab,
  onTabChange,
  mode,
}: {
  tab: SidebarTab
  onTabChange: (tab: SidebarTab) => void
  mode: ViewMode
}) {
  const { t } = useTranslation()
  const tabs: Array<[SidebarTab, string]> = mode === 'xy'
    ? [
        ['xy', t('generate.xyAxes')],
        ['lora', 'LoRA'],
        ['prompts', t('generate.prompts')],
        ['config', t('generate.samplingParams')],
      ]
    : [
        ['lora', 'LoRA'],
        ['prompts', t('generate.prompts')],
        ['config', t('generate.samplingParams')],
      ]

  return (
    <div
      role="tablist"
      className="flex items-center gap-1"
      style={{ background: 'var(--bg-sunken)', borderRadius: 'var(--r-md)', padding: 3 }}
    >
      {tabs.map(([key, label], index) => {
        const active = tab === key
        return (
          <button
            key={key}
            id={`generate-sidebar-tab-${key}`}
            type="button"
            role="tab"
            aria-controls={`generate-sidebar-panel-${key}`}
            tabIndex={active ? 0 : -1}
            onClick={() => onTabChange(key)}
            onKeyDown={(event) => {
              let nextIndex: number | null = null
              if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length
              if (event.key === 'ArrowLeft') nextIndex = (index - 1 + tabs.length) % tabs.length
              if (event.key === 'Home') nextIndex = 0
              if (event.key === 'End') nextIndex = tabs.length - 1
              if (nextIndex == null) return
              event.preventDefault()
              const nextKey = tabs[nextIndex][0]
              onTabChange(nextKey)
              document.getElementById(`generate-sidebar-tab-${nextKey}`)?.focus()
            }}
            aria-selected={active}
            className="flex-1 min-w-0 truncate text-xs text-center transition-colors"
            style={{
              padding: '5px 6px',
              borderRadius: 'var(--r-sm)',
              border: `1px solid ${active ? 'var(--border-subtle)' : 'transparent'}`,
              background: active ? 'var(--bg-surface)' : 'transparent',
              color: active ? 'var(--fg-primary)' : 'var(--fg-tertiary)',
              fontWeight: active ? 600 : 500,
              boxShadow: active ? 'var(--sh-sm)' : 'none',
              cursor: 'pointer',
            }}
          >
            {label}
          </button>
        )
      })}
    </div>
  )
}
