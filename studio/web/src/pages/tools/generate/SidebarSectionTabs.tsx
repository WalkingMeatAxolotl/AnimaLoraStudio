import { useTranslation } from 'react-i18next'
import { Tabs, type TabItem } from '../../../components/SelectionGroup'
import { SidebarToolIcon } from './SidebarToolbar'
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
  const allTabs: Record<SidebarTab, TabItem<SidebarTab>> = {
    xy: {
      value: 'xy',
      label: t('generate.xyAxes'),
      icon: <SidebarToolIcon name="axes" />,
      controls: 'generate-sidebar-panel-xy',
    },
    lora: {
      value: 'lora',
      label: 'LoRA',
      icon: <SidebarToolIcon name="layers" />,
      controls: 'generate-sidebar-panel-lora',
    },
    prompts: {
      value: 'prompts',
      label: t('generate.prompts'),
      icon: <SidebarToolIcon name="text" />,
      controls: 'generate-sidebar-panel-prompts',
    },
    config: {
      value: 'config',
      label: t('generate.parametersShort'),
      icon: <SidebarToolIcon name="sliders" />,
      controls: 'generate-sidebar-panel-config',
    },
  }
  const tabs = mode === 'xy'
    ? [allTabs.xy, allTabs.lora, allTabs.prompts, allTabs.config]
    : [allTabs.lora, allTabs.prompts, allTabs.config]

  return (
    <Tabs
      appearance="segmented"
      size="sm"
      items={tabs}
      value={tab}
      onChange={onTabChange}
      ariaLabel={t('generate.sidebarSections')}
      idPrefix="generate-sidebar-tab"
      className="w-full"
    />
  )
}
