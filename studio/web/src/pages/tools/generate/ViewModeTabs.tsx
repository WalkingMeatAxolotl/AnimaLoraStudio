import { useTranslation } from 'react-i18next'
import { SegmentedControl, SidebarToolIcon } from './SidebarToolbar'

/** 视图模式选择：单图 / XY 矩阵。
 *
 * 这是工作模式的互斥选择，不是内容 tab；使用 radiogroup 语义。
 * 双图对比合并进 XY 模式内部（selectedIndices=2 时自动切到 compare sub-view）。 */
export type ViewMode = 'single' | 'xy'

export default function ViewModeTabs({
  mode, onModeChange,
}: {
  mode: ViewMode
  onModeChange: (m: ViewMode) => void
}) {
  const { t } = useTranslation()
  return (
    <SegmentedControl
      items={[
        {
          value: 'single',
          label: t('generate.singleMode'),
          icon: <SidebarToolIcon name="image" />,
        },
        {
          value: 'xy',
          label: t('generate.xyMode'),
          icon: <SidebarToolIcon name="grid" />,
        },
      ]}
      value={mode}
      onChange={onModeChange}
      ariaLabel={t('generate.viewModes')}
      semantics="radio"
      idPrefix="generate-view-mode"
      className="w-full"
    />
  )
}
