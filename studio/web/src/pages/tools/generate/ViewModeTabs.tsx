/** 视图模式 tab：单图 / XY 矩阵。
 *
 * 用户决策：双图对比合并进 XY 模式内部（selectedIndices=2 时自动切到
 * compare sub-view，不再单独占顶部 tab）。 */

export type ViewMode = 'single' | 'xy'

export default function ViewModeTabs({
  mode, onModeChange,
}: {
  mode: ViewMode
  onModeChange: (m: ViewMode) => void
}) {
  const tab = (m: ViewMode, label: string) => (
    <button
      onClick={() => onModeChange(m)}
      className={`btn btn-sm text-xs ${
        mode === m ? 'btn-primary' : 'btn-ghost text-fg-secondary'
      }`}
    >
      {label}
    </button>
  )
  return (
    <div className="flex items-center gap-1.5" role="tablist">
      {tab('single', '单图')}
      {tab('xy', 'XY 矩阵')}
    </div>
  )
}
