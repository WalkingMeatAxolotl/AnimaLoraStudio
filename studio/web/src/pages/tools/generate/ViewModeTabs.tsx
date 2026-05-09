/** 视图模式 tab：单图 / XY 矩阵 / 双图对比（compare 仅 XY 完成后选 2 张时可用）。 */

export type ViewMode = 'single' | 'xy' | 'compare'

export default function ViewModeTabs({
  mode, onModeChange, compareEnabled,
}: {
  mode: ViewMode
  onModeChange: (m: ViewMode) => void
  /** 双图对比仅在 XY 已生成 + 选了 2 张时可用 —— commit 6 用 */
  compareEnabled: boolean
}) {
  const tab = (m: ViewMode, label: string, disabled = false) => (
    <button
      onClick={() => !disabled && onModeChange(m)}
      disabled={disabled}
      className={`btn btn-sm text-xs ${
        mode === m ? 'btn-primary' : 'btn-ghost text-fg-secondary'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      {label}
    </button>
  )
  return (
    <div className="flex items-center gap-1.5" role="tablist">
      {tab('single', '单图')}
      {tab('xy', 'XY 矩阵')}
      {tab('compare', '双图对比', !compareEnabled)}
    </div>
  )
}
