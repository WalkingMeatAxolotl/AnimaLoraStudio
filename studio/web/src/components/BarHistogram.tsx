/** 通用水平 bar histogram —— 复用于：
 *  - Preprocess 像素分布（PixelHist）
 *  - PreprocessCrop 长宽比分布（ArHist）
 *  - Overview 详情 tab 内的同款统计
 *
 *  渲染：每行 grid 96px label / 1fr 进度条 / 30px 数字
 *  bar 用全局 `.ar-bar` / `.ar-bar-fill` CSS class。
 */
interface HistBin {
  /** React key（可选，fallback 用 label） */
  key?: string
  label: string
  n: number
}

interface Props {
  bins: HistBin[]
  /** bins 为空时的 placeholder 文本 */
  emptyHint?: string
}

export default function BarHistogram({ bins, emptyHint }: Props) {
  if (bins.length === 0) {
    return emptyHint ? (
      <p className="text-xs text-fg-tertiary italic m-0">{emptyHint}</p>
    ) : null
  }
  const max = Math.max(1, ...bins.map((b) => b.n))
  return (
    <div className="flex flex-col gap-1">
      {bins.map((b) => (
        <div
          key={b.key ?? b.label}
          className="grid items-center gap-1.5 text-[11px]"
          style={{ gridTemplateColumns: '96px 1fr 30px' }}
        >
          <span className="text-fg-tertiary font-mono">{b.label}</span>
          <div className="ar-bar">
            <div className="ar-bar-fill" style={{ width: `${(b.n / max) * 100}%` }} />
          </div>
          <span className="font-mono text-right text-fg-secondary">{b.n}</span>
        </div>
      ))}
    </div>
  )
}
