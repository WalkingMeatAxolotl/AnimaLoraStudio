/** 带 label 的数值输入框（共享 sidebar 参数面板）。
 *
 * 容器 flex-1 min-w-0 + input w-full：横向并排两个 NumField 时两列等宽，
 * 不被 label 长度撑歪。
 */
export default function NumField({ label, value, onChange, min, max, step }: {
  label: string; value: number
  onChange: (v: number) => void
  min?: number; max?: number; step?: number
}) {
  return (
    <div className="flex flex-col gap-1 flex-1 min-w-0">
      <label className="caption truncate">{label}</label>
      <input
        type="number"
        className="input w-full"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}
