/** 带 label 的数值输入框（共享 sidebar 参数面板）。 */
export default function NumField({ label, value, onChange, min, max, step }: {
  label: string; value: number
  onChange: (v: number) => void
  min?: number; max?: number; step?: number
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="caption">{label}</label>
      <input
        type="number"
        className="input"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}
