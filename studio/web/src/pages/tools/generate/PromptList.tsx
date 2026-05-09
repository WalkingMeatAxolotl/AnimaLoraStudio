/** 多 prompt 编辑列表（每条 prompt 各生成 count 张）。 */
export default function PromptList({ prompts, onChange }: {
  prompts: string[]
  onChange: (p: string[]) => void
}) {
  const add = () => onChange([...prompts, ''])
  const del = (i: number) => onChange(prompts.filter((_, idx) => idx !== i))
  const set = (i: number, v: string) => onChange(prompts.map((p, idx) => idx === i ? v : p))

  return (
    <div className="flex flex-col gap-2">
      {prompts.map((p, i) => (
        <div key={i} className="flex gap-1.5">
          {prompts.length > 1 && (
            <span className="caption mt-2.5 w-4 text-center shrink-0">{i + 1}</span>
          )}
          <textarea
            className="input flex-1 font-mono text-sm resize-y"
            rows={3}
            value={p}
            onChange={(e) => set(i, e.target.value)}
            placeholder="输入正向提示词…"
          />
          {prompts.length > 1 && (
            <button onClick={() => del(i)} className="btn btn-ghost btn-sm text-fg-tertiary hover:text-err self-start px-1.5">×</button>
          )}
        </div>
      ))}
      <button onClick={add} className="btn btn-ghost btn-sm self-start text-xs text-fg-secondary">
        + 添加 prompt（轮换生成）
      </button>
    </div>
  )
}
