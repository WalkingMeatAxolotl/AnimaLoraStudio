/** 正向提示词输入。
 *
 * 之前支持多 prompt 轮换（"+ 添加 prompt"），用户决策"隐藏前端轮换功能"
 * → 简化成单 textarea。后端仍接 list[str]，发请求时仍然包成数组。
 */
export default function PromptList({ prompts, onChange }: {
  prompts: string[]
  onChange: (p: string[]) => void
}) {
  // 当前只显示第一条 prompt；用户编辑时同步成 [value]
  const value = prompts[0] ?? ''
  return (
    <textarea
      className="input w-full font-mono text-sm resize-y"
      rows={5}
      value={value}
      onChange={(e) => onChange([e.target.value])}
      placeholder="输入正向提示词…"
    />
  )
}
