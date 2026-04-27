/** PP2-PP6 的步骤页占位。每个 PP 落地时会被实际页面替换。 */
export default function StepPlaceholder({
  step,
  doc,
}: {
  step: string
  doc: string
}) {
  return (
    <div className="max-w-xl mt-8 space-y-3">
      <h2 className="text-lg font-semibold">{step}</h2>
      <p className="text-sm text-slate-400">
        该步骤将在 <code className="text-cyan-300">{doc}</code> 阶段实现。
      </p>
      <p className="text-xs text-slate-500">
        当前 PP1 仅落地了 Project / Version 数据模型与导航；下一步是 PP2
        下载集成。
      </p>
    </div>
  )
}
