interface Props {
  title: string
  phase: string
  description: string
}

export default function Placeholder({ title, phase, description }: Props) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800/40 p-8 text-center">
      <h1 className="text-2xl font-semibold mb-2">{title}</h1>
      <p className="text-cyan-400 text-sm mb-4">即将上线 · {phase}</p>
      <p className="text-slate-400 max-w-md mx-auto">{description}</p>
    </div>
  )
}
