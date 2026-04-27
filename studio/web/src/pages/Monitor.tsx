import { useEffect, useState } from 'react'
import { api, type HealthResponse } from '../api/client'

export default function MonitorPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e) => setError(String(e)))
  }, [])

  const ok = !error && health?.status === 'ok'

  return (
    <div className="space-y-4">
      <section className="rounded-xl border border-slate-700 bg-slate-800/40 p-5">
        <h2 className="text-base font-semibold mb-2 text-slate-200">
          守护进程状态
        </h2>
        <div className="flex items-center gap-3">
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              ok ? 'bg-emerald-400' : 'bg-red-400'
            }`}
          />
          <span className={ok ? 'text-emerald-400' : 'text-red-400'}>
            {error ? 'offline' : health?.status ?? '...'}
          </span>
          {health && (
            <span className="text-slate-500 text-sm">v{health.version}</span>
          )}
        </div>
        {error && (
          <p className="text-red-400 text-sm mt-2 font-mono">{error}</p>
        )}
      </section>

      <section className="rounded-xl border border-slate-700 bg-slate-800/40 p-5">
        <h2 className="text-base font-semibold mb-3 text-slate-200">
          实时训练监控
        </h2>
        <p className="text-slate-400 text-sm mb-3">
          loss 曲线、采样图、训练进度仍在旧 UI 上。后续 P5 会把它内嵌到这里。
        </p>
        <a
          href="/"
          className="inline-block px-3 py-1.5 rounded text-sm bg-cyan-600 hover:bg-cyan-500"
        >
          打开监控面板 →
        </a>
      </section>
    </div>
  )
}
