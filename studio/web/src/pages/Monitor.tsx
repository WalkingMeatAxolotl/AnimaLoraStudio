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
    <div className="flex flex-col h-full gap-3">
      <section className="rounded-xl border border-slate-700 bg-slate-800/40 px-5 py-3 flex items-center gap-3">
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
        <span className="flex-1" />
        <a
          href="/"
          target="_blank"
          rel="noreferrer"
          className="text-xs text-slate-400 hover:text-slate-200"
        >
          在新窗口打开 ↗
        </a>
      </section>

      <iframe
        src="/"
        title="Anima Training Monitor"
        className="flex-1 w-full rounded-xl border border-slate-700 bg-black/30 min-h-0"
      />
    </div>
  )
}
