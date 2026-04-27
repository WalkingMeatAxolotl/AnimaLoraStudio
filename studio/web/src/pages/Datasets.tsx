import { useCallback, useEffect, useState } from 'react'
import { api, type DatasetScan } from '../api/client'
import PathPicker from '../components/PathPicker'

export default function DatasetsPage() {
  const [path, setPath] = useState('')
  const [scan, setScan] = useState<DatasetScan | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [picking, setPicking] = useState(false)

  const reload = useCallback(async (p: string) => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.listDatasets(p || undefined)
      setScan(data)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void reload('')
  }, [reload])

  return (
    <div className="space-y-4">
      <section className="rounded-xl border border-slate-700 bg-slate-800/40 p-4">
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <label className="block text-xs text-slate-500 mb-1">
              数据集根目录
            </label>
            <input
              type="text"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && void reload(path)}
              placeholder="留空使用 ./dataset"
              className="w-full px-3 py-1.5 bg-slate-900 border border-slate-700
                rounded-md text-sm font-mono focus:outline-none focus:border-cyan-500"
            />
          </div>
          <button
            onClick={() => setPicking(true)}
            className="px-3 py-1.5 rounded text-sm bg-slate-700 hover:bg-slate-600"
          >
            浏览
          </button>
          <button
            onClick={() => void reload(path)}
            disabled={loading}
            className="px-3 py-1.5 rounded text-sm bg-cyan-600 hover:bg-cyan-500
              disabled:bg-slate-700"
          >
            {loading ? '扫描中...' : '扫描'}
          </button>
        </div>
        {picking && (
          <PathPicker
            initialPath={path || undefined}
            dirOnly
            onPick={(p) => {
              setPath(p)
              setPicking(false)
              void reload(p)
            }}
            onClose={() => setPicking(false)}
          />
        )}
        {scan && (
          <div className="mt-3 text-xs text-slate-500 font-mono break-all">
            {scan.root}
          </div>
        )}
      </section>

      {error && (
        <div className="p-3 rounded bg-red-900/40 border border-red-700 text-red-300 text-sm font-mono">
          {error}
        </div>
      )}

      {scan && !scan.exists && (
        <div className="p-4 rounded border border-slate-700 bg-slate-800/30 text-slate-400 text-sm">
          目录不存在或不可访问。检查上方路径是否正确。
        </div>
      )}

      {scan?.exists && scan.folders.length === 0 && (
        <div className="p-4 rounded border border-slate-700 bg-slate-800/30 text-slate-400 text-sm">
          目录为空。
        </div>
      )}

      {scan?.exists && scan.folders.length > 0 && (
        <>
          <div className="flex gap-6 px-1 text-sm">
            <div>
              <span className="text-slate-500">总图数：</span>
              <span className="text-slate-200 font-mono">
                {scan.total_images}
              </span>
            </div>
            <div>
              <span className="text-slate-500">加权 step/epoch：</span>
              <span className="text-cyan-300 font-mono">
                {scan.weighted_steps_per_epoch}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {scan.folders.map((f) => (
              <article
                key={f.path}
                className="rounded-lg border border-slate-700 bg-slate-800/40 p-4"
              >
                <header className="flex items-baseline justify-between mb-2">
                  <h3 className="text-base font-semibold text-slate-200">
                    {f.label}
                  </h3>
                  <div className="text-xs text-slate-500 font-mono">
                    {f.name}
                  </div>
                </header>
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400 mb-3">
                  <span>
                    repeat:{' '}
                    <span className="text-cyan-300 font-mono">{f.repeat}</span>
                  </span>
                  <span>
                    images:{' '}
                    <span className="text-slate-200 font-mono">
                      {f.image_count}
                    </span>
                  </span>
                  <span>
                    json:{' '}
                    <span className="text-emerald-400 font-mono">
                      {f.caption_types.json}
                    </span>
                  </span>
                  <span>
                    txt:{' '}
                    <span className="text-amber-400 font-mono">
                      {f.caption_types.txt}
                    </span>
                  </span>
                  <span>
                    无 caption:{' '}
                    <span
                      className={
                        f.caption_types.none > 0
                          ? 'text-red-400 font-mono'
                          : 'text-slate-500 font-mono'
                      }
                    >
                      {f.caption_types.none}
                    </span>
                  </span>
                </div>
                {f.samples.length > 0 && (
                  <div className="grid grid-cols-4 gap-1.5">
                    {f.samples.map((name) => (
                      <img
                        key={name}
                        src={api.thumbnailUrl(f.path, name)}
                        alt={name}
                        loading="lazy"
                        className="w-full aspect-square object-cover rounded
                          border border-slate-800"
                        onError={(e) =>
                          ((e.target as HTMLImageElement).style.opacity = '0.3')
                        }
                      />
                    ))}
                  </div>
                )}
              </article>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
