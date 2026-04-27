import { useEffect, useState } from 'react'
import { api, type BrowseResult } from '../api/client'

interface Props {
  initialPath?: string
  /** true: 只允许选目录；false: 文件也能选 */
  dirOnly?: boolean
  onPick: (path: string) => void
  onClose: () => void
}

/**
 * 模态目录浏览器：通过 /api/browse 拉条目，点目录进入，点文件（或当前目录）选中。
 * 仅允许 REPO_ROOT 下浏览（后端强制）。
 */
export default function PathPicker({
  initialPath,
  dirOnly = false,
  onPick,
  onClose,
}: Props) {
  const [data, setData] = useState<BrowseResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [path, setPath] = useState(initialPath ?? '')

  const load = async (p?: string) => {
    setError(null)
    try {
      const r = await api.browse(p)
      setData(r)
      setPath(r.path)
    } catch (e) {
      setError(String(e))
    }
  }

  useEffect(() => {
    void load(initialPath)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div
      className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center"
      onClick={onClose}
    >
      <div
        className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl
          w-[640px] max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="px-4 py-3 border-b border-slate-700 flex items-center gap-2">
          <h3 className="text-sm font-semibold flex-1">选择路径</h3>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-200 text-sm"
          >
            ✕
          </button>
        </header>

        <div className="px-4 py-2 border-b border-slate-800 flex items-center gap-2">
          {data?.parent && (
            <button
              onClick={() => void load(data.parent!)}
              className="text-xs text-slate-400 hover:text-slate-200"
            >
              ← 上级
            </button>
          )}
          <input
            type="text"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && void load(path)}
            className="flex-1 px-2 py-1 bg-slate-800 border border-slate-700 rounded
              text-xs font-mono"
          />
          <button
            onClick={() => void load(path)}
            className="text-xs px-2 py-1 rounded bg-slate-700 hover:bg-slate-600"
          >
            前往
          </button>
        </div>

        {error && (
          <div className="px-4 py-2 text-red-400 text-sm font-mono">{error}</div>
        )}

        <div className="flex-1 overflow-y-auto">
          {data?.entries.map((e) => {
            const childPath =
              data.path.replace(/[/\\]+$/, '') +
              (data.path.endsWith('/') || data.path.endsWith('\\') ? '' : '/') +
              e.name
            const enterable = e.type === 'dir'
            const selectable = enterable || !dirOnly
            return (
              <div
                key={e.name}
                className="px-4 py-2 border-b border-slate-800 flex items-center
                  gap-3 hover:bg-slate-800/50 group"
              >
                <span className="text-slate-500 w-4 text-center">
                  {e.type === 'dir' ? '📁' : '📄'}
                </span>
                <span className="flex-1 text-sm font-mono">{e.name}</span>
                {enterable && (
                  <button
                    onClick={() => void load(childPath)}
                    className="text-xs text-slate-400 hover:text-cyan-400 invisible group-hover:visible"
                  >
                    打开
                  </button>
                )}
                {selectable && (
                  <button
                    onClick={() => onPick(childPath)}
                    className="text-xs text-slate-400 hover:text-emerald-400 invisible group-hover:visible"
                  >
                    选这个
                  </button>
                )}
              </div>
            )
          })}
        </div>

        <footer className="px-4 py-3 border-t border-slate-700 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-3 py-1.5 rounded text-sm bg-slate-700 hover:bg-slate-600"
          >
            取消
          </button>
          <button
            onClick={() => onPick(path)}
            className="px-3 py-1.5 rounded text-sm bg-cyan-600 hover:bg-cyan-500"
          >
            选择当前目录
          </button>
        </footer>
      </div>
    </div>
  )
}
