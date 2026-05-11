import { useEffect, useMemo, useState } from 'react'
import { api, type CaptionEntry, type ProjectSummary } from '../../../api/client'

/** 从训练集 caption 里选一条，把 tags 拿到生成 prompt 里。
 *
 * 流程：选项目 → 选版本 → 列 captions → 单选 → 「追加」/「替换」。
 * 保留 inline 展开（不弹 modal），跟 LoRA picker 一致。
 */
export default function PromptFromDatasetPicker({
  onAppend, onReplace, onClose,
}: {
  /** 把选中 caption 的 tags 追加到当前 prompt 末尾 */
  onAppend: (tags: string[]) => void
  /** 用 tags 替换整个 prompt */
  onReplace: (tags: string[]) => void
  onClose: () => void
}) {
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [pid, setPid] = useState<number | null>(null)
  const [vid, setVid] = useState<number | null>(null)
  const [versions, setVersions] = useState<Array<{ id: number; label: string }>>([])
  const [captions, setCaptions] = useState<CaptionEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [search, setSearch] = useState('')

  // 1. 拉项目列表
  useEffect(() => {
    void api.listProjects()
      .then(setProjects)
      .catch((e) => setError(String(e)))
  }, [])

  // 2. 选项目后拉版本列表
  useEffect(() => {
    if (!pid) { setVersions([]); setVid(null); return }
    void api.getProject(pid)
      .then((p) => {
        const vs = p.versions.map((v) => ({ id: v.id, label: v.label }))
        setVersions(vs)
        if (vs.length > 0) setVid(vs[0].id)
        else setVid(null)
      })
      .catch((e) => setError(String(e)))
  }, [pid])

  // 3. 选版本后拉 captions
  useEffect(() => {
    if (!pid || !vid) { setCaptions([]); return }
    setLoading(true)
    setError(null)
    void api.listCaptionsFull(pid, vid)
      .then((r) => { setCaptions(r.items); setLoading(false) })
      .catch((e) => { setError(String(e)); setLoading(false) })
  }, [pid, vid])

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return captions
    return captions.filter((c) =>
      c.name.toLowerCase().includes(q) ||
      c.tags.some((t) => t.toLowerCase().includes(q))
    )
  }, [captions, search])

  const selected = filtered.find((c) => `${c.folder}/${c.name}` === selectedKey)

  return (
    <div
      className="rounded-md border border-subtle bg-overlay p-2.5 flex flex-col gap-2"
      data-testid="prompt-dataset-picker"
    >
      {/* header */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-fg-secondary shrink-0">从训练集选 prompt</span>
        <span className="flex-1" />
        <button
          onClick={onClose}
          className="btn btn-ghost btn-sm text-fg-tertiary px-1.5"
          title="关闭"
          aria-label="关闭"
        >
          ×
        </button>
      </div>

      {/* project / version 选择 */}
      <div className="flex gap-2">
        <select
          className="input text-xs flex-1"
          value={pid ?? ''}
          onChange={(e) => setPid(e.target.value ? Number(e.target.value) : null)}
        >
          <option value="">选项目…</option>
          {projects.map((p) => (
            <option key={p.id} value={p.id}>{p.title}</option>
          ))}
        </select>
        <select
          className="input text-xs flex-1"
          value={vid ?? ''}
          onChange={(e) => setVid(e.target.value ? Number(e.target.value) : null)}
          disabled={versions.length === 0}
        >
          <option value="">选版本…</option>
          {versions.map((v) => (
            <option key={v.id} value={v.id}>{v.label}</option>
          ))}
        </select>
      </div>

      {/* search */}
      <input
        type="text"
        className="input text-xs"
        placeholder="搜索文件名 / tag…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        disabled={!pid || !vid || captions.length === 0}
      />

      {error && <div className="text-2xs text-err">{error}</div>}

      {/* caption 列表 */}
      <div className="flex flex-col gap-px overflow-y-auto" style={{ maxHeight: 280 }}>
        {loading && <div className="text-2xs text-fg-tertiary">加载中…</div>}
        {!loading && pid && vid && captions.length === 0 && (
          <div className="text-2xs text-fg-tertiary">该版本没有 caption</div>
        )}
        {!loading && filtered.map((c) => {
          const k = `${c.folder}/${c.name}`
          const active = selectedKey === k
          return (
            <button
              key={k}
              onClick={() => setSelectedKey(active ? null : k)}
              className="flex items-center gap-2 px-2 py-1.5 rounded text-xs text-left border-none transition-colors"
              style={{
                background: active ? 'var(--accent-soft)' : 'transparent',
                color: active ? 'var(--accent)' : 'var(--fg-secondary)',
                cursor: 'pointer',
              }}
            >
              <span className="font-mono text-2xs shrink-0">{active ? '✓' : '+'}</span>
              <div className="flex-1 min-w-0">
                <div className="font-medium truncate">{c.name}</div>
                <div className="text-2xs text-fg-tertiary truncate">
                  {c.tags.slice(0, 6).join(', ')}{c.tags.length > 6 ? ` (+${c.tags.length - 6})` : ''}
                </div>
              </div>
            </button>
          )
        })}
      </div>

      {/* 选中预览 + 操作 */}
      {selected && (
        <>
          <div
            className="rounded-sm border border-subtle bg-sunken px-2 py-1.5 text-xs font-mono whitespace-pre-wrap"
            style={{ maxHeight: 100, overflowY: 'auto' }}
          >
            {selected.tags.join(', ')}
          </div>
          <div className="flex gap-2 justify-end">
            <button
              onClick={() => { onReplace(selected.tags); onClose() }}
              className="btn btn-ghost btn-sm text-xs"
            >
              替换 prompt
            </button>
            <button
              onClick={() => { onAppend(selected.tags); onClose() }}
              className="btn btn-primary btn-sm text-xs"
            >
              追加到末尾
            </button>
          </div>
        </>
      )}
    </div>
  )
}
