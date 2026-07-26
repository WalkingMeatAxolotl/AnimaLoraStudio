// 「创建新评估」modal —— 从评估作业列表上方发起。
//
// 以前这是指标面板头上的一个内联展开区。评估现在是一类独立作业（有自己的列表和详情
// 页），发起动作就该在列表上方，参数在 modal 里填完再提交 —— 和「新建版本」同款。
// 目前参数只有 checkpoint 选择；样本数 / 指标模型仍走 Settings 默认。
import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, type EvalScale, type LoraCkpt } from '../api/client'

export default function CreateEvalModal({
  pid, vid, taskId, onClose, onCreated,
}: {
  pid: number
  vid: number
  /** 溯源：从训练详情发起时带上；从概览发起时留空。 */
  taskId?: number
  onClose: () => void
  onCreated: (sessionId: number) => void
}) {
  const [ckpts, setCkpts] = useState<LoraCkpt[]>([])
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [scale, setScale] = useState<EvalScale | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    void (async () => {
      try {
        const items = await api.listVersionLoraCkpts(pid, vid)
        if (alive) setCkpts(items)
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e))
      } finally {
        if (alive) setLoading(false)
      }
    })()
    // 规模因子（验证图数 / 指标 runner / baseline 开关）与选了几个无关，拉一次就够
    void api.getEvalScale(pid, vid).then((s) => { if (alive) setScale(s) }).catch(() => {})
    return () => { alive = false }
  }, [pid, vid])

  // Esc 关闭（与其它 modal 一致）
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  const toggle = useCallback((path: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }, [])

  // 候选 = 选中数 + baseline 一份，每个候选出一整套验证图。作业数恒为 1（一次评估一个
  // EvalSession，#465），成本落在出图数和阶段数上 —— 阶段 = 1 个出图 + 每个指标 runner。
  const picked = useMemo(() => {
    if (!scale || selected.size === 0) return null
    const candidates = selected.size + (scale.baseline_enabled ? 1 : 0)
    return {
      candidates,
      images: candidates * scale.validation_images,
      stages: 1 + scale.metric_runners.length,
      validationImages: scale.validation_images,
    }
  }, [scale, selected.size])

  const submit = async () => {
    if (selected.size === 0) return
    setBusy(true)
    setError(null)
    try {
      const r = await api.runTaskEval(pid, vid, { task_id: taskId, checkpoints: [...selected] })
      onCreated(r.session.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="create-eval-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-md"
      data-testid="create-eval-modal"
    >
      <div className="w-[90%] max-w-[680px] flex flex-col gap-4 p-6 bg-elevated border border-dim rounded-lg shadow-xl">
        <div className="flex items-baseline gap-3">
          <h2 id="create-eval-title" className="m-0 text-lg font-semibold text-fg-primary">
            创建新评估
          </h2>
          <span className="text-xs text-fg-tertiary">
            选 LoRA 文件；样本数 / 指标模型用 Settings 默认
          </span>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold">LoRA 文件</span>
            <span className="text-[11px] text-fg-tertiary">选多个可横向对比</span>
            <span className="flex-1" />
            {ckpts.length > 0 && (
              <button
                type="button"
                className="text-[11px] text-fg-tertiary hover:text-fg underline bg-transparent border-none cursor-pointer p-0"
                onClick={() =>
                  setSelected((prev) =>
                    prev.size === ckpts.length ? new Set() : new Set(ckpts.map((c) => c.path)),
                  )
                }
              >
                {selected.size === ckpts.length ? '清空' : '全选'}
              </button>
            )}
          </div>
          {loading ? (
            <div className="text-xs text-fg-tertiary py-1">读取 LoRA 文件…</div>
          ) : ckpts.length === 0 ? (
            <div className="text-xs text-fg-tertiary py-1">output/ 下没有 LoRA 文件。</div>
          ) : (
            <div
              className="grid gap-1.5 overflow-y-auto"
              style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', maxHeight: 260, padding: 2 }}
            >
              {ckpts.map((c) => {
                const isPicked = selected.has(c.path)
                return (
                  <button
                    key={c.path}
                    type="button"
                    onClick={() => toggle(c.path)}
                    className="font-mono flex items-center gap-1 min-w-0"
                    style={{
                      fontSize: 11,
                      padding: '4px 8px',
                      borderRadius: 'var(--r-md)',
                      border: isPicked ? '1px solid transparent' : '1px solid var(--border-subtle)',
                      background: isPicked ? 'var(--accent-soft)' : 'var(--bg-sunken)',
                      color: isPicked ? 'var(--accent)' : 'var(--fg-secondary)',
                      cursor: 'pointer',
                    }}
                    title={c.path}
                  >
                    <span className="shrink-0">{isPicked ? '✓' : '+'}</span>
                    <span className="truncate flex-1 text-left">{c.label}</span>
                  </button>
                )
              })}
            </div>
          )}
        </div>

        {picked && (
          <div className="text-[11px] text-fg-tertiary">
            {picked.validationImages === 0 ? (
              <span className="text-warn">
                验证集为空 —— 先划分或手动放入验证图，否则评估算不出指标。
              </span>
            ) : (
              <>
                将生成 <span className="font-mono text-fg-secondary">{picked.images}</span> 张图
                （{picked.candidates} 个被测对象 × {picked.validationImages} 张
                {scale?.baseline_enabled ? '，含一组纯底模 baseline 对照' : ''}）、
                1 个评估任务（<span className="font-mono text-fg-secondary">{picked.stages}</span> 个阶段）
              </>
            )}
          </div>
        )}

        {error && (
          <div className="rounded-md border border-err bg-err-soft px-3 py-2 text-xs text-err">
            {error}
          </div>
        )}

        <div className="flex items-center justify-end gap-2">
          <button type="button" onClick={onClose} className="btn btn-ghost btn-sm">
            取消
          </button>
          <button
            type="button"
            onClick={() => void submit()}
            disabled={busy || selected.size === 0}
            className="btn btn-primary btn-sm"
          >
            {busy ? '排队中…' : `创建评估${selected.size ? ` (${selected.size})` : ''}`}
          </button>
        </div>
      </div>
    </div>
  )
}
