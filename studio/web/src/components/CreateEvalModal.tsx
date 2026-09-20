// 「创建新评估」modal —— 从评估作业列表上方发起。
//
// 以前这是指标面板头上的一个内联展开区。评估现在是一类独立作业（有自己的列表和详情
// 页），发起动作就该在列表上方，参数在 modal 里填完再提交 —— 和「新建版本」同款。
// 目前参数只有 checkpoint 选择；样本数 / 指标模型仍走 Settings 默认。
import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, type EvalScale, type EvalSessionInfo, type LoraCkpt } from '../api/client'
import ActionGroup from './ActionGroup'
import Alert from './Alert'
import Button from './Button'
import EmptyState from './EmptyState'
import Modal from './Modal'

export default function CreateEvalModal({
  pid, vid, taskId, onClose, onCreated,
}: {
  pid: number
  vid: number
  /** 溯源：从训练详情发起时带上；从概览发起时留空。 */
  taskId?: number
  onClose: () => void
  onCreated: (session: EvalSessionInfo) => void
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
      onCreated(r.session)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  return (
    <Modal
      title="创建新评估"
      description="选择 LoRA 文件；样本数与指标模型使用 Settings 默认值。"
      onClose={onClose}
      closeOnBackdrop={!busy}
      closeOnEscape={!busy}
      as="form"
      onSubmit={(event) => {
        event.preventDefault()
        void submit()
      }}
      size="lg"
      testId="create-eval-modal"
      footer={(
        <ActionGroup
          secondary={(
            <Button variant="ghost" size="sm" onClick={onClose} disabled={busy}>
              取消
            </Button>
          )}
          primary={(
            <Button
              type="submit"
              variant="primary"
              size="sm"
              loading={busy}
              disabled={selected.size === 0}
            >
              {`创建评估${selected.size ? ` (${selected.size})` : ''}`}
            </Button>
          )}
        />
      )}
    >
      <div className="flex flex-col gap-section">
        <section className="flex flex-col gap-related" aria-labelledby="eval-checkpoint-title">
          <div className="flex items-center gap-related">
            <div className="min-w-0 flex-1">
              <h3 id="eval-checkpoint-title" className="type-panel-title">LoRA 文件</h3>
              <p className="type-page-description mt-1">可选择多个 checkpoint 横向比较。</p>
            </div>
            {ckpts.length > 0 && (
              <Button
                variant="ghost"
                size="xs"
                onClick={() =>
                  setSelected((prev) =>
                    prev.size === ckpts.length ? new Set() : new Set(ckpts.map((c) => c.path)),
                  )
                }
              >
                {selected.size === ckpts.length ? '清空' : '全选'}
              </Button>
            )}
          </div>

          {loading ? (
            <div role="status" className="py-related text-sm text-fg-secondary">
              读取 LoRA 文件…
            </div>
          ) : ckpts.length === 0 ? (
            <EmptyState
              embedded
              size="sm"
              description="output/ 下没有 LoRA 文件。"
            />
          ) : (
            <div
              className="grid max-h-[16.25rem] grid-cols-[repeat(auto-fill,minmax(8.125rem,1fr))] gap-related overflow-y-auto p-related"
              aria-label="选择要评估的 LoRA 文件"
            >
              {ckpts.map((c) => {
                const isPicked = selected.has(c.path)
                return (
                  <Button
                    key={c.path}
                    variant="secondary"
                    size="sm"
                    onClick={() => toggle(c.path)}
                    aria-pressed={isPicked}
                    className="min-w-0 justify-start font-mono"
                    title={c.path}
                  >
                    <span className="truncate text-left">{c.label}</span>
                  </Button>
                )
              })}
            </div>
          )}
        </section>

        {picked && (
          picked.validationImages === 0 ? (
            <Alert tone="warning" size="sm">
              验证集为空——先划分或手动放入验证图，否则评估算不出指标。
            </Alert>
          ) : (
            <div className="text-sm text-fg-secondary">
              将生成 <span className="font-mono tabular-nums text-fg-primary">{picked.images}</span> 张图
              （{picked.candidates} 个被测对象 × {picked.validationImages} 张
              {scale?.baseline_enabled ? '，含一组纯底模 baseline 对照' : ''}）、
              1 个评估任务（<span className="font-mono tabular-nums text-fg-primary">{picked.stages}</span> 个阶段）
            </div>
          )
        )}

        {error && (
          <Alert tone="danger" size="sm" role="alert">
            {error}
          </Alert>
        )}
      </div>
    </Modal>
  )
}
