// 评估的**指标**面板：指标卡 + 逐 checkpoint 表格 + 中断 / 重试。
//
// 只管指标。样图矩阵是同级的另一个 tab（EvalSampleGrid），发起评估是列表上方的
// 「创建新评估」—— 都不在这里，一个 tab 一件事。
//
// `sessionId` 钉死看哪一次时（评估作业详情页）不显示历史切换；不钉时（训练详情页）
// 在 taskId 名下的历史里选，默认最新那次。
import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, type EvalMetricResult, type EvalMetricState, type EvalSessionSummary } from '../api/client'
import { evalProgressFromResults } from '../lib/useEvalProgress'
import { InfoButton } from './InfoButton'
import { SeriesChart } from './SeriesChart'

// ── EvalMetricsPanel ──────────────────────────────────────────────────────

const EVAL_METRIC_KEYS = ['clip_t', 'clip_i', 'dino_i', 'ccip_i', 'tag_recall'] as const
type EvalMetricKey = typeof EVAL_METRIC_KEYS[number]
// 核心指标常显；动漫域新指标默认关，只在算过（状态非 not_run）时才显示卡片/列。
const CORE_METRIC_KEYS = new Set<EvalMetricKey>(['clip_t', 'clip_i', 'dino_i'])

const EVAL_LABELS: Record<EvalMetricKey, string> = {
  clip_t: 'CLIP-T',
  clip_i: 'CLIP-I',
  dino_i: 'DINO-I',
  ccip_i: 'CCIP-I',
  tag_recall: 'Tag-Recall',
}

// 每个指标一种线色（并排区分）。深色背景上高对比、可辨。
const EVAL_COLORS: Record<EvalMetricKey, string> = {
  clip_t: '#3fb950',
  clip_i: '#58a6ff',
  dino_i: '#bc8cff',
  ccip_i: '#f778ba',
  tag_recall: '#e3b341',
}

const EVAL_DESCRIPTIONS: Record<EvalMetricKey, string> = {
  clip_t: '生成图和 prompt 文本的 CLIP 相似度，用来看 prompt following；越高越好。',
  clip_i: '生成图和参考图的 CLIP 图像相似度，用来看整体视觉相似度；越高越好。',
  dino_i: '生成图和参考图的 DINO 图像特征相似度，用来看主体或风格特征是否学到；越高越好。',
  ccip_i: '生成图被参考集判为同一动漫角色的比例（CCIP 动漫域角色身份保真）；仅单角色角色 LoRA 有意义；越高越好。',
  tag_recall: '对生成图回标，prompt 里 booru tag 的召回率（动漫原生 prompt following）；仅 booru-tag caption 有意义；越高越好。',
}

function checkpointSortValue(result: EvalMetricResult, index: number): number {
  const value = result.checkpoint?.value
  if (typeof value === 'number') return value
  return result.updated_at ?? result.created_at ?? index
}

function checkpointLabel(result: EvalMetricResult): string {
  return result.checkpoint?.label
    || result.checkpoint?.path?.split(/[\\/]/).pop()
    || result.run_id
}

function metricState(result: EvalMetricResult, key: EvalMetricKey): EvalMetricState | undefined {
  return result.metric_states?.[key]
}

function metricValue(result: EvalMetricResult, key: EvalMetricKey): number | null {
  const stateValue = metricState(result, key)?.value
  if (typeof stateValue === 'number' && Number.isFinite(stateValue)) return stateValue
  const raw = result.metrics?.[key]
  return typeof raw === 'number' && Number.isFinite(raw) ? raw : null
}

function formatEvalValue(value: number | null, state?: EvalMetricState): string {
  if (value != null) return value.toFixed(4)
  const status = state?.status || 'not_run'
  if (status === 'pending') return 'pending'
  if (status === 'running') return 'running'
  if (status === 'failed') return 'failed'
  if (status === 'unavailable') return 'n/a'
  return '--'
}

function stateTone(state?: EvalMetricState, value?: number | null): 'ok' | 'warn' | 'err' | 'muted' {
  if (state?.status === 'failed') return 'err'
  if (state?.status === 'pending' || state?.status === 'running') return 'warn'
  if (value != null || state?.status === 'done') return 'ok'
  return 'muted'
}

function toneClass(tone: 'ok' | 'warn' | 'err' | 'muted'): string {
  if (tone === 'ok') return 'text-ok'
  if (tone === 'warn') return 'text-warn'
  if (tone === 'err') return 'text-err'
  return 'text-fg-tertiary'
}

// 一行 checkpoint 的统一进度文案 + 色调（不管 inline/训练后/手动触发，都从同一份
// run.json/metrics.json 状态推：先出图（sample_run.summary done/total），再算指标。
function evalRowStatus(result: EvalMetricResult): { text: string; tone: 'ok' | 'warn' | 'err' | 'muted' } {
  const s = result.sample_run?.summary
  const total = s?.total ?? 0
  const sampleDone = (s?.done ?? 0) + (s?.failed ?? 0)
  const samplingActive = total > 0 && sampleDone < total &&
    (result.status === 'running' || result.status === 'pending' || (s?.running ?? 0) > 0 || (s?.pending ?? 0) > 0)
  if (samplingActive) return { text: `出图 ${s?.done ?? 0}/${total}`, tone: 'warn' }
  const metricActive = EVAL_METRIC_KEYS.some((k) => {
    const st = metricState(result, k)?.status
    return st === 'pending' || st === 'running'
  })
  if (metricActive) return { text: '算指标…', tone: 'warn' }
  if (result.status === 'failed') return { text: '失败', tone: 'err' }
  if (result.status === 'done') return { text: '完成', tone: 'ok' }
  return { text: result.status, tone: 'muted' }
}

export function EvalMetricsPanel({
  pid, vid, taskId, sessionId, connected,
}: {
  pid: number | undefined
  vid: number | undefined
  /** 带上 = 只在这次训练触发的评估里选；不带 = 整个 version。 */
  taskId?: number
  /** 钉死看哪一次（评估作业详情页）。给了就不显示历史切换。 */
  sessionId?: number | null
  /** SSE 是否连着 —— 训练页据此决定要不要持续轮询；独立页传 false 即可。 */
  connected: boolean
}) {
  const [payload, setPayload] = useState<Awaited<ReturnType<typeof api.listEvalMetrics>> | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // 历史评估（#465 起一次评估一个 Session，全部留档）。null = 看最新那次。
  const [sessions, setSessions] = useState<EvalSessionSummary[]>([])
  const [pickedSession, setPickedSession] = useState<number | null>(sessionId ?? null)

  const load = useCallback(async (quiet = false) => {
    if (!pid || !vid) return
    if (!quiet) setLoading(true)
    try {
      const next = await api.listEvalMetrics(
        pid, vid, taskId, (sessionId ?? pickedSession) ?? undefined,
      )
      setPayload(next)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      if (!quiet) setLoading(false)
    }
    try {
      const { sessions: list } = await api.listEvalSessions(pid, vid, taskId)
      setSessions(list)
    } catch {
      // 历史切换是辅助信息，拉失败不打扰
    }
  }, [pid, vid, taskId, sessionId, pickedSession])

  useEffect(() => {
    setPayload(null)
    setError(null)
    if (!pid || !vid) return
    void load()
  }, [pid, vid, load])

  // 切 task 时清掉选中的 Session（那是上一个 task 的历史）
  useEffect(() => { setPickedSession(null) }, [taskId])

  // 当前在看哪个 Session：钉死的优先，其次下拉选中，最后后端给的那个（最新）。
  // 存量回落（老项目没有 Session）时为 null。
  const activeSessionId = sessionId ?? pickedSession ?? payload?.session?.id ?? null
  // 列表里那条更新（轮询每 5s 重拉），拿不到再退回 /eval/metrics 带的快照
  const activeSession = useMemo(
    () => sessions.find((s) => s.id === activeSessionId) ?? payload?.session ?? null,
    [sessions, activeSessionId, payload?.session],
  )

  const results = useMemo(() => {
    return [...(payload?.results ?? [])]
      .sort((a, b) => checkpointSortValue(a, 0) - checkpointSortValue(b, 0))
  }, [payload?.results])

  // baseline run（纯底模对照）不作为 checkpoint 展示——只用来给各 checkpoint 算 Δ
  // （后端已挂在 result.delta）。每次「运行评估」会自动清空上一轮，所以这里永远只有
  // 这次 run 的结果，每个 checkpoint 一条，无需跨轮去重。
  const displayResults = useMemo(
    () => results.filter((r) => !r.baseline),
    [results],
  )

  const hasActiveMetric = useMemo(() => {
    return results.some((result) =>
      EVAL_METRIC_KEYS.some((key) => {
        const status = metricState(result, key)?.status
        return status === 'pending' || status === 'running'
      }),
    )
  }, [results])

  // Session 还在跑（含「出图」阶段——此时各 metric 状态还是 pending，hasActiveMetric
  // 抓不到）。重跑评估在已 done 的 task 上时，靠这个让轮询继续。
  const hasActiveJob = useMemo(
    () => sessions.some((s) => s.status === 'pending' || s.status === 'running'),
    [sessions],
  )

  // 核心指标常显；动漫域新指标默认关，只有算过（状态非 not_run）才显示，避免空卡。
  const displayKeys = useMemo<EvalMetricKey[]>(
    () => EVAL_METRIC_KEYS.filter((k) =>
      CORE_METRIC_KEYS.has(k) ||
      displayResults.some((r) => {
        const s = metricState(r, k)?.status
        return s != null && s !== 'not_run'
      }),
    ),
    [displayResults],
  )

  // 训练结束后评估进度：复用现有 results 聚合「评估中 done/total」，给面板头部用
  const evalAgg = useMemo(() => evalProgressFromResults(results), [results])

  useEffect(() => {
    if (!pid || !vid) return
    if (!connected && !hasActiveMetric && !hasActiveJob) return
    const id = window.setInterval(() => void load(true), 5000)
    return () => window.clearInterval(id)
  }, [connected, hasActiveMetric, hasActiveJob, load, pid, vid])

  // 中断 / 重试当前 Session。重试走断点续跑（只补没跑完的候选和指标），所以对
  // 「跑了 180 个 checkpoint 才崩」的场景不会全部重来。
  const [sessionBusy, setSessionBusy] = useState(false)
  const [runMsg, setRunMsg] = useState<string | null>(null)
  const sessionAction = useCallback(async (kind: 'cancel' | 'retry') => {
    if (!pid || !vid || !activeSessionId) return
    setSessionBusy(true)
    setRunMsg(null)
    try {
      if (kind === 'cancel') {
        await api.cancelEvalSession(pid, vid, activeSessionId)
        setRunMsg(`已请求中断评估 #${activeSessionId}，已算出的结果保留`)
      } else {
        await api.retryEvalSession(pid, vid, activeSessionId)
        setRunMsg(`已重新排队评估 #${activeSessionId}（跳过已完成的部分）`)
      }
      void load(true)
    } catch (err) {
      setRunMsg(err instanceof Error ? err.message : String(err))
    } finally {
      setSessionBusy(false)
    }
  }, [pid, vid, activeSessionId, load])

  const latestByKey = useMemo(() => {
    const out: Partial<Record<EvalMetricKey, { result: EvalMetricResult; value: number | null; state?: EvalMetricState }>> = {}
    for (const key of EVAL_METRIC_KEYS) {
      for (let i = displayResults.length - 1; i >= 0; i--) {
        const result = displayResults[i]
        const state = metricState(result, key)
        const value = metricValue(result, key)
        if (value != null || state?.status) {
          out[key] = { result, value, state }
          break
        }
      }
    }
    return out
  }, [displayResults])

  const seriesByKey = useMemo(() => {
    const out = Object.fromEntries(
      EVAL_METRIC_KEYS.map((k) => [k, [] as Array<{ x: number; value: number }>]),
    ) as Record<EvalMetricKey, Array<{ x: number; value: number }>>
    displayResults.forEach((result, index) => {
      const x = checkpointSortValue(result, index)
      for (const key of EVAL_METRIC_KEYS) {
        const value = metricValue(result, key)
        if (value != null) out[key].push({ x, value })
      }
    })
    return out
  }, [displayResults])

  // 各指标的纯底模 baseline 值（画成图上的水平参考线）。后端给每条非 baseline 结果
  // 都挂了相同的 baseline_metrics，取任一即可。
  const baselineByKey = useMemo(() => {
    const out: Partial<Record<EvalMetricKey, number>> = {}
    const bm = displayResults.find(
      (r) => r.baseline_metrics && Object.keys(r.baseline_metrics).length,
    )?.baseline_metrics
    if (bm) {
      for (const key of EVAL_METRIC_KEYS) {
        const v = bm[key]
        if (typeof v === 'number' && Number.isFinite(v)) out[key] = v
      }
    }
    return out
  }, [displayResults])

  if (!pid || !vid) {
    return (
      <div className="card px-4 py-3 text-sm text-fg-tertiary">
        当前任务未绑定项目版本，暂不能读取指标。
      </div>
    )
  }

  return (
    <div className="card p-4 flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <div className="text-sm font-semibold">指标</div>
        <span className="flex-1" />
        {/* 历史评估切换：一次评估一个 Session，全部留档（#465）。钉死看某一次时
            （评估作业详情页）没有「切一次」的语义，不显示。 */}
        {sessionId == null && sessions.length > 1 && (
          <select
            className="font-mono cursor-pointer max-w-[260px] truncate"
            style={{
              fontSize: 11,
              padding: '4px 8px',
              borderRadius: 'var(--r-md)',
              border: '1px solid var(--border-subtle)',
              background: 'var(--bg-sunken)',
              color: 'var(--fg-secondary)',
            }}
            value={pickedSession ?? sessions[0].id}
            onChange={(e) => setPickedSession(Number(e.target.value))}
            title="查看历史评估"
            aria-label="选择要查看的评估"
          >
            {sessions.map((s, i) => (
              <option key={s.id} value={s.id}>
                {i === 0 ? '最新 · ' : ''}
                {new Date((s.created_at ?? 0) * 1000).toLocaleString()}
                {` · ${s.candidate_count} 个候选`}
                {s.status !== 'done' ? ` · ${s.status}` : ''}
              </option>
            ))}
          </select>
        )}
        {evalAgg.active && (
          <span
            className="badge badge-accent text-xs"
            title="正在用验证集对各 checkpoint 出图并算指标，完成后消失"
          >
            <span className="dot dot-running" />
            评估中 {evalAgg.done}/{evalAgg.total}
          </span>
        )}
        {loading && <span className="text-xs text-fg-tertiary">读取中…</span>}
        {/* 一次评估就是一个作业，中断 / 重试直接挂在结果面板上——用户看结果的地方
            就是他想操作的地方，不必先去队列里翻出那条 task。 */}
        {activeSession && (activeSession.status === 'pending' || activeSession.status === 'running') && (
          <button
            type="button"
            onClick={() => void sessionAction('cancel')}
            disabled={sessionBusy}
            className="btn btn-ghost btn-sm"
            title="中断这次评估，已算出的结果保留"
          >
            中断
          </button>
        )}
        {activeSession
          && ['failed', 'canceled', 'partial'].includes(activeSession.status) && (
          <button
            type="button"
            onClick={() => void sessionAction('retry')}
            disabled={sessionBusy}
            className="btn btn-secondary btn-sm"
            title="重跑没跑完的候选和指标（已完成的跳过）"
          >
            重试
          </button>
        )}
        <button
          type="button"
          onClick={() => void load()}
          className="btn btn-secondary btn-sm"
        >
          刷新
        </button>
      </div>

      {/* Session 终止原因 —— 之前只写在 DB 和作业日志里，面板上看不到，用户只知道
          「一直在转」。 */}
      {activeSession?.error
        && ['failed', 'canceled'].includes(activeSession.status) && (
        <div className="rounded-md border border-err bg-err-soft px-3 py-2 text-xs text-err">
          评估{activeSession.status === 'canceled' ? '已中断' : '失败'}：{activeSession.error}
        </div>
      )}
      {runMsg && <div className="text-[11px] text-fg-tertiary">{runMsg}</div>}

      {error ? (
        <div className="rounded-md border border-err bg-err-soft px-3 py-2 text-sm text-err">
          评估指标读取失败：{error}
        </div>
      ) : results.length === 0 ? (
        <div className="rounded-md border border-dashed border-subtle px-3 py-3 text-sm text-fg-tertiary">
          暂无评估结果。在概览的「评估」里点「创建新评估」选 LoRA 文件，或在训练配置开启「训练后指标评估」，训练结束后自动用验证集算 CLIP-T、CLIP-I、DINO-I。
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
            {displayKeys.map((key) => {
              const latest = latestByKey[key]
              const tone = stateTone(latest?.state, latest?.value)
              const series = seriesByKey[key].map((p) => ({ step: p.x, value: p.value }))
              return (
                <div key={key} className="card p-4 flex flex-col min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1 shrink-0">
                    <span className="inline-flex items-center gap-1.5 text-sm font-semibold">
                      {EVAL_LABELS[key]}
                      <InfoButton ariaLabel={`${EVAL_LABELS[key]} 指标说明`}>
                        <p>{EVAL_DESCRIPTIONS[key]}</p>
                      </InfoButton>
                    </span>
                    <span className={`text-xs font-mono ${toneClass(tone)}`}>
                      {latest?.state?.status ?? 'not_run'}
                    </span>
                  </div>
                  <div className="flex items-baseline gap-2 shrink-0 mb-1.5">
                    <span className={`text-2xl font-semibold font-mono tabular-nums ${toneClass(tone)}`}>
                      {formatEvalValue(latest?.value ?? null, latest?.state)}
                    </span>
                    {(() => {
                      const d = latest?.result.delta?.[key]
                      return d != null ? (
                        <span
                          className={`text-xs font-mono tabular-nums shrink-0 ${d >= 0 ? 'text-ok' : 'text-err'}`}
                          title="相对纯底模 baseline 的净增益 Δ"
                        >
                          {d >= 0 ? '+' : ''}{d.toFixed(4)}
                        </span>
                      ) : null
                    })()}
                    <span className="text-[11px] text-fg-tertiary truncate">
                      {latest ? checkpointLabel(latest.result) : '等待指标'}
                    </span>
                  </div>
                  <SeriesChart
                    data={series}
                    rawColor={EVAL_COLORS[key]}
                    smoothColor={EVAL_COLORS[key]}
                    emaAlpha={1}
                    yFormat={(v) => v.toFixed(4)}
                    height={132}
                    refLine={baselineByKey[key]}
                  />
                </div>
              )
            })}
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="text-fg-tertiary">
                <tr className="border-b border-subtle">
                  <th className="text-left font-medium py-1.5 pr-3">checkpoint</th>
                  {displayKeys.map((key) => (
                    <th key={key} className="text-right font-medium py-1.5 px-2">
                      {EVAL_LABELS[key]}
                    </th>
                  ))}
                  <th className="text-right font-medium py-1.5 pl-3">状态</th>
                </tr>
              </thead>
              <tbody>
                {displayResults.slice(-8).reverse().map((result) => {
                  const rowStatus = evalRowStatus(result)
                  return (
                    <tr key={result.run_id} className="border-b border-subtle last:border-0">
                      <td className="py-1.5 pr-3 max-w-[220px] truncate font-mono" title={checkpointLabel(result)}>
                        {checkpointLabel(result)}
                      </td>
                      {displayKeys.map((key) => {
                        const state = metricState(result, key)
                        const value = metricValue(result, key)
                        const tone = stateTone(state, value)
                        const d = result.delta?.[key]
                        return (
                          <td key={key} className={`py-1.5 px-2 text-right font-mono tabular-nums ${toneClass(tone)}`}>
                            {formatEvalValue(value, state)}
                            {d != null && value != null && (
                              <span
                                className={`ml-1 text-[10px] ${d >= 0 ? 'text-ok' : 'text-err'}`}
                                title="相对纯底模 baseline 的 Δ"
                              >
                                {d >= 0 ? '+' : ''}{d.toFixed(4)}
                              </span>
                            )}
                          </td>
                        )
                      })}
                      <td className="py-1.5 pl-3 text-right font-mono">
                        <span className={toneClass(rowStatus.tone)}>{rowStatus.text}</span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

    </div>
  )
}
