/** 数据作业共享工具（DataJobsPanel + QueueDetail 共用）。
 *  R-5 台账合并后作业就是 task（task_type = kind），工具全部按 Task 形状取字段。 */
import type { Task, TaskType } from '../../api/client'

/** 数据视图（light + io 档）的 kind 全集。eval_samples 是 exclusive 档，
 *  归 GPU 视图（锚点 §4-2），不在此列。 */
export const DATA_VIEW_KINDS: TaskType[] = [
  'download', 'preprocess', 'tag', 'reg_build',
  'eval_clip', 'eval_dino', 'eval_tag', 'eval_ccip',
]

/** 全部作业 kind（i18n 标签遍历用，含 eval_samples）。 */
export const JOB_KINDS: TaskType[] = ['eval_samples', ...DATA_VIEW_KINDS]

export const JOB_STATUS_TONE: Record<string, string> = {
  pending: 'neutral', running: 'accent', done: 'ok', failed: 'err', canceled: 'neutral',
}

export function fmtJobAgo(ts: number): string {
  const sec = Math.max(0, Date.now() / 1000 - ts)
  if (sec < 60) return '刚刚'
  if (sec < 3600) return `${Math.floor(sec / 60)}m 前`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h 前`
  return `${Math.floor(sec / 86400)}d 前`
}

export function fmtJobDuration(start: number | null, end: number | null): string {
  if (!start) return '—'
  const e = end ?? Date.now() / 1000
  const sec = Math.max(0, e - start)
  if (sec < 60) return `${sec.toFixed(0)}s`
  const m = Math.floor(sec / 60); const s = Math.floor(sec % 60)
  if (m < 60) return `${m}m ${s}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

export function fmtJobTime(ts: number | null | undefined): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false })
}

/** 作业 kind → 原生步骤页深链（download 是 project 级，其余 version 级；
 *  eval_* 落训练页）。非作业类型返回 null（train/generate 的跳转另有专链）。 */
export function jobJumpPath(task: Task): string | null {
  const kind = task.task_type ?? 'train'
  const pid = task.project_id
  const vid = task.version_id
  if (!pid || !JOB_KINDS.includes(kind)) return null
  if (kind === 'download') return `/projects/${pid}/download`
  if (!vid) return null
  switch (kind) {
    case 'preprocess': return `/projects/${pid}/v/${vid}/preprocess`
    case 'tag': return `/projects/${pid}/v/${vid}/tag`
    case 'reg_build': return `/projects/${pid}/v/${vid}/reg`
    default: return `/projects/${pid}/v/${vid}/train`
  }
}

/** params 值 → 人话字符串。布尔走 field.yes/no，数组逗号连接，对象 JSON 截断。 */
export function fmtParamValue(
  v: unknown, t: (key: string) => string,
): string {
  if (typeof v === 'boolean') return v ? t('field.yes') : t('field.no')
  if (v == null) return '—'
  if (Array.isArray(v)) return v.length ? v.map(String).join(', ') : '—'
  if (typeof v === 'object') {
    const s = JSON.stringify(v)
    return s.length > 200 ? `${s.slice(0, 200)}…` : s
  }
  const s = String(v)
  return s.length > 200 ? `${s.slice(0, 200)}…` : s
}

/** param key → 人话标签；没建映射的退回原 key（全字段显示，一个不藏）。 */
export function paramLabel(key: string, t: (k: string) => string): string {
  const i18nKey = `queue.jobs.param.${key}`
  const label = t(i18nKey)
  return label === i18nKey ? key : label
}
