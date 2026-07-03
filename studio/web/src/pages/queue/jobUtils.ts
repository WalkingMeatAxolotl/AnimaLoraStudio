/** 0.17 P-G — 数据作业共享工具（列表 DataJobsPanel + 详情 QueueJobDetail 共用）。 */
import type { Job, JobKind } from '../../api/client'

export const JOB_KINDS: JobKind[] = [
  'download', 'preprocess', 'tag', 'reg_build',
  'eval_samples', 'eval_clip', 'eval_dino', 'eval_tag', 'eval_ccip',
]

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

/** kind → 原生步骤页深链（download 是 project 级，其余 version 级；eval_* 落训练页）。 */
export function jobJumpPath(job: Job): string | null {
  const pid = job.project_id
  const vid = job.version_id
  if (!pid) return null
  if (job.kind === 'download') return `/projects/${pid}/download`
  if (!vid) return null
  switch (job.kind) {
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
