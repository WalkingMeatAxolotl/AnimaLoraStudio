/** ADR-0007 §11.3-B: version.status (5 enum) → 颜色映射。
 *
 *  status：preparing / training / completed / failed / canceled
 *  与老 StageBadge 平行存在；PR-5 v9 destructive 后老 StageBadge 配合 stage
 *  字段一起删，VersionStatusBadge 成为唯一 version 状态展示组件。
 */
import { useTranslation } from 'react-i18next'
import type { VersionPhase, VersionStatus } from '../api/client'

const DOT_RUNNING = (
  <span className="dot dot-running" style={{ flexShrink: 0 }} />
)

type StatusEntry = { badge: string; key: string; dot?: true }

const STATUS_MAP: Record<VersionStatus, StatusEntry> = {
  preparing: { badge: 'badge-warn',    key: 'versionStatus.preparing' },
  training:  { badge: 'badge-accent',  key: 'versionStatus.training', dot: true },
  completed: { badge: 'badge-ok',      key: 'versionStatus.completed' },
  failed:    { badge: 'badge-err',     key: 'versionStatus.failed' },
  canceled:  { badge: 'badge-neutral', key: 'versionStatus.canceled' },
}

/** preparing 时 badge 后缀的 phase 文案。可选 phase（preprocessing /
 * regularizing）刻意缺席 —— 用户决策：optional 步骤不值得占 badge 空间，
 * cursor 落在上面时只显"准备中"。 */
const PHASE_SUFFIX_KEY: Partial<Record<VersionPhase, string>> = {
  curating: 'versionPhase.curating',
  tagging: 'versionPhase.tagging',
  editing: 'versionPhase.editing',
  ready: 'versionPhase.ready',
}

export default function VersionStatusBadge({
  status,
  phase,
}: {
  status: VersionStatus | null | undefined
  /** 传了且 status=preparing 时显示"准备中 · 打标"式后缀（项目卡片用）。 */
  phase?: VersionPhase | null
}) {
  const { t } = useTranslation()
  if (!status) return null
  const entry = STATUS_MAP[status] ?? { badge: 'badge-neutral', key: status }
  const suffixKey =
    status === 'preparing' && phase ? PHASE_SUFFIX_KEY[phase] : undefined
  return (
    <span className={`badge ${entry.badge}`}>
      {entry.dot && DOT_RUNNING}
      {STATUS_MAP[status] ? t(entry.key) : status}
      {suffixKey ? ` · ${t(suffixKey)}` : ''}
    </span>
  )
}
