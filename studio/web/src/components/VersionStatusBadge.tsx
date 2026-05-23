/** ADR-0007 §11.3-B: version.status (5 enum) → 颜色映射。
 *
 *  status：preparing / training / completed / failed / canceled
 *  与老 StageBadge 平行存在；PR-5 v9 destructive 后老 StageBadge 配合 stage
 *  字段一起删，VersionStatusBadge 成为唯一 version 状态展示组件。
 */
import { useTranslation } from 'react-i18next'
import type { VersionStatus } from '../api/client'

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

export default function VersionStatusBadge({ status }: { status: VersionStatus | null | undefined }) {
  const { t } = useTranslation()
  if (!status) return null
  const entry = STATUS_MAP[status] ?? { badge: 'badge-neutral', key: status }
  return (
    <span className={`badge ${entry.badge}`}>
      {entry.dot && DOT_RUNNING}
      {STATUS_MAP[status] ? t(entry.key) : status}
    </span>
  )
}
