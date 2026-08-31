/** ADR-0007 §11.3-B: version.status (5 enum) → 颜色映射。
 *
 *  status：preparing / training / completed / failed / canceled
 *  与老 StageBadge 平行存在；PR-5 v9 destructive 后老 StageBadge 配合 stage
 *  字段一起删，VersionStatusBadge 成为唯一 version 状态展示组件。
 */
import { useTranslation } from 'react-i18next'
import type { VersionPhase, VersionStatus } from '../api/client'
import Badge, { type BadgeTone } from './Badge'

type StatusEntry = { tone: BadgeTone; key: string; active?: true }

const STATUS_MAP: Record<VersionStatus, StatusEntry> = {
  preparing: { tone: 'warning', key: 'versionStatus.preparing' },
  training:  { tone: 'accent',  key: 'versionStatus.training', active: true },
  completed: { tone: 'success', key: 'versionStatus.completed' },
  failed:    { tone: 'danger',  key: 'versionStatus.failed' },
  canceled:  { tone: 'neutral', key: 'versionStatus.canceled' },
}

/** preparing 时 badge 后缀的 phase 文案（PR #265 评审改：可选步骤
 * preprocessing / regularizing 也显示 —— cursor 在哪就显示哪）。 */
const PHASE_SUFFIX_KEY: Record<VersionPhase, string> = {
  curating: 'versionPhase.curating',
  preprocessing: 'versionPhase.preprocessing',
  tagging: 'versionPhase.tagging',
  editing: 'versionPhase.editing',
  regularizing: 'versionPhase.regularizing',
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
  const entry = STATUS_MAP[status] ?? { tone: 'neutral' as const, key: status }
  const suffixKey =
    status === 'preparing' && phase ? PHASE_SUFFIX_KEY[phase] : undefined
  return (
    <Badge tone={entry.tone} active={entry.active}>
      {STATUS_MAP[status] ? t(entry.key) : status}
      {suffixKey ? ` · ${t(suffixKey)}` : ''}
    </Badge>
  )
}
