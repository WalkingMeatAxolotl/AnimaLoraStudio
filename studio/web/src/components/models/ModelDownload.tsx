import { useTranslation } from 'react-i18next'
import type { ModelDownloadStatus } from '../../api/client'
import { InfoButton } from '../InfoButton'
import { fmtBytes } from '../../lib/formatBytes'
import Badge from '../Badge'
import Button from '../Button'
import Card from '../Card'

export function ModelGroupCard({
  title, helpTooltip, children,
}: {
  title: string
  helpTooltip?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <Card tone="sunken" radius="compact" padding="sm">
      <h4 className="type-panel-title mb-related flex items-center gap-related">
        <span>{title}</span>
        {helpTooltip && <InfoButton>{helpTooltip}</InfoButton>}
      </h4>
      {children}
    </Card>
  )
}

export function ModelStatusBadge({ exists, size, status, fileCount, existsCount, sizeEstimate }: {
  exists: boolean; size: number; status?: ModelDownloadStatus['status']; fileCount?: number; existsCount?: number
  /** 下载前的体积估算（bytes）：未下载时并进徽标（「未下载 · ~250 MB」）。 */
  sizeEstimate?: number
}) {
  const { t } = useTranslation()
  if (status === 'running' || status === 'pending') {
    return <Badge tone="accent" active>{t('settings.downloadInProgress')}</Badge>
  }
  if (status === 'failed') return <Badge tone="danger">{t('status.failed')}</Badge>
  if (status === 'canceled') return <Badge tone="neutral">{t('status.canceled')}</Badge>
  if (exists) {
    return <Badge tone="success">{fmtBytes(size)}{fileCount !== undefined ? ` (${existsCount}/${fileCount})` : ''}</Badge>
  }
  if (fileCount !== undefined && existsCount! > 0) {
    return <Badge tone="warning">{t('settings.partialFiles', { exists: existsCount, total: fileCount })}</Badge>
  }
  return (
    <Badge tone="neutral">
      {sizeEstimate ? t('settings.notDownloadedWithSize', { size: fmtBytes(sizeEstimate) }) : t('settings.notDownloaded')}
    </Badge>
  )
}

export function StatusLabel({ bg, fg, text, pulse }: { bg: string; fg: string; text: string; pulse?: boolean }) {
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded-sm font-mono ${bg} ${fg}`}
      style={pulse ? { animation: 'pulse 1.5s infinite' } : undefined}
    >{text}</span>
  )
}

/** 已下载资产的「删除」按钮（下载的逆操作：用户先删再下载）。 */
export function DeleteAssetButton({ onClick }: { onClick: () => void }) {
  const { t } = useTranslation()
  return (
    <Button onClick={onClick} variant="ghost" size="sm"
      title={t('settings.deleteAssetTitle')}>
      {t('settings.deleteAsset')}
    </Button>
  )
}

export function DownloadButton({ exists, status, busy, onClick, onDelete }: {
  exists: boolean; status?: ModelDownloadStatus['status']; busy: boolean; onClick: () => void
  /** 已下载状态的 action：删除（用户先删再下载）。 */
  onDelete?: () => void
}) {
  const { t } = useTranslation()
  const running = status === 'running' || status === 'pending' || busy
  if (running) {
    return <Button disabled loading variant="secondary" size="sm">{t('common.download')}</Button>
  }
  if (exists) return onDelete ? <DeleteAssetButton onClick={onDelete} /> : null
  return (
    <Button onClick={onClick} variant="secondary" size="sm" title={t('common.download')}>
      {t('common.download')}
    </Button>
  )
}
