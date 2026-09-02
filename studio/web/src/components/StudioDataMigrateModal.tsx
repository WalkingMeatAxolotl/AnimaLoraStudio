/** studio_data 迁移确认 + 进度 modal（Settings → 系统 → 存储位置）。
 *
 * 四态：loading（拉 info 扫描）→ confirm（文件数/大小/顶层明细 + 确认）→
 * running（进度条，SSE 驱动）→ done / error。
 *
 * running 期间 modal 不可关（一次性维护操作，复制时长有限，用户等完即可；
 * 不引入"后台迁移中再重开看进度"的游离状态）。完成后新位置**重启 server
 * 生效**（指针文件 import 时求值），done 态给「立即重启」。
 */
import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

import ActionGroup from './ActionGroup'
import Alert from './Alert'
import Button from './Button'
import Modal from './Modal'
import ProgressBar from './ProgressBar'
import { api, type StudioDataInfo } from '../api/client'
import { formatBytes } from '../lib/useUploadProgress'
import { useEventStream } from '../lib/useEventStream'

type Phase = 'loading' | 'confirm' | 'running' | 'done' | 'error'

interface Progress {
  doneFiles: number
  totalFiles: number
  doneBytes: number
  totalBytes: number
  currentFile: string
}

const EMPTY_PROGRESS: Progress = {
  doneFiles: 0, totalFiles: 0, doneBytes: 0, totalBytes: 0, currentFile: '',
}

export default function StudioDataMigrateModal({ target, onClose, onRestart }: {
  target: string
  onClose: () => void
  /** done 态「立即重启」—— 复用 Settings 页现成的重启 + 健康轮询逻辑 */
  onRestart: () => void
}) {
  const { t } = useTranslation()
  // 用户选的是父目录，后端实际把数据复制到 target/studio_data/（display 用，
  // API 仍传 target，由后端拼接）
  const sep = target.includes('\\') ? '\\' : '/'
  const destination = target.endsWith(sep) ? `${target}studio_data` : `${target}${sep}studio_data`
  const [phase, setPhase] = useState<Phase>('loading')
  const [info, setInfo] = useState<StudioDataInfo | null>(null)
  const [progress, setProgress] = useState<Progress>(EMPTY_PROGRESS)
  const [error, setError] = useState('')
  const phaseContentRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (phase === 'running') phaseContentRef.current?.focus()
  }, [phase])

  useEffect(() => {
    let cancelled = false
    void api.getStudioDataInfo().then((i) => {
      if (cancelled) return
      setInfo(i)
      setPhase('confirm')
    }).catch((e) => {
      if (cancelled) return
      setError(String(e))
      setPhase('error')
    })
    return () => { cancelled = true }
  }, [])

  // SSE：实时进度 + 完成事件（只在 running 态响应，防外部杂音翻状态）
  useEventStream((evt) => {
    if (evt.type === 'studio_data_migrate_progress') {
      setProgress({
        doneFiles: Number(evt.done_files) || 0,
        totalFiles: Number(evt.total_files) || 0,
        doneBytes: Number(evt.done_bytes) || 0,
        totalBytes: Number(evt.total_bytes) || 0,
        currentFile: typeof evt.current_file === 'string' ? evt.current_file : '',
      })
    } else if (evt.type === 'studio_data_migrate_done') {
      setPhase((p) => {
        if (p !== 'running') return p
        if (evt.ok) return 'done'
        setError(typeof evt.error === 'string' ? evt.error : 'unknown')
        return 'error'
      })
    }
  }, {
    // SSE 断线重连期间 done 事件会丢，running 态会卡死（modal 不可关）——
    // 重连时冷拉一次状态快照补齐
    onOpen: () => {
      void api.getStudioDataMigrateStatus().then((s) => {
        setPhase((p) => {
          if (p !== 'running') return p
          if (s.state === 'done') return 'done'
          if (s.state === 'error') { setError(s.error); return 'error' }
          return p
        })
      }).catch(() => { /* 下次重连再试 */ })
    },
  })

  const handleStart = async () => {
    setPhase('running')
    setProgress(EMPTY_PROGRESS)
    try {
      await api.startStudioDataMigrate(target)
    } catch (e) {
      setError(String(e))
      setPhase('error')
    }
  }

  const closable = phase !== 'running'
  const pct = progress.totalBytes > 0
    ? Math.min(100, Math.round((progress.doneBytes / progress.totalBytes) * 100))
    : null
  const progressDetail = progress.totalBytes > 0
    ? `${progress.doneFiles}/${progress.totalFiles} · ${formatBytes(progress.doneBytes)}/${formatBytes(progress.totalBytes)} · ${pct}%`
    : null

  const footer = (() => {
    if (phase === 'loading') {
      return (
        <ActionGroup
          secondary={<Button variant="ghost" onClick={onClose}>{t('common.close')}</Button>}
        />
      )
    }
    if (phase === 'confirm') {
      return (
        <ActionGroup
          secondary={<Button variant="ghost" onClick={onClose}>{t('common.cancel')}</Button>}
          primary={(
            <Button variant="primary" onClick={() => void handleStart()}>
              {t('settings.storage.startMigrate')}
            </Button>
          )}
        />
      )
    }
    if (phase === 'done') {
      return (
        <ActionGroup
          secondary={<Button variant="ghost" onClick={onClose}>{t('common.close')}</Button>}
          primary={(
            <Button variant="primary" onClick={onRestart}>
              {t('settings.storage.restartNow')}
            </Button>
          )}
        />
      )
    }
    if (phase === 'error') {
      return (
        <ActionGroup
          primary={<Button variant="primary" onClick={onClose}>{t('common.close')}</Button>}
        />
      )
    }
    return undefined
  })()

  return (
    <Modal
      title={t('settings.storage.migrateTitle')}
      onClose={onClose}
      closeOnBackdrop={closable}
      closeOnEscape={closable}
      footer={footer}
      size="md"
    >
      <div
        ref={phaseContentRef}
        data-testid="studio-data-migration-phase"
        tabIndex={-1}
        className="flex flex-col gap-field focus:outline-none"
        aria-busy={phase === 'loading' || phase === 'running'}
      >
        {phase === 'loading' && (
          <div className="flex flex-col gap-related" role="status" aria-live="polite">
            <div className="text-sm text-fg-secondary">{t('settings.storage.scanning')}</div>
            <ProgressBar label={t('settings.storage.scanning')} value={null} />
          </div>
        )}

        {phase === 'confirm' && info?.scan && (
          <>
            <div className="flex flex-col gap-1 text-xs text-fg-secondary">
              <div>
                <span className="text-fg-tertiary">{t('settings.storage.from')}</span>{' '}
                <code className="font-mono">{info.current}</code>
              </div>
              <div>
                <span className="text-fg-tertiary">{t('settings.storage.to')}</span>{' '}
                <code className="font-mono">{destination}</code>
              </div>
            </div>
            <div className="text-sm font-semibold">
              {t('settings.storage.totalLine', {
                files: info.scan.total_files,
                size: formatBytes(info.scan.total_bytes),
              })}
            </div>
            <div
              className="overflow-y-auto rounded-md border border-subtle bg-sunken font-mono text-xs"
              style={{ maxHeight: 200 }}
            >
              {info.scan.entries.map((entry) => (
                <div key={entry.name} className="flex justify-between gap-related border-b border-subtle px-2.5 py-1 last:border-b-0">
                  <span className="truncate">{entry.is_dir ? `${entry.name}/` : entry.name}</span>
                  <span className="shrink-0 text-fg-tertiary">
                    {t('settings.storage.entryMeta', { files: entry.files, size: formatBytes(entry.bytes) })}
                  </span>
                </div>
              ))}
            </div>
            <div className="text-xs text-fg-tertiary">
              {t('settings.storage.keepOriginalNote')}
            </div>
          </>
        )}

        {phase === 'running' && (
          <>
            <div className="text-sm text-fg-secondary" role="status" aria-live="polite">
              {t('settings.storage.migrating')}
            </div>
            <ProgressBar
              label={t('settings.storage.migrating')}
              value={pct}
              valueText={progressDetail ?? undefined}
              size="md"
            />
            <div className="flex justify-between gap-related font-mono text-xs text-fg-tertiary">
              <span className="truncate">{progress.currentFile}</span>
              {progressDetail && <span className="shrink-0">{progressDetail}</span>}
            </div>
          </>
        )}

        {phase === 'done' && (
          <Alert tone="success" size="sm" role="status" aria-live="polite" title={t('settings.storage.doneTitle')}>
            {t('settings.storage.doneRestartNote')}
          </Alert>
        )}

        {phase === 'error' && (
          <Alert tone="danger" size="sm" role="alert">
            <span className="break-all">{t('settings.storage.failed', { error })}</span>
          </Alert>
        )}
      </div>
    </Modal>
  )
}
