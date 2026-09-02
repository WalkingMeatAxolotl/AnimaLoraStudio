/** 模型根目录迁移确认 + 进度 modal（Settings → 系统 → 存储位置 → 模型根目录）。
 *
 * 镜像 StudioDataMigrateModal，区别：复制完 **立即生效、无需重启**（models_root()
 * 现读 secrets.models.root）。所以 done 态不给「立即重启」，只给「完成」+ 关闭，
 * 并回调 onDone 让父级刷新当前路径显示。
 *
 * 相位：loading（拉 info 扫描）→ confirm（文件数/大小/顶层明细 + 确认）→
 * running（进度条，SSE 驱动）→ done / error。running 期间 modal 不可关。
 * 目标已有 models 数据时后端回 409 target_conflict → conflict 相位（issue #351）：
 * 「跳过已有文件」（合并补齐，同名保留目标现有版本）/「覆盖已有文件」/ 取消。
 */
import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

import ActionGroup from './ActionGroup'
import Alert from './Alert'
import Button from './Button'
import Modal from './Modal'
import ProgressBar from './ProgressBar'
import { api, type ApiError, type ModelsRootInfo } from '../api/client'
import { formatBytes } from '../lib/useUploadProgress'
import { useEventStream } from '../lib/useEventStream'

type Phase = 'loading' | 'confirm' | 'conflict' | 'running' | 'done' | 'error'

interface ConflictInfo {
  existingFiles: number
  existingBytes: number
  sameNameFiles: number
}

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

export default function ModelsRootMigrateModal({ target, onClose, onDone }: {
  target: string
  onClose: () => void
  /** 迁移成功后回调（父级据此重新拉 getModelsRootInfo 刷新当前路径，无需重启） */
  onDone: () => void
}) {
  const { t } = useTranslation()
  // 用户选的是父目录，后端实际把数据复制到 target/models/（display 用，
  // API 仍传 target，由后端拼接）
  const sep = target.includes('\\') ? '\\' : '/'
  const destination = target.endsWith(sep) ? `${target}models` : `${target}${sep}models`
  const [phase, setPhase] = useState<Phase>('loading')
  const [info, setInfo] = useState<ModelsRootInfo | null>(null)
  const [progress, setProgress] = useState<Progress>(EMPTY_PROGRESS)
  const [conflict, setConflict] = useState<ConflictInfo | null>(null)
  const [error, setError] = useState('')
  const phaseContentRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (phase === 'running') phaseContentRef.current?.focus()
  }, [phase])

  useEffect(() => {
    let cancelled = false
    void api.getModelsRootInfo().then((i) => {
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
    if (evt.type === 'models_root_migrate_progress') {
      setProgress({
        doneFiles: Number(evt.done_files) || 0,
        totalFiles: Number(evt.total_files) || 0,
        doneBytes: Number(evt.done_bytes) || 0,
        totalBytes: Number(evt.total_bytes) || 0,
        currentFile: typeof evt.current_file === 'string' ? evt.current_file : '',
      })
    } else if (evt.type === 'models_root_migrate_done') {
      setPhase((p) => {
        if (p !== 'running') return p
        if (evt.ok) { onDone(); return 'done' }
        setError(typeof evt.error === 'string' ? evt.error : 'unknown')
        return 'error'
      })
    }
  }, {
    // SSE 断线重连期间 done 事件会丢，running 态会卡死（modal 不可关）——
    // 重连时冷拉一次状态快照补齐
    onOpen: () => {
      void api.getModelsRootMigrateStatus().then((s) => {
        setPhase((p) => {
          if (p !== 'running') return p
          if (s.state === 'done') { onDone(); return 'done' }
          if (s.state === 'error') { setError(s.error); return 'error' }
          return p
        })
      }).catch(() => { /* 下次重连再试 */ })
    },
  })

  const handleStart = async (onConflict?: 'skip' | 'overwrite') => {
    setPhase('running')
    setProgress(EMPTY_PROGRESS)
    try {
      await api.startModelsRootMigrate(target, onConflict)
    } catch (e) {
      const err = e as ApiError
      if (err.code === 'models_root.target_conflict') {
        const d = (err.detail ?? {}) as Record<string, unknown>
        setConflict({
          existingFiles: Number(d.existing_files) || 0,
          existingBytes: Number(d.existing_bytes) || 0,
          sameNameFiles: Number(d.same_name_files) || 0,
        })
        setPhase('conflict')
        return
      }
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
    if (phase === 'conflict') {
      return (
        <ActionGroup
          secondary={(
            <>
              <Button variant="ghost" onClick={() => setPhase('confirm')}>
                {t('common.cancel')}
              </Button>
              <Button variant="danger" onClick={() => void handleStart('overwrite')}>
                {t('settings.storage.conflictOverwrite')}
              </Button>
            </>
          )}
          primary={(
            <Button variant="primary" onClick={() => void handleStart('skip')}>
              {t('settings.storage.conflictSkip')}
            </Button>
          )}
        />
      )
    }
    if (phase === 'done' || phase === 'error') {
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
      title={t('settings.storage.modelsMigrateTitle')}
      onClose={onClose}
      closeOnBackdrop={closable}
      closeOnEscape={closable}
      footer={footer}
      size="md"
      role={phase === 'conflict' ? 'alertdialog' : 'dialog'}
    >
      <div
        ref={phaseContentRef}
        data-testid="models-root-migration-phase"
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
              {t('settings.storage.modelsKeepOriginalNote')}
            </div>
          </>
        )}

        {phase === 'conflict' && conflict && (
          <Alert tone="warning" size="sm" title={t('settings.storage.conflictTitle')}>
            <span className="block">
              {t('settings.storage.conflictSummary', {
                path: destination,
                files: conflict.existingFiles,
                size: formatBytes(conflict.existingBytes),
                same: conflict.sameNameFiles,
              })}
            </span>
            <span className="mt-related block text-fg-tertiary">
              {t('settings.storage.conflictHint')}
            </span>
          </Alert>
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
            {t('settings.storage.modelsDoneNote')}
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
