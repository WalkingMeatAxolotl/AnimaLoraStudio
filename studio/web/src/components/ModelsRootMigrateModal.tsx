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
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

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
    : 0

  return (
    <div
      className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center"
      onClick={closable ? onClose : undefined}
    >
      <div
        className="bg-elevated border border-dim rounded-lg shadow-xl w-[560px] max-h-[80vh] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="px-4 py-3 border-b border-subtle flex items-center gap-2 shrink-0">
          <h3 className="m-0 text-sm font-semibold flex-1 text-fg-primary">
            {t('settings.storage.modelsMigrateTitle')}
          </h3>
          {closable && (
            <button className="btn btn-ghost text-xs" onClick={onClose} aria-label={t('common.close')}>×</button>
          )}
        </header>

        <div className="p-4 flex flex-col gap-3 overflow-y-auto">
          {phase === 'loading' && (
            <div className="text-sm text-fg-tertiary py-6 text-center">
              {t('settings.storage.scanning')}
            </div>
          )}

          {phase === 'confirm' && info?.scan && (
            <>
              <div className="text-xs text-fg-secondary flex flex-col gap-1">
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
                className="bg-sunken border border-subtle rounded-md text-xs font-mono overflow-y-auto"
                style={{ maxHeight: 200 }}
              >
                {info.scan.entries.map((e) => (
                  <div key={e.name} className="flex justify-between gap-2 px-2.5 py-1 border-b border-subtle last:border-b-0">
                    <span className="truncate">{e.is_dir ? `${e.name}/` : e.name}</span>
                    <span className="text-fg-tertiary shrink-0">
                      {t('settings.storage.entryMeta', { files: e.files, size: formatBytes(e.bytes) })}
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
            <>
              <div className="text-sm font-semibold">
                {t('settings.storage.conflictTitle')}
              </div>
              <div className="text-xs text-fg-secondary">
                {t('settings.storage.conflictSummary', {
                  path: destination,
                  files: conflict.existingFiles,
                  size: formatBytes(conflict.existingBytes),
                  same: conflict.sameNameFiles,
                })}
              </div>
              <div className="text-xs text-fg-tertiary">
                {t('settings.storage.conflictHint')}
              </div>
            </>
          )}

          {phase === 'running' && (
            <>
              <div className="text-sm">{t('settings.storage.migrating')}</div>
              <div className="h-2 rounded-full bg-sunken border border-subtle overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${pct}%`,
                    background: 'var(--accent)',
                    transition: 'width 200ms linear',
                  }}
                />
              </div>
              <div className="text-xs text-fg-tertiary font-mono flex justify-between gap-2">
                <span className="truncate">{progress.currentFile}</span>
                <span className="shrink-0">
                  {progress.doneFiles}/{progress.totalFiles} · {formatBytes(progress.doneBytes)}/{formatBytes(progress.totalBytes)} · {pct}%
                </span>
              </div>
            </>
          )}

          {phase === 'done' && (
            <>
              <div className="text-sm text-ok">{t('settings.storage.doneTitle')}</div>
              <div className="text-xs text-fg-secondary">
                {t('settings.storage.modelsDoneNote')}
              </div>
            </>
          )}

          {phase === 'error' && (
            <div className="text-sm text-err break-all">
              {t('settings.storage.failed', { error })}
            </div>
          )}
        </div>

        <footer className="px-4 py-3 border-t border-subtle flex justify-end gap-2 shrink-0">
          {phase === 'confirm' && (
            <>
              <button className="btn btn-ghost" onClick={onClose}>{t('common.cancel')}</button>
              <button className="btn btn-primary" onClick={() => void handleStart()}>
                {t('settings.storage.startMigrate')}
              </button>
            </>
          )}
          {phase === 'conflict' && (
            <>
              <button className="btn btn-ghost" onClick={() => setPhase('confirm')}>
                {t('common.cancel')}
              </button>
              <button className="btn btn-danger" onClick={() => void handleStart('overwrite')}>
                {t('settings.storage.conflictOverwrite')}
              </button>
              <button className="btn btn-primary" onClick={() => void handleStart('skip')}>
                {t('settings.storage.conflictSkip')}
              </button>
            </>
          )}
          {(phase === 'done' || phase === 'error') && (
            <button className="btn btn-primary" onClick={onClose}>{t('common.close')}</button>
          )}
        </footer>
      </div>
    </div>
  )
}
