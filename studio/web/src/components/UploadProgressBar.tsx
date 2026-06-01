/**
 * UploadProgressBar — 浏览器上传共用进度条 UI。
 *
 * 三个阶段视觉区分：
 *   - uploading  → 实进度 + speed + ETA
 *   - processing → 100% 满条 + "处理中…"（server 同步解包 / 落盘的等待）
 *   - error      → 红色边 + 错误信息
 *
 * 业务侧只负责喂 state（来自 useUploadProgress），其余完全 stateless。
 */
import { useTranslation } from 'react-i18next'

import {
  formatBytes,
  formatEta,
  formatSpeed,
  type UploadProgressState,
} from '../lib/useUploadProgress'

interface Props {
  state: UploadProgressState
  className?: string
}

export default function UploadProgressBar({ state, className }: Props) {
  const { t } = useTranslation()
  if (state.phase === 'idle') return null

  const pct =
    state.total > 0
      ? Math.min(100, Math.max(0, (state.loaded / state.total) * 100))
      : state.phase === 'processing' || state.phase === 'done'
        ? 100
        : 0

  const isErr = state.phase === 'error'
  const isUploading = state.phase === 'uploading'
  const isProcessing = state.phase === 'processing'

  return (
    <div
      className={`flex flex-col gap-1 ${className ?? ''}`}
      role="status"
      aria-live="polite"
    >
      <div
        className={`relative w-full h-1.5 rounded-sm overflow-hidden ${
          isErr ? 'bg-err-soft' : 'bg-overlay'
        }`}
      >
        <div
          className={`absolute inset-y-0 left-0 transition-[width] duration-150 ease-out ${
            isErr ? 'bg-err' : isProcessing ? 'bg-accent animate-pulse' : 'bg-accent'
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex items-center gap-2 text-[11px] text-fg-tertiary font-mono">
        {isErr ? (
          <span className="text-err truncate">
            {t('upload.failed')}: {state.error}
          </span>
        ) : isProcessing ? (
          <span>{t('upload.processing')}</span>
        ) : isUploading ? (
          <>
            <span>{pct.toFixed(0)}%</span>
            <span>·</span>
            <span>
              {formatBytes(state.loaded)}
              {state.total > 0 && ` / ${formatBytes(state.total)}`}
            </span>
            {state.speedBps > 0 && (
              <>
                <span>·</span>
                <span>{formatSpeed(state.speedBps)}</span>
              </>
            )}
            {state.etaSec != null && (
              <>
                <span>·</span>
                <span>
                  {t('upload.etaPrefix')} {formatEta(state.etaSec)}
                </span>
              </>
            )}
          </>
        ) : null}
      </div>
    </div>
  )
}
