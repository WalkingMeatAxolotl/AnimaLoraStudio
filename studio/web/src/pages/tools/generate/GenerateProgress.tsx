import { useTranslation } from 'react-i18next'

/** 生成进度条：单图采样 step / 多图 batch 进度 + phase 文字。
 *
 * 来源：daemon 推 SSE
 *   - generate_image_started  { batch_idx, batch_total, total_steps }
 *   - generate_preview_step   { step, total, image_b64? }
 *
 * Generate.tsx 把这俩聚合成 progress prop 传进来。
 */

export interface GenerateProgress {
  /** 当前在跑哪一张（多张图 / XY 时；count=1 单图时 batchTotal=1） */
  batchIdx: number | null
  batchTotal: number | null
  /** 当前图采样到第几步 */
  currentStep: number | null
  totalSteps: number | null
}

export default function GenerateProgressBar({
  busy, progress,
}: {
  busy: boolean
  progress: GenerateProgress
}) {
  const { t } = useTranslation()
  if (!busy && progress.currentStep == null) return null

  const showBatch =
    progress.batchTotal != null && progress.batchTotal > 1
  const stepPct =
    progress.currentStep != null && progress.totalSteps && progress.totalSteps > 0
      ? Math.min(100, (progress.currentStep / progress.totalSteps) * 100)
      : 0
  const batchPct =
    progress.batchIdx != null && progress.batchTotal && progress.batchTotal > 0
      ? Math.min(100, ((progress.batchIdx + (stepPct / 100)) / progress.batchTotal) * 100)
      : 0

  // phase 文字
  let phase = t('generate.progressPreparing')
  if (progress.currentStep != null && progress.totalSteps) {
    phase = t('generate.progressSampling', { step: progress.currentStep, total: progress.totalSteps })
  }
  if (showBatch && progress.batchIdx != null && progress.batchTotal) {
    phase = t('generate.progressBatch', {
      current: progress.batchIdx + 1,
      total: progress.batchTotal,
      detail: progress.currentStep != null && progress.totalSteps
        ? `step ${progress.currentStep} / ${progress.totalSteps}`
        : t('generate.progressPreparingShort'),
    })
  }

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-fg-secondary font-mono">{phase}</span>
        {showBatch && (
          <span className="text-fg-tertiary font-mono text-2xs">
            {t('generate.totalProgress', { pct: batchPct.toFixed(0) })}
          </span>
        )}
      </div>

      {/* 当前图 step 进度 */}
      <div
        style={{
          height: 4,
          borderRadius: 999,
          background: 'var(--bg-sunken)',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${stepPct}%`,
            height: '100%',
            background: 'var(--accent)',
            transition: 'width 100ms linear',
          }}
        />
      </div>

      {/* batch 进度（XY / 多 batch 时） */}
      {showBatch && (
        <div
          style={{
            height: 3,
            borderRadius: 999,
            background: 'var(--bg-sunken)',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              width: `${batchPct}%`,
              height: '100%',
              background: 'var(--fg-tertiary)',
              transition: 'width 100ms linear',
            }}
          />
        </div>
      )}
    </div>
  )
}
