import { useTranslation } from 'react-i18next'

/** 出图进度：细条 + 相位标签，覆盖**全流程**（不再只有采样 step）。
 *
 * 来源：daemon 推 SSE
 *   - generate_phase          { name: 'load'|'clip'|'sample'|'vae' }  ← 覆盖非采样阶段
 *   - generate_image_started  { batch_idx, batch_total, total_steps }
 *   - generate_preview_step   { step, total, image_b64? }
 *
 * Generate.tsx 聚合成 progress prop。设计成 header 下方**细条 overlay**（absolute、
 * pointer-events-none）：不挤压预览、不增加上下高度；切历史图也照常显示当前进度。
 */

export type GeneratePhase = 'load' | 'clip' | 'sample' | 'vae'

export interface GenerateProgress {
  /** 当前阶段（load/clip/sample/vae）；null = 未知（退回按 step 估算） */
  phase: GeneratePhase | null
  /** 当前在跑哪一张（多张图 / XY 时；单图 batchTotal=1） */
  batchIdx: number | null
  batchTotal: number | null
  /** 当前图采样到第几步 */
  currentStep: number | null
  totalSteps: number | null
}

// 各阶段在「单张图」里占的总进度基点（sample 段按 step 线性铺开 0.20→0.92）
const PHASE_BASE: Record<GeneratePhase, number> = {
  load: 0.03,
  clip: 0.12,
  sample: 0.20,
  vae: 0.95,
}

export default function GenerateProgressBar({
  busy, progress,
}: {
  busy: boolean
  progress: GenerateProgress
}) {
  const { t } = useTranslation()
  if (!busy && progress.currentStep == null && progress.phase == null) return null

  const stepFrac =
    progress.currentStep != null && progress.totalSteps && progress.totalSteps > 0
      ? Math.min(1, progress.currentStep / progress.totalSteps)
      : 0

  // 单张图的进度：sample 段按 step 铺 0.20→0.92，其余阶段用基点；无 phase 时退回 step。
  let frac: number
  if (progress.phase === 'sample') frac = PHASE_BASE.sample + stepFrac * 0.72
  else if (progress.phase) frac = PHASE_BASE[progress.phase]
  else frac = stepFrac > 0 ? PHASE_BASE.sample + stepFrac * 0.72 : 0

  // 多图（batch / XY）：把当前图的 frac 摊进整体
  const bt = progress.batchTotal
  const bi = progress.batchIdx
  const overall = bt && bt > 1 && bi != null ? Math.min(1, (bi + frac) / bt) : frac
  const pct = Math.round(overall * 100)

  // 相位文字
  let phaseLabel: string
  if (progress.phase === 'load') phaseLabel = t('generate.phaseLoad')
  else if (progress.phase === 'clip') phaseLabel = t('generate.phaseClip')
  else if (progress.phase === 'vae') phaseLabel = t('generate.phaseVae')
  else if (progress.phase === 'sample' || stepFrac > 0)
    phaseLabel = t('generate.phaseSample', { step: progress.currentStep ?? 0, total: progress.totalSteps ?? 0 })
  else phaseLabel = t('generate.progressPreparing')

  const batchTag = bt && bt > 1 && bi != null ? `${bi + 1}/${bt} · ` : ''

  return (
    <div className="flex flex-col">
      {/* 细进度条（贴 header 下沿，3px） */}
      <div style={{ height: 3, background: 'var(--bg-sunken)', overflow: 'hidden' }}>
        <div
          style={{
            width: `${pct}%`,
            height: '100%',
            background: 'var(--accent)',
            transition: 'width 150ms linear',
          }}
        />
      </div>
      {/* 相位标签行：小字 + 渐隐底，读得清又不挡图；pointer-events-none 由父层给 */}
      <div
        className="flex items-center justify-between px-2 font-mono text-2xs"
        style={{
          paddingTop: 2, paddingBottom: 2,
          background: 'linear-gradient(to bottom, var(--bg-surface) 55%, transparent)',
        }}
      >
        <span className="text-fg-secondary truncate">{batchTag}{phaseLabel}</span>
        <span className="text-fg-tertiary shrink-0 ml-2">{pct}%</span>
      </div>
    </div>
  )
}
