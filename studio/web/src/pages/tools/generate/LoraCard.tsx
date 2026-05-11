import { useEffect, useState } from 'react'
import { api, type LoraCkpt, type LoraEntry } from '../../../api/client'
import { projectAbbr } from './InlineLoraPicker'
import { ckptStemFromPath } from './xy'

/** 已添加的 LoRA 卡片（对齐 Test 重设计.html 的 .ver-row 风格）：
 *
 *   [44px thumb]  项目 / 版本   [训练中 pill]
 *                 [step 2476 · 最新][step 2000][step 1500]   ← ckpt pills
 *                 ─────────────  权重 slider  [×0.85]    [×]
 *
 * - thumb 用 ASCII 缩写（大写）+ 斜纹背景占位
 * - 训练中 stage 时整张卡片描边走 accent 色
 * - 有 version_id 时 fetch ckpt 列表；点 pill 切 path（同卡片内不同 step）
 */
export default function LoraCard({
  lora, label, stage, onChange, onRemove,
}: {
  lora: LoraEntry
  label: string
  stage?: string
  /** 替换整条 entry（path 切 ckpt 时也走它） */
  onChange: (next: LoraEntry) => void
  onRemove: () => void
}) {
  const filename = ckptStemFromPath(lora.path)
  const live = stage === 'training'

  const [ckpts, setCkpts] = useState<LoraCkpt[]>([])
  useEffect(() => {
    if (!lora.project_id || !lora.version_id) {
      setCkpts([])
      return
    }
    let cancelled = false
    void api.listVersionLoraCkpts(lora.project_id, lora.version_id)
      .then((items) => { if (!cancelled) setCkpts(items) })
      .catch(() => { if (!cancelled) setCkpts([]) })
    return () => { cancelled = true }
  }, [lora.project_id, lora.version_id])

  return (
    <div
      className="rounded-md flex items-stretch gap-2.5"
      style={{
        padding: 10,
        border: live ? '1px solid var(--accent)' : '1px solid var(--border-subtle)',
        background: live
          ? 'var(--accent-soft)'
          : 'linear-gradient(180deg, var(--bg-elevated), var(--bg-surface))',
      }}
    >
      <div
        className="shrink-0 grid place-items-center font-mono text-fg-tertiary"
        style={{
          width: 44, height: 44, borderRadius: 6,
          background: 'var(--bg-sunken)',
          backgroundImage:
            'repeating-linear-gradient(135deg, transparent 0 6px, rgba(255,255,255,0.04) 6px 7px)',
          border: '1px solid var(--border-subtle)',
          fontSize: 11,
        }}
      >
        {projectAbbr(label || filename)}
      </div>

      <div className="flex-1 min-w-0 flex flex-col gap-1.5">
        <div className="flex items-center gap-2 text-sm">
          <span className="font-medium truncate">{label || filename}</span>
          {live && (
            <span
              className="font-mono shrink-0"
              style={{
                fontSize: 10, padding: '1px 7px', borderRadius: 999,
                background: 'var(--accent-soft)', color: 'var(--accent)',
                letterSpacing: '0.03em',
              }}
            >
              训练中
            </span>
          )}
        </div>

        {!label && (
          <div className="text-2xs text-fg-tertiary truncate font-mono" title={lora.path}>
            {lora.path}
          </div>
        )}

        {/* Ckpt step pills（同 version 多个 ckpt 时显示，点击切 path） */}
        {ckpts.length > 1 && (
          <div className="flex flex-wrap gap-1" style={{ marginTop: 2 }}>
            {ckpts.map((c) => {
              const active = c.path === lora.path
              return (
                <button
                  key={c.path}
                  onClick={() => onChange({ ...lora, path: c.path })}
                  className="font-mono"
                  style={{
                    fontSize: 10,
                    padding: '2px 8px',
                    borderRadius: 999,
                    border: active ? '1px solid transparent' : '1px dashed var(--border-default)',
                    background: active ? 'var(--accent-soft)' : 'transparent',
                    color: active ? 'var(--accent)' : 'var(--fg-secondary)',
                    cursor: 'pointer',
                  }}
                  title={c.path}
                >
                  {c.label}
                </button>
              )
            })}
          </div>
        )}

        <div className="flex items-center gap-3 mt-0.5">
          <span
            className="font-mono text-fg-tertiary shrink-0"
            style={{ fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase' }}
          >
            权重
          </span>
          <input
            type="range"
            min={0}
            max={1.5}
            step={0.05}
            value={lora.scale}
            onChange={(e) => onChange({ ...lora, scale: Number(e.target.value) })}
            className="flex-1"
            aria-label="权重滑杆"
            style={{ accentColor: 'var(--accent)' }}
          />
          <input
            type="number"
            min={0}
            max={1.5}
            step={0.05}
            value={lora.scale}
            onChange={(e) => onChange({ ...lora, scale: Number(e.target.value) })}
            className="input font-mono text-center"
            style={{ width: 54, padding: '3px 6px', fontSize: 12 }}
            aria-label="权重数值"
          />
        </div>
      </div>

      <button
        onClick={onRemove}
        className="btn btn-ghost text-fg-tertiary hover:text-err shrink-0"
        style={{ width: 28, height: 28, padding: 0, alignSelf: 'flex-start' }}
        title="移除"
        aria-label="移除 LoRA"
      >
        ×
      </button>
    </div>
  )
}
