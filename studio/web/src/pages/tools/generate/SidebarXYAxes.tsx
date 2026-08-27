import { useTranslation } from 'react-i18next'
import type { LoraEntry } from '../../../api/client'
import { axisLabel, axisView, cellCount, formatAxisValue, type XYAxisDraft } from './xy'

function AxisSummary({
  label,
  draft,
  onEdit,
  onRemove,
}: {
  label: 'X' | 'Y'
  draft: XYAxisDraft
  onEdit: () => void
  onRemove?: () => void
}) {
  const { t } = useTranslation()
  const view = axisView(draft)
  const preview = view.values.slice(0, 5)

  return (
    <div className="rounded-lg border border-subtle bg-overlay p-3">
      <div className="flex items-center gap-2">
        <button
          type="button"
          className="flex-1 min-w-0 text-left"
          onClick={onEdit}
          aria-label={t('generate.editAxis', { label })}
        >
          <span
            className="inline-flex items-center justify-center rounded-sm font-semibold mr-2"
            style={{
              width: 22,
              height: 22,
              background: label === 'X' ? 'var(--accent-soft)' : 'var(--ok-soft)',
              color: label === 'X' ? 'var(--accent)' : 'var(--ok)',
            }}
          >
            {label}
          </span>
          <span className="text-sm font-medium text-fg-primary">{axisLabel(draft.axis)}</span>
        </button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={onEdit}>
          {t('common.edit')}
        </button>
        {onRemove && (
          <button
            type="button"
            className="btn btn-ghost btn-sm text-err"
            onClick={onRemove}
            title={t('generate.xyRemoveAxisTitle', { label })}
            aria-label={t('generate.xyRemoveAxisAria', { label })}
          >
            ×
          </button>
        )}
      </div>

      <button
        type="button"
        className="w-full text-left mt-2"
        onClick={onEdit}
        title={t('generate.editAxis', { label })}
      >
        {preview.length > 0 ? (
          <div className="flex flex-wrap gap-1">
            {preview.map((value, index) => (
              <span
                key={`${value}-${index}`}
                className="font-mono text-2xs px-1.5 py-0.5 rounded border border-subtle bg-surface"
                title={value}
              >
                {formatAxisValue(draft.axis, value)}
              </span>
            ))}
            {view.values.length > preview.length && (
              <span className="text-2xs text-fg-tertiary px-1 py-0.5">
                +{view.values.length - preview.length}
              </span>
            )}
          </div>
        ) : (
          <span className="text-xs text-fg-tertiary">{t('generate.axisNoValues')}</span>
        )}
      </button>
    </div>
  )
}

export default function SidebarXYAxes({
  xDraft,
  yDraft,
  fixedLoras,
  fp8BaseModel,
  onEditX,
  onEditY,
  onSwap,
  onAddY,
  onRemoveY,
}: {
  xDraft: XYAxisDraft
  yDraft: XYAxisDraft | null
  fixedLoras: LoraEntry[]
  fp8BaseModel: boolean
  onEditX: () => void
  onEditY: () => void
  onSwap: () => void
  onAddY: () => void
  onRemoveY: () => void
}) {
  const { t } = useTranslation()
  const xCount = axisView(xDraft).values.length
  const yCount = yDraft ? axisView(yDraft).values.length : null
  const total = cellCount(xCount, yCount)
  const fp8MergeHeavy = fp8BaseModel
    && Boolean(yDraft)
    && [xDraft.axis, yDraft?.axis].includes('lora_ckpt')
    && [xDraft.axis, yDraft?.axis].includes('lora_scale')

  return (
    <div className="flex flex-col gap-3" data-testid="xy-axes-panel">
      <div className="flex items-center justify-between">
        <h3 className="m-0 text-md font-semibold">{t('generate.xyAxes')}</h3>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          disabled={!yDraft}
          onClick={onSwap}
          title={t('generate.swapAxes')}
        >
          ⇅ {t('generate.swapAxes')}
        </button>
      </div>

      <AxisSummary label="X" draft={xDraft} onEdit={onEditX} />
      {yDraft ? (
        <AxisSummary label="Y" draft={yDraft} onEdit={onEditY} onRemove={onRemoveY} />
      ) : (
        <button type="button" className="btn btn-secondary w-full" onClick={onAddY}>
          + {t('generate.addYAxis')}
        </button>
      )}

      <div className="rounded-lg border border-subtle bg-sunken p-3">
        <div className="flex items-baseline justify-between">
          <span className="text-xs text-fg-secondary">
            X {xCount}{yDraft ? ` × Y ${yCount ?? 0}` : ''}
          </span>
          <span className="font-mono text-sm font-semibold text-fg-primary">
            {total} {t('generate.xyImages')}
          </span>
        </div>
        {total > 50 && (
          <div className="text-2xs text-warn mt-2">{t('generate.xyLargeMatrixWarning')}</div>
        )}
        {fp8MergeHeavy && (
          <div className="text-2xs text-warn mt-2">{t('generate.xyFp8MergeWarning')}</div>
        )}
      </div>

      <div className="text-xs text-fg-tertiary">
        {fixedLoras.length > 0
          ? t('generate.xyFixedLoraCount', { count: fixedLoras.length })
          : t('generate.xyFixedLorasEmpty')}
      </div>
    </div>
  )
}
