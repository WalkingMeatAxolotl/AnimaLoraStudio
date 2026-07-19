// RuleImpactDialog —— 规则联动改值的确认对话框（刀 2 / R6，D6）。
//
// 用户改动 gate 字段（如开 InfoNoise、切 optimizer 到 prodigy）触发 disable_when
// 规则 takeover 时，若有「有损」写值（目标字段当前值 ≠ 将写入值），SchemaForm
// 在 setField 入口拦截：不让违反态进入表单 state，先弹本对话框列出变更清单
// （from → to + 领域理由），确认才应用（触发改动 + 全部联动写值一次性提交），
// 取消则什么都不发生。无损（联动目标本来就在钉值上）时静默应用不弹窗。
//
// FamilySwitchDialog 是同一交互范式的族切换特例（那边要后端重算路径，这边
// 纯前端元数据求值即可）；UI 骨架保持一致。
import { useTranslation } from 'react-i18next'
import { fieldLabel } from '../lib/schema'

export interface RuleImpactChange {
  field: string
  from: unknown
  to: unknown
  /** 领域理由（字段 disable_hint / 建议文案），显示在变更行下方。 */
  reason?: string
}

interface Props {
  /** 用户触发的原始改动（gate 字段），显示在标题区。 */
  trigger: { field: string; from: unknown; to: unknown }
  changes: RuleImpactChange[]
  onApply: () => void
  onCancel: () => void
}

export default function RuleImpactDialog({ trigger, changes, onApply, onCancel }: Props) {
  const { t } = useTranslation()

  const fmt = (v: unknown): string => {
    if (v === null || v === undefined || v === '') return t('familySwitch.empty')
    if (typeof v === 'boolean') return v ? t('field.yes') : t('field.no')
    return String(v)
  }

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/50"
      onClick={onCancel}
    >
      <div
        className="bg-elevated border border-dim rounded-lg w-[92%] max-w-[560px] max-h-[80vh] p-6 flex flex-col gap-4 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <h3 className="m-0 text-base font-semibold text-fg-primary">
            {t('ruleImpact.title')}
          </h3>
          <p className="m-0 mt-1 text-sm text-fg-secondary">
            {t('ruleImpact.intro', {
              field: fieldLabel(trigger.field),
              value: fmt(trigger.to),
            })}
          </p>
        </div>

        <div className="overflow-y-auto flex flex-col gap-3 pr-1">
          {changes.map((c) => (
            <div key={c.field} className="text-sm">
              <div className="flex items-baseline gap-2">
                <span className="font-medium text-fg-secondary">{fieldLabel(c.field)}</span>
                <span className="text-fg-tertiary">{fmt(c.from)}</span>
                <span className="text-accent">→</span>
                <span className="text-fg-primary">{fmt(c.to)}</span>
              </div>
              {c.reason && (
                <div className="mt-0.5 text-xs text-fg-tertiary">{c.reason}</div>
              )}
            </div>
          ))}
        </div>

        <div className="flex gap-2 justify-end mt-1">
          <button
            type="button"
            onClick={onCancel}
            className="btn btn-secondary min-w-[96px] justify-center"
          >
            {t('common.cancel')}
          </button>
          <button
            type="button"
            onClick={onApply}
            className="btn btn-primary min-w-[96px] justify-center"
          >
            {t('ruleImpact.ok')}
          </button>
        </div>
      </div>
    </div>
  )
}
