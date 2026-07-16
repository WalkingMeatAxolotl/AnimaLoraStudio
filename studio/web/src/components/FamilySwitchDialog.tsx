// FamilySwitchDialog —— 模型族切换的确认对话框（多模型 P4-3）。
//
// 翻 model_family 不是普通字段编辑：路径按目标族重算、族风味字段重置、
// 目标族不支持的能力字段关闭。本组件在打开时调后端预览计算
// （/api/models/family-switch，纯计算不落盘），把变更分「模型路径 /
// 参数调整」两区结构化展示，确认才把切换后的完整 config 交回调用方
// （走各页正常保存链路）。
//
// 不用通用 Dialog.confirm 的文本槽：变更清单是结构化数据（长路径 +
// 新旧对照），塞纯文本里换行混乱不可读 —— 按 Dialog.tsx 自己的约定，
// 复杂内容走声明式 JSX modal（NewVersionDialog 同款范式）。
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  type ConfigData,
  type FamilySwitchChange,
} from '../api/client'
import { fieldLabel, schemaEnumLabel } from '../lib/schema'

interface Props {
  /** 目标族 id（用户在下拉里选的新值）。 */
  target: string
  /** 当前 config（切换前，model_family 仍是旧值）。 */
  config: ConfigData
  /** 用户确认：应用后端重算的完整 config。 */
  onApply: (switched: ConfigData) => void
  /** 用户取消 / 预览失败：调用方保持旧值不动。 */
  onCancel: () => void
}

/** 4 个权重路径字段 —— 展示用等宽字体 + 上下对照布局。 */
const PATH_FIELDS = new Set([
  'transformer_path', 'vae_path', 'text_encoder_path', 't5_tokenizer_path',
])

function useSwitchPreview(target: string, config: ConfigData) {
  const [preview, setPreview] = useState<{
    config: ConfigData
    changes: FamilySwitchChange[]
  } | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let alive = true
    api.switchModelFamily(target, config)
      .then((r) => { if (alive) setPreview(r) })
      .catch((e) => { if (alive) setError(String(e)) })
    return () => { alive = false }
    // config 引用在对话框生命周期内不变（打开时快照）
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target])
  return { preview, error }
}

export default function FamilySwitchDialog({ target, config, onApply, onCancel }: Props) {
  const { t } = useTranslation()
  const { preview, error } = useSwitchPreview(target, config)

  const fmt = (v: unknown): string => {
    if (v === null || v === undefined || v === '') return t('familySwitch.empty')
    if (typeof v === 'boolean') return v ? t('field.yes') : t('field.no')
    return String(v)
  }

  const changes = (preview?.changes ?? []).filter((c) => c.field !== 'model_family')
  const pathChanges = changes.filter((c) => PATH_FIELDS.has(c.field))
  const paramChanges = changes.filter((c) => !PATH_FIELDS.has(c.field))
  const fromLabel = schemaEnumLabel('model_family', String(config.model_family ?? 'anima'), t)
  const toLabel = schemaEnumLabel('model_family', target, t)

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/50"
      onClick={onCancel}
    >
      <div
        className="bg-elevated border border-dim rounded-lg w-[92%] max-w-[640px] max-h-[85vh] p-6 flex flex-col gap-4 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <h3 className="m-0 text-base font-semibold text-fg-primary">
            {t('familySwitch.title')}
          </h3>
          <p className="m-0 mt-1 text-sm text-fg-secondary">
            {t('familySwitch.intro', { from: fromLabel, to: toLabel })}
          </p>
        </div>

        {error ? (
          <p className="m-0 text-sm text-err">{t('familySwitch.failed', { error })}</p>
        ) : !preview ? (
          <p className="m-0 text-sm text-fg-tertiary">{t('familySwitch.loading')}</p>
        ) : changes.length === 0 ? (
          <p className="m-0 text-sm text-fg-secondary">{t('familySwitch.noChanges')}</p>
        ) : (
          <div className="overflow-y-auto flex flex-col gap-4 pr-1">
            {pathChanges.length > 0 && (
              <section>
                <div className="text-xs font-semibold text-fg-tertiary uppercase tracking-wide mb-2">
                  {t('familySwitch.pathsSection')}
                </div>
                <div className="flex flex-col gap-2.5">
                  {pathChanges.map((c) => (
                    <div key={c.field} className="text-sm">
                      <div className="font-medium text-fg-secondary mb-0.5">
                        {fieldLabel(c.field)}
                      </div>
                      <div className="font-mono text-xs break-all text-fg-tertiary">
                        {fmt(c.from)}
                      </div>
                      <div className="font-mono text-xs break-all text-fg-primary">
                        <span className="text-accent mr-1">→</span>
                        {fmt(c.to)}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}
            {paramChanges.length > 0 && (
              <section>
                <div className="text-xs font-semibold text-fg-tertiary uppercase tracking-wide mb-2">
                  {t('familySwitch.paramsSection')}
                </div>
                <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
                  {paramChanges.map((c) => (
                    <div key={c.field} className="contents">
                      <div className="font-medium text-fg-secondary">
                        {fieldLabel(c.field)}
                      </div>
                      <div className="text-fg-primary">
                        <span className="text-fg-tertiary">{fmt(c.from)}</span>
                        <span className="text-accent mx-1.5">→</span>
                        {fmt(c.to)}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>
        )}

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
            disabled={!preview || !!error}
            onClick={() => preview && onApply(preview.config)}
            className="btn btn-primary min-w-[96px] justify-center"
          >
            {t('familySwitch.ok')}
          </button>
        </div>
      </div>
    </div>
  )
}
