// FamilySwitchDialog —— 模型族切换的确认对话框（多模型 P4-3）。
//
// 翻 model_family 不是普通字段编辑：路径按目标族重算、族风味字段重置、
// 目标族不支持的能力字段关闭。本组件在打开时调后端预览计算
// （/api/models/family-switch，纯计算不落盘），把变更分「模型路径 /
// 参数调整」两区结构化展示，确认才把切换后的完整 config 交回调用方
// （走各页正常保存链路）。
//
// 使用统一 Modal Pattern 承载结构化内容；命令式 Dialog.confirm 仍只用于纯文本确认。
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  type ConfigData,
  type FamilySwitchChange,
} from '../api/client'
import { fieldLabel, schemaEnumLabel } from '../lib/schema'
import ActionGroup from './ActionGroup'
import Button from './Button'
import Modal from './Modal'

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
    <Modal
      title={t('familySwitch.title')}
      description={t('familySwitch.intro', { from: fromLabel, to: toLabel })}
      onClose={onCancel}
      size="lg"
      bodyClassName="flex flex-col gap-section"
      footer={(
        <ActionGroup
          secondary={(
            <Button
              type="button"
              variant="secondary"
              onClick={onCancel}
              className="min-w-[96px] justify-center"
            >
              {t('common.cancel')}
            </Button>
          )}
          primary={(
            <Button
              type="button"
              variant="primary"
              loading={!preview && !error}
              disabled={!!error}
              onClick={() => preview && onApply(preview.config)}
              className="min-w-[96px] justify-center"
            >
              {t('familySwitch.ok')}
            </Button>
          )}
        />
      )}
    >
      {error ? (
        <p className="m-0 text-sm text-err">{t('familySwitch.failed', { error })}</p>
      ) : !preview ? (
        <p className="m-0 text-sm text-fg-tertiary">{t('familySwitch.loading')}</p>
      ) : changes.length === 0 ? (
        <p className="m-0 text-sm text-fg-secondary">{t('familySwitch.noChanges')}</p>
      ) : (
        <div className="flex flex-col gap-section pr-1">
          {pathChanges.length > 0 && (
            <section>
              <div className="type-section-label mb-related">
                {t('familySwitch.pathsSection')}
              </div>
              <div className="flex flex-col gap-field">
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
              <div className="type-section-label mb-related">
                {t('familySwitch.paramsSection')}
              </div>
              <div className="grid grid-cols-[auto_1fr] gap-x-section gap-y-related text-sm">
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
    </Modal>
  )
}
