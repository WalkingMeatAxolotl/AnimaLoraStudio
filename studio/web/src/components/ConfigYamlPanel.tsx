import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import type { ConfigData, SchemaResponse } from '../api/client'
import { pruneInactiveConfig } from '../lib/schema'
import { configToYaml } from '../lib/yamlPreview'
import { useToast } from './Toast'

/**
 * 「生效配置」查看器：把当前表单 config 按 show_when 裁掉未启用字段后,
 * 渲染成与落盘 yaml 一致的文本(键序 / 引号规则对齐 yaml.safe_dump)。
 * Presets 页包在居中 modal 里,Train 页作为右栏预览抽屉的 tab。
 * 高度由外层控制:className 传 flex 布局类,<pre> 自己滚动。
 */
export default function ConfigYamlPanel({
  config,
  schema,
  fileLabel,
  hint,
  className,
}: {
  config: ConfigData
  schema: SchemaResponse | null
  /** 落盘文件语境,如 `config.yaml` / `my-preset.yaml`。 */
  fileLabel: string
  /** 顶部警示条(如「包含未保存修改」),缺省不显示。 */
  hint?: string
  className?: string
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const active = useMemo(
    () => (schema ? pruneInactiveConfig(config, schema.schema.properties) : config),
    [config, schema],
  )
  const yamlText = useMemo(() => configToYaml(active), [active])

  return (
    <div className={className ?? 'flex flex-col min-h-0'}>
      <div className="flex items-center gap-2 mb-2 shrink-0">
        <span className="font-mono text-xs font-semibold text-fg-secondary truncate">{fileLabel}</span>
        <span className="text-xs text-fg-tertiary shrink-0">
          {t('schema.fieldCount', { n: Object.keys(active).length })}
        </span>
        {hint && <span className="text-xs text-warn truncate">{hint}</span>}
        <span className="flex-1" />
        <button
          type="button"
          className="btn btn-ghost btn-sm text-xs shrink-0"
          onClick={() => {
            navigator.clipboard.writeText(yamlText)
              .then(() => toast(t('presets.copied'), 'success'))
              .catch(() => toast(t('presets.copyFailed'), 'error'))
          }}
        >{t('common.copy')}</button>
      </div>
      <pre className="flex-1 min-h-0 m-0 p-3 bg-sunken rounded-sm font-mono text-xs text-fg-secondary leading-[1.7] whitespace-pre-wrap break-words overflow-auto">
        {yamlText}
      </pre>
    </div>
  )
}
