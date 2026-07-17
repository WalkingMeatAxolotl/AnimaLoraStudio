import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type ConfigData } from '../api/client'
import { useToast } from './Toast'

/**
 * YAML 预览(R4,D3):当前表单 config 经后端 /api/schema/preview-yaml 渲染 ——
 * 与保存后落盘文件走同一条 tolerant + 裁剪 + safe_dump 序列化路径,预览与
 * 落盘物理一致(历史上前端 pruneInactiveConfig + configToYaml 双镜像靠注释
 * 纪律「声称一致」,已删除)。300ms debounce 跟手;请求失败保留上次文本。
 * Presets 页包在居中 modal 里,Train 页作为右栏预览抽屉的 tab。
 * 高度由外层控制:className 传 flex 布局类,<pre> 自己滚动;长行不折行、
 * 横向滚动(路径等长值折行后完全没法读)。
 */
export default function ConfigYamlPanel({
  config,
  fileLabel,
  hint,
  className,
}: {
  config: ConfigData
  /** 落盘文件语境,如 `config.yaml` / `my-preset.yaml`。 */
  fileLabel: string
  /** 顶部警示条(如「包含未保存修改」),缺省不显示。 */
  hint?: string
  className?: string
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [yamlText, setYamlText] = useState('')

  useEffect(() => {
    let alive = true
    const timer = setTimeout(() => {
      api.previewConfigYaml(config)
        .then((r) => { if (alive) setYamlText(r.yaml) })
        .catch(() => { /* 网络抖动保留上次文本;下次变更会重试 */ })
    }, 300)
    return () => { alive = false; clearTimeout(timer) }
  }, [config])

  // safe_dump 顶级键顶格 —— 行首非空白即一个落盘字段(多行字符串续行有缩进)
  const fieldCount = yamlText
    ? yamlText.split('\n').filter((l) => /^[^\s#]/.test(l)).length
    : 0

  return (
    <div className={className ?? 'flex flex-col min-h-0'}>
      <div className="flex items-center gap-2 mb-2 shrink-0">
        <span className="font-mono text-xs font-semibold text-fg-secondary truncate">{fileLabel}</span>
        <span className="text-xs text-fg-tertiary shrink-0">
          {t('schema.fieldCount', { n: fieldCount })}
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
      <pre className="flex-1 min-h-0 m-0 p-3 bg-sunken rounded-sm font-mono text-xs text-fg-secondary leading-[1.7] whitespace-pre overflow-auto">
        {yamlText}
      </pre>
    </div>
  )
}
