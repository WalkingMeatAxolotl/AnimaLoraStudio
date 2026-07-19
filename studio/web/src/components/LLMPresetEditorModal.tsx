// LLMPresetEditorModal —— LLM tagger 预设的全字段编辑器（居中 modal）。
//
// 全局唯一的预设编辑入口：设置页预设列表的「编辑」和打标页的「编辑预设」都开
// 这个 modal，字段只在这里维护一份。数据面走 SettingsDataProvider（instant-apply
// commitSecrets 即时落盘），modal 自身无草稿/保存按钮。
//
// 布局参考打标页字段范式：label 上 / 控件全宽在下 / 两列 grid，分节平铺不折叠。
import type { TFunction } from 'i18next'
import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { Trans, useTranslation } from 'react-i18next'
import { api, type LLMPreset, type SecretsPatch } from '../api/client'
import { MASK } from '../pages/tools/settings/constants'
import { useSettingsData } from '../lib/SettingsData'
import { useDialog } from './Dialog'
import { useToast } from './Toast'
import LLMMessagesEditor from './LLMMessagesEditor'

// 内置预设 label 的 i18n 映射（id → key）；自定义预设直接显示 label。
const LLM_PRESET_LABEL_KEYS: Record<string, string> = {
  style_json: 'llmPreset.presetLabels.styleJson',
  general_json: 'llmPreset.presetLabels.generalJson',
  txt_tags: 'llmPreset.presetLabels.txtTags',
  joycaption: 'llmPreset.presetLabels.joycaption',
  assist_json: 'llmPreset.presetLabels.assistJson',
  assist_text: 'llmPreset.presetLabels.assistText',
}

export function llmPresetLabel(preset: LLMPreset, t: TFunction): string {
  const key = LLM_PRESET_LABEL_KEYS[preset.id]
  return key ? t(key, { defaultValue: preset.label }) : preset.label
}

export default function LLMPresetEditorModal({ presetId, onClose }: {
  presetId: string
  onClose: () => void
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const { secrets, setSecrets, commitSecrets } = useSettingsData()
  // 另存为副本后切到新预设继续编辑
  const [editingId, setEditingId] = useState(presetId)
  const [modelsBusy, setModelsBusy] = useState(false)
  const [testBusy, setTestBusy] = useState(false)

  const presets = secrets?.llm_tagger.presets ?? []
  const preset = presets.find((p) => p.id === editingId) ?? null

  if (!secrets || !preset) return null

  const patchPreset = (patch: Partial<LLMPreset>) => {
    const next = presets.map((p) => (p.id === editingId ? { ...p, ...patch } : p))
    commitSecrets({ llm_tagger: { presets: next } } as SecretsPatch)
  }

  const refreshModels = async () => {
    setModelsBusy(true)
    try {
      const result = await api.refreshLLMModels({
        preset_id: preset.id,
        base_url: preset.base_url,
        api_key: preset.api_key,
        timeout: preset.timeout,
      })
      setSecrets(result.secrets)
      toast(t('settings.modelsLoaded', { n: result.items.length }), 'success')
    } catch (e) {
      toast(t('settings.modelsLoadFailed', { error: String(e) }), 'error')
    } finally {
      setModelsBusy(false)
    }
  }

  const testConnection = async () => {
    setTestBusy(true)
    try {
      const result = await api.testLLMConnection({
        preset_id: preset.id,
        base_url: preset.base_url,
        api_key: preset.api_key,
        model: preset.model,
        endpoint: preset.endpoint,
        timeout: preset.timeout,
        max_tokens: Math.max(512, preset.max_tokens),
        temperature: preset.temperature,
      })
      // 延迟 / HTTP 状态 / 错误预览拼进 toast，让用户不打开日志也能拿到详情。
      const parts: string[] = [result.ok ? t('settings.llmTestOk') : t('settings.llmTestNotOk')]
      if (result.elapsed_ms > 0) parts.push(`${result.elapsed_ms} ms`)
      if (result.status_code !== null) parts.push(`HTTP ${result.status_code}`)
      if (!result.ok) {
        const detail = result.error || result.response_preview
        if (detail) parts.push(detail.slice(0, 120))
      }
      toast(parts.join(' · '), result.ok ? 'success' : 'error')
    } catch (e) {
      toast(t('settings.llmTestFailed', { error: String(e) }), 'error')
    } finally {
      setTestBusy(false)
    }
  }

  // 另存为副本：免弹窗（Dialog 弹层叠 modal 之上体验差）——直接用「原名 - Copy」
  // 命名并切到副本继续编辑，名称字段随时可改。
  const saveAsCopy = () => {
    const label = `${llmPresetLabel(preset, t)} - Copy`
    const slug = label.toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '') || 'preset'
    const used = new Set(presets.map((p) => p.id))
    let idx = 1
    let id = slug
    while (used.has(id)) {
      idx += 1
      id = `${slug}_${idx}`
    }
    const next: LLMPreset = {
      ...preset,
      // deep-copy 可变字段避免共享引用
      messages: preset.messages.map((m) => ({ ...m })),
      model_ids: [...preset.model_ids],
      // api_key 不复制：这里拿到的是 MASK 掩码，后端 deep-merge 对新 id 会把
      // MASK leaf 直接丢弃 → 静默落成空 key。明确置空让用户重填。
      api_key: '',
      id,
      label,
      builtin: false,
    }
    commitSecrets({ llm_tagger: { presets: [...presets, next] } } as SecretsPatch)
    setEditingId(id)
  }

  const deletePreset = async () => {
    if (preset.builtin || presets.length <= 1) return
    if (!(await confirm(t('settings.confirmDeletePreset', { label: preset.label }), { tone: 'danger' }))) return
    const next = presets.filter((p) => p.id !== editingId)
    // 删的是全局默认时把默认转给列表第一个；一次 commit 避免两个 patch 竞态。
    commitSecrets({
      llm_tagger: {
        presets: next,
        ...(secrets.llm_tagger.current_preset === editingId
          ? { current_preset: next[0]?.id ?? 'style_json' }
          : {}),
      },
    } as SecretsPatch)
    onClose()
  }

  const resetToBuiltin = async () => {
    if (!preset.builtin) return
    if (!(await confirm(t('settings.confirmResetPreset', { label: llmPresetLabel(preset, t) }), { tone: 'danger' }))) return
    // 从列表移除，后端 validator 在 PUT 后会用程序默认值补回同 id 的内置预设。
    // 乐观更新窗口内本 preset 短暂不存在，直接关掉 modal 避免闪烁。
    commitSecrets({ llm_tagger: { presets: presets.filter((p) => p.id !== editingId) } } as SecretsPatch)
    onClose()
  }

  const assistNeedsTags =
    !!preset.assist_tagger
    && !preset.messages.some((m) => m.type === 'text' && m.content.includes('{{tags}}'))
  const assistHelp = t('llmPreset.assistTaggerHelp').split('%TAGS%').join('{{tags}}')

  // Portal 到 body：modal 是全局层（打标页/设置抽屉两处入口同款），不挂在抽屉
  // DOM 里；幕布与公告中心同款（bg-black/35，无磨砂）。
  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="llm-preset-editor-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/35 p-4"
      data-testid="llm-preset-editor-modal"
      onClick={onClose}
    >
      <div
        className="w-[80vw] max-h-[88vh] flex flex-col bg-elevated border border-dim rounded-lg shadow-xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* header */}
        <div className="flex items-center gap-2.5 px-6 py-4 border-b border-subtle shrink-0">
          <h2 id="llm-preset-editor-title" className="m-0 text-base font-semibold text-fg-primary">
            {t('llmPreset.title')}
          </h2>
          <span className="text-sm text-fg-tertiary truncate">{llmPresetLabel(preset, t)}</span>
          {preset.builtin && (
            <span className="text-xs px-1.5 py-0.5 rounded-sm font-mono bg-overlay text-fg-tertiary shrink-0">
              {t('llmPreset.builtin')}
            </span>
          )}
          <span className="flex-1" />
          <button
            type="button"
            onClick={onClose}
            aria-label={t('common.close', { defaultValue: 'Close' })}
            className="w-7 h-7 grid place-items-center text-fg-tertiary bg-transparent border-none rounded-sm cursor-pointer hover:bg-overlay hover:text-fg-primary transition-colors"
          >
            ✕
          </button>
        </div>

        {/* body：左（参数）/ 右（提示词消息）= 1:2，两列各自独立滚动 */}
        <div className="flex-1 min-h-0 grid grid-cols-1 md:grid-cols-[1fr_2fr]">
          <div className="min-h-0 overflow-y-auto px-5 py-4 flex flex-col gap-4 border-r border-subtle">
          <EditorSection title={t('llmPreset.sectionBasic')}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
              <MTextField
                label={t('llmPreset.fieldLabel')}
                value={preset.label}
                onCommit={(v) => { if (v.trim()) patchPreset({ label: v.trim() }) }}
              />
            </div>
          </EditorSection>

          <EditorSection title={t('llmPreset.sectionConnection')}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
              <MField
                label={t('settings.fieldBaseUrl')}
                className="md:col-span-2"
                help={<Trans i18nKey="llmPreset.baseUrlHelp" components={{ code: <code /> }} />}
              >
                <MInput
                  type="text"
                  value={preset.base_url}
                  placeholder="https://api.openai.com/v1"
                  onCommit={(v) => patchPreset({ base_url: v.trim() })}
                />
              </MField>
              {/* API Key + Endpoint 同一行；模型独占一行全宽（模型名/下拉常很长） */}
              <MField label="API Key">
                <ApiKeyInput
                  value={preset.api_key}
                  onCommit={(v) => patchPreset({ api_key: v })}
                />
              </MField>
              <MField label={t('llmPreset.fieldEndpoint')}>
                <div className="flex gap-1.5">
                  <select
                    value={preset.endpoint}
                    onChange={(e) => patchPreset({ endpoint: e.target.value as LLMPreset['endpoint'] })}
                    className="input input-mono flex-1 min-w-0"
                    style={mInputStyle}
                  >
                    <option value="chat_completions">Chat Completions</option>
                    <option value="responses">Responses</option>
                  </select>
                  <button
                    type="button"
                    onClick={() => void testConnection()}
                    disabled={testBusy || !preset.base_url.trim() || !preset.model.trim()}
                    className="btn btn-secondary btn-sm shrink-0"
                  >
                    {testBusy ? t('llmPreset.testing') : t('llmPreset.testConnection')}
                  </button>
                </div>
              </MField>
              <MField label={t('llmPreset.fieldModel')} className="md:col-span-2">
                <div className="flex gap-1.5">
                  {preset.model_ids.length > 0 ? (
                    <select
                      value={preset.model}
                      onChange={(e) => patchPreset({ model: e.target.value })}
                      className="input input-mono flex-1 min-w-0"
                      style={mInputStyle}
                    >
                      {!preset.model_ids.includes(preset.model) && preset.model && (
                        <option value={preset.model}>{preset.model}</option>
                      )}
                      {preset.model_ids.map((m) => <option key={m} value={m}>{m}</option>)}
                    </select>
                  ) : (
                    <MInput
                      type="text"
                      value={preset.model}
                      placeholder={t('llmPreset.modelPlaceholder')}
                      onCommit={(v) => patchPreset({ model: v.trim() })}
                      className="flex-1 min-w-0"
                    />
                  )}
                  <button
                    type="button"
                    onClick={() => void refreshModels()}
                    disabled={modelsBusy || !preset.base_url.trim()}
                    className="btn btn-secondary btn-sm shrink-0"
                  >
                    {modelsBusy ? t('llmPreset.fetchingModels') : t('llmPreset.fetchModels')}
                  </button>
                </div>
              </MField>
            </div>
          </EditorSection>

          <EditorSection title={t('llmPreset.sectionOutput')}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
              <MField label={t('llmPreset.fieldOutputFormat')}>
                <select
                  value={preset.output_format}
                  onChange={(e) => patchPreset({ output_format: e.target.value as LLMPreset['output_format'] })}
                  className="input input-mono w-full"
                  style={mInputStyle}
                >
                  <option value="json">{t('llmPreset.jsonCaption')}</option>
                  <option value="text">{t('llmPreset.textCaption')}</option>
                </select>
              </MField>
              <MField
                label={t('llmPreset.assistTagger')}
                helpTooltipText={assistHelp}
                help={assistNeedsTags && (
                  <span className="text-warn">
                    {t('llmPreset.assistNeedsTags').split('%TAGS%').join('{{tags}}')}
                  </span>
                )}
              >
                <select
                  value={preset.assist_tagger}
                  onChange={(e) => patchPreset({ assist_tagger: e.target.value })}
                  className="input input-mono w-full"
                  style={mInputStyle}
                >
                  <option value="">{t('llmPreset.assistOff')}</option>
                  <option value="wd14">WD14</option>
                  <option value="cltagger">CLTagger</option>
                </select>
              </MField>
            </div>
          </EditorSection>

          <EditorSection title={t('llmPreset.sectionSampling')}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
              <MNumField label={t('llmPreset.fieldTemperature')} value={preset.temperature} min={0} max={2} step={0.05} onCommit={(v) => patchPreset({ temperature: v })} />
              <MNumField label={t('llmPreset.fieldMaxTokens')} value={preset.max_tokens} min={64} max={4096} onCommit={(v) => patchPreset({ max_tokens: Math.round(v) })} />
              <MNumField label={t('llmPreset.fieldTimeout')} value={preset.timeout} min={5} max={600} onCommit={(v) => patchPreset({ timeout: Math.round(v) })} />
              <MNumField label={t('llmPreset.fieldMaxRetries')} value={preset.max_retries} min={1} max={10} onCommit={(v) => patchPreset({ max_retries: Math.round(v) })} />
              <MNumField label={t('llmPreset.fieldConcurrency')} value={preset.concurrency} min={1} max={8} onCommit={(v) => patchPreset({ concurrency: Math.round(v) })} />
              <MNumField label={t('llmPreset.fieldRequestsPerSecond')} value={preset.requests_per_second} min={0} max={60} step={0.1} onCommit={(v) => patchPreset({ requests_per_second: v })} />
              <MNumField label={t('llmPreset.fieldMaxRequestsPerMinute')} value={preset.max_requests_per_minute} min={0} max={3600} onCommit={(v) => patchPreset({ max_requests_per_minute: Math.round(v) })} />
            </div>
          </EditorSection>

          <EditorSection title={t('llmPreset.sectionImage')}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
              <MNumField label={t('llmPreset.fieldMaxSide')} value={preset.max_side} min={512} max={4096} step={64} onCommit={(v) => patchPreset({ max_side: Math.round(v) })} />
              <MNumField label={t('llmPreset.fieldJpegQuality')} value={preset.jpeg_quality} min={1} max={100} onCommit={(v) => patchPreset({ jpeg_quality: Math.round(v) })} />
              <MNumField
                label={t('llmPreset.fieldMaxImageMb')} value={preset.max_image_mb} min={0.1} max={25} step={0.1}
                onCommit={(v) => patchPreset({ max_image_mb: v })}
                help={<Trans i18nKey="llmPreset.imageSizeHint" components={{ limit: <b className="text-fg-secondary" /> }} />}
              />
            </div>
          </EditorSection>

        </div>

          {/* 右列：提示词消息（自适应高度的消息编辑器，长了随本列滚动） */}
          <div className="min-h-0 overflow-y-auto px-5 py-4">
            <EditorSection title={t('llmPreset.sectionMessages')}>
              <div className="flex flex-col gap-1.5">
                {preset.endpoint === 'responses' && (
                  <div className="text-xs text-warn">{t('llmPreset.responsesWarning')}</div>
                )}
                <LLMMessagesEditor
                  messages={preset.messages}
                  onChange={(msgs) => patchPreset({ messages: msgs })}
                />
              </div>
            </EditorSection>
          </div>
        </div>

        {/* footer */}
        <div className="flex items-center gap-2 px-6 py-3.5 border-t border-subtle shrink-0">
          {preset.builtin && (
            <button type="button" onClick={() => void resetToBuiltin()} className="btn btn-ghost btn-sm text-err">
              {t('llmPreset.resetBuiltin')}
            </button>
          )}
          <button type="button" onClick={saveAsCopy} className="btn btn-ghost btn-sm">
            {t('llmPreset.saveAsCopy')}
          </button>
          <span className="flex-1" />
          {!preset.builtin && presets.length > 1 && (
            <button type="button" onClick={() => void deletePreset()} className="btn btn-ghost btn-sm text-err">
              {t('common.delete')}
            </button>
          )}
          <button type="button" onClick={onClose} className="btn btn-primary">
            {t('llmPreset.done')}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  )
}

// ── 字段原语（打标页范式：label 上 / 控件全宽在下 / help 在控件下方） ─────────

const mInputStyle: React.CSSProperties = {
  width: '100%', padding: '5px 10px',
  background: 'var(--bg-canvas)', border: '1px solid var(--border-default)',
  borderRadius: 'var(--r-sm)', fontSize: 'var(--t-sm)',
  color: 'var(--fg-primary)',
}

function EditorSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5">
      <div className="flex items-center gap-2 mb-1">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
        <span className="caption">{title}</span>
      </div>
      {children}
    </section>
  )
}

function MField({ label, helpTooltipText, help, className = '', children }: {
  label: string
  helpTooltipText?: string
  help?: React.ReactNode
  className?: string
  children: React.ReactNode
}) {
  return (
    <div className={`py-1.5 ${className}`}>
      <div className="flex items-center gap-2 text-sm font-medium text-fg-secondary mb-1" title={helpTooltipText}>
        <span>{label}</span>
      </div>
      {children}
      {help && <div className="text-xs text-fg-tertiary mt-1">{help}</div>}
    </div>
  )
}

// 文本输入：本地缓冲，失焦 / Enter 才提交（instant-apply 下避免逐字 PUT）。
function MInput({ value, onCommit, className = '', ...rest }: Omit<React.InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange' | 'className'> & {
  value: string
  onCommit: (v: string) => void
  className?: string
}) {
  const [local, setLocal] = useState(value)
  useEffect(() => { setLocal(value) }, [value])
  return (
    <input
      {...rest}
      value={local}
      onChange={(e) => setLocal(e.target.value)}
      onBlur={() => { if (local !== value) onCommit(local) }}
      onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
      className={`input input-mono ${className}`}
      style={mInputStyle}
    />
  )
}

function MTextField({ label, value, onCommit, className }: {
  label: string; value: string; onCommit: (v: string) => void; className?: string
}) {
  return (
    <MField label={label} className={className}>
      <MInput type="text" value={value} onCommit={onCommit} />
    </MField>
  )
}

function MNumField({ label, value, min, max, step = 1, onCommit, help }: {
  label: string; value: number; min: number; max: number; step?: number
  onCommit: (v: number) => void; help?: React.ReactNode
}) {
  const [local, setLocal] = useState(String(value))
  useEffect(() => { setLocal(String(value)) }, [value])
  const commit = () => {
    const n = Number(local)
    if (Number.isNaN(n)) { setLocal(String(value)); return }
    const clamped = Math.max(min, Math.min(max, n))
    if (clamped !== value) onCommit(clamped)
    else setLocal(String(value))
  }
  return (
    <MField label={label} help={help}>
      <input
        type="number" min={min} max={max} step={step}
        value={local}
        onChange={(e) => setLocal(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
        className="input input-mono"
        style={mInputStyle}
      />
    </MField>
  )
}

// API key：password 输入 + MASK 哨兵语义（已保存值显示占位符；清空 = 保持不变）。
function ApiKeyInput({ value, onCommit }: {
  value: string
  onCommit: (v: string) => void
}) {
  const { t } = useTranslation()
  const [local, setLocal] = useState(value)
  useEffect(() => { setLocal(value) }, [value])
  const masked = local === MASK
  return (
    <input
      type="password"
      value={masked ? '' : local}
      placeholder={value === MASK ? t('llmPreset.apiKeySavedPlaceholder') : ''}
      onChange={(e) => setLocal(e.target.value || MASK)}
      onBlur={() => { if (local !== value) onCommit(local) }}
      onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
      autoComplete="new-password"
      data-lpignore="true"
      data-1p-ignore
      data-form-type="other"
      className="input input-mono"
      style={mInputStyle}
    />
  )
}
