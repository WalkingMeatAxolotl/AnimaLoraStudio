// LLMPresetEditorModal —— LLM tagger 预设的全字段编辑器（居中 modal）。
//
// 全局唯一的预设编辑入口：设置页预设列表的「编辑」和打标页的「编辑预设」都开
// 这个 modal，字段只在这里维护一份。preset/credential 各走资源 API，
// SettingsDataProvider 只承载保存状态；字段即时提交，没有整页草稿/保存按钮。
//
// 参数与提示词保留专业双栏；Modal 拥有键盘生命周期，危险确认复用同一个壳层。
import type { TFunction } from 'i18next'
import { useEffect, useRef, useState } from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { api, type LLMPreset } from '../api/client'
import { MASK } from '../pages/tools/settings/constants'
import { useSettingsData } from '../lib/SettingsData'
import { useToast } from './Toast'
import ActionGroup from './ActionGroup'
import Alert from './Alert'
import Badge from './Badge'
import Button from './Button'
import Modal from './Modal'
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

interface EditorConfirmation {
  action: 'delete' | 'reset'
  presetId: string
  label: string
}

export default function LLMPresetEditorModal({ presetId, onClose }: {
  presetId: string
  onClose: () => void
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { secrets, reloadSecrets, runSave } = useSettingsData()
  // 另存为副本后切到新预设继续编辑
  const [editingId, setEditingId] = useState(presetId)
  const [modelsBusy, setModelsBusy] = useState(false)
  const [testBusy, setTestBusy] = useState(false)
  const [confirmation, setConfirmation] = useState<EditorConfirmation | null>(null)
  const [confirmationBusy, setConfirmationBusy] = useState(false)
  const [confirmationError, setConfirmationError] = useState('')
  const confirmationBusyRef = useRef(false)
  const editorContentRef = useRef<HTMLDivElement>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)
  const cancelButtonRef = useRef<HTMLButtonElement>(null)
  const returnActionRef = useRef<HTMLButtonElement | null>(null)
  const scrollSnapshotRef = useRef<{ element: HTMLElement; top: number; left: number }[]>([])
  const mutationQueueRef = useRef<Promise<unknown>>(Promise.resolve())
  const etagRef = useRef('')

  const presets = secrets?.llm_tagger.presets ?? []
  const preset = presets.find((p) => p.id === editingId) ?? null
  const presetRef = useRef<LLMPreset | null>(preset)
  useEffect(() => {
    presetRef.current = preset
    etagRef.current = preset?.etag ?? ''
  }, [preset])

  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const surface = opener?.closest<HTMLElement>('[role="dialog"], [role="main"], main')
    return () => {
      // Deleting a preset can remove its Settings row before Modal restores the
      // opener. Return to that still-active task surface, not the inert page.
      if (opener && !opener.isConnected && surface?.isConnected &&
        !surface.closest('[hidden], [aria-hidden="true"], [inert]')) {
        surface.focus({ preventScroll: true })
      }
    }
  }, [])

  // These are view transitions inside one Modal, not another focus trap.
  useEffect(() => {
    if (!confirmation && !returnActionRef.current) return
    let restoreFrame: number | undefined
    const frame = requestAnimationFrame(() => {
      if (confirmation) {
        if (confirmationBusy) {
          editorContentRef.current?.closest<HTMLElement>('[role="alertdialog"]')?.focus()
        } else {
          cancelButtonRef.current?.focus()
        }
      } else {
        // Auto-growing textareas remeasure in ResizeObserver after the reveal
        // frame, temporarily shrinking the scroll range. Restore only after
        // that measurement so it cannot clamp the retained offset back to zero.
        restoreFrame = requestAnimationFrame(() => {
          for (const { element, top, left } of scrollSnapshotRef.current) {
            element.scrollTop = top
            element.scrollLeft = left
          }
          const target = returnActionRef.current?.isConnected ? returnActionRef.current : closeButtonRef.current
          target?.focus({ preventScroll: true })
          returnActionRef.current = null
          scrollSnapshotRef.current = []
        })
      }
    })
    return () => {
      cancelAnimationFrame(frame)
      if (restoreFrame !== undefined) cancelAnimationFrame(restoreFrame)
    }
  }, [confirmation, confirmationBusy])

  // A queued Save as copy may change the edited object. Never reuse a confirmation
  // for a different preset, even when an earlier resource request finishes later.
  useEffect(() => {
    if (confirmation && confirmation.presetId !== editingId && !confirmationBusyRef.current) {
      setConfirmation(null)
      setConfirmationError('')
    }
  }, [confirmation, editingId])

  if (!secrets || !preset) return null

  const enqueueMutation = (operation: () => Promise<void>, onError?: (error: unknown) => void) => {
    mutationQueueRef.current = mutationQueueRef.current
      .then(() => runSave(operation))
      .catch((error) => {
        if (onError) onError(error)
        else toast(String(error), 'error')
      })
    return mutationQueueRef.current
  }

  const patchPreset = (patch: Partial<LLMPreset>) => {
    enqueueMutation(async () => {
      const current = presetRef.current
      if (!current?.etag) throw new Error('Preset ETag is missing; reload settings')
      const updated = await api.patchLLMPreset(current.id, patch, etagRef.current || current.etag)
      etagRef.current = updated.etag ?? ''
      await reloadSecrets()
    })
  }

  const patchApiKey = (value: string) => {
    if (value === MASK) return
    enqueueMutation(async () => {
      const current = presetRef.current
      if (!current?.etag) throw new Error('Preset ETag is missing; reload settings')
      if (current.credential_ref) {
        const metadata = (await api.listCredentials()).find((item) => item.id === current.credential_ref)
        if (!metadata) throw new Error(`Credential not found: ${current.credential_ref}`)
        await api.replaceCredentialSecret(metadata.id, value, metadata.etag)
      } else if (value) {
        const credential = await api.createCredential({
          label: `LLM · ${current.label}`,
          secret: value,
        })
        const updated = await api.patchLLMPreset(
          current.id,
          { credential_ref: credential.id },
          etagRef.current || current.etag,
        )
        etagRef.current = updated.etag ?? ''
      }
      await reloadSecrets()
    })
  }

  const refreshModels = async () => {
    setModelsBusy(true)
    try {
      await mutationQueueRef.current
      const current = presetRef.current
      if (!current) return
      const result = await api.refreshLLMModels(current.id, current.timeout)
      await reloadSecrets()
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
      await mutationQueueRef.current
      const current = presetRef.current
      if (!current) return
      const result = await api.testLLMConnection(current.id, current.timeout)
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

  const saveAsCopy = () => {
    const label = `${llmPresetLabel(preset, t)} - Copy`
    enqueueMutation(async () => {
      const current = presetRef.current
      if (!current) return
      const copied = await api.duplicateLLMPreset(current.id, label)
      await reloadSecrets()
      setEditingId(copied.id)
      toast(t('llmPreset.savedAsCopy', { label }), 'success')
    })
  }

  const exportPreset = () => {
    const a = document.createElement('a')
    a.href = api.llmPresetExportUrl(editingId)
    a.download = `llm-preset-${editingId}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const blurEditorField = () => {
    const active = document.activeElement
    if (active instanceof HTMLElement && editorContentRef.current?.contains(active)) active.blur()
  }

  const beginConfirmation = (action: EditorConfirmation['action'], initiator: HTMLButtonElement) => {
    if (confirmationBusyRef.current || !preset.etag) return
    if (action === 'delete' && (preset.builtin || presets.length <= 1)) return
    if (action === 'reset' && !preset.builtin) return
    blurEditorField()
    returnActionRef.current = initiator
    scrollSnapshotRef.current = Array.from(
      editorContentRef.current?.querySelectorAll<HTMLElement>('[data-llm-editor-scroll]') ?? [],
    ).map((element) => ({ element, top: element.scrollTop, left: element.scrollLeft }))
    setConfirmationError('')
    setConfirmation({ action, presetId: preset.id, label: llmPresetLabel(preset, t) })
  }

  const cancelConfirmation = () => {
    if (confirmationBusyRef.current) return
    setConfirmation(null)
    setConfirmationError('')
  }

  const requestClose = () => {
    if (confirmationBusyRef.current) return
    if (confirmation) {
      cancelConfirmation()
      return
    }
    // The existing sortable owns Escape while a message is being dragged. Its
    // document listener still receives this event and cancels before a later exit.
    if (editorContentRef.current?.querySelector('[aria-roledescription="sortable"][aria-pressed="true"]')) return
    blurEditorField()
    onClose()
  }

  const runConfirmedAction = () => {
    if (!confirmation || confirmationBusyRef.current) return
    const pending = confirmation
    confirmationBusyRef.current = true
    setConfirmationBusy(true)
    setConfirmationError('')
    void enqueueMutation(async () => {
      const current = presetRef.current
      if (!current || current.id !== pending.presetId) {
        setConfirmation(null)
        return
      }
      if (!current.etag) throw new Error('Preset ETag is missing; reload settings')
      if (pending.action === 'delete') {
        if (current.builtin || presets.length <= 1) {
          setConfirmation(null)
          return
        }
        if (secrets.llm_tagger.current_preset === current.id) {
          const replacement = presets.find((item) => item.id !== current.id)
          if (!replacement) return
          await api.setDefaultLLMPreset(replacement.id)
        }
        await api.deleteLLMPreset(current.id, etagRef.current || current.etag)
        await reloadSecrets()
        toast(t('llmPreset.deleted', { label: pending.label }), 'success')
      } else {
        if (!current.builtin) {
          setConfirmation(null)
          return
        }
        await api.resetLLMPreset(current.id, etagRef.current || current.etag)
        await reloadSecrets()
      }
      onClose()
    }, (error) => setConfirmationError(String(error))).finally(() => {
      confirmationBusyRef.current = false
      setConfirmationBusy(false)
    })
  }

  const assistNeedsTags =
    !!preset.assist_tagger
    && !preset.messages.some((m) => m.type === 'text' && m.content.includes('{{tags}}'))
  const assistHelp = t('llmPreset.assistTaggerHelp').split('%TAGS%').join('{{tags}}')

  return (
    <Modal
      title={confirmation ? t('common.dialogConfirmTitle') : (
        <span className="flex min-w-0 flex-wrap items-baseline gap-related">
          <span>{t('llmPreset.title')}</span>
          <span className="min-w-0 break-words text-sm font-normal text-fg-tertiary">{llmPresetLabel(preset, t)}</span>
          {preset.builtin && <Badge tone="neutral" size="sm">{t('llmPreset.builtin')}</Badge>}
        </span>
      )}
      description={confirmation ? t(
        confirmation.action === 'delete' ? 'settings.confirmDeletePreset' : 'settings.confirmResetPreset',
        { label: confirmation.label },
      ) : undefined}
      role={confirmation ? 'alertdialog' : 'dialog'}
      size={confirmation ? 'sm' : 'wide'}
      panelClassName={confirmation ? '' : 'h-[88dvh]'}
      bodyClassName={confirmation ? '!p-0' : '!overflow-hidden !p-0 flex flex-1 flex-col'}
      initialFocusRef={closeButtonRef}
      closeOnEscape={!confirmationBusy}
      closeOnBackdrop={!confirmationBusy}
      onClose={requestClose}
      testId="llm-preset-editor-modal"
      headerActions={!confirmation && (
        <Button ref={closeButtonRef} variant="ghost" size="sm" iconOnly onClick={requestClose}
          aria-label={t('common.close', { defaultValue: 'Close' })}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            <path d="M6 6l12 12M18 6 6 18" />
          </svg>
        </Button>
      )}
      footer={(
        <>
          <div hidden={Boolean(confirmation)}>
            <div className="flex flex-wrap items-center gap-related">
              <div className="mr-auto flex flex-wrap items-center gap-related">
                {preset.builtin && (
                  <Button variant="danger" size="sm" onClick={(event) => beginConfirmation('reset', event.currentTarget)}>
                    {t('llmPreset.resetBuiltin')}
                  </Button>
                )}
                <Button variant="ghost" size="sm" onClick={saveAsCopy}>{t('llmPreset.saveAsCopy')}</Button>
                <Button variant="ghost" size="sm" onClick={exportPreset} title={t('llmPreset.exportTitle')}>
                  {t('llmPreset.export')}
                </Button>
              </div>
              <ActionGroup
                secondary={!preset.builtin && presets.length > 1 ? (
                  <Button variant="danger" size="sm" onClick={(event) => beginConfirmation('delete', event.currentTarget)}>
                    {t('common.delete')}
                  </Button>
                ) : undefined}
                primary={<Button variant="primary" onClick={requestClose}>{t('llmPreset.done')}</Button>}
              />
            </div>
          </div>
          {confirmation && (
            <ActionGroup
              secondary={(
                <Button ref={cancelButtonRef} variant="secondary" onClick={cancelConfirmation} disabled={confirmationBusy}>
                  {t('common.cancel')}
                </Button>
              )}
              primary={(
                <Button variant="danger" loading={confirmationBusy} onClick={runConfirmedAction}>
                  {t('common.confirm')}
                </Button>
              )}
            />
          )}
        </>
      )}
    >
      <div ref={editorContentRef} hidden={Boolean(confirmation)} className="h-full min-h-0">
        {/* Preserve both input subtrees and their scroll owners across confirmation. */}
        <div className="h-full min-h-0 grid grid-cols-1 md:grid-cols-[1fr_2fr]">
          <div data-llm-editor-scroll="parameters" className="min-h-0 overflow-y-auto px-5 py-4 flex flex-col gap-4 border-r border-subtle">
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
                  onCommit={patchApiKey}
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
          <div data-llm-editor-scroll="messages" className="min-h-0 overflow-y-auto px-5 py-4">
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
      </div>
      {confirmation && confirmationError && (
        <div className="p-page">
          <Alert tone="danger" role="alert">{confirmationError}</Alert>
        </div>
      )}
    </Modal>
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
