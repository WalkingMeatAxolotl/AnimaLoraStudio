import { useCallback, useEffect, useRef, useState } from 'react'
import type { TFunction } from 'i18next'
import { Trans, useTranslation } from 'react-i18next'
import {
  api,
  type FlashAttnStatus,
  type ModelsCatalog,
  type Secrets,
  type TorchCuTag,
  type TorchStatus,
  type WD14Runtime,
  type XformersStatus,
} from '../../../api/client'
import { useDialog } from '../../../components/Dialog'
import { InfoButton } from '../../../components/InfoButton'
import PathPicker from '../../../components/PathPicker'
import { useShowTagTranslation } from '../../../tagDict/showToggle'
import { useTagDict, reloadDict } from '../../../tagDict/store'
import { useToast } from '../../../components/Toast'
import { applyDensity, applyTheme, getStoredDensity, getStoredTheme, setStoredDensity, setStoredTheme, type Density, type Theme } from '../../../lib/theme'
import i18n, { getStoredLangWithDefault, setStoredLang } from '../../../i18n'
import { MODEL_DESCRIPTION_KEYS, textInputClass, translatedCatalogText, UPSCALER_DESCRIPTION_KEYS, type Section } from './constants'
import { Bool, SettingsField, SettingsInput, SettingsSection } from './fields'
import { DownloadButton, ModelGroupCard, ModelStatusBadge, SourceSelect, StatusLabel } from './modelCards'

// ── 训练参数 Section ─────────────────────────────────────────────────
//
// 训练相关的全局开关。当前只有「自动配置模型路径」(auto_sync_paths)：独立 PUT，
// 不进全局 dirty 流程。原先夹在「训练模型」section 里，挪到这里跟模型下载分开。
export function TrainingParamsSection() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [autoSyncPaths, setAutoSyncPaths] = useState<boolean>(true)
  const [savingAutoSync, setSavingAutoSync] = useState(false)

  useEffect(() => {
    void api.getSecrets().then((sec) => {
      setAutoSyncPaths(sec.models?.auto_sync_paths ?? true)
    }).catch(() => { /* 显示用，拉不到不阻塞 */ })
  }, [])

  const saveAutoSync = async (next: boolean) => {
    setSavingAutoSync(true)
    const prev = autoSyncPaths
    setAutoSyncPaths(next)
    try {
      await api.updateSecrets({ models: { auto_sync_paths: next } })
      toast(next ? t('settings.autoSyncPathsOn') : t('settings.autoSyncPathsOff'), 'success')
    } catch (e) {
      setAutoSyncPaths(prev)
      toast(String(e), 'error')
    } finally {
      setSavingAutoSync(false)
    }
  }

  return (
    <SettingsSection id="training-params" title={t('settings.trainingParams')}>
      <SettingsField
        label={t('settings.autoSyncPathsLabel')}
        helpTooltip={<p>{t('settings.autoSyncPathsHelp')}</p>}
      >
        <Bool value={autoSyncPaths} onChange={(v) => void saveAutoSync(v)} disabled={savingAutoSync} />
      </SettingsField>
    </SettingsSection>
  )
}

export function ModelsSection({ catalog, busy, start, setSource, reloadCatalog, catalogError, t }: {
  catalog: ModelsCatalog | null
  busy: Set<string>
  start: (model_id: string, variant?: string) => Promise<void>
  setSource: (type: string, source: string) => Promise<void>
  reloadCatalog: () => Promise<ModelsCatalog | null>
  catalogError: string | null
  t: TFunction
}) {
  const { toast } = useToast()
  const [selectedAnima, setSelectedAnima] = useState<string>('1.0')
  const [showPicker, setShowPicker] = useState(false)
  const [addingCustom, setAddingCustom] = useState(false)

  // 一次性拉 secrets 取 selected_anima（独立 PUT，不进全局 dirty 流程）。模型
  // 根目录已挪到「系统 → 存储位置」、自动配置模型路径已挪到「训练参数」，不在此渲染。
  useEffect(() => {
    void api.getSecrets().then((sec) => {
      setSelectedAnima(sec.models?.selected_anima ?? '1.0')
    }).catch(() => { /* 显示用，拉不到不阻塞 */ })
  }, [])

  const pickAnima = async (variant: string) => {
    if (variant === selectedAnima) return
    setSelectedAnima(variant)
    try {
      await api.updateSecrets({ models: { selected_anima: variant } })
      toast(t('settings.mainModelSelected', { name: variant }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
      void reloadCatalog()
    }
  }

  // PathPicker 选中本地 .safetensors → 注册到 custom_anima_paths（仅登记路径）。
  // 注册后不自动选中：与官方 variant「下载完再点 radio」的流程一致。
  const addCustom = async (picked: string) => {
    setShowPicker(false)
    const p = picked.trim()
    if (!p) return
    if (!p.toLowerCase().endsWith('.safetensors')) {
      toast(t('settings.localModelInvalidExt'), 'error')
      return
    }
    setAddingCustom(true)
    try {
      await api.addCustomAnima(p)
      toast(t('settings.localModelAdded', { name: p.split(/[\\/]/).pop() }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setAddingCustom(false)
    }
  }

  const removeCustom = async (p: string) => {
    try {
      await api.removeCustomAnima(p)
      if (p === selectedAnima) setSelectedAnima('1.0')
      toast(t('settings.localModelRemoved'), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const error = catalogError

  return (
    <SettingsSection id="models" title={t('settings.trainingModelsOneClick')}>
      <SourceSelect
        opt={catalog?.download_source_options?.training}
        onChange={(s) => void setSource('training', s)}
      />

      {error && <div className="text-err text-xs font-mono">{error}</div>}
      {!catalog ? (
        <p className="text-fg-tertiary text-xs">{t('settings.loadingModelCatalog')}</p>
      ) : (
        <div className="flex flex-col gap-2">
          {/* Anima 主模型 */}
          <ModelGroupCard
            title={catalog.anima_main.name}
            helpTooltip={
              <>
                <p><Trans i18nKey="settings.repoHelp" values={{ desc: translatedCatalogText(MODEL_DESCRIPTION_KEYS, 'anima_main', catalog.anima_main.description, t), repo: catalog.anima_main.repo }} components={{ code: <code /> }} /></p>
                <p><Trans i18nKey="settings.defaultTransformerHelp" components={{ strong: <strong /> }} /></p>
              </>
            }
          >
            <ul className="list-none m-0 p-0 flex flex-col gap-1">
              {catalog.anima_main.variants.map((v) => {
                const key = `anima_main:${v.variant}`
                const dl = catalog.downloads[key]
                const isSel = v.variant === selectedAnima
                const canSelect = v.exists && dl?.status !== 'running'
                return (
                  <li key={v.variant} className={`flex items-center gap-2 text-xs px-1.5 py-1 rounded-sm ${
                    isSel ? 'bg-accent-soft border border-accent' : 'bg-transparent border border-transparent'
                  }`}>
                    <input type="radio" name="anima_variant" checked={isSel} disabled={!canSelect}
                      onChange={() => void pickAnima(v.variant)}
                      className="shrink-0"
                      style={{ accentColor: 'var(--accent)' }}
                      title={canSelect ? t('settings.selectDefaultMainModel') : v.exists ? t('settings.downloadInProgress') : t('settings.downloadRequiredFirst')}
                    />
                    <code className="font-mono text-fg-primary w-32 shrink-0">{v.variant}</code>
                    <ModelStatusBadge exists={v.exists} size={v.size} status={dl?.status} />
                    <span style={{ flex: 1 }} />
                    <DownloadButton exists={v.exists} status={dl?.status} busy={busy.has(key)} onClick={() => void start('anima_main', v.variant)} />
                  </li>
                )
              })}
              {/* 用户注册的本地 custom 主模型（微调权重 / 在微调上测试） */}
              {catalog.anima_main.custom.map((c) => {
                const isSel = c.path === selectedAnima
                return (
                  <li key={c.path} className={`flex items-center gap-2 text-xs px-1.5 py-1 rounded-sm ${
                    isSel ? 'bg-accent-soft border border-accent' : 'bg-transparent border border-transparent'
                  }`}>
                    <input type="radio" name="anima_variant" checked={isSel} disabled={!c.exists}
                      onChange={() => void pickAnima(c.path)}
                      className="shrink-0"
                      style={{ accentColor: 'var(--accent)' }}
                      title={c.exists ? t('settings.selectDefaultMainModel') : t('settings.localModelMissing')}
                    />
                    <code className="font-mono text-fg-primary w-32 shrink-0 truncate" title={c.path}>{c.name}</code>
                    {c.exists
                      ? <ModelStatusBadge exists size={c.size} />
                      : <span className="text-err text-2xs">{t('settings.localModelMissing')}</span>}
                    <span className="text-2xs px-1 py-0.5 rounded-sm bg-overlay text-fg-tertiary shrink-0">{t('settings.storage.customBadge')}</span>
                    <span style={{ flex: 1 }} />
                    <button
                      onClick={() => void removeCustom(c.path)}
                      className="btn btn-secondary btn-sm shrink-0 min-w-[5rem] justify-center"
                      title={t('settings.removeLocalModel')}
                    >🗑 {t('settings.removeLocalModelShort')}</button>
                  </li>
                )
              })}
            </ul>
            <button
              onClick={() => setShowPicker(true)}
              disabled={addingCustom}
              className="btn btn-ghost btn-sm self-start mt-1"
            >
              {addingCustom ? t('common.saving') : t('settings.addLocalModel')}
            </button>
          </ModelGroupCard>

          {/* VAE */}
          <ModelGroupCard title={catalog.anima_vae.name}>
            <div className="flex items-center gap-2 text-xs">
              <span className="text-fg-tertiary">{translatedCatalogText(MODEL_DESCRIPTION_KEYS, 'anima_vae', catalog.anima_vae.description, t)} · <code>{catalog.anima_vae.repo}</code></span>
              <span style={{ flex: 1 }} />
              <ModelStatusBadge exists={catalog.anima_vae.exists} size={catalog.anima_vae.size} status={catalog.downloads.anima_vae?.status} />
              <DownloadButton exists={catalog.anima_vae.exists} status={catalog.downloads.anima_vae?.status} busy={busy.has('anima_vae')} onClick={() => void start('anima_vae')} />
            </div>
          </ModelGroupCard>

          {/* Qwen3 + T5（CLTagger 已挪到「打标」tab） */}
          {(['qwen3', 't5_tokenizer'] as const).map((id) => {
            const m = catalog[id]
            const dl = catalog.downloads[id]
            const allExist = m.files.every((f) => f.exists)
            const totalSize = m.files.reduce((s, f) => s + f.size, 0)
            return (
              <ModelGroupCard key={id} title={m.name}>
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-fg-tertiary">{translatedCatalogText(MODEL_DESCRIPTION_KEYS, id, m.description, t)} · <code>{m.repo}</code></span>
                  <span style={{ flex: 1 }} />
                  <ModelStatusBadge exists={allExist} size={totalSize} status={dl?.status} fileCount={m.files.length} existsCount={m.files.filter((f) => f.exists).length} />
                  <DownloadButton exists={allExist} status={dl?.status} busy={busy.has(id)} onClick={() => void start(id)} />
                </div>
              </ModelGroupCard>
            )
          })}

          {/* 下载日志 */}
          {Object.values(catalog.downloads).filter((d) => d.status === 'running' || d.status === 'failed').length > 0 && (
            <details className="text-xs">
              <summary className="cursor-pointer text-fg-tertiary">
                {t('settings.downloadLogs', { n: Object.values(catalog.downloads).filter((d) => d.status === 'running' || d.status === 'failed').length })}
              </summary>
              <div className="mt-1 flex flex-col gap-2">
                {Object.values(catalog.downloads).map((d) => (
                  <div key={d.key} className="rounded-sm border border-subtle bg-sunken p-2">
                    <div className="flex items-center gap-2 mb-1">
                      <code className="font-mono text-fg-secondary">{d.key}</code>
                      <ModelStatusBadge exists={d.status === 'done'} size={0} status={d.status} />
                      {d.message && <span className="text-err overflow-hidden text-ellipsis whitespace-nowrap">{d.message}</span>}
                    </div>
                    <pre className="text-xs font-mono text-fg-tertiary max-h-32 overflow-auto whitespace-pre-wrap m-0">
                      {d.log_tail.join('\n') || t('settings.emptyLog')}
                    </pre>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
      {showPicker && (
        <PathPicker
          initialPath={catalog?.models_root ?? undefined}
          onPick={(p) => void addCustom(p)}
          onClose={() => setShowPicker(false)}
        />
      )}
    </SettingsSection>
  )
}

export function UpscalerSection({
  catalog, busy, start, setSource, reloadCatalog, t,
}: {
  catalog: ModelsCatalog | null
  busy: Set<string>
  start: (model_id: string, variant?: string) => Promise<void>
  setSource: (type: string, source: string) => Promise<void>
  reloadCatalog: () => Promise<ModelsCatalog | null>
  t: TFunction
}) {
  const { toast } = useToast()
  const [customSource, setCustomSource] = useState<'hf' | 'ms'>('hf')
  const [customRepo, setCustomRepo] = useState('')
  const [customFile, setCustomFile] = useState('')
  const [customBusy, setCustomBusy] = useState(false)

  const pickUpscaler = async (label: string) => {
    try {
      await api.selectUpscaler(label)
      toast(t('settings.defaultUpscaler', { name: label }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const submitCustom = async () => {
    const repo = customRepo.trim()
    const file = customFile.trim()
    if (!repo || !file) {
      toast(t('settings.repoAndFilenameRequired'), 'error')
      return
    }
    setCustomBusy(true)
    try {
      await api.startUpscalerCustomDownload({
        source: customSource, repo_id: repo, filename: file,
      })
      toast(t('settings.downloadStarted', { name: file }), 'success')
      setCustomRepo('')
      setCustomFile('')
      // SSE 推 model_download_changed 会刷 catalog；这里兜底
      setTimeout(() => void reloadCatalog(), 1500)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setCustomBusy(false)
    }
  }

  const variants = catalog?.upscalers?.variants ?? []
  const current = catalog?.upscalers?.current ?? ''

  return (
    <SettingsSection id="upscalers" title={t('settings.upscalersPreprocess')}>
      <SourceSelect
        opt={catalog?.download_source_options?.upscaler}
        onChange={(s) => void setSource('upscaler', s)}
      />
      {!catalog ? (
        <p className="text-fg-tertiary text-xs">{t('common.loading')}</p>
      ) : (
        <div className="flex flex-col gap-2">
          <ModelGroupCard
            title={t('settings.availableUpscalers')}
            helpTooltip={
              <>
                <p><Trans i18nKey="settings.upscalersHelpPath" values={{ path: catalog.upscalers?.target_dir }} components={{ code: <code /> }} /></p>
                <p>{t('settings.upscalersHelpDefault')}</p>
              </>
            }
          >
            <ul className="list-none m-0 p-0 flex flex-col gap-1">
              {variants.map((v) => {
                const key = v.kind === 'custom'
                  ? `upscaler:custom:${v.filename}`
                  : `upscaler:${v.label}`
                const dl = catalog.downloads[key]
                const isSel = v.label === current
                const canSelect = v.exists && dl?.status !== 'running'
                return (
                  <li key={v.label} className={`flex items-center gap-2 text-xs px-1.5 py-1 rounded-sm ${
                    isSel ? 'bg-accent-soft border border-accent' : 'bg-transparent border border-transparent'
                  }`}>
                    <input
                      type="radio"
                      name="selected_upscaler"
                      checked={isSel}
                      disabled={!canSelect}
                      onChange={() => void pickUpscaler(v.label)}
                      className="shrink-0"
                      style={{ accentColor: 'var(--accent)' }}
                      title={canSelect ? t('settings.selectDefaultPreprocess') : v.exists ? t('settings.downloadInProgress') : t('settings.notDownloaded')}
                    />
                    <div className="flex flex-col min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <code className="font-mono text-fg-primary truncate">{v.label}</code>
                        {v.kind === 'custom' && (
                          <span className="text-[10px] px-1 py-0 rounded-sm bg-sunken text-fg-tertiary">custom</span>
                        )}
                      </div>
                      <span className="text-fg-tertiary text-[11px] truncate">
                        {translatedCatalogText(UPSCALER_DESCRIPTION_KEYS, v.label, v.description, t)}
                        {v.hf_repo && <> · HF <code>{v.hf_repo}</code></>}
                        {v.ms_repo && <> · MS <code>{v.ms_repo}</code></>}
                        {v.size_mb != null && <> · ~{v.size_mb} MB</>}
                      </span>
                    </div>
                    <ModelStatusBadge exists={v.exists} size={v.size} status={dl?.status} />
                    {v.kind === 'preset' && (
                      <DownloadButton
                        exists={v.exists}
                        status={dl?.status}
                        busy={busy.has(`upscaler:${v.label}`)}
                        onClick={() => void start('upscaler', v.label)}
                      />
                    )}
                  </li>
                )
              })}
            </ul>
          </ModelGroupCard>

          <ModelGroupCard
            title={t('settings.customDownload')}
            helpTooltip={
              <>
                <p><Trans i18nKey="settings.customUpscalerHelpTypes" components={{ code: <code /> }} /></p>
                <p><Trans i18nKey="settings.customUpscalerHelpSources" components={{ code: <code /> }} /></p>
                <p>{t('settings.customUpscalerHelpEnable')}</p>
              </>
            }
          >
            <div className="flex flex-col gap-2 text-xs">
              <SettingsField label={t('settings.source')}>
                <select
                  value={customSource}
                  onChange={(e) => setCustomSource(e.target.value as 'hf' | 'ms')}
                  className={`${textInputClass} max-w-32`}
                >
                  <option value="hf">HuggingFace</option>
                  <option value="ms">ModelScope</option>
                </select>
              </SettingsField>
              <SettingsField label={t('settings.repoId')}>
                <input
                  type="text"
                  value={customRepo}
                  onChange={(e) => setCustomRepo(e.target.value)}
                  placeholder={customSource === 'hf' ? 'Kim2091/UltraSharp' : 'libfishopen/upscaler'}
                  className={`${textInputClass} flex-1 font-mono`}
                />
              </SettingsField>
              <SettingsField label={t('common.filename')}>
                <input
                  type="text"
                  value={customFile}
                  onChange={(e) => setCustomFile(e.target.value)}
                  placeholder="4x-UltraSharp.pth"
                  className={`${textInputClass} flex-1 font-mono`}
                />
              </SettingsField>
              <div className="flex justify-end">
                <button
                  onClick={() => void submitCustom()}
                  disabled={customBusy || !customRepo.trim() || !customFile.trim()}
                  className="btn btn-primary btn-sm"
                >
                  {customBusy ? t('settings.downloadInProgress') : t('common.download')}
                </button>
              </div>
            </div>
          </ModelGroupCard>

          {/* 下载日志 */}
          {Object.values(catalog.downloads).filter((d) => d.key.startsWith('upscaler') && (d.status === 'running' || d.status === 'failed')).length > 0 && (
            <details className="text-xs">
              <summary className="cursor-pointer text-fg-tertiary">{t('settings.upscalerDownloadLogs')}</summary>
              <div className="mt-1 flex flex-col gap-2">
                {Object.values(catalog.downloads).filter((d) => d.key.startsWith('upscaler')).map((d) => (
                  <div key={d.key} className="rounded-sm border border-subtle bg-sunken p-2">
                    <div className="flex items-center gap-2 mb-1">
                      <code className="font-mono text-fg-secondary">{d.key}</code>
                      <ModelStatusBadge exists={d.status === 'done'} size={0} status={d.status} />
                      {d.message && <span className="text-err overflow-hidden text-ellipsis whitespace-nowrap">{d.message}</span>}
                    </div>
                    <pre className="text-xs font-mono text-fg-tertiary max-h-32 overflow-auto whitespace-pre-wrap m-0">
                      {d.log_tail.join('\n') || t('settings.emptyLog')}
                    </pre>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </SettingsSection>
  )
}

// ── Tag 翻译词典：上传 / 恢复默认 / 全局 chip toggle ──────────────────────

export function TagDictionarySection() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const dict = useTagDict()
  const [show, setShow] = useShowTagTranslation()
  const [busy, setBusy] = useState<null | 'reset' | 'upload'>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const meta = dict.meta
  const sourceLabel = meta?.kind === 'default'
    ? t('settings.tagDictionary.sourceDefault')
    : meta?.kind === 'user'
      ? t('settings.tagDictionary.sourceUser')
      : t('settings.tagDictionary.sourceUnknown')

  const downloadedAt = meta?.downloaded_at
    ? new Date(meta.downloaded_at * 1000).toLocaleString()
    : '—'

  const reset = async () => {
    if (busy) return
    setBusy('reset')
    try {
      await api.resetTagDictionary()
      await reloadDict()
      toast(t('settings.tagDictionary.resetOk'))
    } catch (err) {
      toast(`${t('settings.tagDictionary.resetFail')}: ${err instanceof Error ? err.message : String(err)}`)
    } finally { setBusy(null) }
  }

  const upload = async (file: File) => {
    setBusy('upload')
    try {
      await api.uploadTagDictionary(file)
      await reloadDict()
      toast(t('settings.tagDictionary.uploadOk', { name: file.name }))
    } catch (err) {
      toast(`${t('settings.tagDictionary.uploadFail')}: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setBusy(null)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  return (
    <SettingsSection id="tag-dictionary" title={t('settings.tagDictionary.title')}>
      <SettingsField label={t('settings.tagDictionary.statusLabel')}>
        {dict.status === 'loading' && (
          <span className="text-xs text-fg-tertiary">{t('settings.tagDictionary.loading')}</span>
        )}
        {dict.status === 'empty' && (
          <span className="text-xs text-warn">{t('settings.tagDictionary.empty')}</span>
        )}
        {dict.status === 'error' && (
          <span className="text-xs text-err">{dict.error ?? t('settings.tagDictionary.error')}</span>
        )}
        {dict.status === 'ready' && meta && (
          <div className="text-xs flex flex-col gap-0.5">
            <div>
              <span className="text-fg-tertiary">{sourceLabel}：</span>
              <code className="font-mono text-fg-primary">{meta.source_name}</code>
            </div>
            <div className="text-fg-tertiary">
              {t('settings.tagDictionary.entryCount', { n: meta.entry_count })} · {downloadedAt}
            </div>
          </div>
        )}
      </SettingsField>

      <SettingsField
        label={t('settings.tagDictionary.uploadLabel')}
        desc={t('settings.tagDictionary.uploadHint')}
      >
        <div className="flex gap-1.5 items-center flex-wrap">
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.txt"
            disabled={busy !== null}
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) void upload(f)
            }}
            className="text-xs"
          />
          <button
            type="button"
            disabled={busy !== null}
            onClick={() => void reset()}
            className="btn btn-secondary btn-sm"
            title={t('settings.tagDictionary.resetHint')}
          >
            {busy === 'reset' ? t('common.downloading') : t('settings.tagDictionary.resetButton')}
          </button>
        </div>
      </SettingsField>

      <SettingsField
        label={t('settings.tagDictionary.showToggleLabel')}
        desc={t('settings.tagDictionary.showToggleHint')}
      >
        <Bool value={show} onChange={setShow} />
      </SettingsField>
    </SettingsSection>
  )
}

// ── ONNX Runtime Section（WD14 + CLTagger 共用 onnxruntime 包管理） ─────────

export function ONNXRuntimeSection() {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [rt, setRt] = useState<WD14Runtime | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<null | 'auto' | 'gpu' | 'cpu' | 'directml'>(null)
  const [reinstallOpen, setReinstallOpen] = useState(false)
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const r = await api.getWD14Runtime()
      setRt(r)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const install = async (target: 'auto' | 'gpu' | 'cpu' | 'directml') => {
    const confirmKey = target === 'auto'
      ? 'settings.confirmInstallOnnxAuto'
      : target === 'gpu'
        ? 'settings.confirmInstallOnnxGpu'
        : target === 'directml'
          ? 'settings.confirmInstallOnnxDirectml'
          : 'settings.confirmInstallOnnxCpu'
    const ok = await dialog.confirm(
      t(confirmKey),
      { tone: 'warn', okText: t('settings.startInstall') },
    )
    if (!ok) return
    setBusy(target)
    try {
      const result = await api.installWD14Runtime(target)
      setRt({
        installed: result.installed, version: result.version, providers: result.providers,
        cuda_available: result.cuda_available,
        directml_available: result.directml_available,
        platform: result.platform,
        restart_required: result.restart_required,
        cuda_load_error: result.cuda_load_error, preload: result.preload, cuda_detect: result.cuda_detect,
        torch_cuda_major: result.torch_cuda_major, ort_cuda_major_mismatch: result.ort_cuda_major_mismatch,
      })
      const newPkg = result.installed_pkg ?? result.installed ?? '?'
      const newVer = result.installed_version ?? result.version ?? '?'
      toast(t('settings.packageInstalledRestart', { pkg: newPkg, version: newVer }), 'success')
    } catch (e) {
      toast(t('settings.packageInstallFailed', { error: String(e) }), 'error')
    } finally {
      setBusy(null)
    }
  }

  const cuda = rt?.cuda_detect ?? { available: false, driver_version: null, gpu_name: null }
  const notInstalled = !!rt && rt.installed === null
  const gpuAccel = !!rt && (rt.cuda_available || rt.directml_available)
  const isWindows = !rt || rt.platform === 'win32'
  // mismatched: 装了某个包 + 有 GPU 但没用上任何加速 EP（CUDA 或 DirectML）。
  // 未装由 notInstalled 接管；已装且已经在用 DirectML 不算 mismatch。
  const mismatched = !!rt && rt.installed !== null && cuda.available && !gpuAccel
  // 默认状态正常时整体折叠；未装 / 有错 / mismatch / 需重启时自动展开
  const hasIssue = !!error || (rt && (
    notInstalled || !!rt.cuda_load_error || rt.restart_required || mismatched
  ))

  // summary 里显示一行简短状态，用户不展开就能扫到
  const epShort = !rt
    ? '?'
    : rt.cuda_available ? 'CUDA' : rt.directml_available ? 'DirectML' : 'CPU'
  const statusLabel = error
    ? `⚠ ${t('settings.statusLoadFailed')}`
    : !rt
      ? t('settings.loadingStatus')
      : notInstalled
        ? `⚠ ${t('settings.notInstalledShort')}`
        : rt.cuda_load_error
          ? `⚠ ${t('settings.cudaLoadFailed')}`
          : rt.restart_required
            ? `⚠ ${t('settings.restartStudioRequired')}`
            : mismatched
              ? `⚠ ${t('settings.gpuRunningCpuEp')}`
              : `${epShort} · ${rt.installed ?? '?'}`
  const statusOk = rt && !hasIssue

  return (
    <details id="onnxruntime" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">ONNX Runtime</h2>
        <span className="text-xs text-fg-tertiary">{t('settings.sharedByWd14ClTagger')}</span>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : 'text-warn'}`}>{statusLabel}</span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !rt && <div className="text-xs text-fg-tertiary">{t('settings.loadingRuntimeStatus')}</div>}
        {rt && (
          <>
            <div className="rounded-sm border border-subtle bg-sunken p-2 flex flex-col gap-1 text-xs">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-fg-tertiary shrink-0">runtime:</span>
                <code className="font-mono text-fg-primary">{rt.installed ?? t('settings.notInstalledParen')}{rt.version ? `==${rt.version}` : ''}</code>
                <StatusLabel bg={gpuAccel ? 'bg-ok-soft' : 'bg-warn-soft'} fg={gpuAccel ? 'text-ok' : 'text-warn'} text={rt.cuda_available ? 'CUDA' : rt.directml_available ? 'DirectML' : 'CPU only'} />
              </div>
              <div className="text-fg-tertiary">EP: <code className="text-fg-secondary font-mono">{(rt.providers ?? []).map((p) => p.replace('ExecutionProvider', '')).join(' / ') || '(none)'}</code></div>
              <div className="text-fg-tertiary">{t('settings.gpuDetect')}: <span className="text-fg-secondary">{cuda.available ? `${cuda.gpu_name ?? '?'} (driver ${cuda.driver_version ?? '?'})` : t('settings.noNvidiaGpu')}</span></div>
              {rt.torch_cuda_major != null && (
                <div className="text-fg-tertiary">
                  {t('settings.torchCudaMajor')}: <span className="text-fg-secondary font-mono">{rt.torch_cuda_major}</span>
                  {rt.ort_cuda_major_mismatch && (
                    <span className="text-warn ml-2">⚠ {t('settings.ortCudaMismatch')}</span>
                  )}
                </div>
              )}
            </div>

            {rt.restart_required && (
              <div className="rounded-sm border border-err bg-err-soft px-2 py-1.5 text-err text-xs">
                <Trans i18nKey="settings.onnxRestartRequired" components={{ strong: <strong /> }} />
              </div>
            )}
            {!rt.restart_required && notInstalled && (
              <div className="rounded-sm border border-info bg-info-soft px-2 py-1.5 text-info text-xs">
                {cuda.available ? t('settings.onnxNotInstalledHintGpu') : t('settings.onnxNotInstalledHintCpu')}
              </div>
            )}
            {!rt.restart_required && mismatched && (
              <div className="rounded-sm border border-info bg-info-soft px-2 py-1.5 text-info text-xs">
                {t('settings.onnxCpuEpWarning')}
              </div>
            )}
            {rt.cuda_load_error && (
              <div className="rounded-sm border border-err bg-err-soft px-2 py-1.5 text-xs text-err">
                <div>{t('settings.cudaEpFailedCpu')}</div>
                <code className="block font-mono text-xs text-err break-all whitespace-pre-wrap mt-1">
                  {rt.cuda_load_error}
                </code>
              </div>
            )}

            <div className="flex gap-1.5 items-center flex-wrap">
              <button onClick={() => install('auto')} disabled={busy !== null} className="btn btn-primary btn-sm">
                {busy === 'auto' ? t('settings.installingPackage') : t('settings.autoDetectInstall')}
              </button>
              <button onClick={() => void refresh()} disabled={busy !== null} title={t('settings.refreshStatus')}
                className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm">↻</button>
              <button type="button" onClick={() => setReinstallOpen(!reinstallOpen)}
                className="btn btn-ghost btn-sm text-xs text-fg-tertiary ml-auto">
                {reinstallOpen ? '▾' : '▸'} {t('settings.forceReinstallAdvanced')}
              </button>
            </div>
            {reinstallOpen && (
              <div className="flex flex-col gap-2 pt-2 border-t border-subtle">
                <div className="flex gap-1.5 items-center flex-wrap">
                  <button
                    onClick={() => install('directml')}
                    disabled={busy !== null || !isWindows}
                    title={isWindows ? t('settings.directmlPackageHint') : t('settings.directmlWinOnlyHint')}
                    className="btn btn-secondary btn-sm"
                  >
                    {busy === 'directml' ? t('settings.installingPackage') : t('settings.reinstallDirectml')}
                  </button>
                  <button
                    onClick={() => install('gpu')}
                    disabled={busy !== null}
                    title={t('settings.cudaPackageHint')}
                    className="btn btn-secondary btn-sm"
                  >
                    {busy === 'gpu' ? t('settings.installingPackage') : t('settings.reinstallGpu')}
                  </button>
                  <button
                    onClick={() => install('cpu')}
                    disabled={busy !== null}
                    title={t('settings.cpuPackageHint')}
                    className="btn btn-secondary btn-sm"
                  >
                    {busy === 'cpu' ? t('settings.installingPackage') : t('settings.reinstallCpu')}
                  </button>
                </div>
                <span className="text-[10px] text-fg-tertiary">{t('settings.onnxForceHint')}</span>
              </div>
            )}
          </>
        )}
      </div>
    </details>
  )
}

// ── PyTorch Section（训练 tab）──────────────────────────────────────────────
//
// 已有 venv 用户的「一键修」入口。PR-4 启动期会 warn「检测到 GPU 但 torch 是
// CPU 版」并给 pip 命令；这里把命令 UI 化，普通用户不用进终端。
//
// 三种状态：
// - cuda_available=True               → ✓ 一切 OK（折叠默认；提供「换 CUDA 版本」高级选项）
// - is_cpu_with_gpu=True               → 红色误装提示 + 显著「重装为 CUDA」主按钮
// - is_cuda_build_unavailable=True     → 黄色驱动警告（pip 修不了，给文档链接）

export function PyTorchSection() {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [status, setStatus] = useState<TorchStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const s = await api.getTorchStatus()
      setStatus(s)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const reinstall = async (target: 'auto' | TorchCuTag) => {
    const tag = target === 'auto' ? status?.recommended_cu_tag ?? '?' : target
    // 注册 → 用户 Ctrl+C 重启 → launcher 进程跑 pip。Windows 上 torch.pyd 被
    // 当前 server 进程锁住，没法直接 replace；只能 defer 到 launcher。
    if (!(await dialog.confirm(
      t('settings.confirmRegisterTorch', { tag }),
      { tone: 'warn', okText: t('settings.registerRequest') },
    ))) return
    setBusy(true)
    try {
      const result = await api.reinstallTorch(target)
      // 后端已写 marker，server 进程没真装；提示用户去重启
      toast(result.message, 'success')
    } catch (e) {
      toast(t('settings.registerFailed', { error: String(e) }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const hasIssue = !!error || (status && (status.is_cpu_with_gpu || status.is_cuda_build_unavailable || !status.installed))
  const statusOk = status?.cuda_available && !error
  const statusLabel = error
    ? t('settings.loadFailedShort')
    : !status
      ? t('settings.loadingStatus')
      : !status.installed
        ? t('settings.notInstalledShort')
        : status.is_cpu_with_gpu
          ? t('settings.cpuBuildMisinstalled')
          : !status.cuda_available && status.cuda_build !== 'cpu'
            ? t('settings.cudaUnavailableDriver')
            : status.cuda_available
              ? `CUDA ✓ ${status.cuda_build}`
              : `CPU ${status.cuda_build}`

  return (
    <details id="pytorch" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">PyTorch</h2>
        <span className="text-xs text-fg-tertiary">{t('settings.trainingCoreDependency')}</span>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : status?.is_cpu_with_gpu ? 'text-err' : 'text-warn'}`}>
          {statusLabel}
        </span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !status && <div className="text-xs text-fg-tertiary">{t('settings.loadingStatus')}</div>}

        {status && (<>
          {/* 当前状态卡 */}
          <div className="rounded-sm border border-subtle bg-sunken p-2 flex flex-col gap-1 text-xs">
            <div className="flex gap-4 flex-wrap">
              <span className="text-fg-tertiary">torch: <code className="text-fg-secondary font-mono">{status.version ?? t('settings.notInstalledParen')}</code></span>
              {status.cuda_build && (
                <span className="text-fg-tertiary">build: <code className="text-fg-secondary font-mono">{status.cuda_build}</code></span>
              )}
              {status.cuda_available && status.device_name && (
                <span className="text-fg-tertiary">GPU: <code className="text-fg-secondary font-mono">{status.device_name}</code></span>
              )}
            </div>
            <div className="flex gap-4 flex-wrap">
              <span className="text-fg-tertiary">
                {t('settings.driverLabel')}:{' '}
                <code className="text-fg-secondary font-mono">
                  {status.cuda_detect.driver_version ?? t('settings.notDetected')}
                </code>
              </span>
              {status.cuda_detect.gpu_name && !status.cuda_available && (
                <span className="text-fg-tertiary">
                  {t('settings.systemGpu')}:{' '}
                  <code className="text-fg-secondary font-mono">{status.cuda_detect.gpu_name}</code>
                </span>
              )}
            </div>
          </div>

          {/* 误装：CPU torch + 有 GPU */}
          {status.is_cpu_with_gpu && (
            <div className="rounded-sm border border-err bg-err-soft px-2 py-1.5 text-err text-xs">
              <Trans
                i18nKey="settings.torchCpuWithGpuWarning"
                values={{ tag: status.recommended_cu_tag }}
                components={{ code: <code className="font-mono" /> }}
              />
            </div>
          )}

          {/* CUDA build 但运行时不可用：驱动 / WSL 问题 */}
          {status.is_cuda_build_unavailable && (
            <div className="rounded-sm border border-warn bg-warn-soft px-2 py-1.5 text-warn text-xs">
              <Trans
                i18nKey="settings.torchCudaUnavailableWarning"
                components={{ code: <code className="font-mono" /> }}
              />
            </div>
          )}

          {/* 操作按钮 */}
          <div className="flex gap-1.5 items-center flex-wrap">
            <button
              onClick={() => void reinstall('auto')}
              disabled={busy || !status.cuda_detect.available}
              className={status.is_cpu_with_gpu ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm'}
              title={status.cuda_detect.available
                ? t('settings.autoSelect', { tag: status.recommended_cu_tag })
                : t('settings.noNvidiaDriverCannotCuda')}
            >
              {busy ? t('settings.installing') : status.is_cpu_with_gpu
                ? t('settings.reinstallCudaBuild', { tag: status.recommended_cu_tag })
                : t('settings.reinstallAuto', { tag: status.recommended_cu_tag })}
            </button>
            <button onClick={() => void refresh()} disabled={busy}
              className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm">↻</button>
            <button type="button" onClick={() => setAdvancedOpen(!advancedOpen)}
              className="btn btn-ghost btn-sm text-xs text-fg-tertiary ml-auto">
              {advancedOpen ? '▾' : '▸'} {t('settings.advancedManualCuda')}
            </button>
          </div>

          {/* 手动选版本 */}
          {advancedOpen && (
            <div className="flex flex-col gap-1.5 pt-2 border-t border-subtle text-xs">
              <p className="text-fg-tertiary m-0">
                {t('settings.manualCudaHint')}
              </p>
              <div className="flex gap-1.5 flex-wrap">
                {(['cu128', 'cu126', 'cu124', 'cu118', 'cpu'] as const).map((tag) => (
                  <button
                    key={tag}
                    onClick={() => void reinstall(tag)}
                    disabled={busy}
                    className={`btn btn-secondary btn-sm ${
                      status.cuda_build === tag ? 'border-accent' : ''
                    }`}
                    title={
                      tag === 'cpu'
                        ? t('settings.installCpuBuildHint')
                        : t('settings.installCudaBuildHint', { tag })
                    }
                  >
                    {tag}{status.cuda_build === tag ? ' ✓' : ''}
                  </button>
                ))}
              </div>
            </div>
          )}
        </>)}
      </div>
    </details>
  )
}

// ── Flash Attention Section（训练 tab）─────────────────────────────────────
//
// 训练加速的可选优化。装好 flash_attn 后启动期会自动 set_flash_attn_enabled(True)。
// 本组件给 UI 一键装 wheel 的能力，复用 PR-7a 的 service：状态 + GitHub 候选 + 安装。
//
// 设计要点：
// - install 是同步 pip（几分钟），用 confirm() + busy 状态防误触
// - Python ABI 不一致的 wheel（usable=false）灰显，但保留「强制安装」按钮（
//   极少数情况用户可能在 ABI 兼容子集里跑）
// - GitHub API 限流时 candidates=[] + fetch_error，给手动 URL 输入兜底

export function FlashAttentionSection() {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [status, setStatus] = useState<FlashAttnStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [candidatesOpen, setCandidatesOpen] = useState(false)
  const [manualUrl, setManualUrl] = useState('')
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const s = await api.getFlashAttnStatus()
      setStatus(s)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const install = async (url: string | null) => {
    const msg = url ? t('settings.confirmInstallFlashUrl') : t('settings.confirmInstallFlashAuto')
    if (!(await dialog.confirm(msg, { tone: 'warn', okText: t('settings.startInstall') }))) return
    setBusy(true)
    try {
      const result = await api.installFlashAttn(url)
      toast(t('settings.flashAttnInstalled', { version: result.version ?? '?' }), 'success')
      await refresh()
    } catch (e) {
      toast(t('settings.installFailed', { error: String(e) }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const env = status?.env
  const candidates = status?.candidates ?? []
  const fetchError = status?.fetch_error ?? null
  const usable = candidates.filter((c) => c.usable)
  const bestCandidate = usable[0] ?? null
  // CPU 版 torch 装不了 flash_attn —— UI 必须显著提示用户先去 PyTorch 那栏重装。
  // 否则用户只会看到「未找到 wheel」/「Internal Server Error」这种误导信息。
  const isCpuTorch = env?.torch_cuda_build === 'cpu'
  const hasIssue = !!error || (status && !status.installed)
  const canAutoInstall = !isCpuTorch && !!env?.torch_tag && !!env?.platform && usable.length > 0

  const statusLabel = error
    ? t('settings.loadFailedShort')
    : !status
      ? t('settings.loadingStatus')
      : status.installed
        ? t('settings.installedVersion', { version: status.version ?? '?' })
        : t('settings.notInstalledShort')
  const statusOk = status?.installed && !error

  return (
    <details id="flash-attn" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">Flash Attention</h2>
        <span className="text-xs text-fg-tertiary">{t('settings.trainingAccelerationOptional')}</span>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : 'text-warn'}`}>{statusLabel}</span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !status && <div className="text-xs text-fg-tertiary">{t('settings.loadingStatus')}</div>}

        {status && env && (<>
          {/* 环境信息 */}
          <div className="rounded-sm border border-subtle bg-sunken p-2 flex flex-col gap-1 text-xs">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-fg-tertiary shrink-0">flash_attn:</span>
              <code className="font-mono text-fg-primary">
                {status.installed ? `v${status.version ?? '?'}` : t('settings.notInstalledParen')}
              </code>
              {status.installed && <StatusLabel bg="bg-ok-soft" fg="text-ok" text={t('settings.installed')} />}
            </div>
            <div className="flex gap-4 flex-wrap">
              <span className="text-fg-tertiary">Python: <code className="text-fg-secondary font-mono">{env.python_tag}</code></span>
              <span className="text-fg-tertiary">CUDA: <code className="text-fg-secondary font-mono">{env.cuda_tag ?? t('settings.notDetected')}</code></span>
              <span className="text-fg-tertiary">PyTorch: <code className="text-fg-secondary font-mono">{env.torch_tag ?? t('settings.notDetected')}</code></span>
              <span className="text-fg-tertiary">{t('settings.platform')}: <code className="text-fg-secondary font-mono">{env.platform ?? t('settings.unsupported')}</code></span>
            </div>
          </div>

          {/* CPU 版 torch：根本装不了 flash_attn，优先显示这条 */}
          {isCpuTorch && (
            <div className="rounded-sm border border-warn bg-warn-soft px-2 py-1.5 text-warn text-xs">
              {t('settings.flashAttnNeedsCudaTorch')}
            </div>
          )}

          {/* GitHub API 失败 */}
          {!isCpuTorch && fetchError && (
            <div className="rounded-sm border border-err bg-err-soft px-2 py-1.5 text-err text-xs">
              {t('settings.githubApiFailed')}
              <code className="block mt-0.5 break-all">{fetchError}</code>
            </div>
          )}

          {/* 没匹配 wheel */}
          {!isCpuTorch && !canAutoInstall && !fetchError && env.platform && env.torch_tag && (
            <div className="rounded-sm border border-warn bg-warn-soft px-2 py-1.5 text-warn text-xs">
              {t('settings.noWheelForPython', { python: env.python_tag })}
            </div>
          )}

          {/* 操作按钮 */}
          <div className="flex gap-1.5 items-center flex-wrap">
            <button
              onClick={() => void install(null)}
              disabled={busy || !canAutoInstall}
              className="btn btn-primary btn-sm"
              title={canAutoInstall
                ? t('settings.autoSelect', { tag: bestCandidate?.name ?? '' })
                : t('settings.noWheelManual')}
            >
              {busy ? t('settings.installing') : status.installed ? t('settings.reinstallAutoMatch') : t('settings.autoMatchInstall')}
            </button>
            <button onClick={() => void refresh()} disabled={busy}
              className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm">↻</button>
            <button type="button" onClick={() => setCandidatesOpen(!candidatesOpen)}
              className="btn btn-ghost btn-sm text-xs text-fg-tertiary ml-auto">
              {candidatesOpen ? '▾' : '▸'} {t('settings.candidateWheels', { n: usable.length })}
            </button>
          </div>

          {/* 候选列表 + 手动 URL */}
          {candidatesOpen && (
            <div className="flex flex-col gap-2 pt-2 border-t border-subtle">
              {candidates.length === 0 ? (
                <p className="text-xs text-fg-tertiary m-0">{t('settings.wheelQueryFailed')}</p>
              ) : (
                <ul className="list-none m-0 p-0 flex flex-col gap-1">
                  {candidates.map((c) => (
                    <li key={c.url} className={`flex items-start gap-2 text-xs px-2 py-1.5 rounded-sm border ${
                      c.usable ? 'border-subtle bg-sunken' : 'border-transparent bg-transparent opacity-50'
                    }`}>
                      <div className="flex flex-col gap-0.5 flex-1 min-w-0">
                        <code className="font-mono text-fg-primary text-[11px] break-all">{c.name}</code>
                        {c.notes.map((n, i) => (
                          <span key={i} className="text-warn text-[10px]">{n}</span>
                        ))}
                      </div>
                      <button
                        onClick={() => void install(c.url)}
                        disabled={busy}
                        className={c.usable ? 'btn btn-primary btn-sm shrink-0' : 'btn btn-secondary btn-sm shrink-0'}
                        title={c.usable ? t('settings.installWheel') : t('settings.wheelAbiIncompatible')}
                      >
                        {c.usable ? t('settings.installAction') : t('settings.forceInstall')}
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              <div className="flex flex-col gap-1 pt-1 border-t border-subtle">
                <p className="text-xs text-fg-tertiary m-0">{t('settings.manualUrl')}</p>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    value={manualUrl}
                    onChange={(e) => setManualUrl(e.target.value)}
                    placeholder="https://github.com/.../flash_attn-...whl"
                    className={`${textInputClass} flex-1`}
                  />
                  <button
                    onClick={() => { if (manualUrl.trim()) void install(manualUrl.trim()) }}
                    disabled={busy || !manualUrl.trim()}
                    className="btn btn-secondary btn-sm shrink-0"
                  >{t('settings.install')}</button>
                </div>
              </div>
            </div>
          )}
        </>)}
      </div>
    </details>
  )
}

// ── xformers Section（训练 tab）─────────────────────────────────────────────
//
// 简化版 attention 加速（替代 flash_attn 的另一选项）。xformers 走 PyPI 直装，
// 不需要 flash_attn 那种 GitHub 候选 wheel 列表。失败时给 stderr 让用户排错。

export function XformersSection() {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [status, setStatus] = useState<XformersStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const s = await api.getXformersStatus()
      setStatus(s)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const install = async () => {
    if (
      !(await dialog.confirm(
        t('settings.confirmInstallXformers'),
        { tone: 'warn', okText: t('settings.startInstall') },
      ))
    ) return
    setBusy(true)
    try {
      const r = await api.installXformers()
      toast(t('settings.xformersInstalled', { version: r.version ?? '?' }), 'success')
      await refresh()
    } catch (e) {
      toast(t('settings.installFailed', { error: String(e) }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const statusLabel = error
    ? t('settings.loadFailedShort')
    : !status
      ? t('settings.loadingStatus')
      : status.installed
        ? t('settings.installedVersion', { version: status.version ?? '?' })
        : t('settings.notInstalledShort')
  const statusOk = status?.installed && !error
  const hasIssue = !!error

  return (
    <details id="xformers" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">xformers</h2>
        <span className="text-xs text-fg-tertiary">{t('settings.xformersSubtitle')}</span>
        <InfoButton>
          <p><Trans i18nKey="settings.xformersHelp1" components={{ strong: <strong />, code: <code /> }} /></p>
          <p>{t('settings.xformersHelp2')}</p>
          <p>{t('settings.xformersHelp3')}</p>
        </InfoButton>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : 'text-warn'}`}>{statusLabel}</span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !status && <div className="text-xs text-fg-tertiary">{t('settings.loadingStatus')}</div>}

        {status && (<>
          <div className="rounded-sm border border-subtle bg-sunken p-2 flex items-center gap-2 text-xs">
            <span className="text-fg-tertiary shrink-0">xformers:</span>
            <code className="font-mono text-fg-primary">
              {status.installed ? `v${status.version ?? '?'}` : t('settings.notInstalledParen')}
            </code>
            {status.installed && <StatusLabel bg="bg-ok-soft" fg="text-ok" text={t('settings.installed')} />}
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => void install()}
              disabled={busy}
              className="btn btn-primary btn-sm"
            >
              {busy
                ? t('settings.installing')
                : status.installed
                  ? t('settings.reinstallAutoMatchPlain')
                  : t('settings.installAutoMatchPlain')}
            </button>
            <button
              onClick={() => void refresh()}
              disabled={busy}
              className="btn btn-ghost btn-sm"
              title={t('settings.refreshStatus')}
            >↻</button>
          </div>
        </>)}
      </div>
    </details>
  )
}

// ── 中间步预览（节流） ────────────────────────────────────────────────────
//
// TAEFlux 模型 server 启动时后台下载（lifespan startup）；UI 只暴露用户必须
// 控制的「节流 N」一个输入，其他状态/下载/帮助文字全删（用户决策）。

export function IdleTimeoutSection({
  draft, update,
}: {
  draft: Secrets
  update: <S extends Section, K extends keyof Secrets[S]>(
    section: S, key: K, value: Secrets[S][K],
  ) => void
}) {
  const { t } = useTranslation()
  const minutes = draft.generate.idle_timeout_minutes
  return (
    <SettingsSection id="idle-timeout" title={t('settings.idleTimeout.title')}>
      <SettingsField
        label={t('settings.idleTimeout.label')}
        desc={t('settings.idleTimeout.desc')}
        helpTooltip={<p>{t('settings.idleTimeout.help')}</p>}
      >
        <div className="flex items-center gap-2">
          <SettingsInput
            type="number"
            min={0}
            max={240}
            value={minutes}
            onChange={(v) => update('generate', 'idle_timeout_minutes', Math.max(0, Number(v) || 0))}
            className={`${textInputClass} max-w-32`}
          />
          <span className="text-xs text-fg-tertiary">
            {minutes === 0
              ? t('settings.idleTimeout.offHint')
              : t('settings.idleTimeout.minutesSuffix')}
          </span>
        </div>
      </SettingsField>
    </SettingsSection>
  )
}


export function VaePrecisionSection({
  draft, update,
}: {
  draft: Secrets
  update: <S extends Section, K extends keyof Secrets[S]>(
    section: S, key: K, value: Secrets[S][K],
  ) => void
}) {
  const { t } = useTranslation()
  return (
    <SettingsSection id="vae-precision" title={t('settings.vaePrecision.title')}>
      <SettingsField
        label={t('settings.vaePrecision.label')}
        desc={t('settings.vaePrecision.desc')}
        helpTooltip={<p>{t('settings.vaePrecision.help')}</p>}
      >
        <select
          value={draft.generate.vae_precision ?? 'bf16'}
          onChange={(e) => update('generate', 'vae_precision', e.target.value as 'bf16' | 'fp32')}
          className={`${textInputClass} max-w-32`}
        >
          <option value="bf16">bf16</option>
          <option value="fp32">fp32</option>
        </select>
      </SettingsField>
    </SettingsSection>
  )
}


export function TaeFluxSection({
  draft, update,
}: {
  draft: Secrets
  update: <S extends Section, K extends keyof Secrets[S]>(
    section: S, key: K, value: Secrets[S][K],
  ) => void
}) {
  const { t } = useTranslation()
  const n = draft.generate.preview_every_n_steps
  return (
    <SettingsSection id="preview" title={t('settings.intermediatePreview')}>
      <SettingsField
        label={t('settings.previewThrottle')}
        desc={t('settings.previewThrottleDesc')}
        helpTooltip={
          <p>{t('settings.taeFluxHelp')}</p>
        }
      >
        <SettingsInput
          type="number"
          min={0}
          max={50}
          value={n}
          onChange={(v) => update('generate', 'preview_every_n_steps', Number(v) || 0)}
          className={`${textInputClass} max-w-32`}
        />
      </SettingsField>
    </SettingsSection>
  )
}


export function SaveTestImagesSection({
  draft, update,
}: {
  draft: Secrets
  update: <S extends Section, K extends keyof Secrets[S]>(
    section: S, key: K, value: Secrets[S][K],
  ) => void
}) {
  const { t } = useTranslation()
  return (
    <SettingsSection id="save-test-images" title={t('settings.saveTestImages.title')}>
      <SettingsField
        label={t('settings.saveTestImages.label')}
        helpTooltip={t('settings.saveTestImages.tooltip')}
      >
        <Bool
          value={draft.generate.save_test_images}
          onChange={(v) => update('generate', 'save_test_images', v)}
        />
      </SettingsField>
    </SettingsSection>
  )
}


// ── Display Section ────────────────────────────────────────────────────────

export function DisplaySection() {
  const { t } = useTranslation()
  const [theme, setTheme] = useState<Theme>(() => getStoredTheme())
  const [density, setDensity] = useState<Density>(() => getStoredDensity())
  const [lang, setLang] = useState<string>(() => getStoredLangWithDefault())

  const handleThemeChange = (t: Theme) => {
    setTheme(t)
    setStoredTheme(t)
    applyTheme(t)
  }

  const handleDensityChange = (d: Density) => {
    setDensity(d)
    setStoredDensity(d)
    applyDensity(d)
  }

  const handleLangChange = (newLang: string) => {
    setLang(newLang)
    setStoredLang(newLang)
    void i18n.changeLanguage(newLang)
  }

  const densityLabel = (d: Density): string => {
    if (d === 'tight') return t('settings.densityTight')
    if (d === 'loose') return t('settings.densityLoose')
    return t('settings.densityDefault')
  }

  return (
    <SettingsSection id="display" title={t('settings.display')}>
      <SettingsField label={t('settings.language')}>
        <div className="flex gap-1">
          {[
            { id: 'zh', label: t('settings.languageZh') },
            { id: 'en', label: t('settings.languageEn') },
          ].map((l) => (
            <button
              key={l.id}
              onClick={() => handleLangChange(l.id)}
              className={`btn btn-sm ${lang === l.id ? 'btn-primary' : 'btn-secondary'}`}
            >
              {l.label}
            </button>
          ))}
        </div>
      </SettingsField>

      <SettingsField label={t('settings.theme')}>
        <div className="flex gap-1">
          {(['light', 'dark'] as Theme[]).map((themeOption) => (
            <button
              key={themeOption}
              onClick={() => handleThemeChange(themeOption)}
              className={`btn btn-sm ${theme === themeOption ? 'btn-primary' : 'btn-secondary'}`}
            >
              {themeOption === 'light' ? t('settings.themeLight') : t('settings.themeDark')}
            </button>
          ))}
        </div>
      </SettingsField>

      <SettingsField
        label={t('settings.uiScale')}
        helpTooltip={
          <>
            <p><strong>{t('settings.densityTight')}</strong>：{t('settings.densityTightHelp')}</p>
            <p><strong>{t('settings.densityDefault')}</strong>：{t('settings.densityDefaultHelp')}</p>
            <p><strong>{t('settings.densityLoose')}</strong>：{t('settings.densityLooseHelp')}</p>
          </>
        }
      >
        <div className="flex gap-1">
          {(['tight', 'default', 'loose'] as Density[]).map((d) => (
            <button
              key={d}
              onClick={() => handleDensityChange(d)}
              className={`btn btn-sm ${density === d ? 'btn-primary' : 'btn-secondary'}`}
            >
              {densityLabel(d)}
            </button>
          ))}
        </div>
      </SettingsField>
    </SettingsSection>
  )
}
