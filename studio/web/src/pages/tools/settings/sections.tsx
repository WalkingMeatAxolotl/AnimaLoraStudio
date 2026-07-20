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
import { useTagAutocompleteEnabled } from '../../../tagDict/autocompleteToggle'
import { useShowTagTranslation } from '../../../tagDict/showToggle'
import { useTagDict, reloadDict } from '../../../tagDict/store'
import { useToast } from '../../../components/Toast'
import { useSettingsData } from '../../../lib/SettingsData'
import { applyDensity, applyTheme, getStoredDensity, getStoredTheme, setStoredDensity, setStoredTheme, type Density, type Theme } from '../../../lib/theme'
import i18n, { getStoredLangWithDefault, setStoredLang } from '../../../i18n'
import { MODEL_DESCRIPTION_KEYS, textInputClass, translatedCatalogText, UPSCALER_DESCRIPTION_KEYS, type Section } from './constants'
import { Bool, PillRadioGroup, SettingsField, SettingsInput, SettingsSection } from './fields'
import { DownloadButton, ModelGroupCard, ModelSourceCard, ModelStatusBadge, SourceSelect, StatusLabel } from './modelCards'

// ── 训练参数 Section ─────────────────────────────────────────────────
//
// 训练相关的全局开关。当前只有「自动配置模型路径」(auto_sync_paths)：独立 PUT，
// 不进全局 dirty 流程。原先夹在「训练模型」section 里，挪到这里跟模型下载分开。
export function TrainingParamsSection() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { runSave } = useSettingsData()
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
      await runSave(() => api.updateSecrets({ models: { auto_sync_paths: next } }))
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

/** 每个模型族在 Settings 页的区块配置（多模型 P4-5）。
 *  加第 3 个族 = 此表加一行 + i18n 标题键；渲染完全遍历化。
 *  encoderIds 是该族的目录型资产（文本编码器 / tokenizer 等 catalog section id）。 */
const FAMILY_MODEL_SECTIONS = [
  {
    family: 'anima' as const,
    sectionId: 'models',
    titleKey: 'settings.animaModels',
    mainKey: 'anima_main' as const,
    fallbackSelected: '1.0',
    encoderIds: ['qwen3', 't5_tokenizer'] as const,
    // 下载日志分组：该族相关的下载 key 前缀（正向声明，不再反向 startsWith 排除）
    downloadKeyPrefixes: ['anima_main', 'anima_vae', 'qwen3', 't5_tokenizer'],
  },
  {
    family: 'krea2' as const,
    sectionId: 'krea2-models',
    titleKey: 'settings.krea2Models',
    mainKey: 'krea2_main' as const,
    fallbackSelected: 'raw',
    // krea2 的 TE 卡是 variant 合并卡（bf16/fp8 radio），单独渲染不走
    // encoderIds 的文件列表卡机制
    encoderIds: [] as const,
    downloadKeyPrefixes: ['krea2_main', 'krea2_text_encoder'],
  },
]

type FamilyId = (typeof FAMILY_MODEL_SECTIONS)[number]['family']

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
  const { runSave, deleteAsset } = useSettingsData()
  const [selected, setSelected] = useState<Record<FamilyId, string>>({
    anima: '1.0', krea2: 'raw',
  })

  // 一次性拉 secrets 取各族 selected（独立 PUT，不进全局 dirty 流程）。模型
  // 根目录已挪到「系统 → 存储位置」、自动配置模型路径已挪到「训练参数」，不在此渲染。
  useEffect(() => {
    void api.getSecrets().then((sec) => {
      setSelected((prev) => ({
        ...prev,
        anima: sec.models?.selected?.anima ?? sec.models?.selected_anima ?? '1.0',
        krea2: sec.models?.selected?.krea2 ?? 'raw',
      }))
    }).catch(() => { /* 显示用，拉不到不阻塞 */ })
  }, [])

  // 统一写新结构 selected.{family}（update() 已剥 merge base 的 read-compat
  // computed 键，不会被过期 legacy 值覆盖——P4-5 前 anima 必须写 legacy 键）
  const pick = async (family: FamilyId, variant: string) => {
    if (variant === selected[family]) return
    setSelected((prev) => ({ ...prev, [family]: variant }))
    try {
      await runSave(() => api.updateSecrets({ models: { selected: { [family]: variant } } }))
      toast(t('settings.mainModelSelected', { name: variant }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
      void reloadCatalog()
    }
  }

  // krea2 TE variant 选择（bf16/fp8）：与主模型 pick 同款直写 secrets
  const [teSelected, setTeSelected] = useState<'bf16' | 'fp8'>('bf16')
  useEffect(() => {
    const sel = (catalog?.krea2_text_encoder as { selected?: string } | undefined)?.selected
    if (sel === 'bf16' || sel === 'fp8') setTeSelected(sel)
  }, [catalog])
  const pickTe = async (variant: 'bf16' | 'fp8') => {
    if (variant === teSelected) return
    setTeSelected(variant)
    try {
      await runSave(() => api.updateSecrets({ models: { selected_te: { krea2: variant } } }))
      toast(t('settings.teVariantSelected', { name: variant }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
      void reloadCatalog()
    }
  }

  const error = catalogError

  const renderSharedVae = () => {
    if (!catalog) return null
    return (
      <ModelGroupCard title={catalog.anima_vae.name}>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-fg-tertiary">{translatedCatalogText(MODEL_DESCRIPTION_KEYS, 'anima_vae', catalog.anima_vae.description, t)} · <code>{catalog.anima_vae.repo}</code></span>
          <span style={{ flex: 1 }} />
          <ModelStatusBadge exists={catalog.anima_vae.exists} size={catalog.anima_vae.size} status={catalog.downloads.anima_vae?.status} />
          <DownloadButton exists={catalog.anima_vae.exists} status={catalog.downloads.anima_vae?.status} busy={busy.has('anima_vae')} onClick={() => void start('anima_vae')} onDelete={() => void deleteAsset('anima_vae', undefined, catalog.anima_vae.name)} />
        </div>
      </ModelGroupCard>
    )
  }

  const renderDownloadLogs = (prefixes: readonly string[]) => {
    if (!catalog) return null
    // 正向按前缀声明分组（此前用 startsWith('krea2_') 反判——第三族会全落
    // 进 anima 桶）
    const downloads = Object.values(catalog.downloads).filter((download) =>
      prefixes.some((prefix) => download.key.startsWith(prefix)))
    const visible = downloads.filter((download) => download.status === 'running' || download.status === 'failed')
    if (visible.length === 0) return null
    return (
      <details className="text-xs">
        <summary className="cursor-pointer text-fg-tertiary">
          {t('settings.downloadLogs', { n: visible.length })}
        </summary>
        <div className="mt-1 flex flex-col gap-2">
          {visible.map((download) => (
            <div key={download.key} className="rounded-sm border border-subtle bg-sunken p-2">
              <div className="flex items-center gap-2 mb-1">
                <code className="font-mono text-fg-secondary">{download.key}</code>
                <ModelStatusBadge exists={download.status === 'done'} size={0} status={download.status} />
                {download.message && <span className="text-err overflow-hidden text-ellipsis whitespace-nowrap">{download.message}</span>}
              </div>
              <pre className="text-xs font-mono text-fg-tertiary max-h-32 overflow-auto whitespace-pre-wrap m-0">
                {download.log_tail.join('\n') || t('settings.emptyLog')}
              </pre>
            </div>
          ))}
        </div>
      </details>
    )
  }

  return (
    <>
    {FAMILY_MODEL_SECTIONS.map((section) => {
      const main = catalog?.[section.mainKey]
      return (
        <SettingsSection key={section.sectionId} id={section.sectionId} title={t(section.titleKey)}>
          <SourceSelect
            opt={catalog?.download_source_options?.training}
            onChange={(s) => void setSource('training', s)}
          />

          {error && <div className="text-err text-xs font-mono">{error}</div>}
          {!catalog || !main ? (
            <p className="text-fg-tertiary text-xs">{t('settings.loadingModelCatalog')}</p>
          ) : (
            <div className="flex flex-col gap-2">
              {/* 族主模型：官方 variants + 用户候选（下载型第三方微调 / 本地文件） */}
              <ModelSourceCard
                domain={section.family}
                title={main.name}
                helpTooltip={
                  <>
                    <p><Trans i18nKey="settings.repoHelp" values={{ desc: translatedCatalogText(MODEL_DESCRIPTION_KEYS, section.mainKey, main.description, t), repo: main.repo }} components={{ code: <code /> }} /></p>
                    {section.family === 'anima' && (
                      <p><Trans i18nKey="settings.defaultTransformerHelp" components={{ strong: <strong /> }} /></p>
                    )}
                    {main.license_url && (
                      <p>
                        <a href={main.license_url} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                          {t('settings.krea2LicenseNotice', { license: main.license })}
                        </a>
                      </p>
                    )}
                  </>
                }
                catalog={catalog}
                currentValue={selected[section.family]}
                onSelect={(v) => void pick(section.family, v)}
                addDownload={{
                  filenameField: true,
                  repoPlaceholder: 'author/finetune-repo',
                  filenamePlaceholder: 'model.safetensors',
                }}
                addLocal={{}}
                selectRequiresExists
                renderRowMeta={(row) => (
                  row.extra.purpose ? (
                    <span className="text-2xs px-1 py-0.5 rounded-sm bg-overlay text-fg-tertiary shrink-0">
                      {t(`baseModel.purpose.${row.extra.purpose}`)}
                    </span>
                  ) : null
                )}
                t={t}
              />

              {/* 共享 Qwen-Image VAE（两族同一份文件，族无关资产） */}
              {renderSharedVae()}

              {/* krea2 TE variant 合并卡：bf16/fp8 两行 radio + 各自下载 */}
              {section.family === 'krea2' && catalog.krea2_text_encoder && catalog.krea2_text_encoder_fp8 && (
                <ModelGroupCard title={t('settings.krea2TeCardTitle')}>
                  <ul className="flex flex-col gap-1.5 text-xs">
                    {([
                      ['bf16', catalog.krea2_text_encoder],
                      ['fp8', catalog.krea2_text_encoder_fp8],
                    ] as const).map(([variant, m]) => {
                      const allExist = m.files.length > 0 && m.files.every((f) => f.exists)
                      const totalSize = m.files.reduce((s, f) => s + f.size, 0)
                      const dl = catalog.downloads[m.id]
                      return (
                        <li key={variant} className="flex items-center gap-2">
                          <input
                            type="radio"
                            name="krea2-te-variant"
                            checked={teSelected === variant}
                            disabled={!allExist}
                            onChange={() => void pickTe(variant)}
                            style={{ accentColor: 'var(--accent)' }}
                            title={allExist ? t('settings.selectTeVariant') : t('settings.teVariantNotDownloaded')}
                          />
                          <code className="font-mono text-fg-primary w-16 shrink-0">{variant}</code>
                          <span className="text-fg-tertiary truncate">
                            {translatedCatalogText(MODEL_DESCRIPTION_KEYS, m.id, m.description, t)}
                          </span>
                          <span style={{ flex: 1 }} />
                          <ModelStatusBadge exists={allExist} size={totalSize} status={dl?.status} fileCount={m.files.length} existsCount={m.files.filter((f) => f.exists).length} />
                          <DownloadButton exists={allExist} status={dl?.status} busy={busy.has(m.id)} onClick={() => void start(m.id)} onDelete={() => void deleteAsset(m.id, undefined, `Qwen3-VL ${variant}`)} />
                        </li>
                      )
                    })}
                  </ul>
                </ModelGroupCard>
              )}

              {/* 该族的目录型资产（文本编码器 / tokenizer；CLTagger 在「打标」tab） */}
              {section.encoderIds.map((id) => {
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
                      <DownloadButton exists={allExist} status={dl?.status} busy={busy.has(id)} onClick={() => void start(id)} onDelete={() => void deleteAsset(id, undefined, m.name)} />
                    </div>
                  </ModelGroupCard>
                )
              })}

              {renderDownloadLogs(section.downloadKeyPrefixes)}
            </div>
          )}
        </SettingsSection>
      )
    })}
    </>
  )
}

export function UpscalerSection({
  catalog, setSource, reloadCatalog, t,
}: {
  catalog: ModelsCatalog | null
  setSource: (type: string, source: string) => Promise<void>
  reloadCatalog: () => Promise<ModelsCatalog | null>
  t: TFunction
}) {
  const { toast } = useToast()
  const { runSave } = useSettingsData()

  const pickUpscaler = async (label: string) => {
    try {
      await runSave(() => api.selectUpscaler(label))
      toast(t('settings.defaultUpscaler', { name: label }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

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
          <ModelSourceCard
            domain="upscaler"
            title={t('settings.availableUpscalers')}
            helpTooltip={
              <>
                <p><Trans i18nKey="settings.upscalersHelpPath" values={{ path: catalog.upscalers?.target_dir }} components={{ code: <code /> }} /></p>
                <p>{t('settings.upscalersHelpDefault')}</p>
                <p><Trans i18nKey="settings.customUpscalerHelpTypes" components={{ code: <code /> }} /></p>
              </>
            }
            catalog={catalog}
            currentValue={catalog.upscalers?.current ?? ''}
            onSelect={(v) => void pickUpscaler(v)}
            addDownload={{
              filenameField: true,
              repoPlaceholder: 'Kim2091/UltraSharp',
              filenamePlaceholder: '4x-UltraSharp.pth',
            }}
            addLocal={{}}
            selectRequiresExists
            describeRow={(row) => {
              const base = row.kind === 'preset'
                ? translatedCatalogText(UPSCALER_DESCRIPTION_KEYS, row.value, row.description, t)
                : row.description
              const repo = row.extra.hf_repo
                ? ` · HF ${row.extra.hf_repo}` : ''
              return `${base ?? ''}${repo}`
            }}
            t={t}
          />

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
  const dialog = useDialog()
  const dict = useTagDict()
  const { runSave } = useSettingsData()
  const [show, setShow] = useShowTagTranslation()
  const [acEnabled, setAcEnabled] = useTagAutocompleteEnabled()
  const [busy, setBusy] = useState<null | 'reset' | 'upload'>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  // localStorage 即时设置也包一层 runSave：驱动右上角「已保存」指示，反馈跟
  // 其他 instant-apply 设置一致（fn 同步落盘，不会失败）。
  const applyLocal = (fn: () => void) => { void runSave(async () => { fn() }) }

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
    if (!(await dialog.confirm(t('settings.tagDictionary.confirmReset'), { tone: 'danger' }))) return
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
            {busy === 'reset' ? t('settings.tagDictionary.resetBusy') : t('settings.tagDictionary.resetButton')}
          </button>
        </div>
      </SettingsField>

      <SettingsField
        label={t('settings.tagDictionary.showToggleLabel')}
        desc={t('settings.tagDictionary.showToggleHint')}
      >
        <Bool value={show} onChange={(v) => applyLocal(() => setShow(v))} />
      </SettingsField>

      <SettingsField
        label={t('settings.tagDictionary.autocompleteToggleLabel')}
        desc={t('settings.tagDictionary.autocompleteToggleHint')}
      >
        <Bool value={acEnabled} onChange={(v) => applyLocal(() => setAcEnabled(v))} />
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
      <SettingsField
        label={t('settings.idleTimeout.taskTimeoutLabel')}
        desc={t('settings.idleTimeout.taskTimeoutDesc')}
        helpTooltip={<p>{t('settings.idleTimeout.taskTimeoutHelp')}</p>}
      >
        <div className="flex items-center gap-2">
          <SettingsInput
            type="number"
            min={0}
            max={240}
            value={draft.generate.task_timeout_minutes ?? 0}
            onChange={(v) => update('generate', 'task_timeout_minutes', Math.max(0, Number(v) || 0))}
            className={`${textInputClass} max-w-32`}
          />
          <span className="text-xs text-fg-tertiary">
            {(draft.generate.task_timeout_minutes ?? 0) === 0
              ? t('settings.idleTimeout.taskTimeoutOffHint')
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
      <SettingsField
        label={t('settings.loraMergePrecision.label')}
        desc={t('settings.loraMergePrecision.desc')}
        helpTooltip={<p>{t('settings.loraMergePrecision.help')}</p>}
      >
        <select
          value={draft.generate.lora_merge_precision ?? 'fp32'}
          onChange={(e) => update('generate', 'lora_merge_precision', e.target.value as 'fp32' | 'bf16')}
          className={`${textInputClass} max-w-32`}
        >
          <option value="fp32">fp32</option>
          <option value="bf16">bf16</option>
        </select>
      </SettingsField>
    </SettingsSection>
  )
}


export function VramPolicySection({
  draft, update,
}: {
  draft: Secrets
  update: <S extends Section, K extends keyof Secrets[S]>(
    section: S, key: K, value: Secrets[S][K],
  ) => void
}) {
  const { t } = useTranslation()
  return (
    <SettingsSection id="vram-policy" title={t('settings.vramPolicy.title')}>
      <SettingsField
        label={t('settings.vramPolicy.label')}
        desc={t('settings.vramPolicy.desc')}
        helpTooltip={<p>{t('settings.vramPolicy.help')}</p>}
      >
        <select
          value={draft.generate.vram_policy ?? 'auto'}
          onChange={(e) => update('generate', 'vram_policy', e.target.value as 'auto' | 'save_vram' | 'performance')}
          className={`${textInputClass} max-w-32`}
        >
          <option value="auto">{t('settings.vramPolicy.optAuto')}</option>
          <option value="save_vram">{t('settings.vramPolicy.optSaveVram')}</option>
          <option value="performance">{t('settings.vramPolicy.optPerformance')}</option>
        </select>
      </SettingsField>
      <SettingsField
        label={t('settings.vramPolicy.ramGuardLabel')}
        desc={t('settings.vramPolicy.ramGuardDesc')}
        helpTooltip={<p>{t('settings.vramPolicy.ramGuardHelp')}</p>}
      >
        <Bool
          value={draft.generate.ram_guard ?? true}
          onChange={(v) => update('generate', 'ram_guard', v)}
        />
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
        <PillRadioGroup
          options={[
            { id: 'zh', label: t('settings.languageZh') },
            { id: 'en', label: t('settings.languageEn') },
          ]}
          value={lang}
          onChange={handleLangChange}
        />
      </SettingsField>

      <SettingsField label={t('settings.theme')}>
        <PillRadioGroup
          options={(['light', 'dark'] as Theme[]).map((themeOption) => ({
            id: themeOption,
            label: themeOption === 'light' ? t('settings.themeLight') : t('settings.themeDark'),
          }))}
          value={theme}
          onChange={handleThemeChange}
        />
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
        <PillRadioGroup
          options={(['tight', 'default', 'loose'] as Density[]).map((d) => ({
            id: d, label: densityLabel(d),
          }))}
          value={density}
          onChange={handleDensityChange}
        />
      </SettingsField>
    </SettingsSection>
  )
}
