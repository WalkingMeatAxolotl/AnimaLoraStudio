import { useState } from 'react'
import type { TFunction } from 'i18next'
import { Trans, useTranslation } from 'react-i18next'
import {
  type CLTaggerVariantInfo,
  type ModelDownloadStatus,
  type ModelsCatalog,
} from '../../../api/client'
import { InfoButton } from '../../../components/InfoButton'
import { fmtBytes, MODEL_DESCRIPTION_KEYS, textInputClass, translatedCatalogText } from './constants'
import { SettingsField, SettingsInput } from './fields'

// ── HFEndpointSelect ────────────────────────────────────────────────────────
//
// HF 模型下载 endpoint 选择器：preset + 自定义 URL 输入。
// 0.8.2 hotfix：hf-mirror.com preset 暂时隐藏（服务端 redirect 改动后所有
// huggingface_hub 版本均失败，详见 docs/todo/hf-mirror-recheck.md）。endpoint
// 字段本身仍接受任意 URL，用户可通过「自定义 URL」粘贴 hf-mirror / sjtug /
// 腾讯镜像 / 自建反代。复活后把 preset 加回来即可。

export const HF_ENDPOINT_PRESETS: { value: string; label: string; hintKey: string }[] = [
  { value: '', label: 'huggingface.co', hintKey: 'settings.hfOfficialHint' },
  { value: '__custom__', label: 'Custom URL...', hintKey: 'settings.hfCustomHint' },
]

export function HFEndpointSelect({ value, onChange }: {
  value: string; onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  const isPreset = HF_ENDPOINT_PRESETS.some(p => p.value !== '__custom__' && p.value === value)
  const [mode, setMode] = useState<'preset' | 'custom'>(isPreset ? 'preset' : 'custom')
  const selectedPreset = isPreset
    ? value
    : (mode === 'custom' ? '__custom__' : '')

  return (
    <div className="flex flex-col gap-1.5">
      <select
        value={selectedPreset}
        onChange={(e) => {
          const v = e.target.value
          if (v === '__custom__') {
            setMode('custom')
            // 不清当前值，让用户在下方输入
          } else {
            setMode('preset')
            onChange(v)
          }
        }}
        className={textInputClass}
      >
        {HF_ENDPOINT_PRESETS.map(p => (
          <option key={p.value} value={p.value}>
            {p.label}{p.hintKey ? ` — ${t(p.hintKey)}` : ''}
          </option>
        ))}
      </select>
      {mode === 'custom' && (
        <SettingsInput
          type="text"
          value={value && !isPreset ? value : ''}
          placeholder="https://your-mirror.example.com"
          onChange={(v) => onChange(v.trim())}
          className={textInputClass}
        />
      )}
    </div>
  )
}

// ── DownloadSourceSelect ────────────────────────────────────────────────────

// 按类型的下载源选择器。available 多于 1 项才是真 dropdown；固定单源（如
// CLTagger/T5/TAEFlux）渲染成禁用框，纯指示「这个到底从哪下」。
export function SourceSelect({ opt, onChange }: {
  opt?: { current: string; available: string[] }
  onChange: (source: string) => void
}) {
  const { t } = useTranslation()
  if (!opt) return null
  const single = opt.available.length <= 1
  const labelOf = (s: string) =>
    s === 'modelscope' ? t('settings.downloadSourceModelscope') : t('settings.downloadSourceHuggingface')
  return (
    <SettingsField
      label={t('settings.downloadSource')}
      desc={single ? t('settings.singleSourceFixed') : undefined}
    >
      <select
        value={opt.current}
        disabled={single}
        onChange={(e) => onChange(e.target.value)}
        className={`${textInputClass} disabled:opacity-60`}
      >
        {opt.available.map((s) => <option key={s} value={s}>{labelOf(s)}</option>)}
      </select>
    </SettingsField>
  )
}

// ── ModelIdsEditor ──────────────────────────────────────────────────────────

export function ModelIdsEditor({ ids, currentId, onChange }: {
  ids: string[]; currentId: string; onChange: (next: string[]) => void
}) {
  const { t } = useTranslation()
  const [draft, setDraft] = useState('')
  const seen = new Set(ids)

  const add = () => {
    const v = draft.trim()
    if (!v) return
    if (seen.has(v)) { setDraft(''); return }
    onChange([...ids, v])
    setDraft('')
  }
  const remove = (m: string) => {
    if (m === currentId) return
    onChange(ids.filter((x) => x !== m))
  }

  return (
    <div className="flex flex-col gap-1.5">
      <ul className="flex flex-col gap-1 list-none m-0 p-0">
        {ids.map((m) => {
          const isCurrent = m === currentId
          return (
            <li key={m} className={`flex items-center gap-2 px-2 py-1 rounded-sm text-xs ${
              isCurrent ? 'border border-accent bg-accent-soft' : 'border border-subtle bg-sunken'
            }`}>
              <code className="font-mono text-fg-primary flex-1 min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">{m}</code>
              {isCurrent ? (
                <span className="text-xs text-accent">{t('settings.current')}</span>
              ) : (
                <button onClick={() => remove(m)} className="text-xs text-fg-tertiary hover:text-err bg-transparent border-none cursor-pointer transition-colors">×</button>
              )}
            </li>
          )
        })}
      </ul>
      <div className="flex gap-1.5">
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); add() } }}
          placeholder={t('settings.addHfModelId')}
          className={`${textInputClass} flex-1`}                            />
        <button onClick={add} disabled={!draft.trim() || seen.has(draft.trim())} className="btn btn-secondary btn-sm">{t('settings.add')}</button>
      </div>
    </div>
  )
}

// ── WD14 / CLTagger Model Cards（打标 tab 内嵌的模型管理器） ─────────────────

export function WD14ModelCard({
  catalog, busy, start,
  currentModelId, onSelectModelId,
  candidates, onCandidatesChange, t,
}: {
  catalog: ModelsCatalog | null
  busy: Set<string>
  start: (model_id: string, variant?: string) => Promise<void>
  currentModelId: string
  onSelectModelId: (id: string) => void
  candidates: string[]
  onCandidatesChange: (next: string[]) => void
  t: TFunction
}) {
  const [advOpen, setAdvOpen] = useState(false)
  const wd14 = catalog?.wd14
  const wd14Description = translatedCatalogText(MODEL_DESCRIPTION_KEYS, 'wd14', wd14?.description, t)
  if (!wd14) {
    return <p className="text-fg-tertiary text-xs">{t('settings.loadingModelCatalog')}</p>
  }
  return (
    <ModelGroupCard
      title={t('settings.wd14CandidateTitle', { name: wd14.name })}
      helpTooltip={
        <p><Trans i18nKey="settings.wd14CandidateHelp" values={{ desc: wd14Description }} components={{ code: <code /> }} /></p>
      }
    >
      <ul className="list-none m-0 p-0 flex flex-col gap-1">
        {wd14.variants.map((v) => {
          const key = `wd14:${v.model_id}`
          const dl = catalog.downloads[key]
          const isSel = v.model_id === currentModelId
          return (
            <li key={v.model_id} className={`flex items-center gap-2 text-xs px-1.5 py-1 rounded-sm ${
              isSel ? 'bg-accent-soft border border-accent' : 'bg-transparent border border-transparent'
            }`}>
              <input type="radio" name="wd14_variant" checked={isSel}
                onChange={() => onSelectModelId(v.model_id)}
                className="shrink-0"
                style={{ accentColor: 'var(--accent)' }}
                title={t('settings.selectWd14ModelId')}
              />
              <code className="font-mono text-fg-primary flex-1 min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">{v.model_id}</code>
              <ModelStatusBadge
                exists={v.exists} size={v.size} status={dl?.status}
                fileCount={v.files.length}
                existsCount={v.files.filter((f) => f.exists).length}
              />
              <DownloadButton
                exists={v.exists} status={dl?.status} busy={busy.has(key)}
                onClick={() => void start('wd14', v.model_id)}
              />
            </li>
          )
        })}
      </ul>
      <button type="button" onClick={() => setAdvOpen(!advOpen)}
        className="btn btn-ghost btn-sm text-xs text-fg-tertiary self-start">
        {advOpen ? '▾' : '▸'} {t('settings.candidateEditor')}
      </button>
      {advOpen && (
        <ModelIdsEditor
          ids={candidates} currentId={currentModelId}
          onChange={onCandidatesChange}
        />
      )}
    </ModelGroupCard>
  )
}

export function EvalMetricModelCard({
  catalog, busy, start, kind, dlId, titleKey, helpKey, modelId, onModelIdChange, t,
}: {
  catalog: ModelsCatalog | null
  busy: Set<string>
  start: (model_id: string, variant?: string) => Promise<void>
  kind: 'clip' | 'dino' | 'ccip'
  dlId: 'eval_clip' | 'eval_dino' | 'eval_ccip'
  titleKey: string
  helpKey: string
  modelId: string
  onModelIdChange: (id: string) => void
  t: TFunction
}) {
  const [advOpen, setAdvOpen] = useState(false)
  const em = catalog?.eval_metrics
  if (!em) {
    return <p className="text-fg-tertiary text-xs">{t('settings.loadingModelCatalog')}</p>
  }
  const variant = em.variants.find((x) => x.kind === kind)
  const key = `${dlId}:${modelId}`
  const dl = catalog.downloads[key]
  // catalog 的 exists 按已保存的 model_id 算；草稿改了未保存时按未下载显示。
  const exists = variant?.model_id === modelId ? !!variant?.exists : false
  return (
    <ModelGroupCard title={t(titleKey)} helpTooltip={<p>{t(helpKey)}</p>}>
      <div className="flex items-center gap-2 text-xs">
        <code className="font-mono text-fg-primary min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">{modelId}</code>
        {variant?.size_estimate ? (
          <span className="text-fg-tertiary shrink-0">~{fmtBytes(variant.size_estimate)}</span>
        ) : null}
        <span style={{ flex: 1 }} />
        <ModelStatusBadge exists={exists} size={variant?.size ?? 0} status={dl?.status} />
        <DownloadButton
          exists={exists} status={dl?.status} busy={busy.has(key)}
          onClick={() => void start(dlId, modelId)}
        />
      </div>
      <button type="button" onClick={() => setAdvOpen(!advOpen)}
        className="btn btn-ghost btn-sm text-xs text-fg-tertiary self-start">
        {advOpen ? '▾' : '▸'} {t('settings.customRepoAdvanced')}
      </button>
      {advOpen && (
        <SettingsField label={t('settings.fieldModelId')}>
          <SettingsInput
            type="text"
            value={modelId}
            onChange={onModelIdChange}
            className={textInputClass}
            placeholder={t('settings.addHfModelId')}
          />
        </SettingsField>
      )}
    </ModelGroupCard>
  )
}

export function CLTaggerModelCard({
  catalog, busy, start,
  currentModelPath, currentTagMappingPath, onSelectVariant,
  modelId, onModelIdChange, t,
}: {
  catalog: ModelsCatalog | null
  busy: Set<string>
  start: (model_id: string, variant?: string) => Promise<void>
  currentModelPath: string
  currentTagMappingPath: string
  onSelectVariant: (v: CLTaggerVariantInfo) => void
  modelId: string
  onModelIdChange: (id: string) => void
  t: TFunction
}) {
  const [advOpen, setAdvOpen] = useState(false)
  const cl = catalog?.cltagger
  const clDescription = translatedCatalogText(MODEL_DESCRIPTION_KEYS, 'cltagger', cl?.description, t)
  if (!cl) {
    return <p className="text-fg-tertiary text-xs">{t('settings.loadingModelCatalog')}</p>
  }
  return (
    <ModelGroupCard
      title={t('settings.clTaggerVersionTitle', { name: cl.name })}
      helpTooltip={
        <p><Trans i18nKey="settings.repoHelp" values={{ desc: clDescription, repo: cl.repo }} components={{ code: <code /> }} /></p>
      }
    >
      <ul className="list-none m-0 p-0 flex flex-col gap-1">
        {cl.variants.map((v) => {
          const key = `cltagger:${v.label}`
          const dl = catalog.downloads[key]
          const isSel =
            v.model_id === modelId &&
            v.model_path === currentModelPath &&
            v.tag_mapping_path === currentTagMappingPath
          return (
            <li key={v.label} className={`flex items-center gap-2 text-xs px-1.5 py-1 rounded-sm ${
              isSel ? 'bg-accent-soft border border-accent' : 'bg-transparent border border-transparent'
            }`}>
              <input type="radio" name="cltagger_variant" checked={isSel}
                onChange={() => onSelectVariant(v)}
                className="shrink-0"
                style={{ accentColor: 'var(--accent)' }}
                title={t('settings.selectClTaggerVersion')}
              />
              <div className="flex flex-col flex-1 min-w-0">
                <code className="font-mono text-fg-primary overflow-hidden text-ellipsis whitespace-nowrap">{v.label}</code>
                {v.version_dir && (
                  <span className="text-[10px] text-fg-tertiary overflow-hidden text-ellipsis whitespace-nowrap" title={v.version_dir}>
                    {t('settings.clTaggerFilesAt')}: <code>{v.version_dir}</code>
                  </span>
                )}
              </div>
              <ModelStatusBadge
                exists={v.exists} size={v.size} status={dl?.status}
                fileCount={v.files.length}
                existsCount={v.files.filter((f) => f.exists).length}
              />
              <DownloadButton
                exists={v.exists} status={dl?.status} busy={busy.has(key)}
                onClick={() => void start('cltagger', v.label)}
              />
            </li>
          )
        })}
      </ul>
      <button type="button" onClick={() => setAdvOpen(!advOpen)}
        className="btn btn-ghost btn-sm text-xs text-fg-tertiary self-start">
        {advOpen ? '▾' : '▸'} {t('settings.customRepoAdvanced')}
      </button>
      {advOpen && (
        <SettingsField label={t('settings.fieldModelId')}>
          <SettingsInput
            type="text"
            value={modelId}
            onChange={onModelIdChange}
            className={textInputClass}
            placeholder="cella110n/cl_tagger"
          />
        </SettingsField>
      )}
    </ModelGroupCard>
  )
}

export function ModelGroupCard({
  title, helpTooltip, children,
}: {
  title: string
  helpTooltip?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="rounded-sm border border-subtle bg-sunken p-2.5">
      <h4 className="text-xs font-semibold text-fg-primary mb-1.5 flex items-center gap-2">
        <span>{title}</span>
        {helpTooltip && <InfoButton>{helpTooltip}</InfoButton>}
      </h4>
      {children}
    </div>
  )
}

export function ModelStatusBadge({ exists, size, status, fileCount, existsCount }: {
  exists: boolean; size: number; status?: ModelDownloadStatus['status']; fileCount?: number; existsCount?: number
}) {
  const { t } = useTranslation()
  if (status === 'running') {
    return <StatusLabel bg="bg-warn-soft" fg="text-warn" text={t('settings.downloadInProgress')} pulse />
  }
  if (status === 'failed') {
    return <StatusLabel bg="bg-err-soft" fg="text-err" text={t('status.failed')} />
  }
  if (exists) {
    return <StatusLabel bg="bg-ok-soft" fg="text-ok" text={`✓ ${fmtBytes(size)}${fileCount !== undefined ? ` (${existsCount}/${fileCount})` : ''}`} />
  }
  if (fileCount !== undefined && existsCount! > 0) {
    return <StatusLabel bg="bg-warn-soft" fg="text-warn" text={t('settings.partialFiles', { exists: existsCount, total: fileCount })} />
  }
  return <StatusLabel bg="bg-overlay" fg="text-fg-tertiary" text={t('settings.notDownloaded')} />
}

export function StatusLabel({ bg, fg, text, pulse }: { bg: string; fg: string; text: string; pulse?: boolean }) {
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded-sm font-mono ${bg} ${fg}`}
      style={pulse ? { animation: 'pulse 1.5s infinite' } : undefined}
    >{text}</span>
  )
}

export function DownloadButton({ exists, status, busy, onClick }: {
  exists: boolean; status?: ModelDownloadStatus['status']; busy: boolean; onClick: () => void
}) {
  const { t } = useTranslation()
  const running = status === 'running' || busy
  if (running) {
    return <button disabled className="btn btn-secondary btn-sm min-w-[5rem] justify-center" style={{ opacity: 0.5 }}>...</button>
  }
  return (
    <button onClick={onClick} className="btn btn-secondary btn-sm min-w-[5rem] justify-center"
      title={exists ? t('settings.redownloadTitle') : t('common.download')}>
      {exists ? t('settings.redownload') : t('settings.downloadAction')}
    </button>
  )
}
