import { useState } from 'react'
import type { TFunction } from 'i18next'
import { useTranslation } from 'react-i18next'
import {
  api,
  type ModelDownloadStatus,
  type ModelsCatalog,
  type ModelSourceCandidate,
  type ModelSourceRow,
} from '../../../api/client'
import { InfoButton } from '../../../components/InfoButton'
import PathPicker from '../../../components/PathPicker'
import { useToast } from '../../../components/Toast'
import { useSettingsData } from '../../../lib/SettingsData'
import { fmtBytes, textInputClass } from './constants'
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

// ── ModelSourceCard（统一候选卡：候选列表 + 下载/本地文件双添加入口） ────────
//
// docs/design/model-source-unification.md：所有「这个功能用哪个模型」的卡片
// 共用本组件。行数据来自 catalog.model_sources[domain]（后端拼好能力位）；
// 当前选中受控于调用方（Settings draft / Tagging form），不读 is_current。
// 动作矩阵（D2）：内置 preset 只有下载/删除；download 候选加 × 移除（不动
// 磁盘）；local 候选只有 × 移除，永不删用户文件。

export function ModelSourceCard({
  domain, title, helpTooltip, catalog, currentValue, onSelect,
  addDownload, addLocal, selectRequiresExists = false, renderRowMeta, describeRow, t,
}: {
  domain: string
  title: string
  helpTooltip?: React.ReactNode
  catalog: ModelsCatalog | null
  currentValue: string
  onSelect: (value: string, row: ModelSourceRow) => void
  /** 下载型添加表单；undefined = 该 domain 不支持下载型添加。 */
  addDownload?: { filenameField?: boolean; repoPlaceholder?: string; filenamePlaceholder?: string }
  /** 本地文件添加；undefined = 不支持。secondFileKey：双文件资产（如
   *  cltagger 的 tag_mapping），第一次选主文件后再弹一次 PathPicker，第二个
   *  路径写进候选 extra[secondFileKey]。 */
  addLocal?: { dirOnly?: boolean; secondFileKey?: string }
  /** true：未下载的候选禁选（主模型/放大器语义）；false：可选中，运行时懒下载（wd14/eval 语义）。 */
  selectRequiresExists?: boolean
  /** 行副内容逃生舱（label 旁 chip，如主模型 purpose 徽标）。 */
  renderRowMeta?: (row: ModelSourceRow) => React.ReactNode
  /** 行描述文本覆盖（如放大器 preset 描述的 i18n 翻译）；默认 row.description。 */
  describeRow?: (row: ModelSourceRow) => string
  t: TFunction
}) {
  const { toast } = useToast()
  const { deleteAsset, downloadBusy, startDownload, reloadCatalog } = useSettingsData()
  const [adding, setAdding] = useState<null | 'download' | 'local' | 'local2'>(null)
  const [repoDraft, setRepoDraft] = useState('')
  const [fileDraft, setFileDraft] = useState('')
  const [firstLocalPath, setFirstLocalPath] = useState('')
  const [submitBusy, setSubmitBusy] = useState(false)

  const rows = catalog?.model_sources?.[domain]
  if (!catalog || !rows) {
    return <p className="text-fg-tertiary text-xs">{t('settings.loadingModelCatalog')}</p>
  }

  const submitCandidate = async (cand: ModelSourceCandidate) => {
    setSubmitBusy(true)
    try {
      await api.addModelSource(domain, cand)
      toast(t('settings.candidateAdded', {
        name: cand.kind === 'local' ? cand.path : cand.repo,
      }), 'success')
      setAdding(null)
      setRepoDraft('')
      setFileDraft('')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setSubmitBusy(false)
    }
  }

  const removeCandidate = async (row: ModelSourceRow) => {
    const cand: ModelSourceCandidate = row.candidate ?? (row.kind === 'local'
      ? { kind: 'local', path: row.value }
      : { kind: 'download', repo: row.value, extra: row.extra })
    try {
      await api.removeModelSource(domain, cand)
      toast(t('settings.candidateRemoved'), 'success')
      // 移除的是当前选中：服务端已把 secrets 回退默认；本地受控值同步到
      // 首个内置 preset，避免 draft 悬空指向已移除的候选。
      if (row.value === currentValue) {
        const fallback = rows.find((r) => r.kind === 'preset')
        if (fallback) onSelect(fallback.value, fallback)
      }
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const submitDownloadForm = () => {
    const repo = repoDraft.trim()
    if (!repo) return
    const cand: ModelSourceCandidate = { kind: 'download', repo }
    if (addDownload?.filenameField) {
      const fname = fileDraft.trim()
      if (!fname) return
      cand.filename = fname
    }
    void submitCandidate(cand)
  }

  return (
    <ModelGroupCard title={title} helpTooltip={helpTooltip}>
      <ul className="list-none m-0 p-0 flex flex-col gap-1">
        {rows.map((row) => {
          const dl = row.status_key ? catalog.downloads[row.status_key] : undefined
          const isSel = row.value === currentValue
          const canSelect = row.kind === 'local'
            ? row.exists
            : (dl?.status !== 'running' && (!selectRequiresExists || row.exists))
          return (
            <li key={`${row.kind}:${row.value}`} className={`flex items-center gap-2 text-xs px-1.5 py-1 rounded-sm ${
              isSel ? 'bg-selected-soft border border-selected' : 'bg-transparent border border-transparent'
            }`}>
              <input type="radio" name={`source_${domain}`} checked={isSel} disabled={!canSelect}
                onChange={() => onSelect(row.value, row)}
                className="shrink-0"
                style={{ accentColor: 'var(--accent)' }}
                title={canSelect ? t('settings.selectSourceCandidate') : t('settings.downloadRequiredFirst')}
              />
              <div className="flex flex-col flex-1 min-w-0">
                <div className="flex items-center gap-2 min-w-0">
                  <code className="font-mono text-fg-primary min-w-0 overflow-hidden text-ellipsis whitespace-nowrap" title={row.value}>
                    {row.label}
                  </code>
                  {row.kind === 'local' && (
                    <span className="text-[10px] px-1 py-0 rounded-sm bg-overlay text-fg-tertiary shrink-0">
                      {t('settings.localBadge')}
                    </span>
                  )}
                  {(row.kind === 'download' || row.kind === 'scanned') && (
                    <span className="text-[10px] px-1 py-0 rounded-sm bg-overlay text-fg-tertiary shrink-0">custom</span>
                  )}
                  {renderRowMeta?.(row)}
                </div>
                {(describeRow ? describeRow(row) : row.description) && (
                  <span className="text-fg-tertiary text-[11px] truncate">
                    {describeRow ? describeRow(row) : row.description}
                  </span>
                )}
              </div>
              {!row.exists && row.size_estimate ? (
                <span className="text-fg-tertiary shrink-0">~{fmtBytes(row.size_estimate)}</span>
              ) : null}
              {row.kind === 'local' && !row.exists ? (
                <span className="text-err text-2xs shrink-0">{t('settings.localModelMissing')}</span>
              ) : (
                <ModelStatusBadge
                  exists={row.exists} size={row.size} status={dl?.status}
                  fileCount={row.files?.length}
                  existsCount={row.files ? row.files.filter((f) => f.exists).length : undefined}
                />
              )}
              {row.download_id && (
                <DownloadButton
                  exists={row.exists} status={dl?.status}
                  busy={row.status_key ? downloadBusy.has(row.status_key) : false}
                  onClick={() => void startDownload(row.download_id!, row.download_variant ?? row.value)}
                  onDelete={() => void deleteAsset(row.download_id!, row.download_variant ?? row.value, row.label)}
                />
              )}
              {row.removable && (
                <button
                  onClick={() => void removeCandidate(row)}
                  className="btn btn-ghost btn-sm min-w-[5rem] justify-center shrink-0"
                  title={t('settings.removeCandidate')}
                >✕ {t('settings.removeCandidateShort')}</button>
              )}
            </li>
          )
        })}
      </ul>

      <div className="flex gap-1.5 mt-1">
        {addDownload && (
          <button type="button"
            onClick={() => setAdding(adding === 'download' ? null : 'download')}
            className="btn btn-ghost btn-sm text-xs text-fg-tertiary">
            + {t('settings.addDownloadCandidate')}
          </button>
        )}
        {addLocal && (
          <button type="button" onClick={() => setAdding('local')}
            className="btn btn-ghost btn-sm text-xs text-fg-tertiary">
            + {t('settings.addLocalCandidate')}
          </button>
        )}
      </div>

      {adding === 'download' && addDownload && (
        <div className="flex flex-col gap-1.5">
          <div className="flex gap-1.5">
            <input
              type="text"
              value={repoDraft}
              onChange={(e) => setRepoDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !addDownload.filenameField) {
                  e.preventDefault(); submitDownloadForm()
                }
              }}
              placeholder={addDownload.repoPlaceholder ?? t('settings.addHfModelId')}
              className={`${textInputClass} flex-1 font-mono`}
            />
            {addDownload.filenameField && (
              <input
                type="text"
                value={fileDraft}
                onChange={(e) => setFileDraft(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); submitDownloadForm() } }}
                placeholder={addDownload.filenamePlaceholder ?? t('common.filename')}
                className={`${textInputClass} flex-1 font-mono`}
              />
            )}
            <button
              onClick={submitDownloadForm}
              disabled={submitBusy || !repoDraft.trim() || (addDownload.filenameField ? !fileDraft.trim() : false)}
              className="btn btn-secondary btn-sm"
            >{t('settings.add')}</button>
          </div>
        </div>
      )}

      {adding === 'local' && addLocal && (
        <PathPicker
          initialPath={catalog.models_root}
          dirOnly={addLocal.dirOnly}
          onPick={(p) => {
            if (addLocal.secondFileKey) {
              // 双文件资产：记住主文件，再选第二个文件
              setFirstLocalPath(p)
              setAdding('local2')
            } else {
              setAdding(null)
              void submitCandidate({ kind: 'local', path: p })
            }
          }}
          onClose={() => setAdding(null)}
        />
      )}
      {adding === 'local2' && addLocal?.secondFileKey && (
        <PathPicker
          initialPath={firstLocalPath}
          onPick={(p) => {
            setAdding(null)
            void submitCandidate({
              kind: 'local', path: firstLocalPath,
              extra: { [addLocal.secondFileKey!]: p },
            })
          }}
          onClose={() => setAdding(null)}
        />
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

/** 已下载资产的「删除」按钮（下载的逆操作：用户先删再下载）。 */
export function DeleteAssetButton({ onClick }: { onClick: () => void }) {
  const { t } = useTranslation()
  return (
    <button onClick={onClick} className="btn btn-ghost btn-sm min-w-[5rem] justify-center"
      title={t('settings.deleteAssetTitle')}>
      🗑 {t('settings.deleteAsset')}
    </button>
  )
}

export function DownloadButton({ exists, status, busy, onClick, onDelete }: {
  exists: boolean; status?: ModelDownloadStatus['status']; busy: boolean; onClick: () => void
  /** 已下载状态的 action：删除（用户先删再下载）。 */
  onDelete: () => void
}) {
  const { t } = useTranslation()
  const running = status === 'running' || busy
  if (running) {
    return <button disabled className="btn btn-secondary btn-sm min-w-[5rem] justify-center" style={{ opacity: 0.5 }}>...</button>
  }
  if (exists) {
    return <DeleteAssetButton onClick={onDelete} />
  }
  return (
    <button onClick={onClick} className="btn btn-secondary btn-sm min-w-[5rem] justify-center"
      title={t('common.download')}>
      {t('settings.downloadAction')}
    </button>
  )
}
