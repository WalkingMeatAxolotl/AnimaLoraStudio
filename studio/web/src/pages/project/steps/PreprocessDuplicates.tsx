import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type DuplicateScanOptions,
  type DuplicateScanResult,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import DuplicateReviewPanel, {
  DEFAULT_DUPLICATE_OPTIONS,
} from '../../../components/DuplicateReviewPanel'
import { useDialog } from '../../../components/Dialog'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import StepShell from '../../../components/StepShell'
import PreprocessToolsBar from '../../../components/preprocess/PreprocessToolsBar'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

interface DuplicateLog {
  ts: number
  text: string
  status?: string
}

export default function PreprocessDuplicatesPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const vid = activeVersion?.id ?? 0
  const [options, setOptions] = useState<DuplicateScanOptions>(DEFAULT_DUPLICATE_OPTIONS)
  const [result, setResult] = useState<DuplicateScanResult | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [busy, setBusy] = useState(false)
  const [logs, setLogs] = useState<DuplicateLog[]>([])
  const [scanLogVisible, setScanLogVisible] = useState(false)
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)
  const [reviewMode, setReviewMode] = useState<'groups' | 'quality'>('groups')
  const lastLogAtRef = useRef(0)

  useEffect(() => {
    lastLogAtRef.current = 0
    setLogs([])
    setScanLogVisible(false)
  }, [project.id])

  useEventStream((evt) => {
    if (evt.type !== 'duplicate_scan_progress' || evt.project_id !== project.id) return
    const now = Date.now()
    const status = String(evt.status ?? '')
    if (status === 'running' && now - lastLogAtRef.current < 1000) return
    lastLogAtRef.current = now
    setLogs((prev) => [
      ...prev.slice(-119),
      {
        ts: now,
        status,
        text: String(evt.text ?? status),
      },
    ])
  })

  const previewNames = useMemo(
    () => result
      ? Array.from(new Set([
          ...result.groups.flatMap((group) => group.items.map((item) => item.name)),
          ...result.blur_candidates.map((item) => item.name),
          ...result.crop_relations.flatMap((item) => [item.source, item.crop_candidate]),
        ]))
      : [],
    [result],
  )
  const hasQualityCandidates = !!result && (
    result.blur_candidates.length > 0 || result.crop_relations.length > 0
  )
  // 渲染期推导兜底：质量候选清空后自动落回 groups，不需要 effect 改 state
  //（scan() 重置 reviewMode，质量计数只会在重扫时变化）。
  const activeReviewMode = reviewMode === 'quality' && hasQualityCandidates ? 'quality' : 'groups'

  const scan = async () => {
    if (busy) return
    setBusy(true)
    setResult(null)
    setSelected(new Set())
    setReviewMode('groups')
    setScanLogVisible(true)
    setLogs([{ ts: Date.now(), status: 'running', text: t('duplicates.logStarted') }])
    try {
      const next = await api.scanDuplicatesTrain(project.id, vid, options)
      setResult(next)
      setSelected(new Set(
        next.groups.flatMap((group) =>
          group.items.filter((item) => !item.keep).map((item) => item.name),
        ),
      ))
      toast(
        t('duplicates.scanDone', {
          groups: next.group_count,
          candidates: next.candidate_count,
          blur: next.blur_candidate_count,
          crops: next.crop_relation_count,
        }),
        'success',
      )
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const apply = async () => {
    if (busy || selected.size === 0) return
    const names = Array.from(selected)
    const ok = await confirm(
      t('duplicates.confirmApply', { n: names.length }),
      { tone: 'warn', okText: t('duplicates.applyOk') },
    )
    if (!ok) return
    setBusy(true)
    try {
      const res = await api.applyDuplicateActionTrain(project.id, vid, { names })
      toast(
        t('duplicates.appliedToast', { n: res.removed.length }) +
          (res.skipped.length ? t('duplicates.appliedSkipped', { n: res.skipped.length }) : '') +
          (res.missing.length ? t('duplicates.appliedMissing', { n: res.missing.length }) : ''),
        'success',
      )
      setSelected(new Set())
      setResult(null)
      setPreviewIdx(null)
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const openPreview = (name: string) => {
    const index = previewNames.indexOf(name)
    setPreviewIdx(index >= 0 ? index : 0)
  }

  // ADR 0010: hooks 之后再做 vid guard
  if (!activeVersion) {
    return (
      <div className="p-6 text-fg-secondary">
        {t('projectStepper.selectVersion')}
      </div>
    )
  }

  return (
    <StepShell
      idx={2}
      title={t('steps.preprocess.title')}
      subtitle={t('duplicates.subtitle')}
    >
      <div className="flex flex-col h-full gap-3 min-h-0">
        <div className="grid gap-3 flex-1 min-h-0" style={{ gridTemplateColumns: '1fr 260px' }}>
          <div className="flex flex-col gap-2 min-h-0 min-w-0">
            <PreprocessToolsBar current="dedupe" projectId={project.id} versionId={vid} />
            <DuplicateOperationPanel
              options={options}
              result={result}
              selectedCount={selected.size}
              sourceTotal={project.download_image_count}
              busy={busy}
              onOptionsChange={setOptions}
              onScan={scan}
              onApply={apply}
            />
            {scanLogVisible && <DuplicateLogStrip logs={logs} busy={busy} />}
            <div className="flex flex-col flex-1 min-h-0 gap-2">
              <div className="shrink-0 rounded-md border border-subtle bg-surface px-2 py-1.5 flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => setReviewMode('groups')}
                  className={`btn btn-sm !py-1 text-xs ${
                    activeReviewMode === 'groups'
                      ? 'border-warn bg-warn-soft text-warn'
                      : 'btn-secondary'
                  }`}
                >
                  {t('duplicates.reviewTitle')} ({result?.group_count ?? 0})
                </button>
                <button
                  type="button"
                  disabled={!hasQualityCandidates}
                  onClick={() => setReviewMode('quality')}
                  className={`btn btn-sm !py-1 text-xs ${
                    activeReviewMode === 'quality'
                      ? 'border-accent bg-accent-soft text-accent'
                      : 'btn-secondary'
                  }`}
                >
                  {t('duplicates.qualityTitle')} ({(result?.blur_candidate_count ?? 0) + (result?.crop_relation_count ?? 0)})
                </button>
                <span className="text-xs text-fg-tertiary truncate">
                  {activeReviewMode === 'quality'
                    ? t('duplicates.qualitySummary', {
                        blur: result?.blur_candidate_count ?? 0,
                        crops: result?.crop_relation_count ?? 0,
                      })
                    : t('duplicates.panelSummary', {
                        total: result?.total_images ?? project.download_image_count ?? 0,
                        groups: result?.group_count ?? 0,
                        selected: selected.size,
                      })}
                </span>
              </div>

              {activeReviewMode === 'groups' ? (
                <DuplicateReviewPanel
                  projectId={project.id}
                  versionId={vid}
                  result={result}
                  selected={selected}
                  busy={busy}
                  onSelect={setSelected}
                  onPreview={openPreview}
                />
              ) : (
                <QualityReviewPanel
                  projectId={project.id}
                  versionId={vid}
                  result={result}
                  selected={selected}
                  busy={busy}
                  onSelectNames={(names) => {
                    const next = new Set(selected)
                    names.forEach((name) => next.add(name))
                    setSelected(next)
                  }}
                  onToggle={(name) => {
                    const next = new Set(selected)
                    if (next.has(name)) next.delete(name)
                    else next.add(name)
                    setSelected(next)
                  }}
                  onPreview={openPreview}
                />
              )}
            </div>
          </div>
          <DuplicateStatsSidebar
            result={result}
            selectedCount={selected.size}
            sourceTotal={project.download_image_count}
          />
        </div>
      </div>

      {previewIdx !== null && previewNames[previewIdx] && (() => {
        const rel = previewNames[previewIdx]
        const i = rel.lastIndexOf('/')
        const folder = i >= 0 ? rel.slice(0, i) : ''
        const filename = i >= 0 ? rel.slice(i + 1) : rel
        return (
        <ImagePreviewModal
          src={api.versionThumbUrl(project.id, vid, 'train', filename, folder, 1600)}
          caption={rel}
          hasPrev={previewIdx > 0}
          hasNext={previewIdx < previewNames.length - 1}
          onClose={() => setPreviewIdx(null)}
          onPrev={() => previewIdx > 0 && setPreviewIdx(previewIdx - 1)}
          onNext={() => previewIdx < previewNames.length - 1 && setPreviewIdx(previewIdx + 1)}
          shortcutHint={t('duplicates.previewHint')}
        />
        )
      })()}
    </StepShell>
  )
}

interface DuplicateOperationPanelProps {
  options: DuplicateScanOptions
  result: DuplicateScanResult | null
  selectedCount: number
  sourceTotal?: number | null
  busy: boolean
  onOptionsChange: (next: DuplicateScanOptions) => void
  onScan: () => void
  onApply: () => void
}

function DuplicateOperationPanel({
  options,
  result,
  selectedCount,
  sourceTotal,
  busy,
  onOptionsChange,
  onScan,
  onApply,
}: DuplicateOperationPanelProps) {
  const { t } = useTranslation()
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const total = result?.total_images ?? sourceTotal ?? 0
  const tileGridValue = options.tile_grids.join(',')
  const [tileGridsText, setTileGridsText] = useState(tileGridValue)
  useEffect(() => {
    setTileGridsText(tileGridValue)
  }, [tileGridValue])
  const patch = <K extends keyof DuplicateScanOptions>(key: K, value: DuplicateScanOptions[K]) => {
    onOptionsChange({ ...options, [key]: value })
  }
  const resetDefaults = () => {
    onOptionsChange(DEFAULT_DUPLICATE_OPTIONS)
  }
  const updateTileGrids = (value: string) => {
    setTileGridsText(value)
    const grids = value
      .split(',')
      .map((part) => Number(part.trim()))
      .filter((n) => Number.isFinite(n) && n > 0)
    if (grids.length) patch('tile_grids', grids)
  }

  return (
    <section className="flex flex-col gap-1.5 rounded-md border border-subtle bg-surface px-3 py-2.5 shrink-0">
      <h3 className="caption flex items-center gap-1.5">
        <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-warn" />
        {t('duplicates.panelTitle')}
      </h3>

      <div className="flex items-center gap-2 text-sm flex-wrap">
        <label className="flex items-center gap-1.5">
          <input
            type="checkbox"
            checked={options.detect_blur}
            disabled={busy}
            onChange={(e) => patch('detect_blur', e.target.checked)}
          />
          <span className="text-fg-tertiary">{t('duplicates.detectBlur')}</span>
        </label>
        <label className="flex items-center gap-1.5">
          <input
            type="checkbox"
            checked={options.detect_crops}
            disabled={busy}
            onChange={(e) => patch('detect_crops', e.target.checked)}
          />
          <span className="text-fg-tertiary">{t('duplicates.detectCrops')}</span>
        </label>
        <span className="text-dim">·</span>
        <label className="flex items-center gap-1.5">
          <span className="text-fg-tertiary">{t('duplicates.scope')}</span>
          <select
            className="input text-sm"
            style={{ width: 'auto', padding: '2px 6px' }}
            value={options.match_scope}
            onChange={(e) => patch('match_scope', e.target.value as DuplicateScanOptions['match_scope'])}
            disabled={busy}
          >
            <option value="strict">{t('duplicates.scopeStrict')}</option>
            <option value="both">{t('duplicates.scopeBoth')}</option>
          </select>
        </label>
        <span className="text-dim">·</span>
        <NumberOption
          label={t('duplicates.variantScore')}
          value={options.variant_score}
          min={40}
          max={98}
          step={1}
          disabled={busy}
          onChange={(value) => patch('variant_score', value)}
        />
        <span className="text-dim">·</span>
        <NumberOption
          label={t('duplicates.workers')}
          value={options.hash_workers}
          min={1}
          max={32}
          step={1}
          disabled={busy}
          onChange={(value) => patch('hash_workers', value)}
          width={56}
        />
        <span className="text-fg-secondary text-xs">
          {t('duplicates.panelSummary', { total, groups: result?.group_count ?? 0, selected: selectedCount })}
        </span>
        <div className="flex items-center gap-2 ml-auto shrink-0">
          <button type="button" onClick={onScan} disabled={busy} className="btn btn-primary btn-sm">
            {busy ? t('duplicates.scanning') : t('duplicates.scanBtn')}
          </button>
          <button
            type="button"
            onClick={onApply}
            disabled={busy || selectedCount === 0}
            className="btn btn-sm bg-warn-soft text-warn border-warn"
          >
            {t('duplicates.applyBtn', { n: selectedCount })}
          </button>
        </div>
      </div>

      <div className="flex flex-col gap-1.5 rounded-sm bg-sunken/40 border border-subtle px-2.5 py-1.5">
        <div className="flex items-baseline gap-2 text-xs">
          <button
            type="button"
            onClick={() => setAdvancedOpen((v) => !v)}
            className="flex items-baseline gap-2 text-left bg-transparent border-0 p-0 cursor-pointer flex-1 min-w-0"
          >
            <span className="text-fg-tertiary w-3 inline-block shrink-0">{advancedOpen ? '▾' : '▸'}</span>
            <span className="text-accent shrink-0">✦</span>
            <span className="font-medium text-fg-secondary shrink-0">{t('duplicates.advanced')}</span>
            <span className="text-fg-tertiary truncate">{t('duplicates.advancedDesc')}</span>
          </button>
          <button
            type="button"
            onClick={resetDefaults}
            disabled={busy}
            className="btn btn-ghost btn-sm !py-0.5 text-[11px]"
          >
            {t('duplicates.resetDefaults')}
          </button>
        </div>
        {advancedOpen && (
          <div className="flex items-center gap-2 text-sm flex-wrap">
            <NumberOption label={t('duplicates.hashSize')} value={options.hash_size} min={0} max={2048} step={64} disabled={busy} onChange={(value) => patch('hash_size', value)} width={74} />
            <NumberOption label={t('duplicates.structure')} value={options.structure_threshold} min={0} max={24} step={1} disabled={busy} onChange={(value) => patch('structure_threshold', value)} width={58} />
            <NumberOption label={t('duplicates.aspect')} value={options.aspect_tolerance} min={0.005} max={0.2} step={0.005} disabled={busy} onChange={(value) => patch('aspect_tolerance', value)} width={74} />
            <NumberOption label={t('duplicates.closeTiles')} value={options.min_close_tiles} min={0} max={1} step={0.01} disabled={busy} onChange={(value) => patch('min_close_tiles', value)} width={66} />
            <NumberOption label={t('duplicates.tileMedian')} value={options.tile_median} min={0} max={40} step={1} disabled={busy} onChange={(value) => patch('tile_median', value)} width={58} />
            <NumberOption label={t('duplicates.grayClose')} value={options.min_gray_close} min={0} max={1} step={0.01} disabled={busy} onChange={(value) => patch('min_gray_close', value)} width={66} />
            <NumberOption label={t('duplicates.blurScore')} value={options.blur_score_threshold} min={0} max={1000} step={5} disabled={busy || !options.detect_blur} onChange={(value) => patch('blur_score_threshold', value)} width={72} />
            <NumberOption label={t('duplicates.blurLocal')} value={options.blur_local_ratio} min={0} max={0.5} step={0.005} disabled={busy || !options.detect_blur} onChange={(value) => patch('blur_local_ratio', value)} width={72} />
            <NumberOption label={t('duplicates.cropScore')} value={options.crop_score} min={0.3} max={0.98} step={0.01} disabled={busy || !options.detect_crops} onChange={(value) => patch('crop_score', value)} width={66} />
            <NumberOption label={t('duplicates.cropHash')} value={options.crop_hash_threshold} min={0} max={64} step={1} disabled={busy || !options.detect_crops} onChange={(value) => patch('crop_hash_threshold', value)} width={58} />
            <NumberOption label={t('duplicates.cropSide')} value={options.crop_max_side} min={128} max={768} step={32} disabled={busy || !options.detect_crops} onChange={(value) => patch('crop_max_side', value)} width={66} />
            <NumberOption label={t('duplicates.cropWorkers')} value={options.crop_workers} min={1} max={32} step={1} disabled={busy || !options.detect_crops} onChange={(value) => patch('crop_workers', value)} width={56} />
            <NumberOption label={t('duplicates.cropSegments')} value={options.crop_prefilter_min_segments} min={1} max={8} step={1} disabled={busy || !options.detect_crops} onChange={(value) => patch('crop_prefilter_min_segments', value)} width={54} />
            <NumberOption label={t('duplicates.cropCoverage')} value={options.crop_prefilter_min_coverage} min={0} max={1} step={0.01} disabled={busy || !options.detect_crops} onChange={(value) => patch('crop_prefilter_min_coverage', value)} width={66} />
            <NumberOption label={t('duplicates.cropAspectPrefilter')} value={options.crop_prefilter_aspect_tolerance} min={0} max={1.5} step={0.01} disabled={busy || !options.detect_crops} onChange={(value) => patch('crop_prefilter_aspect_tolerance', value)} width={66} />
            <NumberOption label={t('duplicates.cropCandidateCap')} value={options.crop_max_candidates_per_image} min={0} max={100} step={1} disabled={busy || !options.detect_crops} onChange={(value) => patch('crop_max_candidates_per_image', value)} width={58} />
            <label className="flex items-center gap-1.5">
              <span className="text-fg-tertiary">{t('duplicates.tileGrids')}</span>
              <input
                className="input input-mono text-sm"
                style={{ width: 86, padding: '2px 6px' }}
                value={tileGridsText}
                disabled={busy}
                onChange={(e) => updateTileGrids(e.target.value)}
                onBlur={() => setTileGridsText(options.tile_grids.join(','))}
              />
            </label>
          </div>
        )}
      </div>
    </section>
  )
}

function NumberOption({
  label,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
  width = 68,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  disabled: boolean
  onChange: (value: number) => void
  width?: number
}) {
  return (
    <label className="flex items-center gap-1.5">
      <span className="text-fg-tertiary">{label}</span>
      <input
        type="number"
        className="input input-mono text-sm"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width, padding: '2px 6px' }}
      />
    </label>
  )
}

function DuplicateLogStrip({ logs, busy }: { logs: DuplicateLog[]; busy: boolean }) {
  const { t } = useTranslation()
  const lastLine = logs[logs.length - 1]?.text ?? ''
  return (
    <details open={busy} className="group rounded-md border border-subtle bg-surface overflow-hidden shrink-0">
      <summary className="cursor-pointer flex items-center gap-2 list-none px-2.5 py-1.5 text-sm select-none">
        <span className="inline-block transition-transform group-open:rotate-90 text-fg-tertiary w-3">▸</span>
        <span className={busy ? 'badge badge-warn' : 'badge badge-neutral'}>
          {busy ? t('duplicates.scanning') : t('duplicates.logTitle')}
        </span>
        <span className="mono truncate flex-1 min-w-0 text-fg-secondary text-xs">{lastLine}</span>
      </summary>
      <pre className="px-3 py-2 text-xs font-mono text-fg-secondary bg-sunken max-h-[224px] overflow-auto whitespace-pre-wrap border-t border-subtle m-0">
        {logs.length === 0 ? t('jobProgress.waitingLogs') : logs.map((line) => line.text).join('\n')}
      </pre>
    </details>
  )
}

function QualityReviewPanel({
  projectId,
  versionId,
  result,
  selected,
  busy,
  onSelectNames,
  onToggle,
  onPreview,
}: {
  projectId: number
  versionId: number
  result: DuplicateScanResult | null
  selected: Set<string>
  busy: boolean
  onSelectNames: (names: string[]) => void
  onToggle: (name: string) => void
  onPreview: (name: string) => void
}) {
  const { t } = useTranslation()
  const blurCandidates = result?.blur_candidates ?? []
  const cropRelations = result?.crop_relations ?? []
  const [activeTab, setActiveTab] = useState<'blur' | 'crop'>('blur')
  const blurNames = useMemo(
    () => Array.from(new Set((result?.blur_candidates ?? []).map((item) => item.name))),
    [result],
  )
  const cropCandidateNames = useMemo(
    () => Array.from(new Set((result?.crop_relations ?? []).map((item) => item.crop_candidate))),
    [result],
  )
  const cropSourceNames = useMemo(
    () => Array.from(new Set((result?.crop_relations ?? []).map((item) => item.source))),
    [result],
  )
  const cropBothNames = useMemo(
    () => Array.from(new Set((result?.crop_relations ?? []).flatMap((item) => [item.source, item.crop_candidate]))),
    [result],
  )
  const cropRelationKindLabel = (kind: string) => {
    if (kind === 'crop_upscaled') return t('duplicates.cropKindUpscaled')
    if (kind === 'crop_same_area') return t('duplicates.cropKindSameArea')
    return t('duplicates.cropKindSmaller')
  }
  const cropLargerLabel = (name: string) => (
    name === 'same_area'
      ? t('duplicates.cropLargerSameArea')
      : t('duplicates.cropLarger', { name })
  )
  if (!result || (blurCandidates.length === 0 && cropRelations.length === 0)) return null
  // 渲染期推导：某一侧为空时强制落到另一侧（tab 按钮对空侧也是 disabled），
  // 不用 effect 写回 state。
  const activeQualityTab =
    blurCandidates.length === 0 && cropRelations.length > 0
      ? 'crop'
      : cropRelations.length === 0 && blurCandidates.length > 0
        ? 'blur'
        : activeTab
  const showBlur = activeQualityTab === 'blur'
  return (
    <section className="flex flex-col flex-1 min-h-0 rounded-md border border-subtle bg-surface overflow-hidden">
      <div className="h-0.5 bg-accent" />
      <header className="flex flex-wrap items-center gap-2 px-2.5 py-1.5 border-b border-subtle text-sm">
        <h3 className="font-semibold">{t('duplicates.qualityTitle')}</h3>
        <span className="text-xs text-fg-tertiary">
          {t('duplicates.qualitySummary', { blur: blurCandidates.length, crops: cropRelations.length })}
        </span>
        <span className="text-xs text-fg-tertiary min-w-full">
          {t('duplicates.qualityHint')}
        </span>
        <div className="grid grid-cols-2 gap-1.5 min-w-full">
          <button
            type="button"
            disabled={blurCandidates.length === 0}
            onClick={() => setActiveTab('blur')}
            className={`btn btn-sm !py-1 text-xs justify-center ${
              showBlur
                ? 'border-accent bg-accent-soft text-accent'
                : 'btn-secondary'
            }`}
          >
            {t('duplicates.blurTitle')} ({blurCandidates.length})
          </button>
          <button
            type="button"
            disabled={cropRelations.length === 0}
            onClick={() => setActiveTab('crop')}
            className={`btn btn-sm !py-1 text-xs justify-center ${
              !showBlur
                ? 'border-accent bg-accent-soft text-accent'
                : 'btn-secondary'
            }`}
          >
            {t('duplicates.cropTitle')} ({cropRelations.length})
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-1.5 min-w-full">
          {showBlur ? (
            <button
              type="button"
              disabled={busy || blurNames.length === 0}
              onClick={() => onSelectNames(blurNames)}
              className="btn btn-secondary btn-sm !py-0.5 text-[11px]"
            >
              {t('duplicates.selectBlur')}
            </button>
          ) : (
            <>
              <button
                type="button"
                disabled={busy || cropCandidateNames.length === 0}
                onClick={() => onSelectNames(cropCandidateNames)}
                className="btn btn-secondary btn-sm !py-0.5 text-[11px]"
              >
                {t('duplicates.selectCropCandidates')}
              </button>
              <button
                type="button"
                disabled={busy || cropSourceNames.length === 0}
                onClick={() => onSelectNames(cropSourceNames)}
                className="btn btn-secondary btn-sm !py-0.5 text-[11px]"
              >
                {t('duplicates.selectCropSources')}
              </button>
              <button
                type="button"
                disabled={busy || cropBothNames.length === 0}
                onClick={() => onSelectNames(cropBothNames)}
                className="btn btn-secondary btn-sm !py-0.5 text-[11px]"
              >
                {t('duplicates.selectCropBoth')}
              </button>
            </>
          )}
        </div>
      </header>
      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        {showBlur && (
          <QualitySection
            title={t('duplicates.blurTitle')}
            empty={t('duplicates.blurEmpty')}
            items={blurCandidates.map((item) => ({
              key: item.name,
              images: [{ name: item.name }],
              meta: `${item.width}x${item.height} · ${item.filesize_kb}KB`,
              score: t('duplicates.blurMetric', {
                score: Math.round(item.blur_score),
                local: Math.round(item.largest_blur_region_ratio * 100),
              }),
              note: item.reason,
            }))}
            projectId={projectId}
            versionId={versionId}
            selected={selected}
            busy={busy}
            onToggle={onToggle}
            onPreview={onPreview}
          />
        )}
        {!showBlur && (
          <QualitySection
            title={t('duplicates.cropTitle')}
            empty={t('duplicates.cropEmpty')}
            items={cropRelations.map((item, index) => ({
              key: `${item.source}:${item.crop_candidate}:${index}`,
              images: [
                { name: item.source, label: t('duplicates.cropSource') },
                { name: item.crop_candidate, label: t('duplicates.cropCandidate') },
              ],
              meta: `${item.source_width}x${item.source_height} → ${item.crop_width}x${item.crop_height}`,
              score: t('duplicates.cropMetric', {
                score: Math.round(item.score * 100),
                area: Math.round(item.window_ratio * 100),
              }),
              note: [
                cropRelationKindLabel(item.relation_kind),
                cropLargerLabel(item.larger_image),
                t('duplicates.cropAreaRatio', { ratio: item.area_ratio.toFixed(2) }),
                `${item.source_window.x},${item.source_window.y},${item.source_window.width},${item.source_window.height}`,
                item.note,
              ].join(' · '),
            }))}
            projectId={projectId}
            versionId={versionId}
            selected={selected}
            busy={busy}
            onToggle={onToggle}
            onPreview={onPreview}
          />
        )}
      </div>
    </section>
  )
}

function QualitySection({
  title,
  empty,
  items,
  projectId,
  versionId,
  selected,
  busy,
  onToggle,
  onPreview,
}: {
  title: string
  empty: string
  items: Array<{ key: string; images: Array<{ name: string; label?: string }>; meta: string; score: string; note: string }>
  projectId: number
  versionId: number
  selected: Set<string>
  busy: boolean
  onToggle: (name: string) => void
  onPreview: (name: string) => void
}) {
  return (
    <div className="rounded-sm border border-subtle bg-sunken/40 overflow-hidden">
      <div className="px-2 py-1.5 border-b border-subtle text-xs font-medium text-fg-secondary">{title}</div>
      {items.length === 0 ? (
        <div className="px-2 py-3 text-xs text-fg-tertiary">{empty}</div>
      ) : (
        // 卡片铺响应式多列：缩略图只有 256px，卡片不能拉满面板宽（会放大到糊）。
        // 滚动交给面板内容区的 overflow-y-auto，这里不再嵌套 max-h 滚动窗口。
        <div
          className="p-2 grid gap-2"
          style={{
            gridTemplateColumns: `repeat(auto-fill, minmax(${
              items.some((item) => item.images.length > 1) ? 340 : 200
            }px, 1fr))`,
          }}
        >
          {items.map((item) => (
            <article key={item.key} className="rounded-sm border border-subtle bg-surface p-1.5">
              <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${item.images.length}, minmax(0, 1fr))` }}>
                {item.images.map((image) => (
                  <QualityImageCell
                    key={image.name}
                    projectId={projectId}
                    versionId={versionId}
                    name={image.name}
                    label={image.label}
                    selected={selected.has(image.name)}
                    busy={busy}
                    onToggle={() => onToggle(image.name)}
                    onPreview={() => onPreview(image.name)}
                  />
                ))}
              </div>
              <div className="mt-1.5 flex flex-col gap-0.5 text-[11px]">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="badge badge-neutral shrink-0">{item.score}</span>
                  <span className="text-fg-tertiary truncate">{item.meta}</span>
                </div>
                <code className="mono text-fg-secondary truncate">{item.images.map((image) => image.name).join(' <-> ')}</code>
                <div className="text-fg-tertiary truncate" title={item.note}>{item.note}</div>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  )
}

function QualityImageCell({
  projectId,
  versionId,
  name,
  label,
  selected,
  busy,
  onToggle,
  onPreview,
}: {
  projectId: number
  versionId: number
  name: string
  label?: string
  selected: boolean
  busy: boolean
  onToggle: () => void
  onPreview: () => void
}) {
  const { t } = useTranslation()
  return (
    <div
      className={
        'rounded-sm border overflow-hidden bg-surface ' +
        (selected ? 'border-warn ring-2 ring-warn-soft' : 'border-subtle')
      }
    >
      <button
        type="button"
        disabled={busy}
        onClick={onPreview}
        className="block w-full aspect-square bg-sunken disabled:opacity-70"
        title={name}
      >
        {(() => {
          const i = name.lastIndexOf('/')
          const folder = i >= 0 ? name.slice(0, i) : ''
          const filename = i >= 0 ? name.slice(i + 1) : name
          return (
            <img
              src={api.versionThumbUrl(projectId, versionId, 'train', filename, folder, 256)}
              alt={name}
              loading="lazy"
              decoding="async"
              className="w-full h-full object-cover"
            />
          )
        })()}
      </button>
      <div className="px-1.5 py-1 flex items-center gap-1 min-w-0">
        <button
          type="button"
          onClick={onToggle}
          disabled={busy}
          className={`shrink-0 px-1.5 py-0.5 rounded-sm border text-[11px] font-medium ${
            selected
              ? 'bg-warn text-white border-warn'
              : 'bg-ok-soft text-ok border-ok'
          } disabled:opacity-60 disabled:cursor-not-allowed`}
          aria-label={`${selected ? t('duplicates.restoreCandidate') : t('duplicates.removeCandidate')} ${name}`}
        >
          {selected ? t('duplicates.selectedRemove') : t('duplicates.keep')}
        </button>
        {label && <span className="badge badge-neutral shrink-0">{label}</span>}
        <code className="mono truncate min-w-0 text-[11px]">{name}</code>
      </div>
    </div>
  )
}

function DuplicateStatsSidebar({
  result,
  selectedCount,
  sourceTotal,
}: {
  result: DuplicateScanResult | null
  selectedCount: number
  sourceTotal?: number | null
}) {
  const { t } = useTranslation()
  const total = result?.total_images ?? sourceTotal ?? 0
  const candidateCount = result?.candidate_count ?? 0
  const remaining = Math.max(0, total - selectedCount)
  return (
    <aside className="flex flex-col gap-3 min-w-0">
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h3 className="caption flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-warn" />
          {t('duplicates.statsTitle')}
        </h3>
        <StatRow label={t('duplicates.statsTotal')} value={total} />
        <StatRow label={t('duplicates.statsGroups')} value={result?.group_count ?? 0} accent={(result?.group_count ?? 0) > 0 ? 'warn' : undefined} />
        <StatRow label={t('duplicates.statsCandidates')} value={candidateCount} accent={candidateCount > 0 ? 'warn' : undefined} />
        <StatRow label={t('duplicates.statsBlur')} value={result?.blur_candidate_count ?? 0} accent={(result?.blur_candidate_count ?? 0) > 0 ? 'warn' : undefined} />
        <StatRow label={t('duplicates.statsCrops')} value={result?.crop_relation_count ?? 0} accent={(result?.crop_relation_count ?? 0) > 0 ? 'warn' : undefined} />
        <StatRow label={t('duplicates.statsSelected')} value={selectedCount} accent={selectedCount > 0 ? 'err' : undefined} />
        <StatRow label={t('duplicates.statsAfter')} value={remaining} accent="ok" />
      </div>
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h3 className="caption flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent" />
          {t('duplicates.statsScan')}
        </h3>
        <StatRow label={t('duplicates.statsReadable')} value={result?.readable_images ?? 0} />
        <StatRow label={t('duplicates.statsCompared')} value={result?.stats.compared_pairs ?? 0} />
        <StatRow label={t('duplicates.statsElapsed')} value={result ? `${result.elapsed_seconds}s` : '—'} />
      </div>
    </aside>
  )
}

function StatRow({
  label,
  value,
  accent,
}: {
  label: string
  value: string | number
  accent?: 'ok' | 'warn' | 'err'
}) {
  const cls =
    accent === 'ok' ? 'text-ok' :
    accent === 'warn' ? 'text-warn' :
    accent === 'err' ? 'text-err' :
    'text-fg-primary'
  return (
    <div className="flex justify-between items-baseline mt-1.5 text-xs gap-2">
      <span className="text-fg-tertiary">{label}</span>
      <span className={`font-mono font-medium ${cls}`}>{value}</span>
    </div>
  )
}
