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
  const { project, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const [options, setOptions] = useState<DuplicateScanOptions>(DEFAULT_DUPLICATE_OPTIONS)
  const [result, setResult] = useState<DuplicateScanResult | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [busy, setBusy] = useState(false)
  const [logs, setLogs] = useState<DuplicateLog[]>([])
  const [scanLogVisible, setScanLogVisible] = useState(false)
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)
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
    () => result ? result.groups.flatMap((group) => group.items.map((item) => item.name)) : [],
    [result],
  )

  const scan = async () => {
    if (busy) return
    setBusy(true)
    setResult(null)
    setSelected(new Set())
    setScanLogVisible(true)
    setLogs([{ ts: Date.now(), status: 'running', text: t('duplicates.logStarted') }])
    try {
      const next = await api.scanDuplicates(project.id, options)
      setResult(next)
      setSelected(new Set(
        next.groups.flatMap((group) =>
          group.items.filter((item) => !item.keep).map((item) => item.name),
        ),
      ))
      toast(t('duplicates.scanDone', { groups: next.group_count, candidates: next.candidate_count }), 'success')
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
      const res = await api.applyDuplicateAction(project.id, { names })
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

  return (
    <StepShell
      idx={2}
      title={t('steps.preprocess.title')}
      subtitle={t('duplicates.subtitle')}
    >
      <div className="flex flex-col h-full gap-3 min-h-0">
        <div className="grid gap-3 flex-1 min-h-0" style={{ gridTemplateColumns: '1fr 260px' }}>
          <div className="flex flex-col gap-2 min-h-0 min-w-0">
            <PreprocessToolsBar current="dedupe" projectId={project.id} />
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
            <DuplicateReviewPanel
              projectId={project.id}
              result={result}
              selected={selected}
              busy={busy}
              onSelect={setSelected}
              onPreview={openPreview}
            />
          </div>
          <DuplicateStatsSidebar
            result={result}
            selectedCount={selected.size}
            sourceTotal={project.download_image_count}
          />
        </div>
      </div>

      {previewIdx !== null && previewNames[previewIdx] && (
        <ImagePreviewModal
          src={api.projectThumbUrl(project.id, previewNames[previewIdx], 'download', 1600)}
          caption={previewNames[previewIdx]}
          hasPrev={previewIdx > 0}
          hasNext={previewIdx < previewNames.length - 1}
          onClose={() => setPreviewIdx(null)}
          onPrev={() => previewIdx > 0 && setPreviewIdx(previewIdx - 1)}
          onNext={() => previewIdx < previewNames.length - 1 && setPreviewIdx(previewIdx + 1)}
          shortcutHint={t('duplicates.previewHint')}
        />
      )}
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
