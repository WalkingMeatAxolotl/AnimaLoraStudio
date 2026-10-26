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
import ActionGroup from '../../../components/ActionGroup'
import Button from '../../../components/Button'
import { useDialog } from '../../../components/Dialog'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import { SegmentedControl } from '../../../components/SelectionGroup'
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

function candidateNames(result: DuplicateScanResult | null): Set<string> {
  return new Set(
    result
      ? result.groups.flatMap((group) =>
          group.items.filter((item) => !item.keep).map((item) => item.name),
        )
      : [],
  )
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
  const [scanning, setScanning] = useState(false)
  const [logs, setLogs] = useState<DuplicateLog[]>([])
  const [scanLogVisible, setScanLogVisible] = useState(false)
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)
  const lastLogAtRef = useRef(0)
  const scanStartedRef = useRef(false)

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

  // 裁剪/缩放候选已并入分组（更严格的重复判断），故预览名单直接取分组成员即可。
  const previewNames = useMemo(
    () => result
      ? Array.from(new Set(
          result.groups.flatMap((group) => group.items.map((item) => item.name)),
        ))
      : [],
    [result],
  )

  const showResult = (next: DuplicateScanResult | null) => {
    setResult(next)
    setSelected(candidateNames(next))
  }

  const loadSavedScan = async () => {
    try {
      showResult((await api.getLastDuplicateScanTrain(project.id, vid)).result)
    } catch {
      showResult(null)
    }
  }

  useEffect(() => {
    if (!vid) return
    let cancelled = false
    scanStartedRef.current = false
    api.getLastDuplicateScanTrain(project.id, vid)
      .then((res) => {
        if (cancelled || scanStartedRef.current) return
        setResult(res.result)
        setSelected(candidateNames(res.result))
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [project.id, vid])

  const scan = async () => {
    if (busy) return
    scanStartedRef.current = true
    setBusy(true)
    setScanning(true)
    showResult(null)
    setScanLogVisible(true)
    setLogs([{ ts: Date.now(), status: 'running', text: t('duplicates.logStarted') }])
    try {
      const next = await api.scanDuplicatesTrain(project.id, vid, options)
      showResult(next)
      toast(
        t('duplicates.scanDone', {
          groups: next.group_count,
          candidates: next.candidate_count,
          crops: next.crop_relation_count,
        }),
        'success',
      )
    } catch (e) {
      toast(String(e), 'error')
      setLogs((prev) => [...prev, { ts: Date.now(), status: 'error', text: String(e) }])
    } finally {
      setScanning(false)
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
      setPreviewIdx(null)
      // 后端返回的上次扫描已裁掉刚去除的图，剩下的就是还要审的组。
      await loadSavedScan()
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
      title={t('steps.preprocess.title')}
      subtitle={t('duplicates.subtitle')}
      actions={
        <ActionGroup
          aria-label={t('duplicates.actionsLabel')}
          secondary={(
            <Button
              variant="ghost"
              size="sm"
              loading={scanning}
              disabled={busy}
              onClick={() => void scan()}
            >
              {scanning ? t('duplicates.scanning') : t('duplicates.scanBtn')}
            </Button>
          )}
          primary={(
            <Button
              variant="primary"
              size="sm"
              onClick={() => void apply()}
              disabled={busy || selected.size === 0}
            >
              {t('duplicates.applyBtn', { n: selected.size })}
            </Button>
          )}
        />
      }
      belowHeader={<PreprocessToolsBar current="dedupe" projectId={project.id} versionId={vid} />}
      logSources={[
        scanLogVisible && logs.length > 0 && {
          key: 'dup_scan',
          label: t('logDrawer.dupScan'),
          // 扫描是同步 HTTP + SSE 进度，前端合成状态：跑着 = running，
          // 否则按最后一条日志判 failed/done。不可取消。
          status: scanning
            ? ('running' as const)
            : logs[logs.length - 1]?.status === 'error'
              ? ('failed' as const)
              : ('done' as const),
          // 错误行加裸 `ERROR:` 前缀：LogView 按行契约的兼容规则识别级别着色
          lines: logs.map((l) => (l.status === 'error' ? `ERROR: ${l.text}` : l.text)),
          startedAt: logs[0] ? logs[0].ts / 1000 : null,
          finishedAt: scanning ? null : logs[logs.length - 1] ? logs[logs.length - 1].ts / 1000 : null,
        },
      ]}
    >
      <div className="flex flex-col h-full gap-3 min-h-0">
        <div className="grid gap-3 flex-1 min-h-0" style={{ gridTemplateColumns: '1fr 260px' }}>
          <div className="flex flex-col gap-2 min-h-0 min-w-0">
            <DuplicateOperationPanel
              options={options}
              busy={busy}
              onOptionsChange={setOptions}
            />
            {result?.saved_at && !scanning && (
              <p className="text-xs text-fg-tertiary px-1 shrink-0">
                {t('duplicates.savedScan', {
                  time: new Date(result.saved_at * 1000).toLocaleString(),
                })}
              </p>
            )}
            <DuplicateReviewPanel
              projectId={project.id}
              versionId={vid}
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

      {previewIdx !== null && previewNames[previewIdx] && (() => {
        const rel = previewNames[previewIdx]
        const i = rel.lastIndexOf('/')
        const folder = i >= 0 ? rel.slice(0, i) : ''
        const filename = i >= 0 ? rel.slice(i + 1) : rel
        return (
        <ImagePreviewModal
          src={api.versionThumbUrl(project.id, vid, 'train', filename, folder, 1600)}
          caption={rel}
          index={previewIdx}
          total={previewNames.length}
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

// 灵敏度三档：宽松/标准/严格。只对「差分 + 裁剪」判定生效（驱动后端 variant_score
// + crop_score），严格重复模式下 gate 掉。
const SENSITIVITY_OPTIONS = [
  { id: 'loose', key: 'sensitivityLoose' },
  { id: 'standard', key: 'sensitivityStandard' },
  { id: 'strict', key: 'sensitivityStrict' },
] as const

interface DuplicateOperationPanelProps {
  options: DuplicateScanOptions
  busy: boolean
  onOptionsChange: (next: DuplicateScanOptions) => void
}

function DuplicateOperationPanel({
  options,
  busy,
  onOptionsChange,
}: DuplicateOperationPanelProps) {
  const { t } = useTranslation()
  const patch = <K extends keyof DuplicateScanOptions>(key: K, value: DuplicateScanOptions[K]) => {
    onOptionsChange({ ...options, [key]: value })
  }
  // 灵敏度只影响「差分/裁剪」判定；严格重复模式只看全图哈希，故 gate 掉。
  const sensitivityLocked = options.match_scope !== 'both'

  return (
    <section className="flex flex-col gap-1.5 rounded-md border border-subtle bg-surface px-3 py-2.5 shrink-0">
      <h3 className="caption flex items-center gap-1.5">
        <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-warn" />
        {t('duplicates.panelTitle')}
      </h3>

      <div className="flex items-center gap-2 text-sm flex-wrap">
        <div className="flex items-center gap-1.5">
          <span className="text-fg-tertiary">{t('duplicates.scope')}</span>
          <SegmentedControl
            items={([
              ['strict', 'scopeStrict'],
              ['both', 'scopeBoth'],
            ] as const).map(([value, key]) => ({
              value,
              label: t(`duplicates.${key}`),
              disabled: busy,
            }))}
            value={options.match_scope}
            onChange={(value) => patch('match_scope', value as DuplicateScanOptions['match_scope'])}
            ariaLabel={t('duplicates.scope')}
            idPrefix="duplicate-scope"
            size="sm"
            layout="content"
          />
        </div>
        <span className="text-dim">·</span>
        <div
          className={'flex items-center gap-1.5' + (sensitivityLocked ? ' opacity-50' : '')}
          title={sensitivityLocked ? t('duplicates.sensitivityLockedHint') : undefined}
        >
          <span className="text-fg-tertiary">{t('duplicates.sensitivity')}</span>
          <SegmentedControl
            items={SENSITIVITY_OPTIONS.map(({ id, key }) => ({
              value: id,
              label: t(`duplicates.${key}`),
              disabled: busy || sensitivityLocked,
            }))}
            value={options.sensitivity}
            onChange={(value) => patch('sensitivity', value)}
            ariaLabel={t('duplicates.sensitivity')}
            idPrefix="duplicate-sensitivity"
            size="sm"
            layout="content"
          />
        </div>
      </div>
    </section>
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
