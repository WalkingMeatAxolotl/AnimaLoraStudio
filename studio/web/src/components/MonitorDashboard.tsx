/**
 * MonitorDashboard — native React training monitor.
 * Data source: GET /api/state?task_id=N snapshot + SSE monitor_progress deltas.
 * The dashboard is intentionally read-only; task mutations belong to QueueDetail.
 */
import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from 'react'
import { useTranslation } from 'react-i18next'

import { api, type TaskStatus } from '../api/client'
import { useMonitorProgress } from '../lib/useMonitorProgress'
import Alert from './Alert'
import Badge from './Badge'
import Button from './Button'
import Card from './Card'
import EmptyState from './EmptyState'
import ImagePreviewModal from './ImagePreviewModal'
import ProgressBar from './ProgressBar'
import { SeriesChart } from './SeriesChart'

function formatDuration(seconds: number, units: { hour: string; minute: string; second: string }): string {
  if (!seconds || seconds < 0) return '--'
  const hour = Math.floor(seconds / 3600)
  const minute = Math.floor((seconds % 3600) / 60)
  const second = Math.floor(seconds % 60)
  if (hour > 0) return `${hour}${units.hour} ${String(minute).padStart(2, '0')}${units.minute}`
  if (minute > 0) return `${minute}${units.minute} ${String(second).padStart(2, '0')}${units.second}`
  return `${second}${units.second}`
}

function formatLearningRate(value: number | null): string {
  if (value === null) return '--'
  if (value < 0.0001) return value.toExponential(1)
  return value.toFixed(5).replace(/0+$/, '').replace(/\.$/, '')
}

function formatMetric(value: number | null): string {
  if (value === null) return '--'
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 10000) return value.toExponential(2)
  return value.toFixed(5).replace(/0+$/, '').replace(/\.$/, '')
}

function StatCard({ label, value, sub, tone }: {
  label: string
  value: string
  sub?: string
  tone?: 'accent'
}) {
  return (
    <div className="card min-w-0 px-field py-related">
      <dt className="type-data-label truncate" title={label}>{label}</dt>
      <dd className={`m-0 mt-1 font-mono text-3xl font-semibold leading-none tabular-nums ${tone === 'accent' ? 'text-accent' : 'text-fg-primary'}`}>
        {value}
      </dd>
      {sub && (
        <dd className="m-0 mt-related truncate text-xs leading-snug text-fg-tertiary" title={sub}>
          {sub}
        </dd>
      )}
    </div>
  )
}

function SmoothControl({ label, alpha, setAlpha, min, max, step, smoothLabel, offLabel, disabled }: {
  label: string
  alpha: number
  setAlpha: (value: number) => void
  min: number
  max: number
  step: number
  smoothLabel: string
  offLabel: string
  disabled?: boolean
}) {
  const inputId = useId()
  return (
    <label htmlFor={inputId} className="flex items-center gap-1.5 text-xs text-fg-tertiary">
      <span>{smoothLabel}</span>
      <input
        id={inputId}
        type="range"
        min={min}
        max={max}
        step={step}
        value={alpha}
        disabled={disabled}
        onChange={(event) => setAlpha(parseFloat(event.target.value))}
        aria-label={`${label} · ${smoothLabel}`}
        className="w-20 accent-[var(--accent)] disabled:cursor-not-allowed disabled:opacity-50"
      />
      <output htmlFor={inputId} className="w-[4ch] text-right font-mono tabular-nums">
        {alpha >= 0.999 ? offLabel : alpha.toFixed(alpha < 0.1 ? 3 : 2)}
      </output>
    </label>
  )
}

function sampleMarks(sample: { path: string; step?: number }): { step: number | null; epoch: number | null } {
  const filename = sample.path.split(/[\\/]/).pop() ?? sample.path
  const epochMatch = /^epoch_(\d+)/i.exec(filename)
  const stepMatch = /^step_(\d+)/i.exec(filename)
  return {
    epoch: epochMatch ? Number(epochMatch[1]) : null,
    step: stepMatch ? Number(stepMatch[1]) : (sample.step ?? null),
  }
}

function SampleViewer({ samples, taskId }: {
  samples: Array<{ path: string; step?: number }>
  taskId: number
}) {
  const { t, i18n } = useTranslation()
  const [active, setActive] = useState(Math.max(0, samples.length - 1))
  const [zoomOpen, setZoomOpen] = useState(false)
  const stripRef = useRef<HTMLDivElement | null>(null)
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([])
  const previousLengthRef = useRef(0)

  useEffect(() => {
    if (samples.length === 0) {
      setActive(0)
      previousLengthRef.current = 0
      return
    }
    if (active >= previousLengthRef.current - 1) setActive(samples.length - 1)
    previousLengthRef.current = samples.length
  }, [active, samples.length])

  const activeIndex = samples.length > 0 ? Math.min(active, samples.length - 1) : 0

  useEffect(() => {
    const target = stripRef.current?.children[activeIndex] as HTMLElement | undefined
    if (!target) return
    const reducedMotion = typeof window.matchMedia === 'function'
      && window.matchMedia('(prefers-reduced-motion: reduce)').matches
    target.scrollIntoView({
      behavior: reducedMotion ? 'auto' : 'smooth',
      block: 'nearest',
      inline: 'nearest',
    })
  }, [activeIndex])

  if (samples.length === 0) {
    return (
      <div className="grid min-h-[220px] flex-1 place-items-center text-sm text-fg-tertiary" role="status">
        {t('monitor.waitingSamples')}
      </div>
    )
  }

  const selectAndFocus = (index: number) => {
    const next = Math.max(0, Math.min(samples.length - 1, index))
    setActive(next)
    optionRefs.current[next]?.focus()
  }

  const handleOptionKeyDown = (event: ReactKeyboardEvent<HTMLButtonElement>, index: number) => {
    let next: number | null = null
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = index - 1
    else if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = index + 1
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = samples.length - 1
    if (next === null) return
    event.preventDefault()
    selectAndFocus(next)
  }

  const current = samples[activeIndex]
  const filename = current.path.split(/[\\/]/).pop() ?? current.path
  const fullUrl = api.sampleImageUrl(filename, taskId)
  const currentMarks = sampleMarks(current)
  const markText = [
    currentMarks.epoch != null ? `${t('monitor.epochShort')} ${currentMarks.epoch.toLocaleString(i18n.language)}` : null,
    currentMarks.step != null ? `${t('monitor.stepShort')} ${currentMarks.step.toLocaleString(i18n.language)}` : null,
  ].filter(Boolean).join(' · ')
  const currentLabel = t('monitor.samplePreviewLabel', {
    index: activeIndex + 1,
    total: samples.length,
    mark: markText || filename,
  })

  return (
    <div className="flex min-h-0 w-full flex-1 flex-col gap-related">
      <div
        ref={stripRef}
        role="listbox"
        aria-label={t('monitor.sampleListLabel')}
        aria-orientation="horizontal"
        className="flex shrink-0 gap-related overflow-x-auto pb-1"
        style={{ scrollbarWidth: 'thin' }}
      >
        {samples.map((sample, index) => {
          const itemFilename = sample.path.split(/[\\/]/).pop() ?? sample.path
          const thumbnailUrl = api.sampleImageUrl(itemFilename, taskId, 128)
          const isActive = index === activeIndex
          const marks = sampleMarks(sample)
          const marksText = [
            marks.epoch != null ? `${t('monitor.epochShort')} ${marks.epoch.toLocaleString(i18n.language)}` : null,
            marks.step != null ? `${t('monitor.stepShort')} ${marks.step.toLocaleString(i18n.language)}` : null,
          ].filter(Boolean).join(' · ') || itemFilename
          const caption = [
            marks.epoch != null ? `${t('monitor.epochShort')}${marks.epoch}` : null,
            marks.step != null ? `${marks.step}` : null,
          ].filter(Boolean).join('·')
          return (
            <button
              key={`${itemFilename}-${index}`}
              ref={(node) => { optionRefs.current[index] = node }}
              type="button"
              role="option"
              aria-selected={isActive}
              aria-label={t('monitor.sampleOptionLabel', {
                index: index + 1,
                total: samples.length,
                mark: marksText,
              })}
              tabIndex={isActive ? 0 : -1}
              onClick={() => setActive(index)}
              onKeyDown={(event) => handleOptionKeyDown(event, index)}
              className="flex shrink-0 flex-col items-center gap-0.5 rounded-sm border-0 bg-transparent p-0 text-fg-tertiary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              title={marksText}
            >
              <span
                className={`block h-16 w-16 overflow-hidden rounded-sm border bg-sunken transition-colors ${isActive ? 'border-accent ring-2 ring-accent-soft' : 'border-subtle hover:border-bold'}`}
              >
                <img src={thumbnailUrl} alt="" loading="lazy" className="block h-full w-full object-cover" />
              </span>
              {caption && (
                <span className={`text-center font-mono text-2xs leading-tight ${isActive ? 'text-fg-primary' : 'text-fg-tertiary'}`}>
                  {caption}
                </span>
              )}
            </button>
          )
        })}
      </div>

      <button
        type="button"
        onClick={() => setZoomOpen(true)}
        aria-label={t('monitor.openSamplePreview', { label: currentLabel })}
        className="relative min-h-[220px] flex-1 overflow-hidden rounded-sm border border-subtle bg-sunken p-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-canvas"
      >
        <img
          key={fullUrl}
          src={fullUrl}
          alt={currentLabel}
          loading="lazy"
          className="absolute inset-0 block h-full w-full cursor-zoom-in object-contain"
        />
        {(currentMarks.epoch != null || currentMarks.step != null) && (
          <span className="absolute bottom-related left-1/2 -translate-x-1/2 rounded-sm border border-subtle bg-surface/90 px-2.5 py-0.5 font-mono text-xs text-fg-secondary">
            {markText}
            <span className="ml-2 text-fg-tertiary">{activeIndex + 1} / {samples.length}</span>
          </span>
        )}
      </button>

      {zoomOpen && (
        <ImagePreviewModal
          src={fullUrl}
          alt={currentLabel}
          caption={[markText, filename].filter(Boolean).join(' · ')}
          index={activeIndex}
          total={samples.length}
          hasPrev={activeIndex > 0}
          hasNext={activeIndex < samples.length - 1}
          onClose={() => setZoomOpen(false)}
          onPrev={() => setActive((index) => Math.max(0, index - 1))}
          onNext={() => setActive((index) => Math.min(samples.length - 1, index + 1))}
        />
      )}
    </div>
  )
}

function ChartPanel({ title, control, children }: {
  title: string
  control: ReactNode
  children: ReactNode
}) {
  const titleId = useId()
  return (
    <Card as="section" padding="sm" className="flex min-h-[140px] max-h-[300px] min-w-0 flex-1 flex-col" aria-labelledby={titleId}>
      <div className="mb-related flex shrink-0 flex-wrap items-center justify-between gap-related border-b border-subtle pb-related">
        <h2 id={titleId} className="type-panel-title">{title}</h2>
        {control}
      </div>
      {children}
    </Card>
  )
}

export default function MonitorDashboard({ taskId, taskStatus }: {
  taskId: number
  taskStatus?: TaskStatus
}) {
  const { t, i18n } = useTranslation()
  const {
    state,
    status,
    streamStatus,
    lastUpdatedAt,
    refreshing,
    error,
    refetch,
  } = useMonitorProgress(taskId)
  const [emaAlpha, setEmaAlpha] = useState(0.02)
  const [lrAlpha, setLrAlpha] = useState(1)
  const [dAlpha, setDAlpha] = useState(1)

  const losses = useMemo(() => state?.losses ?? [], [state?.losses])
  const lrHistory = useMemo(() => state?.lr_history ?? [], [state?.lr_history])
  const optimizerMetricsHistory = useMemo(
    () => state?.optimizer_metrics_history ?? [],
    [state?.optimizer_metrics_history],
  )
  const samples = useMemo(() => state?.samples ?? [], [state?.samples])

  const lossInfo = useMemo(() => {
    if (!losses.length) return null
    const windowSize = Math.min(50, Math.floor(losses.length / 3)) || losses.length
    const raw = losses.map((loss) => loss.loss)
    const recent = raw.slice(-windowSize)
    const previous = raw.length > windowSize ? raw.slice(-windowSize * 2, -windowSize) : null
    const recentAverage = recent.reduce((sum, value) => sum + value, 0) / recent.length
    if (!previous?.length) return { value: recentAverage, delta: null, windowSize }
    const previousAverage = previous.reduce((sum, value) => sum + value, 0) / previous.length
    return { value: recentAverage, delta: recentAverage - previousAverage, windowSize }
  }, [losses])
  const averageLoss = useMemo(
    () => losses.length
      ? losses.reduce((sum, loss) => sum + loss.loss, 0) / losses.length
      : null,
    [losses],
  )
  const lastLearningRate = lrHistory.length ? lrHistory[lrHistory.length - 1].lr : null
  const lastOptimizerMetrics = optimizerMetricsHistory.length
    ? optimizerMetricsHistory[optimizerMetricsHistory.length - 1]
    : null
  const lastD = lastOptimizerMetrics?.d ?? null

  const lossSeries = useMemo(
    () => losses.map((loss) => ({ step: loss.step, value: loss.loss })),
    [losses],
  )
  const learningRateSeries = useMemo(
    () => lrHistory.map((item) => ({ step: item.step, value: item.lr })),
    [lrHistory],
  )
  const dSeries = useMemo(
    () => optimizerMetricsHistory
      .map((metric) => ({ step: metric.step, d: metric.d }))
      .filter((metric): metric is { step: number; d: number } => typeof metric.d === 'number')
      .map((metric) => ({ step: metric.step, value: metric.d })),
    [optimizerMetricsHistory],
  )

  const chartSummaries = useMemo(() => {
    const describe = (
      series: Array<{ step: number; value: number }>,
      formatter: (value: number) => string,
    ) => {
      if (!series.length) return t('monitor.chartSummaryEmpty')
      let minimum = series[0].value
      let maximum = series[0].value
      for (const point of series) {
        minimum = Math.min(minimum, point.value)
        maximum = Math.max(maximum, point.value)
      }
      const current = series[series.length - 1]
      return t('monitor.chartSummary', {
        count: series.length.toLocaleString(i18n.language),
        start: series[0].step.toLocaleString(i18n.language),
        end: current.step.toLocaleString(i18n.language),
        current: formatter(current.value),
        min: formatter(minimum),
        max: formatter(maximum),
      })
    }
    return {
      loss: describe(lossSeries, (value) => value.toFixed(4)),
      learningRate: describe(learningRateSeries, formatLearningRate),
      optimizerD: describe(dSeries, formatMetric),
    }
  }, [dSeries, i18n.language, learningRateSeries, lossSeries, t])

  if (!state) {
    if (status === 'error') {
      return (
        <div className="p-page">
          <Alert
            tone="danger"
            title={t('monitor.dataErrorTitle')}
            action={(
              <Button size="sm" onClick={() => void refetch()} loading={refreshing}>
                {t('common.retry')}
              </Button>
            )}
          >
            <span title={error ?? undefined}>{t('monitor.dataError')}</span>
          </Alert>
        </div>
      )
    }
    if (status === 'unavailable') {
      const activeTask = taskStatus === 'running' || taskStatus === 'pending' || taskStatus === 'scheduled'
      return (
        <EmptyState
          embedded
          className="h-full"
          title={t(activeTask ? 'monitor.waitingForDataTitle' : 'monitor.noDataTitle')}
          description={t(activeTask ? 'monitor.waitingForDataDescription' : 'monitor.noDataDescription')}
          action={(
            <Button size="sm" onClick={() => void refetch()} loading={refreshing}>
              {t('monitor.readAgain')}
            </Button>
          )}
        />
      )
    }
    return (
      <div className="grid h-[200px] place-items-center text-sm text-fg-tertiary" role="status">
        {t('monitor.loadingData')}
      </div>
    )
  }

  const number = (value: number) => value.toLocaleString(i18n.language)
  const units = {
    hour: t('monitor.duration.hourShort'),
    minute: t('monitor.duration.minuteShort'),
    second: t('monitor.duration.secondShort'),
  }
  const step = state.step ?? 0
  const totalSteps = state.total_steps ?? 0
  const epoch = state.epoch
  const totalEpochs = state.total_epochs
  const speed = state.speed
  const etaSeconds = speed && speed > 0 && totalSteps > step ? (totalSteps - step) / speed : 0
  const eta = etaSeconds > 0 ? formatDuration(etaSeconds, units) : '--'
  const elapsedSeconds = state.start_time ? Date.now() / 1000 - state.start_time : 0
  const elapsed = elapsedSeconds > 0 ? formatDuration(elapsedSeconds, units) : '--'
  const progress = totalSteps > 0 ? Math.min(100, (step / totalSteps) * 100) : 0

  const terminal = taskStatus === 'done' || taskStatus === 'failed' || taskStatus === 'canceled'
  const evidenceState = terminal
    ? 'historical'
    : streamStatus === 'live'
      ? 'live'
      : streamStatus === 'reconnecting'
        ? 'reconnecting'
        : 'synced'
  const lastUpdatedLabel = lastUpdatedAt == null
    ? null
    : new Intl.DateTimeFormat(i18n.language, {
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      }).format(lastUpdatedAt)
  const vramUsed = state.vram_used_gb
  const vramTotal = state.vram_total_gb
  const smoothingLabel = t('monitor.smoothing')
  const smoothingOffLabel = t('monitor.smoothingOff')

  return (
    <div className="flex h-full min-h-0 flex-col gap-field overflow-y-auto p-section">
      {(error || streamStatus === 'reconnecting') && (
        <Alert
          tone="warning"
          size="sm"
          title={t(error ? 'monitor.snapshotStaleTitle' : 'monitor.reconnectingTitle')}
          action={(
            <Button size="xs" onClick={() => void refetch()} loading={refreshing}>
              {t('monitor.readAgain')}
            </Button>
          )}
        >
          <span title={error ?? undefined}>
            {t(error ? 'monitor.snapshotStaleDescription' : 'monitor.reconnectingDescription')}
          </span>
        </Alert>
      )}

      <section
        aria-label={t('monitor.statusSummary')}
        className="flex shrink-0 flex-wrap items-center gap-field border-y border-subtle py-related text-xs text-fg-tertiary"
      >
        <Badge
          size="sm"
          tone={evidenceState === 'live' ? 'success' : evidenceState === 'reconnecting' ? 'warning' : 'neutral'}
          active={evidenceState === 'live'}
        >
          {t(`monitor.evidenceStatus.${evidenceState}`)}
        </Badge>
        {lastUpdatedLabel && <span>{t('monitor.lastUpdated', { time: lastUpdatedLabel })}</span>}
        {totalSteps > 0 && (
          <div className="ml-auto flex items-center gap-section font-mono tabular-nums">
            <span className="whitespace-nowrap">
              {t('monitor.progressSteps', { current: number(step), total: number(totalSteps) })}
            </span>
            <ProgressBar
              className="w-[clamp(16rem,36vw,42rem)] shrink-0"
              size="xs"
              label={t('monitor.trainingProgress')}
              value={step}
              max={totalSteps}
              valueText={`${progress.toFixed(1)}%`}
            />
            <span className="whitespace-nowrap">{progress.toFixed(1)}%</span>
            <span className="whitespace-nowrap text-fg-secondary">
              {t('monitor.metric.elapsed')} {elapsed}
            </span>
            <span className="whitespace-nowrap text-fg-secondary">
              {t('monitor.metric.eta')} {eta}
            </span>
          </div>
        )}
      </section>

      <dl className="m-0 grid shrink-0 grid-cols-6 gap-related">
        <StatCard
          label={t('monitor.metric.step')}
          value={number(step)}
          sub={[
            totalSteps > 0 ? t('monitor.ofTotal', { total: number(totalSteps) }) : t('monitor.totalUnknown'),
            epoch == null
              ? null
              : `${t('monitor.metric.epoch')} ${number(epoch)}${totalEpochs == null ? '' : ` / ${number(totalEpochs)}`}`,
          ].filter(Boolean).join(' · ')}
          tone="accent"
        />
        <StatCard
          label={t('monitor.metric.recentLoss')}
          value={lossInfo ? lossInfo.value.toFixed(4) : '--'}
          sub={lossInfo
            ? t(lossInfo.delta == null ? 'monitor.recentAverage' : 'monitor.recentDelta', {
                count: lossInfo.windowSize,
                delta: lossInfo.delta == null ? '' : `${lossInfo.delta >= 0 ? '+' : ''}${lossInfo.delta.toFixed(4)}`,
              })
            : t('monitor.awaitingMetric')}
        />
        <StatCard
          label={t('monitor.metric.averageLoss')}
          value={averageLoss == null ? '--' : averageLoss.toFixed(4)}
          sub={losses.length ? t('monitor.rawMeanPoints', { count: number(losses.length) }) : t('monitor.awaitingMetric')}
        />
        <StatCard
          label={t('monitor.metric.learningRate')}
          value={formatLearningRate(lastLearningRate)}
          sub={lastD == null
            ? t(lrHistory.length ? 'monitor.actualLearningRate' : 'monitor.awaitingMetric')
            : `d ${formatMetric(lastD)} · ${t('monitor.actualLearningRate')}`}
        />
        <StatCard
          label={t('monitor.metric.vram')}
          value={vramUsed == null ? '--' : `${vramUsed.toFixed(1)} GB`}
          sub={vramUsed != null && vramTotal != null
            ? t('monitor.vramOfTotal', {
                total: vramTotal.toFixed(1),
                percent: ((vramUsed / vramTotal) * 100).toFixed(0),
              })
            : t('monitor.notReported')}
        />
        <StatCard
          label={t('monitor.metric.eta')}
          value={eta}
          sub={speed == null ? t('monitor.notReported') : `${speed.toFixed(2)} it/s · ${t('monitor.iterationSpeed')}`}
        />
      </dl>

      <div className="grid min-h-[440px] flex-1 grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)] gap-related">
        <Card as="section" padding="none" className="flex min-h-[420px] min-w-0 flex-col overflow-hidden">
          <div className="flex shrink-0 items-center justify-between border-b border-subtle px-related py-related">
            <h2 className="type-panel-title">{t('monitor.samplesTitle')}</h2>
            <span className="font-mono text-xs text-fg-tertiary tabular-nums">
              {t('monitor.sampleCount', { count: number(samples.length) })}
            </span>
          </div>
          <div className="flex min-h-0 flex-1 flex-col p-related">
            <SampleViewer key={taskId} samples={samples} taskId={taskId} />
          </div>
        </Card>

        <div className="flex min-h-0 min-w-0 flex-col gap-related">
          <ChartPanel
            title={t('monitor.chart.loss')}
            control={(
              <SmoothControl
                label={t('monitor.chart.loss')}
                alpha={emaAlpha}
                setAlpha={setEmaAlpha}
                min={0.001}
                max={0.3}
                step={0.001}
                smoothLabel={smoothingLabel}
                offLabel={smoothingOffLabel}
                disabled={lossSeries.length === 0}
              />
            )}
          >
            <SeriesChart
              data={lossSeries}
              rawColor="var(--fg-tertiary)"
              smoothColor="var(--accent)"
              fillColor="var(--accent-soft)"
              emaAlpha={emaAlpha}
              yFormat={(value) => value.toFixed(4)}
              minHeight={110}
              ariaLabel={t('monitor.chart.loss')}
              summary={chartSummaries.loss}
              emptyLabel={t('monitor.chartWaiting')}
            />
          </ChartPanel>

          <ChartPanel
            title={t('monitor.chart.learningRate')}
            control={(
              <SmoothControl
                label={t('monitor.chart.learningRate')}
                alpha={lrAlpha}
                setAlpha={setLrAlpha}
                min={0.005}
                max={1}
                step={0.005}
                smoothLabel={smoothingLabel}
                offLabel={smoothingOffLabel}
                disabled={learningRateSeries.length === 0}
              />
            )}
          >
            <SeriesChart
              data={learningRateSeries}
              rawColor="var(--warn)"
              smoothColor="var(--warn)"
              emaAlpha={lrAlpha}
              yFormat={formatLearningRate}
              minHeight={110}
              ariaLabel={t('monitor.chart.learningRate')}
              summary={chartSummaries.learningRate}
              emptyLabel={t('monitor.chartWaiting')}
            />
          </ChartPanel>

          <ChartPanel
            title={t('monitor.chart.optimizerD')}
            control={(
              <SmoothControl
                label={t('monitor.chart.optimizerD')}
                alpha={dAlpha}
                setAlpha={setDAlpha}
                min={0.005}
                max={1}
                step={0.005}
                smoothLabel={smoothingLabel}
                offLabel={smoothingOffLabel}
                disabled={dSeries.length === 0}
              />
            )}
          >
            <SeriesChart
              data={dSeries}
              rawColor="var(--accent)"
              smoothColor="var(--accent)"
              emaAlpha={dAlpha}
              yFormat={formatMetric}
              minHeight={110}
              ariaLabel={t('monitor.chart.optimizerD')}
              summary={chartSummaries.optimizerD}
              emptyLabel={t('monitor.chartWaiting')}
            />
          </ChartPanel>
        </div>
      </div>
    </div>
  )
}
