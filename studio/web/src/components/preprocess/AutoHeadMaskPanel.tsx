import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

import {
  api,
  type HeadMaskProposalImage,
  type Job,
  type ModelsCatalog,
} from '../../api/client'
import { useToast } from '../Toast'
import { useEventStream } from '../../lib/useEventStream'

export interface AutoHeadMaskState {
  images: HeadMaskProposalImage[]
  selections: Record<string, string[]>
}

interface Props {
  projectId: number
  versionId: number
  activeName: string | null
  unsavedCount: number
  onStateChange: (state: AutoHeadMaskState | null) => void
  onShowUndetected: (names: string[]) => void
  onWorkspaceChanged: () => Promise<void>
}

function isHeadMaskJob(job: Job | null): boolean {
  if (!job) return false
  let params = job.params_decoded
  if (!params && typeof job.params === 'string') {
    try { params = JSON.parse(job.params) as Record<string, unknown> } catch { return false }
  }
  return params?.stage === 'head_mask'
}

export default function AutoHeadMaskPanel({
  projectId,
  versionId,
  activeName,
  unsavedCount,
  onStateChange,
  onShowUndetected,
  onWorkspaceChanged,
}: Props) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [catalog, setCatalog] = useState<ModelsCatalog | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [proposal, setProposal] = useState<Awaited<ReturnType<typeof api.getHeadMaskProposals>> | null>(null)
  const [selections, setSelections] = useState<Record<string, string[]>>({})
  const [busy, setBusy] = useState(false)
  const [downloadRequested, setDownloadRequested] = useState(false)
  const [progress, setProgress] = useState({ done: 0, total: 0, heads: 0 })
  const [params, setParams] = useState({
    confidence: 0.413,
    iou_threshold: 0.7,
    padding_ratio: 0.10,
    feather_ratio: 0.03,
  })
  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = job?.id ?? null

  const reloadCatalog = useCallback(() => {
    void api.getModelsCatalog().then(setCatalog).catch(() => setCatalog(null))
  }, [])

  const loadProposal = useCallback(async (jobId: number) => {
    const result = await api.getHeadMaskProposals(projectId, versionId, jobId)
    setProposal(result)
    setSelections((previous) => {
      const hasMatchingState = Object.keys(previous).some((name) =>
        result.images.some((image) => image.name === name),
      )
      if (hasMatchingState) return previous
      return Object.fromEntries(
        result.images.map((image) => [image.name, image.regions.map((region) => region.id)]),
      )
    })
  }, [projectId, versionId])

  useEffect(() => {
    reloadCatalog()
    void api.getPreprocessStatusTrain(projectId, versionId).then((status) => {
      if (!isHeadMaskJob(status.job)) return
      setJob(status.job)
      if (status.job?.status === 'done') void loadProposal(status.job.id)
    }).catch(() => {})
  }, [projectId, versionId, reloadCatalog, loadProposal])

  useEffect(() => {
    if (!downloadRequested || catalog?.head_detector?.valid) return
    const timer = window.setInterval(reloadCatalog, 1000)
    return () => window.clearInterval(timer)
  }, [downloadRequested, catalog?.head_detector?.valid, reloadCatalog])

  // SSE is the fast path. Polling is the recovery path for a sleeping browser,
  // a proxy that buffered events, or a reconnect that missed the terminal event.
  useEffect(() => {
    if (!job || (job.status !== 'pending' && job.status !== 'running')) return
    let active = true
    const poll = async () => {
      try {
        const latest = await api.getJob(job.id)
        if (!active) return
        setJob(latest)
        if (latest.status === 'done') await loadProposal(latest.id)
        else if (latest.status === 'failed') toast(t('preprocessInpaint.headMask.detectFailed'), 'error')
        else if (latest.status === 'canceled') toast(t('preprocessInpaint.headMask.detectCanceled'), 'info')
      } catch {
        // A transient status read must not replace a still-running job with an error.
      }
    }
    const timer = window.setInterval(() => { void poll() }, 1500)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [job, loadProposal, t, toast])

  useEffect(() => {
    onStateChange(proposal ? { images: proposal.images, selections } : null)
  }, [proposal, selections, onStateChange])

  useEventStream((event) => {
    if (event.type === 'model_download_changed' && event.key === 'head_detector') {
      reloadCatalog()
    }
    const currentJobId = jobIdRef.current
    if (!currentJobId || event.job_id !== currentJobId) return
    if (event.type === 'head_mask_progress') {
      setProgress((current) => ({
        done: Number(event.idx ?? current.done),
        total: Number(event.total ?? current.total),
        heads: current.heads + (event.status === 'done' ? Number(event.detections ?? 0) : 0),
      }))
    } else if (event.type === 'job_state_changed') {
      const status = String(event.status) as Job['status']
      setJob((current) => current ? { ...current, status } : current)
      if (status === 'done') {
        void loadProposal(currentJobId).catch((error) => toast(String(error), 'error'))
      } else if (status === 'failed') {
        toast(t('preprocessInpaint.headMask.detectFailed'), 'error')
      } else if (status === 'canceled') {
        toast(t('preprocessInpaint.headMask.detectCanceled'), 'info')
      }
    }
  })

  const startDetection = async (scope: 'all' | 'selected') => {
    if (unsavedCount > 0) {
      toast(t('preprocessInpaint.headMask.saveFirst', { n: unsavedCount }), 'error')
      return
    }
    if (!catalog?.head_detector?.valid) {
      toast(t('preprocessInpaint.headMask.modelRequired'), 'error')
      return
    }
    if (scope === 'selected' && !activeName) return
    setBusy(true)
    setProposal(null)
    setSelections({})
    setProgress({ done: 0, total: scope === 'selected' ? 1 : 0, heads: 0 })
    try {
      const next = await api.startHeadMaskDetection(projectId, versionId, {
        scope,
        ...(scope === 'selected' && activeName ? { filenames: [activeName] } : {}),
        ...params,
      })
      setJob(next)
      toast(t('preprocessInpaint.headMask.detectStarted', { id: next.id }), 'success')
    } catch (error) {
      toast(String(error), 'error')
    } finally {
      setBusy(false)
    }
  }

  const downloadModel = async () => {
    setDownloadRequested(true)
    try {
      await api.startModelDownload({ model_id: 'head_detector' })
      toast(t('preprocessInpaint.headMask.downloadStarted'), 'success')
      reloadCatalog()
    } catch (error) {
      toast(String(error), 'error')
    }
  }

  const activeProposal = proposal?.images.find((image) => image.name === activeName) ?? null
  const selectedCount = Object.values(selections).reduce((total, ids) => total + ids.length, 0)
  const totalHeads = proposal?.images.reduce((total, image) => total + image.regions.length, 0) ?? 0
  const undetected = useMemo(
    () => proposal?.images.filter((image) => image.regions.length === 0).map((image) => image.name) ?? [],
    [proposal],
  )

  const setActiveSelection = (all: boolean) => {
    if (!activeProposal) return
    setSelections((current) => ({
      ...current,
      [activeProposal.name]: all ? activeProposal.regions.map((region) => region.id) : [],
    }))
  }

  const toggleRegion = (regionId: string) => {
    if (!activeProposal) return
    setSelections((current) => {
      const existing = current[activeProposal.name] ?? []
      return {
        ...current,
        [activeProposal.name]: existing.includes(regionId)
          ? existing.filter((id) => id !== regionId)
          : [...existing, regionId],
      }
    })
  }

  const apply = async () => {
    if (!proposal || selectedCount === 0) return
    setBusy(true)
    try {
      const result = await api.applyHeadMaskProposals(
        projectId, versionId, proposal.job_id, selections,
      )
      toast(t('preprocessInpaint.headMask.applied', { n: result.applied }), 'success')
      await onWorkspaceChanged()
      await loadProposal(proposal.job_id)
    } catch (error) {
      toast(String(error), 'error')
    } finally {
      setBusy(false)
    }
  }

  const undoApply = async () => {
    if (!proposal) return
    setBusy(true)
    try {
      const result = await api.undoHeadMaskApply(projectId, versionId, proposal.job_id)
      toast(t('preprocessInpaint.headMask.undone', { n: result.undone }), 'success')
      await onWorkspaceChanged()
      await loadProposal(proposal.job_id)
    } catch (error) {
      toast(String(error), 'error')
    } finally {
      setBusy(false)
    }
  }

  const running = job?.status === 'pending' || job?.status === 'running'
  const modelReady = catalog?.head_detector?.valid === true

  return (
    <div className="flex flex-col gap-2 border-t border-subtle pt-2 mt-1" data-testid="auto-head-mask-panel">
      <div className="flex items-center justify-between gap-2">
        <h4 className="caption">{t('preprocessInpaint.headMask.title')}</h4>
        <span className={`text-[10px] ${modelReady ? 'text-ok' : 'text-warn'}`}>
          {modelReady
            ? t('preprocessInpaint.headMask.modelReady')
            : t('preprocessInpaint.headMask.modelMissing')}
        </span>
      </div>
      <p className="text-[11px] text-fg-tertiary leading-relaxed m-0">
        {t('preprocessInpaint.headMask.boundary')}
      </p>

      {!modelReady && (
        <button
          type="button"
          className="btn btn-secondary btn-sm justify-center"
          disabled={downloadRequested && catalog?.downloads.head_detector?.status === 'running'}
          onClick={() => void downloadModel()}
        >
          {catalog?.downloads.head_detector?.status === 'running'
            ? t('preprocessInpaint.headMask.downloading')
            : t('preprocessInpaint.headMask.downloadModel')}
        </button>
      )}

      <details className="text-[11px]">
        <summary className="cursor-pointer text-fg-secondary">
          {t('preprocessInpaint.headMask.parameters')}
        </summary>
        <div className="grid grid-cols-2 gap-1.5 mt-1.5">
          {(['confidence', 'iou_threshold', 'padding_ratio', 'feather_ratio'] as const).map((key) => (
            <label key={key} className="flex flex-col gap-0.5 text-fg-tertiary">
              {t(`preprocessInpaint.headMask.${key}`)}
              <input
                className="input input-mono text-xs"
                type="number"
                min={0} max={key === 'feather_ratio' ? 0.5 : 1}
                step={0.01}
                value={params[key]}
                onChange={(event) => setParams((current) => ({
                  ...current,
                  [key]: Number(event.target.value),
                }))}
              />
            </label>
          ))}
        </div>
      </details>

      <div className="grid grid-cols-2 gap-1.5">
        <button type="button" className="btn btn-secondary btn-sm justify-center"
          disabled={busy || running || !modelReady}
          onClick={() => void startDetection('all')}>
          {t('preprocessInpaint.headMask.detectAll')}
        </button>
        <button type="button" className="btn btn-secondary btn-sm justify-center"
          disabled={busy || running || !modelReady || !activeName}
          onClick={() => void startDetection('selected')}>
          {t('preprocessInpaint.headMask.detectCurrent')}
        </button>
      </div>

      {running && (
        <div className="rounded-sm bg-overlay px-2 py-1.5 text-[11px] text-fg-secondary">
          {t('preprocessInpaint.headMask.progress', progress)}
          <button type="button" className="ml-2 text-err underline"
            onClick={() => job && void api.cancelJob(job.id)}>
            {t('common.cancel')}
          </button>
        </div>
      )}

      {proposal && (
        <>
          <div className="flex items-center gap-1.5 text-[11px] text-fg-secondary flex-wrap">
            <span>{t('preprocessInpaint.headMask.summary', {
              images: proposal.images.length, heads: totalHeads, selected: selectedCount,
            })}</span>
            <button type="button" className="underline text-accent"
              onClick={() => onShowUndetected(undetected)}>
              {t('preprocessInpaint.headMask.showUndetected', { n: undetected.length })}
            </button>
          </div>
          {proposal.stale_count > 0 && (
            <p className="m-0 text-[11px] text-err">
              {t('preprocessInpaint.headMask.stale', { n: proposal.stale_count })}
            </p>
          )}
          <div className="flex items-center gap-1">
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setActiveSelection(true)}>
              {t('preprocessInpaint.headMask.selectCurrent')}
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setActiveSelection(false)}>
              {t('preprocessInpaint.headMask.clearCurrent')}
            </button>
          </div>
          <div className="flex flex-col gap-1 max-h-28 overflow-auto">
            {!activeProposal || activeProposal.regions.length === 0 ? (
              <span className="text-[11px] text-fg-tertiary">
                {t('preprocessInpaint.headMask.noneCurrent')}
              </span>
            ) : activeProposal.regions.map((region, index) => (
              <label key={region.id} className="flex items-center gap-1.5 text-[11px]">
                <input type="checkbox"
                  checked={(selections[activeProposal.name] ?? []).includes(region.id)}
                  onChange={() => toggleRegion(region.id)} />
                <span>{t('preprocessInpaint.headMask.region', {
                  n: index + 1, score: Math.round(region.score * 100),
                })}</span>
              </label>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            <button type="button" className="btn btn-primary btn-sm justify-center"
              disabled={busy || selectedCount === 0 || proposal.stale_count > 0}
              onClick={() => void apply()}>
              {t('preprocessInpaint.headMask.applySelected', { n: selectedCount })}
            </button>
            <button type="button" className="btn btn-ghost btn-sm justify-center"
              disabled={busy || !proposal.undo_available}
              onClick={() => void undoApply()}>
              {t('preprocessInpaint.headMask.undoApply')}
            </button>
          </div>
        </>
      )}
    </div>
  )
}
