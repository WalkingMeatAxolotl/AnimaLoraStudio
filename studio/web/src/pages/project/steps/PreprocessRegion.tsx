import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type CropWorkspaceItem,
  type ProjectDetail,
  type RegionAnnotation,
  type RegionBox,
  type Version,
} from '../../../api/client'
import RegionCanvas from '../../../components/preprocess/RegionCanvas'
import PreprocessToolsBar from '../../../components/preprocess/PreprocessToolsBar'
import StepShell from '../../../components/StepShell'
import { useToast } from '../../../components/Toast'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

export default function PreprocessRegionPage() {
  const { t } = useTranslation()
  const { project, activeVersion } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const vid = activeVersion?.id ?? 0
  const [images, setImages] = useState<CropWorkspaceItem[]>([])
  const [activeName, setActiveName] = useState('')
  const [box, setBox] = useState<RegionBox | null>(null)
  const [label, setLabel] = useState('face')
  const [classWord, setClassWord] = useState('1girl')
  const [caption, setCaption] = useState('')
  const [weight, setWeight] = useState(1)
  const [loading, setLoading] = useState(false)

  const loadWorkspace = useCallback(async () => {
    if (!vid) return
    const result = await api.listCropWorkspaceTrain(project.id, vid)
    setImages(result.images)
    setActiveName((current) => current || result.images[0]?.name || '')
  }, [project.id, vid])

  useEffect(() => { void loadWorkspace().catch((e) => toast(String(e), 'error')) }, [loadWorkspace, toast])
  const active = useMemo(() => images.find((item) => item.name === activeName) ?? null, [images, activeName])

  useEffect(() => {
    let cancelled = false
    setBox(null); setLabel('face'); setCaption(''); setWeight(1)
    if (!active || active.region_mtime == null) {
      setLoading(false)
      return () => { cancelled = true }
    }
    setLoading(true)
    void api.getRegionTrain(project.id, vid, active.name)
      .then((doc) => {
        if (cancelled) return
        const region = doc.regions[0]
        if (!region) return
        setBox(region.box); setLabel(region.label); setClassWord(region.class_word || '1girl')
        setCaption(region.caption); setWeight(region.weight)
      })
      .catch((e) => { if (!cancelled) toast(String(e), 'error') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [active, project.id, vid, toast])

  if (!activeVersion) return <p className="p-6 text-fg-tertiary">{t('preprocessRegion.noVersion')}</p>

  const save = async () => {
    if (!active || !box) return
    const annotation: RegionAnnotation = {
      version: 1,
      image_size: { w: active.w, h: active.h },
      regions: [{ id: 'primary', label, class_word: classWord, caption, weight, box }],
    }
    try {
      await api.saveRegionTrain(project.id, vid, active.name, annotation)
      toast(t('preprocessRegion.saved'), 'success')
      await loadWorkspace()
    } catch (e) { toast(String(e), 'error') }
  }
  const clear = async () => {
    if (!active) return
    try {
      await api.deleteRegionTrain(project.id, vid, active.name)
      setBox(null); setCaption('')
      toast(t('preprocessRegion.cleared'), 'success')
      await loadWorkspace()
    } catch (e) { toast(String(e), 'error') }
  }
  const annotated = images.filter((item) => item.region_mtime != null).length

  return (
    <StepShell
      idx={2}
      title={t('preprocessRegion.title')}
      subtitle={t('preprocessRegion.subtitle')}
      belowHeader={<PreprocessToolsBar current="region" projectId={project.id} versionId={vid} />}
      actions={<span className="badge badge-neutral">{annotated} / {images.length}</span>}
    >
      <div className="flex flex-1 min-h-0 gap-3">
        <aside className="w-56 shrink-0 rounded-md border border-subtle bg-surface overflow-y-auto p-2 space-y-1">
          {images.map((item) => (
            <button
              key={item.name}
              type="button"
              onClick={() => setActiveName(item.name)}
              className={`w-full flex items-center gap-2 rounded px-2 py-1.5 text-left text-xs ${item.name === activeName ? 'bg-accent-soft text-accent' : 'hover:bg-overlay'}`}
            >
              <span className={item.region_mtime != null ? 'text-ok' : 'text-fg-disabled'}>{item.region_mtime != null ? '●' : '○'}</span>
              <span className="truncate">{item.name}</span>
            </button>
          ))}
        </aside>

        <section className="flex-1 min-w-0 rounded-md border border-subtle bg-surface overflow-hidden p-2">
          {active ? (
            <RegionCanvas
              key={active.name}
              src={api.versionThumbUrl(project.id, vid, 'train', active.name.split('/').pop() || active.name, active.name.includes('/') ? active.name.split('/')[0] : '', 0)}
              width={active.w}
              height={active.h}
              box={box}
              onChange={setBox}
            />
          ) : <div className="h-full grid place-items-center text-fg-tertiary">{t('preprocessRegion.empty')}</div>}
        </section>

        <aside className="w-80 shrink-0 rounded-md border border-subtle bg-surface p-4 space-y-4 overflow-y-auto">
          <p className="text-sm text-fg-secondary">{loading ? t('common.loading') : t('preprocessRegion.drawHint')}</p>
          <label className="block text-sm">
            <span className="caption">{t('preprocessRegion.label')}</span>
            <input className="input mt-1 w-full" value={label} onChange={(e) => setLabel(e.target.value)} />
          </label>
          <label className="block text-sm">
            <span className="caption">{t('preprocessRegion.classWord')}</span>
            <input className="input mt-1 w-full" value={classWord} onChange={(e) => setClassWord(e.target.value)} />
          </label>
          <label className="block text-sm">
            <span className="caption">{t('preprocessRegion.caption')}</span>
            <textarea className="input mt-1 w-full min-h-28" value={caption} onChange={(e) => setCaption(e.target.value)} />
          </label>
          <label className="block text-sm">
            <span className="caption">{t('preprocessRegion.weight')}</span>
            <input type="number" min={0.1} max={10} step={0.1} className="input mt-1 w-full" value={weight} onChange={(e) => setWeight(Number(e.target.value))} />
          </label>
          {box && <code className="block text-xs text-fg-tertiary">x={box.x.toFixed(3)} y={box.y.toFixed(3)} w={box.w.toFixed(3)} h={box.h.toFixed(3)}</code>}
          <div className="flex gap-2">
            <button type="button" className="btn btn-primary flex-1" disabled={!active || !box} onClick={() => void save()}>{t('common.save')}</button>
            <button type="button" className="btn btn-secondary" disabled={!active} onClick={() => void clear()}>{t('common.clear')}</button>
          </div>
        </aside>
      </div>
    </StepShell>
  )
}
