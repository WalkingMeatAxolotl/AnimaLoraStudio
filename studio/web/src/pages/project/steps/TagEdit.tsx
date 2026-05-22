import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type CommitItem,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import BulkActionBar from '../../../components/BulkActionBar'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import SaveBar from '../../../components/SaveBar'
import StepShell from '../../../components/StepShell'
import TagEditor from '../../../components/TagEditor'
import TagStatsPanel from '../../../components/TagStatsPanel'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

const keyOf = (folder: string, name: string) => `${folder}/${name}`

interface CaptionMeta {
  folder: string
  name: string
  format: 'txt' | 'json' | 'none'
}

function arraysEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

export default function TagEditPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const versionId = activeVersion?.id ?? null

  const [cache, setCache] = useState<Map<string, string[]>>(new Map())
  const [initial, setInitial] = useState<Map<string, string[]>>(new Map())
  const [meta, setMeta] = useState<Map<string, CaptionMeta>>(new Map())
  const [keys, setKeys] = useState<string[]>([])

  const [activeKey, setActiveKey] = useState<string>('')
  const [sel, setSel] = useState<Set<string>>(new Set())
  const [anchor, setAnchor] = useState<string | null>(null)
  // '' = 全部；否则限定到该 folder（1_data / 2_data ...）。命名特意区分于下面
  // editing 时用的 `activeFolder`（那个是当前编辑图所在 folder，纯展示）。
  const [folderFilter, setFolderFilter] = useState<string>('')
  const [exporting, setExporting] = useState(false)

  const reloadCache = useCallback(async () => {
    if (versionId == null) return
    try {
      const r = await api.listCaptionsFull(project.id, versionId)
      const c = new Map<string, string[]>()
      const m = new Map<string, CaptionMeta>()
      const ks: string[] = []
      for (const it of r.items) {
        const k = keyOf(it.folder, it.name)
        c.set(k, it.tags)
        m.set(k, { folder: it.folder, name: it.name, format: it.format })
        ks.push(k)
      }
      setCache(c); setInitial(new Map(c)); setMeta(m); setKeys(ks)
    } catch (e) { toast(String(e), 'error') }
  }, [project.id, versionId, toast])

  useEffect(() => { void reloadCache() }, [reloadCache])

  useEventStream((evt) => {
    if (
      evt.type === 'version_state_changed' &&
      versionId != null &&
      evt.version_id === versionId
    ) {
      void reloadCache(); void reload()
    } else if (
      evt.type === 'job_state_changed' &&
      evt.project_id === project.id &&
      (evt.status === 'done' || evt.status === 'failed')
    ) {
      void reloadCache(); void reload()
    } else if (
      // train.zip 打包结果 SSE —— <a> 直链发完后端 publish ready/_failed,这里清
      // app-side "打包中..." 状态 + 失败弹 toast。和 Layout.tsx 的导出共用同一对事件,
      // 两个页面同时打开时各自只响应 project_id+version_id 匹配的那一条。
      (evt.type === 'version_train_zip_ready' || evt.type === 'version_train_zip_failed') &&
      evt.project_id === project.id &&
      versionId != null &&
      evt.version_id === versionId
    ) {
      setExporting(false)
      if (evt.type === 'version_train_zip_failed') {
        const err = typeof evt.error === 'string' ? evt.error : '?'
        toast(t('tagEdit.downloadFailed', { error: err }), 'error')
      }
    }
  })

  // 兜底：SSE 事件丢失时 60s 强制清 exporting,不让按钮卡死。
  useEffect(() => {
    if (!exporting) return
    const tid = window.setTimeout(() => setExporting(false), 60_000)
    return () => window.clearTimeout(tid)
  }, [exporting])

  const dirtyKeys = useMemo(() => {
    const out: string[] = []
    for (const k of keys) {
      const cur = cache.get(k) ?? []
      const ini = initial.get(k) ?? []
      if (!arraysEqual(cur, ini)) out.push(k)
    }
    return out
  }, [cache, initial, keys])
  const dirty = dirtyKeys.length > 0

  useEffect(() => {
    if (!dirty) return
    const handler = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = '' }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [dirty])

  // folder 列表 + 每个 folder 的原始张数（不受 filterTag 影响，让 tab 数字稳定
  // 不抖动 — 同 Preprocess chip 风格）。单 folder 项目时 UI 不显示 tabs。
  const folderNames = useMemo(() => {
    const set = new Set<string>()
    for (const m of meta.values()) set.add(m.folder)
    return Array.from(set).sort()
  }, [meta])
  const folderCounts = useMemo(() => {
    const c = new Map<string, number>()
    for (const m of meta.values()) c.set(m.folder, (c.get(m.folder) ?? 0) + 1)
    return c
  }, [meta])

  const filteredKeys = useMemo(() => {
    if (!folderFilter) return keys
    return keys.filter((k) => meta.get(k)?.folder === folderFilter)
  }, [keys, meta, folderFilter])

  const captionItems = useMemo(
    () =>
      filteredKeys.map((k) => {
        const m = meta.get(k)!
        const tags = cache.get(k) ?? []
        return {
          name: k,
          thumbUrl:
            activeVersion != null
              ? api.versionThumbUrl(project.id, activeVersion.id, 'train', m.name, m.folder)
              : '',
          meta: tags.slice(0, 5).join(', '),
        }
      }),
    [filteredKeys, meta, cache, project.id, activeVersion]
  )

  const selectedKeys = useMemo(
    () => filteredKeys.filter((k) => sel.has(k)),
    [filteredKeys, sel]
  )
  const navKeys = selectedKeys.length > 0 ? selectedKeys : filteredKeys
  const activeIndex = activeKey ? navKeys.indexOf(activeKey) : -1

  const tagSuggestions = useMemo(() => {
    const set = new Set<string>()
    for (const tags of cache.values()) for (const tag of tags) set.add(tag)
    return Array.from(set).sort((a, b) => a.localeCompare(b))
  }, [cache])

  const handlePickTag = useCallback(
    (tag: string) => {
      const matched = new Set<string>()
      for (const k of keys) {
        if ((cache.get(k) ?? []).includes(tag)) matched.add(k)
      }
      setSel(matched); setAnchor(null)
      toast(t('tagEdit.selectedContaining', { tag, n: matched.size }), 'success')
    },
    [keys, cache, toast, t]
  )

  if (!activeVersion) {
    return <p className="text-fg-tertiary p-6">{t('tagEdit.noVersion')}</p>
  }

  const handleClick = (key: string, e: React.MouseEvent) => {
    const r = applySelection(sel, key, e, filteredKeys, anchor)
    setSel(r.next); setAnchor(r.anchor)
  }

  const navActive = (delta: number) => {
    if (navKeys.length === 0) return
    const i = activeKey ? navKeys.indexOf(activeKey) : -1
    const next = i < 0 ? 0 : (i + delta + navKeys.length) % navKeys.length
    setActiveKey(navKeys[next])
  }

  const updateActiveTags = (tags: string[]) => {
    if (!activeKey) return
    setCache((prev) => {
      const next = new Map(prev); next.set(activeKey, [...tags]); return next
    })
  }

  const applyBulkUpdates = (updates: Map<string, string[]>) => {
    setCache((prev) => {
      const next = new Map(prev)
      for (const [k, v] of updates) next.set(k, v)
      return next
    })
  }

  // 标签分布行内 × 触发：从当前选中图删除该 tag。toast 主要是反馈"动作生效"
  // —— 因为 UI 上 tag 行马上消失/数字下降已经是可视反馈，这里只补一句简短的。
  const removeTagFromSelected = (tag: string) => {
    if (selectedKeys.length === 0) return
    const updates = new Map<string, string[]>()
    for (const k of selectedKeys) {
      const cur = cache.get(k) ?? []
      if (!cur.includes(tag)) continue
      updates.set(k, cur.filter((tt) => tt !== tag))
    }
    if (updates.size === 0) return
    applyBulkUpdates(updates)
    toast(t('tagEdit.removedFromN', { tag, n: updates.size }), 'success')
  }

  // 标签分布行内 ✎ inline edit 提交：把选中图里的 oldTag 替换成 newTag，去重。
  const replaceTagInSelected = (oldTag: string, newTag: string) => {
    if (selectedKeys.length === 0 || !newTag || newTag === oldTag) return
    const updates = new Map<string, string[]>()
    for (const k of selectedKeys) {
      const cur = cache.get(k) ?? []
      if (!cur.includes(oldTag)) continue
      const next: string[] = []
      const seen = new Set<string>()
      for (const tt of cur) {
        const out = tt === oldTag ? newTag : tt
        if (seen.has(out)) continue
        seen.add(out); next.push(out)
      }
      updates.set(k, next)
    }
    if (updates.size === 0) return
    applyBulkUpdates(updates)
    toast(t('tagEdit.replacedInN', { from: oldTag, to: newTag, n: updates.size }), 'success')
  }

  const onSave = async () => {
    if (!dirty || versionId == null) return
    const items: CommitItem[] = dirtyKeys.map((k) => {
      const m = meta.get(k)!
      return { folder: m.folder, name: m.name, tags: cache.get(k) ?? [] }
    })
    try {
      const r = await api.commitCaptions(project.id, versionId, items)
      setInitial(new Map(cache))
      toast(t('tagEdit.savedToast', { written: r.written, id: r.snapshot.id }), 'success')
      void reload()
    } catch (e) { toast(String(e), 'error') }
  }

  const onAfterRestore = async () => {
    await reloadCache()
    setActiveKey('')
    setSel(new Set())
    setAnchor(null)
    setFolderFilter('')
    await reload()
  }

  const downloadTrainZip = () => {
    if (dirty) {
      toast(t('tagEdit.saveThenDownloadToast'), 'error')
      return
    }
    if (exporting) return
    setExporting(true)
    // <a download> 直链 —— 浏览器原生接管下载（进度条 / 暂停 / 切 tab 不中断）。
    // app-side "打包中..." 由 version_train_zip_ready/_failed SSE 清。
    const filename = `${project.slug}-${activeVersion.label}.train.zip`
    const a = document.createElement('a')
    a.href = api.versionTrainZipUrl(project.id, activeVersion.id)
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const stats = activeVersion.stats
  const trainTotal = stats?.train_image_count ?? 0
  const taggedTotal = stats?.tagged_image_count ?? 0
  const allTagged = trainTotal > 0 && taggedTotal >= trainTotal

  const activeMeta = activeKey ? meta.get(activeKey) : undefined
  const activeFolder = activeMeta?.folder ?? ''
  const activeName = activeMeta?.name ?? ''
  const activeTags = activeKey ? cache.get(activeKey) ?? [] : []

  const isEditing = Boolean(activeKey)

  return (
    <StepShell
      idx={4}
      title={t('tagEdit.title')}
      subtitle={t('tagEdit.subtitle')}
      actions={
        <>
          {activeVersion.trigger_word && (
            <span className="badge badge-neutral" title={t('tagEdit.triggerWordHint')}>
              {t('tagEdit.triggerWord')}:{' '}
              <code className="font-mono">{activeVersion.trigger_word}</code>
            </span>
          )}
          {stats && (
            <span className={allTagged ? 'badge badge-ok' : 'badge badge-neutral'}>
              {t('tagEdit.taggedBadge', { tagged: taggedTotal, total: trainTotal })}
            </span>
          )}
          <button
            className="btn btn-primary btn-sm"
            disabled={exporting || dirty || trainTotal === 0}
            onClick={downloadTrainZip}
            title={dirty ? t('tagEdit.saveThenDownload') : t('tagEdit.downloadTitle')}
          >
            {exporting ? t('tagEdit.zipping') : t('tagEdit.downloadZip')}
          </button>
          <SaveBar
            pid={project.id}
            vid={activeVersion.id}
            dirtyCount={dirtyKeys.length}
            onSave={onSave}
            onAfterRestore={onAfterRestore}
          />
        </>
      }
    >
      <div className="flex flex-1 min-h-0 gap-2.5">

        <section
          className="rounded-md border border-subtle bg-surface flex flex-col min-w-0 overflow-hidden"
          style={{ flex: isEditing ? 1.5 : 1 }}
        >
          {folderNames.length > 1 && (
            <div className="px-2 pt-2 pb-1.5 flex items-center gap-1 flex-wrap shrink-0 border-b border-subtle">
              {['', ...folderNames].map((f) => {
                const isActive = f === folderFilter
                const label = f || t('common.all')
                const count = f ? folderCounts.get(f) ?? 0 : keys.length
                return (
                  <button
                    key={f || '__all__'}
                    type="button"
                    onClick={() => setFolderFilter(f)}
                    className={
                      'px-2 py-0.5 rounded-full text-xs font-medium transition-colors ' +
                      (isActive
                        ? 'bg-accent text-white'
                        : 'bg-overlay text-fg-secondary hover:bg-accent-soft')
                    }
                  >
                    {label} {count}
                  </button>
                )
              })}
            </div>
          )}
          <div className="flex-1 overflow-y-auto p-2">
            <ImageGrid
              items={captionItems}
              selected={sel}
              activeName={activeKey || undefined}
              onSelect={handleClick}
              onActivate={setActiveKey}
              clickMode="activate"
              ariaLabel="tag-edit-grid"
              emptyHint={
                folderFilter
                  ? t('tagEdit.noImagesInFolder', { folder: folderFilter })
                  : t('tagEdit.noImagesHint')
              }
            />
          </div>
        </section>

        {isEditing && (
          <section className="flex-1 rounded-md border border-subtle bg-surface flex flex-col min-w-0 overflow-hidden">
            <div className="px-3 py-2 border-b border-subtle shrink-0 flex items-center gap-2">
              <span className="text-xs text-fg-tertiary">{t('tagEdit.singleEdit')}</span>
              <code className="flex-1 min-w-0 text-xs font-mono text-fg-secondary truncate">
                {activeFolder}/{activeName}
              </code>
            </div>
            <div className="flex-1 relative bg-sunken">
              <img
                key={activeKey}
                src={api.versionThumbUrl(project.id, activeVersion.id, 'train', activeName, activeFolder, 800)}
                alt={activeName}
                className="absolute inset-2 object-contain rounded-sm"
                style={{ width: 'calc(100% - 16px)', height: 'calc(100% - 16px)' }}
              />
            </div>
          </section>
        )}

        <div className="flex flex-col gap-2.5 min-w-0 flex-1 min-h-0" style={{ flex: '0 0 32%' }}>
          {isEditing ? (
            // editing 时：bulk + 标签分布 都和"调单图标签"无关，整个侧栏让位给
            // TagEditor。退出 editing 后自动回来，sel / folderFilter 等 state
            // 保留（隐藏的是 UI 不是状态）。
            <section className="flex-1 rounded-md border border-subtle bg-surface p-2.5 flex flex-col gap-2 min-h-0 overflow-hidden">
              <div className="flex items-center gap-1.5 shrink-0">
                <button onClick={() => navActive(-1)} disabled={navKeys.length === 0} aria-label={t('tagEdit.prevImage')} className="btn btn-secondary btn-sm">◀</button>
                <span className="text-xs text-fg-tertiary font-mono flex-1 text-center">
                  {activeIndex >= 0 ? `${activeIndex + 1} / ${navKeys.length}` : `– / ${navKeys.length}`}
                </span>
                <button onClick={() => navActive(1)} disabled={navKeys.length === 0} aria-label={t('tagEdit.nextImage')} className="btn btn-secondary btn-sm">▶</button>
                <button onClick={() => setActiveKey('')} className="btn btn-ghost btn-sm ml-1" aria-label={t('tagEdit.closeEdit')}>✕</button>
              </div>
              <TagEditor tags={activeTags} onChange={updateActiveTags} />
            </section>
          ) : (
            // BulkActionBar + TagStatsPanel 合到同一个外框 section（"标签编辑
            // 工作区"），视觉上是一个面板：上半是 batch 输入区，下半是标签分布
            // 兼快捷单 tag 操作区。两者共享"操作 = 给当前选中图做"的语义。
            <section className="flex-1 min-h-0 rounded-md border border-subtle bg-surface flex flex-col overflow-hidden">
              <BulkActionBar
                cache={cache}
                selectedKeys={selectedKeys}
                onApply={applyBulkUpdates}
                tagSuggestions={tagSuggestions}
                onClearSelection={() => setSel(new Set())}
                onSelectAll={() => setSel(new Set(filteredKeys))}
                totalCount={filteredKeys.length}
              />
              <TagStatsPanel
                cache={cache}
                selectedKeys={selectedKeys}
                onPickTag={handlePickTag}
                onRemoveTag={removeTagFromSelected}
                onReplaceTag={replaceTagInSelected}
              />
            </section>
          )}
        </div>
      </div>
    </StepShell>
  )
}
