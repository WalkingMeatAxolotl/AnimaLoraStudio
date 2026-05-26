import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  type ApiError,
  type ConfigData,
  type PresetSummary,
  type SchemaResponse,
} from '../../api/client'
import ConfigSkeleton from '../../components/ConfigSkeleton'
import { useDialog } from '../../components/Dialog'
import PathPicker from '../../components/PathPicker'
import SchemaForm from '../../components/SchemaForm'
import { useToast } from '../../components/Toast'
import { useSettingsDrawer } from '../../lib/SettingsDrawer'
import { useAdvancedMode } from '../../lib/useAdvancedMode'
import {
  PRESET_NAME_RE,
  defaultsFromSchema,
  loadPresetDescriptions,
  savePresetDescriptions,
} from '../../lib/preset-helpers'

// ── TOML 生成（键按字母排序，值尽量保留原始类型） ──────────────────────────
function toTomlValue(v: unknown): string {
  if (v === null || v === undefined) return ''
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number') return String(v)
  if (Array.isArray(v)) return '[' + v.map(toTomlValue).join(', ') + ']'
  if (typeof v === 'object') {
    const lines: string[] = []
    for (const [k, vv] of Object.entries(v as Record<string, unknown>)) {
      lines.push(`  ${k} = ${toTomlValue(vv)}`)
    }
    return '{\n' + lines.join('\n') + '\n}'
  }
  const s = String(v)
  if (/[\n"'#[\]{}]/.test(s)) return `'''\n${s}\n'''`
  if (s.includes(' ') || s === '' || /[^\w.\-]/.test(s)) return `"${s}"`
  return s
}

function generateToml(config: ConfigData): string {
  const keys = Object.keys(config).sort()
  return keys.map((k) => `${k} = ${toTomlValue(config[k])}`).join('\n')
}

// 预设名校验 / 描述存储 / schema 默认值 抽到 lib/preset-helpers.ts，
// 跟 Train 页面「新建预设」内联表单共享，避免两份维护。

// 上传冲突时,后端 409 body 透传到这里;用户决定覆盖 / 另存为 / 取消。
interface ConflictState {
  config: ConfigData
  desc: string
  suggestedName: string
}

type ConflictChoice =
  | { kind: 'overwrite' }
  | { kind: 'saveAs'; name: string }
  | { kind: 'cancel' }

export default function PresetsPage() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const settingsDrawer = useSettingsDrawer()

  // ── backend state ──
  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [presets, setPresets] = useState<PresetSummary[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [busy, setBusy] = useState(false)
  const [autoSyncPaths, setAutoSyncPaths] = useState<boolean>(true)
  // 4 个模型字段当前 Settings 算出的绝对路径（reset 按钮 + 新建预设默认值）
  const [modelPathDefaults, setModelPathDefaults] = useState<Record<string, string>>({})

  // 已保存快照，用于 dirty 判定
  const savedJsonRef = useRef<string | null>(null)

  // 描述
  const [descriptions, setDescriptions] = useState<Record<string, string>>(loadPresetDescriptions)
  const [descDraft, setDescDraft] = useState('')
  const [descDirty, setDescDirty] = useState(false)

  // 新建模式输入
  const [newName, setNewName] = useState('')
  const [newNameError, setNewNameError] = useState('')
  const isNew = selected === null

  // ── 上传冲突 dialog 状态 + 命令式 resolver ──
  // handleImportFile await 一个 Promise 直到用户在 dialog 里选了"覆盖/另存为/取消"。
  // resolver 是个 ref 函数,dialog 的 3 个按钮各调一次 → resolve Promise + 清状态。
  const [conflict, setConflict] = useState<ConflictState | null>(null)
  const conflictResolveRef = useRef<((c: ConflictChoice) => void) | null>(null)
  const askConflict = (state: ConflictState): Promise<ConflictChoice> =>
    new Promise((resolve) => {
      conflictResolveRef.current = resolve
      setConflict(state)
    })
  const resolveConflict = (choice: ConflictChoice) => {
    setConflict(null)
    const r = conflictResolveRef.current
    conflictResolveRef.current = null
    r?.(choice)
  }

  // ── UI 状态 ──
  const [pickerOpen, setPickerOpen] = useState(false)
  const [pickerSearch, setPickerSearch] = useState('')
  const [tomlOpen, setTomlOpen] = useState(false)
  const [exportDialogOpen, setExportDialogOpen] = useState(false)
  const [showImportPathPicker, setShowImportPathPicker] = useState(false)
  const [advancedMode, toggleAdvancedMode] = useAdvancedMode()
  const pickerAnchorRef = useRef<HTMLButtonElement | null>(null)
  const pickerPopRef = useRef<HTMLDivElement | null>(null)
  const newNameInputRef = useRef<HTMLInputElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // 4 个模型字段（用于新建预设默认值 / reset 按钮）。同 Train.tsx 的 GLOBAL_MODEL_FIELDS。
  const MODEL_PATH_FIELDS = useMemo(() => [
    'transformer_path', 'vae_path', 'text_encoder_path', 't5_tokenizer_path',
  ], [])

  // ── 加载 schema + 预设列表 + Settings toggle + 模型路径默认 ──
  useEffect(() => {
    api.schema().then(setSchema).catch((e) => toast(t('presets.loadSchemaFailed', { error: e }), 'error'))
    refreshList()
    api.getSecrets().then((s) => setAutoSyncPaths(s.models?.auto_sync_paths ?? true)).catch(() => {})
    api.getModelPathDefaults().then(setModelPathDefaults).catch(() => {})
  }, [t, toast])

  const refreshList = () => {
    api.listPresets().then(setPresets).catch(() => setPresets([]))
  }

  // ── 选 preset 切换 ──
  // 新建模式（selected=null）：用 schema 默认值预填表单,用户输名字 + 编辑后点保存。
  // 导入 / 复制副本 现在都走"一键落盘 + 自动选中"路径,不再走"切到新建模式预填表单"
  // 的中间态,所以这里不再有 draftSeed 分支。
  // modelPathDefaults 在此处只读初值快照、不进 deps：异步晚到的情况由下面那个
  // 带「用户没改过」guard 的 useEffect 覆盖，避免重入这里把用户编辑清掉。
  useEffect(() => {
    if (!selected) {
      if (schema) {
        // 用 modelPathDefaults 覆盖 schema 里 4 字段的相对默认值，保证新建预设
        // 表单里看到的是当前 Settings 算出的绝对路径，跟 fork 后实际落盘一致。
        const defaults = { ...defaultsFromSchema(schema), ...modelPathDefaults }
        setConfig(defaults)
        savedJsonRef.current = JSON.stringify(defaults)
        setNewName('')
        setDescDraft('')
        setDescDirty(false)
      } else {
        setConfig(null)
        savedJsonRef.current = null
        setNewName('')
        setDescDraft('')
        setDescDirty(false)
      }
      setNewNameError('')
      return
    }
    api.getPreset(selected).then((data) => {
      setConfig(data)
      savedJsonRef.current = JSON.stringify(data)
      setDescDraft(descriptions[selected] ?? '')
      setDescDirty(false)
    }).catch((e) => {
      toast(t('presets.loadFailed', { error: e }), 'error')
      setSelected(null)
    })
    // modelPathDefaults 故意排除：late-arrival 由下一个 useEffect 处理
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, schema, descriptions, t, toast])

  // modelPathDefaults 异步拉取，可能晚于主 init useEffect 到达。新建模式下
  // 用户没改过时（current JSON === saved JSON）就地覆盖 4 字段为绝对路径，
  // 避免 UI 一开始显示相对默认、稍后才换成绝对的视觉跳变。
  useEffect(() => {
    if (selected !== null) return
    if (!schema || !config) return
    if (Object.keys(modelPathDefaults).length === 0) return
    const currentJson = JSON.stringify(config)
    if (currentJson !== savedJsonRef.current) return  // 用户改过了，不要覆盖
    let needsUpdate = false
    for (const f of MODEL_PATH_FIELDS) {
      if (modelPathDefaults[f] && config[f] !== modelPathDefaults[f]) {
        needsUpdate = true
        break
      }
    }
    if (!needsUpdate) return
    const next = { ...config, ...modelPathDefaults }
    setConfig(next)
    savedJsonRef.current = JSON.stringify(next)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelPathDefaults, selected, schema])

  // ── 首次拿到列表后：自动选最近一个，省一次「切换」点击 ──
  const autoSelectedRef = useRef(false)
  useEffect(() => {
    if (autoSelectedRef.current) return
    if (presets.length > 0 && selected === null) {
      autoSelectedRef.current = true
      setSelected(presets[0].name)
    } else if (presets.length === 0 && schema) {
      autoSelectedRef.current = true
      // 列表为空 → 落到新建模式（schema 默认已经预填）
    }
  }, [presets, selected, schema])

  // ── 派生 ──
  const dirty = useMemo(() => {
    if (!config) return false
    return JSON.stringify(config) !== savedJsonRef.current
  }, [config])
  const hasAnyChange = dirty || descDirty

  const filteredPresets = useMemo(
    () => presets.filter((p) => !pickerSearch || p.name.toLowerCase().includes(pickerSearch.toLowerCase())),
    [presets, pickerSearch],
  )

  // auto_sync_paths ON：预设里 4 模型字段灰显（fork 反正会覆盖，编了无意义）。
  // OFF：可编辑，旁边挂「↺ 重置为全局默认」按钮把字段值刷成当前 Settings 算的绝对路径。
  const disabledFields = autoSyncPaths ? MODEL_PATH_FIELDS : []
  const disabledHints = useMemo(() => {
    const h: Record<string, React.ReactNode> = {}
    if (autoSyncPaths) {
      const node = (
        <>
          {t('train.globalAutoLockedPrefix')} ·{' '}
          <button
            type="button"
            onClick={() => settingsDrawer.open({ section: 'models' })}
            className="bg-transparent border-none p-0 underline text-warn hover:opacity-80 cursor-pointer"
          >
            {t('train.globalAutoLockedLink')}
          </button>
        </>
      )
      for (const f of MODEL_PATH_FIELDS) h[f] = node
    }
    return h
  }, [t, autoSyncPaths, MODEL_PATH_FIELDS, settingsDrawer])
  const autoHints = useMemo(() => {
    const h: Record<string, string> = {}
    if (!autoSyncPaths) {
      for (const f of MODEL_PATH_FIELDS) h[f] = t('train.globalAutoEditableHint')
    }
    return h
  }, [t, autoSyncPaths, MODEL_PATH_FIELDS])

  const fieldSuffixes = useMemo(() => {
    if (autoSyncPaths) return {}
    if (!config) return {}
    if (Object.keys(modelPathDefaults).length === 0) return {}
    const out: Record<string, React.ReactNode> = {}
    for (const f of MODEL_PATH_FIELDS) {
      const dv = modelPathDefaults[f]
      if (typeof dv !== 'string' || !dv) continue
      out[f] = (
        <button
          type="button"
          onClick={() => setConfig({ ...config, [f]: dv })}
          className="btn btn-ghost btn-sm shrink-0"
          title={t('train.resetToGlobalDefaultTitle')}
        >
          {t('train.resetToGlobalDefault')}
        </button>
      )
    }
    return out
  }, [autoSyncPaths, modelPathDefaults, config, t, MODEL_PATH_FIELDS])

  // ── popover 关闭：点外面关 ──
  useEffect(() => {
    if (!pickerOpen) return
    const onDocClick = (e: MouseEvent) => {
      const t = e.target as Node
      if (
        pickerPopRef.current?.contains(t) ||
        pickerAnchorRef.current?.contains(t)
      ) return
      setPickerOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setPickerOpen(false) }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [pickerOpen])

  // ── 操作 ──
  const handleSave = async () => {
    const name = isNew ? newName.trim() : selected
    if (!name) {
      setNewNameError(t('presets.nameRequired'))
      newNameInputRef.current?.focus()
      return
    }
    if (!config) return
    if (isNew) {
      if (!PRESET_NAME_RE.test(name)) { setNewNameError(t('presets.nameInvalid')); return }
      if (presets.find((p) => p.name === name)) { setNewNameError(t('presets.nameExists')); return }
    }
    setBusy(true)
    try {
      await api.savePreset(name, config)
      if (descDraft) {
        const next = { ...descriptions, [name]: descDraft }
        setDescriptions(next); savePresetDescriptions(next)
      } else if (descriptions[name]) {
        const { [name]: _, ...rest } = descriptions
        setDescriptions(rest); savePresetDescriptions(rest)
      }
      savedJsonRef.current = JSON.stringify(config)
      setDescDirty(false)
      if (isNew) {
        setSelected(name)
        setNewName('')
        setNewNameError('')
        toast(t('presets.created', { name }), 'success')
      } else {
        toast(t('presets.saved'), 'success')
      }
      refreshList()
    } catch (e) { toast(String(e), 'error') }
    finally { setBusy(false) }
  }

  // "复制副本":Save-As 语义 —— 把当前内存里的 config(含未保存编辑)写到新名字下,
  // refresh + 自动选中。原 preset 的 on-disk 内容不动;原 preset 的内存里"未保存编辑"
  // 仍属未保存状态(用户切回原 preset 时按现有 dirty 检查处理)。
  const handleDuplicate = async () => {
    if (!config || busy) return
    const baseName = selected ?? 'preset'
    let candidate = `${baseName}-copy`
    let i = 2
    while (presets.find((p) => p.name === candidate)) {
      candidate = `${baseName}-copy-${i++}`
    }
    setBusy(true)
    setPickerOpen(false)
    try {
      await api.savePreset(candidate, config)
      // 同步描述(描述字段当前是本地存的,不走后端)
      if (descDraft) {
        const next = { ...descriptions, [candidate]: descDraft }
        setDescriptions(next); savePresetDescriptions(next)
      }
      refreshList()
      setSelected(candidate)
      toast(t('presets.duplicated', { name: candidate }), 'success')
    } catch (e) { toast(String(e), 'error') }
    finally { setBusy(false) }
  }

  const handleNew = () => {
    setSelected(null)
    setPickerOpen(false)
  }

  const handleDelete = async () => {
    if (!selected) return
    if (!(await confirm(t('presets.confirmDelete', { name: selected }), { tone: 'danger', okText: t('common.delete') }))) return
    setBusy(true)
    api.deletePreset(selected).then(() => {
      const { [selected]: _, ...rest } = descriptions
      setDescriptions(rest); savePresetDescriptions(rest)
      setSelected(null)
      refreshList()
      toast(t('presets.deleted'), 'success')
    }).catch((e) => toast(String(e), 'error')).finally(() => setBusy(false))
  }

  const currentExportName = () => (isNew ? newName.trim() : selected) || 'preset'

  const downloadCurrentPreset = () => {
    if (!config) return
    if (isNew || !selected || hasAnyChange) {
      toast(t('presets.saveBeforeDownload'), 'info')
      return
    }
    // server FileResponse 直发磁盘上的原 yaml，已设 Content-Disposition。
    const a = document.createElement('a')
    a.href = api.presetDownloadUrl(selected)
    a.download = `${selected}.yaml`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const exportCurrentPresetToDataExports = async () => {
    if (!config) return
    setBusy(true)
    try {
      const result = await api.exportPresetToDataExports(currentExportName(), config)
      toast(t('presets.exportedToDataExports', { filename: result.filename, path: result.path }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  // 「导入」：上传 → 后端 yaml + pydantic 校验 + 直接落盘 + 返回 name。
  // 不冲突 → refresh + setSelected(name),一步到位出现在 picker 并选中。
  // 冲突(409)→ 弹 ImportConflictDialog 让用户选覆盖 / 另存为 / 取消;选定后
  // PUT /api/presets/{name} 落盘,refresh + setSelected。
  // 解析 / schema 校验失败 → toast。
  const handleImportedPreset = (name: string) => {
    refreshList()
    setSelected(name)
    toast(t('presets.imported', { name }), 'success')
  }

  const handleImportConflict = async (err: ApiError): Promise<boolean> => {
    if (err.status === 409 && err.detail && typeof err.detail === 'object') {
      const d = err.detail as { config?: ConfigData; suggested_name?: string }
      if (!d.config || !d.suggested_name) { toast(String(err), 'error'); return true }
      const choice = await askConflict({
        config: d.config, desc: '', suggestedName: d.suggested_name,
      })
      if (choice.kind === 'cancel') return true
      const target = choice.kind === 'overwrite' ? d.suggested_name : choice.name
      setBusy(true)
      try {
        await api.savePreset(target, d.config)
        handleImportedPreset(target)
      } catch (saveErr) { toast(String(saveErr), 'error') }
      finally { setBusy(false) }
      return true
    }
    return false
  }

  const handleImportFile = async (f: File) => {
    let imported: { name: string }
    try {
      imported = await api.importPreset(f)
    } catch (e) {
      const err = e as ApiError
      if (await handleImportConflict(err)) return
      toast(String(e), 'error')
      return
    }
    handleImportedPreset(imported.name)
  }

  const handleImportFromPath = async (path: string) => {
    setShowImportPathPicker(false)
    setBusy(true)
    try {
      const imported = await api.importPresetFromPath(path)
      handleImportedPreset(imported.name)
    } catch (e) {
      const err = e as ApiError
      if (await handleImportConflict(err)) return
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const onImportClick = () => fileInputRef.current?.click()

  const saveDisabled =
    busy
    || !config
    || (isNew && !newName.trim())
    || (!isNew && !hasAnyChange)

  // ── 渲染 ──
  return (
    <div className="fade-in flex flex-col h-full">

      {/* ── 单行 header：picker + 状态 + 全部操作 ──
        Topbar 已经显示「预设」面包屑，这里不再重复 h1。把上一版的页面标题
        和底部操作栏并成一行，picker 当做"当前编辑上下文"的标识，状态 +
        所有动作（导入 / 复制 / 导出 / 删除 / 保存）右侧排齐。 */}
      <div className="py-3 px-6 border-b border-subtle bg-canvas shrink-0 flex items-center gap-3.5 relative">
        <button
          ref={pickerAnchorRef}
          onClick={() => { setPickerOpen((v) => !v); setPickerSearch('') }}
          disabled={busy}
          className={[
            'flex items-center gap-3 min-w-[300px] pl-3.5 pr-3 py-2.5',
            'rounded-md border transition-[border-color,background] duration-100',
            pickerOpen
              ? 'border-accent bg-accent-soft'
              : 'border-dim bg-surface shadow-sm hover:border-bold',
            busy ? 'cursor-default' : 'cursor-pointer',
          ].join(' ')}
          title={t('presets.switchTitle')}
        >
          <span className="text-[10px] uppercase tracking-[0.08em] text-fg-tertiary font-semibold">
            {t('presets.label')}
          </span>
          <span className="font-mono text-md font-semibold text-fg-primary flex-1 text-left truncate">
            {selected ?? (newName.trim() || t('presets.creating'))}
          </span>
          <span className="text-fg-tertiary text-md">▾</span>
        </button>

        {/* 状态指示 */}
        <div className="flex items-center gap-2 min-w-0">
          <span className={[
            'inline-block w-2 h-2 rounded-full shrink-0',
            hasAnyChange ? 'bg-warn' : isNew ? 'bg-accent' : 'bg-ok',
          ].join(' ')} />
          <span className="text-sm text-fg-secondary whitespace-nowrap">
            {isNew ? t('presets.creating') : hasAnyChange ? t('presets.unsaved') : t('presets.savedStatus')}
          </span>
        </div>

        <span style={{ flex: 1 }} />

        {/* 全局动作 */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,.yaml,.yml"
          style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) void handleImportFile(f)
            if (fileInputRef.current) fileInputRef.current.value = ''
          }}
        />
        <button onClick={onImportClick} disabled={busy} className="btn btn-ghost btn-sm">
          {t('presets.importUpload')}
        </button>
        <button onClick={() => setShowImportPathPicker(true)} disabled={busy} className="btn btn-ghost btn-sm">
          {t('presets.importPath')}
        </button>

        {/* 编辑模式下的预设级动作 */}
        {!isNew && (
          <>
            <span style={{ width: 1, height: 22, background: 'var(--border-subtle)' }} />
            <button onClick={handleDuplicate} disabled={busy || !config} className="btn btn-ghost btn-sm">
              {t('presets.duplicate')}
            </button>
            <button onClick={() => setExportDialogOpen(true)} disabled={busy || !config} className="btn btn-ghost btn-sm">
              {t('presets.exportYaml')}
            </button>
            <button onClick={handleDelete} disabled={busy} className="btn btn-ghost btn-sm" style={{ color: 'var(--err)' }}>
              {t('common.delete')}
            </button>
          </>
        )}

        {/* 主操作 */}
        <button
          onClick={handleSave}
          disabled={saveDisabled}
          className="btn btn-primary btn-sm inline-flex items-center justify-center"
          style={{ minWidth: 0, paddingLeft: 12, paddingRight: 12 }}
        >
          {t('common.save')}
        </button>

        {/* popover */}
        {pickerOpen && (
          <div
            ref={pickerPopRef}
            role="dialog"
            aria-label={t('presets.switchPreset')}
            style={{
              position: 'absolute', top: 'calc(100% - 1px)', left: 24,
              width: 480, maxHeight: 480, overflow: 'hidden',
              borderRadius: 'var(--r-md)', border: '1px solid var(--border-subtle)',
              background: 'var(--bg-surface)', boxShadow: 'var(--sh-lg)',
              display: 'flex', flexDirection: 'column',
              zIndex: 50,
            }}
          >
              {/* search */}
              <div style={{
                padding: 10, borderBottom: '1px solid var(--border-subtle)',
                display: 'flex', alignItems: 'center', gap: 8,
              }}>
                <span style={{ position: 'relative', flex: 1, display: 'inline-flex', alignItems: 'center' }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                    strokeWidth="2" strokeLinecap="round"
                    style={{ position: 'absolute', left: 8, color: 'var(--fg-tertiary)', pointerEvents: 'none' }}>
                    <circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>
                  </svg>
                  <input
                    autoFocus
                    className="input"
                    placeholder={t('presets.filterPlaceholder')}
                    value={pickerSearch}
                    onChange={(e) => setPickerSearch(e.target.value)}
                    style={{ width: '100%', paddingLeft: 28, fontSize: 'var(--t-sm)' }}
                  />
                </span>
                <button
                  onClick={refreshList}
                  className="btn btn-ghost btn-sm"
                  style={{ fontSize: 'var(--t-xs)' }}
                  title={t('presets.refreshList')}
                >{t('common.refresh')}</button>
              </div>

              {/* grid */}
              <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: 10 }}>
                <div className="grid grid-cols-2 gap-2">
                  {/* + 新建（永远第一格） */}
                  <button
                    onClick={handleNew}
                    style={{
                      borderRadius: 'var(--r-sm)',
                      border: '1px dashed var(--border-default)',
                      background: 'transparent',
                      padding: '10px 12px',
                      textAlign: 'left',
                      cursor: 'pointer',
                      color: 'var(--accent)',
                      fontWeight: 600, fontSize: 'var(--t-sm)',
                    }}
                    onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.borderColor = 'var(--accent)'; (e.currentTarget as HTMLElement).style.background = 'var(--accent-soft)' }}
                    onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-default)'; (e.currentTarget as HTMLElement).style.background = 'transparent' }}
                  >
                    {t('presets.newPreset')}
                  </button>
                  {filteredPresets.map((p) => {
                    const active = p.name === selected
                    return (
                      <button
                        key={p.name}
                        onClick={() => { setSelected(p.name); setPickerOpen(false) }}
                        style={{
                          borderRadius: 'var(--r-sm)',
                          border: active ? '1px solid var(--accent)' : '1px solid var(--border-subtle)',
                          background: active ? 'var(--accent-soft)' : 'var(--bg-sunken)',
                          padding: '8px 10px',
                          textAlign: 'left',
                          cursor: 'pointer',
                        }}
                        onMouseEnter={(e) => { if (!active) (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-strong)' }}
                        onMouseLeave={(e) => { if (!active) (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-subtle)' }}
                      >
                        <div style={{
                          fontSize: 'var(--t-sm)', fontFamily: 'var(--font-mono)',
                          color: active ? 'var(--accent)' : 'var(--fg-primary)',
                          fontWeight: 600,
                          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                        }}>{p.name}</div>
                        <div style={{
                          fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)',
                          marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                        }}>
                          {descriptions[p.name] || '—'}
                        </div>
                      </button>
                    )
                  })}
                </div>
                {presets.length > 0 && filteredPresets.length === 0 && (
                  <div style={{
                    color: 'var(--fg-tertiary)', fontSize: 'var(--t-sm)',
                    textAlign: 'center', padding: '16px 0',
                  }}>
                    {t('presets.noMatch', { search: pickerSearch })}
                  </div>
                )}
              </div>
            </div>
          )}
      </div>

      {/* ── content（scroll） ── */}
      <div className="flex-1 min-h-0 overflow-auto p-4">
        <div className="flex flex-col gap-3">

          {/* 名称 / 描述 */}
          <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5">
            <div className="flex gap-2.5">
              {isNew ? (
                <label className="flex-1 flex flex-col gap-1">
                  <span className="text-sm font-medium text-fg-secondary">{t('presets.presetName')}</span>
                  <input
                    ref={newNameInputRef}
                    className="input input-mono font-mono"
                    placeholder="my-training-preset"
                    value={newName}
                    onChange={(e) => { setNewName(e.target.value); setNewNameError('') }}
                    disabled={busy}
                  />
                  {newNameError && (
                    <span className="text-xs text-err">{newNameError}</span>
                  )}
                </label>
              ) : (
                <label className="flex-1 flex flex-col gap-1">
                  <span className="text-sm font-medium text-fg-secondary">{t('presets.nameReadonly')}</span>
                  <div className="py-1.5 px-3 rounded-md border border-subtle bg-sunken font-mono text-sm text-fg-primary">{selected}</div>
                </label>
              )}
              <label className="flex-[1.5] flex flex-col gap-1">
                <span className="text-sm font-medium text-fg-secondary">{t('presets.description')}</span>
                <input
                  className="input"
                  placeholder={t('presets.descPlaceholder')}
                  value={descDraft}
                  onChange={(e) => { setDescDraft(e.target.value); setDescDirty(true) }}
                  disabled={busy}
                />
              </label>
            </div>
          </section>

          {/* schema 表单 */}
          {!schema || !config ? (
            <div className="h-[200px] rounded-md border border-subtle bg-surface p-3.5">
              <ConfigSkeleton variant="flat" label={t('presets.loadingConfig')} />
            </div>
          ) : (
            <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5">
              <div className="flex items-center gap-2 mb-2.5">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-fg-tertiary shrink-0" />
                <span className="caption uppercase tracking-[0.06em] text-xs">{t('presets.trainingParams')}</span>
                <span className="flex-1" />
                <div className="inline-flex rounded-md border border-subtle overflow-hidden text-xs">
                  <button
                    type="button"
                    onClick={() => advancedMode && toggleAdvancedMode()}
                    className={`px-3 py-1 transition-colors ${!advancedMode ? 'bg-accent text-white' : 'bg-surface text-fg-secondary hover:bg-subtle'}`}
                  >
                    {t('train.simpleMode')}
                  </button>
                  <button
                    type="button"
                    onClick={() => !advancedMode && toggleAdvancedMode()}
                    className={`px-3 py-1 transition-colors ${advancedMode ? 'bg-accent text-white' : 'bg-surface text-fg-secondary hover:bg-subtle'}`}
                  >
                    {t('train.advancedMode')}
                  </button>
                </div>
              </div>
              <SchemaForm
                schema={schema}
                values={config}
                onChange={setConfig}
                disabledFields={disabledFields}
                disabledHints={disabledHints}
                autoHints={autoHints}
                fieldSuffixes={fieldSuffixes}
                advancedMode={advancedMode}
              />
            </section>
          )}

          {/* TOML 预览（默认折叠） */}
          {config && Object.keys(config).length > 0 && (
            <section className={`rounded-md border border-subtle bg-surface ${tomlOpen ? 'px-3.5 py-2.5' : 'px-3.5 py-1.5'}`}>
              <button
                type="button"
                onClick={() => setTomlOpen((v) => !v)}
                className="w-full flex items-center gap-2 bg-transparent border-none p-0 cursor-pointer text-left"
              >
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-info shrink-0" />
                <span className="caption uppercase tracking-[0.06em] text-xs">{t('presets.tomlPreview')}</span>
                <span className="text-[10px] text-fg-tertiary">
                  {tomlOpen ? t('presets.tomlReadable') : t('presets.tomlCollapsed')}
                </span>
                <span className="flex-1" />
                {tomlOpen && (
                  <button
                    className="btn btn-ghost btn-sm text-xs"
                    onClick={(e) => {
                      e.stopPropagation()
                      const toml = generateToml(config)
                      navigator.clipboard.writeText(toml)
                        .then(() => toast(t('presets.copied'), 'success'))
                        .catch(() => toast(t('presets.copyFailed'), 'error'))
                    }}
                  >{t('common.copy')}</button>
                )}
                <span className="text-fg-tertiary">{tomlOpen ? '▾' : '▸'}</span>
              </button>
              {tomlOpen && (
                <pre className="m-0 mt-2.5 p-3 bg-sunken rounded-sm font-mono text-xs text-fg-secondary leading-[1.7] whitespace-pre-wrap break-words max-h-80 overflow-auto">
                  {generateToml(config)}
                </pre>
              )}
            </section>
          )}
        </div>
      </div>

      {exportDialogOpen && (
        <PresetExportDialog
          onDownload={() => {
            setExportDialogOpen(false)
            downloadCurrentPreset()
          }}
          onDataExports={() => {
            setExportDialogOpen(false)
            void exportCurrentPresetToDataExports()
          }}
          onCancel={() => setExportDialogOpen(false)}
        />
      )}

      {showImportPathPicker && (
        <PathPicker
          dirOnly={false}
          onClose={() => setShowImportPathPicker(false)}
          onPick={(path) => { void handleImportFromPath(path) }}
        />
      )}

      {conflict && (
        <ImportConflictDialog
          suggestedName={conflict.suggestedName}
          existingNames={presets.map((p) => p.name)}
          onDecide={resolveConflict}
        />
      )}
    </div>
  )
}

function PresetExportDialog({
  onDownload,
  onDataExports,
  onCancel,
}: {
  onDownload: () => void
  onDataExports: () => void
  onCancel: () => void
}) {
  const { t } = useTranslation()
  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/50"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel() }}
    >
      <div className="bg-elevated border border-dim rounded-lg w-[90%] max-w-[420px] p-6 flex flex-col gap-4 shadow-xl">
        <div>
          <h2 className="m-0 text-lg font-semibold text-fg-primary">{t('presets.exportPresetTitle')}</h2>
          <p className="mt-1 mb-0 text-sm text-fg-secondary">{t('presets.exportPresetHint')}</p>
        </div>
        <button type="button" className="card p-4 text-left hover:border-dim" onClick={onDownload}>
          <div className="font-medium text-fg-primary mb-1">{t('presets.exportDownload')}</div>
          <div className="text-xs text-fg-tertiary">{t('presets.exportDownloadHint')}</div>
        </button>
        <button type="button" className="card p-4 text-left hover:border-dim" onClick={onDataExports}>
          <div className="font-medium text-fg-primary mb-1">{t('presets.exportDataExports')}</div>
          <div className="text-xs text-fg-tertiary">{t('presets.exportDataExportsHint')}</div>
        </button>
        <div className="flex justify-end">
          <button type="button" className="btn btn-secondary" onClick={onCancel}>{t('common.cancel')}</button>
        </div>
      </div>
    </div>
  )
}

// ────────────────────────────────────────────────────────────────────────────
// ImportConflictDialog —— 上传 preset 名字撞库时弹三选一
//
// 沿用 NewVersionDialog (Layout.tsx) 的内联声明式风格 —— Dialog.tsx 的 confirm/
// prompt/alert 三件套不够装 "3 个动作 + 一个 input" 这种形态。命令式 await 走
// 父组件的 askConflict / resolveConflict resolver pattern,call site 仍是
// `const choice = await askConflict(...)`。
function ImportConflictDialog({
  suggestedName,
  existingNames,
  onDecide,
}: {
  suggestedName: string
  existingNames: string[]
  onDecide: (c: ConflictChoice) => void
}) {
  const { t } = useTranslation()
  const [newName, setNewName] = useState(() => {
    // 默认 `{suggested}-2`,如果还撞继续 -3 / -4…
    let i = 2
    let cand = `${suggestedName}-${i}`
    while (existingNames.includes(cand)) cand = `${suggestedName}-${++i}`
    return cand
  })
  const [error, setError] = useState('')
  const inputRef = useRef<HTMLInputElement | null>(null)
  useEffect(() => {
    requestAnimationFrame(() => { inputRef.current?.focus(); inputRef.current?.select() })
  }, [])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onDecide({ kind: 'cancel' }) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onDecide])

  const submitSaveAs = (e?: React.FormEvent) => {
    e?.preventDefault()
    const v = newName.trim()
    if (!v) { setError(t('presets.nameRequired')); return }
    if (!PRESET_NAME_RE.test(v)) { setError(t('presets.nameInvalid')); return }
    if (existingNames.includes(v)) { setError(t('presets.nameExists')); return }
    onDecide({ kind: 'saveAs', name: v })
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/50"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onDecide({ kind: 'cancel' }) }}
    >
      <form
        onSubmit={submitSaveAs}
        className="bg-elevated border border-dim rounded-lg w-[90%] max-w-[480px] p-6 flex flex-col gap-4 shadow-xl"
      >
        <h2 className="m-0 text-lg font-semibold text-fg-primary">
          {t('presets.importConflictTitle', { name: suggestedName })}
        </h2>
        <p className="m-0 text-sm text-fg-secondary">
          {t('presets.importConflictBody')}
        </p>
        <label className="flex flex-col gap-1.5">
          <span className="text-sm text-fg-secondary">{t('presets.importSaveAsLabel')}</span>
          <input
            ref={inputRef}
            className="input input-mono font-mono"
            value={newName}
            onChange={(e) => { setNewName(e.target.value); if (error) setError('') }}
          />
          {error && <span className="text-xs text-err">{error}</span>}
        </label>
        <div className="flex gap-2 justify-end mt-1">
          <button
            type="button"
            onClick={() => onDecide({ kind: 'cancel' })}
            className="btn btn-secondary"
          >
            {t('common.cancel')}
          </button>
          <button
            type="button"
            onClick={() => onDecide({ kind: 'overwrite' })}
            className="btn btn-warn"
            title={t('presets.importOverwriteTitle', { name: suggestedName })}
          >
            {t('presets.importOverwrite')}
          </button>
          <button type="submit" className="btn btn-primary">
            {t('presets.importSaveAs')}
          </button>
        </div>
      </form>
    </div>
  )
}

