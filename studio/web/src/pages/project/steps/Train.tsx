import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useBlocker, useNavigate, useOutletContext } from 'react-router-dom'
import {
  api,
  type ConfigData,
  type PresetSummary,
  type ProjectDetail,
  type RegStatus,
  type SchemaResponse,
  type Task,
  type Version,
  type VersionConfigResponse,
} from '../../../api/client'
import ActionGroup from '../../../components/ActionGroup'
import Alert from '../../../components/Alert'
import Button from '../../../components/Button'
import ConfigSkeleton from '../../../components/ConfigSkeleton'
import EmptyState from '../../../components/EmptyState'
import { Input, Select } from '../../../components/FormControl'
import { useDialog } from '../../../components/Dialog'
import SaveIndicator from '../../../components/SaveIndicator'
import SchemaForm, { visibleSchemaGroups } from '../../../components/SchemaForm'
import SchemaSectionIndex from '../../../components/SchemaSectionIndex'
import StepShell from '../../../components/StepShell'
import type { SaveStatus } from '../../../lib/SettingsData'
import { useToast } from '../../../components/Toast'
import { useSettingsDrawer } from '../../../lib/SettingsDrawer'
import { useAdvancedMode } from '../../../lib/useAdvancedMode'
import { useLocalStorageState } from '../../../lib/useLocalStorageState'
import {
  PRESET_NAME_RE,
  defaultsFromSchema,
  generateUniquePresetName,
} from '../../../lib/preset-helpers'
import FamilySwitchDialog from '../../../components/FamilySwitchDialog'
import Modal from '../../../components/Modal'
import { SegmentedControl } from '../../../components/SelectionGroup'
import { useEventStream } from '../../../lib/useEventStream'
import { schemaGroupLabel } from '../../../lib/schema'
import TrainPlanPanel, {
  LoadError,
  TrainRunSummary,
  useTrainDatasetPlan,
} from './train/TrainPlanPanel'

// 全局模型字段来自全局设置，对版本维度只读
const GLOBAL_MODEL_FIELDS = [
  'transformer_path',
  'vae_path',
  'text_encoder_path',
  't5_tokenizer_path',
]

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
  setVersionSwitchGuard: (guard: (() => Promise<boolean>) | null) => void
}

export default function TrainPage() {
  const { t, i18n } = useTranslation()
  const { project, activeVersion, reload, setVersionSwitchGuard } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm, prompt } = useDialog()
  const navigate = useNavigate()
  const settingsDrawer = useSettingsDrawer()

  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [presets, setPresets] = useState<PresetSummary[]>([])
  const [configResp, setConfigResp] = useState<VersionConfigResponse | null>(null)
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [reg, setReg] = useState<RegStatus | null>(null)
  const [schemaError, setSchemaError] = useState<string | null>(null)
  const [configError, setConfigError] = useState<string | null>(null)
  const [presetsError, setPresetsError] = useState<string | null>(null)
  const [regError, setRegError] = useState<string | null>(null)
  const [activeTask, setActiveTask] = useState<Task | null>(null)
  const [taskLoading, setTaskLoading] = useState(true)
  const [taskError, setTaskError] = useState<string | null>(null)
  const taskRequestRef = useRef(0)
  const [busy, setBusy] = useState(false)
  const busyRef = useRef(false)
  const saveErrorRef = useRef<string | null>(null)
  const forceSaveRef = useRef(false)
  const boundaryDoneRef = useRef<{ promise: Promise<void>; resolve: () => void } | null>(null)
  const leavePromiseRef = useRef<Promise<boolean> | null>(null)
  const [autoSyncPaths, setAutoSyncPaths] = useState<boolean>(true)
  const [droppedFields, setDroppedFields] = useState<string[]>([])
  const [defaultedFields, setDefaultedFields] = useState<string[]>([])

  /** 已落盘的 config JSON 快照，dirty 判断的 baseline。 */
  const savedJsonRef = useRef<string | null>(null)
  /** 当前 config 的同步镜像。React setState 是 queued 的，事件 handler 跑完才
   * flush；flush / navigation guards 需要立刻读到最新值，不能等 React
   * commit。所有 setConfig 都走 setConfigSync 包装，写 ref 同步、写 state 异步。 */
  const configRef = useRef<ConfigData | null>(null)
  /** 当前在飞的 save promise，dedup 重叠的保存请求。 */
  const inFlightSaveRef = useRef<Promise<void> | null>(null)
  /** 等待中的 debounce setTimeout id；onEnqueue 需要 cancel 它。 */
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  // 0.8.2 hotfix：删 presetBaselineRef + customized 标签逻辑。fork 之后
  // version yaml 跟全局预设解耦了，承认这是"项目专属配置"。「已自定义」
  // 标签是骗人的（全局模型 4 个字段 fork 时被注入绝对路径，跟全局预设
  // 相对路径 diff 永远存在 → 永远显示已自定义）。

  // 预设 picker（dropdown 模式，与 Presets 页一致）
  const [pickerOpen, setPickerOpen] = useState(false)
  const [pickerSearch, setPickerSearch] = useState('')
  // 0.17 P-B — 定时训练弹层（延迟 N 小时 / 指定绝对时间两种入口，D7）。
  const [scheduleOpen, setScheduleOpen] = useState(false)
  const [scheduleTime, setScheduleTime] = useState('')
  const [advancedMode, toggleAdvancedMode] = useAdvancedMode()
  const pickerAnchorRef = useRef<HTMLButtonElement | null>(null)
  const pickerPopRef = useRef<HTMLDivElement | null>(null)
  const scheduleInputRef = useRef<HTMLInputElement | null>(null)
  const pickerId = useId()
  const scheduleInputId = useId()

  // 「新建预设」=一键创建+套用：点 + 新建预设 卡片直接生成 <slug>_<label> 命名
  // 的预设、写全局池、fork 到当前 version。不弹中间表单，避免用户点了 + 就以为
  // "已创建"但实际什么都没存（state 不持久化，切页面回来又是空）。

  /** 包装 setConfig：先同步写 configRef（绕 React state flush 延迟），再调
   * setConfig 触发 React 渲染。 SchemaForm.onChange / 任何想改 config 的入口
   * 都要走这个，不要直接 setConfig。 */
  const setConfigSync = useCallback((v: ConfigData | null) => {
    configRef.current = v
    setConfig(v)
  }, [])

  /** 待确认的族切换目标（非空时渲染 FamilySwitchDialog）。 */
  const [familySwitchTarget, setFamilySwitchTarget] = useState<string | null>(null)

  /** header 自动保存指示（与 Settings 页同款 SaveIndicator）。 */
  const [saveStatus, setSaveStatus] = useState<SaveStatus>({ state: 'idle' })

  /** SchemaForm.onChange 入口：拦截 model_family 变化走切换动作（P4-3）。
   * 切族不是裸字段编辑——弹结构化确认对话框（后端重算路径 + 重置族风味
   * 字段），用户取消则保持旧值不动。其余字段变更原样透传 setConfigSync。 */
  const onFormChange = useCallback((v: ConfigData) => {
    if (busyRef.current) return
    const prev = configRef.current
    const prevFamily = String(prev?.model_family ?? 'anima')
    const nextFamily = String(v.model_family ?? 'anima')
    if (prev && nextFamily !== prevFamily) {
      setFamilySwitchTarget(nextFamily)
      return
    }
    setConfigSync(v)
  }, [setConfigSync])

  const vid = activeVersion?.id ?? null

  const applyPresetWarnings = useCallback((r: { dropped_fields?: string[]; defaulted_fields?: string[] }) => {
    setDroppedFields(r.dropped_fields ?? [])
    setDefaultedFields(r.defaulted_fields ?? [])
  }, [])

  const refreshConfig = useCallback(async () => {
    if (!vid) return
    setConfigError(null)
    try {
      const r = await api.getVersionConfig(project.id, vid)
      setConfigResp(r)
      setConfigSync(r.config)
      savedJsonRef.current = JSON.stringify(r.config)
      // 老 config 兼容（InfoNoise 互斥被后端自动关掉等）由后端写进 r.defaulted_fields，
      // 顶部 banner 渲染。dropped_fields 兜底 schema 演进时丢弃的旧字段。
      applyPresetWarnings(r)
    } catch (e) {
      setConfigError(String(e))
    }
  }, [project.id, vid, setConfigSync, applyPresetWarnings])

  const refreshSchema = useCallback(async () => {
    setSchemaError(null)
    try {
      setSchema(await api.schema())
    } catch (e) {
      setSchemaError(String(e))
    }
  }, [])

  const refreshPresets = useCallback(async () => {
    setPresetsError(null)
    try {
      setPresets(await api.listPresets())
    } catch (e) {
      setPresetsError(String(e))
    }
  }, [])

  useEffect(() => {
    void refreshSchema()
    void refreshPresets()
    api.getSecrets().then((s) => setAutoSyncPaths(s.models?.auto_sync_paths ?? true)).catch(() => {})
  }, [refreshPresets, refreshSchema])

  useEffect(() => {
    setConfigResp(null)
    setConfigSync(null)
    setSaveStatus({ state: 'idle' })
    savedJsonRef.current = null
    setDroppedFields([])
    setDefaultedFields([])
    void refreshConfig()
  }, [refreshConfig, setConfigSync])

  // 拉 reg 状态用于显示「训练集 + 正则」分布。
  const refreshReg = useCallback(async () => {
    if (!vid) return
    setRegError(null)
    try {
      setReg(await api.getRegStatus(project.id, vid))
    } catch (e) {
      setRegError(String(e))
    }
  }, [project.id, vid])

  useEffect(() => {
    setReg(null)
    void refreshReg()
  }, [refreshReg])

  const refreshActiveTask = useCallback(async (showLoading = false) => {
    if (!vid) return
    const requestId = ++taskRequestRef.current
    if (showLoading) setTaskLoading(true)
    setTaskError(null)
    try {
      const tasks = await api.listQueueLive()
      if (requestId !== taskRequestRef.current) return
      setActiveTask(tasks.find((task) => (
        task.project_id === project.id
        && task.version_id === vid
        && ['train', 'reg_ai', 'generate'].includes(task.task_type ?? 'train')
      )) ?? null)
    } catch (e) {
      if (requestId === taskRequestRef.current) setTaskError(String(e))
    } finally {
      if (requestId === taskRequestRef.current) setTaskLoading(false)
    }
  }, [project.id, vid])

  useEffect(() => {
    setActiveTask(null)
    void refreshActiveTask(true)
  }, [refreshActiveTask])

  useEventStream((event) => {
    if (event.type !== 'task_state_changed') return
    // 事件不携带 project/version；后台刷新不切换首次加载状态，因此可以查询
    // 所有 task 事件而不让 Start/Schedule 闪烁，并能发现另一标签页新建的 task。
    void refreshActiveTask()
    if (activeTask && event.task_id === activeTask.id) void reload()
  }, { onOpen: () => { void refreshActiveTask() } })


  // config 里的值就是训练实际用的值：字段一律可编辑，系统不在背后改写已保存的
  // config（docs/design/version-config-ownership.md）。
  //   - 项目特定字段（data_dir / output_name 等）：创建 version 时按项目结构预填，
  //     挂「自动 · 项目设置」说明这是预填的，不是预设里来的。
  //   - 4 个模型路径：创建时按 auto_sync_paths 取全局设置作初值，之后归用户所有。
  //     全局设置再变也不回头改写本 version（重现性）。徽章里「全局设置」可点，
  //     跳 Settings 模型区；值与全局当前值不一致时再挂「恢复默认」，让对齐成为
  //     用户的一次显式动作。
  const makeAutoHints = useCallback(
    (
      formValues: ConfigData | null,
      setForm: (v: ConfigData) => void,
    ): Record<string, React.ReactNode> => {
      const h: Record<string, React.ReactNode> = {}
      for (const f of configResp?.project_specific_fields ?? []) {
        h[f] = t('train.projectAutoHint')
      }
      const psd = configResp?.project_specific_defaults
      for (const f of GLOBAL_MODEL_FIELDS) {
        const dv = psd?.[f]
        const differs =
          typeof dv === 'string' && !!dv && String(formValues?.[f] ?? '') !== dv
        h[f] = (
          <>
            <button
              type="button"
              onClick={() => settingsDrawer.open({ section: 'models' })}
              className="bg-transparent border-none p-0 underline text-warn hover:opacity-80 cursor-pointer"
            >
              {t('train.globalAutoLockedLink')}
            </button>
            {differs && formValues && (
              <>
                {' · '}
                <button
                  type="button"
                  onClick={() => setForm({ ...formValues, [f]: dv })}
                  title={t('train.restoreGlobalDefaultTitle')}
                  className="bg-transparent border-none p-0 underline text-warn hover:opacity-80 cursor-pointer"
                >
                  {t('train.restoreGlobalDefault')}
                </button>
              </>
            )}
          </>
        )
      }
      return h
    },
    [configResp?.project_specific_fields, configResp?.project_specific_defaults, t, settingsDrawer],
  )

  const clearSaveDebounce = useCallback(() => {
    if (debounceTimerRef.current !== null) {
      clearTimeout(debounceTimerRef.current)
      debounceTimerRef.current = null
    }
  }, [])

  /** One writer drains the latest draft, not a snapshot captured by a timer.
   * A failed drain stops here; only a new edit or explicit action retries it. */
  const flushConfig = useCallback((force = false): Promise<void> => {
    clearSaveDebounce()
    if (force) forceSaveRef.current = true
    if (inFlightSaveRef.current) return inFlightSaveRef.current
    const drain = async () => {
      let wrote = false
      try {
        while (vid && configRef.current) {
          const cfg = configRef.current
          if (!forceSaveRef.current && JSON.stringify(cfg) === savedJsonRef.current) break
          saveErrorRef.current = null
          setSaveStatus({ state: 'saving' })
          const r = await api.putVersionConfig(project.id, vid, cfg)
          wrote = true
          forceSaveRef.current = false
          savedJsonRef.current = JSON.stringify(r.config)
          setConfigResp((prev) => prev ? { ...prev, has_config: true, config: r.config } : prev)
          // An old response must never replace edits made while it was in flight.
          if (configRef.current === cfg) {
            configRef.current = r.config
            setConfig(r.config)
          }
          applyPresetWarnings({})
        }
        saveErrorRef.current = null
        if (wrote && configRef.current) setSaveStatus({ state: 'saved', at: Date.now() })
      } catch (e) {
        saveErrorRef.current = String(e)
        setSaveStatus({ state: 'error', error: String(e) })
        throw e
      } finally {
        clearSaveDebounce()
        inFlightSaveRef.current = null
      }
    }
    // Defer the drain so even a no-op flush releases the assigned promise.
    const promise = Promise.resolve().then(drain)
    inFlightSaveRef.current = promise
    return promise
  }, [project.id, vid, applyPresetWarnings, clearSaveDebounce])

  const finishBoundary = useCallback(() => {
    busyRef.current = false
    setBusy(false)
    boundaryDoneRef.current?.resolve()
    boundaryDoneRef.current = null
  }, [])

  const beginBoundary = useCallback(async () => {
    if (busyRef.current) return false
    busyRef.current = true
    setBusy(true)
    let resolve!: () => void
    const promise = new Promise<void>((done) => { resolve = done })
    boundaryDoneRef.current = { promise, resolve }
    try {
      await flushConfig()
      return true
    } catch {
      // The local recovery alert is the sole announcement for save failures.
      finishBoundary()
      return false
    }
  }, [flushConfig, finishBoundary])

  const saveBeforeLeave = useCallback((): Promise<boolean> => {
    if (leavePromiseRef.current) return leavePromiseRef.current
    const leave = async () => {
      if (boundaryDoneRef.current) {
        await boundaryDoneRef.current.promise
        if (saveErrorRef.current) return false
      }
      if (!(await beginBoundary())) return false
      finishBoundary()
      return true
    }
    const promise = leave().finally(() => { leavePromiseRef.current = null })
    leavePromiseRef.current = promise
    return promise
  }, [beginBoundary, finishBoundary])

  // Debounce ordinary edits. Do not depend on save status or translation/toast
  // callbacks: a failed request or a render must not schedule another retry.
  useEffect(() => {
    if (!config || JSON.stringify(config) === savedJsonRef.current) return
    debounceTimerRef.current = setTimeout(() => {
      debounceTimerRef.current = null
      void flushConfig().catch(() => {})
    }, 600)
    return clearSaveDebounce
  }, [config, flushConfig, clearSaveDebounce])

  const blocker = useBlocker(useCallback(() => (
    busyRef.current || forceSaveRef.current || Boolean(inFlightSaveRef.current)
    || Boolean(configRef.current && JSON.stringify(configRef.current) !== savedJsonRef.current)
  ), []))
  useEffect(() => {
    if (blocker.state !== 'blocked') return
    let cancelled = false
    void saveBeforeLeave().then((saved) => {
      if (cancelled) return
      if (saved) blocker.proceed()
      else blocker.reset()
    })
    return () => { cancelled = true }
  }, [blocker, saveBeforeLeave])

  useEffect(() => {
    setVersionSwitchGuard(saveBeforeLeave)
    return () => setVersionSwitchGuard(null)
  }, [setVersionSwitchGuard, saveBeforeLeave])

  useEffect(() => {
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      const cur = configRef.current
      const dirty = Boolean(cur && JSON.stringify(cur) !== savedJsonRef.current)
      if (!dirty && !forceSaveRef.current && !inFlightSaveRef.current && !busyRef.current) return
      event.preventDefault()
      event.returnValue = ''
    }
    window.addEventListener('beforeunload', onBeforeUnload)
    return () => window.removeEventListener('beforeunload', onBeforeUnload)
  }, [])

  const filteredPresets = useMemo(
    () => presets.filter((p) => !pickerSearch || p.name.toLowerCase().includes(pickerSearch.toLowerCase())),
    [presets, pickerSearch],
  )

  // 配置表单的局部滚动根与章节锚点。
  const schemaScrollRef = useRef<HTMLDivElement | null>(null)
  const previewToggleRef = useRef<HTMLButtonElement | null>(null)
  const [storedPreviewOpen, setPreviewOpen] = useLocalStorageState('train.previewOpen', true)
  const previewOpen = storedPreviewOpen !== false
  const [storedPreviewTab, setPreviewTab] = useLocalStorageState<'stats' | 'config'>('train.previewTab', 'stats')
  const previewTab = storedPreviewTab === 'config' ? 'config' : 'stats'
  const datasetPlan = useTrainDatasetPlan({ projectId: project.id, activeVersion, reg, config })
  const [compactGroup, setCompactGroup] = useState('')
  const visibleGroups = useMemo(
    () => (schema ? visibleSchemaGroups(schema, advancedMode) : []),
    [schema, advancedMode],
  )
  useEffect(() => {
    if (!visibleGroups.some((group) => group.key === compactGroup)) {
      setCompactGroup(visibleGroups[0]?.key ?? '')
    }
  }, [compactGroup, visibleGroups])

  // popover 关闭：点外面 / Esc
  useEffect(() => {
    if (!pickerOpen) return
    const onDocClick = (e: MouseEvent) => {
      const target = e.target as Node
      if (pickerPopRef.current?.contains(target) || pickerAnchorRef.current?.contains(target)) return
      setPickerOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      setPickerOpen(false)
      pickerAnchorRef.current?.focus()
    }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [pickerOpen])

  if (!activeVersion || !vid) {
    return <p className="text-fg-tertiary p-6">{t('train.noVersion')}</p>
  }

  const onForkPreset = async (name: string) => {
    if (!name) return
    if (configResp?.has_config) {
      const ok = await confirm(
        t('train.confirmReset', { name }),
        { tone: 'warn', okText: t('train.resetOkText') },
      )
      if (!ok) return
    }
    if (!(await beginBoundary())) return
    try {
      const r = await api.forkPresetForVersion(project.id, vid, name)
      applyPresetWarnings(r)
      // refreshConfig 刷本页 config state；reload 刷父级 activeVersion，
      // 主表单字段才会同步显示新预设的内容。
      await Promise.all([refreshConfig(), reload()])
      toast(t('train.resetSuccess', { name }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      finishBoundary()
    }
  }

  const onSaveAsPreset = async () => {
    const name = await prompt(t('train.promptPresetName'), {
      placeholder: 'my-preset',
      validate: (v) => {
        const trimmed = v.trim()
        if (!trimmed) return t('train.nameEmpty')
        if (!PRESET_NAME_RE.test(trimmed)) return t('train.nameInvalid')
        return null
      },
    })
    if (!name) return
    const trimmed = name.trim()
    if (!(await beginBoundary())) return
    try {
      await api.saveVersionConfigAsPreset(project.id, vid, trimmed, false)
      const list = await api.listPresets()
      setPresets(list)
      toast(t('train.savedAsPreset', { name: trimmed }), 'success')
    } catch (e) {
      const msg = String(e)
      if (msg.includes('已存在')) {
        const overwrite = await confirm(t('train.alreadyExists', { name: trimmed }), {
          tone: 'danger',
          okText: t('train.overwriteOkText'),
        })
        if (overwrite) {
          try {
            await api.saveVersionConfigAsPreset(project.id, vid, trimmed, true)
            const list = await api.listPresets()
            setPresets(list)
            toast(t('train.overwritePreset', { name: trimmed }), 'success')
          } catch (e2) {
            toast(String(e2), 'error')
          }
        }
      } else {
        toast(msg, 'error')
      }
    } finally {
      finishBoundary()
    }
  }

  /** 默认预设名 = `<slug>_<label>`；label 含非法字符时 fallback 到 `<slug>_v<id>`。
   * 用户在表单输入框里可改。 */
  const defaultPresetName = (): string => {
    if (!activeVersion) return project.slug
    const candidate = `${project.slug}_${activeVersion.label}`
    if (PRESET_NAME_RE.test(candidate)) return candidate
    return `${project.slug}_v${activeVersion.id}`
  }

  /** 一键新建预设 +套用到当前 version。
   *
   * 步骤：
   *   1. version 已有 config → 弹覆盖确认（跟 onForkPreset 一致）
   *   2. 拉最新 project_specific_defaults（缓存的 configResp 可能早于用户在
   *      Settings 里换模型，模型路径初值要取当前值）
   *   3. 配置 = schema 默认 + 项目路径预填（仅 autoSyncPaths 开时）
   *   4. 自动名 = `<slug>_<label>`（PRESET_NAME_RE 兼容），重名加 _1 _2 后缀
   *   5. savePreset → forkPresetForVersion → 刷三处状态
   */
  const startCreatePreset = async () => {
    setPickerOpen(false)
    if (!vid || !schema) return

    if (configResp?.has_config) {
      const ok = await confirm(
        t('train.confirmReset', { name: t('train.newPresetAction') }),
        { tone: 'warn', okText: t('train.resetOkText') },
      )
      if (!ok) return
    }

    if (!(await beginBoundary())) return
    try {
      const fresh = await api.getVersionConfig(project.id, vid).catch(() => null)
      const psd =
        fresh?.project_specific_defaults
        ?? configResp?.project_specific_defaults
        ?? {}

      // 全局预设池不带项目特定字段（数据集路径 / 输出名等）：schema 默认即可，
      // fork 时后端再把项目预填注入到 version 私有 config。
      // 4 个模型字段：autoSyncPaths ON 时用当前 Settings 算的绝对路径，OFF 时
      // 维持 schema 默认（独立模型用户场景）。跟 services/presets.py:
      // save_version_config_as_preset 的清理逻辑对齐。
      const cleaned: ConfigData = { ...defaultsFromSchema(schema) }
      if (autoSyncPaths) {
        for (const f of GLOBAL_MODEL_FIELDS) {
          if (typeof psd[f] === 'string' && psd[f]) cleaned[f] = psd[f]
        }
      }

      const name = generateUniquePresetName(defaultPresetName(), presets)
      await api.savePreset(name, cleaned)
      const r = await api.forkPresetForVersion(project.id, vid, name)
      applyPresetWarnings(r)

      const list = await api.listPresets()
      setPresets(list)
      // refreshConfig 刷本页 config state；reload 刷父级 activeVersion，
      // 主表单字段才会同步显示新预设的内容。
      await Promise.all([refreshConfig(), reload()])
      toast(t('train.createdPreset', { name }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      finishBoundary()
    }
  }

  const submitBlocked = taskLoading || Boolean(taskError) || busy || !configResp?.has_config || Boolean(activeTask)

  const onEnqueue = async (scheduledAt?: number) => {
    if (!configResp?.has_config) {
      toast(t('train.noPresetError'), 'error')
      return
    }
    if (activeTask) {
      toast(t('train.activeTaskBlocks', { id: activeTask.id }), 'error')
      return
    }
    if (!(await beginBoundary())) return
    try {
      const task = await api.enqueueVersionTraining(
        project.id, vid, scheduledAt != null ? { scheduledAt } : undefined,
      )
      if (scheduledAt != null) {
        toast(t('train.scheduledNav', {
          id: task.id,
          time: new Date(scheduledAt * 1000).toLocaleString(
            i18n.resolvedLanguage?.startsWith('zh') ? 'zh-CN' : 'en-US',
            { hour12: false },
          ),
        }), 'success')
      } else {
        toast(t('train.enqueuedNav', { id: task.id }), 'success')
      }
      setScheduleOpen(false)
      setActiveTask(task)
      void reload()
      navigate(`/queue/${task.id}`)
    } catch (e) {
      toast(String(e), 'error')
      void refreshActiveTask()
    } finally {
      finishBoundary()
    }
  }

  // datetime-local 的 value 格式（本地时区，分钟精度）。
  const toLocalInputValue = (d: Date) => {
    const pad = (n: number) => String(n).padStart(2, '0')
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
      + `T${pad(d.getHours())}:${pad(d.getMinutes())}`
  }

  const onScheduleAbsolute = () => {
    if (!scheduleTime) return
    const ts = new Date(scheduleTime).getTime() / 1000
    if (!Number.isFinite(ts) || ts <= Date.now() / 1000 + 30) {
      toast(t('train.schedulePast'), 'error')
      return
    }
    void onEnqueue(ts)
  }

  return (
    <>
    <StepShell
      title={t('steps.train.title')}
      subtitle={t('steps.train.subtitle')}
      actions={(
        <ActionGroup
          role="group"
          aria-label={t('train.pageActions')}
          status={(
            <div className="flex items-center gap-related">
              {activeTask && !previewOpen && (
                <Button variant="ghost" size="sm" onClick={() => navigate(`/queue/${activeTask.id}`)}>
                  #{activeTask.id} · {t(`status.${activeTask.status}`)}
                </Button>
              )}
              <SaveIndicator status={saveStatus} announceError={false} />
            </div>
          )}
          secondary={(
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setScheduleOpen(true)}
              disabled={submitBlocked}
              data-testid="train-schedule-btn"
            >
              {t('train.scheduleBtn')}
            </Button>
          )}
          primary={(
            <Button variant="primary" size="sm" onClick={() => void onEnqueue()} disabled={submitBlocked} loading={busy}>
              {t('train.startTrainBtn')}
            </Button>
          )}
        />
      )}
    >
      <div className="flex flex-col h-full min-h-0">
        <div className="train-workbench flex-1 min-h-0" data-preview-open={previewOpen} data-preview-tab={previewTab}>

          {/* 左栏：配置工作区 */}
          <div className="train-draft flex flex-col gap-3 min-h-0 min-w-0">

          {/* 预设 picker：dropdown 入口。0.8.2 起承认 version yaml 是 first-class
              「项目专属配置」，不再显示「绑定哪个预设」+「已自定义」标签 —— 这套
              判定逻辑骗人（全局模型 4 字段 fork 时被注入绝对路径，跟全局预设
              相对路径 diff 永远存在）。预设变成纯"模板起点"概念。 */}
          <section aria-label={t('train.configToolbar')} className="train-config-toolbar shrink-0 relative">
            <div className="train-config-toolbar-main">
            <button
              ref={pickerAnchorRef}
              type="button"
              aria-expanded={pickerOpen}
              aria-controls={pickerId}
              aria-haspopup="dialog"
              onClick={() => { setPickerOpen((v) => !v); setPickerSearch('') }}
              disabled={busy}
              className={[
                'train-preset-trigger flex items-center gap-related min-w-0 pl-3.5 pr-3 py-2.5',
                'rounded-md border transition-[border-color,background] duration-100',
                pickerOpen
                  ? 'border-accent bg-accent-soft'
                  : 'border-dim bg-surface shadow-sm hover:border-bold',
                busy ? 'cursor-default' : 'cursor-pointer',
              ].join(' ')}
              title={configResp?.has_config
                ? t('train.pickerTitleConfigured')
                : t('train.pickerTitleEmpty')}
            >
              <span className="caption uppercase tracking-[0.08em]">
                {t('train.configChip')}
              </span>
              <span className={[
                'text-md font-semibold flex-1 text-left truncate',
                configResp?.has_config ? 'text-fg-primary' : 'text-fg-tertiary',
              ].join(' ')}>
                {configResp?.has_config
                  ? t('train.scopedConfigLabel', { title: project.title, label: activeVersion.label })
                  : t('train.notConfiguredLabel')}
              </span>
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className={`shrink-0 text-fg-tertiary transition-transform ${pickerOpen ? 'rotate-180' : ''}`}
                aria-hidden="true"
              >
                <path d="m6 9 6 6 6-6" />
              </svg>
            </button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => void onSaveAsPreset()}
              disabled={busy || !configResp?.has_config}
              title={t('train.saveAsPresetTitle')}
            >
              {t('train.saveAsPreset')}
            </Button>
            </div>
            {configResp?.has_config && config && (
              <div className="train-config-toolbar-mode flex shrink-0 items-center gap-related" data-train-display-mode>
                <span className="text-sm text-fg-secondary">{t('train.parameterDisplay')}</span>
                <SegmentedControl
                value={advancedMode ? 'advanced' : 'simple'}
                onChange={(value) => {
                  const wantsAdvanced = value === 'advanced'
                  if (wantsAdvanced !== advancedMode) toggleAdvancedMode()
                }}
                items={[
                  { value: 'simple', label: t('train.simpleMode') },
                  { value: 'advanced', label: t('train.advancedMode') },
                ]}
                ariaLabel={t('train.modeLabel')}
                idPrefix="train-mode"
                layout="content"
                className="shrink-0"
              />
              </div>
            )}

            {/* popover */}
            {pickerOpen && (
              <div
                id={pickerId}
                ref={pickerPopRef}
                role="dialog"
                aria-label={t('train.presetLabel')}
                onKeyDown={(event) => {
                  if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
                  const choices = Array.from(
                    event.currentTarget.querySelectorAll<HTMLButtonElement>('[data-preset-choice]'),
                  )
                  if (choices.length === 0) return
                  const current = choices.indexOf(document.activeElement as HTMLButtonElement)
                  const next = event.key === 'ArrowDown'
                    ? (current + 1) % choices.length
                    : (current <= 0 ? choices.length - 1 : current - 1)
                  event.preventDefault()
                  choices[next]?.focus()
                }}
                className="absolute top-[calc(100%+6px)] left-0 w-[480px] max-w-[calc(100vw-2rem)] max-h-[480px] overflow-hidden rounded-md border border-subtle bg-surface shadow-lg flex flex-col z-50"
              >
                {/* search */}
                <div className="p-2.5 border-b border-subtle flex items-center gap-2">
                  <span className="relative flex-1 inline-flex items-center">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                      strokeWidth="2" strokeLinecap="round"
                      className="absolute left-2 text-fg-tertiary pointer-events-none">
                      <circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>
                    </svg>
                    <Input
                      autoFocus
                      aria-label={t('train.filterPresets')}
                      className="input w-full pl-7 text-sm"
                      placeholder={t('train.filterPresets')}
                      value={pickerSearch}
                      onChange={(e) => setPickerSearch(e.target.value)}
                    />
                  </span>
                </div>

                {/* grid */}
                <div className="flex-1 min-h-0 overflow-y-auto p-2.5">
                  <div className="grid grid-cols-2 gap-2">
                    {/* + 新建预设 永远第一格（跟 Presets 页面一致）。pickerSearch
                        非空时藏起来 —— 用户在搜旧的，新建是另一条意图。 */}
                    {!pickerSearch && (
                      <button
                        data-preset-choice
                        onClick={() => void startCreatePreset()}
                        disabled={busy}
                        className={[
                          'rounded-sm px-2.5 py-2 text-left border border-dashed transition-colors',
                          'border-subtle text-accent hover:border-accent hover:bg-accent-soft',
                          busy ? 'cursor-default' : 'cursor-pointer',
                          'bg-transparent text-sm font-semibold',
                        ].join(' ')}
                      >
                        {t('train.newPreset')}
                      </button>
                    )}
                    {filteredPresets.map((p) => {
                      // 0.8.2 起预设跟 version 脱钩，picker 卡片不再有
                      // "active = 当前绑定" 概念，全部一视同仁地作为可用模板。
                      return (
                        <button
                          data-preset-choice
                          key={p.name}
                          onClick={() => { setPickerOpen(false); void onForkPreset(p.name) }}
                          disabled={busy}
                          className={[
                            'rounded-sm px-2.5 py-2 text-left border transition-colors',
                            'border-subtle bg-sunken hover:border-bold',
                            busy ? 'cursor-default' : 'cursor-pointer',
                          ].join(' ')}
                        >
                          <div className="text-sm font-mono font-semibold truncate text-fg-primary">{p.name}</div>
                          <div className="text-xs text-fg-tertiary mt-0.5">
                            {t('train.readPresetParams')}
                          </div>
                        </button>
                      )
                    })}
                  </div>
                  {presets.length > 0 && filteredPresets.length === 0 && (
                    <div className="text-fg-tertiary text-sm text-center py-4">
                      {t('train.noMatch', { search: pickerSearch })}
                    </div>
                  )}
                </div>
              </div>
            )}
          </section>

            {saveStatus.state === 'error' && (
              <Alert tone="danger" size="sm" role="alert" action={(
                <Button variant="secondary" size="sm" disabled={busy} onClick={() => { void flushConfig().catch(() => {}) }}>
                  {t('common.retry')}
                </Button>
              )}>
                {t('train.saveFailed', { error: saveStatus.error })}
              </Alert>
            )}

            {(schemaError || configError || presetsError || taskError) && (
              <div className="space-y-2 shrink-0" aria-live="polite">
                {schemaError && <LoadError message={t('train.loadSchemaFailed', { error: schemaError })} onRetry={refreshSchema} />}
                {configError && <LoadError message={t('train.loadConfigFailed', { error: configError })} onRetry={refreshConfig} />}
                {presetsError && <LoadError message={t('train.loadPresetsFailed', { error: presetsError })} onRetry={refreshPresets} />}
                {taskError && (
                  <LoadError
                    message={t('train.loadTaskFailed', { error: taskError })}
                    onRetry={() => { void refreshActiveTask(true) }}
                  />
                )}
              </div>
            )}

            {configResp?.has_config && config && visibleGroups.length > 0 && (
              <label className="train-section-select shrink-0">
                <span className="caption block mb-1">{t('settings.pageIndex')}</span>
                <Select
                  value={compactGroup || visibleGroups[0].key}
                  onChange={(event) => {
                    const key = event.target.value
                    setCompactGroup(key)
                    document.getElementById(`schema-group-${key}`)?.scrollIntoView({ block: 'start' })
                  }}
                >
                  {visibleGroups.map((group) => (
                    <option key={group.key} value={group.key}>
                      {schemaGroupLabel(group.key, group.label, t)}
                    </option>
                  ))}
                </Select>
              </label>
            )}

            <div className="train-form-layout flex flex-1 min-h-0 gap-3">
            {configResp === null || !schema ? (
              <ConfigSkeleton label={t('train.loadingConfig')} />
            ) : !configResp.has_config ? (
              <EmptyState
                embedded
                className="flex-1 border border-dashed border-dim"
                title={t('train.notConfiguredLabel')}
                description={t('train.noConfigHint')}
                action={(
                  <ActionGroup
                    secondary={(
                      <Button variant="secondary" onClick={() => { setPickerOpen(true); setPickerSearch(''); pickerAnchorRef.current?.focus() }}>
                        {t('train.choosePreset')}
                      </Button>
                    )}
                    primary={(
                      <Button variant="primary" onClick={() => void startCreatePreset()} disabled={busy}>
                        {t('train.newPreset')}
                      </Button>
                    )}
                  />
                )}
              />
            ) : config ? (
              <section ref={schemaScrollRef} className="train-config-scroll flex-1 min-w-0 min-h-0 overflow-y-auto pr-1">
                {(droppedFields.length > 0 || defaultedFields.length > 0) && (
                  <Alert
                    tone="warning"
                    size="sm"
                    className="mb-3"
                    title={t('presets.compatNoticeTitle')}
                    action={(
                      <Button
                        variant="secondary"
                        size="xs"
                        onClick={() => {
                          const cur = configRef.current
                          if (!cur) return
                          void flushConfig(true)
                            .then(() => toast(t('presets.cleanLegacyDone'), 'success'))
                            .catch(() => {})
                        }}
                        disabled={busy || saveStatus.state === 'saving'}
                        title={t('presets.cleanLegacyTitle')}
                      >
                        {t('presets.cleanLegacyBtn')}
                      </Button>
                    )}
                  >
                    <div className="space-y-1">
                      {droppedFields.length > 0 && (
                        <div>{t('presets.droppedFieldsBody')}<code className="ml-1 text-xs opacity-80">{droppedFields.join(', ')}</code></div>
                      )}
                      {defaultedFields.length > 0 && (
                        <div>{t('presets.defaultedFieldsBody')}<code className="ml-1 text-xs opacity-80">{defaultedFields.join(', ')}</code></div>
                      )}
                    </div>
                  </Alert>
                )}
                <SchemaForm
                  schema={schema}
                  values={config}
                  onChange={onFormChange}
                  autoHints={makeAutoHints(config, onFormChange)}
                  disabledFields={busy ? Object.keys(schema.schema.properties ?? {}) : []}
                  disabledHints={busy ? Object.fromEntries(Object.keys(schema.schema.properties ?? {}).map((name) => [name, false])) : undefined}
                  advancedMode={advancedMode}
                />
                {familySwitchTarget && config && (
                  <FamilySwitchDialog
                    target={familySwitchTarget}
                    config={config}
                    onApply={(switched) => {
                      if (busyRef.current) return
                      setConfigSync(switched)
                      setFamilySwitchTarget(null)
                    }}
                    onCancel={() => setFamilySwitchTarget(null)}
                  />
                )}
              </section>
            ) : (
              <ConfigSkeleton label={t('train.loadingConfig')} />
            )}

            {configResp?.has_config && config && visibleGroups.length > 0 && (
              <div className="train-section-index shrink-0 overflow-y-auto">
                <SchemaSectionIndex
                  groups={visibleGroups}
                  scrollContainer={schemaScrollRef}
                />
              </div>
            )}
            </div>
          </div>

          <div className="train-preview-rail">
            <Button
              ref={previewToggleRef}
              variant="ghost"
              size="xs"
              className="train-preview-toggle"
              aria-label={t(previewOpen ? 'train.collapsePreview' : 'train.expandPreview')}
              title={t(previewOpen ? 'train.collapsePreview' : 'train.expandPreview')}
              aria-expanded={previewOpen}
              aria-controls="train-preview-panel"
              onClick={() => {
                setPreviewOpen(!previewOpen)
                previewToggleRef.current?.focus()
              }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <path d={previewOpen ? 'm9 5 7 7-7 7' : 'm15 5-7 7 7 7'} />
              </svg>
            </Button>
          </div>
          <TrainPlanPanel
            plan={datasetPlan}
            previewOpen={previewOpen}
            previewTab={previewTab}
            onPreviewTabChange={setPreviewTab}
            projectId={project.id}
            activeVersion={activeVersion}
            config={config}
            reg={reg}
            regError={regError}
            onRetryReg={refreshReg}
            activeTask={activeTask}
            taskLoading={taskLoading}
            saveStatus={saveStatus}
          />
      </div>
    </div>
    </StepShell>
    {scheduleOpen && (
    <Modal
      onClose={() => setScheduleOpen(false)}
      title={t('train.scheduleBtn')}
      description={t('train.scheduleDescription')}
      initialFocusRef={scheduleInputRef}
      size="sm"
      footer={(
        <ActionGroup
          secondary={(
            <Button variant="secondary" onClick={() => setScheduleOpen(false)}>
              {t('common.cancel')}
            </Button>
          )}
          primary={(
            <Button
              variant="primary"
              onClick={onScheduleAbsolute}
              disabled={submitBlocked || !scheduleTime}
              loading={busy}
              data-testid="train-schedule-confirm"
            >
              {t('train.scheduleConfirm')}
            </Button>
          )}
        />
      )}
    >
      <div className="space-y-5" data-testid="train-schedule-modal">
        <Alert tone="info" size="sm">{t('train.scheduleSnapshotHint')}</Alert>
        <TrainRunSummary config={config} plan={datasetPlan} />
        <section className="space-y-2">
          <h3 className="caption">{t('train.scheduleDelaySection')}</h3>
          <div className="grid grid-cols-4 gap-2">
            {[1, 2, 4, 8].map((hours) => (
              <Button
                key={hours}
                variant="secondary"
                size="sm"
                onClick={() => void onEnqueue(Date.now() / 1000 + hours * 3600)}
                disabled={submitBlocked}
                data-testid={`train-schedule-delay-${hours}h`}
              >
                +{hours}h
              </Button>
            ))}
          </div>
        </section>
        <section className="space-y-2">
          <label htmlFor={scheduleInputId} className="caption block">
            {t('train.scheduleAbsoluteSection')}
          </label>
          <Input
            ref={scheduleInputRef}
            id={scheduleInputId}
            type="datetime-local"
            value={scheduleTime}
            min={toLocalInputValue(new Date())}
            onChange={(event) => setScheduleTime(event.target.value)}
            data-testid="train-schedule-time"
          />
        </section>
      </div>
    </Modal>
    )}
    </>
  )
}
