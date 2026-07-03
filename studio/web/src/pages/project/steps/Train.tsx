import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate, useOutletContext } from 'react-router-dom'
import {
  api,
  type BucketDistribution,
  type ConfigData,
  type PresetSummary,
  type ProjectDetail,
  type RegStatus,
  type SchemaResponse,
  type Version,
  type VersionConfigResponse,
} from '../../../api/client'
import { parseFolderMeta } from '../../../lib/folderMeta'
import { useLocalStorageState } from '../../../lib/useLocalStorageState'
import ConfigSkeleton from '../../../components/ConfigSkeleton'
import { useDialog } from '../../../components/Dialog'
import SchemaForm, { visibleSchemaGroups } from '../../../components/SchemaForm'
import SchemaSectionIndex from '../../../components/SchemaSectionIndex'
import StepShell from '../../../components/StepShell'
import { useToast } from '../../../components/Toast'
import { useSettingsDrawer } from '../../../lib/SettingsDrawer'
import { useAdvancedMode } from '../../../lib/useAdvancedMode'
import {
  PRESET_NAME_RE,
  defaultsFromSchema,
  generateUniquePresetName,
} from '../../../lib/preset-helpers'

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
}

export default function TrainPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm, prompt } = useDialog()
  const navigate = useNavigate()
  const settingsDrawer = useSettingsDrawer()

  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [presets, setPresets] = useState<PresetSummary[]>([])
  const [configResp, setConfigResp] = useState<VersionConfigResponse | null>(null)
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [reg, setReg] = useState<RegStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [autoSyncPaths, setAutoSyncPaths] = useState<boolean>(true)
  const [droppedFields, setDroppedFields] = useState<string[]>([])
  const [defaultedFields, setDefaultedFields] = useState<string[]>([])

  /** 已落盘的 config JSON 快照，dirty 判断的 baseline。 */
  const savedJsonRef = useRef<string | null>(null)
  /** 当前 config 的同步镜像。React setState 是 queued 的，事件 handler 跑完才
   * flush；onEnqueue / cleanup-on-unmount 需要立刻读到最新值，不能等 React
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

  const vid = activeVersion?.id ?? null

  const applyPresetWarnings = useCallback((r: { dropped_fields?: string[]; defaulted_fields?: string[] }) => {
    setDroppedFields(r.dropped_fields ?? [])
    setDefaultedFields(r.defaulted_fields ?? [])
  }, [])

  const refreshConfig = useCallback(async () => {
    if (!vid) return
    try {
      const r = await api.getVersionConfig(project.id, vid)
      setConfigResp(r)
      setConfigSync(r.config)
      savedJsonRef.current = JSON.stringify(r.config)
      // 老 config 兼容（InfoNoise 互斥被后端自动关掉等）由后端写进 r.defaulted_fields，
      // 顶部 banner 渲染。dropped_fields 兜底 schema 演进时丢弃的旧字段。
      applyPresetWarnings(r)
    } catch (e) {
      toast(t('train.loadConfigFailed', { error: e }), 'error')
    }
  }, [project.id, vid, toast, setConfigSync, t, applyPresetWarnings])

  useEffect(() => {
    api.schema().then(setSchema).catch((e) => toast(t('train.loadSchemaFailed', { error: e }), 'error'))
    api.listPresets().then(setPresets).catch(() => setPresets([]))
    api.getSecrets().then((s) => setAutoSyncPaths(s.models?.auto_sync_paths ?? true)).catch(() => {})
  }, [toast, t])

  useEffect(() => {
    setDroppedFields([])
    setDefaultedFields([])
    void refreshConfig()
  }, [refreshConfig])

  // 拉 reg 状态用于显示「训练集 + 正则」分布
  useEffect(() => {
    if (!vid) return
    api.getRegStatus(project.id, vid).then(setReg).catch(() => setReg(null))
  }, [project.id, vid])


  // auto_sync_paths ON（默认 / 多数用户）：4 个模型路径字段 disabled，fork 时
  // 后端自动用 Settings 全局值覆盖；OFF（独立模型用户）：字段可编辑 + reset
  // 按钮，fork 时尊重预设里的绝对路径。
  const disabledFields = autoSyncPaths ? GLOBAL_MODEL_FIELDS : []
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
      for (const f of GLOBAL_MODEL_FIELDS) h[f] = node
    }
    return h
  }, [t, autoSyncPaths, settingsDrawer])
  // 项目特定字段（data_dir / reg_data_dir / output_dir 等）：值由项目预填，但
  // 不锁定，挂「自动 · 项目设置」徽章让用户知道这是预填的，不是预设里来的。
  // toggle OFF 时全局模型字段也挂 hint「默认来自 Settings · 可改」。
  const autoHints = useMemo(() => {
    const h: Record<string, string> = {}
    for (const f of configResp?.project_specific_fields ?? []) {
      if (!GLOBAL_MODEL_FIELDS.includes(f)) h[f] = t('train.projectAutoHint')
    }
    if (!autoSyncPaths) {
      for (const f of GLOBAL_MODEL_FIELDS) h[f] = t('train.globalAutoEditableHint')
    }
    return h
  }, [configResp?.project_specific_fields, t, autoSyncPaths])

  // toggle OFF 时给 4 个模型字段加「↺ 重置为全局默认」按钮。值取自
  // configResp.project_specific_defaults（后端已用 default_paths_for_new_version
  // 算好的绝对路径）。
  const makeResetSuffixes = useCallback(
    (formValues: ConfigData | null, setForm: (v: ConfigData) => void): Record<string, React.ReactNode> => {
      if (autoSyncPaths) return {}
      const psd = configResp?.project_specific_defaults
      if (!psd || !formValues) return {}
      const out: Record<string, React.ReactNode> = {}
      for (const f of GLOBAL_MODEL_FIELDS) {
        const dv = psd[f]
        if (typeof dv !== 'string' || !dv) continue
        out[f] = (
          <button
            type="button"
            onClick={() => setForm({ ...formValues, [f]: dv })}
            className="btn btn-ghost btn-sm shrink-0"
            title={t('train.resetToGlobalDefaultTitle')}
          >
            {t('train.resetToGlobalDefault')}
          </button>
        )
      }
      return out
    },
    [autoSyncPaths, configResp?.project_specific_defaults, t],
  )

  /** 落盘 cfg。串行化保证：如果上一次 save 还在飞，等它跑完再决定是否要再
   * save；这样多次 setConfig + debounce 不会丢任何一次的内容。
   *
   * 注意 race：用户在 await 期间可能又改了 config —— 那时不能用 server 返回的
   * 归一化结果去覆盖 React state（会清空他正在打字的字段）。靠 reference
   * 比对 configRef.current === cfg 区分：
   *   - 相等 → 用户没动过，安全 sync server 归一化结果到 UI
   *   - 不等 → 用户有新内容，只更新 savedJson baseline，UI state 不动；
   *            useEffect debounce 会自然为新内容触发下一轮 save 收敛 */
  const persistConfig = useCallback(async (cfg: ConfigData): Promise<void> => {
    while (inFlightSaveRef.current) {
      await inFlightSaveRef.current
    }
    if (JSON.stringify(cfg) === savedJsonRef.current) return
    const p = (async () => {
      const r = await api.putVersionConfig(project.id, vid!, cfg)
      setConfigResp((prev) => prev ? { ...prev, has_config: true, config: r.config } : prev)
      // baseline 用 server 归一化后的 r.config，下次 dirty diff 才不会假阳性。
      savedJsonRef.current = JSON.stringify(r.config)
      if (configRef.current === cfg) {
        configRef.current = r.config
        setConfig(r.config)
      }
    })()
    inFlightSaveRef.current = p
    try { await p } finally { inFlightSaveRef.current = null }
  }, [project.id, vid])

  // ── auto-save ─────────────────────────────────────────────────────────
  // config 变化 → 600ms 后没新改动就落盘。中途又改 → cleanup clearTimeout 重置。
  useEffect(() => {
    if (!config) return
    if (JSON.stringify(config) === savedJsonRef.current) return
    debounceTimerRef.current = setTimeout(() => {
      debounceTimerRef.current = null
      void persistConfig(config).catch((e) => toast(t('train.saveFailed', { error: e }), 'error'))
    }, 600)
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
        debounceTimerRef.current = null
      }
    }
  }, [config, persistConfig, toast, t])

  // 卸载时（路由切走）如果还有 dirty 没落盘 → fire-and-forget 把 PUT 发出去。
  // fetch 一旦发起，浏览器会继续送，不需要 await。catch 静默以免 cleanup 抛出。
  useEffect(() => {
    return () => {
      const cur = configRef.current
      if (!cur || !vid) return
      if (JSON.stringify(cur) === savedJsonRef.current) return
      void api.putVersionConfig(project.id, vid, cur).catch(() => {})
    }
  }, [project.id, vid])

  const filteredPresets = useMemo(
    () => presets.filter((p) => !pickerSearch || p.name.toLowerCase().includes(pickerSearch.toLowerCase())),
    [presets, pickerSearch],
  )

  // 右侧 SchemaSectionIndex 的 IntersectionObserver root + 跳转目标
  const schemaScrollRef = useRef<HTMLDivElement | null>(null)
  // 右侧训练集分布预览抽屉的展开/收起（持久化）。收起时把横向空间让给表单。
  const [previewOpen, setPreviewOpen] = useLocalStorageState('train.previewOpen', true)
  const visibleGroups = useMemo(
    () => (schema ? visibleSchemaGroups(schema, advancedMode) : []),
    [schema, advancedMode],
  )

  // popover 关闭：点外面 / Esc
  useEffect(() => {
    if (!pickerOpen) return
    const onDocClick = (e: MouseEvent) => {
      const target = e.target as Node
      if (pickerPopRef.current?.contains(target) || pickerAnchorRef.current?.contains(target)) return
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
    setBusy(true)
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
      setBusy(false)
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
    setBusy(true)
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
      setBusy(false)
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
   *   2. 拉最新 project_specific_defaults（用户常见路径是 fork 之后才跑 reg
   *      build，缓存的 configResp 里 reg 状态过期）
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

    setBusy(true)
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
      setBusy(false)
    }
  }

  const onEnqueue = async (scheduledAt?: number) => {
    if (!configResp?.has_config) {
      toast(t('train.noPresetError'), 'error')
      return
    }
    setBusy(true)
    try {
      // 1. 干掉等待中的 debounce save；不然它可能在 enqueue 之后才 fire，导致
      //    worker 起来时读的是旧 config。
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
        debounceTimerRef.current = null
      }
      // 2. 等任何正在飞的 save 跑完（debounce 刚刚 fire 的那一次）。
      if (inFlightSaveRef.current) await inFlightSaveRef.current
      // 3. 用 configRef（不是 config closure）再 diff 一次。覆盖「用户在 input
      //    里敲完值不离开焦点直接点开始训练」的场景：input.onBlur (commit) 同步
      //    setConfig 入队但 React 还没 flush，config closure 是旧的，但 configRef
      //    在 setConfigSync 里同步更新过了。
      const cur = configRef.current
      if (cur && JSON.stringify(cur) !== savedJsonRef.current) {
        await persistConfig(cur)
      }
      const task = await api.enqueueVersionTraining(
        project.id, vid, scheduledAt != null ? { scheduledAt } : undefined,
      )
      if (scheduledAt != null) {
        toast(t('train.scheduledNav', {
          id: task.id,
          time: new Date(scheduledAt * 1000).toLocaleString('zh-CN', { hour12: false }),
        }), 'success')
      } else {
        toast(t('train.enqueuedNav', { id: task.id }), 'success')
      }
      setScheduleOpen(false)
      void reload()
      navigate('/queue')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
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
    <StepShell
      idx={6}
      title={t('steps.train.title')}
      subtitle={t('steps.train.subtitle')}
      actions={
        <>
          {/* 0.17 P-B — 定时训练：延迟 N 小时 / 指定时间，建成 scheduled task。
              样式对齐项目页「导入项目」（btn-ghost btn-sm）。 */}
          <button
            onClick={() => setScheduleOpen(true)}
            disabled={busy || !configResp?.has_config}
            className="btn btn-ghost btn-sm"
            title={t('train.scheduleHint')}
            data-testid="train-schedule-btn"
          >
            {t('train.scheduleBtn')}
          </button>
          {/* 样式对齐项目页「新建项目」（btn-primary btn-sm + icon + 文字） */}
          <button
            onClick={() => void onEnqueue()}
            disabled={busy || !configResp?.has_config}
            className="btn btn-primary btn-sm"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
            <span>{t('train.startTrainBtn')}</span>
          </button>
          {scheduleOpen && (
            <div
              role="dialog"
              aria-modal="true"
              className="fixed inset-0 z-40 flex items-center justify-center bg-black/50"
              onMouseDown={(e) => { if (e.target === e.currentTarget) setScheduleOpen(false) }}
              data-testid="train-schedule-modal"
            >
              <div className="bg-elevated border border-dim rounded-lg w-[90%] max-w-[440px] p-6 flex flex-col gap-4 shadow-xl">
                <h2 className="m-0 text-lg font-semibold text-fg-primary">
                  {t('train.scheduleBtn')}
                </h2>
                <div className="flex flex-col gap-1.5">
                  <span className="text-xs font-semibold text-fg-tertiary uppercase tracking-wide">
                    {t('train.scheduleDelaySection')}
                  </span>
                  <div className="flex gap-1.5">
                    {[1, 2, 4, 8].map((h) => (
                      <button
                        key={h}
                        onClick={() => void onEnqueue(Date.now() / 1000 + h * 3600)}
                        disabled={busy}
                        className="btn btn-secondary btn-sm flex-1"
                        data-testid={`train-schedule-delay-${h}h`}
                      >
                        +{h}h
                      </button>
                    ))}
                  </div>
                </div>
                <div className="flex flex-col gap-1.5">
                  <span className="text-xs font-semibold text-fg-tertiary uppercase tracking-wide">
                    {t('train.scheduleAbsoluteSection')}
                  </span>
                  <input
                    type="datetime-local"
                    className="input"
                    value={scheduleTime}
                    min={toLocalInputValue(new Date())}
                    onChange={(e) => setScheduleTime(e.target.value)}
                    data-testid="train-schedule-time"
                  />
                </div>
                <div className="flex gap-2 justify-end mt-1">
                  <button
                    onClick={() => setScheduleOpen(false)}
                    className="btn btn-secondary"
                  >
                    {t('common.cancel')}
                  </button>
                  <button
                    onClick={onScheduleAbsolute}
                    disabled={busy || !scheduleTime}
                    className="btn btn-primary"
                    data-testid="train-schedule-confirm"
                  >
                    {t('train.scheduleConfirm')}
                  </button>
                </div>
              </div>
            </div>
          )}
        </>
      }
    >
      <div className="flex flex-col h-full gap-3">

        {/* 两栏布局：左（预设 + config 编辑） / 右（估算面板） */}
        <div className="flex gap-3 flex-1 min-h-0">

          {/* 左栏：配置表单（flex-[3] 与右预览 flex-[1] 还原老 grid 3:1 比例） */}
          <div className="flex flex-col gap-3 min-h-0 min-w-0 overflow-y-auto flex-[3]">

          {/* 预设 picker：dropdown 入口。0.8.2 起承认 version yaml 是 first-class
              「项目专属配置」，不再显示「绑定哪个预设」+「已自定义」标签 —— 这套
              判定逻辑骗人（全局模型 4 字段 fork 时被注入绝对路径，跟全局预设
              相对路径 diff 永远存在）。预设变成纯"模板起点"概念。 */}
          <section className="flex items-center gap-2.5 shrink-0 relative">
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
              title={configResp?.has_config
                ? t('train.pickerTitleConfigured')
                : t('train.pickerTitleEmpty')}
            >
              <span className="text-[10px] uppercase tracking-[0.08em] text-fg-tertiary font-semibold">
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
              <span className="text-fg-tertiary text-md">▾</span>
            </button>
            <button
              onClick={() => void onSaveAsPreset()}
              disabled={busy || !configResp?.has_config}
              className="btn btn-ghost btn-sm"
              title={t('train.saveAsPresetTitle')}
            >
              {t('train.saveAsPreset')}
            </button>

            {/* popover */}
            {pickerOpen && (
              <div
                ref={pickerPopRef}
                role="dialog"
                aria-label={t('train.presetLabel')}
                className="absolute top-[calc(100%+6px)] left-0 w-[480px] max-h-[480px] overflow-hidden rounded-md border border-subtle bg-surface shadow-lg flex flex-col z-50"
              >
                {/* search */}
                <div className="p-2.5 border-b border-subtle flex items-center gap-2">
                  <span className="relative flex-1 inline-flex items-center">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                      strokeWidth="2" strokeLinecap="round"
                      className="absolute left-2 text-fg-tertiary pointer-events-none">
                      <circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>
                    </svg>
                    <input
                      autoFocus
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

            {configResp === null || !schema ? (
              <ConfigSkeleton label={t('train.loadingConfig')} />
            ) : !configResp.has_config ? (
              <div className="flex-1 flex items-center justify-center text-fg-tertiary text-sm rounded-md border border-dashed border-dim">
                {t('train.noConfigHint')}
              </div>
            ) : config ? (
              <section ref={schemaScrollRef} className="flex-1 min-h-0 overflow-y-auto pr-1">
                <div className="flex justify-end mb-2">
                  <div className="inline-flex rounded-md border border-subtle overflow-hidden text-xs">
                    <button
                      type="button"
                      onClick={() => !advancedMode || toggleAdvancedMode()}
                      className={`px-3 py-1 transition-colors ${!advancedMode ? 'bg-accent text-white' : 'bg-surface text-fg-secondary hover:bg-subtle'}`}
                    >
                      {t('train.simpleMode')}
                    </button>
                    <button
                      type="button"
                      onClick={() => advancedMode || toggleAdvancedMode()}
                      className={`px-3 py-1 transition-colors ${advancedMode ? 'bg-accent text-white' : 'bg-surface text-fg-secondary hover:bg-subtle'}`}
                    >
                      {t('train.advancedMode')}
                    </button>
                  </div>
                </div>
                {(droppedFields.length > 0 || defaultedFields.length > 0) && (
                  <div className="mb-3 rounded-md border border-amber-400/50 bg-amber-950/60 px-3.5 py-2.5 text-xs text-amber-100 space-y-1">
                    <span className="font-semibold text-amber-300">{t('presets.compatNoticeTitle')}</span>
                    {droppedFields.length > 0 && (
                      <div>{t('presets.droppedFieldsBody')}<code className="ml-1 text-[11px] opacity-80">{droppedFields.join(', ')}</code></div>
                    )}
                    {defaultedFields.length > 0 && (
                      <div>{t('presets.defaultedFieldsBody')}<code className="ml-1 text-[11px] opacity-80">{defaultedFields.join(', ')}</code></div>
                    )}
                  </div>
                )}
                <SchemaForm
                  schema={schema}
                  values={config}
                  onChange={setConfigSync}
                  disabledFields={disabledFields}
                  disabledHints={disabledHints}
                  autoHints={autoHints}
                  fieldSuffixes={makeResetSuffixes(config, setConfigSync)}
                  advancedMode={advancedMode}
                />
              </section>
            ) : (
              <ConfigSkeleton label={t('train.loadingConfig')} />
            )}
          </div>

        {/* 中栏：章节锚点导航（固定窄列，始终可见） */}
        {configResp?.has_config && config && visibleGroups.length > 0 && (
          <div className="shrink-0 w-[168px] overflow-y-auto">
            <SchemaSectionIndex
              groups={visibleGroups}
              scrollContainer={schemaScrollRef}
            />
          </div>
        )}

        {/* 把手：单竖线 + 顶部圆圈 ›/‹ —— 分隔预览抽屉 */}
        <div className="relative w-3 shrink-0 self-stretch flex justify-center">
          <div className="w-px bg-subtle" />
          <button
            type="button"
            onClick={() => setPreviewOpen((v) => !v)}
            title={previewOpen ? t('train.collapsePreview') : t('train.expandPreview')}
            aria-label={previewOpen ? t('train.collapsePreview') : t('train.expandPreview')}
            className="absolute top-1 left-1/2 -translate-x-1/2 w-6 h-6 rounded-full border border-subtle bg-surface text-fg-tertiary hover:text-accent hover:border-accent flex items-center justify-center text-xs leading-none shadow-sm"
          >
            {previewOpen ? '›' : '‹'}
          </button>
        </div>

        {/* 右栏：训练集分布预览抽屉（可收回）。flex-[1] 与左表单 flex-[3] 还原老
            grid 的 3:1 比例（按比例而非固定宽）；收起时整列不渲染、空间归表单。 */}
        {previewOpen && (
          <div className="flex-[1] min-w-0 overflow-y-auto">
            <DatasetStatsPanel
              projectId={project.id}
              activeVersion={activeVersion}
              reg={reg}
              config={config}
            />
          </div>
        )}
      </div>
    </div>
    </StepShell>
  )
}

/** config.resolution 归一成 number[]（schema 是 list[int]，旧 config / 标量也兜底）。 */
function configResolutions(config: ConfigData | null): number[] {
  const r = config?.resolution as unknown
  if (Array.isArray(r)) return r.length ? (r as number[]) : [1024]
  if (typeof r === 'number') return [r]
  return [1024]
}

/** 文件夹有效样本数 = repeat × 图数 × 分辨率档数（px 文件夹固定 1 档；否则跟 config 列表）。 */
function folderEffective(name: string, imageCount: number, resoCount: number): number {
  const { reso, repeat } = parseFolderMeta(name)
  return repeat * imageCount * (reso ? 1 : resoCount)
}

/** reg.files 形如 `5_concept/12345.png` —— 按首段文件夹聚合计数。 */
function aggregateRegFolders(files: string[]): Array<{ name: string; image_count: number }> {
  const m = new Map<string, number>()
  for (const f of files) {
    const idx = f.indexOf('/')
    if (idx < 0) continue
    const folder = f.slice(0, idx)
    m.set(folder, (m.get(folder) ?? 0) + 1)
  }
  return Array.from(m.entries())
    .map(([name, image_count]) => ({ name, image_count }))
    .sort((a, b) => a.name.localeCompare(b.name))
}

/** 训练集 + 正则集分布右栏面板。
 *
 * 显示每个 repeat 文件夹（Kohya 风格 N_label）的 raw 图数 + 有效图数（repeat × imgs），
 * train / reg 分两块汇总，最后给出有效图数总和——这是 anima_train 单 epoch 的实际样本数。
 */
function DatasetStatsPanel({
  projectId,
  activeVersion,
  reg,
  config,
}: {
  projectId: number
  activeVersion: Version | null
  reg: RegStatus | null
  config: ConfigData | null
}) {
  const { t } = useTranslation()
  const trainFolders = activeVersion?.stats?.train_folders ?? []
  const regFolders = useMemo(
    () => (reg && reg.exists ? aggregateRegFolders(reg.files) : []),
    [reg]
  )

  const resoCount = configResolutions(config).length
  const trainEffective = trainFolders.reduce(
    (s, f) => s + folderEffective(f.name, f.image_count, resoCount),
    0,
  )
  const regEffective = regFolders.reduce(
    (s, f) => s + folderEffective(f.name, f.image_count, resoCount),
    0,
  )
  const totalEffective = trainEffective + regEffective

  // 单 epoch 优化器步数估算（与 sd-scripts max_train_steps 同语义）。
  // 不算 AR bucketing 损失（每桶最后一 batch 可能不满），相同 AR 数据集误差 < 5%。
  // schema 字段：batch_size / grad_accum / epochs / max_steps（max_steps=0 表示不限）。
  const bs = Number(config?.batch_size) || 1
  const ga = Number(config?.grad_accum) || 1
  const epochs = Number(config?.epochs) || 0
  const maxSteps = Number(config?.max_steps) || 0
  const stepsPerEpoch = totalEffective > 0
    ? Math.ceil(totalEffective / (bs * ga))
    : null
  const naturalTotal = stepsPerEpoch !== null && epochs > 0
    ? stepsPerEpoch * epochs
    : null
  const finalTotal = naturalTotal !== null && maxSteps > 0
    ? Math.min(maxSteps, naturalTotal)
    : naturalTotal
  const maxStepsTruncates =
    maxSteps > 0 && naturalTotal !== null && maxSteps < naturalTotal

  return (
    <div className="flex flex-col gap-3 min-w-0">
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <div className="flex items-center gap-1.5 mb-2.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
          <span className="caption uppercase tracking-[0.06em] text-xs">{t('train.statsTitle')}</span>
        </div>

        <FolderSection
          title="train/"
          folders={trainFolders}
          effective={trainEffective}
          resoCount={resoCount}
          empty={t('train.noTrainImages')}
        />

        <div className="h-2" />

        <FolderSection
          title="reg/"
          folders={regFolders}
          effective={regEffective}
          resoCount={resoCount}
          empty={reg && !reg.exists ? t('train.regNotBuilt') : t('train.noRegImages')}
        />

        {/* 总计 + 步数估算（不含 AR bucketing 误差） */}
        <div className="mt-2.5 pt-2 border-t border-subtle flex flex-col gap-1 text-xs">
          <Row label={t('train.effectiveSamples')} value={String(totalEffective)} bold />
          {stepsPerEpoch !== null && (
            <Row
              label={`÷ batch × ga (${bs} × ${ga})`}
              value={`≈ ${stepsPerEpoch} steps/epoch`}
              dim
            />
          )}
          {naturalTotal !== null && (
            <Row
              label={`× epochs (${epochs})`}
              value={`≈ ${naturalTotal} steps`}
              dim
            />
          )}
          {finalTotal !== null && (
            <Row
              label={maxStepsTruncates ? t('train.maxStepsLabel', { n: maxSteps }) : t('train.totalSteps')}
              value={`≈ ${finalTotal}`}
              bold
            />
          )}
        </div>
      </div>

      <BucketPreview
        projectId={projectId}
        vid={activeVersion?.id ?? 0}
        sig={JSON.stringify([
          config?.resolution,
          config?.aspect_ratio_limit,
          // 文件夹名单（含 px 前缀 / repeat / 图数）—— 改名加 px 也要触发重取，不能只看总数
          activeVersion?.stats?.train_folders,
        ])}
      />
    </div>
  )
}

/** 训练集实际桶分布（后端用真 BucketManager 算）。按分辨率档分组、每桶有效图数。
 *  trainer 用 drop_last=False —— 桶不满只出短 batch、不丢图，所以这里不做丢图警告。 */
function BucketPreview({
  projectId, vid, sig,
}: {
  projectId: number
  vid: number
  sig: string
}) {
  const { t } = useTranslation()
  const [dist, setDist] = useState<BucketDistribution | null>(null)

  useEffect(() => {
    if (!projectId || !vid) return
    let cancelled = false
    api.getBucketDistribution(projectId, vid)
      .then((d) => { if (!cancelled) setDist(d) })
      .catch(() => { if (!cancelled) setDist(null) })
    return () => { cancelled = true }
  }, [projectId, vid, sig])

  if (!dist || dist.groups.length === 0) return null

  return (
    <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
      <div className="flex items-center gap-1.5 mb-2.5">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
        <span className="caption uppercase tracking-[0.06em] text-xs">{t('train.bucketDistTitle')}</span>
      </div>
      <div className="flex flex-col gap-2">
        {dist.groups.map((g) => (
          <div key={g.reso}>
            <div className="text-xs font-mono text-fg-secondary mb-1">{g.reso}px</div>
            <div className="flex flex-col gap-0.5">
              {g.buckets.map((b) => (
                <div
                  key={`${b.w}x${b.h}`}
                  className="flex items-baseline gap-1.5 text-xs font-mono pl-1"
                >
                  <span className="text-fg-tertiary">{b.w}×{b.h}</span>
                  <span className="flex-1 border-b border-dotted border-subtle self-end mb-1" />
                  <span className="text-fg-primary">{b.count}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="text-[10px] text-fg-tertiary mt-2">{t('train.bucketDistHint')}</div>
    </div>
  )
}

function FolderSection({
  title,
  folders,
  effective,
  resoCount,
  empty,
}: {
  title: string
  folders: Array<{ name: string; image_count: number }>
  effective: number
  resoCount: number
  empty: string
}) {
  const { t } = useTranslation()
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs mb-1">
        <span className="font-mono text-fg-secondary font-medium">{title}</span>
        {folders.length > 0 && (
          <span className="font-mono text-fg-tertiary">∑ {effective}</span>
        )}
      </div>
      {folders.length === 0 ? (
        <div className="text-xs text-fg-tertiary pl-1">{empty}</div>
      ) : (
        <div className="flex flex-col gap-0.5">
          {folders.map((f) => {
            const { reso, repeat, label } = parseFolderMeta(f.name)
            const folderResos = reso ? 1 : resoCount
            const eff = repeat * f.image_count * folderResos
            const resoTag = reso ? `${reso}px` : null
            return (
              <div
                key={f.name}
                className="flex items-baseline gap-1.5 text-xs font-mono text-fg-secondary pl-1"
                title={folderResos > 1
                  ? t('train.folderTipReso', { name: f.name, repeat, imgs: f.image_count, resos: folderResos, total: eff })
                  : t('train.folderTip', { name: f.name, repeat, imgs: f.image_count, total: eff })}
              >
                <span className="text-fg-tertiary">{label}</span>
                {resoTag && <span className="text-[10px] text-accent">{resoTag}</span>}
                <span className="flex-1 border-b border-dotted border-subtle self-end mb-1" />
                <span>
                  <span className="text-accent">{repeat}</span>
                  <span className="text-fg-tertiary"> × </span>
                  <span className="text-fg-primary">{f.image_count}</span>
                  {folderResos > 1 && (
                    <>
                      <span className="text-fg-tertiary"> × </span>
                      <span className="text-accent">{folderResos}</span>
                    </>
                  )}
                  <span className="text-fg-tertiary"> = </span>
                  <span className="text-fg-primary font-semibold">{eff}</span>
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function Row({
  label,
  value,
  bold,
  dim,
}: {
  label: string
  value: string
  bold?: boolean
  dim?: boolean
}) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
      <span style={{ color: dim ? 'var(--fg-tertiary)' : 'var(--fg-secondary)' }}>{label}</span>
      <span style={{
        fontFamily: 'var(--font-mono)',
        color: bold ? 'var(--accent)' : dim ? 'var(--fg-tertiary)' : 'var(--fg-primary)',
        fontWeight: bold ? 700 : 500,
      }}>{value}</span>
    </div>
  )
}
