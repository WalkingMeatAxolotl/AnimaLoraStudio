import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useOutletContext } from 'react-router-dom'
import {
  api,
  type ConfigData,
  type PresetSummary,
  type ProjectDetail,
  type RegStatus,
  type SchemaResponse,
  type Version,
  type VersionConfigResponse,
} from '../../../api/client'
import SchemaForm from '../../../components/SchemaForm'
import StepShell from '../../../components/StepShell'
import { useToast } from '../../../components/Toast'

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
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const navigate = useNavigate()

  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [presets, setPresets] = useState<PresetSummary[]>([])
  const [configResp, setConfigResp] = useState<VersionConfigResponse | null>(null)
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [reg, setReg] = useState<RegStatus | null>(null)
  const [busy, setBusy] = useState(false)

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
  /** 当前预设的原始 config（fork 时缓的那份），picker「· 已自定义」标签
   * 用它做基准。null = 还没拉到 / 没绑定到任何预设。 */
  const presetBaselineRef = useRef<ConfigData | null>(null)

  // 预设 picker（dropdown 模式，与 Presets 页一致）
  const [pickerOpen, setPickerOpen] = useState(false)
  const [pickerSearch, setPickerSearch] = useState('')
  const pickerAnchorRef = useRef<HTMLButtonElement | null>(null)
  const pickerPopRef = useRef<HTMLDivElement | null>(null)

  /** 包装 setConfig：先同步写 configRef（绕 React state flush 延迟），再调
   * setConfig 触发 React 渲染。 SchemaForm.onChange / 任何想改 config 的入口
   * 都要走这个，不要直接 setConfig。 */
  const setConfigSync = useCallback((v: ConfigData | null) => {
    configRef.current = v
    setConfig(v)
  }, [])

  const vid = activeVersion?.id ?? null

  const refreshConfig = useCallback(async () => {
    if (!vid) return
    try {
      const r = await api.getVersionConfig(project.id, vid)
      setConfigResp(r)
      setConfigSync(r.config)
      savedJsonRef.current = JSON.stringify(r.config)
    } catch (e) {
      toast(`加载训练配置失败: ${e}`, 'error')
    }
  }, [project.id, vid, toast, setConfigSync])

  /** 拉当前 version 绑的预设 config，给 picker「· 已自定义」标签做 baseline。
   * 预设可能已被删（找不到就清掉 baseline，标签自然不显示）。 */
  const refreshPresetBaseline = useCallback(async (name: string | null) => {
    if (!name) {
      presetBaselineRef.current = null
      return
    }
    try {
      presetBaselineRef.current = await api.getPreset(name)
    } catch {
      presetBaselineRef.current = null
    }
  }, [])

  // 进入页面 / 切 version 时拉一次预设 baseline；config_name 变化也跟上
  useEffect(() => {
    void refreshPresetBaseline(activeVersion?.config_name ?? null)
  }, [activeVersion?.config_name, refreshPresetBaseline])

  useEffect(() => {
    api.schema().then(setSchema).catch((e) => toast(`schema 加载失败: ${e}`, 'error'))
    api.listPresets().then(setPresets).catch(() => setPresets([]))
  }, [toast])

  useEffect(() => {
    void refreshConfig()
  }, [refreshConfig])

  // 拉 reg 状态用于显示「训练集 + 正则」分布
  useEffect(() => {
    if (!vid) return
    api.getRegStatus(project.id, vid).then(setReg).catch(() => setReg(null))
  }, [project.id, vid])


  // 全局模型路径仍然灰显 readonly（值来自 Settings.models 配置；version 维度
  // 改了没意义）。PP10.4 起项目特定字段（data_dir 等）改成可编辑：fork preset
  // 时仍然预填项目路径，但用户后续可以自由改（接续训练填 resume_lora 之类）。
  const disabledFields = GLOBAL_MODEL_FIELDS
  const disabledHints = useMemo(() => {
    const h: Record<string, string> = {}
    for (const f of GLOBAL_MODEL_FIELDS) h[f] = '自动 · 全局设置'
    return h
  }, [])
  // 项目特定字段（data_dir / reg_data_dir / output_dir 等）：值由项目预填，但
  // 不锁定，挂「自动 · 项目设置」徽章让用户知道这是预填的，不是预设里来的。
  const autoHints = useMemo(() => {
    const h: Record<string, string> = {}
    for (const f of configResp?.project_specific_fields ?? []) {
      if (!GLOBAL_MODEL_FIELDS.includes(f)) h[f] = '自动 · 项目设置'
    }
    return h
  }, [configResp?.project_specific_fields])

  /** 把项目特定字段（data_dir / reg_data_dir 等）从 config 里拿掉再 JSON。
   * picker「· 已自定义」标签比对预设原值时要排除这几个：fork 时项目预填会把
   * 它们覆盖成项目路径，跟预设原值天然不一样，但这不算用户自己改了。 */
  const stripProjectFields = useCallback((cfg: ConfigData | null): string => {
    if (!cfg) return ''
    const skip = new Set(configResp?.project_specific_fields ?? [])
    const filtered: ConfigData = {}
    for (const k of Object.keys(cfg)) {
      if (!skip.has(k)) filtered[k] = cfg[k]
    }
    return JSON.stringify(filtered)
  }, [configResp?.project_specific_fields])

  /** 当前 config 是否相对预设原值有自定义改动（picker 标签用）。
   * 注意 presetBaselineRef 是 ref 不进 deps 数组；它只在 fork 时变，那时
   * config 也会跟着变，依赖 config 重算就够。 */
  const customized = useMemo(() => {
    if (!config || !presetBaselineRef.current) return false
    return stripProjectFields(config) !== stripProjectFields(presetBaselineRef.current)
  }, [config, stripProjectFields])

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
      void persistConfig(config).catch((e) => toast(`保存失败：${e}`, 'error'))
    }, 600)
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
        debounceTimerRef.current = null
      }
    }
  }, [config, persistConfig, toast])

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

  // popover 关闭：点外面 / Esc
  useEffect(() => {
    if (!pickerOpen) return
    const onDocClick = (e: MouseEvent) => {
      const t = e.target as Node
      if (pickerPopRef.current?.contains(t) || pickerAnchorRef.current?.contains(t)) return
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
    return <p className="text-fg-tertiary p-6">请先选择 / 创建一个版本</p>
  }

  const onForkPreset = async (name: string) => {
    if (!name) return
    if (
      configResp?.has_config &&
      !window.confirm(
        `换预设会覆盖当前 version 的配置（已保存的内容会丢失）。继续？`
      )
    ) {
      return
    }
    setBusy(true)
    try {
      await api.forkPresetForVersion(project.id, vid, name)
      await refreshConfig()
      toast(`已从预设 ${name} 复制`, 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const onSaveAsPreset = async () => {
    const name = window.prompt(
      '保存为新预设。预设名（字母 / 数字 / _ / -，会清掉项目特定字段）：',
      ''
    )
    if (!name) return
    setBusy(true)
    try {
      await api.saveVersionConfigAsPreset(project.id, vid, name, false)
      const list = await api.listPresets()
      setPresets(list)
      toast(`已保存为预设 ${name}`, 'success')
    } catch (e) {
      const msg = String(e)
      if (msg.includes('已存在')) {
        if (window.confirm(`预设 ${name} 已存在，覆盖？`)) {
          try {
            await api.saveVersionConfigAsPreset(project.id, vid, name, true)
            const list = await api.listPresets()
            setPresets(list)
            toast(`已覆盖预设 ${name}`, 'success')
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

  const onEnqueue = async () => {
    if (!configResp?.has_config) {
      toast('先选预设', 'error')
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
      const t = await api.enqueueVersionTraining(project.id, vid)
      toast(`已入队 #${t.id}，去 /queue 查看进度`, 'success')
      void reload()
      navigate('/queue')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <StepShell
      idx={6}
      title="训练"
      subtitle="选预设 → 编辑 config → 入队训练"
      actions={
        <button
          onClick={() => void onEnqueue()}
          disabled={busy || !configResp?.has_config}
          className="btn btn-primary"
        >
          开始训练
        </button>
      }
    >
      <div className="flex flex-col h-full gap-3">

        {/* 两栏布局：左（预设 + config 编辑） / 右（估算面板） */}
        <div className="grid grid-cols-[1.5fr_1fr] gap-3 flex-1 min-h-0">

          {/* 左栏 */}
          <div className="flex flex-col gap-3 min-h-0 min-w-0 overflow-y-auto">

          {/* 预设 picker：dropdown 取代「当前预设条 + 可用预设网格」两块。
              点击展开 popover 含搜索 + 卡片网格，跟全局 Presets 页一致。 */}
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
              title="切换预设"
            >
              <span className="text-[10px] uppercase tracking-[0.08em] text-fg-tertiary font-semibold">
                预设
              </span>
              <span className={[
                'font-mono text-md font-semibold flex-1 text-left truncate',
                configResp?.has_config ? 'text-fg-primary' : 'text-fg-tertiary',
              ].join(' ')}>
                {activeVersion.config_name ?? '(未选)'}
                {customized && (
                  <span
                    className="ml-2 text-xs text-warn font-normal"
                    title="此版本基于预设拷贝。修改只影响当前版本，不影响预设池里的原值。"
                  >
                    · 已自定义
                  </span>
                )}
              </span>
              <span className="text-fg-tertiary text-md">▾</span>
            </button>
            <button
              onClick={() => void onSaveAsPreset()}
              disabled={busy || !configResp?.has_config}
              className="btn btn-ghost btn-sm"
              title="把当前 version 配置另存为一个全局预设"
            >
              另存为新预设
            </button>

            {/* popover */}
            {pickerOpen && (
              <div
                ref={pickerPopRef}
                role="dialog"
                aria-label="切换预设"
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
                      placeholder="筛选预设…"
                      value={pickerSearch}
                      onChange={(e) => setPickerSearch(e.target.value)}
                    />
                  </span>
                </div>

                {/* grid */}
                <div className="flex-1 min-h-0 overflow-y-auto p-2.5">
                  {filteredPresets.length > 0 ? (
                    <div className="grid grid-cols-2 gap-2">
                      {filteredPresets.map((p) => {
                        const active = p.name === activeVersion.config_name
                        return (
                          <button
                            key={p.name}
                            onClick={() => { setPickerOpen(false); void onForkPreset(p.name) }}
                            disabled={busy}
                            className={[
                              'rounded-sm px-2.5 py-2 text-left border transition-colors',
                              active
                                ? 'border-accent bg-accent-soft'
                                : 'border-subtle bg-sunken hover:border-bold',
                              busy ? 'cursor-default' : 'cursor-pointer',
                            ].join(' ')}
                          >
                            <div className={[
                              'text-sm font-mono font-semibold truncate',
                              active ? 'text-accent' : 'text-fg-primary',
                            ].join(' ')}>{p.name}</div>
                            <div className="text-xs text-fg-tertiary mt-0.5">
                              {active ? '当前使用' : '点击套用'}
                            </div>
                          </button>
                        )
                      })}
                    </div>
                  ) : (
                    <div className="text-fg-tertiary text-sm text-center py-4">
                      {pickerSearch
                        ? `没有匹配「${pickerSearch}」`
                        : '尚无预设，去 /tools/presets 创建'}
                    </div>
                  )}
                </div>
              </div>
            )}
          </section>

            {configResp === null || !schema ? (
              <ConfigSkeleton />
            ) : !configResp.has_config ? (
              <div className="flex-1 flex items-center justify-center text-fg-tertiary text-sm rounded-md border border-dashed border-dim">
                请从上方预设卡片选择一个，复制进当前 version 后即可编辑配置。
              </div>
            ) : config ? (
              <section className="flex-1 min-h-0 overflow-y-auto pr-1">
                <SchemaForm
                  schema={schema}
                  values={config}
                  onChange={setConfigSync}
                  disabledFields={disabledFields}
                  disabledHints={disabledHints}
                  autoHints={autoHints}
                />
              </section>
            ) : (
              <ConfigSkeleton />
            )}
          </div>

        {/* 右栏：训练集 + 正则集分布 */}
        <DatasetStatsPanel
          activeVersion={activeVersion}
          reg={reg}
          config={config}
        />
      </div>
    </div>
    </StepShell>
  )
}

/** Kohya 风格文件夹名「N_label」→ {repeat=N, label}。无前缀数字默认 1。 */
function parseFolderRepeat(name: string): { repeat: number; label: string } {
  const m = name.match(/^(\d+)_(.*)$/)
  if (m) return { repeat: parseInt(m[1], 10), label: m[2] }
  return { repeat: 1, label: name }
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
  activeVersion,
  reg,
  config,
}: {
  activeVersion: Version | null
  reg: RegStatus | null
  config: ConfigData | null
}) {
  const trainFolders = activeVersion?.stats?.train_folders ?? []
  const regFolders = useMemo(
    () => (reg && reg.exists ? aggregateRegFolders(reg.files) : []),
    [reg]
  )

  const trainEffective = trainFolders.reduce(
    (s, f) => s + parseFolderRepeat(f.name).repeat * f.image_count,
    0,
  )
  const regEffective = regFolders.reduce(
    (s, f) => s + parseFolderRepeat(f.name).repeat * f.image_count,
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
          <span className="caption uppercase tracking-[0.06em] text-xs">训练集参数</span>
        </div>

        <FolderSection
          title="train/"
          folders={trainFolders}
          effective={trainEffective}
          empty="无训练图"
        />

        <div className="h-2" />

        <FolderSection
          title="reg/"
          folders={regFolders}
          effective={regEffective}
          empty={reg && !reg.exists ? '未生成' : '无正则图'}
        />

        {/* 总计 + 步数估算（不含 AR bucketing 误差） */}
        <div className="mt-2.5 pt-2 border-t border-subtle flex flex-col gap-1 text-xs">
          <Row label="有效样本/epoch" value={String(totalEffective)} bold />
          {stepsPerEpoch !== null && (
            <Row
              label={`÷ batch × ga (${bs} × ${ga})`}
              value={`≈ ${stepsPerEpoch} 步/epoch`}
              dim
            />
          )}
          {naturalTotal !== null && (
            <Row
              label={`× epochs (${epochs})`}
              value={`≈ ${naturalTotal} 步`}
              dim
            />
          )}
          {finalTotal !== null && (
            <Row
              label={maxStepsTruncates ? `max_steps 上限 ${maxSteps}` : '总步数'}
              value={`≈ ${finalTotal}`}
              bold
            />
          )}
        </div>
      </div>
    </div>
  )
}

function FolderSection({
  title,
  folders,
  effective,
  empty,
}: {
  title: string
  folders: Array<{ name: string; image_count: number }>
  effective: number
  empty: string
}) {
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
            const { repeat, label } = parseFolderRepeat(f.name)
            const eff = repeat * f.image_count
            return (
              <div
                key={f.name}
                className="flex items-baseline gap-1.5 text-xs font-mono text-fg-secondary pl-1"
                title={`${f.name}：${repeat} repeat × ${f.image_count} 图 = ${eff}`}
              >
                <span className="text-fg-tertiary">{label}</span>
                <span className="flex-1 border-b border-dotted border-subtle self-end mb-1" />
                <span>
                  <span className="text-accent">{repeat}</span>
                  <span className="text-fg-tertiary"> × </span>
                  <span className="text-fg-primary">{f.image_count}</span>
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


function ConfigSkeleton() {
  // 一个分组卡片：标题条 + 4-6 行字段（label + input 灰条）
  const groups = [5, 6, 4, 5]
  return (
    <section
      className="flex-1 min-h-0 overflow-y-auto pr-1 space-y-3"
      role="status"
      aria-label="加载训练配置中"
    >
      {groups.map((rows, gi) => (
        <div
          key={gi}
          className="animate-pulse rounded-md border border-subtle bg-surface p-3.5"
        >
          <div className="h-3.5 w-32 rounded-sm bg-sunken mb-2.5" />
          <div className="flex flex-col gap-2">
            {Array.from({ length: rows }).map((_, ri) => (
              <div key={ri} className="flex flex-col gap-1">
                <div className="h-2.5 w-24 rounded-sm bg-sunken opacity-70" />
                <div className="h-7 rounded-sm bg-canvas border border-subtle" />
              </div>
            ))}
          </div>
        </div>
      ))}
      <span className="sr-only">加载训练配置中...</span>
    </section>
  )
}
