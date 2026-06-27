import { useEffect, useMemo, useRef, useState } from 'react'
import { type LoraCkpt } from '../../../api/client'
import type { LoraCatalog } from './useLoraCatalog'

function basenameOf(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}

/** 项目缩写图标（2 字符 uppercase）。SidebarLoras / 历史代码引用。 */
export function projectAbbr(title: string): string {
  const cleaned = title.replace(/[^a-zA-Z0-9]/g, '')
  return (cleaned.slice(0, 2) || '??').toUpperCase()
}

export interface PickedLora {
  path: string
  projectId: number | null
  versionId: number | null
}

interface CommonProps {
  /** 懒级联数据源：picker 自己按需拉 projects / versions / ckpts（见 useLoraCatalog） */
  catalog: LoraCatalog
  /** × 按钮回调：单选模式 = 删整个槽；多选模式 = 关 inline 面板 */
  onClose: () => void
  onPickExternal?: () => void
}

interface SingleModeProps extends CommonProps {
  mode: 'single'
  /** 当前槽绑的 ckpt（null = 槽空着）。受控。 */
  value: PickedLora | null
  /** 当前权重。受控。 */
  weight: number
  /** value/weight 任一变更都走这个回调。 */
  onChange: (next: PickedLora | null, weight: number) => void
  /** showWeight 强制为 true（单选模式 = 一个 LoRA 槽，必有权重）。 */
  showWeight?: never
  existingPaths?: never
}

interface MultiModeProps extends CommonProps {
  mode?: 'multi'
  /** 已被 caller 选过的 path（其他 LoRA 槽 / 其他 axis 占用），在 list 标 ✓ 禁用 */
  existingPaths?: Set<string>
  /** chip 切换 / 权重 / pid-vid 变更回调。
   *
   * - 普通 multi 模式：用户点「添加 N 个」commit 时触发，picker 自动 onClose
   * - live 模式：每次 chip toggle / 权重变 / pid-vid 切都立即触发（不依赖 commit 按钮）
   */
  onPick: (picks: PickedLora[], weight: number) => void
  /** XY 轴绑定下应 hide 权重（轴卡片自己有 lora_scale 控制） */
  showWeight?: boolean
  defaultWeight?: number
  /** 即时生效：每次 chip toggle 都 onPick，不再渲染「添加 N 个」commit footer。
   *  用于 XY 轴卡片这种「picker 常驻、用户期望所见即所得」的场景。 */
  live?: boolean
  /** 受控选中集合（仅 live 模式有意义）：picker 同步内部 picked 跟随这个数组。
   *
   * 元素可以是全 path 或 basename：raw 字符串 split 即可塞过来，picker 内会按
   * basename 等价匹配 ckpts 后高亮全 path。命中 basename → path 时立即 onPick
   * 回写全 path（修历史回填时 raw 是 basename 让 daemon "路径不存在"）。
   *
   * undefined = picker 用纯内部 picked state（active task / bulk-add 流程）。 */
  selectedPaths?: string[]
  /** 受控 pid/vid 初值（仅 live 模式）：与 selectedPaths 配套，让 picker mount
   *  时锚到正确 project/version，否则会 fallback 到 projects[0]。 */
  initialPid?: number | null
  initialVid?: number | null
  value?: never
  weight?: never
  onChange?: never
}

type Props = SingleModeProps | MultiModeProps

/** 内嵌 LoRA 选择器：项目 + 版本下拉 → ckpt chip 列表 → 单选 / 多选 + 权重。
 *
 * **single 模式**（受控）：一个 picker = 一个 LoRA 槽。点 chip = 切换当前槽 ckpt；
 *   再点同 chip = 取消（槽空）。weight slider 改 = 立即 onChange。× = 删槽。
 *
 * **multi 模式**（XY 轴 / bulk add）：toggle 多选 + 底部 weight + 「添加 N 个」按钮
 *   一次性 commit；commit 后 onClose 自动触发；× = 取消 inline 面板。XY 场景下
 *   传 `showWeight=false` 隐藏权重栏。
 */
export default function InlineLoraPicker(props: Props) {
  const { catalog, onClose, onPickExternal } = props
  // 解构出稳定的 loader（useCallback）+ 响应式数据，effect deps 用纯标识符。
  const { projects, ensureProjects, ensureVersions, versionsOf, fetchCkpts } = catalog
  const isSingle = props.mode === 'single'
  const showWeight = isSingle ? true : (props.showWeight ?? true)
  const existingPaths = isSingle ? new Set<string>() : (props.existingPaths ?? new Set<string>())

  // 项目下拉：picker mount 即懒拉项目列表（catalog 缓存 + 去重，不开 picker 不发）
  useEffect(() => { ensureProjects() }, [ensureProjects])

  // multi mode 受控锚定：caller 给 initialPid/Vid 时直接采纳（历史回填走这条）
  const multiInitialPid = !isSingle ? (props as MultiModeProps).initialPid ?? null : null
  const multiInitialVid = !isSingle ? (props as MultiModeProps).initialVid ?? null : null
  // 初始 pid/vid：single 用 value 的；multi 用 initialPid/Vid 兜底。
  // 懒级联下项目列表 mount 时还没到 → 起步 null；projects 到了再由下方 effect
  // 自动锚第一个项目（保留「打开 picker 即看到第一个项目 ckpts」的既有 UX）。
  const initialPid = isSingle
    ? (props.value?.projectId ?? null)
    : (multiInitialPid ?? null)
  const initialVid = isSingle
    ? (props.value?.versionId ?? null)
    : (multiInitialVid ?? null)

  const [pid, setPid] = useState<number | null>(initialPid)

  // pid 定下来（含 anchor 回填 / 用户选）→ 懒拉该项目的 versions
  useEffect(() => {
    if (pid != null) ensureVersions(pid)
  }, [ensureVersions, pid])

  // 决策 #8（plan §9.2）：single 模式是受控的，pid 必须跟 props.value.projectId
  // 同步 —— 否则历史回填 / URL ?lora= 流回新 LoraEntry 时，下拉框还卡在
  // 旧值（之前靠父级 bump urlConsumedKey 强制 remount 兜底，Step 6 砍掉）。
  // setPid 函数式更新 + 值未变跳过自动免无限循环。
  // value=null 时不 sync（保留 fallback：未选 LoRA 时给用户看 projects[0] ckpts）
  const singleValue = isSingle ? props.value : null
  useEffect(() => {
    if (!isSingle || singleValue == null) return
    const next = singleValue.projectId
    setPid((cur) => (cur === next ? cur : next))
  }, [isSingle, singleValue])

  // multi mode：当 caller 给 initialPid（XY 历史回填），pid 跟着 prop 走
  useEffect(() => {
    if (isSingle || multiInitialPid == null) return
    setPid((cur) => (cur === multiInitialPid ? cur : multiInitialPid))
  }, [isSingle, multiInitialPid])

  // 无 anchor（single value=null / multi 无 initialPid）时，projects 懒加载到位后
  // 自动锚第一个项目 —— 保留既有 UX：打开 picker 即看到第一个项目的 ckpts（不必
  // 先手选项目）。有 anchor 时上面两个 sync effect 接手，这里跳过。
  const hasAnchor = isSingle ? singleValue != null : multiInitialPid != null
  useEffect(() => {
    if (hasAnchor) return
    if (projects.length === 0) return
    setPid((cur) => (cur != null ? cur : projects[0].id))
  }, [hasAnchor, projects])

  // 版本下拉：来自 catalog（pid 定后由上面 effect 懒拉）。还没到时为空数组。
  const versions = useMemo(
    () => (pid != null ? versionsOf(pid) ?? [] : []),
    [versionsOf, pid],
  )

  const [vid, setVid] = useState<number | null>(initialVid)
  // 同 pid：single 模式下 value 非 null 时 vid 跟 props.value.versionId 同步
  useEffect(() => {
    if (!isSingle || singleValue == null) return
    const next = singleValue.versionId
    setVid((cur) => (cur === next ? cur : next))
  }, [isSingle, singleValue])

  // multi mode：受控 vid（同 multiInitialPid 一对儿）
  useEffect(() => {
    if (isSingle || multiInitialVid == null) return
    setVid((cur) => (cur === multiInitialVid ? cur : multiInitialVid))
  }, [isSingle, multiInitialVid])
  // versions 切换时如果当前 vid 不在新 versions 列表里，自动取第一个
  // （用户切 project 下拉时触发）
  useEffect(() => {
    if (versions.length === 0) {
      setVid((cur) => (cur === null ? cur : null))
    } else if (!versions.some((v) => v.id === vid)) {
      setVid(versions[0].id)
    }
  }, [versions, vid])

  // 拉 ckpt
  const [ckpts, setCkpts] = useState<LoraCkpt[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (pid === null || vid === null) {
      setCkpts([])
      return
    }
    // vid 必须确属于当前 pid 的 versions 才拉 ckpt。切 project 后有一拍 pid 已是
    // 新值、vid 还停在旧 project 的版本（且 versions(newPid) 可能还没懒加载到）——
    // 此时跳过，等 versions 到位 + auto-vid effect 把 vid 纠正到新 project 的版本，
    // 否则会用 (newPid, oldVid) 发请求触发 404。
    if (!versions.some((v) => v.id === vid)) {
      setCkpts([])
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    void fetchCkpts(pid, vid)
      .then((items) => {
        if (cancelled) return
        setCkpts(items)
        setLoading(false)
      })
      .catch((e) => {
        if (cancelled) return
        setError(e instanceof Error ? e.message : String(e))
        setCkpts([])
        setLoading(false)
      })
    return () => { cancelled = true }
  }, [pid, vid, versions, fetchCkpts])

  // 搜索过滤
  const [search, setSearch] = useState('')
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return ckpts
    return ckpts.filter((c) =>
      c.label.toLowerCase().includes(q) || c.path.toLowerCase().includes(q)
    )
  }, [ckpts, search])

  // 权重：single 模式受控；multi 模式内部状态
  const [internalWeight, setInternalWeight] = useState<number>(
    isSingle ? props.weight : (props.mode === 'multi' ? props.defaultWeight ?? 1.0 : 1.0)
  )
  // single 模式 weight 跟着 props 走；multi 模式 singleWeight 恒为 0，不会触发同步
  const singleWeight = isSingle ? props.weight : 0
  useEffect(() => {
    if (isSingle) setInternalWeight(singleWeight)
  }, [isSingle, singleWeight])

  // multi 模式的当前会话选中（single 模式不用，受控走 props.value）
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const isLive = !isSingle && (props as MultiModeProps).live === true
  // 受控选中（live 模式专用）：caller 给 selectedPaths 时 picker 以它为单一来源
  const multiSelectedPaths = !isSingle ? (props as MultiModeProps).selectedPaths : undefined
  const isControlled = !isSingle && multiSelectedPaths !== undefined
  // 区分"prop 同步导致的 pid/vid 变化"vs"用户手点下拉"—— 前者不应清 axis
  const lastPropPidVid = useRef({ pid: multiInitialPid, vid: multiInitialVid })
  useEffect(() => {
    lastPropPidVid.current = { pid: multiInitialPid, vid: multiInitialVid }
  }, [multiInitialPid, multiInitialVid])
  useEffect(() => {
    if (isSingle) return
    // controlled：prop 同步触发的 pid/vid 变化不清 picked（让 selectedPaths sync 接手）
    if (isControlled && pid === lastPropPidVid.current.pid && vid === lastPropPidVid.current.vid) return
    setPicked(new Set())  // pid/vid 切换时清空 UI 选中
    // live 模式：pid/vid 切了把 axis 也清掉（新版本下旧 path 已无意义）；
    // 初始 mount 也会跑一次，但此时 draft.raw 通常已是空，commit 空集合即 no-op。
    if (isLive) {
      (props as MultiModeProps).onPick([], internalWeight)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pid, vid, isSingle, isControlled])

  // 受控同步：selectedPaths + ckpts 决定 picked。
  //   - selectedPaths 元素是全 path → 直接 hit
  //   - 是 basename（历史回填基础形态，避免快照泄露绝对路径） → 按 basename 在 ckpts 里找匹配
  //     全 path，picked 用全 path；同时立刻 onPick 回写让 raw 升级成全 path，
  //     修 daemon 拿到 basename 触发"LoRA 路径不存在"。
  useEffect(() => {
    if (!isControlled || !multiSelectedPaths || loading) return
    if (pid === null || vid === null) return
    const ckptByPath = new Map(ckpts.map((c) => [c.path, c]))
    const ckptByBasename = new Map<string, LoraCkpt>()
    for (const c of ckpts) ckptByBasename.set(basenameOf(c.path), c)
    const resolvedPaths: string[] = []
    let needUpgrade = false
    for (const v of multiSelectedPaths) {
      if (!v) continue
      if (ckptByPath.has(v)) {
        resolvedPaths.push(v)
        continue
      }
      const ck = ckptByBasename.get(basenameOf(v))
      if (ck) {
        resolvedPaths.push(ck.path)
        needUpgrade = true  // raw 里是 basename / 旧 path，要升级到当前机器上的全 path
      }
      // 找不到 → 当前 ckpts 没扫到，跳过（用户可能换了项目或文件被删）
    }
    const resolvedSet = new Set(resolvedPaths)
    // 避免无限循环：picked 实际不变时不 setState
    setPicked((prev) => {
      if (prev.size === resolvedSet.size && [...prev].every((p) => resolvedSet.has(p))) return prev
      return resolvedSet
    })
    if (needUpgrade && isLive) {
      const picks = ckpts
        .filter((c) => resolvedSet.has(c.path))
        .map((c) => ({ path: c.path, projectId: pid, versionId: vid }))
      ;(props as MultiModeProps).onPick(picks, internalWeight)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isControlled, multiSelectedPaths, ckpts, loading, pid, vid, isLive])

  const currentVersion = versions.find((v) => v.id === vid)

  // 选中集合 → picks：按 ckpts 展示顺序排（list_lora_ckpts 的 canonical sort：
  // final → step↓ → epoch↓），而非用户点击顺序。XY ckpt 轴必须单调，否则
  // 网格列/行随点击先后乱跳，读不出过拟合拐点（ep60 应恒在 ep80 / ep40 之间）。
  const orderedPicks = (sel: Set<string>): PickedLora[] =>
    ckpts
      .filter((c) => sel.has(c.path))
      .map((c) => ({ path: c.path, projectId: pid, versionId: vid }))

  // chip 点击
  const onChipClick = (c: LoraCkpt) => {
    if (existingPaths.has(c.path)) return
    if (isSingle) {
      const { value } = props
      const isCurrent = value && value.path === c.path
      if (isCurrent) {
        // 反选：槽内 ckpt 清空（视觉等同初次打开的空槽）；SidebarLoras 收到
        // null 不会删整个槽，只把 path 置空（× 才删槽）。
        props.onChange(null, internalWeight)
        return
      }
      props.onChange(
        { path: c.path, projectId: pid, versionId: vid },
        internalWeight,
      )
      return
    }
    setPicked((s) => {
      const next = new Set(s)
      if (next.has(c.path)) next.delete(c.path); else next.add(c.path)
      // live 模式：每次 chip toggle 都即时 commit，不等用户点「添加 N 个」
      if (isLive && pid !== null && vid !== null) {
        ;(props as MultiModeProps).onPick(orderedPicks(next), internalWeight)
      }
      return next
    })
  }

  const onWeightChange = (w: number) => {
    if (isSingle) {
      props.onChange(props.value, w)
    } else {
      setInternalWeight(w)
      if (isLive && pid !== null && vid !== null && picked.size > 0) {
        ;(props as MultiModeProps).onPick(orderedPicks(picked), w)
      }
    }
  }

  const commitMulti = () => {
    if (isSingle) return
    if (picked.size === 0) return
    props.onPick(orderedPicks(picked), internalWeight)
    onClose()
  }

  // single 模式：选中的 ckpt path（用于 chip 高亮）
  const selectedPath = isSingle ? props.value?.path ?? null : null

  return (
    <div
      className="rounded-md border border-subtle bg-overlay p-2.5 flex flex-col gap-2"
      data-testid="inline-lora-picker"
    >
      {/* header */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-fg-secondary shrink-0">选 LoRA</span>
        <span className="flex-1" />
        {onPickExternal && (
          <button
            onClick={onPickExternal}
            className="btn btn-ghost btn-sm text-2xs text-fg-tertiary"
            title="选系统中任意 .safetensors 文件"
          >
            外部文件
          </button>
        )}
        <button
          onClick={onClose}
          className="btn btn-ghost btn-sm text-fg-tertiary px-1.5"
          title={isSingle ? '移除这个 LoRA 槽' : '关闭面板'}
          aria-label={isSingle ? '移除 LoRA' : '关闭挑选区'}
        >
          ×
        </button>
      </div>

      {/* project / version 下拉 */}
      <div className="flex gap-2">
        <select
          className="input text-xs flex-1"
          value={pid ?? ''}
          onChange={(e) => setPid(e.target.value ? Number(e.target.value) : null)}
          aria-label="选项目"
        >
          <option value="">选项目…</option>
          {projects.map((p) => (
            <option key={p.id} value={p.id}>{p.title}</option>
          ))}
        </select>
        <select
          className="input text-xs flex-1"
          value={vid ?? ''}
          onChange={(e) => setVid(e.target.value ? Number(e.target.value) : null)}
          disabled={versions.length === 0}
          aria-label="选版本"
        >
          <option value="">选版本…</option>
          {versions.map((v) => (
            <option key={v.id} value={v.id}>
              {v.label}{v.status === 'training' ? '（训练中）' : ''}
            </option>
          ))}
        </select>
      </div>

      {/* search */}
      <input
        type="text"
        className="input text-xs"
        placeholder="搜索 ckpt 文件名…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        disabled={!pid || !vid || ckpts.length === 0}
      />

      {error && <div className="text-2xs text-err">{error}</div>}
      {currentVersion?.status === 'training' && (
        <div className="text-2xs text-fg-tertiary">
          <span className="badge badge-info" style={{ fontSize: 10, marginRight: 4 }}>训练中</span>
          ckpt 列表会随训练进度刷新
        </div>
      )}

      {/* ckpt chip 列表 —— 等宽网格（auto-fill）：名字长短不一时也对齐成整齐的列，
          长名在格内 truncate + title 看全名，避免散乱的 ragged 流式排布。 */}
      <div
        className="grid gap-1.5 overflow-y-auto"
        style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', maxHeight: 280, padding: 2 }}
      >
        {loading && <div className="text-2xs text-fg-tertiary px-1 py-2" style={{ gridColumn: '1 / -1' }}>加载中…</div>}
        {!loading && projects.length === 0 && (
          <div className="text-fg-tertiary text-xs px-1 py-4 text-center" style={{ gridColumn: '1 / -1' }}>
            还没有训练好的 LoRA —— 先去训练一个{onPickExternal ? '，或用「外部文件」' : ''}
          </div>
        )}
        {!loading && projects.length > 0 && pid !== null && vid !== null && ckpts.length === 0 && !error && (
          <div className="text-2xs text-fg-tertiary px-1 py-4 text-center" style={{ gridColumn: '1 / -1' }}>
            该版本没扫到 ckpt 文件
          </div>
        )}
        {!loading && filtered.map((c) => {
          const isExisting = existingPaths.has(c.path)
          const isPicked = isSingle ? c.path === selectedPath : picked.has(c.path)
          const marker = isExisting ? '✓' : (isPicked ? '✓' : '+')
          return (
            <button
              key={c.path}
              type="button"
              onClick={() => onChipClick(c)}
              disabled={isExisting}
              className="font-mono flex items-center gap-1 min-w-0"
              style={{
                fontSize: 11,
                padding: '4px 8px',
                borderRadius: 'var(--r-md)',
                border: isPicked
                  ? '1px solid transparent'
                  : (isExisting ? '1px dashed var(--border-default)' : '1px solid var(--border-subtle)'),
                background: isExisting
                  ? 'var(--bg-sunken)'
                  : (isPicked ? 'var(--accent-soft)' : 'var(--bg-sunken)'),
                color: isExisting
                  ? 'var(--fg-tertiary)'
                  : (isPicked ? 'var(--accent)' : 'var(--fg-secondary)'),
                cursor: isExisting ? 'not-allowed' : 'pointer',
              }}
              title={c.path}
            >
              <span className="shrink-0">{marker}</span>
              <span className="truncate flex-1 text-left">{c.label}</span>
            </button>
          )
        })}
        {!loading && ckpts.length > 0 && filtered.length === 0 && (
          <div className="text-fg-tertiary text-xs px-1 py-4 text-center" style={{ gridColumn: '1 / -1' }}>没有匹配的 ckpt</div>
        )}
      </div>

      {/* 权重 slider —— single 模式恒显；multi 模式按 showWeight + 有选时显 */}
      {showWeight && (isSingle || picked.size > 0) && (
        <div
          className="flex items-center gap-2 pt-1"
          style={{ borderTop: '1px solid var(--border-subtle)' }}
        >
          <span
            className="font-mono text-fg-tertiary shrink-0"
            style={{ fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase' }}
          >
            权重
          </span>
          <input
            type="range"
            min={0}
            max={1.5}
            step={0.05}
            value={internalWeight}
            onChange={(e) => onWeightChange(Number(e.target.value))}
            className="flex-1"
            aria-label="LoRA 权重"
            style={{ accentColor: 'var(--accent)' }}
          />
          <input
            type="number"
            min={0}
            max={1.5}
            step={0.05}
            value={internalWeight}
            onChange={(e) => onWeightChange(Number(e.target.value))}
            className="input font-mono text-center"
            style={{ width: 54, padding: '3px 6px', fontSize: 12 }}
            aria-label="LoRA 权重数值"
          />
        </div>
      )}

      {/* multi 模式：commit footer（live 模式下不渲染，chip 即所见即所得） */}
      {!isSingle && !isLive && picked.size > 0 && (
        <div className="flex items-center gap-2 justify-end">
          <span className="text-2xs text-fg-tertiary mr-auto">已选 {picked.size}</span>
          <button
            onClick={() => setPicked(new Set())}
            className="btn btn-ghost btn-sm text-xs"
          >
            取消
          </button>
          <button
            onClick={commitMulti}
            className="btn btn-primary btn-sm text-xs"
          >
            添加 {picked.size} 个
          </button>
        </div>
      )}
    </div>
  )
}
