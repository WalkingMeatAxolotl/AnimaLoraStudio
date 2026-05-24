import { useEffect, useMemo, useState } from 'react'
import { api, type LoraCkpt } from '../../../api/client'
import type { ProjectLora } from './types'

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
  projectLoras: ProjectLora[]
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
  const { projectLoras, onClose, onPickExternal } = props
  const isSingle = props.mode === 'single'
  const showWeight = isSingle ? true : (props.showWeight ?? true)
  const existingPaths = isSingle ? new Set<string>() : (props.existingPaths ?? new Set<string>())

  // 项目下拉：projectLoras 去重 by projectId
  const projects = useMemo(() => {
    const map = new Map<number, { id: number; title: string }>()
    for (const l of projectLoras) {
      if (!map.has(l.projectId)) map.set(l.projectId, { id: l.projectId, title: l.projectTitle })
    }
    return Array.from(map.values())
  }, [projectLoras])

  // 初始 pid/vid：single 模式优先用 value 的；否则取第一个项目
  const initialPid = isSingle ? (props.value?.projectId ?? projects[0]?.id ?? null) : (projects[0]?.id ?? null)
  const initialVid = isSingle
    ? (props.value?.versionId ?? null)
    : (projectLoras.find((l) => l.projectId === projects[0]?.id)?.versionId ?? null)

  const [pid, setPid] = useState<number | null>(initialPid)

  const versions = useMemo(() => {
    if (pid === null) return []
    return projectLoras
      .filter((l) => l.projectId === pid)
      .map((l) => ({ id: l.versionId, label: l.versionLabel, status: l.status }))
  }, [projectLoras, pid])

  const [vid, setVid] = useState<number | null>(initialVid)
  useEffect(() => {
    if (versions.length === 0) {
      setVid(null)
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
    let cancelled = false
    setLoading(true)
    setError(null)
    void api.listVersionLoraCkpts(pid, vid)
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
  }, [pid, vid])

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
  useEffect(() => {
    if (isSingle) return
    setPicked(new Set())  // pid/vid 切换时清空 UI 选中
    // live 模式：pid/vid 切了把 axis 也清掉（新版本下旧 path 已无意义）；
    // 初始 mount 也会跑一次，但此时 draft.raw 通常已是空，commit 空集合即 no-op。
    if (isLive) {
      (props as MultiModeProps).onPick([], internalWeight)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pid, vid, isSingle])

  const currentVersion = versions.find((v) => v.id === vid)

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
        const picks: PickedLora[] = Array.from(next).map((path) => ({
          path, projectId: pid, versionId: vid,
        }))
        ;(props as MultiModeProps).onPick(picks, internalWeight)
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
        const picks: PickedLora[] = Array.from(picked).map((path) => ({
          path, projectId: pid, versionId: vid,
        }))
        ;(props as MultiModeProps).onPick(picks, w)
      }
    }
  }

  const commitMulti = () => {
    if (isSingle) return
    if (picked.size === 0) return
    const picks: PickedLora[] = Array.from(picked).map((path) => ({
      path,
      projectId: pid,
      versionId: vid,
    }))
    props.onPick(picks, internalWeight)
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

      {/* ckpt chip 列表 */}
      <div className="flex flex-wrap gap-1.5 overflow-y-auto" style={{ maxHeight: 280, padding: 2 }}>
        {loading && <div className="text-2xs text-fg-tertiary px-1 py-2">加载中…</div>}
        {!loading && projects.length === 0 && (
          <div className="text-fg-tertiary text-xs px-1 py-4 text-center w-full">
            还没有训练好的 LoRA —— 先去训练一个{onPickExternal ? '，或用「外部文件」' : ''}
          </div>
        )}
        {!loading && projects.length > 0 && pid !== null && vid !== null && ckpts.length === 0 && !error && (
          <div className="text-2xs text-fg-tertiary px-1 py-4 text-center w-full">
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
              className="font-mono inline-flex items-center gap-1"
              style={{
                fontSize: 11,
                padding: '3px 10px',
                borderRadius: 999,
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
                whiteSpace: 'nowrap',
              }}
              title={c.path}
            >
              <span>{marker}</span>
              <span>{c.label}</span>
            </button>
          )
        })}
        {!loading && ckpts.length > 0 && filtered.length === 0 && (
          <div className="text-fg-tertiary text-xs px-1 py-4 text-center w-full">没有匹配的 ckpt</div>
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
