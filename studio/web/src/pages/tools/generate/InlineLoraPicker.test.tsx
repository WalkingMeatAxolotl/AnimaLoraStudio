import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { type LoraCkpt } from '../../../api/client'
import InlineLoraPicker, { projectAbbr, type PickedLora } from './InlineLoraPicker'
import type { ProjectLora } from './types'
import type { LoraCatalog, LoraVersionOption } from './useLoraCatalog'

/** 用 sample（ProjectLora[]）造一个已加载好的 LoraCatalog（懒级联测试替身）：
 *  projects / versionsByPid 同步备好，fetchCkpts 返固定 ckpts。 */
function catalogFrom(samples: ProjectLora[], ckpts: LoraCkpt[]): LoraCatalog {
  const projects = Array.from(
    new Map(samples.map((s) => [s.projectId, { id: s.projectId, title: s.projectTitle }])).values(),
  )
  const versionsByPid: Record<number, LoraVersionOption[]> = {}
  for (const s of samples) {
    (versionsByPid[s.projectId] ??= []).push({ id: s.versionId, label: s.versionLabel, status: s.status })
  }
  return {
    projects,
    projectsLoading: false,
    ensureProjects: () => {},
    loadProjects: () => Promise.resolve(projects),
    versionsOf: (pid) => versionsByPid[pid],
    ensureVersions: () => {},
    fetchCkpts: () => Promise.resolve(ckpts),
  }
}

const sample: ProjectLora[] = [
  {
    projectId: 1, projectTitle: 'cute_chibi',
    versionId: 11, versionLabel: 'v3', status: 'training',
    path: '/loras/cute_chibi/v3.safetensors', createdAt: 300,
  },
  {
    projectId: 1, projectTitle: 'cute_chibi',
    versionId: 12, versionLabel: 'v2', status: 'completed',
    path: '/loras/cute_chibi/v2.safetensors', createdAt: 200,
  },
  {
    projectId: 2, projectTitle: 'noir_portrait',
    versionId: 21, versionLabel: 'v1', status: 'completed',
    path: '/loras/noir/v1.safetensors', createdAt: 100,
  },
]

const ckptsV3: LoraCkpt[] = [
  { kind: 'final', value: 0, label: 'final', path: '/loras/cute_chibi/v3/final.safetensors', mtime: 300 },
  { kind: 'step', value: 2000, label: 'step 2000', path: '/loras/cute_chibi/v3/step_2000.safetensors', mtime: 250 },
  { kind: 'step', value: 1000, label: 'step 1000', path: '/loras/cute_chibi/v3/step_1000.safetensors', mtime: 200 },
]

describe('projectAbbr', () => {
  it('extracts first 2 alphanumerics, uppercase', () => {
    expect(projectAbbr('cute_chibi')).toBe('CU')
    expect(projectAbbr('noir_portrait')).toBe('NO')
  })
  it('falls back to ?? when empty', () => {
    expect(projectAbbr('___')).toBe('??')
    expect(projectAbbr('')).toBe('??')
  })
})

describe('InlineLoraPicker — multi mode (default)', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  function renderMulti(overrides: Partial<{
    projectLoras: ProjectLora[]
    existingPaths: Set<string>
    showWeight: boolean
    ckpts: LoraCkpt[]
    live: boolean
  }> = {}) {
    const catalog = catalogFrom(overrides.projectLoras ?? sample, overrides.ckpts ?? ckptsV3)
    const onPick = vi.fn()
    const onClose = vi.fn()
    const onPickExternal = vi.fn()
    const utils = render(
      <InlineLoraPicker
        mode="multi"
        catalog={catalog}
        existingPaths={overrides.existingPaths ?? new Set()}
        showWeight={overrides.showWeight ?? true}
        live={overrides.live ?? false}
        onPick={onPick}
        onClose={onClose}
        onPickExternal={onPickExternal}
      />
    )
    return { ...utils, onPick, onClose, onPickExternal }
  }

  it('renders project + version dropdowns from projectLoras', async () => {
    renderMulti()
    expect(screen.getByLabelText('选项目')).toBeInTheDocument()
    expect(screen.getByText('cute_chibi')).toBeInTheDocument()
    expect(screen.getByText('noir_portrait')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText(/v3（训练中）/)).toBeInTheDocument())
  })

  it('auto-loads ckpts for the default project/version (as chips)', async () => {
    renderMulti()
    await waitFor(() => {
      expect(screen.getByText('final')).toBeInTheDocument()
      expect(screen.getByText('step 2000')).toBeInTheDocument()
      expect(screen.getByText('step 1000')).toBeInTheDocument()
    })
  })

  it('click toggles picked; 添加 N 个 commits + closes', async () => {
    const user = userEvent.setup()
    const { onPick, onClose } = renderMulti()
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.click(screen.getByText('step 2000').closest('button')!)
    await user.click(screen.getByText('step 1000').closest('button')!)
    expect(screen.getByText(/已选 2/)).toBeInTheDocument()
    await user.click(screen.getByText(/添加 2 个/))
    expect(onPick).toHaveBeenCalledTimes(1)
    const [picks, weight] = onPick.mock.calls[0]
    expect(picks).toHaveLength(2)
    expect(picks.map((p: PickedLora) => p.path).sort()).toEqual([
      '/loras/cute_chibi/v3/step_1000.safetensors',
      '/loras/cute_chibi/v3/step_2000.safetensors',
    ])
    expect(weight).toBe(1.0)
    expect(onClose).toHaveBeenCalled()
  })

  it('commits picks in ckpt display order, not click order (添加 N 个)', async () => {
    const user = userEvent.setup()
    const { onPick } = renderMulti()
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    // 故意乱序点击：step 1000 → final → step 2000
    await user.click(screen.getByText('step 1000').closest('button')!)
    await user.click(screen.getByText('final').closest('button')!)
    await user.click(screen.getByText('step 2000').closest('button')!)
    await user.click(screen.getByText(/添加 3 个/))
    const [picks] = onPick.mock.calls[0]
    // 输出跟随 ckpts 展示序（final → step↓），与点击先后无关
    expect(picks.map((p: PickedLora) => p.path)).toEqual([
      '/loras/cute_chibi/v3/final.safetensors',
      '/loras/cute_chibi/v3/step_2000.safetensors',
      '/loras/cute_chibi/v3/step_1000.safetensors',
    ])
  })

  it('live mode commits in display order on each toggle (XY ckpt 轴单调)', async () => {
    const user = userEvent.setup()
    const { onPick } = renderMulti({ live: true, showWeight: false })
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    // 乱序点击；live 模式每次 toggle 即时 commit，取最后一次（三个都选中）
    await user.click(screen.getByText('step 1000').closest('button')!)
    await user.click(screen.getByText('step 2000').closest('button')!)
    await user.click(screen.getByText('final').closest('button')!)
    const lastPicks = onPick.mock.calls[onPick.mock.calls.length - 1][0]
    expect(lastPicks.map((p: PickedLora) => p.path)).toEqual([
      '/loras/cute_chibi/v3/final.safetensors',
      '/loras/cute_chibi/v3/step_2000.safetensors',
      '/loras/cute_chibi/v3/step_1000.safetensors',
    ])
  })

  it('existingPaths disables the chip; click does not toggle', async () => {
    const user = userEvent.setup()
    const { onPick } = renderMulti({
      existingPaths: new Set(['/loras/cute_chibi/v3/step_2000.safetensors']),
    })
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    const btn = screen.getByText('step 2000').closest('button')!
    expect(btn).toBeDisabled()
    await user.click(btn)
    expect(onPick).not.toHaveBeenCalled()
  })

  it('shows empty state when projectLoras is empty', () => {
    renderMulti({ projectLoras: [] })
    expect(screen.getByText(/还没有训练好的 LoRA/)).toBeInTheDocument()
  })

  it('shows no-ckpt hint when version has no ckpts', async () => {
    renderMulti({ ckpts: [] })
    await waitFor(() =>
      expect(screen.getByText(/该版本没扫到 ckpt 文件/)).toBeInTheDocument()
    )
  })

  it('search filters chip list', async () => {
    const user = userEvent.setup()
    renderMulti()
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.type(screen.getByPlaceholderText('搜索 ckpt 文件名…'), '2000')
    expect(screen.queryByText('final')).not.toBeInTheDocument()
    expect(screen.queryByText('step 1000')).not.toBeInTheDocument()
    expect(screen.getByText('step 2000')).toBeInTheDocument()
  })

  it('× triggers onClose', async () => {
    const user = userEvent.setup()
    const { onClose } = renderMulti()
    await user.click(screen.getByLabelText('关闭挑选区'))
    expect(onClose).toHaveBeenCalled()
  })

  it('外部文件 triggers onPickExternal', async () => {
    const user = userEvent.setup()
    const { onPickExternal } = renderMulti()
    await user.click(screen.getByText('外部文件'))
    expect(onPickExternal).toHaveBeenCalled()
  })

  it('changing project resets picked', async () => {
    const user = userEvent.setup()
    renderMulti()
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.click(screen.getByText('step 2000').closest('button')!)
    expect(screen.getByText(/已选 1/)).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('选项目'), '2')
    expect(screen.queryByText(/已选 1/)).not.toBeInTheDocument()
  })

  it('切 project 不会用 (新 pid, 旧 vid) 拉 ckpt（回归：避免 404）', async () => {
    // sample 合法 (pid:vid)：1:11 / 1:12 / 2:21。切 project 时若 pid 已变、vid 还
    // 是旧 project 的版本就发请求，会得到 (2, 11) 这种非法组合 → 后端 404。
    const user = userEvent.setup()
    const calls: string[] = []
    const base = catalogFrom(sample, ckptsV3)
    const catalog: LoraCatalog = {
      ...base,
      fetchCkpts: (pid, vid) => { calls.push(`${pid}:${vid}`); return base.fetchCkpts(pid, vid) },
    }
    render(
      <InlineLoraPicker
        mode="multi"
        catalog={catalog}
        existingPaths={new Set()}
        showWeight
        onPick={vi.fn()}
        onClose={vi.fn()}
        onPickExternal={vi.fn()}
      />,
    )
    await waitFor(() => expect(calls).toContain('1:11'))  // 初始锚 project 1
    await user.selectOptions(screen.getByLabelText('选项目'), '2')
    await waitFor(() => expect(calls).toContain('2:21'))  // 切到 project 2 的版本
    // 关键：从没出现过 (2, 11) 这种「新 project + 旧 version」非法组合
    const valid = new Set(['1:11', '1:12', '2:21'])
    expect(calls.every((c) => valid.has(c))).toBe(true)
  })

  it('weight slider value used in onPick', async () => {
    const user = userEvent.setup()
    const { onPick } = renderMulti()
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.click(screen.getByText('step 2000').closest('button')!)
    const weightInput = screen.getByLabelText('LoRA 权重数值')
    await user.clear(weightInput)
    await user.type(weightInput, '0.75')
    await user.click(screen.getByText(/添加 1 个/))
    const [, weight] = onPick.mock.calls[0]
    expect(weight).toBe(0.75)
  })

  it('showWeight=false hides weight slider (XY axis use)', async () => {
    const user = userEvent.setup()
    renderMulti({ showWeight: false })
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.click(screen.getByText('step 2000').closest('button')!)
    expect(screen.queryByLabelText('LoRA 权重数值')).not.toBeInTheDocument()
  })
})

describe('InlineLoraPicker — single mode (controlled slot)', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  function renderSingle(overrides: Partial<{
    value: PickedLora | null
    weight: number
    ckpts: LoraCkpt[]
  }> = {}) {
    const catalog = catalogFrom(sample, overrides.ckpts ?? ckptsV3)
    const onChange = vi.fn()
    const onClose = vi.fn()
    const onPickExternal = vi.fn()
    const utils = render(
      <InlineLoraPicker
        mode="single"
        catalog={catalog}
        value={overrides.value ?? null}
        weight={overrides.weight ?? 1.0}
        onChange={onChange}
        onClose={onClose}
        onPickExternal={onPickExternal}
      />
    )
    return { ...utils, onChange, onClose, onPickExternal }
  }

  it('click ckpt chip → onChange(pick, weight) — does not auto-close', async () => {
    const user = userEvent.setup()
    const { onChange, onClose } = renderSingle()
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.click(screen.getByText('step 2000').closest('button')!)
    expect(onChange).toHaveBeenCalledTimes(1)
    const [pick, weight] = onChange.mock.calls[0]
    expect(pick).toEqual({
      path: '/loras/cute_chibi/v3/step_2000.safetensors',
      projectId: 1,
      versionId: 11,
    })
    expect(weight).toBe(1.0)
    expect(onClose).not.toHaveBeenCalled()
  })

  it('click currently-selected chip → onChange(null, weight) 反选（SidebarLoras 把 path 置空保留槽）', async () => {
    const user = userEvent.setup()
    const { onChange, onClose } = renderSingle({
      value: {
        path: '/loras/cute_chibi/v3/step_2000.safetensors',
        projectId: 1, versionId: 11,
      },
    })
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.click(screen.getByText('step 2000').closest('button')!)
    expect(onChange).toHaveBeenCalledWith(null, 1.0)
    expect(onClose).not.toHaveBeenCalled()
  })

  it('weight slider change → onChange(value, new_weight)', async () => {
    const value: PickedLora = {
      path: '/loras/cute_chibi/v3/step_2000.safetensors',
      projectId: 1, versionId: 11,
    }
    const { onChange } = renderSingle({ value, weight: 0.8 })
    await waitFor(() => expect(screen.getByLabelText('LoRA 权重数值')).toBeInTheDocument())
    const weightInput = screen.getByLabelText('LoRA 权重数值') as HTMLInputElement
    expect(weightInput.value).toBe('0.8')
    // 受控输入：用 fireEvent.change 一次性触发，不走 userEvent.type 多次 keystroke
    // （那种方式会被 props.weight 回流覆盖）
    fireEvent.change(weightInput, { target: { value: '1.2' } })
    expect(onChange).toHaveBeenCalledWith(value, 1.2)
  })

  it('× → onClose (deletes slot from parent)', async () => {
    const user = userEvent.setup()
    const { onClose } = renderSingle()
    await user.click(screen.getByLabelText('移除 LoRA'))
    expect(onClose).toHaveBeenCalled()
  })

  it('weight slider always visible in single mode (even without selection)', async () => {
    renderSingle({ value: null })
    // findBy await 让自动锚第一个项目的 effect 级联结算（避免 act 警告）
    expect(await screen.findByLabelText('LoRA 权重数值')).toBeInTheDocument()
  })
})

describe('InlineLoraPicker — controlled sync (Step 6 / 决策 #8)', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('rerender with new props.value → project/version dropdowns reflect new ids', async () => {
    const catalog = catalogFrom(sample, ckptsV3)
    const onChange = vi.fn()
    const onClose = vi.fn()
    const initialValue: PickedLora = {
      path: '/loras/cute_chibi/v3/final.safetensors',
      projectId: 1, versionId: 11,
    }
    const { rerender } = render(
      <InlineLoraPicker
        mode="single"
        catalog={catalog}
        value={initialValue}
        weight={1.0}
        onChange={onChange}
        onClose={onClose}
      />
    )
    await waitFor(() => {
      const projectSelect = screen.getAllByRole('combobox')[0] as HTMLSelectElement
      expect(projectSelect.value).toBe('1')
    })

    // 模拟历史回填：props.value 切换到另一项目 (id=2)
    const newValue: PickedLora = {
      path: '/loras/noir/v1/final.safetensors',
      projectId: 2, versionId: 21,
    }
    rerender(
      <InlineLoraPicker
        mode="single"
        catalog={catalog}
        value={newValue}
        weight={1.0}
        onChange={onChange}
        onClose={onClose}
      />
    )
    // sync useEffect 把 pid 设到 2 —— 项目下拉跟着更新
    await waitFor(() => {
      const projectSelect = screen.getAllByRole('combobox')[0] as HTMLSelectElement
      expect(projectSelect.value).toBe('2')
    })
  })

  it('value=null 时不 sync，保留 fallback 默认（projects[0] ckpts 显示）', async () => {
    const catalog = catalogFrom(sample, ckptsV3)
    const onChange = vi.fn()
    const onClose = vi.fn()
    render(
      <InlineLoraPicker
        mode="single"
        catalog={catalog}
        value={null}
        weight={1.0}
        onChange={onChange}
        onClose={onClose}
      />
    )
    // 没有受控 value 时仍能看到 projects[0] 的 ckpts（fallback 行为）
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
  })
})
