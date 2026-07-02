/** GeneratePage 端到端 smoke：mock fetch，验证 single / xy / 多 prompt+xy
 *  三个关键路径的 enqueue payload 行为。 */
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ToastProvider } from '../../components/Toast'
import GeneratePage from './Generate'
import { useMonitorProgress } from '../../lib/useMonitorProgress'

// monitorState 来自 SSE，smoke 里不驱动 —— 默认返回空 state（samples=[]，等同
// 既有行为）。#1 冻结用例单独 mockReturnValue 注入 XY samples。
vi.mock('../../lib/useMonitorProgress', () => {
  const NULL_MONITOR = { state: null }
  return { useMonitorProgress: vi.fn(() => NULL_MONITOR) }
})

const fetchMock = vi.fn()
let lastEnqueueBody: Record<string, unknown> | null = null

beforeEach(() => {
  lastEnqueueBody = null
  window.localStorage.clear()
  // 每个用例重置 monitor mock 到默认空 state（隔离 #1 用例注入的 samples）
  vi.mocked(useMonitorProgress).mockReturnValue({ state: null } as never)
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset()
  fetchMock.mockImplementation((url: string, init?: RequestInit) => {
    // catalog 懒级联：picker mount 时才拉 /api/projects（这里返回空 = no LoRAs）
    if (url.endsWith('/api/projects') && (init?.method ?? 'GET') === 'GET') {
      return Promise.resolve({
        ok: true, status: 200,
        json: async () => ({ items: [] }),
        text: async () => '{"items":[]}',
        headers: new Headers({ 'content-type': 'application/json' }),
      } as Response)
    }
    // listQueue('running') — 默认无运行中任务（个别 case 内通过 mockImplementationOnce 覆盖）
    if (url.startsWith('/api/queue') && (init?.method ?? 'GET') === 'GET') {
      return Promise.resolve({
        ok: true, status: 200,
        json: async () => ({ items: [] }),
        text: async () => '{"items":[]}',
        headers: new Headers({ 'content-type': 'application/json' }),
      } as Response)
    }
    // enqueueGenerate
    if (url.endsWith('/api/generate') && init?.method === 'POST') {
      lastEnqueueBody = JSON.parse(String(init.body))
      const taskStub = {
        id: 1, name: 'generate', config_name: 'generate', status: 'pending',
        priority: 0, created_at: 0, started_at: null, finished_at: null,
        pid: null, exit_code: null, output_dir: null, error_msg: null,
      }
      return Promise.resolve({
        ok: true, status: 200,
        json: async () => taskStub,
        text: async () => JSON.stringify(taskStub),
        headers: new Headers({ 'content-type': 'application/json' }),
      } as Response)
    }
    // 兜底 404
    return Promise.resolve({
      ok: false, status: 404,
      json: async () => null,
      text: async () => '',
      headers: new Headers(),
    } as Response)
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

function setup() {
  return render(
    <ToastProvider>
      <GeneratePage />
    </ToastProvider>
  )
}

// LoRA 数据现在懒级联（catalog）：/api/projects 只在 picker mount 时才发，不再
// 是 mount 必发。所以这里改成等页面渲染出生成按钮作为「页面就绪」信号；需要
// picker 内容的用例自己再 waitFor 对应 chip（懒加载到位）。
async function waitForInitialLorasLoad() {
  await screen.findByRole('button', { name: /开始生成/ })
}

// 正向 / 负向 textarea 现在归到左侧「提示词」分页 tab（默认 tab 是 LoRA）；
// 要操作 prompt 的用例先切到这一页。
async function openPromptsTab(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: '提示词' }))
}

describe('GeneratePage 端到端 smoke', () => {
  it('mode=single：enqueue payload 含 xy_matrix=null + 完整字段', async () => {
    const user = userEvent.setup()
    setup()

    const btn = screen.getByRole('button', { name: /开始生成/ })
    await user.click(btn)

    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    const body = lastEnqueueBody!
    expect(body.xy_matrix).toBeNull()
    expect(body.prompts).toEqual(['newest, safe, 1girl, masterpiece, best quality'])
    expect(body.count).toBe(1)
    // commit C: attention_backend 从 Generate 页移到 Settings；不再随 enqueue 发
    expect(body.attention_backend).toBeUndefined()
  })

  it('多任务（P-I）：running + pending → 排队列表带取消，提交按钮不禁用', async () => {
    const genTask = (id: number, status: string) => ({
      id, name: 'generate', config_name: 'generate', status, priority: 0,
      created_at: 0, started_at: status === 'running' ? 1 : null, finished_at: null,
      pid: null, exit_code: null, output_dir: null, error_msg: null,
    })
    const jsonOk = (body: unknown) => Promise.resolve({
      ok: true, status: 200, json: async () => body,
      text: async () => JSON.stringify(body),
      headers: new Headers({ 'content-type': 'application/json' }),
    } as Response)
    fetchMock.mockImplementation((url: string) => {
      if (url.endsWith('/api/projects')) return jsonOk({ items: [] })
      // group=live&types=generate → running #5 + pending #6,#7
      if (url.includes('group=live')) {
        return jsonOk({ items: [genTask(5, 'running'), genTask(6, 'pending'), genTask(7, 'pending')] })
      }
      // listQueue('running')（无 group，训练阻塞检测）→ 无阻塞
      if (url.startsWith('/api/queue')) return jsonOk({ items: [] })
      return Promise.resolve({ ok: false, status: 404, json: async () => null, text: async () => '', headers: new Headers() } as Response)
    })

    setup()
    // 排队小列表出现，pending #6/#7 各带取消
    await waitFor(() => expect(screen.getByTestId('generate-queue')).toBeInTheDocument())
    expect(screen.getByTestId('generate-queue-cancel-6')).toBeInTheDocument()
    expect(screen.getByTestId('generate-queue-cancel-7')).toBeInTheDocument()
    // running #5 不算 pending → 无取消项
    expect(screen.queryByTestId('generate-queue-cancel-5')).toBeNull()
    // 正在出图时提交按钮仍可点（能继续入队）
    expect(screen.getByRole('button', { name: /开始生成/ })).not.toBeDisabled()
  })

  it('mode=xy 默认 X=steps 20,25,30：按钮显示「开始生成 · 3 张」并 enqueue 正确 xy_matrix', async () => {
    const user = userEvent.setup()
    setup()

    await user.click(screen.getByRole('button', { name: 'XY 矩阵' }))

    // 按钮文案包含 cell 数
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /开始生成 · 3 张/ })).toBeInTheDocument()
    )

    await user.click(screen.getByRole('button', { name: /开始生成 · 3 张/ }))

    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    const body = lastEnqueueBody!
    const xy = body.xy_matrix as { x: { axis: string; values: number[] }; y: unknown }
    expect(xy).not.toBeNull()
    expect(xy.x.axis).toBe('steps')
    expect(xy.x.values).toEqual([20, 25, 30])
    expect(xy.y).toBeNull()
    // schema 强制 count=1（即使 UI count 字段被隐藏，前端也要把它发对）
    expect(body.count).toBe(1)
  })

  it('多 prompt 轮换功能已隐藏：只有一个 textarea，"添加 prompt"按钮不存在', async () => {
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()
    await openPromptsTab(user)
    // 单 textarea
    const promptInputs = screen.getAllByPlaceholderText('输入正向提示词…')
    expect(promptInputs.length).toBe(1)
    // 「+ 添加 prompt」按钮不再渲染
    expect(screen.queryByRole('button', { name: /添加 prompt/ })).toBeNull()
  })

  it('切到 xy 再切回 single：sidebar 已填的 prompts/seed 等保留', async () => {
    const user = userEvent.setup()
    setup()
    await openPromptsTab(user)

    const promptArea = screen.getAllByPlaceholderText('输入正向提示词…')[0]
    await user.clear(promptArea)
    await user.type(promptArea, 'my custom prompt')

    await user.click(screen.getByRole('button', { name: 'XY 矩阵' }))
    await user.click(screen.getByRole('button', { name: '单图' }))

    expect(promptArea).toHaveValue('my custom prompt')
  })

  it('训练 / reg-ai 等任务在跑时，禁用生成按钮 + 鼠标 hover tooltip 说明原因', async () => {
    // listQueue('running') 默认返 [] —— 覆盖这次返回 1 个 running task。
    // /api/queue 默认排除 generate task（client.ts:1918），所以这里返的就是
    // train / reg-ai 等抢 GPU 的任务。
    const previousImpl = fetchMock.getMockImplementation()
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      // 只覆盖 listQueue('running')（阻塞检测）；group=live&types=generate 走
      // previousImpl 返 [] —— 后端会按 types=generate 过滤掉 train，本用例没提交 generate。
      if (
        url.startsWith('/api/queue') && !url.includes('group=live')
        && (init?.method ?? 'GET') === 'GET'
      ) {
        const running = {
          id: 42, name: 'train', config_name: 'train', status: 'running',
          priority: 0, created_at: 0, started_at: 0, finished_at: null,
          pid: 1234, exit_code: null, output_dir: null, error_msg: null,
        }
        return Promise.resolve({
          ok: true, status: 200,
          json: async () => ({ items: [running] }),
          text: async () => `{"items":[${JSON.stringify(running)}]}`,
          headers: new Headers({ 'content-type': 'application/json' }),
        } as Response)
      }
      return previousImpl ? previousImpl(url, init) : Promise.resolve({
        ok: false, status: 404, json: async () => null, text: async () => '',
        headers: new Headers(),
      } as Response)
    })

    setup()

    const btn = await screen.findByRole('button', { name: /开始生成/ })
    await waitFor(() => expect(btn).toBeDisabled())
    expect(btn).toHaveAttribute('title', expect.stringContaining('#42'))
    expect(screen.getByText(/等队列 #42 完成/)).toBeInTheDocument()
  })

  it('URL ?lora= 进入时 replace 缓存 LoRA list + clamp xDraft.loraIndex', async () => {
    // 用户场景：localStorage 缓存里有旧 LoRA + xDraft 指 loraIndex=1（lora_ckpt 轴
    // 绑第 2 条 LoRA）；从项目页 "在测试中加载" 跳过来，URL 带新 LoRA。
    // 修前：append → loras=[旧, 新]（视觉拥挤）+ loraIndex=1 偶然合法；如果用户
    //   之前没第 2 条 LoRA 但 loraIndex=1（脏 state）→ axisLoraMissing。
    // 修后：replace → loras=[新]，loraIndex 越界 → clamp 到 0。
    window.localStorage.setItem(
      'studio:generate:params:v1',
      JSON.stringify({
        mode: 'single',
        prompts: ['persist'],
        negPrompt: '',
        aspect: '1:1',
        width: 1024, height: 1024,
        steps: 25, cfgScale: 4, count: 1, seed: 0,
        loras: [
          { path: 'G:/old/cached.safetensors', scale: 1, project_id: 1, version_id: 1 },
        ],
        xDraft: { axis: 'lora_ckpt', raw: 'a, b', loraIndex: 5 },
        yDraft: { axis: 'lora_scale', raw: '0.5, 1.0', loraIndex: 3 },
        datasetPick: null,
      })
    )

    const newLoraPath = 'G:/new/from_project.safetensors'
    const search = `?lora=${encodeURIComponent(newLoraPath)}&projectId=2&versionId=3`
    window.history.replaceState({}, '', `/tools/generate${search}`)

    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    // submit single，看 enqueue payload 里的 loras 只剩 URL 来的那条
    await user.click(await screen.findByRole('button', { name: /开始生成/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    const body = lastEnqueueBody!
    expect(body.lora_configs).toEqual([
      { path: newLoraPath, scale: 1.0, project_id: 2, version_id: 3 },
    ])

    // localStorage 里 xDraft/yDraft.loraIndex 应被 clamp（5/3 都越界）
    const stored = JSON.parse(window.localStorage.getItem('studio:generate:params:v1')!)
    expect(stored.xDraft.loraIndex).toBe(0)
    expect(stored.yDraft.loraIndex).toBe(0)
    // URL query 已被 replaceState 清掉
    expect(window.location.search).toBe('')
  })

  it('刷新后恢复左侧生成参数，但不恢复当前生成结果', async () => {
    const user = userEvent.setup()
    const first = setup()
    await waitForInitialLorasLoad()
    await openPromptsTab(user)

    const promptArea = screen.getAllByPlaceholderText('输入正向提示词…')[0]
    await user.clear(promptArea)
    await user.type(promptArea, 'persist me')
    await user.click(screen.getByRole('button', { name: 'XY 矩阵' }))

    first.unmount()
    setup()
    await waitForInitialLorasLoad()

    expect(screen.getAllByPlaceholderText('输入正向提示词…')[0]).toHaveValue('persist me')
    expect(screen.getByRole('button', { name: /开始生成 · 3 张/ })).toBeInTheDocument()
    expect(screen.queryByText('#1')).toBeNull()
    expect(screen.getByText('填写参数后点击「开始生成」')).toBeInTheDocument()
  })

  // ---- LoRA 列表 single / xy 完全独立（2026-05-29 修复跨 mode 串味 bug）----

  const A = { path: 'G:/a.safetensors', scale: 1, project_id: null, version_id: null }
  const B = { path: 'G:/b.safetensors', scale: 1, project_id: null, version_id: null }
  const seedPrefs = (over: Record<string, unknown>) =>
    window.localStorage.setItem(
      'studio:generate:params:v1',
      JSON.stringify({
        mode: 'single', prompts: ['x'], negPrompt: '',
        aspect: '1:1', width: 1024, height: 1024,
        steps: 25, cfgScale: 4, count: 1, seed: 0,
        xDraft: { axis: 'steps', raw: '20, 25, 30', loraIndex: null },
        yDraft: null, datasetPick: null,
        ...over,
      })
    )

  it('single 提交只用 singleLoras（不带 xyLoras）', async () => {
    seedPrefs({ mode: 'single', singleLoras: [A], xyLoras: [B] })
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    await user.click(await screen.findByRole('button', { name: /开始生成/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    expect(lastEnqueueBody!.lora_configs).toEqual([A])
    expect(lastEnqueueBody!.xy_matrix).toBeNull()
  })

  it('xy 提交不带 singleLoras，也不带未被轴引用的 xyLoras 孤儿', async () => {
    // 默认 X 轴是 steps（不引用任何 LoRA）。singleLoras=[A] 不该泄漏到 xy；
    // xyLoras=[B] 是没被轴引用的孤儿（picker 切项目残留），也不该当 base 发。
    // 修前：xy 整桶发 xyLoras → lora_configs=[B]（B 叠到每个 cell）。
    // 修后：steps 轴不引用 anchor → lora_configs=[]。
    // （lora_ckpt 轴的引用/重映射逻辑由 xy.test.ts buildXYMatrix 单测覆盖，
    //  这里不 seed lora_ckpt 轴 —— picker 在无 projects 的 mock 下 mount 即清空它。）
    seedPrefs({ mode: 'xy', singleLoras: [A], xyLoras: [B] })
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    await user.click(await screen.findByRole('button', { name: /开始生成/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    expect(lastEnqueueBody!.lora_configs).toEqual([])
    expect(lastEnqueueBody!.xy_matrix).not.toBeNull()
  })

  it('老版本共享 loras 迁移：拆成 singleLoras/xyLoras 各一份，不丢已选 LoRA', async () => {
    // 老 shape 只有共享 loras=[A]（无 singleLoras/xyLoras）
    seedPrefs({ mode: 'single', loras: [A] })
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    await user.click(await screen.findByRole('button', { name: /开始生成/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    expect(lastEnqueueBody!.lora_configs).toEqual([A])

    // 落库后 shape 已迁移：两边都拿到 A
    const stored = JSON.parse(window.localStorage.getItem('studio:generate:params:v1')!)
    expect(stored.singleLoras).toEqual([A])
    expect(stored.xyLoras).toEqual([A])
  })

  // ---- 点击 XY 历史 entry 回填 sidebar 参数（含 xDraft）----
  it('点击 XY 落盘历史 → 左侧 XY 轴 dropdown 切到 LoRA + raw 写入', async () => {
    // 用户场景：当前 sidebar 在 XY mode 默认 X=steps；点 XY plot 1 历史 entry
    // 回填后 X 轴应切到 lora_ckpt + raw=basenames（picker 后续会按 basename 升级
    // 成全 path 给 daemon；这里只验 xDraft 同步进 prefs 这一步）。
    seedPrefs({ mode: 'xy' })  // 起步默认 X=steps
    const xySnapshotParams = {
      schema_version: 1,
      mode: 'xy',
      prompts: ['recall-prompt'],
      negative_prompt: 'recall-neg',
      width: 768, height: 1344,
      steps: 25, cfg_scale: 5, count: 1, seed: 7,
      loras: [
        { name: 'chen-bin_V3.7_step5500.safetensors', scale: 1,
          project_id: 19, version_id: 44 },
      ],
      xy_draft: {
        x: {
          axis: 'lora_ckpt',
          raw: 'epoch40.safetensors, epoch38.safetensors, epoch24.safetensors',
          loraIndex: 0,
        },
        y: null,
      },
      dataset_pick: null,
    }
    const diskEntry = {
      id: 'disk:abc123',
      date: '2026-06-09',
      mode: 'xy',
      folder: 'xy plot 1',
      path: '/tmp/test/2026-06-09/xy/xy plot 1',
      image_url: '/api/generate/disk/image/2026-06-09/xy/xy%20plot%201/xy%20plot.png',
      thumb_url: '/api/generate/disk/thumb/2026-06-09/xy/xy%20plot%201/xy%20plot.png?w=128',
      created_at: 1717900000,
      schema_version: 2,
      params: xySnapshotParams,
      xy_meta: {
        x_axis: 'lora_ckpt',
        y_axis: null,
        x_values: ['epoch40.safetensors', 'epoch38.safetensors', 'epoch24.safetensors'],
        y_values: [null],
        samples: [],
      },
    }
    const previousImpl = fetchMock.getMockImplementation()
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/api/generate/disk/history') && (init?.method ?? 'GET') === 'GET') {
        return Promise.resolve({
          ok: true, status: 200,
          json: async () => ({ entries: [diskEntry] }),
          text: async () => JSON.stringify({ entries: [diskEntry] }),
          headers: new Headers({ 'content-type': 'application/json' }),
        } as Response)
      }
      return previousImpl ? previousImpl(url, init) : Promise.resolve({
        ok: false, status: 404, json: async () => null, text: async () => '',
        headers: new Headers(),
      } as Response)
    })

    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    // 默认 X 轴是 steps —— 文本输入框显示 "20, 25, 30"
    const initialAxisInput = await screen.findByDisplayValue(/20, 25, 30/)
    expect(initialAxisInput).toBeInTheDocument()

    // 等历史栏的 thumbnail 出现（HistoryItem div 的 title 含 folder 名）
    const thumb = await screen.findByTitle(/xy plot 1 ·/)
    await user.click(thumb)

    // 回填后：X 轴 dropdown 切到 LoRA，raw 写入新值。
    // 因为 axis=lora_ckpt 渲染的是 AxisLoraCkptPicker（不是 text input）—— 直接
    // 看 X 轴 select 的 value（一行 select 元素，AxisCard.label='X'）。
    await waitFor(() => {
      const xLabel = screen.getAllByText('X 轴')[0]
      // AxisCard 外框跟 LoRA 槽对齐用 bg-overlay（原 bg-sunken 在 dark 下近黑显突兀）
      const card = xLabel.closest('div.bg-overlay')!
      const axisSelect = card.querySelector('select') as HTMLSelectElement
      expect(axisSelect.value).toBe('lora_ckpt')
    })
    // 原 "20, 25, 30" 文本框该消失（切到 lora_ckpt 渲染的是 picker）
    expect(screen.queryByDisplayValue(/20, 25, 30/)).not.toBeInTheDocument()
  })

  // ---- #1：XY 开始后改轴只影响下次，不串改右侧已出结果 ----
  it('XY 开始后改 X 轴：sidebar 改了但右侧结果网格冻结（30 列仍在）', async () => {
    // 注入 3 个 cell 的 XY samples（X=steps 20/25/30，无 Y 轴）
    vi.mocked(useMonitorProgress).mockReturnValue({
      state: {
        samples: [
          { path: 'cell x0 y0.png', xy: { xi: 0, yi: 0, xv: 20, yv: null } },
          { path: 'cell x1 y0.png', xy: { xi: 1, yi: 0, xv: 25, yv: null } },
          { path: 'cell x2 y0.png', xy: { xi: 2, yi: 0, xv: 30, yv: null } },
        ],
      },
    } as never)
    seedPrefs({ mode: 'xy' })  // 默认 X=steps raw "20, 25, 30"
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    // 开始生成（3 张）→ dispatch 定格本次运行态 run
    await user.click(await screen.findByRole('button', { name: /开始生成 · 3 张/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())

    // 网格渲染出 steps 三档表头（20/25/30）
    await waitFor(() => expect(screen.getByText('30')).toBeInTheDocument())

    // 取消 30 那一档：X 轴 raw "20, 25, 30" → "20, 25"
    const axisInput = screen.getByDisplayValue('20, 25, 30')
    await user.clear(axisInput)
    await user.type(axisInput, '20, 25')

    // live sidebar 确实改成了 "20, 25"（下次生成会用它）……
    await waitFor(() => expect(axisInput).toHaveValue('20, 25'))
    // ……但右侧已出结果网格冻结：仍含 "30" 表头，未被 live 编辑串改
    expect(screen.getByText('30')).toBeInTheDocument()
  })
})
