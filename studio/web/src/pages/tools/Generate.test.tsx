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
  await screen.findByRole('button', { name: /开始生成|生成 \d+ 张/ })
}

// 正向 / 负向 textarea 现在归到左侧「提示词」分页 tab（默认 tab 是 LoRA）；
// 要操作 prompt 的用例先切到这一页。
async function openPromptsTab(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('tab', { name: '提示词' }))
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

  it('keeps the page title, puts the accessible mode switch in the left card, and hides result/history chrome', async () => {
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    expect(screen.getByRole('heading', { name: '测试' })).toBeInTheDocument()
    expect(screen.queryByText('生成结果')).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'LoRA' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '刷新' })).not.toBeInTheDocument()

    const modeSwitch = screen.getByRole('radio', { name: '单图' })
    expect(screen.getByRole('radiogroup', { name: '生成模式' })).toBeInTheDocument()
    expect(modeSwitch).toHaveAttribute('aria-checked', 'true')
    const loraTab = screen.getByRole('tab', { name: 'LoRA' })
    expect(modeSwitch.compareDocumentPosition(loraTab) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

    modeSwitch.focus()
    await user.keyboard('{ArrowRight}')
    expect(screen.getByRole('radio', { name: 'XY 矩阵' })).toHaveFocus()
    expect(screen.getByRole('radio', { name: 'XY 矩阵' })).toHaveAttribute('aria-checked', 'true')
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
    // 右栏时间线出现 pending #6/#7（各带取消），running #5 也在（同为 live 项、可取消）
    await waitFor(() => expect(screen.getByTestId('timeline-cancel-6')).toBeInTheDocument())
    expect(screen.getByTestId('timeline-cancel-7')).toBeInTheDocument()
    expect(screen.getByTestId('timeline-cancel-5')).toBeInTheDocument()
    // 正在出图时提交按钮仍可点（能继续入队）
    expect(screen.getByRole('button', { name: /开始生成/ })).not.toBeDisabled()
  })

  it('mode=xy 默认 X=steps、Y=weight 1.0：生成按钮显示「3 张」且 enqueue 保持单维矩阵', async () => {
    const user = userEvent.setup()
    setup()

    await user.click(screen.getByRole('radio', { name: 'XY 矩阵' }))
    const secondaryTabs = screen.getByTestId('xy-axis-secondary-tabs')
    const sectionTabs = screen.getByRole('tablist', { name: '生成配置' })
    expect(secondaryTabs.compareDocumentPosition(sectionTabs) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(screen.getByRole('tab', { name: 'Y · 权重' })).toBeInTheDocument()
    const editAxisButton = screen.getByRole('button', { name: '编辑 X 轴' })
    expect(editAxisButton).toHaveTextContent('编辑 X 轴')
    expect(editAxisButton).toHaveAttribute('aria-controls', 'xy-axis-editor-drawer')
    expect(secondaryTabs).not.toContainElement(editAxisButton)
    await user.click(screen.getByRole('tab', { name: 'Y · 权重' }))
    expect(screen.getByTestId('xy-axis-selected-value')).toHaveTextContent('1.0')
    await user.click(screen.getByRole('tab', { name: 'X · 步数' }))
    // Numeric values are edited in the XY Axis Editor Drawer, not inline in
    // the summary card.
    await user.click(await screen.findByRole('button', { name: '编辑 X 轴' }))

    // cell 数归入主操作按钮，轴工具栏不再重复显示。
    expect(screen.queryByTestId('xy-image-count')).not.toBeInTheDocument()
    const generateButton = screen.getByRole('button', { name: '生成 3 张' })
    expect(generateButton).toBeInTheDocument()

    await user.click(generateButton)

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

  it('single 与 XY 模式的 LoRA tab 复用同一套既有内容', async () => {
    const user = userEvent.setup()
    setup()

    expect(screen.getByTestId('current-lora-panel')).toBeVisible()
    expect(screen.getByRole('textbox', { name: 'LoRA 文本' })).toBeVisible()

    await user.click(screen.getByRole('radio', { name: 'XY 矩阵' }))
    await user.click(screen.getByRole('tab', { name: 'LoRA' }))

    expect(screen.getByTestId('current-lora-panel')).toBeVisible()
    expect(screen.getByRole('textbox', { name: 'LoRA 文本' })).toBeVisible()
    expect(screen.queryByTestId('current-fixed-lora-panel')).not.toBeInTheDocument()
  })

  it('opens the training-set picker in the shared attached drawer instead of inline', async () => {
    const user = userEvent.setup()
    setup()

    await openPromptsTab(user)
    const trigger = screen.getByRole('button', { name: '从训练集选取' })
    expect(trigger).toHaveAttribute('aria-controls', 'prompt-dataset-drawer')
    expect(screen.queryByTestId('prompt-dataset-drawer')).not.toBeInTheDocument()

    await user.click(trigger)

    const drawer = screen.getByTestId('prompt-dataset-drawer')
    const picker = screen.getByTestId('prompt-dataset-picker')
    expect(drawer).toHaveClass('generate-attached-drawer')
    expect(drawer).toContainElement(picker)
    expect(screen.getByRole('button', { name: '收起' })).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByRole('tabpanel', { name: '提示词' })).not.toContainElement(picker)

    await user.click(screen.getByRole('button', { name: '收起' }))
    expect(screen.queryByTestId('prompt-dataset-drawer')).not.toBeInTheDocument()
  })

  it('sidebar tabs support arrow-key navigation', async () => {
    const user = userEvent.setup()
    setup()

    const loraTab = screen.getByRole('tab', { name: 'LoRA' })
    loraTab.focus()
    await user.keyboard('{ArrowRight}')

    const promptsTab = screen.getByRole('tab', { name: '提示词' })
    expect(promptsTab).toHaveFocus()
    expect(promptsTab).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tabpanel', { name: '提示词' })).toBeVisible()
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

    await user.click(screen.getByRole('radio', { name: 'XY 矩阵' }))
    await user.click(screen.getByRole('radio', { name: '单图' }))

    expect(promptArea).toHaveValue('my custom prompt')
  })

  it('训练 / reg-ai 等任务在跑时，按钮可用（提交排队）+ tooltip 说明会排队', async () => {
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
    // R-5：后端准入（R-1）已保证互斥，前端不再硬禁用——提交只是入队排队；
    // tooltip 说明当前 GPU 被 #42 占用、提交会排队。
    await waitFor(() =>
      expect(btn).toHaveAttribute('title', expect.stringContaining('#42')),
    )
    expect(btn).not.toBeDisabled()
  })

  it('URL ?lora= 进入时 replace 缓存 LoRA list + 迁移旧 checkpoint 绑定', async () => {
    // 用户场景：localStorage 缓存里有旧 LoRA + xDraft 指 loraIndex=1（lora_ckpt 轴
    // 绑第 2 条 LoRA）；从项目页 "在测试中加载" 跳过来，URL 带新 LoRA。
    // 修前：append → loras=[旧, 新]；脏索引还可能让 submit 抛 axisLoraMissing。
    // 修后：single 列表 replace 为新 LoRA；旧 XY checkpoint 索引迁移成稳定 anchor。
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

    // localStorage 里 checkpoint 轴的旧索引迁移成稳定 anchor；
    // 非 checkpoint 轴不再保留已被 schema 禁止的 loraIndex。
    const stored = JSON.parse(window.localStorage.getItem('studio:generate:params:v1')!)
    expect(stored.xDraft.loraIndex).toBeNull()
    expect(stored.xDraft.checkpointAnchor).toEqual({
      path: 'G:/old/cached.safetensors', scale: 1, project_id: 1, version_id: 1,
    })
    expect(stored.yDraft.loraIndex).toBeNull()
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
    await user.click(screen.getByRole('radio', { name: 'XY 矩阵' }))

    first.unmount()
    setup()
    await waitForInitialLorasLoad()

    expect(screen.getAllByPlaceholderText('输入正向提示词…')[0]).toHaveValue('persist me')
    expect(screen.queryByTestId('xy-image-count')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '生成 3 张' })).toBeInTheDocument()
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

  it('single sidecar：停用 LoRA 保留在 prefs，但生成和快照都只提交启用项', async () => {
    seedPrefs({
      mode: 'single',
      singleLoras: [A, B],
      singleLoraUi: [
        { id: 'stable-a', enabled: false },
        { id: 'stable-b', enabled: true },
      ],
      xyLoras: [],
    })
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    await user.click(await screen.findByRole('button', { name: /开始生成/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    expect(lastEnqueueBody!.lora_configs).toEqual([B])
    expect((lastEnqueueBody!.params_snapshot as { loras: Array<{ name: string }> }).loras)
      .toEqual([{ name: 'b.safetensors', scale: 1, project_id: null, version_id: null }])

    const stored = JSON.parse(window.localStorage.getItem('studio:generate:params:v1')!)
    expect(stored.singleLoras).toEqual([A, B])
    expect(stored.singleLoraUi).toEqual([
      { id: 'stable-a', enabled: false },
      { id: 'stable-b', enabled: true },
    ])
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

    await user.click(await screen.findByRole('button', { name: '生成 3 张' }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    expect(lastEnqueueBody!.lora_configs).toEqual([])
    expect(lastEnqueueBody!.xy_matrix).not.toBeNull()
  })

  const mockKrea2Catalog = (selected: 'raw' | 'raw_fp8', selectedTe: 'bf16' | 'fp8') => {
    const previousImpl = fetchMock.getMockImplementation()
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/api/models/catalog')) {
        const body = {
          krea2_main: {
            variants: [
              { variant: 'raw', exists: true, purpose: 'training' },
              { variant: 'raw_fp8', exists: true, purpose: 'training' },
            ],
            custom: [],
            selected,
          },
          krea2_text_encoder: { selected: selectedTe },
          krea2_text_encoder_fp8: { files: [] },
        }
        return Promise.resolve({
          ok: true, status: 200,
          json: async () => body,
          text: async () => JSON.stringify(body),
          headers: new Headers({ 'content-type': 'application/json' }),
        } as Response)
      }
      return previousImpl!(url, init)
    })
  }

  it('FP8 性能警告跟随底模，而不是文本编码器', async () => {
    seedPrefs({
      mode: 'xy',
      modelFamily: 'krea2',
      textEncoder: 'fp8',
      xDraft: { axis: 'lora_ckpt', raw: A.path, loraIndex: null, checkpointAnchor: A },
      yDraft: { axis: 'lora_scale', raw: '0.5, 1.0', loraIndex: null },
    })
    mockKrea2Catalog('raw', 'fp8')
    setup()

    await waitFor(() => expect(screen.getByLabelText('底模')).toHaveValue('raw'))
    expect(screen.queryByText(/每个单元格都可能重新合并/)).not.toBeInTheDocument()
  })

  it('FP8 底模的 checkpoint × weight 组合显示性能警告', async () => {
    seedPrefs({
      mode: 'xy',
      modelFamily: 'krea2',
      textEncoder: 'bf16',
      xDraft: { axis: 'lora_ckpt', raw: A.path, loraIndex: null, checkpointAnchor: A },
      yDraft: { axis: 'lora_scale', raw: '0.5, 1.0', loraIndex: null },
    })
    mockKrea2Catalog('raw_fp8', 'bf16')
    setup()

    expect(await screen.findByText(/每个单元格都可能重新合并/)).toBeInTheDocument()
  })

  it('xy 新草稿持久化 checkpointAnchor，不依赖旧 xyLoras 索引', async () => {
    seedPrefs({
      mode: 'xy',
      xyLoras: [],
      xyFixedLoras: [],
      xyFixedLoraUi: [],
      xDraft: {
        axis: 'lora_ckpt',
        raw: A.path,
        loraIndex: null,
        checkpointAnchor: A,
      },
    })
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    await user.click(await screen.findByRole('button', { name: '生成 1 张' }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    expect(lastEnqueueBody!.lora_configs).toEqual([A])
    expect((lastEnqueueBody!.xy_matrix as { x: { lora_index: number } }).x.lora_index).toBe(0)
    expect((lastEnqueueBody!.params_snapshot as {
      xy_draft: { x: { loraIndex: number | null } }
    }).xy_draft.x.loraIndex).toBe(0)
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
    const timelineEntry = {
      task_id: 91,
      status: 'done',
      created_at: 1717900000,
      mode: 'xy',
      storage: 'disk',
      params: xySnapshotParams,
      images: [
        {
          url: '/api/generate/disk/image/2026-06-09/xy/xy%20plot%201/cell%20x0%20y0.png',
          thumb_url: '/api/generate/disk/thumb/2026-06-09/xy/xy%20plot%201/cell%20x0%20y0.png?w=128',
          xi: 0, yi: 0,
        },
      ],
      available: true,
      xy_folder: 'xy plot 1',
    }
    const previousImpl = fetchMock.getMockImplementation()
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes('/api/generate/timeline') && (init?.method ?? 'GET') === 'GET') {
        const body = { entries: [timelineEntry], total: 1, offset: 0 }
        return Promise.resolve({
          ok: true, status: 200,
          json: async () => body,
          text: async () => JSON.stringify(body),
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
    // Numeric values are edited in the XY Axis Editor Drawer; the sidebar only
    // renders the compact summary card.
    await user.click(await screen.findByRole('button', { name: '编辑 X 轴' }))

    // 默认 X 轴编辑器中的文本输入框显示 "20, 25, 30"
    const initialAxisInput = await screen.findByDisplayValue(/20, 25, 30/)
    expect(initialAxisInput).toBeInTheDocument()

    // 等历史栏的 thumbnail 出现（HistoryItem div 的 title 含 folder 名）
    const thumb = await screen.findByTitle(/xy plot 1 ·/)
    await user.click(thumb)

    // 回填后：X 轴编辑器中的 dropdown 切到 LoRA，raw 写入新值。
    const axisEditor = screen.getByTestId('xy-axis-editor-drawer')
    await waitFor(() => {
      const axisSelect = axisEditor.querySelector('select') as HTMLSelectElement
      expect(axisSelect.value).toBe('lora_ckpt')
    })
    // 原 "20, 25, 30" 文本框该消失（切到 lora_ckpt 后显示 checkpoint 列表）
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
    await user.click(await screen.findByRole('button', { name: '编辑 X 轴' }))

    // 生成 3 张 → dispatch 定格本次运行态 run
    await user.click(await screen.findByRole('button', { name: '生成 3 张' }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())

    // 网格渲染出 steps 三档表头（20/25/30）
    await waitFor(() => expect(screen.getAllByText('30').length).toBeGreaterThan(0))

    // Drawer 中编辑当前 X 轴的值；左侧只显示摘要卡。
    const axisInput = screen.getByDisplayValue('20, 25, 30')
    await user.clear(axisInput)
    await user.type(axisInput, '20, 25')

    // live sidebar 确实改成了 "20, 25"（下次生成会用它）……
    await waitFor(() => expect(axisInput).toHaveValue('20, 25'))
    // ……但右侧已出结果网格冻结：仍含 "30" 表头，未被 live 编辑串改
    expect(screen.getAllByText('30').length).toBeGreaterThan(0)
  })

  // ---- 回看历史 XY 时点「开始生成」→ 回到实时视图（P-I 回归修复）----
  it('回看 XY 历史时点开始生成：清掉历史 override，结果区回到实时新任务', async () => {
    // 修前（P-I 删了「currentTask.id 变自动清 override」的 effect）：点开始生成不清
    // override，结果区停留在正回看的老 XY 图，看不到新入队/正在跑的这次。
    // 修后：提交即清 override，回到实时视图。
    vi.mocked(useMonitorProgress).mockReturnValue({
      state: {
        samples: [{ path: 'cell x0 y0.png', xy: { xi: 0, yi: 0, xv: 20, yv: null } }],
      },
    } as never)
    seedPrefs({ mode: 'xy' })  // 默认 X=steps raw "20, 25, 30"
    const xySnapshotParams = {
      schema_version: 1, mode: 'xy',
      prompts: ['recall'], negative_prompt: '',
      width: 1024, height: 1024, steps: 25, cfg_scale: 4, count: 1, seed: 0,
      loras: [],
      // 用 steps 轴（非 lora_ckpt）→ 回填后按钮保持简洁，不引入 picker 异步
      xy_draft: { x: { axis: 'steps', raw: '20, 25, 30', loraIndex: null }, y: null },
      dataset_pick: null,
    }
    const timelineEntry = {
      task_id: 92,
      status: 'done',
      created_at: 1717900000,
      mode: 'xy',
      storage: 'disk',
      params: xySnapshotParams,
      images: [
        {
          url: '/api/generate/disk/image/2026-06-09/xy/xy%20plot%201/cell%20x0%20y0.png',
          thumb_url: '/api/generate/disk/thumb/2026-06-09/xy/xy%20plot%201/cell%20x0%20y0.png?w=128',
          xi: 0, yi: 0,
        },
      ],
      available: true,
      xy_folder: 'xy plot 1',
    }
    const previousImpl = fetchMock.getMockImplementation()
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes('/api/generate/timeline') && (init?.method ?? 'GET') === 'GET') {
        const body = { entries: [timelineEntry], total: 1, offset: 0 }
        return Promise.resolve({
          ok: true, status: 200,
          json: async () => body,
          text: async () => JSON.stringify(body),
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

    // 点历史缩略图进入回看：结果区底部显示文件夹名 "xy plot 1"（override 生效标志）
    const thumb = await screen.findByTitle(/xy plot 1 ·/)
    await user.click(thumb)
    await waitFor(() => expect(screen.getByText('xy plot 1')).toBeInTheDocument())

    // 点生成按钮 → 清掉 override，结果区回到实时任务：底部 "xy plot 1" 文本消失
    await user.click(screen.getByRole('button', { name: '生成 3 张' }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    await waitFor(() => expect(screen.queryByText('xy plot 1')).not.toBeInTheDocument())
  })
})
