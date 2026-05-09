/** GeneratePage 端到端 smoke：mock fetch，验证 single / xy / 多 prompt+xy
 *  三个关键路径的 enqueue payload 行为。 */
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ToastProvider } from '../../components/Toast'
import GeneratePage from './Generate'

const fetchMock = vi.fn()
let lastEnqueueBody: Record<string, unknown> | null = null

beforeEach(() => {
  lastEnqueueBody = null
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset()
  fetchMock.mockImplementation((url: string, init?: RequestInit) => {
    // useProjectLoras 启动时 listProjects → 返回空（no LoRAs in picker）
    if (url.endsWith('/api/projects') && (init?.method ?? 'GET') === 'GET') {
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
    expect(body.attention_backend).toBe('flash_attn')
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

  it('mode=xy + 多 prompt → toast 报错，不调 /api/generate', async () => {
    const user = userEvent.setup()
    setup()

    // 先加一个 prompt 并填内容（空 prompt 会被 filter 掉，不触发互斥校验）
    await user.click(screen.getByRole('button', { name: /添加 prompt/ }))
    const promptInputs = screen.getAllByPlaceholderText('输入正向提示词…')
    expect(promptInputs.length).toBe(2)
    await user.type(promptInputs[1], 'second prompt')

    await user.click(screen.getByRole('button', { name: 'XY 矩阵' }))
    await user.click(screen.getByRole('button', { name: /开始生成 · 3 张/ }))

    await waitFor(() =>
      expect(screen.getByText(/XY 模式只支持单条 prompt/)).toBeInTheDocument()
    )
    expect(lastEnqueueBody).toBeNull()
  })

  it('切到 xy 再切回 single：sidebar 已填的 prompts/seed 等保留', async () => {
    const user = userEvent.setup()
    setup()

    const promptArea = screen.getAllByPlaceholderText('输入正向提示词…')[0]
    await user.clear(promptArea)
    await user.type(promptArea, 'my custom prompt')

    await user.click(screen.getByRole('button', { name: 'XY 矩阵' }))
    await user.click(screen.getByRole('button', { name: '单图' }))

    expect(promptArea).toHaveValue('my custom prompt')
  })
})
