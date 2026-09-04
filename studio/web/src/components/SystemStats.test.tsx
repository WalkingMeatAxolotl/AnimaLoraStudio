import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import SystemStats from './SystemStats'
import { api, type SystemStats as Stats } from '../api/client'

// useEventStream 在 jsdom 下不会真起 EventSource (源码有 typeof 守卫)，所以
// 这里测的主要是：mount 时 GET 一次冷启动 + 各种 stats 形态下的渲染。SSE
// delta 的合并行为另测（手动验证或 e2e）。

function makeStats(overrides: Partial<Stats> = {}): Stats {
  return {
    cpu_pct: 12.5,
    ram_used_gb: 8.0,
    ram_total_gb: 32.0,
    gpu: [
      {
        index: 0,
        name: 'Test GPU',
        util_pct: 50,
        vram_used_gb: 4.0,
        vram_total_gb: 24.0,
        temp_c: 55,
      },
    ],
    ...overrides,
  }
}

describe('SystemStats', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders nothing before first fetch resolves', () => {
    vi.spyOn(api, 'systemStats').mockReturnValue(new Promise(() => {}))
    const { container } = render(<SystemStats />)
    expect(container.firstChild).toBeNull()
  })

  it('shows CPU / MEM / GPU / VRAM pills with values after mount fetch', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats())
    render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('CPU')).toBeInTheDocument())
    expect(screen.getByText('CPU').closest('.ui-app-shell-topbar-stats')).toBeInTheDocument()
    expect(screen.getByText('13%')).toBeInTheDocument()
    expect(screen.getByText('MEM')).toBeInTheDocument()
    expect(screen.getByText('8.0/32G')).toBeInTheDocument()
    expect(screen.getByText('GPU')).toBeInTheDocument()
    expect(screen.getByText('50%')).toBeInTheDocument()
    expect(screen.getByText('VRAM')).toBeInTheDocument()
    expect(screen.getByText('4.0/24G')).toBeInTheDocument()

    const cpu = screen.getByRole('meter', { name: 'CPU' })
    expect(cpu).toHaveAttribute('aria-valuemin', '0')
    expect(cpu).toHaveAttribute('aria-valuemax', '100')
    expect(cpu).toHaveAttribute('aria-valuenow', '12.5')
    expect(cpu).toHaveAttribute('aria-valuetext', 'CPU 占用 12.5%')
    expect(screen.getByRole('meter', { name: '内存' })).toHaveAttribute(
      'aria-valuetext',
      '内存 8.0 / 32.0 GB（25%）',
    )
    expect(screen.getByRole('meter', { name: 'GPU' })).toHaveAttribute(
      'aria-valuetext',
      expect.stringMatching(/GPU 利用率 50% · Test GPU/),
    )
    expect(screen.getByRole('meter', { name: '显存' })).toHaveAttribute(
      'aria-valuetext',
      expect.stringMatching(/显存 4.0 \/ 24.0 GB（17%）· Test GPU/),
    )
  })

  it('hides GPU / VRAM when stats.gpu is null', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({ gpu: null }))
    render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('CPU')).toBeInTheDocument())
    expect(screen.queryByText('GPU')).toBeNull()
    expect(screen.queryByText('VRAM')).toBeNull()
  })

  it('hides GPU / VRAM when stats.gpu is empty array', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({ gpu: [] }))
    render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('CPU')).toBeInTheDocument())
    expect(screen.queryByText('GPU')).toBeNull()
    expect(screen.queryByText('VRAM')).toBeNull()
  })

  it('shows high-tone class when util exceeds 90%', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({ cpu_pct: 95 }))
    render(<SystemStats />)
    const el = await screen.findByText('95%')
    expect(el.className).toContain('text-err')
  })

  it('multi-GPU: shows the active card, not blindly gpu[0]', async () => {
    // #491：NVML 序 0=2080(8G)、1=3070(16G)，torch 实际在 3070 上
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({
      gpu: [
        { index: 0, name: 'RTX 2080', util_pct: 1, vram_used_gb: 1.1, vram_total_gb: 8.0, temp_c: 57, active: false },
        { index: 1, name: 'RTX 3070', util_pct: 80, vram_used_gb: 12.0, vram_total_gb: 16.0, temp_c: 43, active: true },
      ],
    }))
    render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('GPU')).toBeInTheDocument())
    expect(screen.getByText('80%')).toBeInTheDocument()
    expect(screen.getByText('12.0/16G')).toBeInTheDocument()
    expect(screen.queryByText('1.1/8G')).toBeNull()
    expect(screen.getByRole('meter', { name: 'GPU' })).toHaveAttribute(
      'aria-valuetext',
      expect.stringMatching(/GPU 利用率 80% · RTX 3070 · 43°C（另有 1 张显卡）/),
    )
  })

  it('multi-GPU: falls back to gpu[0] when no card is marked active', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({
      gpu: [
        { index: 0, name: 'RTX 2080', util_pct: 1, vram_used_gb: 1.1, vram_total_gb: 8.0, temp_c: 57 },
        { index: 1, name: 'RTX 3070', util_pct: 80, vram_used_gb: 12.0, vram_total_gb: 16.0, temp_c: 43 },
      ],
    }))
    render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('GPU')).toBeInTheDocument())
    expect(screen.getByText('1.1/8G')).toBeInTheDocument()
  })

  it('only fetches once on mount (SSE 化后无轮询)', async () => {
    const spy = vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats())
    render(<SystemStats />)
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1))
    // 等一段实际时间让任何潜在的轮询有机会触发
    await new Promise((r) => setTimeout(r, 200))
    expect(spy).toHaveBeenCalledTimes(1)
  })
})
