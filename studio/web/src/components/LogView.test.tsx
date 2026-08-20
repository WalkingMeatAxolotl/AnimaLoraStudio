import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { api } from '../api/client'
import { _resetLogDebugPrefForTests } from '../lib/logDebugPref'
import LogView from './LogView'

const H = (lvl: string, logger: string, msg: string) => `2026-08-19 14:03:22.417 ${lvl.padEnd(5)} ${logger}: ${msg}`

function mockGlobalDefault(v: boolean) {
  vi.spyOn(api, 'getSecrets').mockResolvedValue({ system: { log_debug_default: v } } as never)
}

beforeEach(() => {
  _resetLogDebugPrefForTests()
  vi.restoreAllMocks()
})

describe('LogView', () => {
  it('按级别着色，默认（全局关）隐藏 DEBUG 行；视图开关临时打开', async () => {
    mockGlobalDefault(false)
    render(<LogView lines={[H('DEBUG', 'training.loop', 'dbg-line'), H('INFO', 'training.loop', 'info-line'), H('WARNING', 'utils.x', 'warn-line'), H('ERROR', 'a.b', 'err-line'), 'Traceback (most recent call last):']} />)
    await waitFor(() => expect(screen.queryByText('dbg-line')).not.toBeInTheDocument())
    expect(screen.getByText('err-line')).toHaveClass('text-err')
    expect(screen.getByText('warn-line')).toHaveClass('text-warn')
    // 续行继承 ERROR 且缩进
    const cont = screen.getByText('Traceback (most recent call last):')
    expect(cont).toHaveClass('text-err')
    expect(cont.getAttribute('style')).toMatch(/padding-left/)

    await userEvent.click(screen.getByLabelText(/调试/))
    expect(screen.getByText('dbg-line')).toBeInTheDocument()
    expect(screen.getByText('dbg-line')).toHaveClass('text-fg-tertiary')
  })

  it('全局默认开 → 视图初值开；用户改过后全局再变不回推', async () => {
    mockGlobalDefault(true)
    const { rerender } = render(<LogView lines={[H('DEBUG', 'x', 'dbg-line')]} />)
    await waitFor(() => expect(screen.getByText('dbg-line')).toBeInTheDocument())
    await userEvent.click(screen.getByLabelText(/调试/))  // 用户关掉
    expect(screen.queryByText('dbg-line')).not.toBeInTheDocument()
    rerender(<LogView lines={[H('DEBUG', 'x', 'dbg-line'), H('INFO', 'x', 'i')]} />)
    expect(screen.queryByText('dbg-line')).not.toBeInTheDocument()
  })

  it('行头拆成 时间 / 级别 / 来源 / 消息；老格式行原样', async () => {
    mockGlobalDefault(false)
    render(<LogView lines={[H('INFO', 'training.progress', 'epoch=0 step=10'), '2026-08-10 03:43:23,610 - INFO - 训练完成!']} />)
    expect(screen.getByText('14:03:22.417')).toBeInTheDocument()
    expect(screen.getByText('INFO')).toBeInTheDocument()
    expect(screen.getByText('training.progress')).toBeInTheDocument()
    expect(screen.getByText('epoch=0 step=10')).toBeInTheDocument()
    expect(screen.getByText('2026-08-10 03:43:23,610 - INFO - 训练完成!')).toBeInTheDocument()
  })

  it('空态按 status 选文案；error 态显示错误与重试；加载更早 / 下载按钮按 props 出现', async () => {
    mockGlobalDefault(false)
    const onLoad = vi.fn()
    const onRefresh = vi.fn()
    const { rerender } = render(<LogView lines={[]} status="waiting" />)
    expect(screen.getByText('（等待日志…）')).toBeInTheDocument()
    rerender(<LogView lines={[]} status="finished" />)
    expect(screen.getByText('（没有日志）')).toBeInTheDocument()
    rerender(<LogView lines={['x']} status="error" error="boom" onRefresh={onRefresh} hasMoreBefore onLoadEarlier={onLoad} downloadUrl="/api/logs/7/raw" />)
    expect(screen.getByText('boom')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: '重试' }))
    expect(onRefresh).toHaveBeenCalled()
    await userEvent.click(screen.getByRole('button', { name: '加载更早' }))
    expect(onLoad).toHaveBeenCalled()
    expect(screen.getByRole('link', { name: '下载' })).toHaveAttribute('href', '/api/logs/7/raw')
  })

  it('toolbar=false 不渲染工具栏', () => {
    mockGlobalDefault(false)
    render(<LogView lines={['a']} toolbar={false} />)
    expect(screen.queryByLabelText(/调试/)).not.toBeInTheDocument()
    expect(screen.getByText('a')).toBeInTheDocument()
  })
})
