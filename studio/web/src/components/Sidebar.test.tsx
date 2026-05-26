import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { SettingsDrawerProvider } from '../lib/SettingsDrawer'
import { DialogProvider } from './Dialog'
import Sidebar from './Sidebar'

function renderAt(path: string) {
  return render(
    <MemoryRouter
      initialEntries={[path]}
      future={{ v7_relativeSplatPath: true, v7_startTransition: true }}
    >
      <DialogProvider>
        <SettingsDrawerProvider>
          <Sidebar />
        </SettingsDrawerProvider>
      </DialogProvider>
    </MemoryRouter>
  )
}

describe('Sidebar (PP0)', () => {
  it('shows main items + tools with all 5 destinations', () => {
    renderAt('/')
    // 主导航
    expect(screen.getByRole('link', { name: /项目/ })).toHaveAttribute(
      'href',
      '/'
    )
    expect(screen.getByRole('link', { name: /队列/ })).toHaveAttribute(
      'href',
      '/queue'
    )
    // 工具区（重设计后没有 "工具" 分组 label，只是用 border-top 分隔）
    expect(screen.getByRole('link', { name: /预设/ })).toHaveAttribute(
      'href',
      '/tools/presets'
    )
    expect(screen.getByRole('link', { name: /监控/ })).toHaveAttribute(
      'href',
      '/tools/monitor'
    )
    // 设置不再是路由 link，而是打开右侧抽屉的 button；没有 href
    expect(screen.getByRole('button', { name: /设置/ })).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: /设置/ })).toBeNull()
  })

  it('marks the active route', () => {
    renderAt('/tools/presets')
    const link = screen.getByRole('link', { name: /预设/ })
    // 活跃 link：bg-surface + font-semibold（重设计 token 化后的活跃态）
    expect(link.className).toMatch(/bg-surface/)
    expect(link.className).toMatch(/font-semibold/)
    // 非活跃 link 没有这俩
    const queue = screen.getByRole('link', { name: /队列/ })
    expect(queue.className).not.toMatch(/bg-surface/)
    expect(queue.className).not.toMatch(/font-semibold/)
  })

  it('does not include the removed Datasets link', () => {
    renderAt('/')
    expect(screen.queryByRole('link', { name: /数据集/ })).toBeNull()
    expect(screen.queryByRole('link', { name: /配置/ })).toBeNull()
  })
})
