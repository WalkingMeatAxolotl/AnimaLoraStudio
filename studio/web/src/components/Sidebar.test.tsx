import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { SettingsDrawerProvider } from '../lib/SettingsDrawer'
import {
  SelectedProjectContext,
  type SelectedProjectValue,
} from '../context/ProjectContext'
import type { ProjectDetail, Version } from '../api/client'
import { DialogProvider } from './Dialog'
import { ToastProvider } from './Toast'
import Sidebar from './Sidebar'

function renderAt(path: string, sticky: SelectedProjectValue | null = null) {
  return render(
    <MemoryRouter
      initialEntries={[path]}
      future={{ v7_relativeSplatPath: true, v7_startTransition: true }}
    >
      <ToastProvider>
        <DialogProvider>
          <SettingsDrawerProvider>
            <SelectedProjectContext.Provider value={sticky}>
              <Sidebar />
            </SelectedProjectContext.Provider>
          </SettingsDrawerProvider>
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>
  )
}

const MOCK_VERSION: Version = {
  id: 7, project_id: 3, label: 'v1', config_name: null, status: 'preparing',
  phase: 'curating', last_failure_reason: null, created_at: 0,
  output_lora_path: null, note: null, trigger_word: '',
}
const MOCK_PROJECT: ProjectDetail = {
  id: 3, slug: 'ganyu', title: '甘雨', active_version_id: 7,
  active_version_label: 'v1', active_version_status: 'preparing',
  active_version_phase: 'curating', created_at: 0, updated_at: 0,
  archived_at: null, note: null, versions: [MOCK_VERSION],
  download_image_count: 0, preprocess_image_count: 0,
}
const STICKY: SelectedProjectValue = { project: MOCK_PROJECT, activeVersion: MOCK_VERSION }

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

  // 粘性"已选中项目"：离开项目页（如在队列页）后项目区仍保留，用于跨页导航
  it('keeps the selected project section on a global page (queue)', () => {
    renderAt('/queue', STICKY)
    // 项目名仍显示
    expect(screen.getByText('甘雨')).toBeInTheDocument()
    // 概览链接指向该项目，可点回去
    const overview = screen.getByRole('link', { name: /概览/ })
    expect(overview).toHaveAttribute('href', '/projects/3')
  })

  it('does not highlight overview when off the project route', () => {
    renderAt('/queue', STICKY)
    // 在队列页：队列高亮，概览不高亮（inRoute 门控，避免 currentStep===null 误判）
    const queue = screen.getByRole('link', { name: /队列/ })
    expect(queue.className).toMatch(/bg-surface/)
    const overview = screen.getByRole('link', { name: /概览/ })
    expect(overview.className).not.toMatch(/bg-surface/)
  })

  it('shows no project section without a sticky selection', () => {
    renderAt('/queue')
    expect(screen.queryByText('甘雨')).toBeNull()
    expect(screen.queryByRole('link', { name: /概览/ })).toBeNull()
  })
})
