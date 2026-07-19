import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ConfigData } from '../api/client'
import { api } from '../api/client'
import ConfigYamlPanel from './ConfigYamlPanel'
import { ToastProvider } from './Toast'

// R4(D3):组件不再本地裁剪渲染(pruneInactiveConfig/configToYaml 已删),
// 而是 300ms debounce 后调 POST /api/schema/preview-yaml —— 与落盘同一条
// 序列化路径。测试 mock 该端点,锁「请求带当前 config + 展示返回文本」。

function renderPanel(config: ConfigData, hint?: string) {
  return render(
    <ToastProvider>
      <ConfigYamlPanel config={config} fileLabel="config.yaml" hint={hint} />
    </ToastProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('ConfigYamlPanel', () => {
  it('debounces then renders backend preview text and counts top-level keys', async () => {
    const spy = vi
      .spyOn(api, 'previewConfigYaml')
      .mockResolvedValue({ yaml: 'optimizer_type: adamw\nlora_rank: 64\n' })
    renderPanel({ optimizer_type: 'adamw', lora_rank: 64 })
    await waitFor(
      () => {
        const pre = document.querySelector('pre')
        expect(pre?.textContent).toContain('optimizer_type: adamw')
      },
      { timeout: 2000 },
    )
    expect(spy).toHaveBeenCalledWith({ optimizer_type: 'adamw', lora_rank: 64 })
    expect(screen.getByText('config.yaml')).toBeInTheDocument()
    // schema.fieldCount → 顶级键 2 项(顶格行计数)
    expect(screen.getByText('2 项')).toBeInTheDocument()
  })

  it('shows the hint when provided', async () => {
    vi.spyOn(api, 'previewConfigYaml').mockResolvedValue({ yaml: 'lora_rank: 64\n' })
    renderPanel({ lora_rank: 64 }, '包含未保存的修改')
    expect(screen.getByText('包含未保存的修改')).toBeInTheDocument()
    await waitFor(
      () => expect(document.querySelector('pre')?.textContent).toContain('lora_rank'),
      { timeout: 2000 },
    )
  })
})
