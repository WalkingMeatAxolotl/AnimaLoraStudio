import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ConfigData, SchemaResponse } from '../api/client'
import ConfigYamlPanel from './ConfigYamlPanel'
import { ToastProvider } from './Toast'

const schema = {
  schema: {
    properties: {
      optimizer_type: {},
      lora_rank: {},
      came_beta1: { show_when: 'optimizer_type==came' },
    },
  },
  groups: [],
} as unknown as SchemaResponse

function renderPanel(config: ConfigData) {
  return render(
    <ToastProvider>
      <ConfigYamlPanel config={config} schema={schema} fileLabel="config.yaml" />
    </ToastProvider>,
  )
}

describe('ConfigYamlPanel', () => {
  it('renders yaml with inactive fields pruned and counts active fields', () => {
    renderPanel({ optimizer_type: 'adamw', lora_rank: 64, came_beta1: 0.5 })
    const pre = document.querySelector('pre')
    expect(pre?.textContent).toBe('optimizer_type: adamw\nlora_rank: 64')
    expect(screen.getByText('config.yaml')).toBeInTheDocument()
    // schema.fieldCount → 「2 项」（came_beta1 被裁掉不计入）
    expect(screen.getByText('2 项')).toBeInTheDocument()
  })

  it('keeps active conditional fields', () => {
    renderPanel({ optimizer_type: 'came', lora_rank: 64, came_beta1: 0.5 })
    const pre = document.querySelector('pre')
    expect(pre?.textContent).toContain('came_beta1: 0.5')
  })

  it('shows the hint when provided', () => {
    render(
      <ToastProvider>
        <ConfigYamlPanel
          config={{ lora_rank: 64 }}
          schema={schema}
          fileLabel="p.yaml"
          hint="包含未保存的修改"
        />
      </ToastProvider>,
    )
    expect(screen.getByText('包含未保存的修改')).toBeInTheDocument()
  })
})
