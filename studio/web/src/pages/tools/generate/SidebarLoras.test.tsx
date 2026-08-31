import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import type { LoraEntry } from '../../../api/client'
import SidebarLoras, { reorderLoraSelection } from './SidebarLoras'
import type { LoraUiState } from './loraSelection'

function Harness() {
  const [loras, setLoras] = useState<LoraEntry[]>([
    { path: 'D:/ComfyUI/models/loras/styles/ink.safetensors', scale: 1 },
  ])
  const [ui, setUi] = useState<LoraUiState[]>([{ id: 'ink', enabled: true }])
  return (
    <SidebarLoras
      loras={loras}
      ui={ui}
      onChange={(nextLoras, nextUi) => { setLoras(nextLoras); setUi(nextUi) }}
    />
  )
}

describe('SidebarLoras', () => {
  it('uses a directly editable textarea and applies on blur while preserving invalid input', async () => {
    render(<Harness />)
    const textarea = screen.getByRole('textbox', { name: 'LoRA 文本' })
    expect(textarea).toHaveValue('<lora:ink:1>')
    expect(screen.queryByRole('button', { name: '复制' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '编辑' })).not.toBeInTheDocument()

    fireEvent.focus(textarea)
    fireEvent.change(textarea, { target: { value: '<lora:ink:0.75>' } })
    fireEvent.blur(textarea)
    await waitFor(() => expect(screen.getByRole('spinbutton', { name: /ink/ })).toHaveValue(0.75))

    fireEvent.focus(textarea)
    fireEvent.change(textarea, { target: { value: 'invalid' } })
    fireEvent.blur(textarea)
    expect(await screen.findByRole('alert')).toBeInTheDocument()
    expect(textarea).toHaveValue('invalid')
    expect(textarea).toHaveAttribute('aria-invalid', 'true')
  })

  it('applies with Ctrl+Enter and renders a compact native number weight control', async () => {
    const user = userEvent.setup()
    const { container } = render(<Harness />)
    const textarea = screen.getByRole('textbox', { name: 'LoRA 文本' })

    fireEvent.focus(textarea)
    fireEvent.change(textarea, { target: { value: '<lora:ink:0.5>' } })
    fireEvent.keyDown(textarea, { key: 'Enter', ctrlKey: true })
    await waitFor(() => expect(screen.getByRole('spinbutton', { name: /ink/ })).toHaveValue(0.5))

    expect(screen.getByText('ink')).toBeInTheDocument()
    expect(screen.queryByText('ink.safetensors')).not.toBeInTheDocument()
    expect(screen.queryByText('D:/ComfyUI/models/loras/styles/ink.safetensors')).not.toBeInTheDocument()
    expect(container.querySelector('input[type="range"]')).toBeNull()
    const weight = screen.getByRole('spinbutton', { name: /ink/ })
    expect(weight).toHaveAttribute('min', '0')
    expect(weight).toHaveAttribute('max', '1.5')
    expect(weight).toHaveAttribute('step', '0.05')
    const remove = screen.getByRole('button', { name: /移除 LoRA ink/ })
    expect(remove).toHaveClass('opacity-0', 'pointer-events-none', 'group-hover:opacity-100', 'group-hover:pointer-events-auto', 'group-focus-within:opacity-100')
    expect(remove.nextElementSibling).toBe(weight)

    expect(screen.getByRole('button', { name: '拖动调整顺序 ink' })).toHaveClass('cursor-grab')

    fireEvent.blur(textarea)
    await user.click(screen.getByRole('checkbox', { name: /启用 LoRA ink/ }))
    await waitFor(() => expect(textarea).toHaveValue(''))
  })

  it('reorders the LoRA entries and their UI sidecars together', () => {
    const loras: LoraEntry[] = [
      { path: 'D:/ComfyUI/models/loras/styles/ink.safetensors', scale: 1 },
      { path: 'D:/ComfyUI/models/loras/styles/watercolor.safetensors', scale: 0.8 },
    ]
    const ui: LoraUiState[] = [
      { id: 'ink', enabled: true },
      { id: 'watercolor', enabled: false },
    ]

    const result = reorderLoraSelection(loras, ui, 'ink', 'watercolor')

    expect(result?.loras.map((entry) => entry.path)).toEqual([
      'D:/ComfyUI/models/loras/styles/watercolor.safetensors',
      'D:/ComfyUI/models/loras/styles/ink.safetensors',
    ])
    expect(result?.ui).toEqual([
      { id: 'watercolor', enabled: false },
      { id: 'ink', enabled: true },
    ])
  })
})
