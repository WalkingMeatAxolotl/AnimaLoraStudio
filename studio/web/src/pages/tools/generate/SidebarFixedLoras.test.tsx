import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import type { LoraEntry } from '../../../api/client'
import SidebarFixedLoras from './SidebarFixedLoras'
import type { LoraUiState } from './loraSelection'

function Harness() {
  const [loras, setLoras] = useState<LoraEntry[]>([
    { path: 'D:/ComfyUI/models/loras/styles/ink.safetensors', scale: 1 },
  ])
  const [ui, setUi] = useState<LoraUiState[]>([{ id: 'ink', enabled: true }])
  return (
    <SidebarFixedLoras
      loras={loras}
      ui={ui}
      onChange={(nextLoras, nextUi) => { setLoras(nextLoras); setUi(nextUi) }}
    />
  )
}

describe('SidebarFixedLoras', () => {
  it('keeps raw text secondary and exposes slider plus exact weight', async () => {
    const user = userEvent.setup()
    const { container } = render(<Harness />)

    const details = screen.getByText('编辑').closest('details')
    expect(details).not.toHaveAttribute('open')
    await user.click(screen.getByText('编辑'))

    const textarea = screen.getByRole('textbox', { name: 'LoRA 文本' })
    fireEvent.focus(textarea)
    fireEvent.change(textarea, { target: { value: '<lora:ink:0.5>' } })
    fireEvent.keyDown(textarea, { key: 'Enter', ctrlKey: true })

    await waitFor(() => expect(screen.getByRole('spinbutton', { name: /ink/ })).toHaveValue(0.5))
    expect(container.querySelector('input[type="range"]')).not.toBeNull()
    expect(screen.getByRole('slider', { name: /ink/ })).toHaveValue('0.5')
    expect(screen.queryByText('ink.safetensors')).not.toBeInTheDocument()
  })
})
