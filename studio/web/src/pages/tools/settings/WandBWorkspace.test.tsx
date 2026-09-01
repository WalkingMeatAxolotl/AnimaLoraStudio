import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { WandBConfig, WandBPreset } from '../../../api/client'
import WandBWorkspace from './WandBWorkspace'

const preset: WandBPreset = {
  id: 'default',
  label: 'Default',
  api_key: '',
  project: 'AnimaLoraStudio',
  entity: '',
  base_url: 'https://api.wandb.ai',
  mode: 'online',
  log_samples: false,
  sample_max_side: 1216,
  sample_every_n_steps: 0,
  upload_model: false,
  upload_model_policy: 'last',
  upload_state_manual: false,
  upload_state_manual_policy: 'last',
  upload_state_auto: false,
  upload_state_auto_policy: 'last',
}

const config: WandBConfig = {
  enabled: false,
  current_preset: preset.id,
  presets: [preset],
}

describe('WandBWorkspace form layout', () => {
  it('spaces field rows and gives every body control one shared width rail', () => {
    render(
      <WandBWorkspace
        title="Weights & Biases"
        config={config}
        serverPresets={[preset]}
        currentPreset={preset}
        onToggleEnabled={vi.fn()}
        onSelectPreset={vi.fn()}
        onUpdatePreset={vi.fn()}
        onAddPreset={vi.fn()}
        onSaveAs={vi.fn()}
        onDeletePreset={vi.fn()}
        onExport={vi.fn()}
        onImportFile={vi.fn()}
      />,
    )

    const fields = screen.getByTestId('wandb-fields')
    expect(fields).toHaveClass('flex', 'flex-col', 'gap-field')

    const controls = fields.querySelectorAll('.form-control')
    expect(controls).toHaveLength(9)
    controls.forEach((control) => expect(control).toHaveClass('max-w-md'))
  })
})
