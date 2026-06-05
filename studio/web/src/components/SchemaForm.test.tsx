import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { ConfigData, SchemaResponse } from '../api/client'
import SchemaForm from './SchemaForm'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, opts?: { defaultValue?: string; n?: number }) => {
      const dict: Record<string, string> = {
        'schema.groups.training': 'Training',
        'schema.groups.timestepSampling': 'Timestep Sampling',
        'schema.disableHints.learning_rate': 'Prodigy controls the learning rate',
        'schema.disableHints.lr_scheduler': 'Schedule-Free has its own scheduler',
        'schema.disableHints.timestep_sampling': 'InfoNoise controls timestep sampling',
        'schema.descriptions.timestep_sampling': 'Normal timestep description',
        'schema.altDescriptions.timestep_sampling': 'InfoNoise takes over timestep sampling',
        'schema.enums.optimizer_type.adamw': 'AdamW',
        'schema.enums.optimizer_type.prodigy': 'Prodigy',
        'schema.enums.optimizer_type.prodigy_plus_schedulefree': 'Prodigy+ Schedule-Free',
        'schema.enums.lr_scheduler.none': 'None',
        'schema.enums.lr_scheduler.cosine': 'Cosine',
        'schema.enums.timestep_sampling.logit_normal': 'Logit Normal',
        'schema.enums.timestep_sampling.uniform': 'Uniform',
        'field.useGlobal': 'Use global',
        'field.yes': 'Yes',
        'field.no': 'No',
      }
      if (key === 'schema.fieldCount') return `${opts?.n ?? 0} fields`
      return dict[key] ?? opts?.defaultValue ?? key
    },
  }),
}))

const schema: SchemaResponse = {
  groups: [
    { key: 'training', label: '训练' },
    { key: 'timestep_sampling', label: '时间步采样' },
  ],
  schema: {
    properties: {
      optimizer_type: {
        type: 'string',
        enum: ['adamw', 'prodigy', 'prodigy_plus_schedulefree'],
        default: 'adamw',
        group: 'training',
        description: 'Optimizer',
      },
      learning_rate: {
        type: 'number',
        default: 0.0001,
        group: 'training',
        description: 'Learning rate',
        disable_when: 'optimizer_type==prodigy||optimizer_type==prodigy_plus_schedulefree',
        disable_value: 1,
      },
      lr_scheduler: {
        type: 'string',
        enum: ['none', 'cosine'],
        default: 'none',
        group: 'training',
        description: 'Scheduler',
        disable_when: 'optimizer_type==prodigy_plus_schedulefree',
      },
      timestep_sampling: {
        type: 'string',
        enum: ['logit_normal', 'uniform'],
        default: 'logit_normal',
        group: 'timestep_sampling',
        description: '后端普通说明',
        alt_description_when: 'infonoise_enabled==true',
        disable_when: 'infonoise_enabled==true',
        advanced: true,
      },
      infonoise_enabled: {
        type: 'boolean',
        default: false,
        group: 'timestep_sampling',
        description: 'InfoNoise',
        advanced: true,
      },
      wandb_enabled: {
        anyOf: [{ type: 'boolean' }, { type: 'null' }],
        default: null,
        group: 'training',
        description: 'WandB',
      },
    },
  },
}

describe('SchemaForm takeover behavior', () => {
  it('resets and disables fields taken over by disable_when', async () => {
    const onChange = vi.fn()
    const values: ConfigData = {
      optimizer_type: 'prodigy_plus_schedulefree',
      learning_rate: 0.0001,
      lr_scheduler: 'cosine',
      infonoise_enabled: false,
      timestep_sampling: 'logit_normal',
    }

    render(
      <SchemaForm
        schema={schema}
        values={values}
        onChange={onChange}
        advancedMode
      />,
    )

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith({
        ...values,
        learning_rate: 1,
        lr_scheduler: 'none',
      })
    })
    const learningRateInput = screen.getByRole('textbox') as HTMLInputElement
    expect(learningRateInput).toBeDisabled()
    expect(learningRateInput.value).toBe('0.0001')
    expect(screen.getByText('Prodigy controls the learning rate')).toBeInTheDocument()
    const schedulerSelect = screen.getAllByRole('combobox')[1] as HTMLSelectElement
    expect(schedulerSelect).toBeDisabled()
    expect(schedulerSelect.value).toBe('cosine')
    expect(screen.getByText('Schedule-Free has its own scheduler')).toBeInTheDocument()
  })

  it('locks learning rate to 1 for regular Prodigy', async () => {
    const onChange = vi.fn()
    const values: ConfigData = {
      optimizer_type: 'prodigy',
      learning_rate: 0.0001,
      lr_scheduler: 'none',
      infonoise_enabled: false,
      timestep_sampling: 'logit_normal',
    }

    render(
      <SchemaForm
        schema={schema}
        values={values}
        onChange={onChange}
        advancedMode
      />,
    )

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith({
        ...values,
        learning_rate: 1,
      })
    })
    expect(screen.getByRole('textbox')).toBeDisabled()
    expect(screen.getByText('Prodigy controls the learning rate')).toBeInTheDocument()
  })

  it('renders nullable booleans as tri-state selects without coercing null to false', () => {
    const onChange = vi.fn()

    render(
      <SchemaForm
        schema={schema}
        values={{
          optimizer_type: 'adamw',
          learning_rate: 0.0001,
          lr_scheduler: 'none',
          infonoise_enabled: false,
          timestep_sampling: 'logit_normal',
          wandb_enabled: null,
        }}
        onChange={onChange}
        advancedMode
      />,
    )

    const wandbSelect = screen.getByDisplayValue('Use global') as HTMLSelectElement
    expect(wandbSelect.value).toBe('')
    expect(screen.queryByRole('checkbox', { name: /wandb/i })).not.toBeInTheDocument()
    expect(onChange).not.toHaveBeenCalled()
  })

  it('disables InfoNoise-controlled timestep fields and uses frontend alt descriptions', () => {
    render(
      <SchemaForm
        schema={schema}
        values={{
          optimizer_type: 'adamw',
          learning_rate: 0.0001,
          lr_scheduler: 'none',
          infonoise_enabled: true,
          timestep_sampling: 'uniform',
        }}
        onChange={() => {}}
        advancedMode
      />,
    )

    const timestepSelect = screen.getAllByRole('combobox').find((el) => (el as HTMLSelectElement).value === 'uniform') as HTMLSelectElement
    expect(timestepSelect).toBeDisabled()
    expect(screen.getByText('InfoNoise controls timestep sampling')).toBeInTheDocument()
    expect(screen.getByText('InfoNoise takes over timestep sampling')).toBeInTheDocument()
    expect(screen.queryByText('Normal timestep description')).not.toBeInTheDocument()
  })

  it('disable_when without disable_value preserves the existing value (no silent reset)', async () => {
    const onChange = vi.fn()
    // timestep_sampling has disable_when='infonoise_enabled==true' but NO disable_value
    // in the mock schema. Old behavior reset to prop.default; new behavior keeps the
    // user's value so the backend model_validator surfaces the real conflict combo.
    render(
      <SchemaForm
        schema={schema}
        values={{
          optimizer_type: 'adamw',
          learning_rate: 0.0001,
          lr_scheduler: 'none',
          infonoise_enabled: true,
          timestep_sampling: 'uniform',
        }}
        onChange={onChange}
        advancedMode
      />,
    )

    // Yield once to let the disable-takeover useEffect run (it would have called
    // onChange under the old behavior).
    await new Promise((r) => setTimeout(r, 0))
    expect(onChange).not.toHaveBeenCalled()
    const timestepSelect = screen.getAllByRole('combobox').find((el) => (el as HTMLSelectElement).value === 'uniform') as HTMLSelectElement
    expect(timestepSelect.value).toBe('uniform')
    expect(timestepSelect).toBeDisabled()
  })
})
