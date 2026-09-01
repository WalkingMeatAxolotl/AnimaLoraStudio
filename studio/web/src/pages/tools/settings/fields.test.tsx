import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { Bool, SettingsField, SettingsSection } from './fields'

describe('settings hierarchy', () => {
  it('uses shared panel, field, and spacing roles without treating UI labels as code', () => {
    const { container } = render(
      <SettingsSection title="Runtime">
        <SettingsField label="Cache path">
          <input aria-label="Cache path value" />
        </SettingsField>
      </SettingsSection>,
    )

    expect(container.querySelector('section')).toHaveClass('p-section', 'gap-field')
    expect(screen.getByRole('heading', { level: 2, name: 'Runtime' }))
      .toHaveClass('type-panel-title')
    expect(screen.getByText('Cache path')).toHaveClass('type-field-label')
    expect(screen.getByText('Cache path').parentElement?.parentElement)
      .toHaveClass('gap-field')
    expect(screen.getByText('Cache path')).not.toHaveClass('font-mono')
  })

  it('uses the compact sunken form-control contract for boolean settings', () => {
    render(<Bool value onChange={() => {}} />)
    expect(screen.getByRole('combobox'))
      .toHaveClass('form-control', 'form-control-sm', 'form-control-sunken')
  })
})
