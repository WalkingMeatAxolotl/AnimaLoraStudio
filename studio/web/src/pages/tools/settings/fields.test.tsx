import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SettingsField, SettingsSection } from './fields'

describe('settings typography hierarchy', () => {
  it('uses panel and field roles without treating UI labels as code', () => {
    render(
      <SettingsSection title="Runtime">
        <SettingsField label="Cache path">
          <input aria-label="Cache path value" />
        </SettingsField>
      </SettingsSection>,
    )

    expect(screen.getByRole('heading', { level: 2, name: 'Runtime' }))
      .toHaveClass('type-panel-title')
    expect(screen.getByText('Cache path')).toHaveClass('type-field-label')
    expect(screen.getByText('Cache path')).not.toHaveClass('font-mono')
  })
})
