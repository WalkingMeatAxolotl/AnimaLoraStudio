import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import {
  Checkbox,
  Input,
  Select,
  Textarea,
  controlClassName,
} from './FormControl'

describe('FormControl primitives', () => {
  it('applies the default input contract without changing native semantics', () => {
    render(<Input aria-label="Name" />)
    expect(screen.getByRole('textbox', { name: 'Name' }))
      .toHaveClass('form-control', 'form-control-surface')
  })

  it('maps compact, surface, and technical-value options', () => {
    render(<Input aria-label="Path" controlSize="sm" surface="canvas" mono />)
    expect(screen.getByRole('textbox', { name: 'Path' }))
      .toHaveClass('form-control-sm', 'form-control-canvas', 'form-control-mono')
  })

  it('preserves native select and textarea elements', () => {
    render(
      <>
        <Select aria-label="Mode" surface="sunken"><option>Auto</option></Select>
        <Textarea aria-label="Prompt" rows={3} />
      </>,
    )

    expect(screen.getByRole('combobox', { name: 'Mode' }))
      .toHaveClass('form-control-sunken')
    expect(screen.getByRole('textbox', { name: 'Prompt' }))
      .toHaveClass('form-control-textarea')
  })

  it('exposes invalid state to assistive technology', () => {
    render(<Input aria-label="JSON" invalid />)
    expect(screen.getByRole('textbox', { name: 'JSON' }))
      .toHaveAttribute('aria-invalid', 'true')
  })

  it('keeps checkbox behavior native while applying the shared size', () => {
    render(<Checkbox aria-label="Enabled" controlSize="sm" disabled />)
    const checkbox = screen.getByRole('checkbox', { name: 'Enabled' })
    expect(checkbox).toHaveClass('form-checkbox', 'form-checkbox-sm')
    expect(checkbox).toBeDisabled()
  })

  it('builds classes for consumers that cannot use the React primitive yet', () => {
    expect(controlClassName({ size: 'sm', surface: 'sunken', className: 'max-w-32' }))
      .toBe('form-control form-control-sm form-control-sunken max-w-32')
  })
})
