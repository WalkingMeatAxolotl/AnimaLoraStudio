import { fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import CheckboxDropdown, { type CheckboxDropdownOption } from './CheckboxDropdown'

const OPTIONS: CheckboxDropdownOption[] = [
  { value: 'a', label: 'Alpha' },
  { value: 'b', label: 'Beta' },
]

function Harness() {
  const [selected, setSelected] = useState(new Set(['a']))
  return (
    <CheckboxDropdown
      label="Candidates"
      options={OPTIONS}
      selected={selected}
      onChange={setSelected}
    />
  )
}

describe('CheckboxDropdown', () => {
  it('uses the compact button primitive and exposes expanded state', () => {
    render(<Harness />)
    const trigger = screen.getByRole('button', { name: 'Candidates' })
    expect(trigger).toHaveClass('btn', 'btn-secondary', 'btn-xs')
    expect(trigger).toHaveAttribute('aria-expanded', 'false')

    fireEvent.click(trigger)
    expect(trigger).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText('已选 1 / 2')).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: 'Alpha' }))
      .toHaveClass('form-checkbox', 'form-checkbox-sm')
  })

  it('selects and clears the complete option set', () => {
    render(<Harness />)
    fireEvent.click(screen.getByRole('button', { name: 'Candidates' }))
    fireEvent.click(screen.getByRole('button', { name: '全选' }))
    expect(screen.getByRole('button', { name: 'Candidates' })).toHaveTextContent('2/2')
    expect(screen.getByRole('button', { name: '取消选择' })).toBeInTheDocument()
  })

  it('uses the localized empty state unless a custom hint is supplied', () => {
    const { rerender } = render(
      <CheckboxDropdown label="Empty" options={[]} selected={new Set()} onChange={() => {}} />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Empty' }))
    expect(screen.getByText('暂无可选项')).toBeInTheDocument()

    rerender(
      <CheckboxDropdown
        label="Empty"
        options={[]}
        selected={new Set()}
        onChange={() => {}}
        emptyHint="Nothing configured"
      />,
    )
    expect(screen.getByText('Nothing configured')).toBeInTheDocument()
  })
})
