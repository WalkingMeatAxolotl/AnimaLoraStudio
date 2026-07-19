import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { SourceSegmented } from './Regularization'

describe('regularization source selector', () => {
  it('uses the shared pill style and switches source', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<SourceSegmented source="ai" onChange={onChange} />)

    const ai = screen.getByRole('radio', { name: /AI 先验/ })
    const booru = screen.getByRole('radio', { name: /Booru 抓取/ })

    expect(ai).toHaveClass('pill-radio', 'pill-radio-content', 'on')
    expect(booru).toHaveClass('pill-radio', 'pill-radio-content')
    expect(booru).not.toHaveClass('on')
    expect(ai.querySelector('.pill-radio-dot')).not.toBeNull()
    expect(booru.querySelector('.pill-radio-dot')).not.toBeNull()

    await user.click(booru)
    expect(onChange).toHaveBeenCalledWith('booru')
  })
})
