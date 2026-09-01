import { render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) => {
      if (key === 'saveBar.save') return `Save (${String(options?.n ?? '')})`
      if (key === 'saveBar.saved') return 'Saved'
      if (key === 'saveBar.restorePoints') return 'Restore points'
      if (key === 'saveBar.tooltip') return 'Save captions'
      return key
    },
  }),
}))

vi.mock('../api/client', () => ({
  api: {
    listCaptionSnapshots: vi.fn().mockResolvedValue([]),
    restoreCaptionSnapshot: vi.fn(),
    deleteCaptionSnapshot: vi.fn(),
  },
}))

vi.mock('./Dialog', () => ({
  useDialog: () => ({ confirm: vi.fn().mockResolvedValue(true) }),
}))

vi.mock('./Toast', () => ({
  useToast: () => ({ toast: vi.fn() }),
}))

import SaveBar from './SaveBar'

describe('SaveBar', () => {
  it('keeps the text-first save action last and restore secondary', () => {
    const { container } = render(
      <SaveBar
        pid={1}
        vid={2}
        dirtyCount={3}
        onSave={vi.fn().mockResolvedValue(undefined)}
        onAfterRestore={vi.fn().mockResolvedValue(undefined)}
      />,
    )

    const group = container.querySelector('[data-action-slot="secondary"]')?.parentElement
    expect(group).not.toBeNull()
    expect(within(group as HTMLElement).getAllByRole('button').map((button) => button.textContent))
      .toEqual(['Restore points', 'Save (3)'])
    expect(screen.getByRole('button', { name: 'Save (3)' })).toHaveClass('btn-primary')
    expect(screen.getByRole('button', { name: 'Restore points' })).toHaveClass('btn-ghost')
    expect(group?.textContent).not.toMatch(/[💾🕒]/u)
  })

  it('keeps the clean save position stable without primary emphasis', () => {
    render(
      <SaveBar
        pid={1}
        vid={2}
        dirtyCount={0}
        onSave={vi.fn().mockResolvedValue(undefined)}
        onAfterRestore={vi.fn().mockResolvedValue(undefined)}
      />,
    )

    expect(screen.getByRole('button', { name: 'Saved' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Saved' })).toHaveClass('btn-secondary')
    expect(screen.getByRole('button', { name: 'Saved' }).closest('[data-action-slot]'))
      .toHaveAttribute('data-action-slot', 'primary')
  })
})
