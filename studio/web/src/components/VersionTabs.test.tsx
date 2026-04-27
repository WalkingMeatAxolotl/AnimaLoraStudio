import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { Version } from '../api/client'
import VersionTabs from './VersionTabs'

const versions: Version[] = [
  {
    id: 1,
    project_id: 10,
    label: 'baseline',
    config_name: null,
    stage: 'curating',
    created_at: 0,
    output_lora_path: null,
    note: null,
  },
  {
    id: 2,
    project_id: 10,
    label: 'high-lr',
    config_name: null,
    stage: 'training',
    created_at: 1,
    output_lora_path: null,
    note: null,
  },
]

describe('VersionTabs (PP1)', () => {
  it('renders one tab per version + a + button', () => {
    render(
      <VersionTabs
        versions={versions}
        activeId={1}
        onSelect={() => {}}
        onCreate={() => {}}
        onDelete={() => {}}
      />
    )
    expect(screen.getByRole('tab', { name: /baseline/ })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /high-lr/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /新版本/ })).toBeInTheDocument()
  })

  it('marks the active tab via aria-selected', () => {
    render(
      <VersionTabs
        versions={versions}
        activeId={2}
        onSelect={() => {}}
        onCreate={() => {}}
        onDelete={() => {}}
      />
    )
    expect(screen.getByRole('tab', { name: /high-lr/ })).toHaveAttribute(
      'aria-selected',
      'true'
    )
    expect(screen.getByRole('tab', { name: /baseline/ })).toHaveAttribute(
      'aria-selected',
      'false'
    )
  })

  it('clicking a tab calls onSelect with its id', async () => {
    const onSelect = vi.fn()
    const user = userEvent.setup()
    render(
      <VersionTabs
        versions={versions}
        activeId={1}
        onSelect={onSelect}
        onCreate={() => {}}
        onDelete={() => {}}
      />
    )
    await user.click(screen.getByRole('tab', { name: /high-lr/ }))
    expect(onSelect).toHaveBeenCalledWith(2)
  })

  it('shows × delete button only on active tab when >1 version exists', () => {
    render(
      <VersionTabs
        versions={versions}
        activeId={1}
        onSelect={() => {}}
        onCreate={() => {}}
        onDelete={() => {}}
      />
    )
    // 仅 active tab (baseline) 旁有删除按钮
    expect(
      screen.getByRole('button', { name: /删除版本 baseline/ })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole('button', { name: /删除版本 high-lr/ })
    ).toBeNull()
  })

  it('hides × when only one version remains', () => {
    render(
      <VersionTabs
        versions={[versions[0]]}
        activeId={1}
        onSelect={() => {}}
        onCreate={() => {}}
        onDelete={() => {}}
      />
    )
    expect(
      screen.queryByRole('button', { name: /删除版本/ })
    ).toBeNull()
  })
})
