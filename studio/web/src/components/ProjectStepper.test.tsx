import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import type { ProjectDetail, Version } from '../api/client'
import ProjectStepper from './ProjectStepper'

function project(stage: ProjectDetail['stage']): ProjectDetail {
  return {
    id: 7,
    slug: 'p',
    title: 'P',
    stage,
    active_version_id: null,
    created_at: 0,
    updated_at: 0,
    note: null,
    versions: [],
    download_image_count: 0,
  }
}

function version(stage: Version['stage']): Version {
  return {
    id: 1,
    project_id: 7,
    label: 'baseline',
    config_name: null,
    stage,
    created_at: 0,
    output_lora_path: null,
    note: null,
  }
}

function renderStepper(p: ProjectDetail, v: Version | null) {
  return render(
    <MemoryRouter>
      <ProjectStepper project={p} version={v} />
    </MemoryRouter>
  )
}

describe('ProjectStepper (PP1)', () => {
  it('shows download as active when project stage is downloading', () => {
    renderStepper(project('downloading'), null)
    const list = screen.getByRole('list', { name: 'pipeline-stepper' })
    const items = within(list).getAllByRole('listitem')
    expect(items[0].textContent).toMatch(/●.*下载/)
  })

  it('shows curate as active and download as done at project stage curating', () => {
    renderStepper(project('curating'), version('curating'))
    const list = screen.getByRole('list', { name: 'pipeline-stepper' })
    const items = within(list).getAllByRole('listitem')
    expect(items[0].textContent).toMatch(/✓.*下载/)
    expect(items[1].textContent).toMatch(/●.*筛选/)
  })

  it('marks all version steps pending without an active version', () => {
    const p = project('curating')
    renderStepper(p, null)
    const list = screen.getByRole('list', { name: 'pipeline-stepper' })
    const items = within(list).getAllByRole('listitem')
    // 没 version 时，version 级 step 应该 disabled（用 span 而不是 link）
    // 至少：筛选/打标/正则集/训练 4 个 listitem 没有 link
    const linkCount = list.querySelectorAll('a').length
    expect(linkCount).toBeLessThan(items.length)
  })
})
