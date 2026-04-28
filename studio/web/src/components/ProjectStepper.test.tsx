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
    // 至少：筛选/打标/编辑/正则集/训练 5 个 listitem 没有 link
    const linkCount = list.querySelectorAll('a').length
    expect(linkCount).toBeLessThan(items.length)
  })

  it('exposes 6 steps including the split tag/edit pair', () => {
    renderStepper(project('curating'), version('curating'))
    const list = screen.getByRole('list', { name: 'pipeline-stepper' })
    const items = within(list).getAllByRole('listitem')
    expect(items).toHaveLength(6)
    expect(items[2].textContent).toMatch(/打标/)
    expect(items[3].textContent).toMatch(/标签编辑/)
  })

  it('marks tag/edit done when all train images have captions', () => {
    const v: Version = {
      ...version('tagging'),
      stats: {
        train_image_count: 10,
        tagged_image_count: 10,
        train_folders: [],
        reg_image_count: 0,
        reg_meta_exists: false,
        has_output: false,
      },
    }
    renderStepper(project('tagging'), v)
    const list = screen.getByRole('list', { name: 'pipeline-stepper' })
    const items = within(list).getAllByRole('listitem')
    expect(items[2].textContent).toMatch(/✓.*打标/)
    expect(items[3].textContent).toMatch(/✓.*标签编辑/)
  })

  it('keeps tag active while some images lack captions', () => {
    const v: Version = {
      ...version('tagging'),
      stats: {
        train_image_count: 10,
        tagged_image_count: 5,
        train_folders: [],
        reg_image_count: 0,
        reg_meta_exists: false,
        has_output: false,
      },
    }
    renderStepper(project('tagging'), v)
    const list = screen.getByRole('list', { name: 'pipeline-stepper' })
    const items = within(list).getAllByRole('listitem')
    expect(items[2].textContent).toMatch(/●.*打标/)
  })

  it('marks reg done when reg meta exists and images present', () => {
    const v: Version = {
      ...version('regularizing'),
      stats: {
        train_image_count: 10,
        tagged_image_count: 10,
        train_folders: [],
        reg_image_count: 8,
        reg_meta_exists: true,
        has_output: false,
      },
    }
    renderStepper(project('tagging'), v)
    const list = screen.getByRole('list', { name: 'pipeline-stepper' })
    const items = within(list).getAllByRole('listitem')
    expect(items[4].textContent).toMatch(/✓.*正则集/)
  })
})
