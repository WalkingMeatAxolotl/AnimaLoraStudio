import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { ProjectSummary } from '../api/client'
import { filterProjects, ProjectFilterBar } from './Projects'

function mk(over: Partial<ProjectSummary> & { id: number }): ProjectSummary {
  return {
    slug: `p${over.id}`,
    title: `Project ${over.id}`,
    active_version_id: null,
    active_version_label: null,
    active_version_status: null,
    active_version_phase: null,
    created_at: over.id,
    updated_at: over.id,
    archived_at: null,
    note: null,
    ...over,
  }
}

const ITEMS: ProjectSummary[] = [
  mk({ id: 1, title: 'Kaguya', slug: 'kaguya', active_version_status: 'completed', note: 'moon princess' }),
  mk({ id: 2, title: 'Miku', slug: 'miku', active_version_status: 'training' }),
  mk({ id: 3, title: 'Asuka', slug: 'asuka-style', active_version_status: 'preparing' }),
]

describe('filterProjects', () => {
  it('default: no filter, sorted by updated_at desc', () => {
    const r = filterProjects(ITEMS, { query: '', status: 'all', sort: 'updated' })
    expect(r.map((p) => p.id)).toEqual([3, 2, 1])
  })

  it('query matches title / slug / note, case-insensitive', () => {
    const byTitle = filterProjects(ITEMS, { query: 'kagu', status: 'all', sort: 'updated' })
    expect(byTitle.map((p) => p.id)).toEqual([1])
    const bySlug = filterProjects(ITEMS, { query: 'STYLE', status: 'all', sort: 'updated' })
    expect(bySlug.map((p) => p.id)).toEqual([3])
    const byNote = filterProjects(ITEMS, { query: 'princess', status: 'all', sort: 'updated' })
    expect(byNote.map((p) => p.id)).toEqual([1])
  })

  it('status filter narrows to matching active version status', () => {
    const r = filterProjects(ITEMS, { query: '', status: 'training', sort: 'updated' })
    expect(r.map((p) => p.id)).toEqual([2])
  })

  it('query and status compose', () => {
    const r = filterProjects(ITEMS, { query: 'miku', status: 'completed', sort: 'updated' })
    expect(r).toEqual([])
  })

  it('sort by title uses locale compare', () => {
    const r = filterProjects(ITEMS, { query: '', status: 'all', sort: 'title' })
    expect(r.map((p) => p.title)).toEqual(['Asuka', 'Kaguya', 'Miku'])
  })

  it('does not mutate the input array', () => {
    const before = ITEMS.map((p) => p.id)
    filterProjects(ITEMS, { query: '', status: 'all', sort: 'title' })
    expect(ITEMS.map((p) => p.id)).toEqual(before)
  })
})

describe('ProjectFilterBar', () => {
  it('keeps a named hidden target and forwards search, filter, and sort changes', () => {
    const onQuery = vi.fn()
    const onStatus = vi.fn()
    const onSort = vi.fn()
    const { rerender } = render(
      <ProjectFilterBar
        hidden
        query=""
        onQuery={onQuery}
        status="all"
        onStatus={onStatus}
        sort="updated"
        onSort={onSort}
      />,
    )

    const toolbar = screen.getByTestId('projects-list-toolbar')
    expect(toolbar).toHaveAttribute('id', 'projects-list-toolbar')
    expect(toolbar).toHaveAttribute('role', 'region')
    expect(toolbar).toHaveAttribute('aria-label')
    expect(toolbar).toHaveAttribute('hidden')

    rerender(
      <ProjectFilterBar
        query=""
        onQuery={onQuery}
        status="all"
        onStatus={onStatus}
        sort="updated"
        onSort={onSort}
      />,
    )

    const controls = within(toolbar)
    expect(toolbar).toHaveAccessibleName()
    fireEvent.change(controls.getByRole('textbox'), { target: { value: 'kagu' } })
    const selects = controls.getAllByRole('combobox')
    fireEvent.change(selects[0], { target: { value: 'training' } })
    fireEvent.change(selects[1], { target: { value: 'title' } })

    expect(onQuery).toHaveBeenCalledWith('kagu')
    expect(onStatus).toHaveBeenCalledWith('training')
    expect(onSort).toHaveBeenCalledWith('title')
  })
})
