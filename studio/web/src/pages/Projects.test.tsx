import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { ProjectSummary } from '../api/client'
import i18n from '../i18n'
import {
  filterProjects,
  ProjectCard,
  ProjectFilterBar,
  ProjectsCollectionSurface,
} from './Projects'

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

describe('ProjectCard', () => {
  it('uses a composite card without nested controls and isolates local actions', () => {
    const onOpen = vi.fn()
    const onEdit = vi.fn()
    const onArchive = vi.fn()
    render(
      <ProjectCard
        project={ITEMS[0]}
        onClick={onOpen}
        onEdit={onEdit}
        onArchive={onArchive}
      />,
    )

    const card = screen.getByRole('article', { name: 'Kaguya' })
    expect(card).toHaveClass('card', 'card-hover', 'card-pad-md')
    expect(card.querySelector('button button')).toBeNull()

    const [openButton, editButton, archiveButton] = within(card).getAllByRole('button')
    const title = within(card).getByRole('heading', { level: 2, name: 'Kaguya' })
    expect(title).toHaveAttribute('title', 'Kaguya')
    expect(title.parentElement).toHaveClass('flex-col', 'gap-related')
    expect(openButton).toHaveAccessibleName(i18n.t('projects.openProject', { title: 'Kaguya' }))
    expect(openButton).toHaveClass(
      'absolute',
      'inset-0',
      'z-0',
      'focus-visible:ring-inset',
    )
    expect(openButton).not.toContainElement(editButton)
    expect(editButton).toHaveClass('btn', 'btn-ghost', 'btn-xs', 'btn-icon')
    expect(editButton.parentElement).toHaveClass(
      'pointer-events-none',
      'gap-related',
      'motion-reduce:transition-none',
      'group-hover:pointer-events-auto',
      'group-focus-within:pointer-events-auto',
    )

    openButton.focus()
    expect(openButton).toHaveFocus()
    editButton.focus()
    expect(editButton).toHaveFocus()
    fireEvent.click(openButton)
    fireEvent.click(editButton)
    fireEvent.click(archiveButton)

    expect(onOpen).toHaveBeenCalledTimes(1)
    expect(onEdit).toHaveBeenCalledTimes(1)
    expect(onArchive).toHaveBeenCalledTimes(1)
  })

  it('keeps archived restore and delete actions outside the primary card action', () => {
    const onOpen = vi.fn()
    const onUnarchive = vi.fn()
    const onDelete = vi.fn()
    render(
      <ProjectCard
        project={ITEMS[0]}
        archived
        onClick={onOpen}
        onUnarchive={onUnarchive}
        onDelete={onDelete}
      />,
    )

    const buttons = within(screen.getByRole('article', { name: 'Kaguya' })).getAllByRole('button')
    expect(buttons).toHaveLength(4)
    fireEvent.click(buttons[2])
    fireEvent.click(buttons[3])

    expect(onOpen).not.toHaveBeenCalled()
    expect(onUnarchive).toHaveBeenCalledTimes(1)
    expect(onDelete).toHaveBeenCalledTimes(1)
  })
})

describe('ProjectsCollectionSurface', () => {
  function surface(
    props: Partial<{
      loading: boolean
      error: string | null
      itemCount: number
      visibleCount: number
    }> = {},
  ) {
    return (
      <ProjectsCollectionSurface
        loading={false}
        error={null}
        itemCount={0}
        visibleCount={0}
        {...props}
      >
        <div data-testid="project-content">projects</div>
      </ProjectsCollectionSurface>
    )
  }

  it('keeps loading, error, empty, no-match, and content states mutually exclusive', () => {
    const { container, rerender } = render(surface({ loading: true }))
    expect(screen.getByRole('status')).toHaveAttribute('aria-busy', 'true')
    expect(container.querySelectorAll('.ui-project-card-skeleton')).toHaveLength(3)
    expect(container.querySelector('.ui-project-card-skeleton')).toHaveClass('ui-skeleton')
    expect(screen.queryByTestId('project-content')).not.toBeInTheDocument()

    rerender(surface({ error: 'network unavailable' }))
    expect(screen.getByRole('alert')).toHaveTextContent('network unavailable')
    expect(container.querySelector('.empty-state')).not.toBeInTheDocument()
    expect(screen.queryByTestId('project-content')).not.toBeInTheDocument()

    rerender(surface())
    expect(screen.getByText(i18n.t('projects.noProjects'))).toBeInTheDocument()
    expect(container.querySelector('.empty-state')).toBeInTheDocument()

    rerender(surface({ itemCount: 2, visibleCount: 0 }))
    expect(screen.getByText(i18n.t('common.noResults'))).toBeInTheDocument()
    expect(container.querySelector('.empty-state-sm')).toBeInTheDocument()

    rerender(surface({ itemCount: 2, visibleCount: 2 }))
    expect(screen.getByTestId('project-content')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('keeps stale visible content available alongside a refresh error', () => {
    render(surface({ error: 'refresh failed', itemCount: 2, visibleCount: 1 }))
    expect(screen.getByRole('alert')).toHaveTextContent('refresh failed')
    expect(screen.getByTestId('project-content')).toBeInTheDocument()
    expect(document.querySelector('.empty-state')).not.toBeInTheDocument()
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
