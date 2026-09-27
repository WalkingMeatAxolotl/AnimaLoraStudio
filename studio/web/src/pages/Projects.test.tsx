import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type ProjectSummary } from '../api/client'
import { DialogProvider } from '../components/Dialog'
import { ToastProvider } from '../components/Toast'
import i18n from '../i18n'
import ProjectsPage, {
  filterProjects,
  ProjectBatchBar,
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

class FakeEventSource {
  static readonly OPEN = 1
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  readyState = FakeEventSource.OPEN
  close() { this.readyState = 2 }
}

beforeEach(() => {
  localStorage.clear()
  vi.stubGlobal('EventSource', FakeEventSource)
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

function renderProjectsPage() {
  return render(
    <MemoryRouter>
      <ToastProvider>
        <DialogProvider>
          <ProjectsPage />
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>,
  )
}

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

  it('turns the card into one checkbox-backed selection target in batch mode', () => {
    const onOpen = vi.fn()
    const onToggleSelected = vi.fn()
    render(
      <ProjectCard
        project={ITEMS[0]}
        selectable
        selected
        onToggleSelected={onToggleSelected}
        onClick={onOpen}
      />,
    )

    const card = screen.getByRole('article', { name: 'Kaguya' })
    const checkbox = within(card).getByRole('checkbox', {
      name: i18n.t('projects.selectProject', { title: 'Kaguya' }),
    })
    expect(card).toHaveAttribute('data-selected', 'true')
    expect(checkbox).toBeChecked()
    expect(within(card).queryByRole('button')).not.toBeInTheDocument()

    fireEvent.click(card)
    fireEvent.click(checkbox)

    expect(onToggleSelected).toHaveBeenCalledTimes(2)
    expect(onOpen).not.toHaveBeenCalled()
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

describe('ProjectBatchBar', () => {
  const handlers = {
    onSelectCurrent: vi.fn(),
    onClear: vi.fn(),
    onArchive: vi.fn(),
    onRestore: vi.fn(),
    onDelete: vi.fn(),
  }

  it('uses archive as the active-view batch action', () => {
    render(
      <ProjectBatchBar
        archived={false}
        selectedCount={2}
        allCurrentSelected={false}
        busyAction={null}
        {...handlers}
      />,
    )

    expect(screen.getByRole('status')).toHaveTextContent(
      i18n.t('projects.batchSelectedCount', { n: 2 }),
    )
    fireEvent.click(screen.getByRole('button', {
      name: i18n.t('projects.selectCurrentResults'),
    }))
    fireEvent.click(screen.getByRole('button', {
      name: i18n.t('projects.archiveSelected', { n: 2 }),
    }))
    expect(handlers.onSelectCurrent).toHaveBeenCalledTimes(1)
    expect(handlers.onArchive).toHaveBeenCalledTimes(1)
    expect(screen.queryByText(i18n.t('projects.deleteSelected', { n: 2 }))).not.toBeInTheDocument()
  })

  it('offers restore before permanent delete in the archived view', () => {
    render(
      <ProjectBatchBar
        archived
        selectedCount={3}
        allCurrentSelected
        busyAction="delete"
        {...handlers}
      />,
    )

    const selectionGroup = screen.getByRole('group', {
      name: i18n.t('projects.batchSelectionActions'),
    })
    const mutationGroup = screen.getByRole('group', {
      name: i18n.t('projects.batchActions'),
    })
    const restore = within(mutationGroup).getByRole('button', {
      name: i18n.t('projects.restoreSelected', { n: 3 }),
    })
    const remove = within(mutationGroup).getByRole('button', {
      name: i18n.t('projects.deleteSelected', { n: 3 }),
    })
    expect(within(selectionGroup).getByRole('button', {
      name: i18n.t('projects.selectCurrentResults'),
    })).toBeDisabled()
    expect(within(selectionGroup).getByRole('button', {
      name: i18n.t('common.deselect'),
    })).toBeDisabled()
    expect(within(selectionGroup).queryByRole('button', {
      name: i18n.t('projects.restoreSelected', { n: 3 }),
    })).not.toBeInTheDocument()
    expect(restore).toBeDisabled()
    expect(remove).toBeDisabled()
    expect(remove).toHaveAttribute('aria-busy', 'true')
    expect(restore.compareDocumentPosition(remove) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })
})

describe('ProjectsPage batch workflow', () => {
  it('archives the current active result selection through one confirmation', async () => {
    vi.spyOn(api, 'listProjects').mockResolvedValue(ITEMS)
    const archiveProjects = vi.spyOn(api, 'archiveProjects').mockResolvedValue({
      updated: [3, 2, 1],
    })
    renderProjectsPage()

    fireEvent.click(await screen.findByRole('button', { name: i18n.t('projects.batchManage') }))
    expect(i18n.t('common.done')).toBe('完成')
    expect(screen.getByRole('button', { name: '完成' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: i18n.t('projects.selectCurrentResults') }))
    fireEvent.click(screen.getByRole('button', {
      name: i18n.t('projects.archiveSelected', { n: 3 }),
    }))
    fireEvent.click(await screen.findByRole('button', {
      name: i18n.t('projects.batchArchiveAction', { n: 3 }),
    }))

    await waitFor(() => expect(archiveProjects).toHaveBeenCalledWith([3, 2, 1]))
    await waitFor(() => expect(screen.getByRole('button', {
      name: i18n.t('projects.batchManage'),
    })).toHaveFocus())
  })

  it('restores selected archived projects from the same batch mode', async () => {
    const archived = mk({ id: 4, title: 'Archived', archived_at: 123 })
    vi.spyOn(api, 'listProjects').mockResolvedValue([...ITEMS, archived])
    const unarchiveProjects = vi.spyOn(api, 'unarchiveProjects').mockResolvedValue({ updated: [4] })
    renderProjectsPage()

    fireEvent.click(await screen.findByRole('button', {
      name: i18n.t('projects.archivedToggle', { n: 1 }),
    }))
    fireEvent.click(screen.getByRole('button', { name: i18n.t('projects.batchManage') }))
    fireEvent.click(screen.getByRole('checkbox', {
      name: i18n.t('projects.selectProject', { title: 'Archived' }),
    }))
    fireEvent.click(screen.getByRole('button', {
      name: i18n.t('projects.restoreSelected', { n: 1 }),
    }))
    fireEvent.click(await screen.findByRole('button', {
      name: i18n.t('projects.batchRestoreAction', { n: 1 }),
    }))

    await waitFor(() => expect(unarchiveProjects).toHaveBeenCalledWith([4]))
  })

  it('permanently deletes selected archived projects through the batch endpoint', async () => {
    const archived = mk({ id: 4, title: 'Archived', archived_at: 123 })
    vi.spyOn(api, 'listProjects').mockResolvedValue([...ITEMS, archived])
    const deleteProjects = vi.spyOn(api, 'deleteProjects').mockResolvedValue({
      deleted: [4],
      failed: [],
    })
    renderProjectsPage()

    fireEvent.click(await screen.findByRole('button', {
      name: i18n.t('projects.archivedToggle', { n: 1 }),
    }))
    fireEvent.click(screen.getByRole('button', { name: i18n.t('projects.batchManage') }))
    fireEvent.click(screen.getByRole('checkbox', {
      name: i18n.t('projects.selectProject', { title: 'Archived' }),
    }))
    fireEvent.click(screen.getByRole('button', {
      name: i18n.t('projects.deleteSelected', { n: 1 }),
    }))
    fireEvent.click(await screen.findByRole('button', {
      name: i18n.t('projects.batchDeleteAction', { n: 1 }),
    }))

    await waitFor(() => expect(deleteProjects).toHaveBeenCalledWith([4]))
  })

  it('keeps only filesystem failures selected after a partial delete', async () => {
    const deleted = mk({ id: 4, title: 'Deleted', archived_at: 123 })
    const failed = mk({ id: 5, title: 'Failed', archived_at: 124 })
    vi.spyOn(api, 'listProjects')
      .mockResolvedValueOnce([...ITEMS, deleted, failed])
      .mockResolvedValue([...ITEMS, failed])
    vi.spyOn(api, 'deleteProjects').mockResolvedValue({
      deleted: [4],
      failed: [{ id: 5, code: 'project.delete_failed', message: 'failed' }],
    })
    renderProjectsPage()

    fireEvent.click(await screen.findByRole('button', {
      name: i18n.t('projects.archivedToggle', { n: 2 }),
    }))
    fireEvent.click(screen.getByRole('button', { name: i18n.t('projects.batchManage') }))
    fireEvent.click(screen.getByRole('button', { name: i18n.t('projects.selectCurrentResults') }))
    fireEvent.click(screen.getByRole('button', {
      name: i18n.t('projects.deleteSelected', { n: 2 }),
    }))
    fireEvent.click(await screen.findByRole('button', {
      name: i18n.t('projects.batchDeleteAction', { n: 2 }),
    }))

    expect(await screen.findByText(i18n.t('projects.batchDeletePartial', {
      deleted: 1,
      failed: 1,
    }))).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText(
      i18n.t('projects.batchSelectedCount', { n: 1 }),
    )).toBeInTheDocument())
    expect(screen.getByRole('checkbox', {
      name: i18n.t('projects.selectProject', { title: 'Failed' }),
    })).toBeChecked()
    expect(screen.getByRole('button', { name: i18n.t('common.done') })).toBeInTheDocument()
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
