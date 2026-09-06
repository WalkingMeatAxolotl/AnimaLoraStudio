import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Outlet, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  api,
  type CurationValidationView,
  type CurationView,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import CurationPage from './Curation'

const mocks = vi.hoisted(() => ({
  toast: vi.fn(),
  confirm: vi.fn(async () => true),
  reload: vi.fn(async () => undefined),
  onEvent: undefined as undefined | ((event: unknown) => void),
}))

vi.mock('../../../components/Toast', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Toast')>()
  return { ...actual, useToast: () => ({ toast: mocks.toast }) }
})
vi.mock('../../../components/Dialog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Dialog')>()
  return { ...actual, useDialog: () => ({ confirm: mocks.confirm }) }
})
vi.mock('../../../lib/useEventStream', () => ({
  useEventStream: (callback: (event: unknown) => void) => { mocks.onEvent = callback },
}))
vi.mock('../../../components/ImageGrid', () => ({
  applySelection: (selected: Set<string>, name: string) => {
    const next = new Set(selected)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    return { next, anchor: name }
  },
  default: ({
    items,
    selected,
    onSelect,
    onActivate,
    ariaLabel,
    className,
    contentClassName,
    emptyHint,
  }: {
    items: { name: string }[]
    selected: Set<string>
    onSelect: (name: string, event: React.MouseEvent) => void
    onActivate?: (name: string) => void
    ariaLabel?: string
    className?: string
    contentClassName?: string
    emptyHint?: string
  }) => (
    <div role="grid" aria-label={ariaLabel} className={className} data-content-class={contentClassName}>
      {items.length === 0 ? <span>{emptyHint}</span> : items.map((item) => (
        <div key={item.name}>
          <button type="button" onClick={(event) => onSelect(item.name, event)} aria-pressed={selected.has(item.name)}>
            选择 {item.name}
          </button>
          <button type="button" onClick={() => onActivate?.(item.name)}>
            预览 {item.name}
          </button>
        </div>
      ))}
    </div>
  ),
}))
vi.mock('../../../components/ImagePreviewModal', () => ({
  default: ({ shortcutHint, onClose }: { shortcutHint?: string; onClose: () => void }) => (
    <div role="dialog" aria-label="图片预览">
      <span>{shortcutHint}</span>
      <button type="button" onClick={onClose}>关闭预览</button>
    </div>
  ),
}))

const trainView: CurationView = {
  left: [
    { name: 'a.png', mtime: 2 },
    { name: 'b.png', mtime: 1 },
  ],
  right: {
    '1_data': [{ name: 'train.png', origin: 'train.png', mtime: 3 }],
  },
  download_total: 3,
  train_total: 1,
  folders: ['1_data'],
}

const validationView: CurationValidationView = {
  left: trainView.left,
  right: [{ name: 'val.png', folder: 'manual', mtime: 4 }],
  download_total: 3,
  val_total: 1,
}

function renderPage() {
  const project = { id: 1 } as ProjectDetail
  const activeVersion = { id: 2 } as Version
  return render(
    <MemoryRouter
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      initialEntries={['/curation']}
    >
      <Routes>
        <Route element={<Outlet context={{ project, activeVersion, reload: mocks.reload }} />}>
          <Route path="/curation" element={<CurationPage />} />
        </Route>
      </Routes>
    </MemoryRouter>,
  )
}

async function ready() {
  await screen.findByRole('grid', { name: '未分配的数据集图片' })
  await screen.findByRole('grid', { name: '当前分组中的训练图片' })
}

beforeEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
  localStorage.clear()
  mocks.onEvent = undefined
  vi.spyOn(api, 'getCuration').mockResolvedValue(trainView)
  vi.spyOn(api, 'getCurationValidation').mockResolvedValue(validationView)
  vi.spyOn(api, 'copyToTrain').mockResolvedValue({ copied: ['a.png'], skipped: [], missing: [] })
  vi.spyOn(api, 'copyToValidation').mockResolvedValue({ copied: ['a.png'], skipped: [], missing: [] })
  vi.spyOn(api, 'removeFromTrain').mockResolvedValue({ removed: ['train.png'], missing: [] })
  vi.spyOn(api, 'removeFromValidation').mockResolvedValue({ removed: ['val.png'], missing: [] })
  vi.spyOn(api, 'folderOp').mockResolvedValue({})
})

describe('Curation workspace', () => {
  it('shows training as the explicit default and adds selected images to the active folder', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    const destination = screen.getByRole('combobox', { name: '加入目标' })
    expect(destination).toHaveValue('train')
    expect(within(destination).getByRole('option', { name: '训练分组（默认）' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '选择 a.png' }))
    const addButton = screen.getByRole('button', { name: '加入 1 张到 1_data' })
    expect(addButton).toHaveTextContent(/^加入 1 张$/)
    await user.click(addButton)

    expect(api.copyToTrain).toHaveBeenCalledWith(1, 2, {
      files: ['a.png'],
      dest_folder: '1_data',
    })
  })

  it('disables Add until a training folder exists', async () => {
    vi.mocked(api.getCuration).mockResolvedValue({
      ...trainView,
      right: {},
      folders: [],
      train_total: 0,
    })
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('grid', { name: '未分配的数据集图片' })

    await user.click(screen.getByRole('button', { name: '选择 a.png' }))
    const disabledAdd = screen.getByRole('button', { name: '加入 1 张到训练分组' })
    expect(disabledAdd).toHaveTextContent(/^加入 1 张$/)
    expect(disabledAdd).toBeDisabled()
    expect(screen.getByText('还没有训练分组，请在上方输入名称并选择「创建分组」')).toBeInTheDocument()
  })

  it('treats validation as a secondary destination and clears incompatible selections', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    const sourceSelection = screen.getByRole('button', { name: '选择 a.png' })
    const trainSelection = screen.getByRole('button', { name: '选择 train.png' })
    await user.click(sourceSelection)
    await user.click(trainSelection)
    expect(sourceSelection).toHaveAttribute('aria-pressed', 'true')
    expect(trainSelection).toHaveAttribute('aria-pressed', 'true')

    await user.selectOptions(screen.getByRole('combobox', { name: '加入目标' }), 'validation')
    expect(screen.getByRole('combobox', { name: '加入目标' })).toHaveValue('validation')
    expect(await screen.findByRole('grid', { name: '留出的验证集图片' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '选择 a.png' })).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByRole('button', { name: '选择 val.png' })).toHaveAttribute('aria-pressed', 'false')

    await user.click(screen.getByRole('button', { name: '选择 a.png' }))
    const addToValidation = screen.getByRole('button', { name: '加入 1 张到验证集' })
    expect(addToValidation).toHaveTextContent(/^加入 1 张$/)
    await user.click(addToValidation)
    expect(api.copyToValidation).toHaveBeenCalledWith(1, 2, { files: ['a.png'] })

    await user.click(screen.getByRole('button', { name: '选择 val.png' }))
    const removeFromValidation = screen.getByRole('button', { name: '从验证集移除 1 张' })
    expect(removeFromValidation).toHaveTextContent(/^移除 1 张$/)
    await user.click(removeFromValidation)
    expect(api.removeFromValidation).toHaveBeenCalledWith(1, 2, {
      items: [{ name: 'val.png', folder: 'manual' }],
    })

    await user.click(screen.getByRole('button', { name: '预览 a.png' }))
    expect(screen.getByText('←/→ 浏览 · Enter/Space 加入验证集')).toBeInTheDocument()
  })

  it('keeps folder creation and named folder actions with the folder selector', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    const folderGroup = screen.getByRole('group', { name: '训练分组' })
    const activeFolder = within(folderGroup).getByRole('button', { name: /^1_data/ })
    expect(activeFolder).toHaveAttribute('aria-pressed', 'true')
    expect(activeFolder.parentElement).not.toHaveClass('border')
    expect(screen.getByRole('button', { name: '重命名训练分组 1_data' })).toHaveClass('btn-ghost')
    expect(screen.getByRole('button', { name: '删除训练分组 1_data' }))
      .toHaveClass('btn-ghost', 'text-err')
    expect(screen.getByRole('button', { name: '删除训练分组 1_data' }))
      .not.toHaveClass('btn-danger')

    const createForm = screen.getByRole('form', { name: '创建训练分组' })
    await user.type(within(createForm).getByRole('textbox', { name: '新训练分组名称' }), '5_concept')
    await user.click(within(createForm).getByRole('button', { name: '创建分组' }))
    expect(api.folderOp).toHaveBeenCalledWith(1, 2, { op: 'create', name: '5_concept' })
  })

  it('keeps loaded grids mounted after refresh failure and retries in place', async () => {
    renderPage()
    await ready()
    vi.mocked(api.getCuration).mockRejectedValueOnce(new Error('offline'))

    act(() => mocks.onEvent?.({
      type: 'version_state_changed',
      project_id: 1,
      version_id: 2,
    }))

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('无法刷新筛选数据')
    expect(screen.getByRole('grid', { name: '未分配的数据集图片' })).toBeInTheDocument()
    expect(screen.getByRole('grid', { name: '当前分组中的训练图片' })).toBeInTheDocument()

    await userEvent.click(within(alert).getByRole('button', { name: '重试' }))
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument())
  })

  it('renders a retryable blocking alert when the first load fails', async () => {
    vi.mocked(api.getCuration).mockRejectedValueOnce(new Error('offline'))
    renderPage()

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('无法加载筛选工作区')
    expect(screen.queryByRole('grid')).not.toBeInTheDocument()

    await userEvent.click(within(alert).getByRole('button', { name: '重试' }))
    await ready()
  })

  it('keeps the browser Alt behavior while retaining the hold-to-preview state', async () => {
    renderPage()
    await ready()
    const event = new KeyboardEvent('keydown', {
      key: 'Alt',
      code: 'AltLeft',
      bubbles: true,
      cancelable: true,
    })

    act(() => window.dispatchEvent(event))
    expect(event.defaultPrevented).toBe(false)
  })
})
