import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Outlet, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  api,
  type Job,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import i18n from '../../../i18n'
import TaggingPage, { availabilityOverrides } from './Tagging'

const mocks = vi.hoisted(() => ({
  toast: vi.fn(),
  reload: vi.fn(async () => undefined),
  openSettings: vi.fn(),
  setDownloadSource: vi.fn(async () => undefined),
}))

vi.mock('../../../components/Toast', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Toast')>()
  return { ...actual, useToast: () => ({ toast: mocks.toast }) }
})
vi.mock('../../../lib/SettingsDrawer', () => ({
  useSettingsDrawer: () => ({ open: mocks.openSettings }),
}))
vi.mock('../../../lib/SettingsData', () => {
  const secrets = {
    wd14: {
      threshold_general: 0.35,
      threshold_character: 0.85,
      model_id: 'SmilingWolf/wd-swinv2-tagger-v3',
      blacklist_tags: [],
    },
    cltagger: {
      threshold_general: 0.35,
      threshold_character: 0.85,
      model_id: 'cella110n/cl_tagger',
      model_path: 'model.onnx',
      tag_mapping_path: 'tags.json',
      add_copyright_tag: true,
      add_artist_tag: true,
      add_meta_tag: false,
      add_model_tag: false,
      add_rating_tag: false,
      add_quality_tag: false,
      blacklist_tags: [],
    },
    llm_tagger: {
      current_preset: 'builtin-default',
      presets: [{
        id: 'builtin-default',
        label: 'Default',
        builtin: true,
        model: 'vision-model',
        output_format: 'text',
        temperature: 0.2,
        concurrency: 1,
        assist_tagger: null,
        messages: [],
      }],
    },
  }
  return {
    useSettingsData: () => ({
      catalog: null,
      setDownloadSource: mocks.setDownloadSource,
      secrets,
    }),
  }
})
vi.mock('../../../lib/useEventStream', () => ({ useEventStream: () => undefined }))
vi.mock('../../../components/LLMPresetEditorModal', () => ({
  default: () => <div role="dialog">Preset editor</div>,
  llmPresetLabel: (preset: { label: string }) => preset.label,
}))
vi.mock('../../tools/settings/modelCards', () => ({
  SourceSelect: () => (
    <select aria-label="Download source" defaultValue="huggingface">
      <option value="huggingface">Hugging Face</option>
    </select>
  ),
  ModelSourceCard: () => <button type="button">Download model</button>,
}))

const completedJob: Job = {
  id: 31,
  project_id: 7,
  version_id: 11,
  kind: 'tag',
  params: JSON.stringify({ tagger: 'wd14', on_existing: 'skip', scope: 'all' }),
  params_decoded: { tagger: 'wd14', on_existing: 'skip', scope: 'all' },
  status: 'done',
  started_at: 100,
  finished_at: 200,
  pid: null,
  log_path: null,
  error_msg: null,
}

const runningJob: Job = { ...completedJob, id: 32, status: 'running', finished_at: null }

function makeVersion(overrides: Record<string, unknown> = {}): Version {
  return {
    id: 11,
    project_id: 7,
    label: 'v1',
    trigger_word: '',
    stats: {
      train_image_count: 10,
      tagged_image_count: 4,
      validation_image_count: 2,
      validation_tagged_count: 1,
      train_folders: [{ name: '10_character', image_count: 6 }],
    },
    ...overrides,
  } as unknown as Version
}

function renderPage(version = makeVersion()) {
  const project = { id: 7, name: 'Project' } as unknown as ProjectDetail
  return render(
    <MemoryRouter
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      initialEntries={['/tagging']}
    >
      <Routes>
        <Route element={<Outlet context={{ project, activeVersion: version, reload: mocks.reload }} />}>
          <Route path="/tagging" element={<TaggingPage />} />
        </Route>
      </Routes>
    </MemoryRouter>,
  )
}

async function ready() {
  return screen.findByRole('button', { name: '开始打标' })
}

beforeEach(async () => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
  await i18n.changeLanguage('zh')
  vi.spyOn(api, 'checkTagger').mockResolvedValue({
    name: 'wd14', ok: true, msg: 'ready', requires_service: false,
  })
  vi.spyOn(api, 'getLatestVersionJob').mockResolvedValue({ job: completedJob, log: '' })
  vi.spyOn(api, 'getCuration').mockResolvedValue({ folders: ['10_character'] } as never)
  vi.spyOn(api, 'startTag').mockResolvedValue(runningJob)
  vi.spyOn(api, 'cancelJob').mockResolvedValue({ task_id: runningJob.id, canceled: true })
})

describe('Tagging workspace contracts', () => {
  it('uses a stacked compact layout, named controls, and an accessible Advanced disclosure', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    const workspace = document.querySelector('[data-tagging-workspace]')
    expect(workspace).toHaveClass(
      'grid-cols-1',
      'overflow-y-auto',
      'xl:grid-cols-[minmax(0,3fr)_minmax(20rem,2fr)]',
      'xl:overflow-hidden',
    )
    expect(screen.getByRole('combobox', { name: '打标器' })).toHaveValue('wd14')
    expect(screen.getByRole('combobox', { name: '范围' })).toHaveValue('all')
    expect(screen.getByRole('combobox', { name: '已有 caption' })).toHaveValue('skip')
    expect(screen.getByRole('textbox', { name: '触发词' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '本次打标计划' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '当前打标状态' })).toBeInTheDocument()

    const advanced = screen.getByRole('button', { name: '高级参数' })
    const controlledId = advanced.getAttribute('aria-controls')
    expect(advanced).toHaveAttribute('aria-expanded', 'false')
    expect(controlledId).toBeTruthy()
    expect(document.getElementById(controlledId!)).toHaveAttribute('hidden')
    await user.click(advanced)
    expect(advanced).toHaveAttribute('aria-expanded', 'true')
    expect(document.getElementById(controlledId!)).not.toHaveAttribute('hidden')
    const blacklist = screen.getByRole('button', { name: '屏蔽 Tags' })
    await user.click(blacklist)
    expect(screen.getByRole('textbox', { name: '屏蔽 Tags' })).toBeInTheDocument()
    expect(screen.getByRole('spinbutton', { name: '通用阈值' })).toBeInTheDocument()
  })

  it('starts with skip by default and preserves the startTag payload', async () => {
    const user = userEvent.setup()
    renderPage()
    const start = await ready()

    expect(screen.getByText('预计处理').parentElement).toHaveTextContent('7 张')
    await user.click(start)

    await waitFor(() => expect(api.startTag).toHaveBeenCalledWith(7, 11, {
      tagger: 'wd14',
      on_existing: 'skip',
      scope: 'all',
      wd14_overrides: undefined,
      cltagger_overrides: undefined,
      llm_overrides: undefined,
      trigger_word: '',
    }))
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
  })

  it('requires confirmation before overwriting known captions', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.selectOptions(screen.getByRole('combobox', { name: '已有 caption' }), 'overwrite')
    await user.click(screen.getByRole('button', { name: '开始打标' }))

    const dialog = screen.getByRole('alertdialog', { name: '确认覆盖已有 Caption' })
    expect(dialog).toHaveTextContent('已有 5 张图片带 caption')
    expect(api.startTag).not.toHaveBeenCalled()
    await user.click(screen.getByRole('button', { name: '取消' }))
    expect(api.startTag).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: '开始打标' }))
    await user.click(screen.getByRole('button', { name: '确认覆盖并开始' }))
    await waitFor(() => expect(api.startTag).toHaveBeenCalledWith(
      7, 11, expect.objectContaining({ on_existing: 'overwrite', scope: 'all' }),
    ))
  })

  it('allows an exact zero-work run and shows unknown work for folder plus skip', async () => {
    const user = userEvent.setup()
    renderPage(makeVersion({
      stats: {
        train_image_count: 10,
        tagged_image_count: 10,
        validation_image_count: 2,
        validation_tagged_count: 2,
        train_folders: [{ name: '10_character', image_count: 6 }],
      },
    }))
    const start = await ready()

    expect(screen.getByText('预计处理').parentElement).toHaveTextContent('0 张')
    expect(start).toBeEnabled()
    await user.selectOptions(screen.getByRole('combobox', { name: '范围' }), '10_character')
    expect(screen.getByText('预计处理').parentElement).toHaveTextContent('启动后扫描')
  })

  it('separates the current task from an editable next-run draft and keeps cancel wired', async () => {
    const user = userEvent.setup()
    vi.mocked(api.getLatestVersionJob).mockResolvedValue({ job: runningJob, log: 'running' })
    renderPage()

    const start = await screen.findByRole('button', { name: '打标中…' })
    expect(start).toHaveAttribute('aria-busy', 'true')
    expect(start).toBeDisabled()
    expect(screen.getByText('当前任务：WD14 #32')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '下一轮设置' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '下一轮打标计划' })).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '打标器' })).toBeEnabled()
    expect(screen.getByRole('combobox', { name: '范围' })).toBeEnabled()
    expect(screen.getByRole('combobox', { name: '已有 caption' })).toBeEnabled()
    expect(screen.getByRole('textbox', { name: '触发词' })).toBeEnabled()

    await user.click(screen.getByRole('button', { name: '高级参数' }))
    expect(screen.getByRole('combobox', { name: 'Download source' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Download model' })).toBeEnabled()
    await user.selectOptions(screen.getByRole('combobox', { name: '范围' }), 'validation')
    expect(screen.getByRole('combobox', { name: '范围' })).toHaveValue('validation')
    expect(screen.getAllByText('验证集')).not.toHaveLength(0)
    expect(api.getLatestVersionJob).toHaveBeenCalledWith(7, 11, 'tag')

    await user.click(screen.getByRole('button', { name: '取消' }))
    await waitFor(() => expect(api.cancelJob).toHaveBeenCalledWith(32))
  })

  it('shows a visible warning when train-folder options cannot load', async () => {
    vi.mocked(api.getCuration).mockRejectedValue(new Error('folders unavailable'))
    renderPage()
    await ready()

    expect(screen.getByText('训练分组加载失败')).toBeInTheDocument()
    expect(screen.getByText('Error: folders unavailable')).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '范围' })).toBeInTheDocument()
  })

  it('marks the header action busy while tagger availability is being checked', async () => {
    vi.mocked(api.checkTagger).mockImplementation(() => new Promise(() => undefined))
    renderPage()

    const checking = await screen.findByRole('button', { name: '检查中…' })
    expect(checking).toHaveAttribute('aria-busy', 'true')
    expect(checking).toBeDisabled()
  })

  it('confirms an unknown folder overwrite instead of assuming it is safe', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.selectOptions(screen.getByRole('combobox', { name: '范围' }), '10_character')
    await user.selectOptions(screen.getByRole('combobox', { name: '已有 caption' }), 'overwrite')
    await user.click(screen.getByRole('button', { name: '开始打标' }))

    expect(screen.getByRole('alertdialog')).toHaveTextContent('可能包含已有 caption')
    expect(api.startTag).not.toHaveBeenCalled()
  })

  it('localizes tagger options without changing their values', async () => {
    await i18n.changeLanguage('en')
    renderPage()
    await screen.findByRole('button', { name: 'Start tagging' })

    const tagger = screen.getByRole('combobox', { name: 'Tagger' })
    expect(screen.getByRole('option', { name: 'WD14 (local ONNX)' })).toHaveValue('wd14')
    expect(screen.getByRole('option', { name: 'CLTagger (local ONNX)' })).toHaveValue('cltagger')
    expect(screen.getByRole('option', { name: 'LLM (OpenAI-compatible, including JoyCaption presets)' })).toHaveValue('llm')
    expect(tagger).toHaveValue('wd14')
  })
})

// issue #477：可用性检查必须带上页面的模型版本覆盖，helper 负责挑出
// 「与全局默认不同、且影响可用性」的字段。
describe('availabilityOverrides', () => {
  const defaults = {
    model_id: 'cella110n/cl_tagger',
    model_path: 'cl_tagger_1_02/model.onnx',
    tag_mapping_path: 'cl_tagger_1_02/tag_mapping.json',
    threshold_general: 0.35,
  }

  it('form / defaults 未加载时不产生覆盖', () => {
    expect(availabilityOverrides(null, defaults, ['model_id'])).toBeUndefined()
    expect(availabilityOverrides(defaults, null, ['model_id'])).toBeUndefined()
  })

  it('与默认一致时不产生覆盖', () => {
    expect(availabilityOverrides({ ...defaults }, defaults, ['model_id', 'model_path'])).toBeUndefined()
  })

  it('只挑列出的且不同的字段（改版本 → 三元组里变了的进覆盖，阈值不进）', () => {
    const form = {
      ...defaults,
      model_id: 'cella110n/cl_tagger_v2',
      model_path: 'v2_01a/model.onnx',
      tag_mapping_path: 'v2_01a/model_vocabulary.json',
      threshold_general: 0.5,
    }
    expect(
      availabilityOverrides(form, defaults, ['model_id', 'model_path', 'tag_mapping_path']),
    ).toEqual({
      model_id: 'cella110n/cl_tagger_v2',
      model_path: 'v2_01a/model.onnx',
      tag_mapping_path: 'v2_01a/model_vocabulary.json',
    })
  })
})
