/**
 * ModelSourceCard —— 统一候选卡的动作矩阵（docs/design/model-source-unification.md D2）：
 * 内置 preset 不可移除；download 候选 = 下载/删除 + × 移除；local 候选只有
 * × 移除（无下载按钮、缺失显示「文件缺失」）；添加入口走 /api/model-sources。
 */
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '../../../i18n'
import { DialogProvider } from '../../../components/Dialog'
import { ToastProvider } from '../../../components/Toast'
import { SettingsDataProvider } from '../../../lib/SettingsData'
import type { ModelSourceRow, ModelsCatalog } from '../../../api/client'
import { ModelSourceCard } from './modelCards'

const t = i18n.t.bind(i18n)

function row(partial: Partial<ModelSourceRow>): ModelSourceRow {
  return {
    kind: 'preset',
    value: 'SmilingWolf/wd-eva02-large-tagger-v3',
    label: 'SmilingWolf/wd-eva02-large-tagger-v3',
    download_id: 'wd14',
    status_key: 'wd14:SmilingWolf/wd-eva02-large-tagger-v3',
    exists: false,
    size: 0,
    files: null,
    size_estimate: 0,
    is_current: false,
    removable: false,
    deletable: true,
    extra: {},
    ...partial,
  }
}

const PRESET = row({})
const DOWNLOAD = row({
  kind: 'download',
  value: 'Custom/tagger-x',
  label: 'Custom/tagger-x',
  status_key: 'wd14:Custom/tagger-x',
  removable: true,
})
const LOCAL = row({
  kind: 'local',
  value: 'D:/models/wd14-local',
  label: 'D:/models/wd14-local',
  download_id: null,
  status_key: null,
  removable: true,
  deletable: false,
  exists: false,
})

const catalog = {
  models_root: 'D:/models',
  model_sources: { wd14: [PRESET, DOWNLOAD, LOCAL] },
  downloads: {},
} as unknown as ModelsCatalog

const fetchMock = vi.fn()

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset()
  fetchMock.mockImplementation((url: string) => {
    if (typeof url === 'string' && url.includes('/api/models/catalog')) {
      return Promise.resolve(new Response(JSON.stringify(catalog), { status: 200 }))
    }
    if (typeof url === 'string' && url.includes('/api/model-sources/')) {
      return Promise.resolve(new Response(JSON.stringify(catalog), { status: 200 }))
    }
    return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }))
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

function renderCard(overrides?: {
  currentValue?: string
  onSelect?: (v: string, r: ModelSourceRow) => void
}) {
  return render(
    <ToastProvider>
      <DialogProvider>
        <SettingsDataProvider>
          <ModelSourceCard
            domain="wd14"
            title="WD14"
            catalog={catalog}
            currentValue={overrides?.currentValue ?? PRESET.value}
            onSelect={overrides?.onSelect ?? (() => {})}
            addDownload={{}}
            addLocal={{ dirOnly: true }}
            t={t}
          />
        </SettingsDataProvider>
      </DialogProvider>
    </ToastProvider>
  )
}

describe('ModelSourceCard action matrix', () => {
  it('preset rows have no remove button; user rows do', () => {
    renderCard()
    const items = screen.getAllByRole('listitem')
    expect(items).toHaveLength(3)
    const removeTitle = i18n.t('settings.removeCandidate')
    expect(within(items[0]).queryByTitle(removeTitle)).toBeNull()
    expect(within(items[1]).getByTitle(removeTitle)).toBeTruthy()
    expect(within(items[2]).getByTitle(removeTitle)).toBeTruthy()
  })

  it('local row has no download button and shows missing-file hint', () => {
    renderCard()
    const items = screen.getAllByRole('listitem')
    // preset / download 行有下载按钮，local 行没有
    expect(within(items[0]).getByTitle(i18n.t('common.download'))).toBeTruthy()
    expect(within(items[2]).queryByTitle(i18n.t('common.download'))).toBeNull()
    expect(
      within(items[2]).getByText(i18n.t('settings.localModelMissing'))
    ).toBeTruthy()
    // 缺失的 local 候选不可选中
    expect(within(items[2]).getByRole('radio')).toBeDisabled()
  })

  it('removing a candidate calls DELETE /api/model-sources/wd14', async () => {
    const user = userEvent.setup()
    renderCard()
    const items = screen.getAllByRole('listitem')
    await user.click(
      within(items[1]).getByTitle(i18n.t('settings.removeCandidate'))
    )
    await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([url, init]) =>
          String(url).includes('/api/model-sources/wd14')
          && init?.method === 'DELETE'
      )
      expect(call).toBeTruthy()
      const body = JSON.parse(String((call![1] as RequestInit).body))
      expect(body).toMatchObject({ kind: 'download', repo: 'Custom/tagger-x' })
    })
  })

  it('removing the currently selected candidate falls back to first preset', async () => {
    const user = userEvent.setup()
    const onSelect = vi.fn()
    renderCard({ currentValue: DOWNLOAD.value, onSelect })
    const items = screen.getAllByRole('listitem')
    await user.click(
      within(items[1]).getByTitle(i18n.t('settings.removeCandidate'))
    )
    await waitFor(() => {
      expect(onSelect).toHaveBeenCalledWith(PRESET.value, expect.anything())
    })
  })

  it('add-download form POSTs a download candidate', async () => {
    const user = userEvent.setup()
    renderCard()
    await user.click(
      screen.getByText(`+ ${i18n.t('settings.addDownloadCandidate')}`)
    )
    await user.type(
      screen.getByPlaceholderText(i18n.t('settings.addHfModelId')),
      'New/candidate{Enter}'
    )
    await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([url, init]) =>
          String(url).includes('/api/model-sources/wd14')
          && init?.method === 'POST'
      )
      expect(call).toBeTruthy()
      const body = JSON.parse(String((call![1] as RequestInit).body))
      expect(body).toMatchObject({ kind: 'download', repo: 'New/candidate' })
    })
  })
})
