import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { api, type Secrets } from '../api/client'
import {
  __resetTagPrefsForTest,
  __setTagPrefsForTest,
  LEGACY_AUTOCOMPLETE_KEY,
  LEGACY_SHOW_KEY,
  loadTagPrefs,
  setTagAutocompleteEnabled,
  useShowTagTranslation,
  useTagAutocompleteEnabled,
} from './prefs'

/** mount 触发的异步加载在 act 里跑完（loadTagPrefs 复用 in-flight Promise）。 */
const flushLoad = () => act(async () => { await loadTagPrefs() })

/** 只给 tag_dictionary 形状；store 只读这一段。 */
function secretsWith(td: { show_translation: boolean | null; autocomplete: boolean | null }): Secrets {
  return { tag_dictionary: td } as unknown as Secrets
}

function mockApi(td: { show_translation: boolean | null; autocomplete: boolean | null }) {
  const getSecrets = vi.spyOn(api, 'getSecrets').mockResolvedValue(secretsWith(td))
  const updateSecrets = vi.spyOn(api, 'updateSecrets').mockImplementation(
    async (patch) => secretsWith({
      show_translation: patch.tag_dictionary?.show_translation ?? td.show_translation,
      autocomplete: patch.tag_dictionary?.autocomplete ?? td.autocomplete,
    }),
  )
  return { getSecrets, updateSecrets }
}

/** 两个独立消费者：Settings 侧（带 setter）+ 任意别处的只读订阅者。 */
function ShowConsumer() {
  const [show, setShow] = useShowTagTranslation()
  return (
    <div>
      <span data-testid="show">{show ? 'on' : 'off'}</span>
      <button type="button" onClick={() => void setShow(!show).catch(() => {})}>toggle-show</button>
    </div>
  )
}

function ShowObserver() {
  const [show] = useShowTagTranslation()
  return <span data-testid="show-observer">{show ? 'on' : 'off'}</span>
}

function AcConsumer() {
  const [ac, setAc] = useTagAutocompleteEnabled()
  return (
    <div>
      <span data-testid="ac">{ac ? 'on' : 'off'}</span>
      <button type="button" onClick={() => void setAc(!ac).catch(() => {})}>toggle-ac</button>
    </div>
  )
}

describe('tagDict/prefs — 后端 secrets.tag_dictionary 镜像 store', () => {
  beforeEach(() => {
    localStorage.clear()
    __resetTagPrefsForTest()
  })

  afterEach(() => {
    // 先卸组件再动 store：否则 reset 会在 act 外触发已挂载订阅者重渲染
    cleanup()
    vi.restoreAllMocks()
    localStorage.clear()
    // 还原 setup.ts 的预热态，别污染同文件后续用例 / 其它组件测试
    __resetTagPrefsForTest()
    __setTagPrefsForTest({ loaded: true })
  })

  it('订阅：Settings 侧切开关，别处的订阅者即时跟着变，且 PUT 单字段 patch', async () => {
    const { updateSecrets } = mockApi({ show_translation: true, autocomplete: true })
    render(<><ShowConsumer /><ShowObserver /></>)
    await flushLoad()
    expect(screen.getByTestId('show')).toHaveTextContent('on')
    expect(screen.getByTestId('show-observer')).toHaveTextContent('on')

    await act(async () => { screen.getByText('toggle-show').click() })
    expect(screen.getByTestId('show')).toHaveTextContent('off')
    expect(screen.getByTestId('show-observer')).toHaveTextContent('off')
    await waitFor(() => expect(updateSecrets).toHaveBeenCalledWith({
      tag_dictionary: { show_translation: false },
    }))
  })

  it('首次加载：后端 null + 旧 localStorage 有值 → 用旧值 seed 一次并清掉旧键', async () => {
    localStorage.setItem(LEGACY_SHOW_KEY, '0')
    localStorage.setItem(LEGACY_AUTOCOMPLETE_KEY, '0')
    __resetTagPrefsForTest() // 让加载前快照也读到旧值
    const { getSecrets, updateSecrets } = mockApi({ show_translation: null, autocomplete: null })
    render(<><ShowConsumer /><AcConsumer /></>)
    // 加载前：本地推导值 = 旧 localStorage（首屏不闪）
    expect(screen.getByTestId('show')).toHaveTextContent('off')
    expect(screen.getByTestId('ac')).toHaveTextContent('off')

    await flushLoad()
    expect(getSecrets).toHaveBeenCalledTimes(1)
    expect(updateSecrets).toHaveBeenCalledTimes(1)
    expect(updateSecrets).toHaveBeenCalledWith({
      tag_dictionary: { show_translation: false, autocomplete: false },
    })
    expect(localStorage.getItem(LEGACY_SHOW_KEY)).toBeNull()
    expect(localStorage.getItem(LEGACY_AUTOCOMPLETE_KEY)).toBeNull()
    expect(screen.getByTestId('show')).toHaveTextContent('off')
    expect(screen.getByTestId('ac')).toHaveTextContent('off')
  })

  it('首次加载：后端 null + 无旧值 → show 按界面语言 seed（en 关 / zh 开），autocomplete 不写', async () => {
    localStorage.setItem('studio.lang', 'en')
    const { updateSecrets } = mockApi({ show_translation: null, autocomplete: null })
    await loadTagPrefs()
    expect(updateSecrets).toHaveBeenCalledTimes(1)
    expect(updateSecrets).toHaveBeenCalledWith({ tag_dictionary: { show_translation: false } })

    vi.restoreAllMocks()
    localStorage.setItem('studio.lang', 'zh-CN')
    __resetTagPrefsForTest()
    const second = mockApi({ show_translation: null, autocomplete: null })
    await loadTagPrefs()
    expect(second.updateSecrets).toHaveBeenCalledWith({ tag_dictionary: { show_translation: true } })
  })

  it('不覆盖手动值：后端已是 bool 时旧 localStorage 值被忽略（不 PUT），只清旧键', async () => {
    localStorage.setItem(LEGACY_SHOW_KEY, '0')
    localStorage.setItem(LEGACY_AUTOCOMPLETE_KEY, '1')
    __resetTagPrefsForTest()
    const { updateSecrets } = mockApi({ show_translation: true, autocomplete: false })
    render(<><ShowConsumer /><AcConsumer /></>)
    await flushLoad()
    expect(screen.getByTestId('show')).toHaveTextContent('on')
    expect(screen.getByTestId('ac')).toHaveTextContent('off')
    expect(updateSecrets).not.toHaveBeenCalled()
    expect(localStorage.getItem(LEGACY_SHOW_KEY)).toBeNull()
    expect(localStorage.getItem(LEGACY_AUTOCOMPLETE_KEY)).toBeNull()
  })

  it('切语言不覆盖已 seed 的值：后端 show=true、界面语言 en → 仍显示开', async () => {
    localStorage.setItem('studio.lang', 'en')
    __resetTagPrefsForTest()
    const { updateSecrets } = mockApi({ show_translation: true, autocomplete: null })
    render(<ShowConsumer />)
    // 加载前按语言推导是关；后端 true 到手后翻成开，且不写回
    expect(screen.getByTestId('show')).toHaveTextContent('off')
    await flushLoad()
    expect(screen.getByTestId('show')).toHaveTextContent('on')
    expect(updateSecrets).not.toHaveBeenCalled()
  })

  it('加载是幂等的：多个消费者同时 mount 只拉一次 secrets', async () => {
    const { getSecrets } = mockApi({ show_translation: true, autocomplete: true })
    render(<><ShowConsumer /><ShowObserver /><AcConsumer /></>)
    await flushLoad()
    await loadTagPrefs()
    expect(getSecrets).toHaveBeenCalledTimes(1)
  })

  it('PUT 失败：乐观值回滚并向 caller 抛错', async () => {
    vi.spyOn(api, 'getSecrets').mockResolvedValue(secretsWith({ show_translation: false, autocomplete: true }))
    vi.spyOn(api, 'updateSecrets').mockRejectedValue(new Error('boom'))
    render(<AcConsumer />)
    await flushLoad()
    expect(screen.getByTestId('ac')).toHaveTextContent('on')
    await act(async () => {
      await expect(setTagAutocompleteEnabled(false)).rejects.toThrow('boom')
    })
    expect(screen.getByTestId('ac')).toHaveTextContent('on')
  })

  it('seed PUT 失败：保留旧 localStorage 键等下次重试，本地先用推导值', async () => {
    localStorage.setItem(LEGACY_SHOW_KEY, '0')
    __resetTagPrefsForTest()
    vi.spyOn(api, 'getSecrets').mockResolvedValue(secretsWith({ show_translation: null, autocomplete: null }))
    vi.spyOn(api, 'updateSecrets').mockRejectedValue(new Error('boom'))
    await loadTagPrefs()
    expect(localStorage.getItem(LEGACY_SHOW_KEY)).toBe('0')
    render(<ShowConsumer />)
    expect(screen.getByTestId('show')).toHaveTextContent('off')
  })
})
