import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useRef, useState } from 'react'
import { beforeEach, describe, expect, it } from 'vitest'

import { __setStateForTest } from '../../tagDict/store'
import { TagSuggestList } from './TagSuggestList'
import { useTagSuggest } from './useTagSuggest'

/** 注入一个小词典（绕过网络），让 findSuggestions 有东西可匹配。 */
function seedDict() {
  const entries = new Map<string, string[]>([
    ['solo', ['单人']],
    ['long hair', ['长发']],
  ])
  const tagKeys = Array.from(entries.keys())
  __setStateForTest({
    status: 'ready',
    entries,
    tagKeys,
    compactedKeys: tagKeys.map((t) => t.replace(/[\s_]/g, '')),
    reverse: [],
    meta: null,
    error: null,
  })
}

/** 复刻真实 caller 的接线（PromptList / TagsInput 同构）。 */
function Harness({ initial = '' }: { initial?: string }) {
  const [value, setValue] = useState(initial)
  const inputRef = useRef<HTMLInputElement>(null)
  const suggest = useTagSuggest({
    value,
    inputRef,
    onPick: ({ suggestion }) => setValue(suggestion.tag),
  })
  return (
    <div>
      <input
        ref={inputRef}
        value={value}
        onChange={(e) => { setValue(e.target.value); suggest.notifyChange() }}
        onKeyDown={(e) => { suggest.handleKeyDown(e) }}
        onKeyUp={() => suggest.notifySelect()}
        onClick={() => suggest.notifyClick()}
        onFocus={() => suggest.notifyFocus()}
        onBlur={() => suggest.notifyBlur()}
      />
      <TagSuggestList
        open={suggest.open}
        suggestions={suggest.suggestions}
        activeIdx={suggest.activeIdx}
        onPick={(s) => suggest.pickAt(suggest.suggestions.indexOf(s))}
        onHover={suggest.setActiveIdx}
        inputRef={inputRef}
        cursor={suggest.cursor}
        positionDeps={[value]}
      />
    </div>
  )
}

describe('useTagSuggest 弹出规则', () => {
  beforeEach(() => {
    localStorage.clear()
    seedDict()
  })

  it('鼠标点进已有内容的输入框不弹候选（只有输入变化才弹）', async () => {
    const user = userEvent.setup()
    render(<Harness initial="sol" />)
    await user.click(screen.getByRole('textbox'))
    expect(screen.queryByRole('listbox')).toBeNull()
  })

  it('输入变化后弹出候选', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    await user.type(screen.getByRole('textbox'), 'sol')
    expect(await screen.findByRole('listbox')).toBeInTheDocument()
    expect(screen.getByText('solo')).toBeInTheDocument()
  })

  it('候选弹出后鼠标点击输入框（挪光标）→ 候选消失', async () => {
    const user = userEvent.setup()
    const input = () => screen.getByRole('textbox')
    render(<Harness />)
    await user.type(input(), 'sol')
    expect(await screen.findByRole('listbox')).toBeInTheDocument()
    await user.click(input())
    expect(screen.queryByRole('listbox')).toBeNull()
  })

  it('全局开关关闭后，输入也不弹候选', async () => {
    localStorage.setItem('studio.tag.autocomplete', '0')
    const user = userEvent.setup()
    render(<Harness />)
    await user.type(screen.getByRole('textbox'), 'sol')
    expect(screen.queryByRole('listbox')).toBeNull()
  })

  it('鼠标点选候选仍然生效（onMouseDown pick 不被 click-close 挡掉）', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    await user.type(screen.getByRole('textbox'), 'sol')
    const option = await screen.findByText('solo')
    await user.click(option)
    expect(screen.getByRole('textbox')).toHaveValue('solo')
    expect(screen.queryByRole('listbox')).toBeNull()
  })
})
