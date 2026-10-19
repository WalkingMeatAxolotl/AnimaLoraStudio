import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import LLMMessagesEditor from './LLMMessagesEditor'
import type { LLMMessage } from '../api/client'

// 受控 textarea：父组件持 state，把每次编辑写回，模拟真实 Settings 用法。
function Harness() {
  const [messages, setMessages] = useState<LLMMessage[]>([
    { type: 'text', role: 'system', content: '' },
    { type: 'text', role: 'user', content: '' },
  ])
  return <LLMMessagesEditor messages={messages} onChange={setMessages} />
}

// 模拟 LLM 预设编辑器：上抛后落盘 → reloadSecrets 整组重载，消息全是新对象。
function ReloadHarness() {
  const [messages, setMessages] = useState<LLMMessage[]>([
    { type: 'text', role: 'system', content: 'sys' },
    { type: 'text', role: 'user', content: 'first' },
    { type: 'text', role: 'user', content: 'second' },
  ])
  return (
    <LLMMessagesEditor
      messages={messages}
      onChange={(next) => setMessages(next.map((m) => ({ ...m })))}
    />
  )
}

describe('LLMMessagesEditor', () => {
  // regression：不可变更新换掉 message 对象引用后，若没把稳定 id 迁到新对象，
  // idOf 会发新 id → key 变 → SortableMessage（含 textarea）重挂 → 输入一个字
  // 就失焦、无法连续输入。
  it('编辑消息内容时 textarea 不重挂、保持焦点、可连续输入', async () => {
    const user = userEvent.setup()
    render(<Harness />)

    // 两条消息 → 两个 textarea；第二个是 user 消息体。
    const before = screen.getAllByRole('textbox')[1] as HTMLTextAreaElement
    before.focus()
    expect(before).toHaveFocus()

    await user.type(before, 'hello')

    const after = screen.getAllByRole('textbox')[1] as HTMLTextAreaElement
    expect(after).toBe(before)         // 同一 DOM 节点 —— 没有重挂
    expect(after).toHaveFocus()        // 焦点保留
    expect(after).toHaveValue('hello') // 连续多字符全部落入（不止首字）
  })

  // regression：落盘后整组重载换掉全部对象引用 → 所有 key 变 → textarea 全部重挂，
  // 撑高前内容瞬间变矮，外层滚动容器被夹回顶部（改完一条提示词就跳到顶）。
  it('提交编辑后整组重载，textarea 不重挂', async () => {
    const user = userEvent.setup()
    render(<ReloadHarness />)

    const before = screen.getAllByRole('textbox')
    await user.click(before[2])
    await user.type(before[2], ' edited')
    await user.tab()

    const after = screen.getAllByRole('textbox')
    expect(after).toHaveLength(3)
    after.forEach((el, i) => expect(el).toBe(before[i]))
    expect(after[2]).toHaveValue('second edited')
  })

  it('删除消息后整组重载，剩余消息按原身份保留', async () => {
    const user = userEvent.setup()
    render(<ReloadHarness />)

    const before = screen.getAllByRole('textbox')
    await user.click(screen.getAllByRole('button', { name: '删除消息' })[1])

    const after = screen.getAllByRole('textbox')
    expect(after).toHaveLength(2)
    expect(after[0]).toBe(before[0])
    expect(after[1]).toBe(before[2])
    expect(after[1]).toHaveValue('second')
  })
})
