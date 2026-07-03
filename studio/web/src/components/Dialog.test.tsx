import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { DialogProvider, useDialog } from './Dialog'
import { useState } from 'react'

// Test harness:外层组件触发 confirm/prompt/alert,把 Promise resolve 值放到
// 屏幕上,断言 user 交互结果。
function Harness({
  run,
}: {
  run: (api: ReturnType<typeof useDialog>) => Promise<unknown>
}) {
  const api = useDialog()
  const [result, setResult] = useState<string>('')
  return (
    <div>
      <button
        onClick={() =>
          void run(api).then((v) => setResult('resolved:' + JSON.stringify(v)))
        }
      >
        trigger
      </button>
      <div data-testid="result">{result}</div>
    </div>
  )
}

const wrap = (run: (api: ReturnType<typeof useDialog>) => Promise<unknown>) =>
  render(
    <DialogProvider>
      <Harness run={run} />
    </DialogProvider>,
  )

describe('Dialog (useDialog API)', () => {
  it('confirm 点确认返回 true', async () => {
    const user = userEvent.setup()
    wrap((api) => api.confirm('删除？'))
    await user.click(screen.getByText('trigger'))
    await user.click(screen.getByText('确认'))
    expect(screen.getByTestId('result')).toHaveTextContent('resolved:true')
  })

  it('confirm 点取消返回 false', async () => {
    const user = userEvent.setup()
    wrap((api) => api.confirm('删除？'))
    await user.click(screen.getByText('trigger'))
    await user.click(screen.getByText('取消'))
    expect(screen.getByTestId('result')).toHaveTextContent('resolved:false')
  })

  it('confirm Esc 关闭返回 false', async () => {
    const user = userEvent.setup()
    wrap((api) => api.confirm('删除？'))
    await user.click(screen.getByText('trigger'))
    await user.keyboard('{Escape}')
    expect(screen.getByTestId('result')).toHaveTextContent('resolved:false')
  })

  it('confirm 自定义按钮文案', async () => {
    const user = userEvent.setup()
    wrap((api) => api.confirm('删除？', { okText: '焚毁', cancelText: '算了' }))
    await user.click(screen.getByText('trigger'))
    expect(screen.getByText('焚毁')).toBeInTheDocument()
    expect(screen.getByText('算了')).toBeInTheDocument()
  })

  it('confirm 确认键恒为 btn-primary（tone 不换按钮底色）', async () => {
    const user = userEvent.setup()
    wrap((api) => api.confirm('删除？', { tone: 'danger' }))
    await user.click(screen.getByText('trigger'))
    expect(screen.getByText('确认')).toHaveClass('btn-primary')
    expect(screen.getByText('确认')).not.toHaveClass('btn-danger')
  })

  it('prompt 输入后点确定返回字符串', async () => {
    const user = userEvent.setup()
    wrap((api) => api.prompt('名称'))
    await user.click(screen.getByText('trigger'))
    const input = screen.getByRole('textbox')
    // 先 click 聚焦再 type：避免 modal 刚开 / autofocus 未 settle 时 userEvent
    // 丢首字符（CI 慢机偶发 "y-preset"）。
    await user.click(input)
    await user.type(input, 'my-preset')
    await user.click(screen.getByText('确定'))
    expect(screen.getByTestId('result')).toHaveTextContent('resolved:"my-preset"')
  })

  it('prompt 取消返回 null', async () => {
    const user = userEvent.setup()
    wrap((api) => api.prompt('名称'))
    await user.click(screen.getByText('trigger'))
    await user.click(screen.getByText('取消'))
    expect(screen.getByTestId('result')).toHaveTextContent('resolved:null')
  })

  it('prompt defaultValue 预填到输入框', async () => {
    const user = userEvent.setup()
    wrap((api) => api.prompt('名称', { defaultValue: 'v3' }))
    await user.click(screen.getByText('trigger'))
    expect(screen.getByRole('textbox')).toHaveValue('v3')
  })

  it('prompt validate 阻止提交并显示错误', async () => {
    const user = userEvent.setup()
    wrap((api) =>
      api.prompt('名称', { validate: (v) => (v.length < 3 ? '至少 3 字符' : null) }),
    )
    await user.click(screen.getByText('trigger'))
    await user.type(screen.getByRole('textbox'), 'ab')
    await user.click(screen.getByText('确定'))
    expect(screen.getByText('至少 3 字符')).toBeInTheDocument()
    // 错误显示后 dialog 未关闭 — result 仍空
    expect(screen.getByTestId('result')).toHaveTextContent('')
  })

  it('alert 点知道了 resolve', async () => {
    const user = userEvent.setup()
    wrap((api) => api.alert('完成'))
    await user.click(screen.getByText('trigger'))
    expect(screen.getByText('知道了')).toBeInTheDocument()
    await user.click(screen.getByText('知道了'))
    expect(screen.getByTestId('result')).toHaveTextContent('resolved:')
  })

  it('alert 没有取消按钮', async () => {
    const user = userEvent.setup()
    wrap((api) => api.alert('完成'))
    await user.click(screen.getByText('trigger'))
    expect(screen.queryByText('取消')).not.toBeInTheDocument()
  })
})
