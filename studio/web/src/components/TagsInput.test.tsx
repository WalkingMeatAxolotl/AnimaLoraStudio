import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import TagsInput, { parseTags } from './TagsInput'

/** 受控 harness：模拟父组件持有 string[] 状态，复刻真实受控回路。 */
function Harness({ initial = [] as string[] }: { initial?: string[] }) {
  const [tags, setTags] = useState<string[]>(initial)
  return (
    <>
      <TagsInput label="bl" value={tags} disabled={false} onChange={setTags} />
      <output data-testid="tags">{JSON.stringify(tags)}</output>
    </>
  )
}

describe('TagsInput', () => {
  it('能打逗号分隔多个 tag —— 逗号不被当场吃掉', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const input = screen.getByRole('textbox')
    await user.type(input, 'monochrome, greyscale')
    expect(input).toHaveValue('monochrome, greyscale')
    expect(screen.getByTestId('tags')).toHaveTextContent('["monochrome","greyscale"]')
  })

  it('能打含空格的多词 tag —— 空格不被 trim 掉', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const input = screen.getByRole('textbox')
    await user.type(input, 'blue eyes')
    expect(input).toHaveValue('blue eyes')
    expect(screen.getByTestId('tags')).toHaveTextContent('["blue eyes"]')
  })

  it('blur 时把文本收成规范形式（去空段 / 统一逗号后单空格）', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const input = screen.getByRole('textbox')
    await user.type(input, 'x,,y , ')
    expect(input).toHaveValue('x,,y , ')   // focus 中保留原始文本
    await user.tab()                       // 触发 blur
    expect(input).toHaveValue('x, y')      // blur 后归整
    expect(screen.getByTestId('tags')).toHaveTextContent('["x","y"]')
  })

  it('外部 value 变化（restore 默认）时同步显示文本', async () => {
    function ResetHarness() {
      const [tags, setTags] = useState<string[]>(['a', 'b'])
      return (
        <>
          <TagsInput label="bl" value={tags} disabled={false} onChange={setTags} />
          <button onClick={() => setTags(['reset'])}>reset</button>
        </>
      )
    }
    const user = userEvent.setup()
    render(<ResetHarness />)
    const input = screen.getByRole('textbox')
    expect(input).toHaveValue('a, b')
    await user.click(screen.getByText('reset'))
    expect(input).toHaveValue('reset')
  })

  it('parseTags 去首尾空格并丢空段', () => {
    expect(parseTags('  a , , b ,')).toEqual(['a', 'b'])
    expect(parseTags('')).toEqual([])
    expect(parseTags('blue eyes, red_hair')).toEqual(['blue eyes', 'red_hair'])
  })
})
