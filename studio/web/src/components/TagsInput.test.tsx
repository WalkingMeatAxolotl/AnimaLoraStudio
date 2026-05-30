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

/** 静止态是 role=button 的 chip 容器；点它进入编辑态后才有 textbox。 */
async function enterEdit(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button'))
  return screen.getByRole('textbox')
}

describe('TagsInput', () => {
  it('编辑态能打逗号分隔多个 tag —— 逗号不被当场吃掉', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const input = await enterEdit(user)
    await user.type(input, 'monochrome, greyscale')
    expect(input).toHaveValue('monochrome, greyscale')
    expect(screen.getByTestId('tags')).toHaveTextContent('["monochrome","greyscale"]')
  })

  it('编辑态能打含空格的多词 tag —— 空格不被 trim 掉', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const input = await enterEdit(user)
    await user.type(input, 'blue eyes')
    expect(input).toHaveValue('blue eyes')
    expect(screen.getByTestId('tags')).toHaveTextContent('["blue eyes"]')
  })

  it('blur 后退出编辑态、以 chip 展示，文本归整', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const input = await enterEdit(user)
    await user.type(input, 'x,,y , ')
    expect(input).toHaveValue('x,,y , ')   // 编辑态保留原始文本
    await user.tab()                       // blur
    expect(screen.queryByRole('textbox')).toBeNull()   // 回到 chip 静止态
    expect(screen.getByText('x')).toBeInTheDocument()
    expect(screen.getByText('y')).toBeInTheDocument()
    expect(screen.getByTestId('tags')).toHaveTextContent('["x","y"]')
    // 再次进编辑态看到的是规范形式
    const input2 = await enterEdit(user)
    expect(input2).toHaveValue('x, y')
  })

  it('blur 把下划线归一成空格（chip 统一空格形式）', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const input = await enterEdit(user)
    await user.type(input, 'cat_girl, blue_eyes')
    expect(input).toHaveValue('cat_girl, blue_eyes')   // 编辑态原样
    await user.tab()                                    // blur 归一
    expect(screen.getByText('cat girl')).toBeInTheDocument()
    expect(screen.getByText('blue eyes')).toBeInTheDocument()
    expect(screen.getByTestId('tags')).toHaveTextContent('["cat girl","blue eyes"]')
  })

  it('静止态把每个 tag 渲染成独立 chip', () => {
    render(<Harness initial={['monochrome', 'blue eyes']} />)
    expect(screen.queryByRole('textbox')).toBeNull()
    expect(screen.getByText('monochrome')).toBeInTheDocument()
    expect(screen.getByText('blue eyes')).toBeInTheDocument()
  })

  it('外部 value 变化（restore 默认）时同步 chip 展示', async () => {
    function ResetHarness() {
      const [tags, setTags] = useState<string[]>(['a', 'b'])
      return (
        <>
          <TagsInput label="bl" value={tags} disabled={false} onChange={setTags} />
          <button onClick={() => setTags(['reset'])}>do-reset</button>
        </>
      )
    }
    const user = userEvent.setup()
    render(<ResetHarness />)
    expect(screen.getByText('a')).toBeInTheDocument()
    expect(screen.getByText('b')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'do-reset' }))
    expect(screen.getByText('reset')).toBeInTheDocument()
    expect(screen.queryByText('a')).toBeNull()
  })

  it('parseTags 去首尾空格并丢空段', () => {
    expect(parseTags('  a , , b ,')).toEqual(['a', 'b'])
    expect(parseTags('')).toEqual([])
    expect(parseTags('blue eyes, red_hair')).toEqual(['blue eyes', 'red_hair'])
  })
})
