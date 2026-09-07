import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import TagEditor from './TagEditor'

describe('TagEditor (PP4 chip mode)', () => {
  it('renders chips for each tag', () => {
    render(<TagEditor tags={['a', 'b']} onChange={() => {}} />)
    expect(screen.getByText('a')).toBeInTheDocument()
    expect(screen.getByText('b')).toBeInTheDocument()
    expect(screen.getByText('2 个标签')).toBeInTheDocument()
  })

  it('can delegate the tag count to a parent panel header', () => {
    render(<TagEditor tags={['a', 'b']} onChange={() => {}} showTagCount={false} />)
    expect(screen.queryByText('2 个标签')).not.toBeInTheDocument()
  })

  it('updates the parent cache immediately in text mode without a sync action', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TagEditor tags={['a']} onChange={onChange} />)

    await user.click(screen.getByText('文本'))
    fireEvent.change(screen.getByRole('textbox', { name: '以文本编辑标签' }), {
      target: { value: 'a, b,\na' },
    })

    expect(onChange).toHaveBeenLastCalledWith(['a', 'b'])
    expect(screen.queryByRole('button', { name: '同步' })).not.toBeInTheDocument()
  })

  it('keeps free-form text intact while the parent cache echoes parsed tags', async () => {
    const user = userEvent.setup()
    function ControlledEditor() {
      const [tags, setTags] = useState(['a'])
      return <TagEditor tags={tags} onChange={setTags} />
    }
    render(<ControlledEditor />)

    await user.click(screen.getByText('文本'))
    const input = screen.getByRole('textbox', { name: '以文本编辑标签' })
    fireEvent.change(input, { target: { value: 'a, b,\na' } })

    expect(input).toHaveValue('a, b,\na')
  })

  it('resets the text buffer on source or external tag changes without resetting mode', async () => {
    const user = userEvent.setup()
    const { rerender } = render(
      <TagEditor resetKey="image-a" tags={['a']} onChange={() => {}} />,
    )
    await user.click(screen.getByText('文本'))
    const input = screen.getByRole('textbox', { name: '以文本编辑标签' })
    fireEvent.change(input, { target: { value: 'a,\n' } })
    expect(input).toHaveValue('a,\n')

    rerender(<TagEditor resetKey="image-b" tags={['a']} onChange={() => {}} />)
    expect(input).toHaveValue('a')
    expect(screen.getByRole('radio', { name: '文本' })).toBeChecked()

    rerender(<TagEditor resetKey="image-b" tags={['different']} onChange={() => {}} />)
    expect(input).toHaveValue('different')
    expect(screen.getByRole('radio', { name: '文本' })).toBeChecked()
  })

  it('Enter adds a tag at the end (chip 拖拽心智 — 新东西落底部)', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TagEditor tags={['a']} onChange={onChange} />)
    const input = screen.getByPlaceholderText(/添加标签/)
    await user.type(input, 'new{Enter}')
    expect(onChange).toHaveBeenCalledWith(['a', 'new'])
  })

  it('comma also adds a tag', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TagEditor tags={[]} onChange={onChange} />)
    await user.type(screen.getByPlaceholderText(/添加标签/), 'foo,')
    expect(onChange).toHaveBeenCalledWith(['foo'])
  })

  it('clicking × removes a tag', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TagEditor tags={['a', 'b']} onChange={onChange} />)
    await user.click(screen.getByLabelText('删除 a'))
    expect(onChange).toHaveBeenCalledWith(['b'])
  })

  it('refuses duplicates silently', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TagEditor tags={['a']} onChange={onChange} />)
    await user.type(screen.getByPlaceholderText(/添加标签/), 'a{Enter}')
    expect(onChange).not.toHaveBeenCalled()
  })

  it('natural mode renders textarea', () => {
    render(<TagEditor natural tags={['a long sentence']} onChange={() => {}} />)
    expect(
      screen.getByPlaceholderText(/自然语言/)
    ).toBeInTheDocument()
  })
})
