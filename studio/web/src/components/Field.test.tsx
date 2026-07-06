import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { SchemaProperty } from '../api/client'
import Field from './Field'

const floatProp: SchemaProperty = {
  type: 'number',
  default: 0.5,
  group: 'misc',
  control: 'auto',
  description: '',
}

const intProp: SchemaProperty = {
  type: 'integer',
  default: 1,
  group: 'misc',
  control: 'auto',
  description: '',
}

const codeProp: SchemaProperty = {
  anyOf: [
    { type: 'object' },
    { type: 'null' },
  ],
  default: null,
  group: 'misc',
  control: 'code',
  description: '',
}

describe('Field number input (PP10.3)', () => {
  it('does not commit on each keystroke — typing 0.05 stays intact', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Field name="lr" prop={floatProp} value={0.5} onChange={onChange} />)

    const input = screen.getByRole('textbox') as HTMLInputElement
    await user.clear(input)
    await user.type(input, '0.05')
    // 输入过程中父 onChange 不应被调用
    expect(onChange).not.toHaveBeenCalled()
    // raw 缓冲保留完整输入
    expect(input.value).toBe('0.05')
  })

  it('commits parsed number on blur', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Field name="lr" prop={floatProp} value={0.5} onChange={onChange} />)

    const input = screen.getByRole('textbox') as HTMLInputElement
    await user.clear(input)
    await user.type(input, '0.05')
    await user.tab() // 触发 blur
    expect(onChange).toHaveBeenCalledWith(0.05)
  })

  it('commits parsed number on Enter', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Field name="lr" prop={floatProp} value={0.5} onChange={onChange} />)

    const input = screen.getByRole('textbox') as HTMLInputElement
    await user.clear(input)
    await user.type(input, '0.05{Enter}')
    expect(onChange).toHaveBeenCalledWith(0.05)
  })

  it('blur with empty input writes back default', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Field name="lr" prop={floatProp} value={0.5} onChange={onChange} />)

    const input = screen.getByRole('textbox') as HTMLInputElement
    await user.clear(input)
    fireEvent.blur(input)
    expect(onChange).toHaveBeenCalledWith(0.5) // floatProp.default
    expect(input.value).toBe('0.5')
  })

  it('blur with invalid input rolls back to current value', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Field name="lr" prop={floatProp} value={0.5} onChange={onChange} />)

    const input = screen.getByRole('textbox') as HTMLInputElement
    await user.clear(input)
    await user.type(input, 'abc')
    fireEvent.blur(input)
    expect(onChange).not.toHaveBeenCalled()
    expect(input.value).toBe('0.5') // 回滚
  })

  it('int field also uses raw buffer', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Field name="batch" prop={intProp} value={4} onChange={onChange} />)

    const input = screen.getByRole('textbox') as HTMLInputElement
    await user.clear(input)
    await user.type(input, '16')
    expect(onChange).not.toHaveBeenCalled()
    fireEvent.blur(input)
    expect(onChange).toHaveBeenCalledWith(16)
  })

  it('external value change syncs raw only when input is not focused', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    const { rerender } = render(
      <Field name="lr" prop={floatProp} value={0.5} onChange={onChange} />,
    )
    const input = screen.getByRole('textbox') as HTMLInputElement

    // 用户 focus 进去开始输入
    await user.click(input)
    await user.clear(input)
    await user.type(input, '0.0')
    expect(input.value).toBe('0.0')

    // 此时外部 value 变化（如 SSE / 别的字段触发的 reset）—— 不应覆盖用户半截输入
    rerender(<Field name="lr" prop={floatProp} value={0.999} onChange={onChange} />)
    expect(input.value).toBe('0.0')

    // blur 后再改外部 value → 这次同步
    await user.tab()
    rerender(<Field name="lr" prop={floatProp} value={0.123} onChange={onChange} />)
    expect(input.value).toBe('0.123')
  })

  it('rolls back values below schema minimum (恢复 type=number min 校验)', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    const propWithRange: SchemaProperty = {
      ...floatProp,
      minimum: 0,
      maximum: 1,
    }
    render(
      <Field name="lr" prop={propWithRange} value={0.5} onChange={onChange} />,
    )

    const input = screen.getByRole('textbox') as HTMLInputElement
    await user.clear(input)
    await user.type(input, '-0.1')
    fireEvent.blur(input)
    expect(onChange).not.toHaveBeenCalled()
    expect(input.value).toBe('0.5')
  })

  it('rolls back values above schema maximum', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    const propWithRange: SchemaProperty = {
      ...floatProp,
      minimum: 0,
      maximum: 1,
    }
    render(
      <Field name="lr" prop={propWithRange} value={0.5} onChange={onChange} />,
    )

    const input = screen.getByRole('textbox') as HTMLInputElement
    await user.clear(input)
    await user.type(input, '5')
    fireEvent.blur(input)
    expect(onChange).not.toHaveBeenCalled()
    expect(input.value).toBe('0.5')
  })

  it('accepts values exactly at min / max boundaries', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    const propWithRange: SchemaProperty = {
      ...floatProp,
      minimum: 0,
      maximum: 1,
    }
    render(
      <Field name="lr" prop={propWithRange} value={0.5} onChange={onChange} />,
    )

    const input = screen.getByRole('textbox') as HTMLInputElement
    await user.clear(input)
    await user.type(input, '0')
    fireEvent.blur(input)
    expect(onChange).toHaveBeenLastCalledWith(0)

    await user.clear(input)
    await user.type(input, '1')
    fireEvent.blur(input)
    expect(onChange).toHaveBeenLastCalledWith(1)
  })

  it('disabled input cannot be typed into', () => {
    render(
      <Field
        name="lr"
        prop={floatProp}
        value={0.5}
        onChange={() => {}}
        disabled
      />,
    )
    expect(screen.getByRole('textbox')).toBeDisabled()
  })
})

const stringListProp: SchemaProperty = {
  type: 'array',
  items: { type: 'string' },
  default: [],
  group: 'sample',
  control: 'string-list',
  description: '',
}

describe('Field string-list input (sample_prompts 换行)', () => {
  it('preserves the newline while typing a second line', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(
      <Field
        name="sample_prompts"
        prop={stringListProp}
        value={['first prompt']}
        onChange={onChange}
      />,
    )

    const input = screen.getByRole('textbox') as HTMLTextAreaElement
    // 敲回车的瞬间（尾部空行还没内容）换行不能被 split+filter 吃掉
    await user.type(input, '{Enter}')
    expect(input.value).toBe('first prompt\n')
    await user.type(input, 'second prompt')
    expect(input.value).toBe('first prompt\nsecond prompt')
    expect(onChange).toHaveBeenLastCalledWith(['first prompt', 'second prompt'])
  })

  it('does not commit empty lines, and blur normalizes the raw text', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(
      <Field
        name="sample_prompts"
        prop={stringListProp}
        value={['first prompt']}
        onChange={onChange}
      />,
    )

    const input = screen.getByRole('textbox') as HTMLTextAreaElement
    await user.type(input, '{Enter}{Enter}  second prompt  ')
    expect(onChange).toHaveBeenLastCalledWith(['first prompt', 'second prompt'])
    fireEvent.blur(input)
    expect(input.value).toBe('first prompt\nsecond prompt')
  })

  it('external value change syncs display only when not focused', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    const { rerender } = render(
      <Field
        name="sample_prompts"
        prop={stringListProp}
        value={['a']}
        onChange={onChange}
      />,
    )
    const input = screen.getByRole('textbox') as HTMLTextAreaElement

    // focus 中外部 value 变化不覆盖用户半截输入
    await user.type(input, '{Enter}b')
    rerender(
      <Field
        name="sample_prompts"
        prop={stringListProp}
        value={['reset']}
        onChange={onChange}
      />,
    )
    expect(input.value).toBe('a\nb')

    // blur 后外部变化才同步
    await user.tab()
    rerender(
      <Field
        name="sample_prompts"
        prop={stringListProp}
        value={['x', 'y']}
        onChange={onChange}
      />,
    )
    expect(input.value).toBe('x\ny')
  })
})

describe('Field code input', () => {
  it('renders object values as formatted JSON and commits parsed objects on blur', () => {
    const onChange = vi.fn()
    render(
      <Field
        name="lora_reg_dims"
        prop={codeProp}
        value={{ 'lora_unet_.*double.*': 16 }}
        onChange={onChange}
      />,
    )

    const input = screen.getByRole('textbox') as HTMLTextAreaElement
    expect(input.value).toContain('"lora_unet_.*double.*": 16')

    fireEvent.change(input, { target: { value: '{"lora_unet_.*single.*": 8}' } })
    fireEvent.blur(input)
    expect(onChange).toHaveBeenCalledWith({ 'lora_unet_.*single.*': 8 })
  })

  it('rejects JSON string fragments instead of committing them as strings', () => {
    const onChange = vi.fn()
    render(
      <Field
        name="lora_reg_dims"
        prop={codeProp}
        value={null}
        onChange={onChange}
      />,
    )

    const input = screen.getByRole('textbox') as HTMLTextAreaElement
    fireEvent.change(input, { target: { value: '"lora_unet_.*double.*": 16' } })
    fireEvent.blur(input)
    expect(onChange).not.toHaveBeenCalled()
    expect(screen.getByText('Invalid JSON')).toBeInTheDocument()
  })
})
