import {
  forwardRef,
  type InputHTMLAttributes,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from 'react'

export type FormControlSize = 'md' | 'sm'
export type FormControlSurface = 'surface' | 'canvas' | 'sunken'

const SIZE_CLASS: Record<FormControlSize, string> = {
  md: '',
  sm: 'form-control-sm',
}

const SURFACE_CLASS: Record<FormControlSurface, string> = {
  surface: 'form-control-surface',
  canvas: 'form-control-canvas',
  sunken: 'form-control-sunken',
}

export function controlClassName({
  size = 'md',
  surface = 'surface',
  mono = false,
  className = '',
}: {
  size?: FormControlSize
  surface?: FormControlSurface
  mono?: boolean
  className?: string
} = {}): string {
  return [
    'form-control',
    SIZE_CLASS[size],
    SURFACE_CLASS[surface],
    mono && 'form-control-mono',
    className,
  ].filter(Boolean).join(' ')
}

interface VisualControlProps {
  controlSize?: FormControlSize
  surface?: FormControlSurface
  mono?: boolean
  invalid?: boolean
}

export interface InputProps
  extends InputHTMLAttributes<HTMLInputElement>,
    VisualControlProps {}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input({
  controlSize = 'md',
  surface = 'surface',
  mono = false,
  invalid = false,
  className,
  'aria-invalid': ariaInvalid,
  ...rest
}, ref) {
  return (
    <input
      {...rest}
      ref={ref}
      aria-invalid={invalid || ariaInvalid || undefined}
      className={controlClassName({ size: controlSize, surface, mono, className })}
    />
  )
})

export interface SelectProps
  extends SelectHTMLAttributes<HTMLSelectElement>,
    VisualControlProps {}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select({
  controlSize = 'md',
  surface = 'surface',
  mono = false,
  invalid = false,
  className,
  'aria-invalid': ariaInvalid,
  children,
  ...rest
}, ref) {
  return (
    <select
      {...rest}
      ref={ref}
      aria-invalid={invalid || ariaInvalid || undefined}
      className={controlClassName({ size: controlSize, surface, mono, className })}
    >
      {children}
    </select>
  )
})

export interface TextareaProps
  extends TextareaHTMLAttributes<HTMLTextAreaElement>,
    VisualControlProps {}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(function Textarea({
  controlSize = 'md',
  surface = 'surface',
  mono = false,
  invalid = false,
  className,
  'aria-invalid': ariaInvalid,
  ...rest
}, ref) {
  return (
    <textarea
      {...rest}
      ref={ref}
      aria-invalid={invalid || ariaInvalid || undefined}
      className={controlClassName({
        size: controlSize,
        surface,
        mono,
        className: ['form-control-textarea', className].filter(Boolean).join(' '),
      })}
    />
  )
})

export interface CheckboxProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  controlSize?: FormControlSize
  invalid?: boolean
}

export const Checkbox = forwardRef<HTMLInputElement, CheckboxProps>(function Checkbox({
  controlSize = 'md',
  invalid = false,
  className,
  'aria-invalid': ariaInvalid,
  ...rest
}, ref) {
  return (
    <input
      {...rest}
      ref={ref}
      type="checkbox"
      aria-invalid={invalid || ariaInvalid || undefined}
      className={[
        'form-checkbox',
        controlSize === 'sm' && 'form-checkbox-sm',
        className,
      ].filter(Boolean).join(' ')}
    />
  )
})
