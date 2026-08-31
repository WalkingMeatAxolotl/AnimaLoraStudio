import {
  forwardRef,
  type ButtonHTMLAttributes,
  type ReactNode,
} from 'react'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'warning' | 'danger'
export type ButtonSize = 'md' | 'sm' | 'xs'

interface ButtonPropsBase extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'aria-label'> {
  variant?: ButtonVariant
  size?: ButtonSize
  loading?: boolean
  iconOnly?: boolean
  children: ReactNode
  'aria-label'?: string
}

export type ButtonProps = ButtonPropsBase & (
  | { iconOnly: true; 'aria-label': string }
  | { iconOnly?: false; 'aria-label'?: string }
)

const VARIANT_CLASS: Record<ButtonVariant, string> = {
  primary: 'btn-primary',
  secondary: 'btn-secondary',
  ghost: 'btn-ghost',
  warning: 'btn-warn',
  danger: 'btn-danger',
}

const SIZE_CLASS: Record<ButtonSize, string> = {
  md: '',
  sm: 'btn-sm',
  xs: 'btn-xs',
}

export function buttonClassName({
  variant = 'secondary',
  size = 'md',
  iconOnly = false,
  loading = false,
  className = '',
}: {
  variant?: ButtonVariant
  size?: ButtonSize
  iconOnly?: boolean
  loading?: boolean
  className?: string
} = {}): string {
  return [
    'btn',
    VARIANT_CLASS[variant],
    SIZE_CLASS[size],
    iconOnly && 'btn-icon',
    loading && 'btn-loading',
    className,
  ].filter(Boolean).join(' ')
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button({
  variant = 'secondary',
  size = 'md',
  loading = false,
  iconOnly = false,
  className,
  children,
  disabled,
  type,
  'aria-label': ariaLabel,
  ...rest
}, ref) {
  return (
    <button
      {...rest}
      ref={ref}
      type={type ?? 'button'}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      aria-label={ariaLabel}
      className={buttonClassName({ variant, size, iconOnly, loading, className })}
    >
      {loading && <span className="btn-spinner" aria-hidden="true" />}
      {children}
    </button>
  )
})

export default Button
