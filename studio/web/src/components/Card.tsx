import type { HTMLAttributes, ReactNode } from 'react'

export type CardTone = 'surface' | 'sunken'
export type CardRadius = 'default' | 'compact'
export type CardPadding = 'none' | 'sm' | 'md' | 'lg'
export type CardElement = 'div' | 'section' | 'article'

export interface CardProps extends HTMLAttributes<HTMLElement> {
  as?: CardElement
  tone?: CardTone
  radius?: CardRadius
  padding?: CardPadding
  interactive?: boolean
  children: ReactNode
}

const TONE_CLASS: Record<CardTone, string> = {
  surface: '',
  sunken: 'card-sunken',
}

const RADIUS_CLASS: Record<CardRadius, string> = {
  default: '',
  compact: 'card-compact',
}

const PADDING_CLASS: Record<CardPadding, string> = {
  none: '',
  sm: 'card-pad-sm',
  md: 'card-pad-md',
  lg: 'card-pad-lg',
}

export function cardClassName({
  tone = 'surface',
  radius = 'default',
  padding = 'none',
  interactive = false,
  className = '',
}: {
  tone?: CardTone
  radius?: CardRadius
  padding?: CardPadding
  interactive?: boolean
  className?: string
} = {}): string {
  return [
    'card',
    TONE_CLASS[tone],
    RADIUS_CLASS[radius],
    PADDING_CLASS[padding],
    interactive && 'card-hover',
    className,
  ].filter(Boolean).join(' ')
}

export default function Card({
  as: Element = 'div',
  tone = 'surface',
  radius = 'default',
  padding = 'none',
  interactive = false,
  className,
  children,
  ...rest
}: CardProps) {
  return (
    <Element
      {...rest}
      className={cardClassName({ tone, radius, padding, interactive, className })}
    >
      {children}
    </Element>
  )
}
