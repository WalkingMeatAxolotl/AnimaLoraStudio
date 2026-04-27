import { describe, expect, it } from 'vitest'
import type { SchemaProperty } from '../api/client'
import { controlKind, evalShowWhen, fieldLabel } from './schema'

describe('controlKind', () => {
  it('uses explicit control field when provided', () => {
    expect(controlKind({ control: 'path' } as SchemaProperty)).toBe('path')
    expect(controlKind({ control: 'textarea' } as SchemaProperty)).toBe(
      'textarea'
    )
    expect(controlKind({ control: 'string-list' } as SchemaProperty)).toBe(
      'string-list'
    )
  })

  it("ignores control='auto' and falls back to type inference", () => {
    expect(
      controlKind({ control: 'auto', type: 'integer' } as SchemaProperty)
    ).toBe('int')
  })

  it('detects enum → select', () => {
    expect(
      controlKind({ enum: ['a', 'b'], type: 'string' } as SchemaProperty)
    ).toBe('select')
  })

  it('maps primitive types', () => {
    expect(controlKind({ type: 'boolean' } as SchemaProperty)).toBe('bool')
    expect(controlKind({ type: 'integer' } as SchemaProperty)).toBe('int')
    expect(controlKind({ type: 'number' } as SchemaProperty)).toBe('float')
    expect(controlKind({ type: 'array' } as SchemaProperty)).toBe('string-list')
    expect(controlKind({ type: 'string' } as SchemaProperty)).toBe('string')
  })

  it('handles Optional[T] via anyOf', () => {
    expect(
      controlKind({
        anyOf: [{ type: 'string' }, { type: 'null' }],
      } as SchemaProperty)
    ).toBe('string')
    expect(
      controlKind({
        anyOf: [{ type: 'integer' }, { type: 'null' }],
      } as SchemaProperty)
    ).toBe('int')
  })
})

describe('evalShowWhen', () => {
  it('returns true when expression is empty', () => {
    expect(evalShowWhen(undefined, {})).toBe(true)
    expect(evalShowWhen('', {})).toBe(true)
  })

  it('handles == matching', () => {
    expect(evalShowWhen('mode == prodigy', { mode: 'prodigy' })).toBe(true)
    expect(evalShowWhen('mode == prodigy', { mode: 'adamw' })).toBe(false)
  })

  it('handles != matching', () => {
    expect(evalShowWhen('lr_scheduler != none', { lr_scheduler: 'cosine' })).toBe(
      true
    )
    expect(evalShowWhen('lr_scheduler != none', { lr_scheduler: 'none' })).toBe(
      false
    )
  })

  it('returns true on unparseable expressions (failsafe)', () => {
    expect(evalShowWhen('garbage', {})).toBe(true)
  })
})

describe('fieldLabel', () => {
  it('capitalizes underscored words', () => {
    expect(fieldLabel('lora_rank')).toBe('Lora Rank')
    expect(fieldLabel('prodigy_d_coef')).toBe('Prodigy D Coef')
    expect(fieldLabel('seed')).toBe('Seed')
  })

  it('handles empty segments gracefully', () => {
    expect(fieldLabel('a__b')).toBe('A  B')
  })
})
