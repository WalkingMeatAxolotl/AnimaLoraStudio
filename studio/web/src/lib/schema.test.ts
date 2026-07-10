import { describe, expect, it } from 'vitest'
import type { SchemaProperty } from '../api/client'
import { controlKind, evalShowWhen, fieldLabel, pruneInactiveConfig } from './schema'

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

  it('maps integer arrays (e.g. resolution) to int-list, string arrays to string-list', () => {
    expect(
      controlKind({ type: 'array', items: { type: 'integer' } } as SchemaProperty)
    ).toBe('int-list')
    expect(
      controlKind({ type: 'array', items: { type: 'number' } } as SchemaProperty)
    ).toBe('int-list')
    expect(
      controlKind({ type: 'array', items: { type: 'string' } } as SchemaProperty)
    ).toBe('string-list')
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

  it('handles || branches', () => {
    const expr = 'optimizer_type==prodigy||optimizer_type==prodigy_plus_schedulefree'
    expect(evalShowWhen(expr, { optimizer_type: 'prodigy' })).toBe(true)
    expect(evalShowWhen(expr, { optimizer_type: 'prodigy_plus_schedulefree' })).toBe(true)
    expect(evalShowWhen(expr, { optimizer_type: 'adamw' })).toBe(false)
  })

  it('returns true on unparseable expressions (failsafe)', () => {
    expect(evalShowWhen('garbage', {})).toBe(true)
  })

  // evalShowWhen 同时被 show_when 和 disable_when 复用（同一表达式语法）
  it('works for PPSF disable_when use case', () => {
    expect(
      evalShowWhen('optimizer_type==prodigy_plus_schedulefree', {
        optimizer_type: 'prodigy_plus_schedulefree',
      })
    ).toBe(true)
    expect(
      evalShowWhen('optimizer_type==prodigy_plus_schedulefree', {
        optimizer_type: 'adamw',
      })
    ).toBe(false)
  })
})

// 语义与后端 studio/domain/config_prune.py 的 prune_inactive_fields 保持对照
describe('pruneInactiveConfig', () => {
  const properties = {
    optimizer_type: {} as SchemaProperty,
    came_beta1: { show_when: 'optimizer_type==came' } as SchemaProperty,
    lr_scheduler: { disable_when: 'optimizer_type==prodigy' } as SchemaProperty,
  }

  it('drops fields whose show_when is false, even with stale values', () => {
    const config = { optimizer_type: 'adamw', came_beta1: 0.5, lr_scheduler: 'cosine' }
    expect(pruneInactiveConfig(config, properties)).toEqual({
      optimizer_type: 'adamw',
      lr_scheduler: 'cosine',
    })
  })

  it('keeps fields whose show_when is true', () => {
    const config = { optimizer_type: 'came', came_beta1: 0.5, lr_scheduler: 'cosine' }
    expect(pruneInactiveConfig(config, properties)).toEqual(config)
  })

  it('keeps disable_when fields and keys without schema entry', () => {
    const config = { optimizer_type: 'prodigy', lr_scheduler: 'none', unknown_key: 1 }
    expect(pruneInactiveConfig(config, properties)).toEqual(config)
  })

  it('drops hidden fields at their schema default, keeps overrides', () => {
    const props = {
      no_progress: { hidden: true, default: true } as SchemaProperty,
      log_every: { hidden: true, default: 10 } as SchemaProperty,
      trigger_word: { hidden: true, default: '' } as SchemaProperty,
    }
    expect(
      pruneInactiveConfig(
        { no_progress: true, log_every: 10, trigger_word: 'miku' },
        props
      )
    ).toEqual({ trigger_word: 'miku' })
    expect(
      pruneInactiveConfig({ no_progress: false, trigger_word: '' }, props)
    ).toEqual({ no_progress: false })
  })

  it('keeps hidden fields when schema carries no default', () => {
    const props = { mystery: { hidden: true } as SchemaProperty }
    expect(pruneInactiveConfig({ mystery: 1 }, props)).toEqual({ mystery: 1 })
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
