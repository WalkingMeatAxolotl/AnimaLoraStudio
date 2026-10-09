import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { expect, it } from 'vitest'

const tokens = readFileSync(resolve('src/styles/tokens.css'), 'utf8')
const crop = readFileSync(resolve('src/styles/preprocess-crop.css'), 'utf8')

function rule(selector, source = tokens) {
  const start = source.indexOf(`${selector} {`)
  expect(start, `Missing CSS rule: ${selector}`).toBeGreaterThanOrEqual(0)
  return source.slice(start).split('}')[0]
}

// 按压 / 悬停位移若写 transform，会整体覆盖 Tailwind 的 -translate-* 定位，
// 绝对定位居中的按钮按下即跳位（XY 轴删除按钮 #598）。必须用独立 translate 属性。
it.each([
  ['.btn:active:not(:disabled)', tokens],
  ['.btn:disabled', tokens],
  ['.card-hover:hover', tokens],
  ['.fs-thumb:hover', crop],
])('%s shifts via translate, not transform', (selector, source) => {
  const body = rule(selector, source)
  expect(body).toMatch(/(^|[\s;{])translate:/)
  expect(body).not.toMatch(/(^|[\s;{])transform:/)
})

it.each(['.btn', '.card-hover'])('%s transitions translate instead of transform', (selector) => {
  const body = rule(selector)
  expect(body).toContain('translate 120ms ease')
  expect(body).not.toMatch(/(^|[\s,])transform 120ms/)
})
