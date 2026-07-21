import { describe, expect, it } from 'vitest'
import { compareImageName, compareImagePath, numericIdKey } from './imageSort'

describe('numericIdKey', () => {
  it('纯数字 stem 取数值', () => {
    expect(numericIdKey('9.jpg')).toBe(9)
    expect(numericIdKey('10.png')).toBe(10)
    expect(numericIdKey('0001.webp')).toBe(1)
  })

  it('非纯数字 stem 落到 +∞', () => {
    expect(numericIdKey('IMG_9.jpg')).toBe(Number.POSITIVE_INFINITY)
    expect(numericIdKey('9_c0.jpg')).toBe(Number.POSITIVE_INFINITY)
    expect(numericIdKey('cover.png')).toBe(Number.POSITIVE_INFINITY)
  })
})

describe('compareImageName', () => {
  it('booru id 按数值排，不是字典序', () => {
    const names = ['10.jpg', '9.jpg', '100.jpg', '2.jpg']
    expect([...names].sort(compareImageName)).toEqual([
      '2.jpg', '9.jpg', '10.jpg', '100.jpg',
    ])
  })

  it('非数字名排在数字名之后，彼此按字典序', () => {
    const names = ['cover.png', '3.jpg', 'aaa.png', '12.jpg']
    expect([...names].sort(compareImageName)).toEqual([
      '3.jpg', '12.jpg', 'aaa.png', 'cover.png',
    ])
  })

  it('两个非数字名不产生 NaN（Infinity - Infinity 陷阱）', () => {
    expect(compareImageName('b.png', 'a.png')).toBeGreaterThan(0)
    expect(compareImageName('a.png', 'a.png')).toBe(0)
  })

  it('扩展名不同但 id 相同时按全名比', () => {
    expect(compareImageName('7.jpg', '7.png')).toBeLessThan(0)
  })
})

describe('compareImagePath', () => {
  it('先按目录字典序，同目录内按 id', () => {
    const paths = ['5_b/10.jpg', '1_a/9.jpg', '5_b/9.jpg', '1_a/10.jpg']
    expect([...paths].sort(compareImagePath)).toEqual([
      '1_a/9.jpg', '1_a/10.jpg', '5_b/9.jpg', '5_b/10.jpg',
    ])
  })

  it('无目录的名字视作根目录，排在有目录的之前', () => {
    const paths = ['1_a/9.jpg', '10.jpg', '9.jpg']
    expect([...paths].sort(compareImagePath)).toEqual([
      '9.jpg', '10.jpg', '1_a/9.jpg',
    ])
  })

  it('嵌套目录按完整前缀比', () => {
    expect(compareImagePath('a/b/1.jpg', 'a/c/1.jpg')).toBeLessThan(0)
  })
})
