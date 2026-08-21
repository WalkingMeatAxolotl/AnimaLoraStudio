import { describe, expect, it } from 'vitest'

import { isLineVisible, lastVisibleLine, levelClass, parseLogLines } from './logLines'

const H = (lvl: string, logger: string, msg: string, t = '14:03:22.417') =>
  `2026-08-19 ${t} ${lvl.padEnd(5)} ${logger}: ${msg}`

describe('parseLogLines（行契约）', () => {
  it('行头解析出 time / level / logger / msg（levelname 只补不截，WARNING 全名）', () => {
    const [a, b] = parseLogLines([H('INFO', 'training.loop', 'step=1'), H('WARNING', 'utils.x', 'careful')])
    expect(a).toMatchObject({ isHeader: true, level: 'INFO', time: '14:03:22.417', logger: 'training.loop', msg: 'step=1' })
    expect(b).toMatchObject({ isHeader: true, level: 'WARNING', logger: 'utils.x', msg: 'careful' })
  })

  it('续行继承上一条记录的级别（traceback 属于 ERROR 记录）', () => {
    const out = parseLogLines([
      H('ERROR', 'studio.workers.tag_worker', 'job crashed'),
      'Traceback (most recent call last):',
      '  File "x.py", line 1',
      'RuntimeError: boom',
      H('INFO', 'training.loop', 'bye'),
    ])
    expect(out.map((l) => l.level)).toEqual(['ERROR', 'ERROR', 'ERROR', 'ERROR', 'INFO'])
    expect(out.slice(1, 4).every((l) => !l.isHeader)).toBe(true)
  })

  it('老格式 / 裸 print 行 level=null，且不被之后的续行逻辑污染', () => {
    const out = parseLogLines([
      '2026-08-10 03:43:23,610 - INFO - 训练完成!',   // 老 basicConfig 格式：逗号毫秒、无 logger
      'epoch=0 step=10 loss=0.1',
      H('DEBUG', 'training.loop', '[显存] ARB 切桶'),
      'still debug continuation',
    ])
    expect(out.map((l) => l.level)).toEqual([null, null, 'DEBUG', 'DEBUG'])
  })

  it('前端合成行的裸 LEVEL 前缀被识别，且不向后传染', () => {
    const out = parseLogLines(['ERROR: 扫描失败: disk', 'next plain line', 'WARNING something', 'INFO: ok'])
    expect(out.map((l) => l.level)).toEqual(['ERROR', null, 'WARNING', 'INFO'])
    expect(out[0].isHeader).toBe(false)
  })

  it('未知级别名（非五级）不当行头', () => {
    const out = parseLogLines(['2026-08-19 14:03:22.417 TRACE x.y: z'])
    expect(out[0]).toMatchObject({ isHeader: true, level: null, logger: 'x.y' })
  })
})

describe('过滤与着色', () => {
  it('showDebug=false 只隐藏 DEBUG；未知级别按 INFO 保留', () => {
    const out = parseLogLines(['plain', H('DEBUG', 'a', 'd'), H('INFO', 'a', 'i'), H('ERROR', 'a', 'e')])
    expect(out.filter((l) => isLineVisible(l, false)).map((l) => l.level)).toEqual([null, 'INFO', 'ERROR'])
    expect(out.filter((l) => isLineVisible(l, true))).toHaveLength(4)
  })

  it('ERROR/CRITICAL 红、WARNING 黄、DEBUG 弱化、其余默认', () => {
    expect(levelClass('ERROR')).toBe('text-err')
    expect(levelClass('CRITICAL')).toBe('text-err')
    expect(levelClass('WARNING')).toBe('text-warn')
    expect(levelClass('DEBUG')).toBe('text-fg-tertiary')
    expect(levelClass('INFO')).toBe(levelClass(null))
  })
})

describe('lastVisibleLine（抽屉收起态预览）', () => {
  it('跳过尾部的 DEBUG 行与续行，取最后一条 INFO+ 记录原文', () => {
    const lines = [
      H('INFO', 'w.tag', 'tagged 43/43'),
      H('DEBUG', 'w.tag', 'internal detail'),
      '  continuation of debug',
    ]
    expect(lastVisibleLine(lines)).toBe(lines[0])
  })

  it('WARNING/ERROR 不被跳过', () => {
    const lines = [H('INFO', 'a', 'x'), H('ERROR', 'a', 'boom')]
    expect(lastVisibleLine(lines)).toBe(lines[1])
  })

  it('全 plain（前端合成日志）退回最后一行原文', () => {
    expect(lastVisibleLine(['a', 'b'])).toBe('b')
  })

  it('有行头但全 DEBUG → 空串（预览留白，不显示调试行）', () => {
    expect(lastVisibleLine([H('DEBUG', 'a', 'x')])).toBe('')
  })

  it('空输入 → 空串', () => {
    expect(lastVisibleLine([])).toBe('')
  })
})
