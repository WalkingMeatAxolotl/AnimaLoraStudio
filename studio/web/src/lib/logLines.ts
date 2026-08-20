/** 行契约解析（docs/design/logging-target-state.md §3.2；后端 `LOG_LINE_RE`）。
 *
 * run.log / daemon stderr 每条记录的行头：
 *   `2026-08-19 14:03:22.417 INFO  training.loop: epoch=0 step=50 ...`
 * 不匹配行头的行（traceback、多行消息）是上一条记录的续行，继承其级别。
 *
 * 兼容两类非契约行：
 *   - 老格式 / 裸 print 行：level=null，显示上按 INFO 对待（不过滤、默认色）
 *   - 前端合成行可用裸 `LEVEL:` / `LEVEL ` 前缀标级别（如去重扫描的错误行）
 */

export type LogLevel = 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL'

export type LogLineKind = 'header' | 'continuation' | 'bare' | 'plain'

export interface LogLine {
  /** 原文（未改动） */
  raw: string
  /** header = 契约行头；continuation = 行头之后的续行（缩进、继承级别）；
   *  bare = 前端合成的 `LEVEL:` 前缀行；plain = 老格式 / 裸 print 行 */
  kind: LogLineKind
  /** 是否契约行头（= kind === 'header'） */
  isHeader: boolean
  /** 本行有效级别（续行继承上一条记录；无法判定 = null） */
  level: LogLevel | null
  /** 行头才有：时间（只留 HH:MM:SS.mmm）、来源 logger、消息体 */
  time?: string
  logger?: string
  msg?: string
}

const LEVELS: readonly LogLevel[] = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

/** 与后端 LOG_LINE_RE 同构：`<YYYY-MM-DD HH:MM:SS.mmm> <LEVEL>\s+<logger>: <msg>` */
const HEADER_RE = /^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d{3}) ([A-Z]+)\s+([^\s:]+): (.*)$/
/** 前端合成行 / 兼容：`ERROR: msg` / `WARNING msg` */
const BARE_LEVEL_RE = /^(DEBUG|INFO|WARNING|ERROR|CRITICAL)(?::| )\s*/

export function isLogLevel(s: string): s is LogLevel {
  return (LEVELS as readonly string[]).includes(s)
}

/** 解析一段连续的行；续行继承前一条记录的级别，所以必须按顺序整段解析。 */
export function parseLogLines(lines: readonly string[]): LogLine[] {
  const out: LogLine[] = []
  let current: LogLevel | null = null
  let inRecord = false
  for (const raw of lines) {
    const m = HEADER_RE.exec(raw)
    if (m) {
      const lvl = isLogLevel(m[3]) ? m[3] : null
      current = lvl
      inRecord = true
      out.push({ raw, kind: 'header', isHeader: true, level: lvl, time: m[2], logger: m[4], msg: m[5] })
      continue
    }
    const b = BARE_LEVEL_RE.exec(raw)
    if (b && isLogLevel(b[1])) {
      // 合成行自成一条记录（不继承、也不被后续续行继承级别）
      current = null
      inRecord = false
      out.push({ raw, kind: 'bare', isHeader: false, level: b[1] })
      continue
    }
    // 续行：继承；老格式 / 裸行：null（显示按 INFO）
    out.push({ raw, kind: inRecord ? 'continuation' : 'plain', isHeader: false, level: inRecord ? current : null })
  }
  return out
}

export const LEVEL_RANK: Record<LogLevel, number> = {
  DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50,
}

/** 显示过滤：`showDebug=false` 时隐藏 DEBUG 行；级别未知的行视为 INFO 保留。 */
export function isLineVisible(line: LogLine, showDebug: boolean): boolean {
  if (showDebug) return true
  return line.level !== 'DEBUG'
}

/** 抽屉收起态的预览行：最后一条非 DEBUG 的记录（跳过续行与调试行）。
 *  日志全无行头（前端合成 plain 文本）时退回最后一行原文；有行头但全是
 *  DEBUG 时返回空串（预览留白好过把调试行顶到 header 上）。 */
export function lastVisibleLine(lines: readonly string[]): string {
  const parsed = parseLogLines(lines)
  let sawLeveled = false
  for (let i = parsed.length - 1; i >= 0; i--) {
    const l = parsed[i]
    if (l.kind === 'header' || l.kind === 'bare') {
      sawLeveled = true
      if (l.level !== 'DEBUG') return l.raw
    }
  }
  return sawLeveled ? '' : (lines[lines.length - 1] ?? '')
}

/** 行内着色 token（与 Tailwind 色类一一对应，整站一套）。 */
export function levelClass(level: LogLevel | null): string {
  switch (level) {
    case 'ERROR':
    case 'CRITICAL':
      return 'text-err'
    case 'WARNING':
      return 'text-warn'
    case 'DEBUG':
      return 'text-fg-tertiary'
    default:
      return 'text-fg-secondary'
  }
}
