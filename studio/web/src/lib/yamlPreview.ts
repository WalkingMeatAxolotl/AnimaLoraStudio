// configToYaml —— 把扁平训练 config 渲染成与后端落盘一致的 YAML 文本。
//
// 对齐 yaml.safe_dump(..., allow_unicode=True, sort_keys=False,
// default_flow_style=False) 在扁平 config 上的输出习惯：
// - 键保持插入顺序（= 后端 model_dump 的字段定义顺序，不排序）
// - 列表用块状写法（`- item` 顶格跟在 key 下一行），空列表 `[]`
// - 字符串仅在有解析歧义（数字/布尔/null 长相、YAML 特殊字符、首尾空格）
//   时才加引号，路径 / 枚举 / CJK 保持裸写
// 已知偏差：JSON 丢掉 int/float 区分，整数值的 float 显示为 `32` 而非后端
// 的 `32.0`（语义等价，加载时 pydantic 会转型）。

/** YAML 会解析成非字符串的裸写形式：布尔 / null / 空 */
const AMBIGUOUS_RE = /^(?:true|false|yes|no|on|off|null|~)$/i
/** 数字长相（含下划线分组 / 科学计数 / hex / octal） */
const NUMBER_LIKE_RE = /^[-+]?(?:\d[\d_]*(?:\.\d*)?(?:e[-+]?\d+)?|\.\d+|0x[\da-f]+|0o[0-7]+)$/i

function yamlScalar(v: unknown): string {
  if (v === null || v === undefined) return 'null'
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number') return String(v)
  if (typeof v === 'object') return JSON.stringify(v) // 兜底：flow mapping 也是合法 YAML
  const s = String(v)
  if (s === '') return "''"
  if (s.includes('\n')) return JSON.stringify(s) // 多行 → 双引号转义
  const needsQuote =
    AMBIGUOUS_RE.test(s) ||
    NUMBER_LIKE_RE.test(s) ||
    /^\s|\s$/.test(s) ||
    /^[-?:,[\]{}#&*!|>'"%@`]/.test(s) ||
    /: |:$| #/.test(s)
  if (!needsQuote) return s
  if (s.includes("'")) return JSON.stringify(s)
  return `'${s}'`
}

export function configToYaml(config: Record<string, unknown>): string {
  return Object.entries(config)
    .map(([key, v]) => {
      if (Array.isArray(v)) {
        if (v.length === 0) return `${key}: []`
        return `${key}:\n` + v.map((x) => `- ${yamlScalar(x)}`).join('\n')
      }
      return `${key}: ${yamlScalar(v)}`
    })
    .join('\n')
}
