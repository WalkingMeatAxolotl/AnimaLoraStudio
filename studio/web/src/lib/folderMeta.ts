/** 数据集文件夹名元信息解析。
 *
 *  SYNC WITH `runtime/training/dataset.py:ImageDataset._parse_folder_meta` —— 两处
 *  解析必须一致，否则前端显示 / 放大目标 ≠ trainer 实际行为。
 *
 *  文法 `[Npx_][R_]label`（token 顺序，均可选）：
 *   - `\d+px` 分辨率前缀（snap 到 64 倍数 + clamp [256,4096]）
 *   - `\d+` Kohya repeat
 *   - 其余为 label
 */
export interface FolderMeta {
  /** px 前缀指定的分辨率；null = 无前缀（用 config 分辨率列表 fan-out）。 */
  reso: number | null
  repeat: number
  label: string
}

export function parseFolderMeta(name: string): FolderMeta {
  let reso: number | null = null
  let repeat = 1
  let rest = name
  let m = rest.match(/^(\d+)px_(.*)$/)
  if (m) {
    reso = Math.max(256, Math.min(4096, Math.round(parseInt(m[1], 10) / 64) * 64))
    rest = m[2]
  }
  m = rest.match(/^(\d+)_(.*)$/)
  if (m) {
    repeat = Math.max(parseInt(m[1], 10), 1)
    rest = m[2]
  }
  return { reso, repeat, label: rest }
}
