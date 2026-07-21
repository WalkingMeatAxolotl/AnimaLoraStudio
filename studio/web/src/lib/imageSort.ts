/**
 * 图片列表统一排序。
 *
 * 各步骤页（概览 / 数据集 / 预处理 / 标签编辑 / 训练集筛选 / 正则集）显示的
 * 是同一批图，顺序必须一致。后端各端点返回的是路径字典序，对 booru 下载的
 * 纯数字文件名会把 `10.jpg` 排在 `9.jpg` 前面 —— 与用户认知的 id 顺序不符。
 * 所以顺序统一在渲染层由这里决定：文件名去后缀是纯数字时按数值排（booru id），
 * 其余按字典序落在数字之后。
 */

/** 排序键：纯数字 stem → 该数值；其余 → +∞（落到数字之后，再按名字比）。 */
export function numericIdKey(name: string): number {
  const stem = name.replace(/\.[^.]+$/, '')
  return /^\d+$/.test(stem) ? Number(stem) : Number.POSITIVE_INFINITY
}

/** 比较单个文件名（不含目录）。 */
export function compareImageName(a: string, b: string): number {
  const ka = numericIdKey(a)
  const kb = numericIdKey(b)
  // 两侧都非数字时 ka === kb === +∞，走 localeCompare，不会算出 NaN
  return ka === kb ? a.localeCompare(b) : ka - kb
}

/** 比较 `folder/name` 形式的相对路径：先按目录字典序，同目录内按文件名。 */
export function compareImagePath(a: string, b: string): number {
  const ia = a.lastIndexOf('/')
  const ib = b.lastIndexOf('/')
  const da = ia >= 0 ? a.slice(0, ia) : ''
  const db = ib >= 0 ? b.slice(0, ib) : ''
  if (da !== db) return da.localeCompare(db)
  return compareImageName(ia >= 0 ? a.slice(ia + 1) : a, ib >= 0 ? b.slice(ib + 1) : b)
}
