import { useEffect, useMemo, useState } from 'react'
import {
  api,
  type ConfigData,
  type ConfigSummary,
  type SchemaResponse,
} from '../api/client'
import SchemaForm from '../components/SchemaForm'

export default function ConfigsPage() {
  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [list, setList] = useState<ConfigSummary[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [values, setValues] = useState<ConfigData>({})
  const [original, setOriginal] = useState<ConfigData>({})
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // 拉 schema + 列表 ----------------------------------------------------
  useEffect(() => {
    api.schema().then(setSchema).catch((e) => setError(String(e)))
    refreshList()
  }, [])

  const refreshList = () => api.listConfigs().then(setList).catch(() => {})

  // 选中某个配置：拉详情填表 ---------------------------------------------
  useEffect(() => {
    if (!selected) {
      setValues({})
      setOriginal({})
      return
    }
    api
      .getConfig(selected)
      .then((data) => {
        setValues(data)
        setOriginal(data)
      })
      .catch((e) => setError(String(e)))
  }, [selected])

  const dirty = useMemo(
    () => JSON.stringify(values) !== JSON.stringify(original),
    [values, original]
  )

  // 默认值（从 schema 抽出来），「新建」时填入 ---------------------------
  const defaultsFromSchema = useMemo(() => {
    if (!schema) return {}
    const out: ConfigData = {}
    for (const [name, prop] of Object.entries(schema.schema.properties)) {
      if (prop.default !== undefined) out[name] = prop.default
    }
    return out
  }, [schema])

  // 操作 ----------------------------------------------------------------
  const handleNew = () => {
    setSelected(null)
    setValues(defaultsFromSchema)
    setOriginal({})
  }

  const handleSave = async () => {
    const name = selected ?? prompt('配置名（字母/数字/下划线/连字符）：')
    if (!name) return
    setBusy(true)
    setError(null)
    try {
      await api.saveConfig(name, values)
      setSelected(name)
      setOriginal(values)
      await refreshList()
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  const handleDuplicate = async () => {
    if (!selected) return
    const name = prompt('新名字：', `${selected}_copy`)
    if (!name) return
    setBusy(true)
    try {
      await api.duplicateConfig(selected, name)
      await refreshList()
      setSelected(name)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  const handleDelete = async () => {
    if (!selected) return
    if (!confirm(`删除 ${selected}?`)) return
    setBusy(true)
    try {
      await api.deleteConfig(selected)
      setSelected(null)
      await refreshList()
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  // 渲染 ----------------------------------------------------------------
  return (
    <div className="grid grid-cols-[260px_1fr] gap-4 h-full">
      {/* 左侧列表 */}
      <aside className="border border-slate-700 rounded-lg bg-slate-800/40 p-3 flex flex-col">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-slate-200">配置</h2>
          <button
            onClick={handleNew}
            className="text-xs px-2 py-1 rounded bg-cyan-600 hover:bg-cyan-500 text-white"
          >
            + 新建
          </button>
        </div>
        <div className="flex-1 overflow-y-auto space-y-1">
          {list.length === 0 && (
            <p className="text-slate-500 text-sm py-4 text-center">
              还没有配置
            </p>
          )}
          {list.map((c) => (
            <button
              key={c.name}
              onClick={() => setSelected(c.name)}
              className={
                'w-full text-left px-2 py-1.5 rounded text-sm ' +
                (selected === c.name
                  ? 'bg-cyan-600/20 text-cyan-300'
                  : 'text-slate-300 hover:bg-slate-800/60')
              }
            >
              {c.name}
            </button>
          ))}
        </div>
      </aside>

      {/* 右侧表单 */}
      <main className="overflow-y-auto pr-2">
        {!schema ? (
          <p className="text-slate-500">加载 schema...</p>
        ) : (
          <>
            <header className="flex items-center gap-2 mb-4 sticky top-0 bg-slate-900 py-2 z-10">
              <h1 className="text-xl font-semibold flex-1">
                {selected ? selected : '新配置'}
                {dirty && (
                  <span className="ml-2 text-amber-400 text-sm">●未保存</span>
                )}
              </h1>
              <button
                disabled={busy || !dirty}
                onClick={handleSave}
                className="px-3 py-1.5 rounded text-sm bg-cyan-600 hover:bg-cyan-500
                  disabled:bg-slate-700 disabled:text-slate-500"
              >
                保存
              </button>
              {selected && (
                <>
                  <button
                    disabled={busy}
                    onClick={handleDuplicate}
                    className="px-3 py-1.5 rounded text-sm bg-slate-700 hover:bg-slate-600"
                  >
                    复制
                  </button>
                  <button
                    disabled={busy}
                    onClick={handleDelete}
                    className="px-3 py-1.5 rounded text-sm bg-red-700/80 hover:bg-red-600"
                  >
                    删除
                  </button>
                </>
              )}
            </header>
            {error && (
              <div className="mb-3 p-3 rounded bg-red-900/40 border border-red-700 text-red-300 text-sm font-mono">
                {error}
              </div>
            )}
            {(selected || Object.keys(values).length > 0) && (
              <SchemaForm
                schema={schema}
                values={values}
                onChange={setValues}
              />
            )}
            {!selected && Object.keys(values).length === 0 && (
              <p className="text-slate-500 mt-12 text-center">
                左侧选一个配置来编辑，或点「+ 新建」开始。
              </p>
            )}
          </>
        )}
      </main>
    </div>
  )
}
