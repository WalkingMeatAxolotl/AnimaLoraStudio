import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type CaptionEntry, type ProjectSummary } from '../../../api/client'
import { useLocalStorageState } from '../../../lib/useLocalStorageState'

// 命名前缀对齐 useAdvancedMode 的 `studio:` 约定（PR #66 P1-4）。旧的
// `anima.generate.promptDataset.*` key 在 mount 时 migrate 一次后丢弃。
const LAST_PROJECT_KEY = 'studio:generate:promptDataset:projectId'
const LAST_VERSION_KEY = 'studio:generate:promptDataset:versionId'
const LEGACY_PROJECT_KEY = 'anima.generate.promptDataset.projectId'
const LEGACY_VERSION_KEY = 'anima.generate.promptDataset.versionId'

function migrateLegacyKey(legacyKey: string, newKey: string): void {
  if (typeof window === 'undefined') return
  if (window.localStorage.getItem(newKey) !== null) return
  const raw = window.localStorage.getItem(legacyKey)
  if (raw === null) return
  const n = Number(raw)
  if (Number.isFinite(n)) {
    window.localStorage.setItem(newKey, JSON.stringify(n))
  }
  window.localStorage.removeItem(legacyKey)
}

export interface DatasetPick {
  projectId: number
  versionId: number
  /** caption 文件名（含目录，例如 "5_concept/0001.txt"） */
  name: string
  /** caption 文本拆出的 tag 列表，按训练集原始顺序 */
  tags: string[]
}

/** 从训练集 caption 里选一条作为生成时的 prompt 后缀（不写入 sidebar 「正向」textarea）。
 *
 * 受控单选：
 * - 父组件控 open / close（× 触发 onClose）；生成不自动关
 * - 选中状态 (DatasetPick) 由父组件持有；**关闭 picker 时父组件应同时清空 value**，
 *   否则 datasetPick.tags 会继续被 handleGenerate 拼到 prompt，用户以为没选还在生效
 * - 点 list 行：未选 → 激活；已选同一行 → 取消（反选）
 * - 选中 caption 的 tags 在底部只读 textarea 展示，不写进上层 prompt 框
 *
 * pid/vid 是「浏览中」的状态，跟 value 解耦 —— 浏览时切别的 project/version 看
 * 不影响 value；用 localStorage 持久化跨 session 记忆浏览位置。
 */
export default function PromptFromDatasetPicker({
  value, onChange, onClose,
}: {
  /** 当前选中 caption（null = 未选） */
  value: DatasetPick | null
  onChange: (next: DatasetPick | null) => void
  onClose: () => void
}) {
  const { t } = useTranslation()
  // 一次性 migrate 旧 anima.* key 到 studio: 命名（PR #66 P1-4 约定）；module 顶部
  // 调用即可，没必要进 useEffect —— 没读 / 写 React state 副作用，只动 localStorage。
  if (typeof window !== 'undefined') {
    migrateLegacyKey(LEGACY_PROJECT_KEY, LAST_PROJECT_KEY)
    migrateLegacyKey(LEGACY_VERSION_KEY, LAST_VERSION_KEY)
  }

  const [projects, setProjects] = useState<ProjectSummary[]>([])
  // useLocalStorageState 默认值仅在 storage 无值时生效；有 value 时用它作初始的"浏览位置"
  const [pid, setPid] = useLocalStorageState<number | null>(LAST_PROJECT_KEY, value?.projectId ?? null)
  const [vid, setVid] = useLocalStorageState<number | null>(LAST_VERSION_KEY, value?.versionId ?? null)
  // 历史回填：value 切到一个 (projectId, versionId) 时，浏览中的 pid/vid 跟随它
  // —— 否则 caption 列表停留在用户上次浏览的版本，看不到当前 value.name 行高亮，
  //    底部 tags 又孤零显示，对不上号。控件外的状态(localStorage)不持久这次切换，
  //    只更新 in-memory state；用户关掉 picker 再开还会回到他们手选的位置。
  useEffect(() => {
    if (value == null) return
    setPid((cur) => (cur === value.projectId ? cur : value.projectId))
    setVid((cur) => (cur === value.versionId ? cur : value.versionId))
  }, [value?.projectId, value?.versionId])  // eslint-disable-line react-hooks/exhaustive-deps
  const [versions, setVersions] = useState<Array<{ id: number; label: string }>>([])
  const [captions, setCaptions] = useState<CaptionEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')

  // 1. 拉项目列表；若上次记的 pid 在新项目列表中不存在则清掉避免幽灵选择
  useEffect(() => {
    void api.listProjects()
      .then((items) => {
        setProjects(items)
        if (pid != null && !items.some((p) => p.id === pid)) setPid(null)
      })
      .catch((e) => setError(String(e)))
    // pid 进依赖会触发反复拉项目；mount 一次就够
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 2. 选项目后拉版本列表；优先复用 vid（如果该版本在新项目里仍存在）
  useEffect(() => {
    if (!pid) { setVersions([]); setVid(null); return }
    void api.getProject(pid)
      .then((p) => {
        const vs = p.versions.map((v) => ({ id: v.id, label: v.label }))
        setVersions(vs)
        if (vs.length > 0) {
          setVid((cur) => (cur && vs.some((v) => v.id === cur) ? cur : vs[0].id))
        } else {
          setVid(null)
        }
      })
      .catch((e) => setError(String(e)))
    // 同上：vid 只在 effect 内部读，不进依赖
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pid])

  // 3. 选版本后拉 captions
  useEffect(() => {
    if (!pid || !vid) { setCaptions([]); return }
    setLoading(true)
    setError(null)
    void api.listCaptionsFull(pid, vid)
      .then((r) => { setCaptions(r.items); setLoading(false) })
      .catch((e) => { setError(String(e)); setLoading(false) })
  }, [pid, vid])

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return captions
    return captions.filter((c) =>
      c.name.toLowerCase().includes(q) ||
      c.tags.some((t) => t.toLowerCase().includes(q))
    )
  }, [captions, search])

  // 当前 list 中匹配选中 caption 的 key（仅当浏览中的 pid/vid 与 value 一致才高亮）
  const selectedKeyInList = useMemo(() => {
    if (!value || value.projectId !== pid || value.versionId !== vid) return null
    return value.name
  }, [value, pid, vid])

  const tagsText = value ? value.tags.join(', ') : ''

  const handleRowClick = (c: CaptionEntry) => {
    if (
      value
      && value.projectId === pid
      && value.versionId === vid
      && value.name === c.name
    ) {
      // 反选
      onChange(null)
      return
    }
    if (!pid || !vid) return
    onChange({
      projectId: pid,
      versionId: vid,
      name: c.name,
      tags: c.tags,
    })
  }

  return (
    <div
      className="rounded-md border border-subtle bg-overlay p-2.5 flex flex-col gap-2"
      data-testid="prompt-dataset-picker"
    >
      {/* header */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-fg-secondary shrink-0">{t('generate.datasetPromptTitle')}</span>
        <span className="flex-1" />
        {value && (
          <button
            onClick={() => onChange(null)}
            className="btn btn-ghost btn-sm text-2xs text-fg-tertiary"
            title={t('generate.clearDatasetPickTitle')}
          >
            {t('generate.clearDatasetPick')}
          </button>
        )}
        <button
          onClick={onClose}
          className="btn btn-ghost btn-sm text-fg-tertiary px-1.5"
          title={t('generate.closeDatasetPickerTitle')}
          aria-label={t('common.close')}
        >
          ×
        </button>
      </div>

      {/* project / version 选择 */}
      <div className="flex gap-2">
        <select
          className="input text-xs flex-1"
          value={pid ?? ''}
          onChange={(e) => setPid(e.target.value ? Number(e.target.value) : null)}
          aria-label={t('generate.selectProjectAria')}
        >
          <option value="">{t('generate.selectProject')}</option>
          {projects.map((p) => (
            <option key={p.id} value={p.id}>{p.title}</option>
          ))}
        </select>
        <select
          className="input text-xs flex-1"
          value={vid ?? ''}
          onChange={(e) => setVid(e.target.value ? Number(e.target.value) : null)}
          disabled={versions.length === 0}
          aria-label={t('generate.selectVersionAria')}
        >
          <option value="">{t('generate.selectVersion')}</option>
          {versions.map((v) => (
            <option key={v.id} value={v.id}>{v.label}</option>
          ))}
        </select>
      </div>

      {/* search */}
      <input
        type="text"
        className="input text-xs"
        placeholder={t('generate.searchFilenameTag')}
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        disabled={!pid || !vid || captions.length === 0}
      />

      {error && <div className="text-2xs text-err">{error}</div>}

      {/* caption 列表 */}
      <div className="flex flex-col gap-px overflow-y-auto" style={{ maxHeight: 240 }}>
        {loading && <div className="text-2xs text-fg-tertiary">{t('common.loading')}</div>}
        {!loading && pid && vid && captions.length === 0 && !error && (
          <div className="text-2xs text-fg-tertiary">{t('generate.noCaptions')}</div>
        )}
        {!loading && filtered.map((c) => {
          const k = `${c.folder}/${c.name}`
          const active = selectedKeyInList === c.name
          return (
            <button
              key={k}
              onClick={() => handleRowClick(c)}
              className="flex items-center gap-2 px-2 py-1.5 rounded text-xs text-left border-none transition-colors"
              style={{
                background: active ? 'var(--accent-soft)' : 'transparent',
                color: active ? 'var(--accent)' : 'var(--fg-secondary)',
                cursor: 'pointer',
              }}
            >
              <span className="font-mono text-2xs shrink-0">{active ? '✓' : '+'}</span>
              <div className="flex-1 min-w-0">
                <div className="font-medium truncate">{c.name}</div>
                <div className="text-2xs text-fg-tertiary truncate">
                  {c.tags.slice(0, 6).join(', ')}{c.tags.length > 6 ? ` (+${c.tags.length - 6})` : ''}
                </div>
              </div>
            </button>
          )
        })}
      </div>

      <label className="caption block mt-1">{t('generate.selectedDatasetTagsLabel')}</label>
      <textarea
        className="input w-full font-mono text-xs resize-y"
        rows={3}
        value={tagsText}
        readOnly
        placeholder={t('generate.selectedDatasetTagsPlaceholder')}
        aria-label={t('generate.selectedDatasetTagsAria')}
      />
    </div>
  )
}
