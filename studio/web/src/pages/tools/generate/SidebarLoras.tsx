import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { LoraEntry } from '../../../api/client'
import PathPicker from '../../../components/PathPicker'
import AddSlotButton from './AddSlotButton'
import InlineLoraPicker, { type PickedLora } from './InlineLoraPicker'
import type { LoraCatalog } from './useLoraCatalog'

/** Sidebar 的 LoRA 区：每个 LoRA = 一个常驻 picker 槽（项目下拉 + ckpt chip + 权重 + ×）。
 *
 * 数据模型统一：所有 picker 槽都用 loras[] 表示，path='' 表示「空槽」（用户
 * 点了 + 添加 LoRA 但还没挑 ckpt）。这样：
 *   - 「+ 添加 LoRA」push 一条空 entry → 新增一个 picker（key 稳定，不会闪）
 *   - 反选 (点已选 chip) = 槽 path 设回 ''，picker 自己仍渲染 → 跟初次打开一样
 *   - × = 真正把 entry 从数组里删掉
 * Generate.tsx handleGenerate 在送 backend 前会 `loras.filter((l) => l.path.trim())`
 * 过滤空槽，不影响 enqueue。 */
export default function SidebarLoras({
  loras, onChange, catalog,
}: {
  loras: LoraEntry[]
  onChange: (l: LoraEntry[]) => void
  catalog: LoraCatalog
}) {
  const { t } = useTranslation()
  const [externalForIdx, setExternalForIdx] = useState<number | null>(null)

  // 已选 path（互相 disable，避免重复添加）—— 排除空槽
  const existingPaths = useMemo(
    () => new Set(loras.filter((l) => l.path).map((l) => l.path)),
    [loras],
  )

  const handleSlotChange = (i: number, picked: PickedLora | null, weight: number) => {
    const entry: LoraEntry = picked
      ? {
          path: picked.path,
          scale: weight,
          project_id: picked.projectId,
          version_id: picked.versionId,
        }
      : {
          // 反选：槽保留但 path 清空，picker 仍渲染（视觉等同初次打开的空槽）
          path: '',
          scale: weight,
          project_id: null,
          version_id: null,
        }
    onChange(loras.map((l, idx) => (idx === i ? entry : l)))
  }

  const handleSlotRemove = (i: number) => {
    onChange(loras.filter((_, idx) => idx !== i))
    if (externalForIdx === i) setExternalForIdx(null)
  }

  const handleAddSlot = () => {
    onChange([
      ...loras,
      { path: '', scale: 1.0, project_id: null, version_id: null },
    ])
  }

  return (
    <div className="flex flex-col gap-2">
      {loras.map((l, i) => {
        const hasCkpt = !!l.path
        // 决策 #8 / plan §3：历史回填后 resolve 失败的 LoRA（path='' && name 保留）
        // 渲染 ⚠ placeholder 卡片，提示用户重选；不要静默 path 空让用户困惑
        if (!hasCkpt && l.name) {
          return (
            <PlaceholderLoraCard
              key={`lora-${i}`}
              name={l.name}
              onPick={() => {
                // 清掉 name 让 InlineLoraPicker 出来供用户重选
                onChange(loras.map((lo, idx) => (
                  idx === i ? { ...lo, name: null } : lo
                )))
              }}
              onRemove={() => handleSlotRemove(i)}
              t={t}
            />
          )
        }
        return (
          <InlineLoraPicker
            // key 只用 index：避免 ckpt 切换 / 反选时整个 picker remount
            key={`lora-${i}`}
            mode="single"
            catalog={catalog}
            value={
              hasCkpt
                ? { path: l.path, projectId: l.project_id ?? null, versionId: l.version_id ?? null }
                : null
            }
            weight={l.scale}
            onChange={(p, w) => handleSlotChange(i, p, w)}
            onClose={() => handleSlotRemove(i)}
            onPickExternal={() => setExternalForIdx(i)}
          />
        )
      })}

      <AddSlotButton onClick={handleAddSlot}>+ 添加 LoRA</AddSlotButton>

      {/* placeholder 卡片渲染：path='' && name 保留 */}
      {externalForIdx !== null && (
        <PathPicker
          dirOnly={false}
          onPick={(p) => {
            const entry: LoraEntry = {
              path: p,
              scale: 1.0,
              project_id: null,
              version_id: null,
            }
            // 覆盖目标槽的内容；existingPaths 已排除空 path，外部文件可叠加
            void existingPaths
            onChange(loras.map((l, idx) => (idx === externalForIdx ? entry : l)))
            setExternalForIdx(null)
          }}
          onClose={() => setExternalForIdx(null)}
        />
      )}
    </div>
  )
}

/** 历史回填后 resolve 失败的 LoRA 槽渲染（决策 #8 / plan §3）。
 *  样式跟 InlineLoraPicker 一致（card 风格 + border），但内容显示 ⚠ + name +
 *  [重选] [移除]，不阻断 submit（path='' 会被过滤）。 */
function PlaceholderLoraCard({
  name, onPick, onRemove, t,
}: {
  name: string
  onPick: () => void
  onRemove: () => void
  t: (key: string) => string
}) {
  return (
    <div
      className="flex items-center gap-2"
      style={{
        border: '1px solid var(--border-warn, var(--border-subtle))',
        background: 'var(--bg-warn-soft, var(--bg-sunken))',
        borderRadius: 'var(--r-md)',
        padding: '8px 10px',
        fontSize: 12,
      }}
    >
      <span aria-hidden="true" style={{ flexShrink: 0 }}>⚠</span>
      <div className="flex-1 min-w-0 flex flex-col gap-0.5">
        <div className="font-mono truncate" style={{ fontSize: 11, color: 'var(--fg-secondary)' }}>
          {name}
        </div>
        <div className="text-2xs text-fg-tertiary">{t('generate.loraNotFoundHint')}</div>
      </div>
      <button
        type="button"
        onClick={onPick}
        className="font-mono"
        style={{
          border: '1px solid var(--border-subtle)',
          background: 'var(--bg-elevated)',
          borderRadius: 'var(--r-sm)',
          padding: '3px 8px',
          fontSize: 11,
          color: 'var(--fg-secondary)',
          cursor: 'pointer',
          flexShrink: 0,
        }}
      >
        {t('generate.repickLora')}
      </button>
      <button
        type="button"
        onClick={onRemove}
        title={t('common.delete')}
        aria-label={t('common.delete')}
        style={{
          width: 20, height: 20,
          display: 'grid', placeItems: 'center',
          borderRadius: 999,
          border: 0,
          background: 'transparent',
          color: 'var(--fg-tertiary)',
          cursor: 'pointer',
          fontSize: 14,
          lineHeight: 1,
          padding: 0,
          flexShrink: 0,
        }}
      >
        ×
      </button>
    </div>
  )
}
