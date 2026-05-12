import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core'
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useRef } from 'react'
import type { LLMMessage } from '../api/client'

interface Props {
  messages: LLMMessage[]
  onChange: (msgs: LLMMessage[]) => void
  disabled?: boolean
}

/** Role 着色（与 design tokens 对齐，参见 LLM Settings redesign.html 的 .msg-role.* 规则）。 */
const roleStyles: Record<string, { bg: string; fg: string }> = {
  system:    { bg: 'var(--info-soft)',   fg: 'var(--info)' },
  user:      { bg: 'var(--accent-soft)', fg: 'var(--accent)' },
  assistant: { bg: 'var(--ok-soft)',     fg: 'var(--ok)' },
  image:     { bg: 'var(--warn-soft)',   fg: 'var(--warn)' },
}

const roleHint: Record<string, string> = {
  system:    '指令性 · 一般写一次',
  user:      'few-shot 示例 / 任务描述',
  assistant: 'few-shot 期望答案',
}

export default function LLMMessagesEditor({ messages, onChange, disabled }: Props) {
  const idRefs = useRef<WeakMap<LLMMessage, string>>(new WeakMap())
  const seq = useRef(0)
  const idOf = (m: LLMMessage): string => {
    let id = idRefs.current.get(m)
    if (!id) {
      seq.current += 1
      id = `msg-${seq.current}`
      idRefs.current.set(m, id)
    }
    return id
  }
  const ids = messages.map(idOf)

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event
    if (!over || active.id === over.id) return
    const oldIdx = ids.indexOf(active.id as string)
    const newIdx = ids.indexOf(over.id as string)
    if (oldIdx === -1 || newIdx === -1) return
    onChange(arrayMove(messages, oldIdx, newIdx))
  }

  const updateMsg = (i: number, patch: Partial<LLMMessage>) => {
    onChange(messages.map((m, idx) => (idx === i ? { ...m, ...patch } : m)))
  }

  const deleteMsg = (i: number) => {
    onChange(messages.filter((_, idx) => idx !== i))
  }

  const addMessage = () => {
    const last = messages[messages.length - 1]
    const nextRole: LLMMessage['role'] =
      !last || last.type === 'image' || last.role === 'system'
        ? 'user'
        : last.role === 'user'
          ? 'assistant'
          : 'user'
    onChange([...messages, { type: 'text', role: nextRole, content: '' }])
  }

  return (
    <div className="grid gap-2.5">
      <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
        <SortableContext items={ids} strategy={verticalListSortingStrategy}>
          {messages.map((m, i) => (
            <SortableMessage
              key={ids[i]}
              id={ids[i]}
              message={m}
              disabled={disabled}
              onChange={(patch) => updateMsg(i, patch)}
              onDelete={() => deleteMsg(i)}
            />
          ))}
        </SortableContext>
      </DndContext>
      <button
        type="button"
        onClick={addMessage}
        disabled={disabled}
        className="bg-transparent text-xs text-fg-tertiary hover:text-accent hover:bg-accent-soft hover:border-accent transition-colors py-2.5 px-3 flex items-center justify-center gap-1.5"
        style={{
          border: '1px dashed var(--border-default)',
          borderRadius: 'var(--r-md)',
          fontFamily: 'var(--font-mono)',
        }}
      >
        + 添加消息
      </button>
    </div>
  )
}

function SortableMessage({
  id,
  message,
  onChange,
  onDelete,
  disabled,
}: {
  id: string
  message: LLMMessage
  onChange: (patch: Partial<LLMMessage>) => void
  onDelete: () => void
  disabled?: boolean
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id })
  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
    background: 'var(--bg-sunken)',
    border: '1px solid var(--border-subtle)',
    borderRadius: 'var(--r-md)',
    overflow: 'hidden',
  }

  const isImage = message.type === 'image'
  const rolePillStyle = roleStyles[isImage ? 'image' : message.role] ?? roleStyles.user

  return (
    <div ref={setNodeRef} style={style}>
      {/* msg-h: header strip */}
      <div
        className="flex items-center gap-2.5 px-2.5 py-2"
        style={{
          background: 'rgba(255,255,255,0.02)',
          borderBottom: '1px solid var(--border-subtle)',
        }}
      >
        <button
          {...attributes}
          {...listeners}
          type="button"
          aria-label="拖动调整位置"
          disabled={disabled}
          className="cursor-grab text-fg-disabled select-none"
          style={{ fontFamily: 'var(--font-mono)', fontSize: 14 }}
        >
          ⋮⋮
        </button>

        {isImage ? (
          <span
            className="inline-flex items-center gap-1.5 text-2xs uppercase tracking-wider"
            style={{
              fontFamily: 'var(--font-mono)',
              padding: '3px 8px',
              borderRadius: 'var(--r-sm)',
              background: rolePillStyle.bg,
              color: rolePillStyle.fg,
              letterSpacing: '0.06em',
            }}
          >
            🖼 当前图片
          </span>
        ) : (
          <select
            value={message.role}
            onChange={(e) => onChange({ role: e.target.value as LLMMessage['role'] })}
            disabled={disabled}
            className="inline-flex items-center gap-1.5 text-2xs uppercase tracking-wider cursor-pointer border-0 outline-none"
            style={{
              fontFamily: 'var(--font-mono)',
              padding: '3px 8px',
              borderRadius: 'var(--r-sm)',
              background: rolePillStyle.bg,
              color: rolePillStyle.fg,
              letterSpacing: '0.06em',
            }}
          >
            <option value="system">system</option>
            <option value="user">user</option>
            <option value="assistant">assistant</option>
          </select>
        )}

        <span
          className="text-2xs text-fg-tertiary"
          style={{ fontFamily: 'var(--font-mono)', letterSpacing: '0.04em' }}
        >
          {isImage ? '打标时把图片塞入此位置 · 可拖动' : roleHint[message.role]}
        </span>

        {!isImage && (
          <button
            type="button"
            onClick={onDelete}
            disabled={disabled}
            className="ml-auto text-fg-tertiary hover:text-err hover:bg-err-soft border-0 bg-transparent"
            style={{ padding: '4px 6px', borderRadius: 'var(--r-sm)', fontSize: 12 }}
            title="删除消息"
            aria-label="删除消息"
          >
            ✕
          </button>
        )}
      </div>

      {/* msg body */}
      {isImage ? (
        <div className="flex items-center gap-3 px-3.5 py-3">
          <div
            className="grid place-items-center text-white"
            style={{
              width: 44,
              height: 44,
              borderRadius: 'var(--r-sm)',
              background:
                'radial-gradient(circle at 30% 30%, oklch(0.75 0.12 60), transparent 60%), radial-gradient(circle at 70% 70%, oklch(0.45 0.10 280), transparent 65%), var(--bg-overlay)',
              border: '1px solid var(--border-default)',
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
            }}
          >
            IMG
          </div>
          <div
            className="text-xs text-fg-tertiary"
            style={{ fontFamily: 'var(--font-mono)', lineHeight: 1.5 }}
          >
            <div>
              占位：<b className="font-medium text-fg-secondary">current_image</b>
            </div>
            <div>每次打标时被替换为正在处理的图片</div>
          </div>
        </div>
      ) : (
        <textarea
          value={message.content}
          onChange={(e) => onChange({ content: e.target.value })}
          disabled={disabled}
          rows={3}
          className="w-full bg-transparent border-0 outline-none text-sm text-fg-primary block"
          style={{
            padding: '12px 14px',
            fontFamily: 'var(--font-mono)',
            resize: 'none',
            lineHeight: 1.55,
            minHeight: 80,
          }}
          placeholder={
            message.role === 'system'
              ? '系统提示，例如：You are an image captioning assistant...'
              : message.role === 'user'
                ? '（可选）用户内容 — few-shot 示例 / 任务描述'
                : 'assistant 期望输出（few-shot 示例答案）'
          }
        />
      )}
    </div>
  )
}
