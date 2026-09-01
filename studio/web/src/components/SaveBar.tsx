import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type CaptionSnapshot } from '../api/client'
import ActionGroup from './ActionGroup'
import Button from './Button'
import { useDialog } from './Dialog'
import { useToast } from './Toast'

interface Props {
  pid: number
  vid: number
  /** 待保存数：0 = 无 dirty。 */
  dirtyCount: number
  /** 触发保存：父组件提供 commit 实现（已经计算 diff）。 */
  onSave: () => Promise<void>
  /** 触发还原后，父组件需要重新拉缓存。 */
  onAfterRestore: () => Promise<void>
}

function fmtTime(epoch: number): string {
  const d = new Date(epoch * 1000)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
}

function fmtSize(b: number): string {
  if (b < 1024) return `${b} B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1024 / 1024).toFixed(1)} MB`
}

export default function SaveBar({
  pid, vid, dirtyCount, onSave, onAfterRestore,
}: Props) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const [open, setOpen] = useState(false)
  const [items, setItems] = useState<CaptionSnapshot[]>([])
  const [busyId, setBusyId] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const refresh = useCallback(async () => {
    try { setItems(await api.listCaptionSnapshots(pid, vid)) }
    catch (e) { toast(String(e), 'error') }
  }, [pid, vid, toast])

  useEffect(() => { if (open) void refresh() }, [open, refresh])

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  const save = async () => {
    setSaving(true)
    try { await onSave() } finally { setSaving(false) }
  }

  const restore = async (sid: string) => {
    if (!(await confirm(t('saveBar.confirmRestore'), { tone: 'warn', okText: t('common.restore') }))) return
    setBusyId(sid)
    try {
      const r = await api.restoreCaptionSnapshot(pid, vid, sid)
      toast(t('saveBar.restoreDone', { n: r.written, m: r.removed_old }), 'success')
      await onAfterRestore()
    } catch (e) { toast(String(e), 'error') }
    finally { setBusyId(null) }
  }

  const del = async (sid: string) => {
    if (!(await confirm(t('saveBar.confirmDeletePoint', { id: sid }), { tone: 'danger', okText: t('common.delete') }))) return
    setBusyId(sid)
    try { await api.deleteCaptionSnapshot(pid, vid, sid); await refresh() }
    catch (e) { toast(String(e), 'error') }
    finally { setBusyId(null) }
  }

  return (
    <div className="relative" ref={ref}>
      <ActionGroup
        secondary={(
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setOpen(!open)}
            aria-expanded={open}
            aria-haspopup="dialog"
          >
            {t('saveBar.restorePoints')}
          </Button>
        )}
        primary={(
          <Button
            variant={dirtyCount > 0 ? 'primary' : 'secondary'}
            size="sm"
            onClick={save}
            disabled={dirtyCount === 0}
            loading={saving}
            title={t('saveBar.tooltip')}
          >
            {saving
              ? t('common.saving')
              : dirtyCount > 0
                ? t('saveBar.save', { n: dirtyCount })
                : t('saveBar.saved')}
          </Button>
        )}
      />

      {open && (
        <div
          role="dialog"
          aria-label={t('saveBar.restorePoints')}
          className="absolute right-0 top-[calc(100%+4px)] w-80 max-h-80 overflow-y-auto rounded-md border border-subtle bg-elevated shadow-xl z-30"
        >
          {items.length === 0 ? (
            <p className="px-3.5 py-3 text-xs text-fg-tertiary m-0">
              {t('saveBar.noRestorePoints')}
            </p>
          ) : (
            <ul className="list-none p-0 m-0">
              {items.map((s) => (
                <li
                  key={s.id}
                  className="px-3 py-2 text-xs flex items-center gap-2 border-b border-subtle"
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-fg-primary font-mono">
                      {fmtTime(s.created_at)}
                    </div>
                    <div className="text-fg-tertiary text-[10px] mt-0.5">
                      {t('saveBar.restoreEntry', { n: s.file_count, size: fmtSize(s.size) })}
                    </div>
                  </div>
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={() => restore(s.id)}
                    disabled={busyId === s.id}
                  >
                    {t('common.restore')}
                  </Button>
                  <Button
                    variant="ghost"
                    size="xs"
                    iconOnly
                    onClick={() => del(s.id)}
                    disabled={busyId === s.id}
                    className="text-fg-tertiary hover:text-err"
                    aria-label={t('common.delete')}
                  >
                    ×
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
