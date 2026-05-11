/** Task 状态徽章（pending / running / done / failed / canceled）。 */
export default function StatusBadge({ status }: { status: string }) {
  const cls =
    status === 'done'    ? 'badge badge-ok'
    : status === 'running'  ? 'badge badge-info'
    : status === 'failed'   ? 'badge badge-err'
    : status === 'canceled' ? 'badge'
    : 'badge'
  const label =
    status === 'done'    ? '已完成'
    : status === 'running'  ? '生成中'
    : status === 'failed'   ? '失败'
    : status === 'pending'  ? '排队中'
    : status === 'canceled' ? '已取消'
    : status
  return <span className={cls}>{label}</span>
}
