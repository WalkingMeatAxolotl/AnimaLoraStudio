import { useTranslation } from 'react-i18next'

type QueueSectionHeaderProps = {
  variant: 'task' | 'job'
  sectionKey: string
  title: string
  count: number
}

/**
 * Queue section title and low-emphasis column guide share the row grid, so the
 * guide improves wide-screen scanning without adding another vertical band.
 */
export default function QueueSectionHeader({
  variant, sectionKey, title, count,
}: QueueSectionHeaderProps) {
  const { t } = useTranslation()
  const gridClass = variant === 'task' ? 'ui-queue-task-grid' : 'ui-queue-job-grid'

  return (
    <div
      className={`${gridClass} ui-queue-section-header px-[22px] grid gap-3 items-end`}
      data-testid={`queue-${variant}-section-header-${sectionKey}`}
    >
      <h3 className="type-section-label col-span-2 min-w-0 truncate">
        {title} ({count})
      </h3>
      {variant === 'task' ? (
        <>
          <span aria-hidden="true" className="ui-queue-column-label type-section-label">
            {t('queue.columns.type')}
          </span>
          <span aria-hidden="true" className="ui-queue-column-label type-section-label text-center">
            {t('queue.columns.status')}
          </span>
          <span aria-hidden="true" className="ui-queue-column-label type-section-label">
            {t('queue.columns.result')}
          </span>
          <span aria-hidden="true" className="ui-queue-column-label ui-queue-task-timing type-section-label text-right">
            {t('queue.columns.time')}
          </span>
          <span aria-hidden="true" className="ui-queue-column-label type-section-label text-right">
            {t('queue.columns.actions')}
          </span>
        </>
      ) : (
        <>
          <span aria-hidden="true" className="ui-queue-column-label type-section-label text-center">
            {t('queue.columns.status')}
          </span>
          <span aria-hidden="true" className="ui-queue-column-label ui-queue-job-timing type-section-label text-right">
            {t('queue.columns.durationTime')}
          </span>
          <span aria-hidden="true" className="ui-queue-column-label type-section-label text-right">
            {t('queue.columns.actions')}
          </span>
        </>
      )}
    </div>
  )
}
