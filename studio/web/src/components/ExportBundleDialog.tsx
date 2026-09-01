// ExportBundleDialog — 选择 bundle.zip 导出内容后触发浏览器下载。
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import ActionGroup from './ActionGroup'
import Button from './Button'
import { Checkbox } from './FormControl'
import Modal from './Modal'

export type BundleExportDestination = 'download' | 'data_exports'

export interface BundleExportOpts {
  train: boolean
  trainCaptions: boolean
  reg: boolean
  regCaptions: boolean
  includeConfig: boolean
  trainLatentCache: boolean
  regLatentCache: boolean
  trainMasks: boolean
  destination: BundleExportDestination
}

interface Props {
  onConfirm: (opts: BundleExportOpts) => void
  onCancel: () => void
}

export default function ExportBundleDialog({ onConfirm, onCancel }: Props) {
  const { t } = useTranslation()
  const [train, setTrain] = useState(true)
  const [trainCaptions, setTrainCaptions] = useState(true)
  const [reg, setReg] = useState(false)
  const [regCaptions, setRegCaptions] = useState(false)
  const [includeConfig, setIncludeConfig] = useState(false)
  const [trainLatentCache, setTrainLatentCache] = useState(false)
  const [regLatentCache, setRegLatentCache] = useState(false)
  const [trainMasks, setTrainMasks] = useState(false)
  const [destination, setDestination] = useState<BundleExportDestination>('download')

  const nothingSelected = !train && !reg && !includeConfig

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (nothingSelected) return
    onConfirm({
      train, trainCaptions, reg, regCaptions, includeConfig,
      trainLatentCache, regLatentCache, trainMasks, destination,
    })
  }

  return (
    <Modal
      as="form"
      title={t('layout.exportBundleTitle')}
      description={t('layout.exportBundleDestinationHint')}
      onClose={onCancel}
      onSubmit={handleSubmit}
      size="sm"
      bodyClassName="flex flex-col gap-section"
      footer={(
        <ActionGroup
          secondary={(
            <Button type="button" variant="secondary" onClick={onCancel}>
              {t('common.cancel')}
            </Button>
          )}
          primary={(
            <Button type="submit" variant="primary" disabled={nothingSelected}>
              {t('common.export')}
            </Button>
          )}
        />
      )}
    >
        <div className="flex flex-col gap-2">
          <div className="text-sm font-medium text-fg-primary">
            {t('layout.exportBundleDestination')}
          </div>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="bundle-export-destination"
              checked={destination === 'download'}
              onChange={() => setDestination('download')}
            />
            <span className="text-sm text-fg-secondary">
              {t('layout.exportBundleDownload')}
            </span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="bundle-export-destination"
              checked={destination === 'data_exports'}
              onChange={() => setDestination('data_exports')}
            />
            <span className="text-sm text-fg-secondary">
              {t('layout.exportBundleDataExports')}
            </span>
          </label>
        </div>

        {/* 训练集 */}
        <div className="flex flex-col gap-2">
          <label className="flex items-center gap-2 cursor-pointer">
            <Checkbox
              checked={train}
              onChange={(e) => setTrain(e.target.checked)}
            />
            <span className="text-sm text-fg-primary font-medium">
              {t('layout.exportBundleTrain')}
            </span>
          </label>
          {train && (
            <>
              <label className="flex items-center gap-2 cursor-pointer pl-5">
                <Checkbox
                  checked={trainCaptions}
                  onChange={(e) => setTrainCaptions(e.target.checked)}
                />
                <span className="text-sm text-fg-secondary">
                  {t('layout.exportBundleCaptions')}
                </span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer pl-5">
                <Checkbox
                  checked={trainLatentCache}
                  onChange={(e) => setTrainLatentCache(e.target.checked)}
                />
                <span className="text-sm text-fg-secondary">
                  {t('layout.exportBundleLatentCache')}
                </span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer pl-5">
                <Checkbox
                  checked={trainMasks}
                  onChange={(e) => setTrainMasks(e.target.checked)}
                />
                <span className="text-sm text-fg-secondary">
                  {t('layout.exportBundleTrainMasks')}
                </span>
              </label>
            </>
          )}
        </div>

        {/* 正则集 */}
        <div className="flex flex-col gap-2">
          <label className="flex items-center gap-2 cursor-pointer">
            <Checkbox
              checked={reg}
              onChange={(e) => setReg(e.target.checked)}
            />
            <span className="text-sm text-fg-primary font-medium">
              {t('layout.exportBundleReg')}
            </span>
          </label>
          {reg && (
            <>
              <label className="flex items-center gap-2 cursor-pointer pl-5">
                <Checkbox
                  checked={regCaptions}
                  onChange={(e) => setRegCaptions(e.target.checked)}
                />
                <span className="text-sm text-fg-secondary">
                  {t('layout.exportBundleCaptions')}
                </span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer pl-5">
                <Checkbox
                  checked={regLatentCache}
                  onChange={(e) => setRegLatentCache(e.target.checked)}
                />
                <span className="text-sm text-fg-secondary">
                  {t('layout.exportBundleLatentCache')}
                </span>
              </label>
            </>
          )}
        </div>

        {/* 训练配置 */}
        <label className="flex items-center gap-2 cursor-pointer">
          <Checkbox
            checked={includeConfig}
            onChange={(e) => setIncludeConfig(e.target.checked)}
          />
          <span className="text-sm text-fg-primary font-medium">
            {t('layout.exportBundleConfig')}
          </span>
          <span className="text-xs text-fg-tertiary">{t('layout.exportBundleConfigHint')}</span>
        </label>

        {nothingSelected && (
          <p className="text-xs text-err m-0">{t('layout.exportBundleAtLeastOne')}</p>
        )}
    </Modal>
  )
}
