// ExportBundleDialog — 选择 bundle.zip 导出内容后触发浏览器下载。
import { useState } from 'react'
import { useTranslation } from 'react-i18next'

export type BundleExportDestination = 'download' | 'data_exports'

export interface BundleExportOpts {
  train: boolean
  trainCaptions: boolean
  reg: boolean
  regCaptions: boolean
  includeConfig: boolean
  latentCache: boolean
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
  const [latentCache, setLatentCache] = useState(false)
  const [destination, setDestination] = useState<BundleExportDestination>('download')

  const nothingSelected = !train && !reg && !includeConfig

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (nothingSelected) return
    onConfirm({ train, trainCaptions, reg, regCaptions, includeConfig, latentCache, destination })
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/50"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel() }}
    >
      <form
        onSubmit={handleSubmit}
        className="bg-elevated border border-dim rounded-lg w-[90%] max-w-[420px] p-6 flex flex-col gap-5 shadow-xl"
      >
        <div>
          <h2 className="m-0 text-lg font-semibold text-fg-primary">
            {t('layout.exportBundleTitle')}
          </h2>
          <p className="mt-1 mb-0 text-sm text-fg-secondary">
            {t('layout.exportBundleDestinationHint')}
          </p>
        </div>

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
            <input
              type="checkbox"
              checked={train}
              onChange={(e) => setTrain(e.target.checked)}
            />
            <span className="text-sm text-fg-primary font-medium">
              {t('layout.exportBundleTrain')}
            </span>
          </label>
          {train && (
            <label className="flex items-center gap-2 cursor-pointer pl-5">
              <input
                type="checkbox"
                checked={trainCaptions}
                onChange={(e) => setTrainCaptions(e.target.checked)}
              />
              <span className="text-sm text-fg-secondary">
                {t('layout.exportBundleCaptions')}
              </span>
            </label>
          )}
        </div>

        {/* 正则集 */}
        <div className="flex flex-col gap-2">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={reg}
              onChange={(e) => setReg(e.target.checked)}
            />
            <span className="text-sm text-fg-primary font-medium">
              {t('layout.exportBundleReg')}
            </span>
          </label>
          {reg && (
            <label className="flex items-center gap-2 cursor-pointer pl-5">
              <input
                type="checkbox"
                checked={regCaptions}
                onChange={(e) => setRegCaptions(e.target.checked)}
              />
              <span className="text-sm text-fg-secondary">
                {t('layout.exportBundleCaptions')}
              </span>
            </label>
          )}
        </div>

        {/* VAE latent 缓存 — 跟着选中的 train/reg 素材走 */}
        {(train || reg) && (
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={latentCache}
              onChange={(e) => setLatentCache(e.target.checked)}
            />
            <span className="text-sm text-fg-primary font-medium">
              {t('layout.exportBundleLatentCache')}
            </span>
            <span className="text-xs text-fg-tertiary">{t('layout.exportBundleLatentCacheHint')}</span>
          </label>
        )}

        {/* 训练配置 */}
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
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

        <div className="flex gap-2 justify-end mt-1">
          <button type="button" onClick={onCancel} className="btn btn-secondary">
            {t('common.cancel')}
          </button>
          <button type="submit" className="btn btn-primary" disabled={nothingSelected}>
            {t('common.export')}
          </button>
        </div>
      </form>
    </div>
  )
}
