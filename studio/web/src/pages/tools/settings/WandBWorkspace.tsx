import { useRef } from 'react'
import { Trans, useTranslation } from 'react-i18next'
import type { WandBConfig, WandBPreset } from '../../../api/client'
import { Caption, PBtn } from '../../../components/LLMTaggerWorkspace'
import { textInputClass } from './constants'
import { Bool, SensitiveInput, SettingsField, SettingsInput } from './fields'

/**
 * WandB 预设工作区 —— 骨架与 LLMTaggerWorkspace 完全同款：
 * 卡片(header 标题 + PresetBar 预设条 + 主体)。区别只有两点：
 * header 右侧带全局 enabled 总开关；主体简化为单 section(wandb 字段少,
 * 不需要 LLM 那种双栏)。预设条右侧多导出/导入(导出不含 API Key)。
 */
export default function WandBWorkspace({
  title,
  config,
  serverPresets,
  currentPreset,
  onToggleEnabled,
  onSelectPreset,
  onUpdatePreset,
  onAddPreset,
  onSaveAs,
  onDeletePreset,
  onExport,
  onImportFile,
}: {
  title: string
  config: WandBConfig
  serverPresets: WandBPreset[]
  currentPreset: WandBPreset
  onToggleEnabled: (v: boolean) => void
  onSelectPreset: (id: string) => void
  onUpdatePreset: <K extends keyof WandBPreset>(field: K, value: WandBPreset[K]) => void
  onAddPreset: () => void
  onSaveAs: () => void
  onDeletePreset: () => void
  onExport: () => void
  onImportFile: (f: File) => void
}) {
  const { t } = useTranslation()
  const importRef = useRef<HTMLInputElement | null>(null)
  const serverPreset = serverPresets.find((p) => p.id === currentPreset.id)

  return (
    <div
      className="bg-surface border border-subtle"
      style={{ borderRadius: 'var(--r-lg)', overflow: 'hidden' }}
    >
      {/* header：标题左 + 全局总开关右(enabled 不属于任何 preset) */}
      <div
        className="flex items-center justify-between gap-3"
        style={{ padding: '14px 16px', borderBottom: '1px solid var(--border-subtle)' }}
      >
        <h2 className="text-sm font-semibold text-fg-primary m-0">{title}</h2>
        <label
          className="flex items-center gap-2 text-xs text-fg-secondary cursor-pointer shrink-0"
          title={t('settings.enableWandbHint')}
        >
          <span className="whitespace-nowrap">{t('settings.enableWandb')}</span>
          <Bool value={config.enabled} onChange={onToggleEnabled} />
        </label>
      </div>

      {/* PresetBar：几何与 LLMTaggerWorkspace.PresetBar 一致 */}
      <div
        style={{
          padding: '10px 12px 10px 16px',
          display: 'grid',
          gridTemplateColumns: '1fr auto',
          alignItems: 'center',
          gap: 12,
          borderBottom: '1px solid var(--border-subtle)',
        }}
      >
        <div className="flex items-center gap-3.5 min-w-0">
          <Caption>{t('settings.wandbPreset')}</Caption>
          <div
            className="flex items-center gap-2.5 cursor-pointer"
            style={{
              background: 'var(--bg-sunken)',
              border: '1px solid var(--border-default)',
              borderRadius: 'var(--r-md)',
              padding: '7px 14px 7px 12px',
              fontSize: 'var(--t-sm)',
              color: 'var(--fg-primary)',
              fontWeight: 500,
              minWidth: 260,
              position: 'relative',
            }}
          >
            {/* 用 select 覆盖整个 pill 让用户能切换；select 透明 */}
            <select
              value={currentPreset.id}
              onChange={(e) => onSelectPreset(e.target.value)}
              className="absolute inset-0 opacity-0 cursor-pointer"
              aria-label={t('settings.wandbPreset')}
            >
              {config.presets.map((p) => (
                <option key={p.id} value={p.id}>{p.label || p.id}</option>
              ))}
            </select>
            <span className="truncate">{currentPreset.label || currentPreset.id}</span>
            <span className="ml-auto" style={{ color: 'var(--fg-tertiary)' }}>▾</span>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <PBtn onClick={onSaveAs}>{t('settings.wandbPresetSaveAs')}</PBtn>
          <PBtn onClick={onAddPreset}>{t('settings.wandbPresetNew')}</PBtn>
          {config.presets.length > 1 && (
            <PBtn variant="danger" onClick={onDeletePreset}>{t('common.delete')}</PBtn>
          )}
          <span style={{ width: 1, height: 18, background: 'var(--border-subtle)' }} />
          <PBtn onClick={onExport} title={t('settings.wandbExportKeyTitle')}>
            {t('settings.wandbPresetExport')}
          </PBtn>
          <PBtn onClick={() => importRef.current?.click()}>{t('settings.wandbPresetImport')}</PBtn>
          <input
            ref={importRef}
            type="file"
            accept=".yaml,.yml,.json"
            style={{ display: 'none' }}
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) onImportFile(f)
              if (importRef.current) importRef.current.value = ''
            }}
          />
        </div>
      </div>

      {/* 主体：单 section（wandb 字段少，不需要 LLM 的双栏 grid） */}
      <div style={{ padding: '12px 16px 14px' }}>
        <p className="m-0 mb-3 text-xs text-fg-tertiary">{t('settings.wandbPresetHint')}</p>

        <SettingsField label={t('settings.fieldApiKey')}>
          <SensitiveInput
            value={currentPreset.api_key}
            serverValue={serverPreset?.api_key ?? ''}
            onChange={(v) => onUpdatePreset('api_key', v)}
          />
        </SettingsField>
        <SettingsField label={t('settings.fieldProject')}>
          <SettingsInput
            type="text"
            value={currentPreset.project}
            onChange={(v) => onUpdatePreset('project', v)}
            placeholder="AnimaLoraStudio"
            className={textInputClass}
          />
        </SettingsField>
        <SettingsField label={t('settings.fieldEntity')} desc={t('settings.wandbEntityHint')}>
          <SettingsInput
            type="text"
            value={currentPreset.entity}
            onChange={(v) => onUpdatePreset('entity', v)}
            className={textInputClass}
          />
        </SettingsField>
        <SettingsField label={t('settings.fieldBaseUrl')} desc={t('settings.wandbBaseUrlHint')}>
          <SettingsInput
            type="text"
            value={currentPreset.base_url}
            onChange={(v) => onUpdatePreset('base_url', v)}
            placeholder="https://api.wandb.ai"
            className={textInputClass}
          />
        </SettingsField>
        <SettingsField label={t('settings.fieldMode')}>
          <select
            value={currentPreset.mode}
            onChange={(e) => onUpdatePreset('mode', e.target.value as WandBPreset['mode'])}
            className={`${textInputClass} max-w-32`}
          >
            <option value="online">online</option>
            <option value="offline">offline</option>
            <option value="disabled">disabled</option>
          </select>
        </SettingsField>
        <SettingsField
          label={t('settings.logSamples')}
          helpTooltip={
            <p><Trans i18nKey="settings.logSamplesHelp" components={{ code: <code /> }} /></p>
          }
        >
          <Bool value={currentPreset.log_samples} onChange={(v) => onUpdatePreset('log_samples', v)} />
        </SettingsField>
        {currentPreset.log_samples && (
          <div className="grid grid-cols-2 gap-3">
            <SettingsField
              label={t('settings.sampleMaxSide')}
              helpTooltip={<p>{t('settings.sampleMaxSideHelp')}</p>}
            >
              <SettingsInput
                type="number"
                min={64}
                step={64}
                value={currentPreset.sample_max_side}
                onChange={(v) => onUpdatePreset('sample_max_side', Math.max(64, parseInt(v) || 1216))}
                className={textInputClass}
              />
            </SettingsField>
            <SettingsField
              label={t('settings.sampleEveryNSteps')}
              helpTooltip={
                <p><Trans i18nKey="settings.sampleEveryNStepsHelp" components={{ code: <code /> }} /></p>
              }
            >
              <SettingsInput
                type="number"
                min={0}
                step={50}
                value={currentPreset.sample_every_n_steps}
                onChange={(v) => onUpdatePreset('sample_every_n_steps', Math.max(0, parseInt(v) || 0))}
                className={textInputClass}
              />
            </SettingsField>
          </div>
        )}

        {/* artifact 上传与采样图无关，不随 log_samples 隐藏 */}
        <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mt-4 mb-2">
          {t('settings.uploadArtifacts')}
        </h4>
        <SettingsField
          label={t('settings.uploadModel')}
          helpTooltip={<p>{t('settings.uploadModelHelp')}</p>}
        >
          <div className="flex items-center gap-3">
            <Bool value={currentPreset.upload_model} onChange={(v) => onUpdatePreset('upload_model', v)} />
            {currentPreset.upload_model && (
              <select
                value={currentPreset.upload_model_policy}
                onChange={(e) => onUpdatePreset('upload_model_policy', e.target.value as 'all' | 'last')}
                className={textInputClass + ' max-w-32'}
              >
                <option value="last">{t('settings.policyLast')}</option>
                <option value="all">{t('settings.policyAll')}</option>
              </select>
            )}
          </div>
        </SettingsField>
        <SettingsField
          label={t('settings.uploadStateManual')}
          helpTooltip={<p>{t('settings.uploadStateManualHelp')}</p>}
        >
          <div className="flex items-center gap-3">
            <Bool value={currentPreset.upload_state_manual} onChange={(v) => onUpdatePreset('upload_state_manual', v)} />
            {currentPreset.upload_state_manual && (
              <select
                value={currentPreset.upload_state_manual_policy}
                onChange={(e) => onUpdatePreset('upload_state_manual_policy', e.target.value as 'all' | 'last')}
                className={textInputClass + ' max-w-32'}
              >
                <option value="last">{t('settings.policyLast')}</option>
                <option value="all">{t('settings.policyAll')}</option>
              </select>
            )}
          </div>
        </SettingsField>
        <SettingsField
          label={t('settings.uploadStateAuto')}
          helpTooltip={<p>{t('settings.uploadStateAutoHelp')}</p>}
        >
          <div className="flex items-center gap-3">
            <Bool value={currentPreset.upload_state_auto} onChange={(v) => onUpdatePreset('upload_state_auto', v)} />
            {currentPreset.upload_state_auto && (
              <select
                value={currentPreset.upload_state_auto_policy}
                onChange={(e) => onUpdatePreset('upload_state_auto_policy', e.target.value as 'all' | 'last')}
                className={textInputClass + ' max-w-32'}
              >
                <option value="last">{t('settings.policyLast')}</option>
                <option value="all">{t('settings.policyAll')}</option>
              </select>
            )}
          </div>
        </SettingsField>
      </div>
    </div>
  )
}
