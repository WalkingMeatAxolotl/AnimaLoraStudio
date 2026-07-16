// familySwitch.ts —— 模型族切换动作（多模型 P4-3，Train / Presets 两页共享）。
//
// 翻 model_family 不是普通字段编辑：4 个权重路径要按目标族重算、族风味字段
// （sampler / scheduler / timestep 等）要重置、目标族不支持的能力字段要关闭。
// 两页在 SchemaForm onChange 拦截该字段的变化，经后端预览计算 + 确认对话框，
// 用户确认才应用（应用后走各页正常保存链路）。
import type { TFunction } from 'i18next'
import { api, type ConfigData, type FamilySwitchChange } from '../api/client'
import { fieldLabel } from './schema'

type ConfirmFn = (
  message: string,
  options?: { title?: string; okText?: string; cancelText?: string; tone?: 'default' | 'danger' | 'warn' },
) => Promise<boolean>

function formatValue(v: unknown, t: TFunction): string {
  if (v === null || v === undefined || v === '') return t('familySwitch.empty')
  if (typeof v === 'boolean') return v ? t('field.yes') : t('field.no')
  return String(v)
}

function formatChanges(changes: FamilySwitchChange[], t: TFunction): string {
  return changes
    .filter((c) => c.field !== 'model_family')
    .map((c) => `· ${fieldLabel(c.field)}: ${formatValue(c.from, t)} → ${formatValue(c.to, t)}`)
    .join('\n')
}

/**
 * 经后端预览 + 确认对话框执行族切换。
 *
 * @param target  目标族 id（用户在下拉里选的新值）
 * @param config  当前 config（切换前，model_family 仍是旧值）
 * @returns 确认后返回切换后的完整 config；用户取消或请求失败返回 null
 *          （调用方保持旧值不动）。
 */
export async function confirmFamilySwitch(
  target: string,
  config: ConfigData,
  confirm: ConfirmFn,
  t: TFunction,
): Promise<ConfigData | null> {
  const { config: switched, changes } = await api.switchModelFamily(target, config)
  const detail = formatChanges(changes, t)
  const ok = await confirm(
    detail
      ? t('familySwitch.message', { family: target, changes: detail })
      : t('familySwitch.messageNoChanges', { family: target }),
    {
      title: t('familySwitch.title'),
      okText: t('familySwitch.ok'),
      tone: 'warn',
    },
  )
  return ok ? switched : null
}
