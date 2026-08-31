import type { LoraEntry } from '../../../api/client'

export interface LoraUiState {
  id: string
  enabled: boolean
}

let idCounter = 0

export function createLoraUiState(enabled = true): LoraUiState {
  idCounter += 1
  return {
    id: `lora-${Date.now().toString(36)}-${idCounter.toString(36)}`,
    enabled,
  }
}

/** Align the UI-only sidecar with persisted LoRA entries without leaking UI fields to the API. */
export function normalizeLoraUi(
  loras: LoraEntry[],
  raw: unknown,
): LoraUiState[] {
  const candidates = Array.isArray(raw) ? raw : []
  const used = new Set<string>()
  return loras.map((_, index) => {
    const candidate = candidates[index] as Partial<LoraUiState> | undefined
    const id = typeof candidate?.id === 'string' && candidate.id.trim() && !used.has(candidate.id)
      ? candidate.id
      : createLoraUiState().id
    used.add(id)
    return { id, enabled: typeof candidate?.enabled === 'boolean' ? candidate.enabled : true }
  })
}

export function enabledLoras(loras: LoraEntry[], ui: LoraUiState[]): LoraEntry[] {
  return loras.filter((lora, index) => ui[index]?.enabled !== false && lora.path.trim())
}

export function normalizeLoraPath(path: string): string {
  const trimmed = path.trim()
  const windowsStyle = /^[a-z]:[\\/]/i.test(trimmed) || trimmed.includes('\\')
  const normalized = trimmed.replace(/\\/g, '/').replace(/\/+$/, '')
  return windowsStyle ? normalized.toLocaleLowerCase() : normalized
}

export function loraTextName(entry: LoraEntry): string {
  const source = entry.path || entry.name || ''
  const basename = source.replace(/\\/g, '/').split('/').pop() ?? source
  return basename.replace(/\.safetensors$/i, '')
}

export function serializeLoraText(loras: LoraEntry[], ui: LoraUiState[]): string {
  return loras
    .filter((_, index) => ui[index]?.enabled !== false)
    .map((lora) => `<lora:${loraTextName(lora)}:${Number(lora.scale).toString()}>`)
    .join('\n')
}

export type LoraTextErrorCode = 'invalid' | 'unknown' | 'ambiguous' | 'duplicate'

export class LoraTextError extends Error {
  constructor(public readonly code: LoraTextErrorCode, public readonly value = '') {
    super(`${code}:${value}`)
  }
}

/** Apply text only to existing structured entries. Omitted entries are disabled, never deleted. */
export function applyLoraText(
  text: string,
  loras: LoraEntry[],
  ui: LoraUiState[],
): { loras: LoraEntry[]; ui: LoraUiState[] } {
  const trimmed = text.trim()
  if (!trimmed) {
    return { loras, ui: ui.map((state) => ({ ...state, enabled: false })) }
  }

  const tokens: Array<{ name: string; scale: number }> = []
  const regex = /<lora:([^:<>]+):([^:<>]+)>/g
  let cursor = 0
  for (const match of trimmed.matchAll(regex)) {
    const index = match.index ?? 0
    if (trimmed.slice(cursor, index).trim()) throw new LoraTextError('invalid')
    const name = match[1].trim()
    const scale = Number(match[2].trim())
    if (!name || !Number.isFinite(scale)) throw new LoraTextError('invalid', match[0])
    tokens.push({ name, scale })
    cursor = index + match[0].length
  }
  if (tokens.length === 0 || trimmed.slice(cursor).trim()) throw new LoraTextError('invalid')

  const seen = new Set<string>()
  const updates = new Map<number, number>()
  for (const token of tokens) {
    const key = token.name.toLocaleLowerCase()
    if (seen.has(key)) throw new LoraTextError('duplicate', token.name)
    seen.add(key)
    const matches = loras
      .map((entry, index) => ({ index, name: loraTextName(entry).toLocaleLowerCase() }))
      .filter((entry) => entry.name === key)
    if (matches.length === 0) throw new LoraTextError('unknown', token.name)
    if (matches.length > 1) throw new LoraTextError('ambiguous', token.name)
    updates.set(matches[0].index, token.scale)
  }

  return {
    loras: loras.map((entry, index) => (
      updates.has(index) ? { ...entry, scale: updates.get(index)! } : entry
    )),
    ui: ui.map((state, index) => ({ ...state, enabled: updates.has(index) })),
  }
}
