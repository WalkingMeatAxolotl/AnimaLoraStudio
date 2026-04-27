// 与 FastAPI 守护进程交互的薄封装。
// 开发时由 Vite proxy 转发到 127.0.0.1:8765；生产部署时与 API 同源。

export interface HealthResponse {
  status: string
  version: string
}

export interface SchemaProperty {
  type?: string | string[]
  default?: unknown
  description?: string
  enum?: unknown[]
  minimum?: number
  maximum?: number
  exclusiveMinimum?: number
  exclusiveMaximum?: number
  group?: string
  control?: string
  cli_alias?: string
  show_when?: string
  anyOf?: Array<{ type?: string }>
  items?: SchemaProperty
}

export interface JsonSchema {
  properties: Record<string, SchemaProperty>
  required?: string[]
}

export interface SchemaResponse {
  schema: JsonSchema
  groups: Array<{ key: string; label: string }>
}

export interface ConfigSummary {
  name: string
  path: string
  updated_at: number
}

export type ConfigData = Record<string, unknown>

async function req<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const resp = await fetch(path, {
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
    },
    ...init,
  })
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`
    try {
      const body = await resp.json()
      if (body?.detail) detail = body.detail
    } catch {
      // ignore
    }
    throw new Error(detail)
  }
  if (resp.status === 204) return undefined as T
  return (await resp.json()) as T
}

export const api = {
  health: () => req<HealthResponse>('/api/health'),
  state: () => req<Record<string, unknown>>('/api/state'),

  schema: () => req<SchemaResponse>('/api/schema'),

  listConfigs: () =>
    req<{ items: ConfigSummary[] }>('/api/configs').then((r) => r.items),
  getConfig: (name: string) => req<ConfigData>(`/api/configs/${name}`),
  saveConfig: (name: string, data: ConfigData) =>
    req<{ name: string; path: string }>(`/api/configs/${name}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  deleteConfig: (name: string) =>
    req<{ deleted: string }>(`/api/configs/${name}`, { method: 'DELETE' }),
  duplicateConfig: (src: string, newName: string) =>
    req<{ name: string; path: string }>(
      `/api/configs/${src}/duplicate`,
      { method: 'POST', body: JSON.stringify({ new_name: newName }) }
    ),
}
