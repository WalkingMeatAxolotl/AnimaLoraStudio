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

export type TaskStatus = 'pending' | 'running' | 'done' | 'failed' | 'canceled'

export interface Task {
  id: number
  name: string
  config_name: string
  status: TaskStatus
  priority: number
  created_at: number
  started_at: number | null
  finished_at: number | null
  pid: number | null
  exit_code: number | null
  output_dir: string | null
  error_msg: string | null
}

export interface LogResponse {
  task_id: number
  content: string
  size: number
}

export interface DatasetFolder {
  name: string
  label: string
  repeat: number
  image_count: number
  caption_types: { json: number; txt: number; none: number }
  samples: string[]
  path: string
}

export interface DatasetScan {
  root: string
  exists: boolean
  folders: DatasetFolder[]
  total_images?: number
  weighted_steps_per_epoch?: number
}

export interface QueueExport {
  version: number
  exported_at: number
  tasks: Array<{
    name: string
    config_name: string
    priority: number
    config: Record<string, unknown> | null
  }>
}

export interface ImportResult {
  imported_count: number
  task_ids: number[]
  renamed: Record<string, string>
}

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

  // Queue --------------------------------------------------------------
  listQueue: (status?: TaskStatus) => {
    const qs = status ? `?status=${status}` : ''
    return req<{ items: Task[] }>(`/api/queue${qs}`).then((r) => r.items)
  },
  getTask: (id: number) => req<Task>(`/api/queue/${id}`),
  enqueue: (payload: { config_name: string; name?: string; priority?: number }) =>
    req<Task>('/api/queue', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  cancelTask: (id: number) =>
    req<{ task_id: number; canceled: boolean }>(`/api/queue/${id}/cancel`, {
      method: 'POST',
    }),
  retryTask: (id: number) =>
    req<Task>(`/api/queue/${id}/retry`, { method: 'POST' }),
  deleteTask: (id: number) =>
    req<{ deleted: number }>(`/api/queue/${id}`, { method: 'DELETE' }),
  reorderQueue: (orderedIds: number[]) =>
    req<{ reordered: number }>('/api/queue/reorder', {
      method: 'POST',
      body: JSON.stringify({ ordered_ids: orderedIds }),
    }),
  getLog: (id: number) => req<LogResponse>(`/api/logs/${id}`),

  // Queue import / export ---------------------------------------------
  exportQueue: (ids?: number[]) => {
    const qs = ids && ids.length ? `?ids=${ids.join(',')}` : ''
    return req<QueueExport>(`/api/queue/export${qs}`)
  },
  importQueue: (payload: unknown) =>
    req<ImportResult>('/api/queue/import', {
      method: 'POST',
      body: JSON.stringify({ payload }),
    }),

  // Datasets -----------------------------------------------------------
  listDatasets: (path?: string) => {
    const qs = path ? `?path=${encodeURIComponent(path)}` : ''
    return req<DatasetScan>(`/api/datasets${qs}`)
  },
  thumbnailUrl: (folder: string, name: string) =>
    `/api/datasets/thumbnail?folder=${encodeURIComponent(folder)}&name=${encodeURIComponent(name)}`,

  // Browse -------------------------------------------------------------
  browse: (path?: string) => {
    const qs = path ? `?path=${encodeURIComponent(path)}` : ''
    return req<BrowseResult>(`/api/browse${qs}`)
  },
}

export interface BrowseEntry {
  name: string
  type: 'dir' | 'file'
}

export interface BrowseResult {
  path: string
  parent: string | null
  entries: BrowseEntry[]
}
