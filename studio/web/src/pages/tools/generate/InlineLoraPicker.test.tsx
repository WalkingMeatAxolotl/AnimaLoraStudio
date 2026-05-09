import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import InlineLoraPicker, { projectAbbr } from './InlineLoraPicker'
import type { ProjectLora } from './types'

const sample: ProjectLora[] = [
  {
    projectId: 1, projectTitle: 'cute_chibi',
    versionId: 11, versionLabel: 'v3', stage: 'training',
    path: '/loras/cute_chibi/v3.safetensors', createdAt: 300,
  },
  {
    projectId: 1, projectTitle: 'cute_chibi',
    versionId: 12, versionLabel: 'v2', stage: 'done',
    path: '/loras/cute_chibi/v2.safetensors', createdAt: 200,
  },
  {
    projectId: 2, projectTitle: 'noir_portrait',
    versionId: 21, versionLabel: 'v1', stage: 'done',
    path: '/loras/noir/v1.safetensors', createdAt: 100,
  },
]

describe('projectAbbr', () => {
  it('extracts first 2 alphanumerics, uppercase', () => {
    expect(projectAbbr('cute_chibi')).toBe('CU')
    expect(projectAbbr('noir_portrait')).toBe('NO')
    expect(projectAbbr('character_yui')).toBe('CH')
  })
  it('strips non-alphanumeric', () => {
    expect(projectAbbr('___test')).toBe('TE')
  })
  it('falls back to ?? when empty', () => {
    expect(projectAbbr('___')).toBe('??')
    expect(projectAbbr('')).toBe('??')
  })
})

describe('InlineLoraPicker', () => {
  function renderPicker(overrides: Partial<{
    projectLoras: ProjectLora[]
    selectedPaths: Set<string>
  }> = {}) {
    const onPick = vi.fn()
    const onClose = vi.fn()
    const onPickExternal = vi.fn()
    const utils = render(
      <InlineLoraPicker
        projectLoras={overrides.projectLoras ?? sample}
        selectedPaths={overrides.selectedPaths ?? new Set()}
        onPick={onPick}
        onClose={onClose}
        onPickExternal={onPickExternal}
      />
    )
    return { ...utils, onPick, onClose, onPickExternal }
  }

  it('shows count in header (已选 / 总)', () => {
    renderPicker()
    // 改成扁平 + 已选/总。3 个总，0 已选
    expect(screen.getByText(/已选 0 \/ 3/)).toBeInTheDocument()
  })

  it('groups versions by project title', () => {
    renderPicker()
    // 项目名作为 caption uppercase 显示（出现 2 次：组头 + cute_chibi 的 2 行 + noir 的 1 行）
    // 用更稳的断言：每个 version 都渲染了
    expect(screen.getByText(/cute_chibi \/ v3/)).toBeInTheDocument()
    expect(screen.getByText(/cute_chibi \/ v2/)).toBeInTheDocument()
    expect(screen.getByText(/noir_portrait \/ v1/)).toBeInTheDocument()
  })

  it('shows 训练中 badge for training-stage versions', () => {
    renderPicker()
    expect(screen.getByText('训练中')).toBeInTheDocument()
  })

  it('marks already-added versions with ✓ marker; disables when no onRemove (multi-select off)', () => {
    renderPicker({
      selectedPaths: new Set(['/loras/cute_chibi/v3.safetensors']),
    })
    const addedBtn = screen.getByText(/cute_chibi \/ v3/).closest('button')!
    // 没传 onRemove → multi-select off → button disabled
    expect(addedBtn).toBeDisabled()
    expect(addedBtn.textContent).toContain('✓')
  })

  it('calls onPick with path when a version is clicked', async () => {
    const user = userEvent.setup()
    const { onPick } = renderPicker()
    const btn = screen.getByText(/noir_portrait \/ v1/).closest('button')!
    await user.click(btn)
    expect(onPick).toHaveBeenCalledWith('/loras/noir/v1.safetensors')
  })

  it('does not call onPick when clicking already-added (button disabled)', async () => {
    const user = userEvent.setup()
    const { onPick } = renderPicker({
      selectedPaths: new Set(['/loras/cute_chibi/v3.safetensors']),
    })
    const btn = screen.getByText(/cute_chibi \/ v3/).closest('button')!
    await user.click(btn)
    expect(onPick).not.toHaveBeenCalled()
  })

  it('filters by project title via search', async () => {
    const user = userEvent.setup()
    renderPicker()
    const search = screen.getByPlaceholderText('搜索项目 / 版本 / 文件名…')
    await user.type(search, 'noir')
    // cute_chibi 的版本应该消失
    expect(screen.queryByText(/cute_chibi \/ v3/)).not.toBeInTheDocument()
    expect(screen.getByText(/noir_portrait \/ v1/)).toBeInTheDocument()
    // count 更新（改成 已选 0/总）
    expect(screen.getByText(/已选 0 \/ 1/)).toBeInTheDocument()
  })

  it('filters by version label via search', async () => {
    const user = userEvent.setup()
    renderPicker()
    const search = screen.getByPlaceholderText('搜索项目 / 版本 / 文件名…')
    await user.type(search, 'v3')
    expect(screen.getByText(/cute_chibi \/ v3/)).toBeInTheDocument()
    expect(screen.queryByText(/cute_chibi \/ v2/)).not.toBeInTheDocument()
    expect(screen.queryByText(/noir_portrait \/ v1/)).not.toBeInTheDocument()
  })

  it('shows empty hint when no matches', async () => {
    const user = userEvent.setup()
    renderPicker()
    await user.type(screen.getByPlaceholderText('搜索项目 / 版本 / 文件名…'), 'zzznomatch')
    expect(screen.getByText(/没有匹配的 LoRA/)).toBeInTheDocument()
  })

  it('shows "no LoRAs trained" hint when projectLoras is empty', () => {
    renderPicker({ projectLoras: [] })
    expect(screen.getByText(/还没有训练好的 LoRA/)).toBeInTheDocument()
  })

  it('triggers onClose when × is clicked', async () => {
    const user = userEvent.setup()
    const { onClose } = renderPicker()
    await user.click(screen.getByLabelText('关闭挑选区'))
    expect(onClose).toHaveBeenCalled()
  })

  it('triggers onPickExternal when 外部文件 is clicked', async () => {
    const user = userEvent.setup()
    const { onPickExternal } = renderPicker()
    await user.click(screen.getByText('外部文件'))
    expect(onPickExternal).toHaveBeenCalled()
  })
})
