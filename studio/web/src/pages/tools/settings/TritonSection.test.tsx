import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type TritonStatus } from '../../../api/client'
import { TritonSection } from './sections'

const mocks = vi.hoisted(() => ({ confirm: vi.fn(), toast: vi.fn() }))
vi.mock('../../../components/Dialog', () => ({ useDialog: () => ({ confirm: mocks.confirm }) }))
vi.mock('../../../components/Toast', () => ({ useToast: () => ({ toast: mocks.toast }) }))
vi.mock('../../../lib/SettingsDrawer', () => ({ useSettingsDrawer: () => ({ sectionRequest: null }) }))

function status(overrides: Partial<TritonStatus> = {}): TritonStatus {
  return {
    state: 'not_installed', installed: false, available: false,
    installed_packages: {}, package: null, version: null,
    expected_package: 'triton-windows', expected_version: '3.8.0.post28',
    compatible: false, reason: 'not_installed', restart_required: false,
    environment: {
      platform: 'Windows', python_version: '3.13', torch_version: '2.11.0',
      torch_cuda_version: '12.8', torch_cuda_available: true,
      supported: true, reason: 'supported',
      expected_package: 'triton-windows', expected_version: '3.8.0.post28',
    },
    ...overrides,
  }
}

async function openSection() {
  await screen.findByText('（未安装）')
  fireEvent.click(screen.getByRole('heading', { name: 'Triton' }))
}

beforeEach(() => {
  mocks.confirm.mockResolvedValue(true)
  vi.spyOn(api, 'getTritonStatus').mockResolvedValue(status())
  vi.spyOn(api, 'installTriton').mockResolvedValue(status({
    state: 'restart_required', installed: true, restart_required: true,
    package: 'triton-windows', version: '3.8.0.post28',
  }))
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

describe('Triton environment row', () => {
  it('uses the shared compact version row without permanent explanatory notices', async () => {
    const { container } = render(<TritonSection />)
    await openSection()
    expect(screen.getByRole('button', { name: '安装（自动匹配）' })).toBeEnabled()
    expect(screen.getByText('Triton:')).toBeInTheDocument()
    expect(container.querySelectorAll('code')).toHaveLength(1)
    expect(container.querySelector('.bg-info-soft')).not.toBeInTheDocument()
    expect(screen.queryByText(/要求制品|Torch 2\.11|CUDA 12\.8/)).not.toBeInTheDocument()
  })

  it('installs only after confirmation and reports restart-required state', async () => {
    render(<TritonSection />)
    await openSection()
    fireEvent.click(screen.getByRole('button', { name: '安装（自动匹配）' }))
    await waitFor(() => expect(api.installTriton).toHaveBeenCalledTimes(1))
    expect(await screen.findByText(/待重启/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '重装（自动匹配）' })).toBeEnabled()
    expect(mocks.toast).toHaveBeenCalledWith(expect.stringContaining('重启 Studio'), 'success')
  })

  it('does not install when confirmation is canceled', async () => {
    mocks.confirm.mockResolvedValue(false)
    render(<TritonSection />)
    await openSection()
    fireEvent.click(screen.getByRole('button', { name: '安装（自动匹配）' }))
    await waitFor(() => expect(mocks.confirm).toHaveBeenCalledTimes(1))
    expect(api.installTriton).not.toHaveBeenCalled()
  })

  it('disables installation for an unsupported environment', async () => {
    vi.mocked(api.getTritonStatus).mockResolvedValue(status({
      environment: { ...status().environment, supported: false, reason: 'torch_unsupported' },
    }))
    render(<TritonSection />)
    await openSection()
    const button = screen.getByRole('button', { name: '安装（自动匹配）' })
    expect(button).toBeDisabled()
    expect(button).toHaveAttribute('title', expect.stringContaining('PyTorch 2.11'))
  })

  it('reports installation failures and allows a retry', async () => {
    vi.mocked(api.installTriton).mockRejectedValue(new Error('wheel unavailable'))
    render(<TritonSection />)
    await openSection()
    fireEvent.click(screen.getByRole('button', { name: '安装（自动匹配）' }))
    await waitFor(() => expect(mocks.toast).toHaveBeenCalledWith(
      expect.stringContaining('wheel unavailable'), 'error',
    ))
    expect(screen.getByRole('button', { name: '安装（自动匹配）' })).toBeEnabled()
  })

  it('exposes a failed status request and recovers when reopened', async () => {
    vi.mocked(api.getTritonStatus).mockRejectedValueOnce(new Error('status unavailable'))
    const view = render(<TritonSection />)
    expect(await screen.findByText('Error: status unavailable')).toBeInTheDocument()
    expect(screen.getByText(/加载失败/)).toBeInTheDocument()
    view.unmount()
    render(<TritonSection />)
    expect(await screen.findByText('（未安装）')).toBeInTheDocument()
  })

  it('refreshes package facts through the shared action', async () => {
    render(<TritonSection />)
    await openSection()
    vi.mocked(api.getTritonStatus).mockResolvedValue(status({
      state: 'available', installed: true, available: true,
      package: 'triton-windows', version: '3.8.0.post28',
    }))
    fireEvent.click(screen.getByTitle('刷新状态'))
    expect(await screen.findByText('v3.8.0.post28')).toBeInTheDocument()
    expect(api.getTritonStatus).toHaveBeenCalledTimes(2)
  })
})
