import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import Filmstrip from './Filmstrip'

describe('Filmstrip', () => {
  it('names the image group and exposes the active image as a pressed choice', async () => {
    const onSelect = vi.fn()
    const user = userEvent.setup()

    render(
      <Filmstrip
        items={[{ name: '1_data/a.png' }, { name: '1_data/b.png' }]}
        activeName="1_data/a.png"
        onSelect={onSelect}
        thumbUrl={(item) => `/thumb/${item.name}`}
        ariaLabel="Crop images"
        itemLabel={(item) => `Edit ${item.name}`}
      />,
    )

    const group = screen.getByRole('group', { name: 'Crop images' })
    const active = screen.getByRole('button', { name: 'Edit 1_data/a.png' })
    const next = screen.getByRole('button', { name: 'Edit 1_data/b.png' })
    expect(group).toContainElement(active)
    expect(active).toHaveAttribute('aria-pressed', 'true')
    expect(next).toHaveAttribute('aria-pressed', 'false')

    await user.click(next)
    expect(onSelect).toHaveBeenCalledWith('1_data/b.png')
  })

  it('keeps list controls outside the thumbnail scrollport, including for empty filters', () => {
    const { container, rerender } = render(
      <Filmstrip
        items={[{ name: '1_data/a.png' }]}
        activeName="1_data/a.png"
        onSelect={() => undefined}
        thumbUrl={(item) => `/thumb/${item.name}`}
        ariaLabel="Workspace images"
        header={<div role="radiogroup" aria-label="Image filter">All</div>}
      />,
    )

    const filter = screen.getByRole('radiogroup', { name: 'Image filter' })
    const scrollport = container.querySelector('.overflow-y-auto')
    expect(scrollport).not.toContainElement(filter)
    expect(filter.compareDocumentPosition(scrollport!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

    rerender(
      <Filmstrip
        items={[]}
        activeName={null}
        onSelect={() => undefined}
        thumbUrl={() => ''}
        ariaLabel="Workspace images"
        header={<div role="radiogroup" aria-label="Image filter">Cropped</div>}
        emptyHint="No cropped images"
      />,
    )
    expect(screen.getByRole('radiogroup', { name: 'Image filter' })).toBeInTheDocument()
    expect(screen.getByRole('group', { name: 'Workspace images' })).toHaveTextContent('No cropped images')
  })
})
