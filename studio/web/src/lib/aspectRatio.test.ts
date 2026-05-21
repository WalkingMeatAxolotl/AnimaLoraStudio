import { describe, expect, it } from 'vitest'
import { arBucket, arLabel } from './aspectRatio'

describe('arLabel', () => {
  it('snaps to common ratios within 1.2%', () => {
    expect(arLabel(1024, 1024)).toBe('1:1')
    expect(arLabel(1920, 1080)).toBe('16:9')
    expect(arLabel(1024, 1536)).toBe('2:3')
    expect(arLabel(1080, 1920)).toBe('9:16')
  })

  it('returns 2-decimal fraction for non-common ratios', () => {
    // 1.43 is between 4:3 (1.33) and 3:2 (1.50) — no snap within 1.2% tolerance.
    expect(arLabel(143, 100)).toBe('1.43')
  })

  it('handles invalid inputs', () => {
    expect(arLabel(0, 100)).toBe('—')
    expect(arLabel(100, 0)).toBe('—')
    expect(arLabel(NaN, 100)).toBe('—')
  })
})

describe('arBucket', () => {
  it('snaps exact common ARs to their canonical label', () => {
    expect(arBucket(1).label).toBe('1:1')
    expect(arBucket(16 / 9).label).toBe('16:9')
    expect(arBucket(3 / 2).label).toBe('3:2')
    expect(arBucket(4 / 3).label).toBe('4:3')
    expect(arBucket(2 / 3).label).toBe('2:3')
    expect(arBucket(9 / 16).label).toBe('9:16')
    expect(arBucket(4 / 5).label).toBe('4:5')
  })

  it('handles ambiguous in-between ratios by snapping to nearest', () => {
    // 1.39 / 1.40 sit between 4:3 (1.333) and 3:2 (1.5); both are closer to 4:3.
    // Before nearest-snap, 1.39 → 4:3 but 1.40 → "其他" — same shape, different bin.
    expect(arBucket(1.39).label).toBe('4:3')
    expect(arBucket(1.40).label).toBe('4:3')
    // Midpoint between 4:3 (1.333) and 3:2 (1.5) by relative distance falls
    // ≈1.42; verify both sides snap to the expected closer target.
    expect(arBucket(1.38).label).toBe('4:3')
    expect(arBucket(1.46).label).toBe('3:2')
  })

  it('snaps extreme ratios to the closest bucket (no "其他" leak)', () => {
    // 3:1 = 3.0 is way wider than any bucket; nearest is 21:9 = 2.33.
    expect(arBucket(3.0).label).toBe('21:9')
    // 1:3 = 0.333 is way taller than 9:21 = 0.428; should snap there.
    expect(arBucket(1 / 3).label).toBe('9:21')
  })

  it('snaps with relative (perceptual) distance, not absolute', () => {
    // Both 0.78 (between 4:5=0.80 and 3:4=0.75) and 1.27 (between 5:4=1.25 and
    // 4:3=1.33) sit symmetric around their neighbors in log space. Sanity: in
    // each case the closer canonical AR wins.
    expect(arBucket(0.78).label).toBe('4:5')
    expect(arBucket(1.27).label).toBe('5:4')
  })

  it('returns sortKey equal to canonical AR for wide→tall ordering', () => {
    expect(arBucket(16 / 9).sortKey).toBeGreaterThan(arBucket(1).sortKey)
    expect(arBucket(1).sortKey).toBeGreaterThan(arBucket(2 / 3).sortKey)
    // Off-bucket inputs sort by the snapped canonical, not the raw value
    expect(arBucket(1.40).sortKey).toBeCloseTo(4 / 3, 5)
  })

  it('handles invalid input', () => {
    expect(arBucket(0).label).toBe('—')
    expect(arBucket(NaN).label).toBe('—')
  })
})
