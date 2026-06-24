import { describe, expect, it } from 'vitest'
import { defaultBuckets, generateBuckets, snapToBucket } from './trainBuckets'

describe('generateBuckets', () => {
  it('produces exactly 37 buckets with default params (sync check)', () => {
    // Strict count match against the Python `BucketManager()` output. If this
    // ever changes, ONE of two things happened:
    //  (a) Someone changed the algorithm or defaults in only one of the two
    //      ports — this test catches the drift. Fix: align both files in the
    //      same commit, see comments at top of `trainBuckets.ts` and
    //      `runtime/training/dataset.py:BucketManager`.
    //  (b) An intentional algorithm change. Re-run
    //      `python -c "from runtime.training.dataset import BucketManager;
    //      print(len(BucketManager().buckets))"` and update both this number
    //      and the Python side together.
    expect(generateBuckets().length).toBe(37)
  })

  it('includes the canonical (1024, 1024) square at base_reso', () => {
    const bs = generateBuckets()
    expect(bs.some((b) => b.w === 1024 && b.h === 1024)).toBe(true)
  })

  it('all buckets satisfy area band ±10% of base²', () => {
    const bs = generateBuckets()
    const baseArea = 1024 * 1024
    for (const b of bs) {
      const dev = Math.abs(b.w * b.h - baseArea) / baseArea
      expect(dev).toBeLessThanOrEqual(0.10 + 1e-9)
    }
  })

  it('all buckets satisfy max AR ≤ 2.0', () => {
    const bs = generateBuckets()
    for (const b of bs) {
      expect(Math.max(b.w / b.h, b.h / b.w)).toBeLessThanOrEqual(2.0 + 1e-9)
    }
  })

  it('all bucket dims are multiples of step (64)', () => {
    const bs = generateBuckets()
    for (const b of bs) {
      expect(b.w % 64).toBe(0)
      expect(b.h % 64).toBe(0)
    }
  })

  it('is sorted by aspect ascending', () => {
    const bs = generateBuckets()
    for (let i = 1; i < bs.length; i++) {
      expect(bs[i].aspect).toBeGreaterThanOrEqual(bs[i - 1].aspect)
    }
  })

  it('derives bounds from (baseReso, R) — small base keeps AR variety', () => {
    // base=512 with the old hard-wired minReso=512 collapsed to the 512×512
    // square only. Derived bounds restore non-square buckets. Mirrors the
    // Python regression in tests/test_bucket_manager.py.
    const bs = generateBuckets({ baseReso: 512 })
    expect(bs.length).toBeGreaterThan(1)
    expect(bs.some((b) => b.w !== b.h)).toBe(true)
    expect(bs.some((b) => b.w > b.h)).toBe(true)
    expect(bs.some((b) => b.h > b.w)).toBe(true)
  })

  it('maxArRatio (R) widens the bucket set symmetrically', () => {
    const narrow = generateBuckets({ maxArRatio: 2.0 })
    const wide = generateBuckets({ maxArRatio: 3.0 })
    const widest = (bs: { w: number; h: number }[]) =>
      Math.max(...bs.map((b) => Math.max(b.w / b.h, b.h / b.w)))
    expect(widest(narrow)).toBeLessThanOrEqual(2.0 + 1e-9)
    expect(widest(wide)).toBeGreaterThan(2.0)
    expect(widest(wide)).toBeLessThanOrEqual(3.0 + 1e-9)
    expect(wide.length).toBeGreaterThan(narrow.length)
  })

  it('contains expected canonical anchors derived from sd-scripts-style buckets', () => {
    // These specific (w, h) pairs are widely cited training buckets at base 1024
    // and exist in the Python BucketManager output. If this test ever changes,
    // verify with `python -c "from runtime.training.dataset import BucketManager; ..."`.
    const bs = generateBuckets()
    const has = (w: number, h: number) => bs.some((b) => b.w === w && b.h === h)
    expect(has(1024, 1024)).toBe(true)
    expect(has(1152, 896)).toBe(true)
    expect(has(1216, 832)).toBe(true)
    expect(has(896, 1152)).toBe(true)   // mirror
    expect(has(832, 1216)).toBe(true)   // mirror
  })
})

describe('snapToBucket', () => {
  const bs = defaultBuckets()

  it('snaps exact 1:1 to (1024, 1024)', () => {
    const b = snapToBucket(1.0, bs)
    expect(b.w).toBe(1024)
    expect(b.h).toBe(1024)
  })

  it('snaps 1.39 and 1.40 to the same bucket (the historical bug case)', () => {
    const b139 = snapToBucket(1.39, bs)
    const b140 = snapToBucket(1.40, bs)
    expect(b139).toEqual(b140)
  })

  it('snaps by absolute AR distance (matching Python BucketManager)', () => {
    // Don't hardcode the expected bucket — the default grid has many buckets
    // near AR 1.4 and the closest one depends on the bucket set. Instead assert
    // the invariant: no other bucket is closer to the input AR than the snap result.
    const target = 1.4
    const snapped = snapToBucket(target, bs)
    const snappedDiff = Math.abs(target - snapped.aspect)
    for (const b of bs) {
      expect(Math.abs(target - b.aspect)).toBeGreaterThanOrEqual(snappedDiff - 1e-12)
    }
  })

  it('uses area as tie-breaker for same-AR buckets', () => {
    const highResBuckets = generateBuckets({ baseReso: 1536, maxArRatio: 2.0 })
    expect(highResBuckets.some((b) => b.w === 1472 && b.h === 1472)).toBe(true)
    expect(highResBuckets.some((b) => b.w === 1536 && b.h === 1536)).toBe(true)
    expect(highResBuckets.some((b) => b.w === 1600 && b.h === 1600)).toBe(true)

    const b = snapToBucket(1.0, highResBuckets, 1536)
    expect(b.w).toBe(1536)
    expect(b.h).toBe(1536)
  })

  it('handles extreme wide → snaps to the widest bucket', () => {
    const b = snapToBucket(5.0, bs)
    // Widest allowed under max_ar=2.0
    expect(b.aspect).toBeCloseTo(2.0, 2)
  })

  it('handles extreme tall → snaps to the tallest bucket', () => {
    const b = snapToBucket(0.1, bs)
    // Tallest allowed under max_ar=2.0 → aspect ≈ 0.5
    expect(b.aspect).toBeCloseTo(0.5, 2)
  })

  it('falls back to (baseReso, baseReso) when buckets is empty', () => {
    const b = snapToBucket(1.5, [], 768)
    expect(b.w).toBe(768)
    expect(b.h).toBe(768)
  })
})

describe('defaultBuckets caching', () => {
  it('returns identical reference across calls', () => {
    expect(defaultBuckets()).toBe(defaultBuckets())
  })
})
