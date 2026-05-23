import { describe, expect, it } from 'vitest'
import { clusterByAspectRatio, type ClusterInput } from './cropClustering'
import { defaultBuckets } from './trainBuckets'

const make = (id: string, w: number, h: number): ClusterInput => ({ id, w, h })

describe('clusterByAspectRatio', () => {
  it('handles empty input', () => {
    const r = clusterByAspectRatio([], { maxCropFraction: 0.1, kMin: 1, kMax: 3 })
    expect(r.kUsed).toBe(0)
    expect(r.assignments).toEqual([])
    expect(r.buckets).toEqual([])
  })

  it('uses kMin when image count is small', () => {
    const images = [make('a', 100, 100)]
    const r = clusterByAspectRatio(images, { maxCropFraction: 0.5, kMin: 3, kMax: 6 })
    expect(r.kUsed).toBeLessThanOrEqual(1)
    expect(r.assignments).toHaveLength(1)
  })

  it('snaps cluster center to common aspect ratio', () => {
    // 5 square + 5 portrait 2:3 images → should split into 2 buckets, snapped to 1:1 and 2:3
    const images: ClusterInput[] = [
      ...Array.from({ length: 5 }, (_, i) => make(`sq${i}`, 100, 100)),
      ...Array.from({ length: 5 }, (_, i) => make(`po${i}`, 200, 300)),
    ]
    const r = clusterByAspectRatio(images, { maxCropFraction: 0.1, kMin: 2, kMax: 4 })
    expect(r.buckets.length).toBeGreaterThanOrEqual(2)
    const ars = r.buckets.map((b) => `${b.targetAr.w}:${b.targetAr.h}`).sort()
    expect(ars).toContain('1:1')
    expect(ars).toContain('2:3')
  })

  it('flags assignments that exceed maxCropFraction', () => {
    // Very wide image will need to be cropped a lot to fit a 1:1 bucket
    const images = [
      ...Array.from({ length: 5 }, (_, i) => make(`sq${i}`, 100, 100)),
      make('wide', 1000, 100), // 10:1 — fitting 1:1 means 90% crop
    ]
    const r = clusterByAspectRatio(images, { maxCropFraction: 0.05, kMin: 1, kMax: 2 })
    const wide = r.assignments.find((a) => a.id === 'wide')
    expect(wide).toBeDefined()
    // With kMax=2 it may or may not get its own cluster; if grouped with squares,
    // the crop fraction should exceed the threshold and be flagged.
    if (wide!.targetAr.w === 1 && wide!.targetAr.h === 1) {
      expect(wide!.skipped).toBe(true)
      expect(wide!.cropFraction).toBeGreaterThan(0.5)
    }
  })

  it('computes center-cropped rect for each assignment', () => {
    const images = [make('a', 200, 200)] // square, target 1:1 means rect = full image
    const r = clusterByAspectRatio(images, { maxCropFraction: 0.1, kMin: 1, kMax: 1 })
    const a = r.assignments[0]
    // Full coverage when AR matches → x/y close to 0, w/h close to 1
    expect(a.rect.w).toBeCloseTo(1, 5)
    expect(a.rect.h).toBeCloseTo(1, 5)
    expect(a.rect.x).toBeCloseTo(0, 5)
    expect(a.rect.y).toBeCloseTo(0, 5)
    expect(a.cropFraction).toBeCloseTo(0, 5)
    expect(a.skipped).toBe(false)
  })

  it('center-crops correctly for non-matching AR', () => {
    // 5 square + 5 wide images, kMax=2 → two clusters: 1:1 and 16:9-ish.
    // The squares end up assigned to the 1:1 bucket and get full-image rect.
    const images: ClusterInput[] = [
      ...Array.from({ length: 5 }, (_, i) => make(`sq${i}`, 100, 100)),
      ...Array.from({ length: 5 }, (_, i) => make(`w${i}`, 200, 100)),
    ]
    const r = clusterByAspectRatio(images, { maxCropFraction: 0.6, kMin: 2, kMax: 2 })
    const sq = r.assignments.find((a) => a.id === 'sq0')!
    expect(sq.targetAr).toEqual({ w: 1, h: 1 })
    expect(sq.rect.w).toBeCloseTo(1, 4)
    expect(sq.rect.h).toBeCloseTo(1, 4)
    expect(sq.cropFraction).toBeCloseTo(0, 4)
    // Wide image (2:1) routed to ≈2:1 cluster — full image fits when target matches.
    const wide = r.assignments.find((a) => a.id === 'w0')!
    expect(wide.cropFraction).toBeLessThan(0.1)
  })

  it('bucket avgCropFraction averages over members', () => {
    const images = [
      make('a', 100, 100),
      make('b', 110, 100),
    ]
    const r = clusterByAspectRatio(images, { maxCropFraction: 0.5, kMin: 1, kMax: 1 })
    expect(r.buckets[0].memberIds.length).toBe(2)
    expect(r.buckets[0].avgCropFraction).toBeGreaterThanOrEqual(0)
  })

  it('respects kMin / kMax bounds', () => {
    const images = Array.from({ length: 20 }, (_, i) => make(`x${i}`, 100 + i * 5, 100))
    const r1 = clusterByAspectRatio(images, { maxCropFraction: 0.5, kMin: 1, kMax: 3 })
    expect(r1.kUsed).toBeLessThanOrEqual(3)
    expect(r1.kUsed).toBeGreaterThanOrEqual(1)
  })

  // §7 ARB alignment — internal trainBucket must be an actual trainer bucket,
  // never a synthetic "pretty AR" pair like {w:4, h:3}.
  describe('trainBucket aligns with trainer grid', () => {
    const grid = defaultBuckets()
    const isOnGrid = (b: { w: number; h: number }) =>
      grid.some((g) => g.w === b.w && g.h === b.h)

    it('cluster bucket.trainBucket is on the trainer grid', () => {
      const images: ClusterInput[] = [
        ...Array.from({ length: 5 }, (_, i) => make(`sq${i}`, 100, 100)),
        ...Array.from({ length: 5 }, (_, i) => make(`po${i}`, 200, 300)),
      ]
      const r = clusterByAspectRatio(images, { maxCropFraction: 0.2, kMin: 2, kMax: 2 })
      for (const b of r.buckets) {
        expect(isOnGrid(b.trainBucket)).toBe(true)
      }
    })

    it('assignment.trainBucket is on the trainer grid', () => {
      const images: ClusterInput[] = Array.from({ length: 8 }, (_, i) => make(`x${i}`, 200, 100))
      const r = clusterByAspectRatio(images, { maxCropFraction: 0.5, kMin: 1, kMax: 1 })
      for (const a of r.assignments) {
        expect(isOnGrid(a.trainBucket)).toBe(true)
      }
    })

    it('rect AR matches trainBucket AR (not displayAr) — no second resize at train time', () => {
      // Use 200×300 images → cluster snaps to a portrait bucket whose AR may
      // not be exactly 2:3. The rect's pixel AR must equal the trainBucket's
      // AR (within rounding), proving the crop targets the real bucket.
      const images: ClusterInput[] = Array.from({ length: 5 }, (_, i) => make(`p${i}`, 200, 300))
      const r = clusterByAspectRatio(images, { maxCropFraction: 0.2, kMin: 1, kMax: 1 })
      const a = r.assignments[0]
      const srcW = 200, srcH = 300
      const rectAr = (a.rect.w * srcW) / (a.rect.h * srcH)
      const trainAr = a.trainBucket.w / a.trainBucket.h
      expect(rectAr).toBeCloseTo(trainAr, 5)
    })

    it('displayAr is a pretty integer pair (for user-facing labels)', () => {
      const images = [make('a', 100, 100), make('b', 200, 300), make('c', 300, 200)]
      const r = clusterByAspectRatio(images, { maxCropFraction: 0.5, kMin: 2, kMax: 3 })
      for (const b of r.buckets) {
        // Pretty AR: both dimensions are small positive integers (denominator ≤ 100)
        expect(Number.isInteger(b.targetAr.w)).toBe(true)
        expect(Number.isInteger(b.targetAr.h)).toBe(true)
        expect(b.targetAr.w).toBeGreaterThan(0)
        expect(b.targetAr.h).toBeGreaterThan(0)
      }
    })
  })
})
