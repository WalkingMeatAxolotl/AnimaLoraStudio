/** 1-D k-means on aspect ratios for the crop "智能聚类" mode.
 *
 *  Why client-side: AR values are tiny (one float per image), and the
 *  algorithm runs in <10ms for thousands of images. Going through a backend
 *  job would buy nothing but latency.
 *
 *  ARB alignment (see docs/design/preprocess-crop-design.md §7): cluster
 *  centers snap **internally** to the trainer's actual bucket grid (via
 *  `lib/trainBuckets.ts`), so cropped outputs land exactly on a training
 *  bucket and the trainer won't re-resize them. Display labels (`assignment.
 *  targetAr`) stay as user-friendly pretty AR (1:1, 4:3, 3:2, ...) — UX
 *  policy is to never surface ARB internals (base_reso, bucket dims) to users.
 */

import { defaultBuckets, snapToBucket } from './trainBuckets'

export interface ClusterInput {
  /** Stable id (= source filename). */
  id: string
  /** Pixel width. */
  w: number
  /** Pixel height. */
  h: number
}

export interface ClusterRect {
  /** Normalized [0..1] coords on the source image. */
  x: number
  y: number
  w: number
  h: number
}

export interface ClusterAssignment {
  id: string
  /** Cluster index (0..k-1). */
  clusterId: number
  /** **Display** aspect ratio for this cluster (pretty integer pair, e.g. 4:3 / 16:9).
   *  Drives only UI labels. Crop dimensions come from `trainBucket` below. */
  targetAr: { w: number; h: number }
  /** **Internal** training bucket the cluster snaps to. Drives the actual crop
   *  rect AR so the cropped image lands exactly on a trainer bucket — no
   *  re-resize at train time. Not surfaced in UI (see §7.2). */
  trainBucket: { w: number; h: number }
  /** Center-cropped rect on the source image, normalized [0..1]. */
  rect: ClusterRect
  /** Area lost to crop, normalized [0..1]. */
  cropFraction: number
  /** Skipped because cropFraction > maxCropFraction. */
  skipped: boolean
}

export interface ClusterParams {
  /** Maximum acceptable crop fraction (skip image if exceeded). */
  maxCropFraction: number
  /** Minimum number of clusters. */
  kMin: number
  /** Maximum number of clusters. */
  kMax: number
}

export interface ClusterSummary {
  /** Selected k after elbow scan. */
  kUsed: number
  /** Image assignments. */
  assignments: ClusterAssignment[]
  /** Per-cluster aggregate info. */
  buckets: {
    clusterId: number
    /** Display AR (pretty pair for label). */
    targetAr: { w: number; h: number }
    /** Internal training bucket (for crop rect). Not surfaced in UI. */
    trainBucket: { w: number; h: number }
    memberIds: string[]
    avgCropFraction: number
  }[]
}

/** Common LoRA bucket ratios we snap to when close (within 4%). */
const SNAP_TARGETS: { w: number; h: number; v: number }[] = [
  { w: 1, h: 1, v: 1 },
  { w: 4, h: 3, v: 4 / 3 },
  { w: 3, h: 2, v: 3 / 2 },
  { w: 16, h: 9, v: 16 / 9 },
  { w: 21, h: 9, v: 21 / 9 },
  { w: 3, h: 4, v: 3 / 4 },
  { w: 2, h: 3, v: 2 / 3 },
  { w: 9, h: 16, v: 9 / 16 },
  { w: 4, h: 5, v: 4 / 5 },
  { w: 5, h: 4, v: 5 / 4 },
]

function snapToCommonAr(v: number, tolerance = 0.04): { w: number; h: number } {
  let best: { w: number; h: number } | null = null
  let bestErr = tolerance
  for (const t of SNAP_TARGETS) {
    const err = Math.abs(v - t.v) / t.v
    if (err < bestErr) {
      best = { w: t.w, h: t.h }
      bestErr = err
    }
  }
  if (best) return best
  // Fallback — reduce decimal AR to a small integer pair (denominator ≤ 16)
  for (let h = 1; h <= 16; h++) {
    const w = Math.round(v * h)
    if (w >= 1 && Math.abs(v - w / h) / v < 0.02) {
      return { w, h }
    }
  }
  return { w: Math.round(v * 100), h: 100 }
}

/** Compute the max rect of target AR centered in source. Returns normalized {w,h}. */
function maxRectForAr(
  srcW: number,
  srcH: number,
  arW: number,
  arH: number,
): { w: number; h: number } {
  const srcAr = srcW / srcH
  const tgtAr = arW / arH
  if (tgtAr > srcAr) {
    // Target wider → fit by width
    return { w: 1, h: srcAr / tgtAr }
  } else {
    return { w: tgtAr / srcAr, h: 1 }
  }
}

/** 1-D k-means with deterministic init (quantiles). */
function kmeans1d(values: number[], k: number, maxIter = 50): { centers: number[]; labels: number[] } {
  const n = values.length
  if (k <= 0 || n === 0) return { centers: [], labels: new Array(n).fill(0) }
  // Init: sort & take k quantile midpoints (stable across runs)
  const sorted = [...values].sort((a, b) => a - b)
  const centers: number[] = []
  for (let i = 0; i < k; i++) {
    const idx = Math.min(n - 1, Math.floor((i + 0.5) * n / k))
    centers.push(sorted[idx])
  }
  const labels = new Array<number>(n).fill(0)
  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false
    // Assign step
    for (let i = 0; i < n; i++) {
      let bestK = 0
      let bestD = Math.abs(values[i] - centers[0])
      for (let j = 1; j < k; j++) {
        const d = Math.abs(values[i] - centers[j])
        if (d < bestD) {
          bestD = d
          bestK = j
        }
      }
      if (labels[i] !== bestK) {
        labels[i] = bestK
        changed = true
      }
    }
    // Update step
    const sum = new Array(k).fill(0)
    const cnt = new Array(k).fill(0)
    for (let i = 0; i < n; i++) {
      sum[labels[i]] += values[i]
      cnt[labels[i]] += 1
    }
    for (let j = 0; j < k; j++) {
      if (cnt[j] > 0) centers[j] = sum[j] / cnt[j]
    }
    if (!changed) break
  }
  return { centers, labels }
}

/** Within-cluster sum of squared distances (inertia). */
function inertia(values: number[], centers: number[], labels: number[]): number {
  let s = 0
  for (let i = 0; i < values.length; i++) {
    const d = values[i] - centers[labels[i]]
    s += d * d
  }
  return s
}

/** Elbow heuristic: pick k where the marginal drop in inertia from k-1 → k stops being meaningful. */
function pickKByElbow(
  values: number[],
  kMin: number,
  kMax: number,
): { k: number; centers: number[]; labels: number[] } {
  // Edge: only one unique value or very few items → use kMin clusters
  if (values.length <= kMin) {
    const r = kmeans1d(values, Math.min(values.length, kMin))
    return { k: r.centers.length, ...r }
  }
  let prev = { k: kMin, ...kmeans1d(values, kMin) }
  let prevInertia = inertia(values, prev.centers, prev.labels)
  let best = prev
  let bestInertia = prevInertia
  // Threshold: pick the largest k where the relative drop > 15% of the original drop
  for (let k = kMin + 1; k <= kMax; k++) {
    const cur = { k, ...kmeans1d(values, k) }
    const curInertia = inertia(values, cur.centers, cur.labels)
    const drop = prevInertia - curInertia
    const relDrop = prevInertia > 0 ? drop / prevInertia : 0
    if (relDrop > 0.15) {
      // Significant gain → adopt this k
      best = cur
      bestInertia = curInertia
    }
    prev = cur
    prevInertia = curInertia
  }
  void bestInertia
  return best
}

export function clusterByAspectRatio(
  images: ClusterInput[],
  params: ClusterParams,
): ClusterSummary {
  if (images.length === 0) {
    return { kUsed: 0, assignments: [], buckets: [] }
  }
  const arValues = images.map((im) => im.w / im.h)
  const kMin = Math.max(1, Math.min(params.kMin, images.length))
  const kMax = Math.max(kMin, Math.min(params.kMax, images.length))
  const { k, centers, labels } = pickKByElbow(arValues, kMin, kMax)

  // For each cluster center, snap to the trainer's bucket grid (internal —
  // drives the actual crop AR so we don't get a second resize at train time)
  // AND compute a pretty AR for the user-facing label (e.g. "聚类 3:2").
  const trainBuckets = defaultBuckets()
  const clusterTargets = centers.map((c) => {
    const bucket = snapToBucket(c, trainBuckets)
    const display = snapToCommonAr(bucket.w / bucket.h)
    return { trainBucket: { w: bucket.w, h: bucket.h }, displayAr: display }
  })

  const assignments: ClusterAssignment[] = images.map((im, i) => {
    const cId = labels[i]
    const { trainBucket, displayAr } = clusterTargets[cId]
    // Rect is computed from the **training bucket** AR, not from displayAr —
    // this is the whole point of §7. The label shows a pretty AR for humans.
    const fit = maxRectForAr(im.w, im.h, trainBucket.w, trainBucket.h)
    const cropFraction = 1 - fit.w * fit.h
    return {
      id: im.id,
      clusterId: cId,
      targetAr: displayAr,
      trainBucket,
      rect: {
        x: (1 - fit.w) / 2,
        y: (1 - fit.h) / 2,
        w: fit.w,
        h: fit.h,
      },
      cropFraction,
      skipped: cropFraction > params.maxCropFraction,
    }
  })

  const buckets = Array.from({ length: k }, (_, j) => {
    const members = assignments.filter((a) => a.clusterId === j)
    const memberIds = members.map((m) => m.id)
    const avgCropFraction =
      members.length > 0
        ? members.reduce((s, m) => s + m.cropFraction, 0) / members.length
        : 0
    return {
      clusterId: j,
      targetAr: clusterTargets[j].displayAr,
      trainBucket: clusterTargets[j].trainBucket,
      memberIds,
      avgCropFraction,
    }
  })

  return { kUsed: k, assignments, buckets }
}
