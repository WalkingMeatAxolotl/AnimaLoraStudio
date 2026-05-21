/** Training-side ARB (aspect-ratio bucketing) mirror.
 *
 *  SYNC WITH `runtime/training/dataset.py:BucketManager`. The algorithm and
 *  defaults here must stay byte-for-byte equivalent to the Python BucketManager
 *  — any change to the algorithm or to the defaults must land in BOTH files in
 *  the same commit, otherwise the frontend's predicted bucket ≠ trainer's
 *  actual bucket and cluster crops will be re-resized by the trainer.
 *
 *  Audience: this file is internal infrastructure for the crop cluster picker.
 *  Users never see bucket dimensions / counts / step / area-tolerance — see
 *  `docs/design/preprocess-crop-design.md` §7.2 for the UX policy.
 */

export interface TrainBucket {
  /** Bucket width in pixels (multiple of step). */
  w: number
  /** Bucket height in pixels (multiple of step). */
  h: number
  /** Cached aspect ratio = w / h. */
  aspect: number
}

/** BucketManager parameters. Defaults mirror Python `BucketManager(...)`. */
export interface BucketParams {
  baseReso?: number       // 1024 — target area ≈ base²
  minReso?: number        // 512  — each side min
  maxReso?: number        // 2048 — each side max
  step?: number           // 64   — w / h granularity
  areaTolerance?: number  // 0.10 — ±10% deviation from base² allowed
  maxArRatio?: number     // 2.0  — max(w/h, h/w) cap
}

const DEFAULTS: Required<BucketParams> = {
  baseReso: 1024,
  minReso: 512,
  maxReso: 2048,
  step: 64,
  areaTolerance: 0.10,
  maxArRatio: 2.0,
}

/** Generate the bucket grid.
 *
 *  Mirrors `BucketManager._generate` in `runtime/training/dataset.py:30-40`:
 *    - double loop over (w, h) in [minReso, maxReso] step `step`
 *    - keep if |w*h - baseReso²| / baseReso² ≤ areaTolerance
 *    - keep if max(w/h, h/w) ≤ maxArRatio
 *
 *  Sorted by aspect ascending (Python returns in nested-loop order; we sort
 *  here for stable consumer-side ordering — does NOT change the bucket set).
 */
export function generateBuckets(p: BucketParams = {}): TrainBucket[] {
  const { baseReso, minReso, maxReso, step, areaTolerance, maxArRatio } = {
    ...DEFAULTS, ...p,
  }
  const baseArea = baseReso * baseReso
  const out: TrainBucket[] = []
  for (let w = minReso; w <= maxReso; w += step) {
    for (let h = minReso; h <= maxReso; h += step) {
      if (Math.abs(w * h - baseArea) / baseArea > areaTolerance) continue
      if (Math.max(w / h, h / w) > maxArRatio) continue
      out.push({ w, h, aspect: w / h })
    }
  }
  out.sort((a, b) => a.aspect - b.aspect)
  return out
}

/** Snap a source aspect ratio to the nearest training bucket.
 *
 *  Mirrors `BucketManager.get_bucket` in `runtime/training/dataset.py:42-51`:
 *  argmin over buckets of **absolute** AR distance |aspect - bucket.aspect|.
 *  NOT relative distance — must match Python exactly so the frontend's
 *  predicted bucket equals what the trainer will choose.
 *
 *  Returns the (baseReso, baseReso) square bucket if `buckets` is empty.
 */
export function snapToBucket(
  aspect: number,
  buckets: TrainBucket[],
  baseReso = DEFAULTS.baseReso,
): TrainBucket {
  if (buckets.length === 0) {
    return { w: baseReso, h: baseReso, aspect: 1 }
  }
  let best = buckets[0]
  let bestDiff = Math.abs(aspect - best.aspect)
  for (let i = 1; i < buckets.length; i++) {
    const b = buckets[i]
    const d = Math.abs(aspect - b.aspect)
    if (d < bestDiff) {
      best = b
      bestDiff = d
    }
  }
  return best
}

// Module-level cache: defaults never change at runtime, so generating once
// covers every consumer. ~30 buckets for the default params; ~kb of memory.
let _defaultBuckets: TrainBucket[] | null = null

/** Get the default bucket set (cached). Use this from cluster code; pass an
 *  explicit `BucketParams` to `generateBuckets()` only if a caller ever needs
 *  non-default params (none today). */
export function defaultBuckets(): TrainBucket[] {
  if (_defaultBuckets === null) _defaultBuckets = generateBuckets()
  return _defaultBuckets
}
