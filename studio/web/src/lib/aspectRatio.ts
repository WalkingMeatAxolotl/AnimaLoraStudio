/** Aspect-ratio display helpers shared between Preprocess pages. */

const COMMON_AR: ReadonlyArray<[number, number]> = [
  [1, 1], [4, 3], [3, 2], [16, 9], [3, 4], [2, 3], [9, 16], [4, 5], [5, 4], [21, 9], [7, 4],
]

/** Format a pixel rect's aspect ratio: snap to common ratios within 1.2 %, else show 2-decimal. */
export function arLabel(w: number, h: number): string {
  if (!isFinite(w) || !isFinite(h) || w <= 0 || h <= 0) return '—'
  const v = w / h
  for (const [cw, ch] of COMMON_AR) {
    const cv = cw / ch
    if (Math.abs(v - cv) / cv < 0.012) return `${cw}:${ch}`
  }
  return v.toFixed(2)
}

/** Common LoRA bucket ARs we snap histogram values to. Ordered wide → tall so
 *  callers can sort buckets in natural visual order without extra logic. */
const HIST_TARGETS: ReadonlyArray<{ w: number; h: number; v: number }> = [
  { w: 21, h: 9, v: 21 / 9 },     // 2.33
  { w: 16, h: 9, v: 16 / 9 },     // 1.78
  { w: 3,  h: 2, v: 3 / 2 },      // 1.50
  { w: 4,  h: 3, v: 4 / 3 },      // 1.33
  { w: 5,  h: 4, v: 5 / 4 },      // 1.25
  { w: 1,  h: 1, v: 1 },          // 1.00
  { w: 4,  h: 5, v: 4 / 5 },      // 0.80
  { w: 3,  h: 4, v: 3 / 4 },      // 0.75
  { w: 2,  h: 3, v: 2 / 3 },      // 0.667
  { w: 9,  h: 16, v: 9 / 16 },    // 0.5625
  { w: 9,  h: 21, v: 9 / 21 },    // 0.428
]

/** Bucket an AR value for histogram display.
 *
 *  Snaps to the **nearest** common LoRA bucket AR (1:1 / 2:3 / 3:2 / 16:9 ...).
 *  No "其他" fallback — trainer (sd-scripts etc.) also lands every image in
 *  some resolution bucket, so histogram showing this is faithful to training
 *  reality. Hard tolerance cutoffs were misleading: a 1.39 image (snapped to
 *  4:3) and a 1.41 image (which would fall into "其他") are semantically the
 *  same shape but appear in different bins.
 *
 *  Returns `{ label, sortKey }`. `sortKey` = the canonical AR value, letting
 *  callers order buckets wide → tall by descending sortKey.
 */
export function arBucket(value: number): { label: string; sortKey: number } {
  if (!isFinite(value) || value <= 0) return { label: '—', sortKey: 0 }
  // argmin over HIST_TARGETS of relative distance |v - t.v| / t.v.
  // Relative (not absolute) so the spacing perceptually matches log-AR — tall
  // and wide buckets get equal "pull" despite their AR magnitudes differing.
  let best = HIST_TARGETS[0]
  let bestErr = Math.abs(value - best.v) / best.v
  for (let i = 1; i < HIST_TARGETS.length; i++) {
    const t = HIST_TARGETS[i]
    const err = Math.abs(value - t.v) / t.v
    if (err < bestErr) {
      best = t
      bestErr = err
    }
  }
  return { label: `${best.w}:${best.h}`, sortKey: best.v }
}
