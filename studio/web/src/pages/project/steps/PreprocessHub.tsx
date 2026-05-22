import { useSearchParams } from 'react-router-dom'
import PreprocessUpscalePage from './Preprocess'
import PreprocessCropPage from './PreprocessCrop'
import PreprocessDuplicatesPage from './PreprocessDuplicates'
import PreprocessOverviewPage from './PreprocessOverview'

/** Route entry for `/projects/:pid/preprocess`.
 *
 *  Dispatches by `?tool=` query param to the corresponding tool's page:
 *    - `?tool=overview` → Overview page (gallery + multi-select + undo)
 *    - `?tool=dedupe` → Duplicate / variant review
 *    - (default) / `?tool=upscale` → Upscale page
 *    - `?tool=crop` → Crop page
 *    - `?tool=inpaint` → not yet implemented; falls back to default
 *
 *  We use query string (not sub-path) so the sidebar's `/preprocess` matcher
 *  stays simple and the parent route doesn't unmount when switching tools.
 *  Inner pages don't try to preserve state across switches — keeping each
 *  tool's local state self-contained is simpler than lifting it here.
 */
export default function PreprocessHub() {
  const [params] = useSearchParams()
  const tool = params.get('tool') ?? 'upscale'
  if (tool === 'overview') return <PreprocessOverviewPage />
  if (tool === 'dedupe') return <PreprocessDuplicatesPage />
  if (tool === 'crop') return <PreprocessCropPage />
  return <PreprocessUpscalePage />
}
