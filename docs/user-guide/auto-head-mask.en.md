# Automatic head masks

Automatic head masks are intended for clothing, pose, and style LoRAs where cartoon faces, hair, ears, and head accessories should not contribute to the loss. The feature writes `.mask` sidecars only; source images are never modified.

## Workflow

1. Open **Preprocess → Retouch → Mask** for a project version. On first use, choose **Download head detector**. The roughly 42.5 MiB weight is integrity-checked after download.
2. Save all manual strokes. Batch detection is disabled while any unsaved strokes exist.
3. Choose **Detect all** or **Detect current**. The defaults are confidence 0.413, NMS IoU 0.7, 10% padding, and 3% feathering.
4. Review the proposal overlays. Every region starts selected; clear individual regions or select/clear all regions on the current image. Use the **No head detected** filter to inspect likely misses.
5. Choose **Apply selected**. The automatic mask is merged with the existing manual mask using a pixelwise minimum, so it can never restore a manually ignored pixel.
6. On the Train page, use **Enable masked loss** when the warning appears. Leap and NaViT Packing remain incompatible with masked loss and must be disabled first.

## Parameters

- **Confidence**: higher values reduce false positives but may miss small heads. Start at 0.413; try 0.5–0.6 if false positives dominate.
- **NMS IoU**: controls removal of overlapping detections. The 0.7 default helps preserve people standing close together.
- **Padding**: expands each box by a percentage of its width and height. Ten percent usually covers hair tips, animal ears, and accessories.
- **Feather**: adds a 0-to-255 transition outside the expanded region. Three percent avoids an unnecessarily hard loss boundary.

## Safety and undo

- Detection writes a proposal only. Application validates the entire batch and prepares all temporary files before atomic replacement; a preparation failure leaves all masks unchanged.
- A proposal is stale and cannot be applied if the source image's dimensions, file size, or modification time changed after detection.
- **Undo this automatic mask** restores the pre-application snapshot. Undo is refused if a related `.mask` was manually changed afterward, protecting the newer edit.
- v1 does not edit captions. Character names, hair colors, eye descriptions, and other identity tags still condition training; remove them manually on the Tagging page when required.

The detector is downloaded on demand from `deepghs/anime_head_detection`. See [ADR 0017](../adr/0017-proposal-based-auto-head-mask.md) for the pinned revision and integrity contract, and `THIRD_PARTY_NOTICES.md` for licensing notes.
