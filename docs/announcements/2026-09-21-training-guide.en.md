---
date: 2026-09-22
tag: guide
title: General LoRA training guide
---
This guide applies to both Anima and Krea 2: establish a trustworthy baseline first, then use separate versions to change data or parameters one variable at a time. Continue with **Krea 2 training guide** for model-specific choices and **Krea 2 on a small GPU** when VRAM is tight.

### Define the training target

- Train one clear target at a time: a character, outfit, style, or concept. Mixed goals make captions and results harder to diagnose.
- Image quality, composition variety, and caption accuracy usually matter more than raw count. Remove duplicates, severe compression, wrong subjects, and distracting watermarks.
- Pick a stable trigger word for each target. Decide before training which traits the trigger should absorb and which should remain explicit in captions.
- Anima commonly uses structured tags; Krea 2 works better with natural-language descriptions. Do not copy one family's caption style into the other unchanged.

### Prepare the training set

1. In **Train Set Curation**, keep images that match the target and still provide useful variation. Hold out a validation set when you want an independent comparison.
2. Use **Preprocess** only for a known problem: deduplicate first, upscale genuinely small images, and crop when composition needs it. Reprocessing everything by habit can damage source quality.
3. In **Tag**, test a small scope before expanding it. The UI skips all existing captions by default, not just hand-written ones, and does not add a trigger to skipped files. Check or update them in Tag Edit; choose overwrite explicitly only when you intend to retag.
4. In **Tag Edit**, correct identity, outfit, action, and scene mistakes, then make trigger placement and caption style consistent.
5. A **regularization set** is optional. Add one when a character or concept needs cleaner trigger control or less base-model forgetting; regularization captions must not contain the training trigger.

### Dataset size, repeats, and total steps

- Do not treat an image count as an admission threshold. A curated set of 10–30 images that covers useful angle and composition changes can be a valid small baseline for one character or one narrow concept. Let checkpoint comparisons reveal missing coverage instead of padding every set to 50 or 100 images first.
- A repeat is a folder's sampling weight, not a "more is better" quality control. Start at `1`; try `2` or higher only to raise one group's relative exposure, while rechecking epochs and total steps. Do not copy `repeat=10/20` and then stack many epochs on top.
- On the Train page, use **Dataset stats** to inspect effective samples, optimizer updates per epoch, and the final total. Image count alone is not comparable across model families, adapters, batch sizes, or gradient accumulation settings.
- This project's historical style-training runs span very different image and update counts. They show that smaller datasets can run successfully, not that a `done` job has proven visual quality. Save checkpoints often enough and compare them under fixed conditions for both character and style work.

### Establish the first baseline

- Choose the correct **model family** and its Raw training base on the Train page. Never train Krea 2 on Turbo.
- Start with that family's defaults and a standard LoRA. Change only fields directly required for this run, such as the output name and epoch count.
- Do not change learning rate, rank, optimizer, timestep distribution, and the dataset all at once. Change one main variable and copy to a new version for comparison.
- Estimated steps combine dataset size, repeats, batch, gradient accumulation, and epochs; they are not a duration estimate. Check the derivation in Dataset stats before submitting.
- `seed=0` randomizes once when a job is created and is then frozen. For comparisons across jobs, fix training, sampling, and validation-split seeds separately. Retain the config snapshot, original data, base model, and software environment; a nonzero seed does not guarantee bitwise reproduction across versions or hardware.

### What to watch during training

- Queueing freezes **config parameters**, not images, captions, or regularization data. You may edit the next config draft, but copy the version before changing data or captions; do not alter material referenced by a queued or running job.
- Use **Job detail** for the effective config, logs, progress, and samples; use **Monitor** to compare active training jobs.
- Loss reveals trends and anomalies, not visual quality by itself. Compare neighboring checkpoints with a fixed prompt and seed to spot undertraining, overfitting, or sudden drift.
- For OOM, first reduce batch size, enable gradient checkpointing, or lower resolution. For Krea 2, then follow the small-GPU guide for fp8, block swap, and generation VRAM policy.

### Evaluate the result

First compare effective update steps with the target and check logs for NaN/Inf, skipped updates, and sampling failures. `done` means the job ended successfully, not that training reached its effective target or produced a good model. A finite loss curve may omit skipped invalid updates.

1. In **Generate**, compare checkpoints with the same base, prompt, sampling seed, LoRA strength, and sampling settings.
2. Compare three cases: base model without a LoRA, LoRA without the trigger, and LoRA with the trigger. Check both learning and unintended changes without the trigger; regularization does not guarantee a strict on/off switch.
3. For undertraining, inspect captions and data coverage before adding training or learning rate. For overfitting, reduce training, lower learning rate, or add diverse data.
4. Keep the best checkpoint with its job snapshot. Change one explainable variable for the next version.

For parameter ranges, caption / regularization strategy, optimizers, troubleshooting, and hardware notes, read [LoRA training tips (Chinese)](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/blob/master/docs/user-guide/training-tips.md).
