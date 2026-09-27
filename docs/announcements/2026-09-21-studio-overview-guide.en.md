---
date: 2026-09-22
tag: guide
title: "Studio tour: pages and workflow"
---
If this is your first time in Studio, start here to learn what each page owns and how a LoRA moves from source images to testing. Studio keeps project data, training versions, and execution jobs separate: **a project owns source material, a version owns one training plan, and a training job saves the config for one run.**

### Global pages

| Page | What it is for |
|---|---|
| **Projects** | Create, search, archive, and manage LoRA projects; open a project to work on its dataset and versions. |
| **Queue** | Inspect running, waiting, scheduled, and finished GPU or data jobs; open job detail for logs, monitoring, the frozen config, and outputs. |
| **Generate** | Load a base model and one or more LoRAs for single-image, batch, or XY comparisons; prompts can come from training data, the gallery, or history. |
| **Presets** | Manage reusable training-config templates. A preset is a starting point and does not keep overwriting configs already copied into versions. |
| **Monitor** | Watch loss, learning rate, progress, and samples from active training jobs in one place. |
| **Settings** | Install or select models and configure download sources, taggers, training, monitoring, generation, page behavior, and the system. |
| **Announcements / Guides** | Open the top-right bell for releases, migrations, and reusable guides; select **Guides** to return here quickly. |

### Inside a project

- **Overview**: review project, version, dataset, and training status; create or switch versions.
- **Dataset**: download from Danbooru / Gelbooru or import local images and zip files. Every version shares this source collection.
- **Train Set Curation**: choose the images for this version, organize training folders, and optionally hold out a validation set.
- **Preprocess (optional)**: deduplicate, upscale, crop, or retouch. These actions change this version's training set, not the shared source collection.
- **Tag**: generate captions with WD14, CLTagger, or an LLM and choose the trigger word and overwrite policy.
- **Tag Edit**: inspect individual captions and bulk add, remove, or replace tags until the training signal matches your goal.
- **Reg set (optional)**: generate base-model images or search Booru for regularization data to reduce forgetting and improve trigger-word control.
- **Train**: copy a preset into this version's private config, review the dataset and estimated steps, then queue now or schedule later.

### Projects, versions, and jobs

- Use one **project** for one character, style, or subject. It retains the shared source images.
- Create multiple **versions** to try different curation, captions, model families, or parameters without overwriting each other.
- Starting training creates a **training-config snapshot**. Later parameter edits cannot change a queued or running job's config. **Images, captions, and regularization data are not frozen with it.**
- Do not edit that version's training material while its job is queued or running. Copy the version before changing data or captions for the next experiment.
- For debugging, start with the job detail's config, seeds, logs, and outputs rather than the current draft. Reproduction also requires the original data, base model, and software environment; older jobs may lack historical fields.

### A shortest path for new users

1. Install and select a model family under **Settings → Training**.
2. Create a **Project**, then import or download source images.
3. Follow the project sidebar through curation, optional preprocessing, tagging, and caption review.
4. If you are unsure about regularization, skip it for the first run; start from the selected model family's defaults.
5. After queueing, follow logs and samples in **Job detail**, then compare checkpoints in **Generate**.
6. Change one main variable per new version so each result has an explainable comparison.

For installation, model downloads, and the complete walkthrough, see [Getting Started](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/blob/master/docs/user-guide/getting-started.en.md).
