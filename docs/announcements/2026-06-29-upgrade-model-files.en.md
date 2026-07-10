---
date: 2026-06-29
tag: migration
title: After upgrading from an older version, check your T5 / text-encoder files before training
version: "0.16.0"
---
If you self-updated from a version **older than 0.15.0**, a few tiny model files (the T5 tokenizer's `spiece.model` and configs, and the text encoder's tokenizer / configs) may have been deleted during the update — so training can fail with a "file not found" error right after you upgrade. A quick check before training takes only a few seconds.

### Why this happens

- These small files used to be committed into git along with the repo; since 0.15.0 the whole `models/` directory became a pure data folder that git no longer tracks.
- Only these small files are affected; the **large weight files** for the base model / VAE / text encoder were never in git and won't be lost.

### What to do

After upgrading, and before you start training, go to **Settings → Training → Base model** and check whether either of these cards reports missing files:

- **T5 tokenizer** (`spiece.model` and 2 config files)
- **Qwen3 / text encoder** (`config.json` / `merges.txt` / `tokenizer_config.json`)

If either is missing files, click download to fill them in (just these few small files — it's quick). Then train as usual.
