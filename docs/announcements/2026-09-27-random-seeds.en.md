---
date: 2026-09-27
tag: migration
title: Seed 0 is now frozen per task or comparison group
pin: true
version: "0.28.0"
---
Starting with v0.28.0, `0` in any seed field means “randomize when the task is created,” with a stable boundary chosen for that workflow.

**This applies automatically; no action is required.**

- A new training task freezes separate training, sampling, and validation-split seeds. Retry, pause, and resume keep the actual values from the task snapshot.
- Regular batch generation randomizes each image. An XY grid or one evaluation session shares a comparison seed so noise changes do not obscure the parameter comparison.
- Generation history, PNG metadata, and task snapshots store the actual seed instead of only the requested `0`.

Use a nonzero seed when you need strict reproduction across separate tasks. See [Random seeds and reproducibility](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/blob/master/docs/user-guide/random-seeds.md) for the complete rules.
