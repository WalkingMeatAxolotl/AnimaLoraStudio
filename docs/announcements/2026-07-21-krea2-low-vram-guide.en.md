---
date: 2026-09-22
tag: guide
title: "Krea 2 on a small GPU"
---
The official fp8 base plus **block swap** can reduce Krea 2's VRAM requirements. This guide helps you choose starting settings and distinguish measured usage from a guarantee that a GPU will work. **8 GB generation and 12 GB training remain trial configurations awaiting full-path validation on those GPUs, not capacity guarantees.**

### Starting settings

Select the **official fp8** base from the download center under **Settings → Training**; selecting the file enables it. These tiers are not universal thresholds for every rank, batch, caption length, and resolution:

| Your VRAM | Training starting point | Generation starting point |
|---|---|---|
| **24 GB and up** | fp8 + gradient checkpointing; try without swap first and add it if peaks exceed capacity | Default VRAM policy; adjust swap based on peaks |
| **16 GB** | fp8 + gradient checkpointing; start with 14 blocks, raise to 28 if needed | Save VRAM; add swap as needed |
| **12 GB** | Swap **28 blocks**, low batch; trial only, not fully validated on a target GPU | Save VRAM + 28 blocks; full-path validation is also needed |
| **8–10 GB** | High headroom risk; experiment at lower resolution, with no training guarantee | Save VRAM + 28 blocks; a roughly 6 GB point-in-time reading is not a generation guarantee |

Leave enough system RAM too: all 28 swapped fp8 blocks hold about 11 GB of pinned weights. **32 GB total RAM is only a starting recommendation**. The OS, model loading, caches, and other programs also need space; current available RAM matters.

### What block swap does and where to set it

Krea 2 and Anima both have 28-block DiTs. Block swap keeps some block weights in RAM and moves them to the GPU for computation, trading transfer time for VRAM. Both families support it from 0.24.0.

- **Training**: enable Advanced on the training config page, then use **System & performance → Block swap**. Default `0` (off), range 0–28.
- **Generation**: **Settings → Generate → VRAM policy → Block swap**, also `0` by default.
- Training includes forward, backward, and gradient-checkpoint recomputation; generation also transfers blocks at each sampling step. Overhead depends on PCIe, RAM, resolution, and configuration, not a universal percentage.
- Transfers introduce no additional quantization, but do not promise bitwise-identical training or images compared with no swap. Gradient differences in the tested, fixed implementation were within the repeated-control noise level.
- Changing the block count reloads the model, increasing the next generation's startup time.

### Reading the measurements

These short experiments used an RTX 5090 environment, official Krea 2 fp8, 1024², standard LoRA rank 32, batch 1 / accumulation 4, and gradient checkpointing. The 14- and 28-block trials were canceled after 6/10 and 8/10 steps respectively. **They are not completed training runs on the target small GPUs.**

| Blocks swapped | Allocated before training | Training-step allocated peak | Sampling allocated peak |
|---|---|---|---|
| 14 | about 7.9 GB | not listed here | not listed here |
| 28 | about 2.2 GB | about 8.4 GB | about 7.1 GB |

`allocated` is PyTorch tensor memory; `reserved` is allocator-held memory, while whole-GPU usage includes context and other costs. Do not combine them into a minimum GPU capacity. After training, the 28-block trial recorded 10.2 GB reserved and 13.4 GB whole-GPU usage. About 6.3 GB was a reading after sampling, not a full-path peak. A separate Task Manager reading of roughly 6 GB after subtracting background use still differed from the roughly 8.8 GB estimate based on the allocated sampling peak plus context.

This does not prove that 12 GB or 8 GB must fail: an allocator may reclaim memory more aggressively on smaller GPUs. It also cannot guarantee success before target-GPU validation. A later completed rank-64 / 14-block run reached a **21.37 GB** training-step allocated peak, showing that residency alone does not predict peaks; rank alone cannot be blamed for the difference either.

The old “about 4% slower” figure compared **14 versus 28 blocks in one sampling task** (52–53 versus 53–55 seconds). It was not training throughput with swap disabled versus 28 blocks.

### Generation VRAM policy

This decides which **model** yields memory to another (text encoder / DiT). It is separate from block swap within a model and can be combined with it. Under **Settings → Generate → VRAM policy**:

- **Default (release after use)**: cached prompts do not reload the text encoder. For new prompts, measured encoding peaks, free VRAM, and WDDM reserve determine whether the DiT moves to RAM first.
- **Save VRAM**: forces the DiT and text encoder to use VRAM sequentially during encoding. Start here on small GPUs, but still validate loading, encoding, merge, sampling, and VAE decode peaks.
- **Performance**: keeps models resident where possible to reduce waits, requiring more VRAM headroom.

The **RAM / VRAM guard** is on by default and tries to reject recognizable shortages before loading. Missing information, other programs, and later peaks can still cause OOM or paging. Keep it enabled, but do not treat it as a guarantee against system stalls.

### LoRA merge precision

On an fp8 base, LoRA deltas are **merged into the weights**. **Settings → Generate → LoRA merge precision** controls temporary computation precision:

- **fp32** (default): aligns with the merge calculation of the validated ComfyUI versions. It does not guarantee bitwise-identical final images across versions, hardware, or workflows.
- **bf16**: lowers temporary delta precision and may change the merged result. Measure switching time and peak memory for your configuration; this does not promise fixed acceleration or halved VRAM.

Both sample with merged fp8 weights, so this does not change their storage size during sampling. Standard LoRA has a chunked-merge optimization; LoHa / LoKr may have different intermediate-matrix costs. Anima and bf16 bases use dynamic LoRA and are unaffected by this option.

For measurement conditions and related design records, see [Training tips → Block swap (Chinese)](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/blob/master/docs/user-guide/training-tips.md#block-交换换出到内存的层数).
