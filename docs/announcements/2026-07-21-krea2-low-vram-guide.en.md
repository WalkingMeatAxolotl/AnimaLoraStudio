---
date: 2026-07-21
tag: notice
title: "Krea 2 on a small GPU"
pin: true
version: "0.21.0"
---
As of 0.21.0, the official fp8 base plus **block swap** brings Krea 2 down from 32 GB to **12 GB for training and 8 GB for generation**. Here are the recommended settings per VRAM tier, and how the three relevant knobs work. All numbers come from an RTX 5090 with the official fp8 base at 1024².

### Recommended settings

Always pick the **official fp8** base (download it from the center under **Settings → Training**; selecting the file is all it takes).

| Your VRAM | Training | Test generation |
|---|---|---|
| **24 GB and up** | No block swap needed | VRAM policy "default", no block swap needed |
| **16 GB** | Block swap from 14, raise to 28 if needed | VRAM policy "default" or "save VRAM"; block swap as needed |
| **12 GB** | Block swap **28** (all blocks) | "Save VRAM" + block swap 28 |
| **8–10 GB** | Too tight — lower the resolution first | "Save VRAM" + block swap 28; 1024² measures about 6.3 GB for this app |

RAM (not VRAM) has to keep up too: swapped-out blocks stay resident and pinned — about 11 GB for all 28 on fp8, so **32 GB of RAM is the floor**.

### Block swap (blocks moved to RAM)

Krea 2's DiT has 28 blocks, and only one is computing at any moment — the rest sit in VRAM purely to save transfer time. Block swap keeps the later blocks in RAM and moves each onto the GPU only when its turn comes, releasing it right after: time traded for VRAM. It **applies to Krea 2 only** (Anima is small enough not to need it, and the option doesn't appear for it).

**Where to turn it on**

- **Training**: switch on "advanced" in the training config, then find "block swap" under **System & performance**. Default `0` (off), range 0–28; anything higher is rejected.
- **Test generation**: **Settings → Test → VRAM policy**, the "block swap" field, also `0` by default.

**How many**

| Blocks swapped | Resident VRAM (training) | Training step peak | Sampling peak | Speed |
|---|---|---|---|---|
| 0 (off) | about 13 GB | — | — | baseline |
| 14 | 7.9 GB | — | — | — |
| **28 (all)** | **2.2 GB** | **8.4 GB** | **7.1 GB** | about 4% slower |

Those are PyTorch allocations; what Task Manager shows adds roughly 1.7 GB of GPU context on top: **training** lands around 10 GB total (16 GB comfortable, 12 GB workable), **generation** around 6.3 GB at 1024² with nothing else on the GPU — **an 8 GB card is enough**. Start at 14, raise toward 28 if you still hit OOM; the lowest number that fits is the fastest.

**Worth knowing**

- **Results don't change.** Only where the weights live changes; training values and generated images match a run without it. This is not precision traded for VRAM.
- **Generation pays more than training.** Every sampling step moves all swapped blocks again, so the cost grows with step count; training pays it once per step, about 4%.
- **The RAM is pinned** and unavailable to other programs (about 0.4 GB per block on fp8, 0.8 GB on bf16). If there isn't enough, the run fails with an error before it starts rather than thrashing your machine.
- **Changing the number reloads the model**, so the next run starts a little slower — swapped-out blocks are kept off the GPU from load time onward.

### The VRAM policy for test generation

This one decides which **model** yields to which (text encoder vs. DiT); block swap works inside a single model. They are separate and combine freely. Under **Settings → Test → VRAM policy**:

- **Default (release after use)**: every prompt for the task is pre-encoded into a cache up front, then the text encoder is released, so it costs nothing during sampling; a later new prompt reloads it from disk (about 3 seconds). Right for most setups.
- **Save VRAM**: on top of that, models are forced to take turns during encoding, for the lowest peak — use this for fp8 on a 16 GB card, and below 12 GB together with block swap. The cost is a few extra seconds of CPU↔GPU transfer per image.
- **Performance**: the text encoder stays resident the whole time, so switching prompts costs nothing — for large-VRAM GPUs.

The same section has the **RAM / VRAM guard** (on by default): before loading, it checks available RAM and free VRAM against the actual weight file size and aborts with an error if either falls short, instead of freezing the machine in page thrash. Keep it on if you're short on memory.

### LoRA merge precision

On an fp8 base, LoRAs are **merged into the weights**. This option (**Settings → Test**) controls the temporary precision of that merge:

- **fp32** (default): matches how ComfyUI computes it, so cross-app reproduction stays exact.
- **bf16**: halves the delta math, so switching LoRAs is faster, at the cost of slight numerical differences in the merged result.

It only affects loading or switching LoRAs on an fp8 base; sampling uses the merged fp8 weights either way, so it **does not change sampling VRAM** — the overall peak is dominated by the base model, the VAE and the work buffers. Anima and bf16 bases use dynamic LoRA and are unaffected.

Full reasoning, cost model and all measurements: [Training tips → Block swap](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/blob/master/docs/user-guide/training-tips.md#block-交换换出到内存的层数).
