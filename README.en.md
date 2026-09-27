# AnimaLoraStudio

[![中文](https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-lightgrey)](README.md) [![English](https://img.shields.io/badge/lang-English-blue)](README.en.md) [![Version](https://img.shields.io/badge/version-0.28.0-blue)](CHANGELOG.md) [![License](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)

**End-to-end pipeline**: Booru scraping → curation → tagging → regularization set → training → image-gen testing, all in one browser panel. Trains LoRAs for two model families: [Anima](https://huggingface.co/circlestone-labs/Anima) (Cosmos DiT, anime-specialized, lightweight) and [Krea 2](https://huggingface.co/krea/Krea-2-Raw) (12.9B single-stream MMDiT; train on Raw, test fast on Turbo).

![Studio training page (older layout illustration)](docs/images/studio-train-en.png)

> This screenshot shows an older layout; see the [Getting Started guide](docs/user-guide/getting-started.en.md) for the current training workflow.

## Features

- **One-stop pipeline**: Booru scraping / curation / preprocessing (dedup · upscale · crop · retouch · [automatic head masks](docs/user-guide/auto-head-mask.en.md)) / tagging / regularization set / training / image-gen testing — all in one browser panel, guided by a stepper.
- **Two model families**: Anima and Krea 2 share the same workflow; switch families right in the training config (weight paths and family defaults are recomputed with an itemized confirmation), options are filtered per family, and one project can hold versions of both families.
- **Three taggers**: WD14, CLTagger (local ONNX), LLM (OpenAI-compatible, long captions); a trigger word entered once is auto-injected into every caption.
- **Separate LLM configuration stores**: presets, ordinary settings, and credentials are stored independently, with built-in template overrides, atomic writes, and backup protection; queued LLM tagging jobs freeze their preset recipe.
- **Booru scraping**: native Gelbooru / Danbooru (Cloudflare-compatible UA, rate limiting, account auth).
- **Automatic regularization sets**: reverse-search by your training set's tag distribution + aspect-ratio clustering, or AI priors from the base model (no LoRA needed).
- **Project / Version two-tier management**: one project holds multiple versions sharing downloaded data, with independent config / output; presets fork both ways with the global pool; projects can be archived, restored, or permanently deleted in batches.
- **Multi-task queue**: training, generation and data jobs in one unified ledger; enqueue, scheduled start, pause (resume from the last epoch boundary), resume, and queue-level hold.
- **Built-in image-gen testing**: single-image / XY-grid eval + a resident inference daemon; one layered catalog covers project checkpoints and external LoRAs, the XY-axis drawer supports checkpoint / LoRA-strength axes and drag ordering, and the Booru gallery can tag a picked image back into your prompt; fp8 base-model inference and LoRA merge are bit-for-bit aligned with ComfyUI; community LoRAs in PEFT / comfy key format (the civitai ecosystem) load directly; output `lora_unet_*` drops straight into ComfyUI, no conversion.
- **LoRA evaluation**: batch-generate validation images for each training checkpoint and score them (CLIP / DINO / CCIP / WD14 tag recall, with a base-model baseline), plus a sample grid for eyeballing; one evaluation = one queue job, resumable after interruption.
- **fp8 & VRAM orchestration**: official fp8 weights support Krea 2 training and inference with lower weight residency; block swap keeps some layers in RAM and transfers them for computation. Minimum GPU capacity needs full-configuration testing, not a guarantee extrapolated from larger cards. Includes text pre-encoding with encoder release, three VRAM policies, and a loading-time RAM guard.
- **Rich training algorithms**: multiple loss / timestep sampling / optimizers (AdamW · Lion · Prodigy · SOAP, etc.) / LoRA · LyCORIS adapters; standard LoRA / LoHa can opt into a restricted [experimental Triton backend](docs/user-guide/triton-backend.md), while Torch remains the default. See [Training algorithm options](docs/user-guide/training-tips.md#训练算法选项).
- **Self-healing setup + in-app self-update**: GPU-aware torch on first install, dependency hash checks, git pull / restart / rollback.
- **Bilingual**: pick a language on first launch, switchable in Settings.

> The training core (`runtime/`) is decoupled from the Studio backend and runs standalone via CLI; model families and adapters / optimizers / schedulers / losses / samplers / timestep sampling are all extensible plugin registries (see [ADR 0003](docs/adr/0003-anima-train-refactor.md)).

## Quick start

**Prerequisites** (install yourself): NVIDIA GPU + CUDA · Python 3.10+ · Node.js 18+ · Git.

```bash
git clone https://github.com/WalkingMeatAxolotl/AnimaLoraStudio
cd AnimaLoraStudio
studio.bat          # Windows
./studio.sh         # Linux / macOS
```

First run automatically creates `venv/` → installs GPU-matched CUDA torch → builds the frontend → starts the backend → opens <http://127.0.0.1:8765/>, with an onboarding modal to one-click install the Anima starter set. Once open, go to the model download center under **Settings → Training** and download the weights for your model family (defaults to `./models/`).

→ Full walkthrough (launch options / model download / mirrors / pipeline steps): see the **[Getting Started guide](docs/user-guide/getting-started.en.md)**.

→ Updating from v0.26 or earlier? Read **[Upgrading to v0.27](docs/user-guide/upgrading-v0.27.en.md)** first for the configuration migration, backups, and LyCORIS dependency changes. Updating from v0.27 to the current release requires no additional data migration.

## Hardware requirements

- **GPU**: NVIDIA (AMD / Apple Silicon not supported), per family:
  - **Anima**: **16 GB+ VRAM recommended**. On smaller GPUs, reduce batch / resolution, adjust sampling, or try block swap; there is no universal minimum-capacity guarantee. Low allocations on a large GPU do not prove **full training on a 6 GB card**; validate loading, training, and sampling peaks.
  - **Krea 2** (12.9B): prefer official fp8 for training on 24 GB-class GPUs; bf16 on 32 GB-class GPUs still needs peak headroom. On 16 GB-class GPUs, try fp8, low batch / gradient checkpointing, and block swap; prefer Save VRAM for generation. **12 GB training and 8 GB generation are trial configurations awaiting full-path validation on those GPUs**, not capacity guarantees. The old “about 4% slower” figure compared 14→28 blocks in a particular sampling task, not training with swap off versus 28 blocks. See [training tips (Chinese)](docs/user-guide/training-tips.md#显存参考) for conditions and measurement limits.
- **RAM**: 16 GB+; consider 32 GB+ for Krea 2, but available RAM must also cover model loading, caches, the OS, and other programs. Block swap pins the transferred weights (about 11 GB for all Krea 2 fp8 blocks, about 3.6 GB for Anima); guards do not guarantee against later OOM or paging.
- **Storage**: SSD strongly recommended (frequent latent-cache + sample IO); budget disk space for Krea 2 weights (Raw / Turbo bf16 26.3 GB each, official fp8 13.1 GB each, text encoder 5.2–8.9 GB)

## Documentation

Entry point: [docs/README.md](docs/README.md).

- **Getting started** → [getting-started.md](docs/user-guide/getting-started.en.md)
- **User guide** → [tag format](docs/user-guide/tagging-guide.md) · [training tips / algorithms](docs/user-guide/training-tips.md) · [random seeds](docs/user-guide/random-seeds.md) · [experimental Triton backend](docs/user-guide/triton-backend.md) · [optimizers](docs/user-guide/optimizers.md) · [caption format](docs/user-guide/caption-format.md)
- **Architecture** → [pipeline overview](docs/architecture/studio-pipeline.md) · [project structure](docs/architecture/project-structure.en.md) · [studio internals](studio/README.md)
- **CLI tools** → [tools/README.md](tools/README.md)
- **Contributing** → [CONTRIBUTING.md](CONTRIBUTING.md) · [docs/AGENTS.md](docs/AGENTS.md)
- **Decision records** → [docs/adr/](docs/adr/) · **Changelog** → [CHANGELOG.md](CHANGELOG.md)

## Upstream and credits

- Core training scripts derived from [**Moeblack/AnimaLoraToolkit**](https://github.com/Moeblack/AnimaLoraToolkit)
- Anima base model / VAE: [circlestone-labs / Anima](https://huggingface.co/circlestone-labs/Anima)
- Krea 2 base models: [krea / Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) · [Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) (official fp8 quantizations from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2))
- Text encoders: [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) (Anima) and [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (Krea 2), by the Qwen team
- Krea 2 training / sampling implementation references and partially derives from [**kohya-ss/musubi-tuner**](https://github.com/kohya-ss/musubi-tuner) (Apache-2.0); model structure derives from ComfyUI, cross-checked against [HuggingFace diffusers](https://github.com/huggingface/diffusers)
- OrthoLoRA / T-LoRA adapters derived from [**sorryhyun/anima_lora**](https://github.com/sorryhyun/anima_lora) (MIT); algorithm from the [ControlGenAI/T-LoRA](https://github.com/ControlGenAI/T-LoRA) paper and official implementation
- Automagic optimizer ported from [**ostris/ai-toolkit**](https://github.com/ostris/ai-toolkit) (MIT); bf16 Kahan path references [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)
- Image-gen / sampling path aligned with and derived from [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) (GPL-3.0)

Full third-party algorithm / code / paper attribution: see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Released under **GPL-3.0** overall (includes / derives from ComfyUI's GPL-3.0 code). It also bundles some Apache-2.0 third-party implementations (NVIDIA Cosmos / Wan2.1 / musubi-tuner derivations, etc.) — see `LICENSE` (GPL-3.0) / `LICENSE-APACHE` / [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); please keep the original file headers.

**Model weights** (Anima / Krea 2 / Qwen / VAE) have their own terms: Anima-related weights carry Non-Commercial restrictions; Krea 2 weights are governed by the [Krea 2 Community License](https://huggingface.co/krea/Krea-2-Raw/blob/main/LICENSE.pdf). Defer to each model card / HF repo license.
