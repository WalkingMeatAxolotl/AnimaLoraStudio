# AnimaLoraStudio

[![中文](https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-lightgrey)](README.md) [![English](https://img.shields.io/badge/lang-English-blue)](README.en.md) [![Version](https://img.shields.io/badge/version-0.15.0-blue)](CHANGELOG.md) [![License](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)

**End-to-end pipeline**: Booru scraping → curation → tagging → regularization set → training → image-gen testing, all in one browser panel. Tuned for [Anima](https://huggingface.co/circlestone-labs/Anima) (Cosmos DiT, anime-specialized).

![Studio training page](docs/images/studio-train-en.png)

## Features

- **One-stop pipeline**: Booru scraping / curation / preprocessing (dedup · upscale · crop) / tagging / regularization set / training / image-gen testing — all in one browser panel, guided by a stepper.
- **Three taggers**: WD14, CLTagger (local ONNX), LLM (OpenAI-compatible, long captions); a trigger word entered once is auto-injected into every caption and sample image.
- **Booru scraping**: native Gelbooru / Danbooru (Cloudflare-compatible UA, rate limiting, account auth).
- **Automatic regularization sets**: reverse-search by your training set's tag distribution + aspect-ratio clustering, or AI priors from the base model (no LoRA needed).
- **Project / Version two-tier management**: one project holds multiple versions sharing downloaded data, with independent config / output; presets fork both ways with the global pool.
- **Multi-task queue**: enqueue, pause (resume from the last epoch boundary), resume, and queue-level hold.
- **Built-in image-gen testing**: single-image / XY-grid eval + a resident inference daemon; output `lora_unet_*` drops straight into ComfyUI, no conversion.
- **Rich training algorithms**: multiple loss / timestep sampling / optimizers (AdamW · Lion · Prodigy · SOAP, etc.) / LoRA · LyCORIS adapters — see [Training algorithm options](docs/user-guide/training-tips.md#训练算法选项).
- **Self-healing setup + in-app self-update**: GPU-aware torch on first install, dependency hash checks, git pull / restart / rollback.
- **Bilingual**: pick a language on first launch, switchable in Settings.

> The training core (`runtime/`) is decoupled from the Studio backend and runs standalone via CLI; adapter / optimizer / scheduler / loss / sampler are five extensible plugin registries (see [ADR 0003](docs/adr/0003-anima-train-refactor.md)).

## Quick start

**Prerequisites** (install yourself): NVIDIA GPU + CUDA (16 GB+ VRAM recommended, 8 GB barely works) · Python 3.10+ · Node.js 18+ · Git.

```bash
git clone https://github.com/WalkingMeatAxolotl/AnimaLoraStudio
cd AnimaLoraStudio
studio.bat          # Windows
./studio.sh         # Linux / macOS
```

First run automatically creates `venv/` → installs GPU-matched CUDA torch → builds the frontend → starts the backend → opens <http://127.0.0.1:8765/>, with an onboarding modal to one-click install models. Once open, go to **Settings → Models** to download the weights (default `./models/`).

→ Full walkthrough (launch options / model download / mirrors / pipeline steps): see the **[Getting Started guide](docs/user-guide/getting-started.md)**.

## Hardware requirements

- **GPU**: NVIDIA, **16 GB+ VRAM recommended** (RTX 4060Ti 16G / 4070Ti / 4080 / 3090 / 4090 / 5090, etc.); **8 GB barely works** (turn off sample output + reduce batch / resolution; noticeably slower). AMD / Apple Silicon not supported.
- **RAM**: 16 GB+
- **Storage**: SSD strongly recommended (frequent latent-cache + sample IO)

## Documentation

Entry point: [docs/README.md](docs/README.md).

- **Getting started** → [getting-started.md](docs/user-guide/getting-started.md)
- **User guide** → [tag format](docs/user-guide/tagging-guide.md) · [training tips / algorithms](docs/user-guide/training-tips.md) · [optimizers](docs/user-guide/optimizers.md) · [caption format](docs/user-guide/caption-format.md)
- **Architecture** → [pipeline overview](docs/architecture/studio-pipeline.md) · [project structure](docs/architecture/project-structure.md) · [studio internals](studio/README.md)
- **CLI tools** → [tools/README.md](tools/README.md)
- **Contributing** → [CONTRIBUTING.md](CONTRIBUTING.md) · [docs/AGENTS.md](docs/AGENTS.md)
- **Decision records** → [docs/adr/](docs/adr/) · **Changelog** → [CHANGELOG.md](CHANGELOG.md)

## Upstream and credits

- Core training scripts derived from [**Moeblack/AnimaLoraToolkit**](https://github.com/Moeblack/AnimaLoraToolkit)
- Base model / VAE: [circlestone-labs / Anima](https://huggingface.co/circlestone-labs/Anima)
- OrthoLoRA / T-LoRA adapters derived from [**sorryhyun/anima_lora**](https://github.com/sorryhyun/anima_lora) (MIT); algorithm from the [ControlGenAI/T-LoRA](https://github.com/ControlGenAI/T-LoRA) paper and official implementation
- Automagic optimizer ported from [**ostris/ai-toolkit**](https://github.com/ostris/ai-toolkit) (MIT); bf16 Kahan path references [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)
- Image-gen / sampling path aligned with and derived from [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) (GPL-3.0)

Full third-party algorithm / code / paper attribution: see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Released under **GPL-3.0** overall (includes / derives from ComfyUI's GPL-3.0 code). It also bundles some Apache-2.0 third-party implementations (NVIDIA Cosmos / Wan2.1, etc.) — see `LICENSE` (GPL-3.0) / `LICENSE-APACHE` / [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); please keep the original file headers.

**Model weights** (Anima / Qwen / VAE) have their own terms (including Non-Commercial restrictions) — defer to each model card / HF repo license.
