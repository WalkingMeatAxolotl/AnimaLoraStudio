# AnimaLoraStudio

[![中文](https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-lightgrey)](README.md) [![English](https://img.shields.io/badge/lang-English-blue)](README.en.md) [![Version](https://img.shields.io/badge/version-0.10.3-blue)](CHANGELOG.md)

**End-to-end pipeline**: scrape from Booru → curate → tag → reg set → train → image testing, all driven from a single browser panel. Optimized for [Anima](https://huggingface.co/circlestone-labs/Anima) (Cosmos DiT, anime-tuned) training.

## Features

### Data preparation

- **Booru scraping**: native support for Gelbooru / Danbooru, with Cloudflare-compatible UA, dual token-bucket rate limiting, and Danbooru account authentication
- **Automatic regularization set generation**: Booru reverse search based on training tag distribution + aspect-ratio clustering; or direct generation from the base model (no LoRA required)
- **Three taggers**: WD14 (local ONNX), CLTagger (external contribution, local ONNX), LLM (OpenAI-compatible API with long captions)
- **Automatic trigger word injection**: enter once at the tagging step, automatically written into every caption and the training sample prompt
- **Pre-training duplicate review**: perceptual-hash scan groups similar / near-duplicate images for manual keep / remove review; soft-deleted entries remain recoverable

### Training and experiment management

- **Project / Version two-tier model**: a project can hold multiple versions sharing the same downloaded data while keeping config / output / reg set independent
- **Bidirectional preset flow**: training configurations can fork between a version's private config and the global preset pool
- **Multi-task queue**: task queueing, pause (saves progress at the most recent epoch end), resume, and queue hold
- **Built-in image testing**: single-image testing and XY matrix evaluation directly in Studio after training, with a long-lived inference daemon to avoid repeated model loading
- **Training set bundles**: one-click export / import (local download or server path), for moving projects between machines

### Training algorithms

- **Loss functions**: MSE / Huber, with configurable weighting curves (min_snr / cosmap / detail_inv_t, etc.)
- **Timestep sampling**: uniform / logit_normal / mode / mixed_uniform, with configurable schedule shift
- **InfoNoise adaptive sampling (optional)**: inverse-CDF timestep sampler based on I-MMSE
- **Optimizers**: AdamW / Prodigy / Prodigy+ScheduleFree
- **Adapter**: LoRA + LyCORIS LoKr (via the [lycoris-lora](https://github.com/KohakuBlueleaf/LyCORIS) library, including DoRA / rs-LoRA / dropout)
- **Per-layer rank**: `lora_rank_rules` assigns different ranks per layer-name regex, useful for biasing parameter budget by module importance
- **Attention backends**: xformers / flash_attn / PyTorch SDPA

### Engineering experience

- **Self-healing environment**: first install automatically selects a GPU-compatible torch (cu118 through cu130), venv synchronizes with requirements.txt via hash comparison, Windows lockfile handling, automatic ONNX CUDA → CPU fallback on failure
- **In-app updates**: Settings supports git pull, restart, and rollback; both master (stable) and dev (rolling) channels
- **Acceleration backend switching**: one-click install of xformers / flash_attn wheels from Settings
- **Internationalization**: bilingual UI (English / Chinese), language picker on first launch, switchable from Settings

![Studio training page](docs/images/studio-train.png)

### Architecture

- The **training core** (`runtime/anima_train.py`) is decoupled from the Studio backend and can be invoked via standalone CLI or spawned by Studio as a subprocess
- **Extensible plugins**: five plugin registries (adapter / optimizer / scheduler / loss / timestep_sampler); adding a custom variant requires only a new builder function, dictionary registration, and a schema Literal (see [`runtime/training/README.md`](runtime/training/README.md) and [ADR 0003](docs/adr/0003-anima-train-refactor.md))

### Studio Web workbench (`studio/`)

8-step pipeline + tool pages:

1. **Download** — Booru scraping / local jpg / png / zip upload
2. **Preprocess** (optional) — multi-tool pipeline: overview (multi-select + one-click undo) + duplicate review (perceptual-hash grouped manual review) + upscale (ESRGAN / Real-ESRGAN presets via ModelScope / HF) + crop (manual rect drawing + smart AR-clustered prefill aligned with sd-scripts training buckets) + inpaint
3. **Curate** — dual download / train panels with multi-select copy / remove and subfolder management
4. **Tag** — choose from WD14 / CLTagger / LLM with automatic GPU EP fallback; trigger_word input at the top
5. **Tag editor** — cached mode with restore points, bulk add / delete / replace
6. **Regularization set** (optional) — Booru reverse search / AI prior generation
7. **Train** — bidirectional preset flow, queues immediately on submit; config edits autosave; Simple / Advanced modes
8. **Image testing** — single image / XY matrix / inference daemon

Common panels:

- Queue / task detail (logs / monitoring / output download / full zip)
- Real-time training monitoring (loss / lr curves + sample images by step)
- Topbar system resources (CPU / GPU / memory / VRAM)
- Settings (credentials / model management / acceleration backend / WandB / auto-update / display)
- Dark / light mode and font density switching

---

## Upstream and credits

- Core training scripts derived from [**Moeblack/AnimaLoraToolkit**](https://github.com/Moeblack/AnimaLoraToolkit).
- Base model / VAE: [circlestone-labs / Anima](https://huggingface.co/circlestone-labs/Anima)

---

## Quick start

### 0. System prerequisites (install yourself)

These are **not** installed by Studio and must be ready beforehand:

- **NVIDIA GPU driver + CUDA runtime** (**16 GB+ VRAM recommended, 8 GB minimum**; AMD GPUs / Apple Silicon are not supported)
- **Python 3.10+** (callable as `python` from PATH)
- **Node.js 18+** (for frontend build, with `npm` on PATH)
- **Git**

### 1. Clone and start Studio

```bash
git clone https://github.com/WalkingMeatAxolotl/AnimaLoraStudio
cd AnimaLoraStudio

# Windows
studio.bat

# Linux / macOS
./studio.sh
```

On first run, the launcher automatically: creates `venv/` → installs the matching CUDA torch (cu118 through cu130) based on detected GPU driver → installs `requirements.txt` → installs onnxruntime based on GPU detection → builds the frontend → starts the backend → opens the browser to <http://127.0.0.1:8765/studio/>.

> If GPU detection falls back to CPU torch, you can reinstall the CUDA build from Settings → System → PyTorch with one click, or specify it explicitly via `studio.bat --torch cu128` (or `studio.sh --torch cu128`).

Alternative launch (equivalent, useful when calling `python` directly):

```bash
python -m studio              # Build frontend if missing, then start backend
python -m studio dev          # Watch mode: vite 5173 + uvicorn 8765 --reload
python -m studio build        # Build frontend only
python -m studio test         # pytest + vitest
```

### 2. Download models from Studio

After launch, go to **Settings → Models** and click to download all required weights and tokenizers.

The default source is the official `huggingface.co`. Users with slow connections can go to **Settings → Training → HuggingFace → endpoint** and switch to "Custom URL" to paste a self-hosted mirror, or switch to **Settings → Training → Download source → ModelScope** (direct connection to ModelScope, requires `pip install modelscope`). CLI users can override via `python tools/download_models.py --endpoint URL` or `--modelscope`.

Downloaded content (defaults to `./models/`):

| Item | Source | Path | Size |
|---|---|---|---|
| Anima base model (latest = 1.0) | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | `models/diffusion_models/` | ~4 GB |
| Anima VAE | Same | `models/vae/` | ~250 MB |
| Qwen3-0.6B-Base text encoder | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | `models/text_encoders/` | ~1.2 GB |
| T5 tokenizer (3 files only, no weights) | [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) | `models/t5_tokenizer/` | <1 MB |

Or via CLI (shares the same code as the UI):

```bash
python tools/download_models.py                   # Download everything (official HF)
python tools/download_models.py --endpoint URL    # Use self-hosted mirror
python tools/download_models.py --modelscope      # Use ModelScope
python tools/download_models.py --variant preview3-base
python tools/download_models.py --skip-main --skip-vae
python tools/download_models.py --output /data/anima
```

WD14 tagger models are not in this list — they are auto-downloaded from HF to `models/wd14/` on first use of the tagging step.

### 3. Follow the stepper

Open <http://127.0.0.1:8765/studio/>:

1. Click "+ New project" on the projects page
2. **① Download**: Booru scraping (fill in Gelbooru / Danbooru credentials in Settings first) or local zip upload
3. **② Curate**: dual grid, select images to copy into train/
4. **③ Tag**: choose WD14 / CLTagger / LLM (OpenAI-compatible, including a JoyCaption preset), set thresholds, run automatically
5. **④ Tag editor**: bulk add / delete / replace, per-image edits, automatic restore points
6. **⑤ Regularization set**: two generation modes —
   - **Booru reverse search**: reverse search Booru based on tag distribution, with automatic WD14 tagging and aspect-ratio clustering
   - **AI prior generation**: use the base model directly to generate the reg set (no LoRA required)
7. **⑥ Train**: pick a preset to copy into the version's private config, edit parameters (autosaved with 600ms debounce, no save button), submit to the queue. The picker label shows "· customized" once the config has diverged from the source preset; the preset pool is never modified
8. View tasks on the **Queue** page; open **task detail** for logs / monitoring / output (with one-click full zip download)

After training, the sidebar **Test** page provides single-image generation / XY matrices / inference daemon for LoRA evaluation. Prompts can be pulled directly from the training set, eliminating round trips to ComfyUI.

The LoRA weights produced are already in `lora_unet_*` format and can be **dropped directly into ComfyUI** without any conversion.

---

## Project structure

```
AnimaLoraStudio/
├── runtime/                       # Anima runtime core (standalone process; launched by Studio as a subprocess or run via CLI)
│   ├── anima_train.py             # Training entry
│   ├── training/                  # Training stack subpackage: context / phases / loop / sample_runner
│   │   ├── adapters/              # plugin: lokr / loha / lora
│   │   ├── optimizers/            # plugin: adamw / prodigy / prodigy_plus_schedulefree
│   │   ├── schedulers/            # plugin: cosine / cosine_with_restart / none
│   │   ├── inference_samplers/    # plugin: er_sde, etc.
│   │   └── phases/                # bootstrap / models / dataset / optimizer / resume / finalize
│   ├── anima_generate.py          # Image generation: single image / XY matrix
│   ├── anima_daemon.py            # Inference daemon: keeps the base model and LoRA loaded in GPU
│   ├── anima_reg_ai.py            # AI prior generation: no LoRA, base model produces reg set
│   └── train_monitor.py           # Training state writer
├── studio/                        # AnimaStudio Web workbench (FastAPI + React)
│   ├── server.py                  # Daemon entry
│   ├── services/                  # Business logic (uploads / tagging / reg set / inference_core /
│   │                              #   torch_setup / xformers_setup / flash_attention_setup, etc.)
│   ├── workers/                   # Background subprocesses (download / tag / reg_build)
│   └── web/                       # React + Vite frontend
├── tools/                         # User CLI / launcher-time setup helpers
│   ├── download_models.py         # One-click download of base model / VAE / Qwen3 / T5 tokenizer
│   ├── install_flash_attn.py     # One-click flash_attn wheel install
│   ├── select_torch_index.py      # GPU-aware torch CUDA index selection (auto-called at launch)
│   ├── check_requirements_changed.py  # venv stale detection (auto-called at launch)
│   └── validate_local_models.py   # Validate local Qwen / T5 for offline loading
├── docs/                          # Three sections: user-guide / architecture / adr (see docs/README.md)
├── utils/                         # Shared utilities for anima_train (model loader / optimizer / lycoris_adapter / ...)
└── models/                        # Model code + bundled tokenizer files + downloaded weights (mixed)
    ├── anima_modeling*.py         # tracked: PyTorch implementation of Anima Cosmos transformer
    ├── cosmos_predict2_modeling.py
    ├── wan/vae2_1.py              # tracked: Wan2.1 VAE implementation
    ├── text_encoders/             # tracked: Qwen tokenizer files + user-downloaded model.safetensors
    ├── t5_tokenizer/              # tracked: T5 tokenizer files (no weights)
    ├── diffusion_models/          # User-downloaded Anima base model (gitignored)
    ├── vae/                       # User-downloaded VAE weights (gitignored)
    ├── wd14/                      # WD14 ONNX models (auto-downloaded from HF, gitignored)
    └── taeflux/                   # TAEFlux intermediate preview weights (gitignored)
```

Runtime data (gitignored):

- `studio_data/` — SQLite + user presets + task logs + per-task monitor state + samples
- `models/diffusion_models/`, `models/vae/`, `models/wd14/` — large weight files
- `studio_data/projects/{id}-{slug}/versions/{label}/output/` — trained LoRA artifacts

---

## CLI tools

The CLIs under `tools/` share the same `services/` code as the Studio UI, convenient for headless environments:

| Script | Purpose |
|---|---|
| `tools/download_models.py` | One-click download of base model / VAE / Qwen3 / T5 tokenizer. Multiple variants supported, with `--no-mirror` / `--endpoint URL` flags |
| `tools/install_flash_attn.py` | Auto-select and install the flash_attn wheel matching your torch ABI |
| `tools/select_torch_index.py` | Detect GPU and recommend the matching PyTorch CUDA index URL (cu130 / cu128 / ...) |
| `tools/validate_local_models.py` | Validate that local Qwen / T5 can be loaded offline |

The runtime scripts under `runtime/` (`anima_train` / `anima_generate` / `anima_daemon` / `anima_reg_ai`) can also be run standalone via CLI — see each script's top-level docstring.

---

## Documentation

Documentation entry: [docs/README.md](docs/README.md). Three sections:

**User-facing** ([`docs/user-guide/`](docs/user-guide/))

- [tagging-guide.md](docs/user-guide/tagging-guide.md) — Anima tag format and best practices
- [training-tips.md](docs/user-guide/training-tips.md) — Training parameters / VRAM configuration matrix / FAQs
- [regularization.md](docs/user-guide/regularization.md) — How regularization set generation works
- [caption-format.md](docs/user-guide/caption-format.md) — JSON tag format + category shuffle

**Developer-facing**

- [docs/architecture/studio-pipeline.md](docs/architecture/studio-pipeline.md) — Cross-step Studio architecture overview
- [studio/README.md](studio/README.md) — Studio internal module structure

**Collaboration conventions**

- [CONTRIBUTING.md](CONTRIBUTING.md) — Workflow / branches / commits / PRs / releases
- [docs/AGENTS.md](docs/AGENTS.md) — Code quality conventions and AI agent collaboration

**Historical decisions** ([`docs/adr/`](docs/adr/))

- Records of "why we chose X over Y"; preserved even after the decision lands

**Version history**

- [CHANGELOG.md](CHANGELOG.md)

---

## Version

Current version is **0.10.3**. See [CHANGELOG.md](CHANGELOG.md) for the full history. The Settings → System → version card inside Studio allows one-click upgrade to the latest version.

---

## Hardware requirements

- **GPU**: NVIDIA, **16 GB+ VRAM recommended** (RTX 4060Ti 16G / 4070Ti / 4080 / 5070+ / 3090 / 4090 / 5090, etc.); **8 GB is the minimum** (some laptop GPUs are confirmed working, requires disabling sample output + reducing batch / resolution, with noticeably slower training). System GPU usage is low; VRAM is mostly for training. AMD GPUs / Apple Silicon are not supported
- **RAM**: 16 GB+
- **Storage**: SSD strongly recommended (latent cache + sample output is I/O heavy)

---

## License

The repository is released under **GPL-3.0** as a whole (includes / derives from ComfyUI's GPL-3.0 code).

Some Apache-2.0 third-party implementations (NVIDIA Cosmos / Wan2.1, etc.) are also included; please preserve their original file headers. See:

- `LICENSE` (GPL-3.0)
- `LICENSE-APACHE` (Apache-2.0 text, applies to in-repo Apache-2.0 components)
- `THIRD_PARTY_NOTICES.md`

**Model weights** (Anima / Qwen / VAE) have their own terms (including Non-Commercial restrictions); refer to the corresponding model card / HF repo for the applicable license.
