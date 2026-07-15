# Getting Started

Run a LoRA training pipeline end to end. This is the full version of the [README](../../README.en.md) "Quick start".

## Prerequisites

These are **not** installed by Studio and must be ready beforehand:

- **NVIDIA GPU driver + CUDA runtime** (16 GB+ VRAM recommended, 8 GB barely works; AMD GPUs / Apple Silicon are not supported)
- **Python 3.10+** (callable as `python` from PATH)
- **Node.js 18+** (for frontend build, with `npm` on PATH)
- **Git**

Hardware details: see [README → Hardware requirements](../../README.en.md#hardware-requirements).

## Start Studio

```bash
git clone https://github.com/WalkingMeatAxolotl/AnimaLoraStudio
cd AnimaLoraStudio

# Windows
studio.bat

# Linux / macOS
./studio.sh
```

On first run, the launcher automatically: creates `venv/` → installs the matching CUDA torch (cu118 through cu130) based on the detected GPU driver → installs `requirements.txt` → builds the frontend → starts the backend → opens the browser to <http://127.0.0.1:8765/>. A first-run onboarding modal then walks through installing base models, ONNX Runtime, and training acceleration with one click.

> If GPU detection falls back to CPU torch, reinstall the CUDA build from Settings → System → PyTorch with one click, or specify it explicitly via `studio.bat --torch cu128` (or `studio.sh --torch cu128`).

### Alternative launch

Equivalent to the above, useful when calling `python` directly:

```bash
python -m studio              # Build frontend if missing, then start backend
python -m studio dev          # Watch mode: vite 5173 + uvicorn 8765 --reload
python -m studio build        # Build frontend only
python -m studio test         # pytest + vitest
```

## Download models

After launch, go to **Settings → Models** and click to download all required weights and tokenizers (defaults to `./models/`):

| Item | Source | Path | Size |
|---|---|---|---|
| Anima base model (latest = 1.0) | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | `models/diffusion_models/` | ~4 GB |
| Qwen-Image VAE (shared by Anima / Krea 2) | Same | `models/vae/` | ~250 MB |
| Qwen3-0.6B-Base text encoder | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | `models/text_encoders/` | ~1.2 GB |
| T5 tokenizer (3 files only, no weights) | [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) | `models/t5_tokenizer/` | <1 MB |
| Krea 2 Raw (LoRA training / samples during training) | [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) | `models/diffusion_models/krea2-raw-bf16.safetensors` | ~26.3 GB |
| Krea 2 Turbo (inference testing) | [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) | `models/diffusion_models/krea2-turbo-bf16.safetensors` | ~26.3 GB |
| Krea 2 text encoder | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | `models/text_encoders/Qwen_Qwen3-VL-4B-Instruct/` | ~8.89 GB |

Krea 2 weights are governed by the [Krea 2 Community License](https://huggingface.co/krea/Krea-2-Raw/blob/main/LICENSE.pdf). Training and samples during training reuse Raw; Turbo is the inference-testing model. A LoRA trained on Raw can be loaded directly on Turbo. Krea 2 reuses Anima's existing VAE, so no duplicate VAE download is needed. When ModelScope is selected, Raw and Turbo are downloaded from [Comfy-Org/Krea-2](https://www.modelscope.cn/models/Comfy-Org/Krea-2).

WD14 tagger models are not in this list — they are auto-downloaded from HF to `models/wd14/` on first use of the tagging step.

**Mirrors / slow connections**: switch the HuggingFace endpoint to a self-hosted mirror under Settings → Training → HuggingFace → endpoint, or switch the download source to ModelScope under Settings → Training → Download source (requires `pip install modelscope`).

Or via CLI (shares the same code as the UI; full flags in [tools/README.md](../../tools/README.md)):

```bash
python tools/download_models.py                   # Anima (default, official HF)
python tools/download_models.py --family krea2    # Krea 2 Raw + shared VAE + Qwen3-VL
python tools/download_models.py --family krea2 --variant turbo
python tools/download_models.py --endpoint URL    # Use self-hosted mirror
python tools/download_models.py --modelscope      # Use ModelScope
```

## Pipeline: follow the stepper

Open <http://127.0.0.1:8765/>, click "+ New project" on the projects page, and the sidebar stepper guides you through 8 steps (those marked ✱ are skippable):

1. **Download** — Booru scraping (fill in Gelbooru / Danbooru credentials in Settings first) or local jpg / png / zip upload.
2. **Curate** — download / train dual panels, multi-select to copy images into train/, subfolder management.
3. **Preprocess** ✱ — overview (multi-select + one-click undo) + duplicate review + upscale (ESRGAN / Real-ESRGAN presets) + crop (manual box + auto AR-clustering prefill) + inpaint. Skip if not needed.
4. **Tag** — WD14 / CLTagger / LLM (OpenAI-compatible, including a JoyCaption preset) + thresholds, automatic GPU EP fallback; a trigger word at the top is auto-injected into every caption and sample image.
5. **Tag editor** — cached mode + restore points, bulk add / delete / replace, per-image edits.
6. **Regularization set** ✱ — two generation modes: **AI prior generation** (default, the base model produces the reg set with no LoRA) or **Booru reverse search** (reverse-search Booru by tag distribution + auto WD14 tagging + aspect-ratio clustering). mirror / flat structure, editable / deletable / auto-dedup / dual tagger.
7. **Train** — pick a preset to copy into the version's private config, edit parameters (autosaved with 600ms debounce, no save button), submit to the queue. The picker label shows "· customized" once the config has diverged; the preset pool is never modified. Simple / Advanced modes.
8. **Test** — single-image / XY matrix / inference daemon.

View tasks on the **Queue** page; open **task detail** for logs / monitoring / output (with one-click full zip download).

## Test your LoRA + ComfyUI

After training, the sidebar **Test** page runs single-image / XY matrix / inference daemon for LoRA evaluation. Prompts can be pulled directly from the training set, eliminating round trips to ComfyUI.

The LoRA weights produced are already in `lora_unet_*` format and can be **dropped directly into ComfyUI** without any conversion.

## Next

- Training parameters / VRAM config / algorithm options → [training-tips.md](training-tips.md)
- Tag format and best practices → [tagging-guide.md](tagging-guide.md)
- Optimizer starting points → [optimizers.md](optimizers.md)
