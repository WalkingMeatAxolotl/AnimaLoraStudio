# Project structure

Top-level repository layout. For Studio's internal module structure see [`studio/README.md`](../../studio/README.md); for the cross-step architecture overview see [`studio-pipeline.md`](studio-pipeline.md).

```
AnimaLoraStudio/
├── runtime/                       # Training/inference runtime core (multi-family; standalone process, launched by Studio as a subprocess or run via CLI)
│   ├── anima_train.py             # Training entry (shared by both families)
│   ├── training/                  # Training stack: context / phases / text cache / loop / sample_runner / sysmem (VRAM/RAM orchestration)
│   │   ├── families/              # Model-family registry: spec / protocol / latent_spaces + anima/ + krea2/ (per-family loader / text stack / sampling / preset)
│   │   ├── adapters/              # plugin: lokr / loha / lora / ortho / tlora
│   │   ├── optimizers/            # plugin: adamw / automagic / came / lion / prodigy / prodigy_plus_schedulefree / soap / soap_sf
│   │   ├── schedulers/            # plugin: cosine / cosine_with_restart / cosine_with_warmup / none
│   │   ├── inference_samplers/    # plugin: er_sde / dpmpp_3m_sde, etc.
│   │   ├── timestep_samplers/     # plugin: baseline / infonoise / krea2_shift
│   │   └── phases/                # bootstrap / models / dataset / text_cache / optimizer / resume / finalize
│   ├── anima_generate.py          # Image generation: single image / XY matrix (family-dispatched)
│   ├── anima_daemon.py            # Inference daemon: keeps the base model and LoRA loaded in GPU (family-dispatched)
│   ├── anima_reg_ai.py            # AI prior generation: no LoRA, base model produces reg set
│   └── train_monitor.py           # Training state writer
├── studio/                        # AnimaStudio Web workbench (FastAPI + React) — 4-layer architecture (ADR 0008)
│   ├── api/                       # HTTP surface: FastAPI app + routers + schemas + deps + exception_handlers
│   ├── services/                  # Business services, 11 subpackages: tagging / booru / reg / inference / models /
│   │                              #   preprocess / projects / dataset / presets / runtime / data_io
│   ├── domain/                    # pydantic models: TrainingConfig / LoRA / XY / Generate / RegAi + migrations
│   ├── infrastructure/            # paths / DB / event bus / secrets / logging / argparse bridge / migrations
│   ├── supervisor/                # Task scheduler daemon thread
│   ├── workers/                   # Background subprocess entries (download / tag / reg_build / preprocess)
│   ├── server.py                  # Compatibility shim, re-exports `app` / `main` (real entries: api/app.py / api/main.py)
│   └── web/                       # React + Vite frontend
├── tools/                         # User CLI / launcher-time setup helpers (see tools/README.md)
├── utils/                         # Shared utilities for anima_train (model loader / optimizer / lycoris_adapter / ...)
├── modeling/                      # Model architecture defs (tracked), organized per family
│   ├── anima/                     # Anima family (anima_modeling.py + cosmos_predict2_modeling.py, based on ComfyUI / vendored diffusion-pipe subset)
│   ├── krea2/                     # Krea 2 family (krea2_modeling.py, single-stream MMDiT, based on ComfyUI / musubi-tuner)
│   └── wan/vae2_1.py              # Wan2.1 VAE implementation (shared across families)
├── docs/                          # user-guide / architecture / adr / design / todo / announcements (see docs/README.md)
└── models/                        # Downloaded weights / tokenizer data dir (gitignored, created on use)
    ├── diffusion_models/          # User-downloaded base models (Anima; Krea 2 Raw / Turbo bf16 / fp8 single files)
    ├── vae/                       # User-downloaded VAE weights (shared across families)
    ├── text_encoders/             # Text encoders + tokenizers (Anima's Qwen3-0.6B; Krea 2's Qwen3-VL-4B bf16 dir / fp8 single-file dir)
    ├── t5_tokenizer/              # T5 tokenizer files (downloaded)
    ├── wd14/                      # WD14 ONNX models (auto-downloaded from HF)
    └── taeflux/                   # TAEFlux intermediate preview weights
```

**Single dependency direction**: `modeling → utils → runtime → studio → tools`, no reverse imports. (`models/` is a pure data dir — no code, not in the dependency chain.)

## Runtime data (gitignored)

- `studio_data/` — SQLite + user presets
- `studio_data/tasks/{id}/` — per-training-task config snapshot + monitor state + samples + run.log (history survives version deletion)
- `studio_data/projects/{id}-{slug}/versions/{label}/output/` — trained LoRA artifacts
- `studio_data/projects/{id}-{slug}/versions/{label}/reg/` — regularization set (shared by tasks under that version)
- `models/diffusion_models/`, `models/vae/`, `models/wd14/` — large weight files
