---
date: 2026-07-09
tag: migration
title: WandB settings moved to global Settings — rotate your API key if you ever put one in a training config
pin: true
version: "0.18.0"
---
Starting with 0.18.0, all WandB settings (API key, entity, project, upload behavior, and so on) live in **Settings → WandB** and are managed as presets. The "WandB (per-config override)" group in the training config has been removed entirely.

### Why

WandB is account-level configuration — it doesn't vary per project. More importantly, an API key stored in a training config was written in plain text into config.yaml, task snapshots, exported presets and bundles — files that exist precisely to be shared and backed up. The key now lives only in the global secrets store and is masked everywhere in the UI.

### What to do

- **You never used WandB** → nothing to do.
- **You only ever configured WandB in global Settings** → your settings are migrated to a "default" preset automatically. **No action needed.**
- **You filled in WandB overrides in a training config** → those fields are ignored when the old config loads (the page tells you which fields were dropped). Reconfigure under **Settings → WandB**; if you need different entity / project per project, create multiple presets and switch between them — import / export is supported too.
- **You ever put an API key in a training config** → **we recommend regenerating (rotating) your API key at wandb.ai**. Old config.yaml files, task snapshots, exported presets and bundles are not rewritten — the plain-text key in them stays on disk. Once rotated, the old key is dead and those leftovers are harmless. Also avoid sharing those older exports.
- **You run training from the command line** → the `--wandb_*` flags were removed along with the fields. Configure WandB in Studio's global Settings (injected automatically at training time), or set the `WANDB_*` environment variables yourself.
