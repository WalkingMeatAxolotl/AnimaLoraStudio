"""Inline checkpoint eval for the training loop.

The Studio project-job workers remain available for manual/bulk eval. This
module covers the training-time path: after a checkpoint is saved, reuse the
already-loaded training model to generate eval samples, then compute metrics
before the loop continues.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch

from training.context import TrainingContext
from training.sampling import sample_image
from utils.optimizer_utils import optimizer_eval_mode

logger = logging.getLogger(__name__)


def run_inline_checkpoint_eval(
    ctx: TrainingContext,
    checkpoint_path: Path,
    *,
    epoch: int,
    step: int,
    trigger: str,
) -> bool:
    """Run auto eval synchronously for one saved checkpoint.

    Returns True when the checkpoint was handled by the inline path, including
    metric/sample failures that were recorded to eval files. Returns False only
    when inline eval is not applicable, so callers may fall back to the legacy
    supervisor event.
    """
    task_id = ctx.lora_task_id
    if task_id is None:
        return False

    try:
        from studio.services import eval_auto as studio_eval_auto
    except Exception:
        logger.exception("inline auto eval unavailable")
        return False

    payload = {
        "checkpoint_path": str(checkpoint_path),
        "epoch": int(epoch),
        "step": int(step),
        "trigger": str(trigger),
    }

    def progress(line: str) -> None:
        ctx.emit(line)

    try:
        result = studio_eval_auto.run_checkpoint_eval_for_task(
            int(task_id),
            payload,
            sample_generator=_make_training_sample_generator(ctx),
            on_progress=progress,
        )
    except Exception as exc:  # noqa: BLE001
        ctx.emit(f"[eval-auto] 评估失败，训练继续: {exc}")
        logger.exception("inline auto eval failed")
        return True

    if result is None:
        return False

    run = result.get("run") if isinstance(result, dict) else None
    run_id = run.get("run_id") if isinstance(run, dict) else None
    if run_id:
        ctx.emit(f"[eval-auto] 完成 checkpoint 评估: {run_id}")
    return True


def _make_training_sample_generator(ctx: TrainingContext):
    def _generator(
        run: dict[str, Any],
        version_dir: Path,
        progress: Callable[[str], None],
    ) -> None:
        from studio.services import eval_samples

        generation = run.get("generation") if isinstance(run.get("generation"), dict) else {}
        args = ctx.args
        width = int(
            generation.get("width")
            or getattr(args, "sample_width", 0)
            or getattr(args, "resolution", 1024)
        )
        height = int(
            generation.get("height")
            or getattr(args, "sample_height", 0)
            or getattr(args, "resolution", 1024)
        )
        width = max(16, (width // 16) * 16)
        height = max(16, (height // 16) * 16)
        steps = int(generation.get("steps") or getattr(args, "sample_infer_steps", 25) or 25)
        cfg_scale = float(
            generation.get("guidance_scale")
            or generation.get("cfg_scale")
            or getattr(args, "sample_cfg_scale", 4.0)
            or 4.0
        )
        negative_prompt = str(
            generation.get("negative_prompt")
            or getattr(args, "sample_negative_prompt", "")
            or ""
        )
        sampler_name = str(
            generation.get("sampler_name")
            or getattr(args, "sample_sampler_name", "er_sde")
            or "er_sde"
        )
        scheduler = str(
            generation.get("scheduler")
            or getattr(args, "sample_scheduler", "simple")
            or "simple"
        )

        items = run.get("items") if isinstance(run.get("items"), list) else []
        with optimizer_eval_mode(ctx.optimizer):
            ctx.model.eval()
            try:
                for idx, item in enumerate(items):
                    current = eval_samples.load_run(version_dir, str(run["run_id"])) or run
                    eval_samples.mark_item_running(version_dir, current, idx)
                    seed = int(item["seed"])
                    output = eval_samples.sample_image_path(
                        version_dir, str(run["run_id"]), str(item["filename"])
                    )
                    output.parent.mkdir(parents=True, exist_ok=True)
                    progress(
                        f"[eval-samples] {idx + 1}/{len(items)} seed={seed} "
                        f"prompt={str(item.get('prompt') or '')[:80]}"
                    )
                    try:
                        torch.manual_seed(seed)
                        img = sample_image(
                            ctx.model,
                            ctx.vae,
                            ctx.qwen_model,
                            ctx.qwen_tok,
                            ctx.t5_tok,
                            str(item.get("prompt") or ""),
                            height=height,
                            width=width,
                            steps=steps,
                            cfg_scale=cfg_scale,
                            negative_prompt=negative_prompt or None,
                            sampler_name=sampler_name,
                            scheduler=scheduler,
                            device=ctx.device,
                            dtype=ctx.dtype,
                            seed=seed,
                        )
                        img.save(output)
                        current = eval_samples.load_run(version_dir, str(run["run_id"])) or current
                        eval_samples.mark_item_done(version_dir, current, idx)
                    except Exception as exc:  # noqa: BLE001
                        current = eval_samples.load_run(version_dir, str(run["run_id"])) or current
                        eval_samples.mark_item_failed(version_dir, current, idx, str(exc))
            finally:
                ctx.model.train()

    return _generator
