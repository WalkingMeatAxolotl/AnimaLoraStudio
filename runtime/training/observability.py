"""训练观测层：Loss 曲线 ASCII 渲染 + Weights & Biases 可选监控。

抽自原 runtime/anima_train.py L183-369（ADR 0003 PR-A）。

公开：
- render_loss_curve / render_curve_panel — ASCII loss 曲线 + Rich Panel 包装
- WandBMonitor / init_wandb_monitor — 可选 W&B 集成；env 变量驱动启停
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def render_loss_curve(losses, width=60, height=10):
    """渲染 ASCII Loss 曲线。"""
    if not losses:
        return ""
    if width < 5:
        width = 5
    values = losses
    if len(values) > width:
        step = len(values) / width
        buckets = []
        for i in range(width):
            start = int(i * step)
            end = int((i + 1) * step)
            end = max(end, start + 1)
            chunk = values[start:end]
            buckets.append(sum(chunk) / len(chunk))
        values = buckets
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        max_v = min_v + 1e-8
    grid = [[" " for _ in range(len(values))] for _ in range(height)]
    for i, v in enumerate(values):
        y = int((v - min_v) / (max_v - min_v) * (height - 1))
        y = height - 1 - y
        grid[y][i] = "*"
    lines = ["".join(row) for row in grid]
    lines.append(f"min={min_v:.4f} max={max_v:.4f}")
    return "\n".join(lines)


def render_curve_panel(losses, width=60, height=10):
    """渲染 Rich Panel 包装的 Loss 曲线。"""
    try:
        from rich.panel import Panel
        from rich.text import Text
    except Exception:
        return None
    chart = render_loss_curve(losses, width=width, height=height)
    return Panel(Text(chart), title="Loss curve (recent)", expand=False)


class WandBMonitor:
    def __init__(
        self,
        wandb_module,
        run,
        *,
        log_samples: bool = False,
        sample_max_side: int = 512,
        sample_every_n_steps: int = 0,
    ) -> None:
        self._wandb = wandb_module
        self._run = run
        self.log_samples = log_samples
        self.sample_max_side = max(64, int(sample_max_side or 512))
        self.sample_every_n_steps = max(0, int(sample_every_n_steps or 0))
        self._last_logged_step: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, data: dict, *, step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        try:
            self._run.log(data, step=step)
        except Exception as exc:
            logger.warning(f"W&B log 失败: {exc}")

    def _should_log_step(self, key: str, step: Optional[int]) -> bool:
        # baseline / epoch 边界一律放行；step 模式按 sample_every_n_steps 节流。
        if self.sample_every_n_steps <= 0:
            return True
        if not key.startswith("samples/step"):
            return True
        if step is None or step <= 0:
            return True
        if step == self._last_logged_step:
            return True  # 同步重复调用允许
        return step % self.sample_every_n_steps == 0

    def _prepare_image(self, image_path: Path, caption: str):
        # 原图常 2K+，wandb 面板浏览 512px 已足够；JPEG 流量比 PNG 小一个数量级。
        try:
            from PIL import Image
        except Exception:
            return self._wandb.Image(str(image_path), caption=caption)
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                max_side = self.sample_max_side
                w, h = img.size
                if max(w, h) > max_side:
                    scale = max_side / float(max(w, h))
                    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                    img = img.resize(new_size, Image.LANCZOS)
                return self._wandb.Image(img, caption=caption)
        except Exception as exc:
            logger.warning(f"W&B 图片缩放失败，改传原图: {exc}")
            return self._wandb.Image(str(image_path), caption=caption)

    def log_image(self, key: str, image_path: Path, *, caption: str, step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        if not self._should_log_step(key, step):
            return
        try:
            self._run.log({key: [self._prepare_image(image_path, caption)]}, step=step)
            self._last_logged_step = step
        except Exception as exc:
            logger.warning(f"W&B 图片记录失败: {exc}")

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            self._run.finish()
        except Exception as exc:
            logger.warning(f"W&B finish 失败: {exc}")


def init_wandb_monitor(args, output_dir: Path, config_path: Optional[Path]) -> WandBMonitor:
    enabled = str(os.environ.get("WANDB_ENABLED", "")).strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return WandBMonitor(None, None)
    mode = str(os.environ.get("WANDB_MODE", "online") or "online")
    if mode == "disabled":
        return WandBMonitor(None, None)
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "已在 Settings 启用 WandB，但当前环境没有安装 wandb。"
            "请先在训练环境安装：pip install wandb，或在 Settings 关闭 WandB。"
        ) from exc

    project = os.environ.get("WANDB_PROJECT") or "AnimaLoraStudio"
    entity = os.environ.get("WANDB_ENTITY") or None
    run_name = os.environ.get("WANDB_RUN_NAME") or str(args.output_name)
    # 默认关 — supervisor 已经按 secrets.wandb.log_samples 设过 env；env 缺省时也保持关闭。
    log_samples = str(os.environ.get("WANDB_LOG_SAMPLES", "0")).strip().lower() not in {
        "0", "false", "no", "off",
    }
    try:
        sample_max_side = int(os.environ.get("WANDB_SAMPLE_MAX_SIDE", "512") or 512)
    except ValueError:
        sample_max_side = 512
    try:
        sample_every_n_steps = int(os.environ.get("WANDB_SAMPLE_EVERY_N_STEPS", "0") or 0)
    except ValueError:
        sample_every_n_steps = 0
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        key: value
        for key, value in vars(args).items()
        if key not in {"interactive", "auto_install"}
    }
    cfg["config_path"] = str(config_path) if config_path else ""
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        mode=mode,
        config=cfg,
        dir=str(wandb_dir),
    )
    logger.info(f"W&B 监控已启用: project={project}, run={run_name}, mode={mode}")
    return WandBMonitor(
        wandb,
        run,
        log_samples=log_samples,
        sample_max_side=sample_max_side,
        sample_every_n_steps=sample_every_n_steps,
    )
