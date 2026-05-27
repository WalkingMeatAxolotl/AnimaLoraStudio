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
        sample_max_side: int = 1216,
        sample_every_n_steps: int = 0,
        upload_model: bool = False,
        upload_model_policy: str = "last",
        upload_state_manual: bool = False,
        upload_state_manual_policy: str = "last",
        upload_state_auto: bool = False,
        upload_state_auto_policy: str = "last",
    ) -> None:
        self._wandb = wandb_module
        self._run = run
        self.log_samples = log_samples
        self.sample_max_side = max(64, int(sample_max_side or 512))
        self.sample_every_n_steps = max(0, int(sample_every_n_steps or 0))
        self._last_logged_step: Optional[int] = None
        self._upload_model_enabled = upload_model
        self._upload_model_policy = upload_model_policy
        self._upload_state_manual_enabled = upload_state_manual
        self._upload_state_manual_policy = upload_state_manual_policy
        self._upload_state_auto_enabled = upload_state_auto
        self._upload_state_auto_policy = upload_state_auto_policy
        self._last_artifact: dict[str, "Any"] = {}

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

    def _upload_artifact(self, file_path: Path, artifact_name: str, artifact_type: str, policy: str) -> None:
        if not self.enabled:
            return
        try:
            artifact = self._wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(str(file_path))
            self._run.log_artifact(artifact)
            logger.info(f"W&B artifact 已上传: {artifact_name} ({file_path.name})")
            if policy == "last":
                prev = self._last_artifact.get(artifact_name)
                if prev is not None:
                    try:
                        prev.delete()
                        logger.info(f"W&B artifact 旧版本已删除: {artifact_name}")
                    except Exception as exc:
                        logger.warning(f"W&B 删除旧 artifact 失败: {exc}")
                self._last_artifact[artifact_name] = artifact
        except Exception as exc:
            logger.warning(f"W&B artifact 上传失败 ({artifact_name}): {exc}")

    def upload_model(self, file_path: Path) -> None:
        if not self._upload_model_enabled or not self.enabled:
            return
        name = f"{self._run.name}-model"
        self._upload_artifact(file_path, name, "model", self._upload_model_policy)

    def upload_state_manual(self, file_path: Path) -> None:
        if not self._upload_state_manual_enabled or not self.enabled:
            return
        name = f"{self._run.name}-state-manual"
        self._upload_artifact(file_path, name, "training-state", self._upload_state_manual_policy)

    def upload_state_auto(self, file_path: Path) -> None:
        if not self._upload_state_auto_enabled or not self.enabled:
            return
        name = f"{self._run.name}-state-auto"
        self._upload_artifact(file_path, name, "training-state", self._upload_state_auto_policy)

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            self._run.finish()
        except Exception as exc:
            logger.warning(f"W&B finish 失败: {exc}")


def _preset_str(args, attr: str) -> str:
    """预设字符串覆盖：非空字符串时使用预设值。"""
    val = getattr(args, attr, None)
    return str(val) if val and str(val).strip() else ""


def _preset_bool(args, attr: str) -> Optional[bool]:
    """预设布尔覆盖：None 表示未设置（回退全局）。"""
    val = getattr(args, attr, None)
    if val is None:
        return None
    return bool(val)


def init_wandb_monitor(args, output_dir: Path, config_path: Optional[Path]) -> WandBMonitor:
    # ---- 预设覆盖优先级：args.wandb_* (非空) > 环境变量 (全局 Settings) ----
    preset_enabled = _preset_bool(args, "wandb_enabled")
    if preset_enabled is not None:
        enabled = preset_enabled
    else:
        enabled = str(os.environ.get("WANDB_ENABLED", "")).strip().lower() in {
            "1", "true", "yes", "on",
        }
    if not enabled:
        return WandBMonitor(None, None)

    mode = _preset_str(args, "wandb_mode") or str(os.environ.get("WANDB_MODE", "online") or "online")
    if mode == "disabled":
        return WandBMonitor(None, None)
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "已在 Settings 启用 WandB，但当前环境没有安装 wandb。"
            "请先在训练环境安装：pip install wandb，或在 Settings 关闭 WandB。"
        ) from exc

    # 预设覆盖 API key：写进 env 让 wandb.init() 识别
    preset_api_key = _preset_str(args, "wandb_api_key")
    if preset_api_key:
        os.environ["WANDB_API_KEY"] = preset_api_key

    project = _preset_str(args, "wandb_project") or os.environ.get("WANDB_PROJECT") or "AnimaLoraStudio"
    entity = _preset_str(args, "wandb_entity") or os.environ.get("WANDB_ENTITY") or None
    run_name = os.environ.get("WANDB_RUN_NAME") or str(args.output_name)

    preset_base_url = _preset_str(args, "wandb_base_url")
    if preset_base_url:
        os.environ["WANDB_BASE_URL"] = preset_base_url

    # log_samples
    preset_log_samples = _preset_bool(args, "wandb_log_samples")
    if preset_log_samples is not None:
        log_samples = preset_log_samples
    else:
        log_samples = str(os.environ.get("WANDB_LOG_SAMPLES", "1")).strip().lower() not in {
            "0", "false", "no", "off",
        }

    # sample_max_side: preset 0 = 使用全局
    preset_max_side = int(getattr(args, "wandb_sample_max_side", 0) or 0)
    if preset_max_side > 0:
        sample_max_side = preset_max_side
    else:
        try:
            sample_max_side = int(os.environ.get("WANDB_SAMPLE_MAX_SIDE", "1216") or 1216)
        except ValueError:
            sample_max_side = 512

    # sample_every_n_steps: preset -1 = 使用全局
    preset_every_n = int(getattr(args, "wandb_sample_every_n_steps", -1) if getattr(args, "wandb_sample_every_n_steps", -1) is not None else -1)
    if preset_every_n >= 0:
        sample_every_n_steps = preset_every_n
    else:
        try:
            sample_every_n_steps = int(os.environ.get("WANDB_SAMPLE_EVERY_N_STEPS", "0") or 0)
        except ValueError:
            sample_every_n_steps = 0

    # artifact 上传
    _env_bool = lambda key, default="0": str(os.environ.get(key, default)).strip().lower() in {"1", "true", "yes", "on"}
    _env_policy = lambda key: "all" if str(os.environ.get(key, "last")).strip().lower() == "all" else "last"

    def _resolve_bool(attr: str, env_key: str) -> bool:
        p = _preset_bool(args, attr)
        return p if p is not None else _env_bool(env_key)

    def _resolve_policy(attr: str, env_key: str) -> str:
        p = _preset_str(args, attr)
        return p if p in {"all", "last"} else _env_policy(env_key)

    upload_model = _resolve_bool("wandb_upload_model", "WANDB_UPLOAD_MODEL")
    upload_model_policy = _resolve_policy("wandb_upload_model_policy", "WANDB_UPLOAD_MODEL_POLICY")
    upload_state_manual = _resolve_bool("wandb_upload_state_manual", "WANDB_UPLOAD_STATE_MANUAL")
    upload_state_manual_policy = _resolve_policy("wandb_upload_state_manual_policy", "WANDB_UPLOAD_STATE_MANUAL_POLICY")
    upload_state_auto = _resolve_bool("wandb_upload_state_auto", "WANDB_UPLOAD_STATE_AUTO")
    upload_state_auto_policy = _resolve_policy("wandb_upload_state_auto_policy", "WANDB_UPLOAD_STATE_AUTO_POLICY")

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
        upload_model=upload_model,
        upload_model_policy=upload_model_policy,
        upload_state_manual=upload_state_manual,
        upload_state_manual_policy=upload_state_manual_policy,
        upload_state_auto=upload_state_auto,
        upload_state_auto_policy=upload_state_auto_policy,
    )
