"""训练观测层：Loss 曲线 ASCII 渲染 + Weights & Biases 可选监控。

抽自原 runtime/anima_train.py L183-369（ADR 0003 PR-A）。

公开：
- render_loss_curve / render_curve_panel — ASCII loss 曲线 + Rich Panel 包装
- WandBMonitor / init_wandb_monitor — 可选 W&B 集成；env 变量驱动启停
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from studio.infrastructure.log_messages import msg


logger = logging.getLogger(__name__)


def _delete_artifact_version(artifact_name: str, artifact) -> bool:
    """删一个旧 artifact 版本；两条清理路径（api 扫描 / 上一次句柄）共用一处实现。"""
    version = getattr(artifact, "version", "?")
    try:
        artifact.delete(delete_aliases=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "W&B artifact version delete failed: %s:%s (%s) — the old version "
            "stays and keeps using storage quota", artifact_name, version, exc,
        )
        return False
    logger.debug(
        "wandb_artifact: old version deleted name=%s version=%s",
        artifact_name, version,
    )
    return True


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
        self._last_artifact: dict[str, Any] = {}
        # W&B 上报失败节流（R8）：log() 每 step 调，首条全文，finish() 汇总
        self._log_fail_count = 0
        self._log_fail_last = ""
        # 采样图上报失败节流（R8）：每张图一条 → warn-once + finish 汇总
        self._image_fail_count = 0
        self._image_fail_last = ""
        self._resize_fail_warned = False

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, data: dict, *, step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        try:
            self._run.log(data, step=step)
        except Exception as exc:
            self._log_fail_count += 1
            self._log_fail_last = str(exc)
            if self._log_fail_count == 1:
                logger.warning(
                    "W&B metric upload failed: %s — metrics for this step are "
                    "missing from the dashboard; further failures are counted "
                    "and reported at the end", exc,
                )

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
            if not self._resize_fail_warned:
                self._resize_fail_warned = True
                logger.warning(
                    "W&B image resize failed: %s — uploading the full-size image "
                    "instead; same warning is not repeated", exc,
                )
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
            self._image_fail_count += 1
            self._image_fail_last = str(exc)
            if self._image_fail_count == 1:
                logger.warning(
                    "W&B image upload failed: %s — this sample image is missing "
                    "from the dashboard; further failures are counted and reported "
                    "at the end", exc,
                )

    def _delete_previous_artifact_versions(self, artifact_name: str, artifact_type: str, keep_artifact) -> None:
        keep_version = getattr(keep_artifact, "version", None)
        collection_name = f"{keep_artifact.entity}/{keep_artifact.project}/{artifact_name}"
        deleted = 0
        try:
            api = self._wandb.Api()
            for artifact in api.artifacts(type_name=artifact_type, name=collection_name):
                if getattr(artifact, "version", None) == keep_version:
                    continue
                if _delete_artifact_version(artifact_name, artifact):
                    deleted += 1
            if deleted:
                logger.debug(
                    "wandb_artifact: old versions cleaned name=%s deleted=%d",
                    artifact_name, deleted,
                )
        except Exception as exc:
            logger.warning(
                "W&B artifact cleanup failed: %s (%s) — old versions keep "
                "accumulating and using storage quota", artifact_name, exc,
            )

    def _upload_artifact(self, file_path: Path, artifact_name: str, artifact_type: str, policy: str) -> None:
        if not self.enabled:
            return
        try:
            artifact = self._wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(str(file_path), name=file_path.name)
            size_mb = file_path.stat().st_size / 1024 / 1024
            logger.debug(
                "wandb_artifact: upload started name=%s file=%s size=%.1f MB",
                artifact_name, file_path.name, size_mb,
            )
            logged_artifact = self._run.log_artifact(artifact)
            start_time = time.monotonic()
            done = threading.Event()

            def report_waiting() -> None:
                slow_warned = False
                # 30s 一条 DEBUG 心跳；超 5 分钟另打一条 WARNING（慢上传从
                # 「刷屏 INFO」改成「一条告警」）。
                while not done.wait(30):
                    elapsed = time.monotonic() - start_time
                    logger.debug(
                        "wandb_artifact: upload in progress name=%s elapsed=%.1f s "
                        "size=%.1f MB", artifact_name, elapsed, size_mb,
                    )
                    if elapsed >= 300 and not slow_warned:
                        slow_warned = True
                        logger.warning(
                            "W&B artifact upload still running after %.1f s: %s "
                            "(%.1f MB) — training continues, the upload may be "
                            "stuck on a slow connection",
                            elapsed, artifact_name, size_mb,
                        )

            progress_thread = threading.Thread(target=report_waiting, daemon=True)
            progress_thread.start()
            try:
                logged_artifact.wait()
            finally:
                done.set()
                progress_thread.join(timeout=1)
            elapsed = time.monotonic() - start_time
            logger.info(msg(
                "train.wandb_artifact_uploaded",
                name=artifact_name, file=file_path.name,
                size=f"{size_mb:.1f}", elapsed=f"{elapsed:.1f}",
            ))
            if policy == "last":
                self._delete_previous_artifact_versions(artifact_name, artifact_type, logged_artifact)
                prev = self._last_artifact.get(artifact_name)
                if prev is not None and getattr(prev, "version", None) != getattr(logged_artifact, "version", None):
                    _delete_artifact_version(artifact_name, prev)
                self._last_artifact[artifact_name] = logged_artifact
        except Exception as exc:
            logger.warning(
                "W&B artifact upload failed: %s (%s) — the file exists locally "
                "only, nothing was uploaded to W&B", artifact_name, exc,
            )

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
        if self._log_fail_count > 1:
            logger.warning(
                "W&B metric upload failed %d time(s) during this run: the "
                "dashboard is incomplete (last error: %s)",
                self._log_fail_count, self._log_fail_last,
            )
        if self._image_fail_count > 1:
            logger.warning(
                "W&B image upload failed %d time(s) during this run: the "
                "dashboard is incomplete (last error: %s)",
                self._image_fail_count, self._image_fail_last,
            )
        try:
            self._run.finish()
        except Exception as exc:
            logger.warning(
                "W&B run failed to close: %s — the run may stay marked as running "
                "on the dashboard", exc,
            )


def init_wandb_monitor(args, output_dir: Path, config_path: Optional[Path]) -> WandBMonitor:
    # ---- 全部配置来自环境变量（全局 Settings 经 supervisor 注入 WANDB_*）。----
    # 0.18 起 per-config wandb_* 覆盖块已从 TrainingConfig 移除：wandb 属于账号/
    # 工作流级配置，api_key 等 secrets 不落 yaml、不进 args（args 会被整体作为
    # run config 上传到 wandb 服务端）。
    enabled = str(os.environ.get("WANDB_ENABLED", "")).strip().lower() in {
        "1", "true", "yes", "on",
    }
    if not enabled:
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

    # api_key / base_url 由 supervisor 直接放进 WANDB_API_KEY / WANDB_BASE_URL，
    # wandb.init() 自己识别，这里无需经手。
    project = os.environ.get("WANDB_PROJECT") or "AnimaLoraStudio"
    entity = os.environ.get("WANDB_ENTITY") or None
    run_name = os.environ.get("WANDB_RUN_NAME") or str(args.output_name)

    log_samples = str(os.environ.get("WANDB_LOG_SAMPLES", "1")).strip().lower() not in {
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

    # artifact 上传
    def _env_bool(key: str, default: str = "0") -> bool:
        return str(os.environ.get(key, default)).strip().lower() in {"1", "true", "yes", "on"}

    def _env_policy(key: str) -> str:
        return "all" if str(os.environ.get(key, "last")).strip().lower() == "all" else "last"

    upload_model = _env_bool("WANDB_UPLOAD_MODEL")
    upload_model_policy = _env_policy("WANDB_UPLOAD_MODEL_POLICY")
    upload_state_manual = _env_bool("WANDB_UPLOAD_STATE_MANUAL")
    upload_state_manual_policy = _env_policy("WANDB_UPLOAD_STATE_MANUAL_POLICY")
    upload_state_auto = _env_bool("WANDB_UPLOAD_STATE_AUTO")
    upload_state_auto_policy = _env_policy("WANDB_UPLOAD_STATE_AUTO_POLICY")

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
    logger.info(msg("train.wandb_enabled", project=project, run=run_name, mode=mode))
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
