"""评估出图 —— 走测试出图那条 daemon，不再自己写一套推理循环。

**为什么**：`eval_samples` 原本自己加载 VAE + 文本栈 + Transformer、自己跑
「encode → sample → decode → save」。而测试出图的常驻 daemon
（`runtime/anima_daemon.py`）早就把这些做全了，还多做了很多：

- `ModelCache` 底模跨 task 常驻，**一次评估只加载一次底模**
- LoRA 拓扑相同时走**权重热换**而不是重 inject（同一次训练的各 epoch 正好命中）
- TE 先行编排：预编码整个 prompt 集合 → 彻底释放 TE → 才加载 DiT
- CUDA OOM 重试 + allocator 恢复、block swap、vram_policy 让位、协议级 cancel

旧路径每个候选完整加载一遍底模：21 个候选就是 21 次 ~28s 的 Transformer 读盘，
约 10 分钟纯开销；而且同进程连跑还得自己管显存释放（见 ADR 0011 的 Addendum）。
现在这些统统归 daemon 的 `ModelCache`。

**为什么不用 server 里那个 daemon 单例**：评估跑在 supervisor 派的独立 worker
子进程里，跨进程拿不到那个单例。而 supervisor 在派 `eval_session` 之前已经调
`_maybe_yield_daemon()` 吊销了测试 daemon 的显存租约，所以卡是空的 —— worker 起
自己的实例即可，两者不会并存。

**为什么不用 daemon 的 XY**：XY 没有 prompt 轴（`_run_xy` 明确说 prompt 已由
`_run_generate` 统一预编码）。但普通 generate 本来就吃 `prompts: list[str]`，
所以「一个候选一个 task、prompts = 全部验证图 caption」就够了，daemon 零改动，
而且保住了候选主序（run / 进度 / 指标 / 样图矩阵的结构全不用动）。
"""
from __future__ import annotations

import base64
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from studio import secrets
from studio.domain.common import FAMILY_CAPABILITIES
from studio.services import version_config
from studio.services.inference.daemon import InferenceDaemon

logger = logging.getLogger(__name__)

# 一个候选的出图上限。daemon 单 task 串行跑 prompts，验证集再大也不该无限等。
_TASK_TIMEOUT_SECONDS = 3600.0


class EvalGenerationError(RuntimeError):
    pass


# 「该族不支持 block swap」只提示一次（见 _generate_settings）
_BLOCK_SWAP_NOTICED: set[str] = set()

_FALLBACK_SETTINGS: dict[str, Any] = {
    "vae_precision": "bf16", "lora_merge_precision": "fp32",
    "vram_policy": "auto", "ram_guard": True, "blocks_to_swap": 0,
}


def _generate_settings(family_id: str) -> dict[str, Any]:
    """daemon 运行时旋钮（显存策略 / 精度 / block swap）取全局出图设置。

    这些是「怎么跑」而不是「跑什么」，评估和测试出图共用一套 —— 但**族条件的能力
    必须门控**：测试出图跑的是用户在那儿选的模型，评估跑的是这个 version 训练时
    那个底模，两者的族可以不同。`blocks_to_swap` 只有 krea2 有
    （`FAMILY_CAPABILITIES`），把为 krea2 调的层数原样喂给 anima，daemon 会
    fail-fast 直接崩掉整个出图阶段。

    读失败给保守默认，不让评估因为设置文件坏了跑不起来。
    """
    try:
        gen = secrets.load().generate
        settings = {
            "vae_precision": str(getattr(gen, "vae_precision", "bf16") or "bf16"),
            "lora_merge_precision": str(
                getattr(gen, "lora_merge_precision", "fp32") or "fp32"
            ),
            "vram_policy": str(getattr(gen, "vram_policy", "auto") or "auto"),
            "ram_guard": bool(getattr(gen, "ram_guard", True)),
            "blocks_to_swap": int(getattr(gen, "blocks_to_swap", 0) or 0),
        }
    except Exception:
        logger.warning("读取出图设置失败，评估出图用保守默认", exc_info=True)
        return dict(_FALLBACK_SETTINGS)

    if settings["blocks_to_swap"] and "block_swap" not in FAMILY_CAPABILITIES.get(
        family_id, frozenset()
    ):
        # 每个候选都会走一遍本函数，逐次打就是 200 个 checkpoint 打 200 行同样的话
        # ——正是 #465 抱怨的那种噪音。按族记一次就够。
        if family_id not in _BLOCK_SWAP_NOTICED:
            _BLOCK_SWAP_NOTICED.add(family_id)
            logger.info(
                "model_family=%s 不支持 block swap，评估出图忽略全局设置的 "
                "blocks_to_swap=%s", family_id, settings["blocks_to_swap"],
            )
        settings["blocks_to_swap"] = 0
    return settings


def build_daemon_config(
    run: dict[str, Any], version_dir: Path, *, output_dir: Path,
) -> dict[str, Any]:
    """一个候选的 run → 一份 daemon generate config。

    prompts 按 items 顺序给，daemon 逐条出一张（count=1），所以第 i 张图对应
    第 i 个 item。生成参数取 run 冻结的 `generation`（EvalPlan 的一部分），模型
    路径取 version config —— 评估必须用 LoRA 训练时那个底模。
    """
    project = {"id": run["project_id"], "slug": run["project_slug"]}
    version = {"id": run["version_id"], "label": run["version_label"]}
    cfg = version_config.read_version_config(project, version)

    generation = run.get("generation") if isinstance(run.get("generation"), dict) else {}
    items = run.get("items") if isinstance(run.get("items"), list) else []
    if not items:
        raise EvalGenerationError("run 没有待出图的 item")

    width = max(16, (int(generation.get("width") or cfg.get("resolution") or 1024) // 16) * 16)
    height = max(16, (int(generation.get("height") or cfg.get("resolution") or 1024) // 16) * 16)
    # baseline run 的 lora_scale 是 0（纯底模对照）→ 干脆不给 lora_configs。
    # 不能写 `or 1.0`：0.0 是 falsy，会被当成「没设」回退成正常 LoRA 跑，Δ 恒为 0。
    raw_scale = generation.get("lora_scale")
    lora_scale = float(raw_scale) if raw_scale is not None else 1.0
    checkpoint = version_dir / run["checkpoint"]["path"]

    family_id = str(cfg.get("model_family") or "anima")
    out: dict[str, Any] = {
        "model_family": family_id,
        # 路径原样传，daemon 侧统一 resolve_path_best_effort
        "transformer_path": str(cfg["transformer_path"]),
        "vae_path": str(cfg["vae_path"]),
        "text_encoder_path": str(cfg["text_encoder_path"]),
        "t5_tokenizer_path": str(cfg.get("t5_tokenizer_path") or ""),
        "prompts": [str(item.get("prompt") or "") for item in items],
        "negative_prompt": str(
            generation.get("negative_prompt") or cfg.get("sample_negative_prompt") or ""
        ),
        "width": width,
        "height": height,
        "steps": int(generation.get("steps") or cfg.get("sample_infer_steps") or 25),
        "cfg_scale": float(
            generation.get("guidance_scale")
            or generation.get("cfg_scale")
            or cfg.get("sample_cfg_scale")
            or 4.0
        ),
        "sampler_name": str(
            generation.get("sampler_name") or cfg.get("sample_sampler_name") or "er_sde"
        ),
        "scheduler": str(
            generation.get("scheduler") or cfg.get("sample_scheduler") or "simple"
        ),
        "count": 1,
        "seed": int(generation.get("seed") or 0),
        "lora_configs": (
            [] if lora_scale == 0.0
            else [{"path": str(checkpoint), "scale": lora_scale}]
        ),
        "output_dir": str(output_dir),
        "mixed_precision": str(cfg.get("mixed_precision") or "bf16"),
        "attention_backend": str(cfg.get("attention_backend") or "flash_attn"),
        # 评估不需要中间预览（没人盯着看），关掉省 b64 编码和管道带宽
        "preview_every_n_steps": 0,
        **_generate_settings(family_id),
    }
    return out


class DaemonSampleGenerator:
    """会话级 daemon 客户端；`__call__` 符合 `eval_samples.SampleGenerator` 契约。

    **生命周期是一次评估**，不是一个候选 —— 底模常驻的收益全在这里：

        with DaemonSampleGenerator(progress) as generate:
            for cand in candidates:
                eval_samples.run_sample_job(..., generator=generate)

    每个候选一个 daemon task；候选之间底模不动，只热换 LoRA 权重。
    """

    def __init__(
        self,
        progress: Callable[[str], None],
        *,
        task_id: int = 0,
        daemon: Optional[InferenceDaemon] = None,
    ) -> None:
        self._progress = progress
        self._task_id = int(task_id)
        # cache_images=False：图归评估的 run 目录，不进测试页的 generate_cache
        self._daemon = daemon or InferenceDaemon(cache_images=False)
        self._owns_daemon = daemon is None

    # ---------------------------------------------------------------- 生命周期
    def __enter__(self) -> "DaemonSampleGenerator":
        self._daemon.start()
        self._progress("[eval-samples] 出图 daemon 就绪（底模按需加载一次，之后候选间只热换 LoRA）")
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if not self._owns_daemon:
            return
        try:
            self._daemon.stop()
            self._progress("[eval-samples] 出图 daemon 已退出，显存释放")
        except Exception:
            logger.warning("停止评估 daemon 失败", exc_info=True)

    # ---------------------------------------------------------------- 出图
    def __call__(
        self,
        run: dict[str, Any],
        version_dir: Path,
        progress: Callable[[str], None],
    ) -> None:
        from studio.services import eval_samples

        items = run.get("items") if isinstance(run.get("items"), list) else []
        if not items:
            return
        eval_root = Path(run["eval_root"]) if run.get("eval_root") else None
        images_dir = eval_samples.run_dir(
            version_dir, run["run_id"], eval_root
        ) / eval_samples.IMAGES_DIRNAME
        images_dir.mkdir(parents=True, exist_ok=True)

        config = build_daemon_config(run, version_dir, output_dir=images_dir)
        done = threading.Event()
        failure: list[str] = []
        # 事件回调跑在 daemon 的 reader 线程上；run.json 的读改写不能两边同时来，
        # 所以本函数在 done 之前只等待、不碰 run。
        state = {"run": run}

        def on_event(evt: dict[str, Any]) -> None:
            kind = str(evt.get("kind") or "")
            try:
                if kind == "image_started":
                    idx = int(evt.get("batch_idx") or 0)
                    if 0 <= idx < len(items):
                        state["run"] = eval_samples.mark_item_running(
                            version_dir, state["run"], idx, eval_root
                        )
                        progress(
                            f"[eval-samples] {idx + 1}/{len(items)} "
                            f"prompt={str(items[idx].get('prompt') or '')[:80]}"
                        )
                elif kind == "image_done":
                    self._write_image(evt, items, images_dir)
                    idx = int(evt.get("step") or 0) - 1
                    if 0 <= idx < len(items):
                        state["run"] = eval_samples.mark_item_done(
                            version_dir, state["run"], idx, eval_root
                        )
                elif kind == "image_error":
                    idx = int(evt.get("step") or 0) - 1
                    message = str(evt.get("message") or "出图失败")
                    if 0 <= idx < len(items):
                        state["run"] = eval_samples.mark_item_failed(
                            version_dir, state["run"], idx, message, eval_root
                        )
                    progress(f"[eval-samples] 第 {idx + 1} 张失败：{message}")
                elif kind in ("done", "error", "canceled"):
                    if kind != "done":
                        failure.append(str(evt.get("message") or kind))
                    done.set()
            except Exception as exc:  # noqa: BLE001
                logger.exception("评估出图事件处理失败 (kind=%s)", kind)
                failure.append(str(exc))
                done.set()

        self._daemon.submit_task(
            task_id=self._task_id or int(run.get("version_id") or 0),
            config=config,
            output_dir=str(images_dir),
            on_event=on_event,
        )
        if not done.wait(timeout=_TASK_TIMEOUT_SECONDS):
            raise EvalGenerationError(
                f"出图超时（>{int(_TASK_TIMEOUT_SECONDS)}s），daemon 无响应"
            )
        if failure:
            raise EvalGenerationError(failure[0])

    def _write_image(
        self, evt: dict[str, Any], items: list[dict[str, Any]], images_dir: Path,
    ) -> None:
        """把 image_done 事件里的 PNG 落到 item 计划好的文件名上。

        daemon 自己的文件名是 `gen_<i>_p<pi>_c<ci>_s<seed>.png`，评估的读侧
        （指标 runner / 样图矩阵）认的是 `item["filename"]` —— 以 item 为准，
        存储结构不变。
        """
        idx = int(evt.get("step") or 0) - 1
        if not (0 <= idx < len(items)):
            raise EvalGenerationError(f"image_done 的序号越界: step={evt.get('step')}")
        raw = evt.get("image_b64")
        if not raw:
            raise EvalGenerationError(
                "image_done 没带图像数据 —— daemon 的 cache_images 应为 False"
            )
        target = images_dir / str(items[idx]["filename"])
        tmp = target.with_suffix(".png.part")
        tmp.write_bytes(base64.b64decode(raw))
        tmp.replace(target)
