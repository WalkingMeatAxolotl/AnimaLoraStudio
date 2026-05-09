#!/usr/bin/env python3
"""测试出图常驻 daemon 子进程。

由 studio/services/inference_daemon.py 启动；JSON-over-stdio 协议：
  stdin  ← {"id": "<req_id>", "action": "generate"|"unload"|"ping", ...}
  stdout → {"id": "<req_id>"|"_evt", "kind": "ready"|"started"|"image_done"|
             "done"|"error"|"loaded"|"unloaded", ...}

stdout 仅协议；日志全走 stderr（避免污染协议流）。

设计：
  - 启动后立即推 _evt ready（说明 import / sys.path 完成；模型未加载）
  - 第一次 generate task 来时 lazy load 模型（30-60s），推 _evt loaded
  - 后续 task 复用模型 + adapters；adapter 卸载/重 inject 仅在 lora_configs 改变时
  - 单线程串行处理（一次一个 task）；server 端保证不并发提交

用法（CLI 调试）：
    python tools/anima_daemon.py
    然后从 stdin 喂一行 JSON：
        {"id":"r1","action":"generate","task_id":1,"output_dir":"/tmp/g","config":{...}}
"""
from __future__ import annotations

import base64
import io
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Optional

import torch

# 同 anima_generate.py 的 sys.path 处理（让 anima_train / studio 可 import）
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _p in (_THIS_DIR, _SCRIPTS_DIR, _REPO_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import anima_train as _T  # noqa: E402

from studio.schema import migrate_legacy_attention  # noqa: E402
from studio.services.inference_core import LoRASpec, apply_loras  # noqa: E402

# 日志走 stderr，stdout 留给协议
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("anima_daemon")


# ---------------------------------------------------------------------------
# 协议输出
# ---------------------------------------------------------------------------


def _emit(msg: dict[str, Any]) -> None:
    """写一条协议消息到 stdout（line-delimited JSON）。"""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _emit_evt(kind: str, **extra: Any) -> None:
    _emit({"id": "_evt", "kind": kind, **extra})


def _emit_for(req_id: str, kind: str, **extra: Any) -> None:
    _emit({"id": req_id, "kind": kind, **extra})


# ---------------------------------------------------------------------------
# 模型管理（lazy load + cache）
# ---------------------------------------------------------------------------


class ModelCache:
    """缓存已加载的模型 / adapters。

    第一次 task 进来 load_model_paths()；之后路径不变则复用；adapters
    在 lora_configs 改变时才重 inject（commit 9 简化：每次都重 inject，
    成本 ~1-2s/LoRA，相比 30s+ model load 可忽略；后续 commit 优化）。
    """

    def __init__(self) -> None:
        self.transformer_path: Optional[str] = None
        self.vae_path: Optional[str] = None
        self.text_encoder_path: Optional[str] = None
        self.t5_tokenizer_path: Optional[str] = None
        self.attention_backend: Optional[str] = None
        self.mixed_precision: Optional[str] = None
        self.device: Optional[str] = None
        self.dtype: Any = None
        self.model: Any = None
        self.vae: Any = None
        self.qwen_model: Any = None
        self.qwen_tok: Any = None
        self.t5_tok: Any = None
        # adapters 必须保持引用，否则 forward hook 失效（lycoris closure）
        self.adapters: list[Any] = []
        self.last_lora_specs: list[LoRASpec] = []

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def ensure_loaded(self, cfg: dict[str, Any]) -> None:
        """按 cfg 决定是否需要 (重新) 加载。路径或后端变了 → 全重载。"""
        cfg = migrate_legacy_attention(dict(cfg))
        backend = cfg.get("attention_backend", "flash_attn")
        precision = cfg.get("mixed_precision", "bf16")
        transformer_path = cfg["transformer_path"]
        vae_path = cfg["vae_path"]
        text_encoder_path = cfg["text_encoder_path"]
        t5_tokenizer_path = cfg.get("t5_tokenizer_path", "")

        # 路径解析
        repo_root = _T.find_diffusion_pipe_root()
        bases = [Path.cwd(), _THIS_DIR, repo_root]
        transformer_path = _T.resolve_path_best_effort(transformer_path, bases)
        vae_path = _T.resolve_path_best_effort(vae_path, bases)
        text_encoder_path = _T.resolve_path_best_effort(text_encoder_path, bases)
        if t5_tokenizer_path:
            t5_tokenizer_path = _T.resolve_path_best_effort(t5_tokenizer_path, bases)

        # 比较是否需要 reload
        needs_reload = (
            not self.loaded
            or self.transformer_path != transformer_path
            or self.vae_path != vae_path
            or self.text_encoder_path != text_encoder_path
            or self.t5_tokenizer_path != t5_tokenizer_path
            or self.attention_backend != backend
            or self.mixed_precision != precision
        )

        if needs_reload:
            self.unload()
            self._load(
                transformer_path=transformer_path,
                vae_path=vae_path,
                text_encoder_path=text_encoder_path,
                t5_tokenizer_path=t5_tokenizer_path,
                backend=backend,
                precision=precision,
            )
            _emit_evt("loaded")

    def _load(
        self,
        *,
        transformer_path: str,
        vae_path: str,
        text_encoder_path: str,
        t5_tokenizer_path: str,
        backend: str,
        precision: str,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        repo_root = _T.find_diffusion_pipe_root()
        use_flash = backend == "flash_attn"
        use_xformers = backend == "xformers"

        logger.info("loading transformer %s", transformer_path)
        model = _T.load_anima_model(
            transformer_path, device, dtype, repo_root, flash_attn=use_flash,
        )
        if use_xformers:
            _T.enable_xformers(model)

        logger.info("loading vae %s", vae_path)
        vae = _T.load_vae(vae_path, device, dtype, repo_root)

        logger.info("loading text encoders %s", text_encoder_path)
        qwen_model, qwen_tok, t5_tok = _T.load_text_encoders(
            text_encoder_path, t5_tokenizer_path or None, device, dtype,
        )

        self.model = model
        self.vae = vae
        self.qwen_model = qwen_model
        self.qwen_tok = qwen_tok
        self.t5_tok = t5_tok
        self.transformer_path = transformer_path
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path
        self.t5_tokenizer_path = t5_tokenizer_path
        self.attention_backend = backend
        self.mixed_precision = precision
        self.device = device
        self.dtype = dtype
        self.adapters = []
        self.last_lora_specs = []

    def apply_loras(self, lora_configs: list[dict[str, Any]]) -> list[Any]:
        """按 lora_configs (重新) inject adapters；commit 9 简化版每次都重 inject。"""
        # 先把旧 adapters 引用丢掉（让 GC 清掉 hook 来源）
        # NOTE：lycoris adapter 没暴露 detach 接口；下次 inject 会在 model 上叠
        # 新 hook，旧 hook 通过 multiplier=0 间接禁用 —— 但这会无限累积。
        # 为简化 commit 9：每次 ensure_loaded 后调 apply_loras，且对 model 做
        # 完全 reload（needs_reload=True 时 unload）。non-reload 路径下 adapters
        # 累积是已知问题，commit 后续 fix（暴露 adapter.detach()）。
        specs = [
            LoRASpec(path=str(lc.get("path", "")), scale=float(lc.get("scale", 1.0)))
            for lc in lora_configs
        ]
        if specs == self.last_lora_specs and self.adapters:
            return self.adapters
        # specs 变了 → 用 model reload 兜底（粗暴但 commit 9 可接受）
        if self.last_lora_specs and self.last_lora_specs != specs:
            logger.info("lora specs changed, reloading model to detach old adapters")
            saved_paths = (
                self.transformer_path, self.vae_path,
                self.text_encoder_path, self.t5_tokenizer_path,
                self.attention_backend, self.mixed_precision,
            )
            self.unload()
            self._load(
                transformer_path=saved_paths[0],
                vae_path=saved_paths[1],
                text_encoder_path=saved_paths[2],
                t5_tokenizer_path=saved_paths[3] or "",
                backend=saved_paths[4],
                precision=saved_paths[5],
            )
            _emit_evt("loaded")
        self.adapters = apply_loras(self.model, specs, self.device, self.dtype)
        self.last_lora_specs = specs
        self.model.eval()
        return self.adapters

    def unload(self) -> None:
        if not self.loaded:
            return
        logger.info("unloading model")
        self.model = None
        self.vae = None
        self.qwen_model = None
        self.qwen_tok = None
        self.t5_tok = None
        self.adapters = []
        self.last_lora_specs = []
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


CACHE = ModelCache()


# ---------------------------------------------------------------------------
# Generate 实现（复用 anima_generate.py 的循环逻辑）
# ---------------------------------------------------------------------------


def _set_lora_multiplier(adapter: Any, scale: float) -> None:
    if adapter.network is None:
        return
    adapter.network.multiplier = float(scale)
    for lora in getattr(adapter.network, "loras", []):
        if hasattr(lora, "multiplier"):
            lora.multiplier = float(scale)


def _apply_axis(
    axis: dict[str, Any],
    value: Any,
    *,
    cur_steps: int,
    cur_cfg_scale: float,
    cur_seed: int,
    cur_sampler: str,
    adapters: list[Any],
) -> tuple[int, float, int, str]:
    axis_type = axis["axis"]
    if axis_type == "steps":
        cur_steps = int(value)
    elif axis_type == "cfg_scale":
        cur_cfg_scale = float(value)
    elif axis_type == "seed":
        cur_seed = int(value)
    elif axis_type == "sampler_name":
        cur_sampler = str(value)
    elif axis_type == "lora_scale":
        idx = int(axis.get("lora_index") or 0)
        if idx < len(adapters):
            _set_lora_multiplier(adapters[idx], float(value))
    return cur_steps, cur_cfg_scale, cur_seed, cur_sampler


def _setup_monitor(cfg: dict[str, Any]) -> Any:
    """初始化 train_monitor（每个 task 一份独立 monitor_state.json）。

    前端通过 SSE monitor_state_updated 拿 samples + xy 元信息；图本身的
    bytes 走协议 image_done 事件入 server 内存 cache（commit 10 起）。
    sample_path 字段写虚拟路径（前端只用 split+pop 拿 filename 来构建
    /api/generate/{tid}/sample/{fn} URL），磁盘上不会有这个文件。
    """
    msf = cfg.get("__monitor_state_file")
    if not msf:
        return None
    try:
        from train_monitor import set_state_file, update_monitor
        set_state_file(msf)
        update_monitor(config={
            "type": "generate",
            "prompts": len(cfg.get("prompts") or []),
            "count": int(cfg.get("count", 1)),
            "steps": int(cfg.get("steps", 25)),
            "cfg_scale": float(cfg.get("cfg_scale", 4.0)),
        })
        return update_monitor
    except Exception as e:
        logger.warning("monitor 初始化失败: %s", e)
        return None


def _encode_png(img: Any) -> tuple[str, int]:
    """PIL.Image → PNG bytes → base64 string。返回 (b64_str, raw_byte_size)。"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), len(raw)


def _virtual_path(task_id: int, filename: str) -> str:
    """前端只用 split+pop 拿 filename，所以给个看起来像绝对路径的字符串。"""
    return f"/anima_gen_{task_id}/{filename}"


def _run_generate(req_id: str, task_id: int, cfg: dict[str, Any], output_dir: Path) -> None:
    """跑一次完整 generate（含可选 XY）。

    commit 10 起：PNG bytes base64 推 stdout（image_done 事件）→ server
    侧 InferenceDaemon 入 generate_cache；output_dir 不再写盘（保留参数
    给 anima_generate.py CLI 用法走 fallback 路径）。

    monitor_state.json 仍写（前端 sample_path SSE 链路兼容），但 sample_path
    是虚拟路径，磁盘上无对应文件 —— 前端只用它 split+pop 拿 filename 来
    构建 /api/generate/{tid}/sample/{fn} URL。
    """
    update_monitor = _setup_monitor(cfg)

    CACHE.ensure_loaded(cfg)
    adapters = CACHE.apply_loras(cfg.get("lora_configs", []))

    prompts: list[str] = cfg.get("prompts") or [
        "newest, safe, 1girl, masterpiece, best quality"
    ]
    negative_prompt: str = cfg.get("negative_prompt", "")
    width: int = int(cfg.get("width", 1024))
    height: int = int(cfg.get("height", 1024))
    steps: int = int(cfg.get("steps", 25))
    cfg_scale: float = float(cfg.get("cfg_scale", 4.0))
    sampler_name: str = cfg.get("sampler_name", "er_sde")
    scheduler: str = cfg.get("scheduler", "simple")
    count: int = max(1, int(cfg.get("count", 1)))
    base_seed: int = int(cfg.get("seed", 0))

    xy_matrix = cfg.get("xy_matrix")
    if xy_matrix is not None:
        _run_xy(
            req_id=req_id, task_id=task_id, cfg=cfg, output_dir=output_dir,
            xy_matrix=xy_matrix, adapters=adapters,
            prompt=prompts[0], negative_prompt=negative_prompt,
            base_seed=base_seed, base_steps=steps, base_cfg_scale=cfg_scale,
            base_sampler=sampler_name, scheduler=scheduler,
            height=height, width=width,
            update_monitor=update_monitor,
        )
        return

    total = count * len(prompts)
    _emit_for(req_id, "started", task_id=task_id, total=total)

    img_idx = 0
    for pi, prompt in enumerate(prompts):
        for ci in range(count):
            seed = (
                (base_seed + img_idx) if base_seed != 0
                else random.randint(0, 2**31 - 1)
            )
            torch.manual_seed(seed)
            random.seed(seed)
            try:
                img = _T.sample_image(
                    CACHE.model, CACHE.vae,
                    CACHE.qwen_model, CACHE.qwen_tok, CACHE.t5_tok,
                    prompt=prompt,
                    height=height,
                    width=width,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    negative_prompt=negative_prompt or None,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    device=CACHE.device,
                    dtype=CACHE.dtype,
                )
                fname = f"gen_{img_idx:04d}_p{pi}_c{ci}_s{seed}.png"
                vpath = _virtual_path(task_id, fname)
                b64, byte_size = _encode_png(img)
                if update_monitor:
                    update_monitor(sample_path=vpath, step=img_idx + 1)
                _emit_for(
                    req_id, "image_done",
                    filename=fname, path=vpath,
                    step=img_idx + 1, total=total,
                    image_b64=b64, byte_size=byte_size,
                )
            except Exception as e:
                logger.exception("generate failed")
                _emit_for(req_id, "image_error", step=img_idx + 1, message=str(e))
            img_idx += 1


def _run_xy(
    *,
    req_id: str,
    task_id: int,
    cfg: dict[str, Any],
    output_dir: Path,
    xy_matrix: dict[str, Any],
    adapters: list[Any],
    prompt: str,
    negative_prompt: str,
    base_seed: int,
    base_steps: int,
    base_cfg_scale: float,
    base_sampler: str,
    scheduler: str,
    height: int,
    width: int,
    update_monitor: Any,
) -> None:
    x_spec = xy_matrix["x"]
    y_spec = xy_matrix.get("y")
    x_values = x_spec["values"]
    y_values = y_spec["values"] if y_spec else [None]

    if base_seed == 0:
        base_seed = random.randint(0, 2**31 - 1)
        logger.info("XY 共享种子（cfg.seed=0 随机化）: %d", base_seed)

    base_scales = [float(s.scale) for s in CACHE.last_lora_specs]
    total = len(x_values) * len(y_values)
    _emit_for(req_id, "started", task_id=task_id, total=total)

    img_idx = 0
    for yi, yv in enumerate(y_values):
        for xi, xv in enumerate(x_values):
            for i, s in enumerate(base_scales):
                if i < len(adapters):
                    _set_lora_multiplier(adapters[i], s)

            cur_steps = base_steps
            cur_cfg_scale = base_cfg_scale
            cur_seed = base_seed
            cur_sampler = base_sampler

            cur_steps, cur_cfg_scale, cur_seed, cur_sampler = _apply_axis(
                x_spec, xv,
                cur_steps=cur_steps, cur_cfg_scale=cur_cfg_scale,
                cur_seed=cur_seed, cur_sampler=cur_sampler,
                adapters=adapters,
            )
            if y_spec is not None and yv is not None:
                cur_steps, cur_cfg_scale, cur_seed, cur_sampler = _apply_axis(
                    y_spec, yv,
                    cur_steps=cur_steps, cur_cfg_scale=cur_cfg_scale,
                    cur_seed=cur_seed, cur_sampler=cur_sampler,
                    adapters=adapters,
                )

            torch.manual_seed(cur_seed)
            random.seed(cur_seed)

            try:
                img = _T.sample_image(
                    CACHE.model, CACHE.vae,
                    CACHE.qwen_model, CACHE.qwen_tok, CACHE.t5_tok,
                    prompt=prompt,
                    height=height,
                    width=width,
                    steps=cur_steps,
                    cfg_scale=cur_cfg_scale,
                    negative_prompt=negative_prompt or None,
                    sampler_name=cur_sampler,
                    scheduler=scheduler,
                    device=CACHE.device,
                    dtype=CACHE.dtype,
                )
                fname = f"xy_x{xi:02d}_y{yi:02d}_s{cur_seed}.png"
                vpath = _virtual_path(task_id, fname)
                b64, byte_size = _encode_png(img)
                if update_monitor:
                    update_monitor(
                        sample_path=vpath,
                        step=img_idx + 1,
                        xy={"xi": xi, "yi": yi, "xv": xv, "yv": yv},
                    )
                _emit_for(
                    req_id, "image_done",
                    filename=fname, path=vpath,
                    step=img_idx + 1, total=total,
                    xy={"xi": xi, "yi": yi, "xv": xv, "yv": yv},
                    image_b64=b64, byte_size=byte_size,
                )
            except Exception as e:
                logger.exception("XY [%d,%d] failed", xi, yi)
                _emit_for(
                    req_id, "image_error",
                    step=img_idx + 1, message=str(e),
                    xy={"xi": xi, "yi": yi, "xv": xv, "yv": yv},
                )
            img_idx += 1


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------


def _handle_message(msg: dict[str, Any]) -> None:
    action = msg.get("action")
    req_id = msg.get("id", "")

    if action == "ping":
        _emit_for(req_id, "pong")
        return

    if action == "unload":
        CACHE.unload()
        _emit_evt("unloaded")
        return

    if action == "generate":
        task_id = int(msg.get("task_id", 0))
        cfg = msg.get("config") or {}
        output_dir = Path(msg.get("output_dir") or ".")
        try:
            _run_generate(req_id, task_id, cfg, output_dir)
            _emit_for(req_id, "done", task_id=task_id)
        except Exception as e:
            logger.exception("generate failed")
            _emit_for(req_id, "error", task_id=task_id, message=str(e))
        return

    logger.warning("unknown action: %r", action)
    _emit_for(req_id, "error", message=f"unknown action: {action!r}")


def main() -> int:
    _emit_evt("ready")
    logger.info("anima daemon ready, waiting for stdin commands")
    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("non-JSON stdin line: %r (%s)", line[:200], e)
                continue
            try:
                _handle_message(msg)
            except Exception:
                logger.exception("message handler crashed")
    except KeyboardInterrupt:
        pass
    finally:
        CACHE.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
