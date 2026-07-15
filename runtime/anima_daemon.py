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
    python runtime/anima_daemon.py
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
import threading
from pathlib import Path
from typing import Any, Optional

import torch

# 同 anima_generate.py 的 sys.path 处理（让 anima_train / studio 可 import）
# anima_train + train_monitor 都在 runtime/ 下，_THIS_DIR 即够。
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _REPO_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import anima_train as _T  # noqa: E402

from studio.domain.comfy_parity import force_comfy_parity_runtime_config  # noqa: E402
from studio.services.inference.core import LoRAMeta, LoRASpec, apply_loras, read_lora_meta  # noqa: E402

# 预热 transformers.generation → sklearn → scipy.special import 链。
# transformers 5.x 的 AutoModelForCausalLM.from_pretrained 在 load text encoder
# 时间接 import 这一串；scipy.special cold import 在 Windows + Python 3.13 + 已
# 加载 GB 级模型（system RAM 紧张）的环境下可能要几分钟（py-spy 实测）。挪到
# daemon import 阶段，趁 RAM 还宽松时一次性付掉。
try:
    import transformers.generation.candidate_generator  # noqa: F401
except Exception:
    pass

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


def _lora_topology(meta: LoRAMeta) -> tuple:
    # weight_decompose / rs_lora 改了网络结构（前者加 dora_scale 张量、后者改
    # effective alpha 公式），不同设置不能走热换权重路径，必须重新 inject。
    # lora_reg_dims 直接改单层 rank → 不同 pattern 配置同 base rank 也不能复用。
    reg = meta.lora_reg_dims
    reg_key: Any = tuple(sorted(reg.items())) if reg else None
    return (meta.rank, meta.alpha, meta.algo, meta.factor,
            meta.weight_decompose, meta.rs_lora, reg_key)


def _load_lora_state_dict(path: str, device: str, dtype: Any) -> dict[str, Any]:
    from safetensors import safe_open

    sd: dict[str, Any] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k).to(device=device, dtype=dtype)
    return sd


def _reload_adapter_weights(adapter: Any, spec: LoRASpec, device: str, dtype: Any) -> None:
    _set_lora_multiplier(adapter, spec.scale)
    result = adapter.load_state_dict(
        _load_lora_state_dict(spec.path, device, dtype),
        strict=False,
    )
    missing = len(getattr(result, "missing_keys", []) or [])
    unexpected = len(getattr(result, "unexpected_keys", []) or [])
    logger.info(
        f"已热换 LoRA 权重: {Path(spec.path).name} "
        f"(scale={spec.scale}; missing={missing}, unexpected={unexpected})"
    )


def _move_module_to_device(module: Any, device: str) -> None:
    if module is None or not hasattr(module, "to"):
        return
    module.to(device)


def _move_adapter_to_device(adapter: Any, device: str, dtype: Any) -> None:
    network = getattr(adapter, "network", None)
    if network is None or not hasattr(network, "to"):
        return
    network.to(device=device, dtype=dtype)


class GenerationCanceled(BaseException):
    """取消信号。

    刻意继承 BaseException 而非 Exception：取消由 step_callback 在采样步内
    抛出，而 sampler 对 step_callback 的调用包在 `except Exception: pass`
    里（回调失败不该毁掉采样）。继承 Exception 会被这层静默吞掉、无法
    中断采样。所有捕获点都显式写 `except GenerationCanceled`。
    """


_CANCEL_EVENTS: dict[str, threading.Event] = {}
_CANCEL_LOCK = threading.Lock()
_ACTIVE_WORKER: threading.Thread | None = None
_ACTIVE_WORKER_LOCK = threading.Lock()


def _register_cancel(req_id: str) -> threading.Event:
    event = threading.Event()
    with _CANCEL_LOCK:
        _CANCEL_EVENTS[req_id] = event
    return event


def _pop_cancel(req_id: str) -> None:
    with _CANCEL_LOCK:
        _CANCEL_EVENTS.pop(req_id, None)


def _request_cancel(req_id: str) -> bool:
    with _CANCEL_LOCK:
        event = _CANCEL_EVENTS.get(req_id)
    if event is None:
        return False
    event.set()
    return True


def _raise_if_canceled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCanceled()


def _torch_dtype_from_precision(value: str | None) -> torch.dtype:
    normalized = str(value or "fp32").lower().strip()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


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
        self.vae_precision: Optional[str] = None
        self.text_encoder_backend: Optional[str] = None
        self.t5_tokenizer_backend: Optional[str] = None
        self.device: Optional[str] = None
        self.dtype: Any = None
        self.lora_dtype: Any = torch.float32
        self.model: Any = None
        self.vae: Any = None
        self.qwen_model: Any = None
        self.qwen_tok: Any = None
        self.t5_tok: Any = None
        # adapters 必须保持引用，否则 forward hook 失效（lycoris closure）
        self.adapters: list[Any] = []
        self.last_lora_specs: list[LoRASpec] = []
        self.last_lora_metas: list[LoRAMeta] = []
        # 中间步预览用 latent2rgb 线性投影（见 _decode_latent2rgb_preview）——
        # 无外部模型 / 无下载，故 CACHE 不再持任何 preview decoder 状态。

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def ensure_loaded(self, cfg: dict[str, Any]) -> None:
        """按 cfg 决定是否需要 (重新) 加载。路径或后端变了 → 全重载。"""
        cfg = force_comfy_parity_runtime_config(
            cfg,
            force_exact_ksampler_backend=False,
        )
        backend = cfg.get("attention_backend", "none")
        precision = cfg.get("mixed_precision", "bf16")
        vae_precision = cfg.get("vae_precision", precision)
        text_encoder_backend = cfg.get("text_encoder_backend", "hf")
        t5_tokenizer_backend = cfg.get("t5_tokenizer_backend", "slow")
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
            or self.vae_precision != vae_precision
            or self.text_encoder_backend != text_encoder_backend
            or self.t5_tokenizer_backend != t5_tokenizer_backend
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
                vae_precision=vae_precision,
                text_encoder_backend=text_encoder_backend,
                t5_tokenizer_backend=t5_tokenizer_backend,
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
        vae_precision: str,
        text_encoder_backend: str,
        t5_tokenizer_backend: str,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = _torch_dtype_from_precision(precision)
        vae_dtype = _torch_dtype_from_precision(vae_precision)
        repo_root = _T.find_diffusion_pipe_root()
        use_flash = backend == "flash_attn"
        use_xformers = backend == "xformers"

        family = _T.get_family("anima")  # daemon 为 Anima 专属常驻壳（K2 接入形态 Phase 4 定）
        logger.info("loading transformer %s", transformer_path)
        model = family.load_dit(
            transformer_path, device, dtype,
            attention_backend=("flash_attn" if use_flash else "none"), repo_root=repo_root,
        )
        if use_xformers and not _T.enable_xformers(model):
            raise RuntimeError(
                "Exact ComfyUI KSampler parity is guaranteed only with xformers, "
                "but xformers could not be enabled"
            )

        logger.info("loading vae %s", vae_path)
        vae = family.load_vae(vae_path, device, vae_dtype)

        logger.info("loading text encoders %s", text_encoder_path)
        qwen_model, qwen_tok, t5_tok = family.load_text(
            text_encoder_path, device, dtype,
            t5_tokenizer_path=t5_tokenizer_path or None,
            comfy_qwen=text_encoder_backend == "comfy_qwen3",
            t5_fast=t5_tokenizer_backend == "fast",
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
        self.vae_precision = vae_precision
        self.text_encoder_backend = text_encoder_backend
        self.t5_tokenizer_backend = t5_tokenizer_backend
        self.device = device
        self.dtype = dtype
        self.lora_dtype = torch.float32
        self.adapters = []
        self.last_lora_specs = []
        self.last_lora_metas = []

    def apply_loras(self, lora_configs: list[dict[str, Any]]) -> list[Any]:
        """按 lora_configs inject adapters；同结构 checkpoint 切换时只热换权重。"""
        self._move_runtime_to_device()

        specs = [
            LoRASpec(path=str(lc.get("path", "")), scale=float(lc.get("scale", 1.0)))
            for lc in lora_configs
        ]
        if specs == self.last_lora_specs and self.adapters:
            return self.adapters

        current_metas: list[LoRAMeta] = []
        if specs:
            try:
                for spec in specs:
                    if not spec.path or not Path(spec.path).exists():
                        current_metas = []
                        break
                    current_metas.append(read_lora_meta(spec.path))
            except Exception:
                logger.exception("read LoRA metadata failed")
                current_metas = []

        can_hot_reload = (
            bool(self.adapters)
            and bool(self.last_lora_specs)
            and len(specs) == len(self.adapters) == len(self.last_lora_metas) == len(current_metas)
            and [_lora_topology(m) for m in current_metas]
            == [_lora_topology(m) for m in self.last_lora_metas]
        )
        if can_hot_reload:
            try:
                for adapter, spec in zip(self.adapters, specs):
                    _reload_adapter_weights(adapter, spec, self.device, self.lora_dtype)
            except Exception:
                logger.exception("LoRA hot reload failed; reinjecting adapters")
            else:
                self.last_lora_specs = specs
                self.last_lora_metas = current_metas
                self.model.eval()
                return self.adapters

        all_detached = True
        for adapter in self.adapters:
            try:
                if not adapter.detach():
                    all_detached = False
            except Exception:
                logger.exception("adapter detach failed")
                all_detached = False
        self.adapters = []

        if not all_detached and self.last_lora_specs:
            logger.warning("detach failed, reloading model to ensure clean state")
            saved_paths = (
                self.transformer_path, self.vae_path,
                self.text_encoder_path, self.t5_tokenizer_path,
                self.attention_backend, self.mixed_precision,
                self.vae_precision, self.text_encoder_backend,
                self.t5_tokenizer_backend,
            )
            self.unload()
            self._load(
                transformer_path=saved_paths[0],
                vae_path=saved_paths[1],
                text_encoder_path=saved_paths[2],
                t5_tokenizer_path=saved_paths[3] or "",
                backend=saved_paths[4],
                precision=saved_paths[5],
                vae_precision=saved_paths[6] or saved_paths[5],
                text_encoder_backend=saved_paths[7] or "hf",
                t5_tokenizer_backend=saved_paths[8] or "slow",
            )
            _emit_evt("loaded")
        self.last_lora_specs = []
        self.last_lora_metas = []

        self.adapters = apply_loras(self.model, specs, self.device, self.lora_dtype)
        self.last_lora_specs = specs
        self.last_lora_metas = current_metas
        self.model.eval()
        return self.adapters

    def _move_runtime_to_device(self) -> None:
        if not self.device:
            return
        _move_module_to_device(self.model, self.device)
        _move_module_to_device(self.qwen_model, self.device)
        for adapter in self.adapters:
            _move_adapter_to_device(adapter, self.device, self.lora_dtype)

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
        self.last_lora_metas = []
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
    adapters: list[Any],
) -> tuple[int, float]:
    """处理纯数值/scale 轴。lora_ckpt 不在这处理（需要重新 inject，由
    _run_xy 单独走 CACHE.apply_loras 路径）。

    lora_scale 是**全局轴** —— 把所有 adapter 的 multiplier 都设成 cell 值；
    原本不同 LoRA 的相对权重会消失，但 UI 上权重轴的语义就是「扫一个绝对值」
    而非「扫某一条 LoRA 的相对值」。
    """
    axis_type = axis["axis"]
    if axis_type == "steps":
        cur_steps = int(value)
    elif axis_type == "cfg_scale":
        cur_cfg_scale = float(value)
    elif axis_type == "lora_scale":
        for ad in adapters:
            _set_lora_multiplier(ad, float(value))
    return cur_steps, cur_cfg_scale


def _setup_monitor(cfg: dict[str, Any]) -> Any:
    """初始化 train_monitor（每个 task 一份独立 monitor_state.json）。

    前端通过 SSE monitor_progress 拿 samples + xy 元信息；图本身的
    bytes 走协议 image_done 事件入 server 内存 cache（commit 10 起）。
    sample_path 字段写虚拟路径（前端只用 split+pop 拿 filename 来构建
    /api/generate/{tid}/sample/{fn} URL），磁盘上不会有这个文件。
    """
    msf = cfg.get("__monitor_state_file")
    if not msf:
        return None
    try:
        from train_monitor import reset_monitor, set_state_file, update_monitor
        # 关键：daemon 复用进程跨 task 时 MONITOR_STATE 残留，必须清。
        # 否则上一 task 的 samples 会混入新 task 的 monitor_state.json，
        # 前端用 currentTask.id 拼 URL 拿旧 filename → 404 破图。
        reset_monitor()
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


def _encode_jpeg(img: Any, quality: int = 80) -> tuple[str, int]:
    """中间步预览编码：JPEG 80% 默认，比 PNG 小 ~5x。返回 (b64_str, byte_size)。"""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), len(raw)


def _build_preview_callback(
    req_id: str,
    every_n: int,
    cancel_event: threading.Event | None = None,
) -> Any:
    """每步推 preview_step 事件；节流命中时附 latent2rgb 预览图 image_b64。

    用户反馈：进度条始终要可见（"当前在做什么，第几步"），预览图按需。
    本 callback 拆成两路：
      - 永远 emit preview_step { step, total } —— 前端进度条
      - every_n>0 且步命中（含末步）→ latent2rgb decode + 附 image_b64
    callback 在 daemon 主线程同步执行；空逻辑 ~微秒级，latent2rgb（纯线性
    投影，无 NN forward）+ JPEG 编码 ~1-2ms。

    cancel_event 注入后每步检查，取消延迟从"整张图"降到"一步"。
    """
    def _cb(step: int, total: int, latent: Any) -> None:
        _raise_if_canceled(cancel_event)
        # 是否带预览图：preview_every_n_steps>0 + 节流命中（含末步）。
        with_image = False
        b64: Optional[str] = None
        byte_size = 0
        if every_n > 0 and (step % every_n == 0 or step == total - 1):
            img = _decode_latent2rgb_preview(latent)
            if img is not None:
                b64, byte_size = _encode_jpeg(img, quality=80)
                with_image = True
        payload: dict[str, Any] = {"step": step + 1, "total": total}
        if with_image:
            payload["image_b64"] = b64
            payload["byte_size"] = byte_size
        _emit_for(req_id, "preview_step", **payload)
    return _cb


# Wan2.1 VAE 的 latent→RGB 线性投影系数（16 通道 → RGB）。
# Anima 的 VAE 是 Qwen-Image VAE（models/vae/qwen_image_vae.safetensors），其 latent
# 空间就是 Wan2.1 的 16-ch 空间——ComfyUI supported_models.py 里
# `QwenImage.latent_format = latent_formats.Wan21`，且本仓库 models.py 的 VAE
# latent2rgb 快速预览系数（"模糊但能看出图"）。系数表已收编进 ModelSpec
# （families/anima，D17）——单一来源，K2 同 latent 空间时直接复用。之前误用
# TAEFlux（Flux VAE 的 tiny decoder）解码这个 WAN latent，空间不匹配 →
# 颜色反相/错乱，已废弃。
from training.families.anima import ANIMA_SPEC as _ANIMA_SPEC  # noqa: E402

_WAN21_LATENT_RGB_FACTORS = [list(r) for r in _ANIMA_SPEC.latent.rgb_factors]
_WAN21_LATENT_RGB_BIAS = list(_ANIMA_SPEC.latent.rgb_bias)
# 预览放大目标（最长边像素）。latent 是 1/8 分辨率（1024²→128²），latent2rgb 直出
# 128²；放大到 512 让前端铺满时不至于过糊，JPEG 仍很小。
_PREVIEW_TARGET_PX = 512


def _decode_latent2rgb_preview(latent: Any) -> Optional[Any]:
    """latent → Wan2.1 latent2rgb 线性投影 → PIL.Image。失败返 None（preview 不阻塞）。

    Anima latent shape：[B, 16, F=1, H, W]。对齐 ComfyUI Latent2RGBPreviewer：
      x0 = latent[0, :, 0]                          # [16, H, W]
      rgb[h,w,r] = Σ_c x0[c,h,w]·factors[c,r] + bias[r]
      img = ((rgb + 1) / 2).clamp(0,1) · 255        # [-1,1] → [0,255]
    """
    try:
        import numpy as np
        from PIL import Image
        with torch.no_grad():
            x = latent[0, :, 0].float()  # [16, H, W]
            factors = torch.tensor(
                _WAN21_LATENT_RGB_FACTORS, device=x.device, dtype=x.dtype,
            )  # [16, 3]
            bias = torch.tensor(
                _WAN21_LATENT_RGB_BIAS, device=x.device, dtype=x.dtype,
            )  # [3]
            rgb = torch.einsum("chw,cr->hwr", x, factors) + bias  # [H, W, 3]
            rgb = ((rgb + 1.0) / 2.0).clamp(0.0, 1.0)
            arr = (rgb.cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3]
            img = Image.fromarray(arr)
            # 放大到 _PREVIEW_TARGET_PX 最长边（保持比例），前端再铺满结果区
            w, h = img.size
            scale = _PREVIEW_TARGET_PX / max(w, h)
            if scale > 1.0:
                img = img.resize(
                    (max(1, round(w * scale)), max(1, round(h * scale))),
                    Image.BILINEAR,
                )
            return img
    except Exception:
        logger.exception("latent2rgb preview decode failed")
        return None


def _virtual_path(task_id: int, filename: str) -> str:
    """前端只用 split+pop 拿 filename，所以给个看起来像绝对路径的字符串。"""
    return f"/anima_gen_{task_id}/{filename}"


def _run_generate(
    req_id: str,
    task_id: int,
    cfg: dict[str, Any],
    output_dir: Path,
    cancel_event: threading.Event | None = None,
) -> None:
    """跑一次完整 generate（含可选 XY）。

    commit 10 起：PNG bytes base64 推 stdout（image_done 事件）→ server
    侧 InferenceDaemon 入 generate_cache；output_dir 不再写盘（保留参数
    给 anima_generate.py CLI 用法走 fallback 路径）。

    monitor_state.json 仍写（前端 sample_path SSE 链路兼容），但 sample_path
    是虚拟路径，磁盘上无对应文件 —— 前端只用它 split+pop 拿 filename 来
    构建 /api/generate/{tid}/sample/{fn} URL。
    """
    cfg = force_comfy_parity_runtime_config(
        cfg,
        force_exact_ksampler_backend=False,
    )
    update_monitor = _setup_monitor(cfg)

    _raise_if_canceled(cancel_event)
    # phase 上报：加载模型/LoRA 阶段（clip/sample/vae 由 sample_image 内部报）→ 进度条覆盖全流程
    _emit_for(req_id, "phase", name="load")
    CACHE.ensure_loaded(cfg)
    adapters = CACHE.apply_loras(cfg.get("lora_configs", []))

    # 进度推送：永远建 callback 推 preview_step（含 step/total）；
    # preview_every_n_steps>0 时附 image_b64 中间预览图（commit 14）。
    preview_every = int(cfg.get("preview_every_n_steps", 0) or 0)
    preview_callback = _build_preview_callback(req_id, preview_every, cancel_event)

    def _phase_cb(name: str) -> None:
        _emit_for(req_id, "phase", name=name)

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
            preview_callback=preview_callback,
            phase_callback=_phase_cb,
            cancel_event=cancel_event,
        )
        return

    total = count * len(prompts)
    _emit_for(req_id, "started", task_id=task_id, total=total)

    img_idx = 0
    image_done_count = 0
    image_errors: list[str] = []
    for pi, prompt in enumerate(prompts):
        for ci in range(count):
            _raise_if_canceled(cancel_event)
            seed = (
                (base_seed + img_idx) if base_seed != 0
                else random.randint(0, 2**31 - 1)
            )
            torch.manual_seed(seed)
            random.seed(seed)
            _emit_for(
                req_id, "image_started",
                batch_idx=img_idx, batch_total=total, total_steps=steps,
            )
            try:
                img = _T.sample_image(
                    CACHE.model, CACHE.vae,
                    CACHE.qwen_model, CACHE.qwen_tok, CACHE.t5_tok,
                    prompt=prompt,
                    height=height,
                    width=width,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    negative_prompt=negative_prompt,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    device=CACHE.device,
                    dtype=CACHE.dtype,
                    step_callback=preview_callback,
                    phase_callback=_phase_cb,
                    seed=seed,
                )
                CACHE._move_runtime_to_device()
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
                image_done_count += 1
            except GenerationCanceled:
                raise
            except Exception as e:
                logger.exception("generate failed")
                image_errors.append(str(e))
                _emit_for(req_id, "image_error", step=img_idx + 1, message=str(e))
            img_idx += 1

    if image_done_count == 0 and image_errors:
        raise RuntimeError(f"all generated images failed: {image_errors[-1]}")


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
    preview_callback: Any = None,
    phase_callback: Any = None,
    cancel_event: threading.Event | None = None,
) -> None:
    x_spec = xy_matrix["x"]
    y_spec = xy_matrix.get("y")
    x_values = x_spec["values"]
    y_values = y_spec["values"] if y_spec else [None]

    if base_seed == 0:
        base_seed = random.randint(0, 2**31 - 1)
        logger.info("XY 共享种子（cfg.seed=0 随机化）: %d", base_seed)

    base_scales = [float(s.scale) for s in CACHE.last_lora_specs]
    base_lora_paths = [str(s.path) for s in CACHE.last_lora_specs]
    total = len(x_values) * len(y_values)
    _emit_for(req_id, "started", task_id=task_id, total=total)

    def _swap_ckpt_for_axis(spec: dict[str, Any], val: Any, lora_configs: list[dict[str, Any]]) -> None:
        """axis=lora_ckpt 时把 lora_configs[lora_index].path 改成 val。"""
        if spec.get("axis") != "lora_ckpt":
            return
        idx = int(spec.get("lora_index") or 0)
        if 0 <= idx < len(lora_configs):
            lora_configs[idx]["path"] = str(val)

    img_idx = 0
    image_done_count = 0
    image_errors: list[str] = []
    for yi, yv in enumerate(y_values):
        for xi, xv in enumerate(x_values):
            _raise_if_canceled(cancel_event)
            # lora_ckpt 切换：mutate cfg.lora_configs 的 path 然后调
            # CACHE.apply_loras —— commit 20 detach 路径会快速 reinject。
            x_is_ckpt = x_spec.get("axis") == "lora_ckpt"
            y_is_ckpt = y_spec is not None and y_spec.get("axis") == "lora_ckpt"
            if x_is_ckpt or y_is_ckpt:
                lora_configs = [
                    {"path": p, "scale": s}
                    for p, s in zip(base_lora_paths, base_scales)
                ]
                _swap_ckpt_for_axis(x_spec, xv, lora_configs)
                if y_spec is not None and yv is not None:
                    _swap_ckpt_for_axis(y_spec, yv, lora_configs)
                adapters = CACHE.apply_loras(lora_configs)
                base_scales = [float(s.scale) for s in CACHE.last_lora_specs]

            for i, s in enumerate(base_scales):
                if i < len(adapters):
                    _set_lora_multiplier(adapters[i], s)

            cur_steps = base_steps
            cur_cfg_scale = base_cfg_scale

            cur_steps, cur_cfg_scale = _apply_axis(
                x_spec, xv,
                cur_steps=cur_steps, cur_cfg_scale=cur_cfg_scale,
                adapters=adapters,
            )
            if y_spec is not None and yv is not None:
                cur_steps, cur_cfg_scale = _apply_axis(
                    y_spec, yv,
                    cur_steps=cur_steps, cur_cfg_scale=cur_cfg_scale,
                    adapters=adapters,
                )

            cur_seed = base_seed
            torch.manual_seed(cur_seed)
            random.seed(cur_seed)

            _emit_for(
                req_id, "image_started",
                batch_idx=img_idx, batch_total=total, total_steps=cur_steps,
            )
            try:
                img = _T.sample_image(
                    CACHE.model, CACHE.vae,
                    CACHE.qwen_model, CACHE.qwen_tok, CACHE.t5_tok,
                    prompt=prompt,
                    height=height,
                    width=width,
                    steps=cur_steps,
                    step_callback=preview_callback,
                    phase_callback=phase_callback,
                    cfg_scale=cur_cfg_scale,
                    negative_prompt=negative_prompt,
                    sampler_name=base_sampler,
                    scheduler=scheduler,
                    device=CACHE.device,
                    dtype=CACHE.dtype,
                    seed=cur_seed,
                )
                CACHE._move_runtime_to_device()
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
                image_done_count += 1
            except GenerationCanceled:
                raise
            except Exception as e:
                logger.exception("XY [%d,%d] failed", xi, yi)
                image_errors.append(str(e))
                _emit_for(
                    req_id, "image_error",
                    step=img_idx + 1, message=str(e),
                    xy={"xi": xi, "yi": yi, "xv": xv, "yv": yv},
                )
            img_idx += 1

    if image_done_count == 0 and image_errors:
        raise RuntimeError(f"all generated images failed: {image_errors[-1]}")


def _run_generate_worker(
    req_id: str,
    task_id: int,
    cfg: dict[str, Any],
    output_dir: Path,
    cancel_event: threading.Event,
) -> None:
    try:
        _run_generate(req_id, task_id, cfg, output_dir, cancel_event)
        _emit_for(req_id, "done", task_id=task_id)
    except GenerationCanceled:
        logger.info("generate canceled: task_id=%s", task_id)
        _emit_for(req_id, "canceled", task_id=task_id)
    except Exception as e:
        logger.exception("generate failed")
        _emit_for(req_id, "error", task_id=task_id, message=str(e))
    finally:
        _pop_cancel(req_id)
        with _ACTIVE_WORKER_LOCK:
            global _ACTIVE_WORKER
            _ACTIVE_WORKER = None


def _start_generate_worker(req_id: str, task_id: int, cfg: dict[str, Any], output_dir: Path) -> bool:
    global _ACTIVE_WORKER
    with _ACTIVE_WORKER_LOCK:
        if _ACTIVE_WORKER is not None and _ACTIVE_WORKER.is_alive():
            return False
        cancel_event = _register_cancel(req_id)
        worker = threading.Thread(
            target=_run_generate_worker,
            args=(req_id, task_id, cfg, output_dir, cancel_event),
            daemon=False,
            name=f"generate-{task_id}",
        )
        _ACTIVE_WORKER = worker
        worker.start()
        return True


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

    if action == "cancel":
        if _request_cancel(str(msg.get("target_id") or req_id)):
            _emit_for(req_id, "cancel_ack")
        else:
            _emit_for(req_id, "cancel_missed")
        return

    if action == "generate":
        task_id = int(msg.get("task_id", 0))
        cfg = msg.get("config") or {}
        output_dir = Path(msg.get("output_dir") or ".")
        if not _start_generate_worker(req_id, task_id, cfg, output_dir):
            _emit_for(req_id, "error", task_id=task_id, message="daemon is already running a task")
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
        with _ACTIVE_WORKER_LOCK:
            worker = _ACTIVE_WORKER
        if worker is not None:
            worker.join()
        CACHE.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
