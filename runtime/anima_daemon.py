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
from studio.services.inference.core import (  # noqa: E402
    DeferredVAE,
    LoRAMeta,
    LoRASpec,
    apply_loras,
    read_lora_meta,
    release_vae_after_decode,
)

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

_GIB = 1024 ** 3
_TE_LOAD_FALLBACK_FP8 = 7 * _GIB
_TE_LOAD_FALLBACK_FULL = 11 * _GIB
_TE_ENCODE_FALLBACK = 2 * _GIB


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


def _swap_discount_ratio(family_id: str, blocks_to_swap: int) -> float:
    """block swap 下不会进显存的权重比例（显存预算折扣）。

    比例而非字节：fp8 与 bf16 的文件大小差一倍，按字节折扣会在 fp8 场景把护栏
    折扣穿（训练侧同款，见 training/sysmem.check_load_budget）。
    族不支持 / 查询失败返回 0，护栏退化成保守。
    """
    if blocks_to_swap <= 0:
        return 0.0
    try:
        family = _T.get_family(family_id)
        ratio_fn = getattr(family, "swapped_param_ratio", None)
        return float(ratio_fn(blocks_to_swap)) if ratio_fn else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _move_adapter_to_device(adapter: Any, device: str, dtype: Any) -> None:
    network = getattr(adapter, "network", None)
    if network is None or not hasattr(network, "to"):
        return
    network.to(device=device, dtype=dtype)


def _cuda_mem_info(device: str) -> tuple[int, int] | None:
    try:
        target = torch.device(device)
        if target.type != "cuda" or not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(target)
        return int(free), int(total)
    except Exception:
        return None


def _is_cuda_oom(exc: BaseException) -> bool:
    """Recognize CUDA allocator failures without hiding unrelated errors."""
    return "cuda out of memory" in str(exc).lower()


def _recover_cuda_allocator(device: str) -> None:
    """Drop failed-forward temporaries before one deterministic retry."""
    import gc

    gc.collect()
    try:
        target = torch.device(device)
    except Exception:
        return
    if target.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize(target)
    except Exception:
        pass
    try:
        torch._C._cuda_clearCublasWorkspaces()
    except Exception:
        pass
    torch.cuda.empty_cache()


def _sample_with_cuda_oom_retry(
    sample_once: Any,
    *,
    seed: int,
    device: str,
    before_retry: Any = None,
) -> Any:
    """Retry one transient CUDA OOM with the exact same random seed.

    A large FP8 DiT consists of many CUDA allocations.  On Windows/WDDM, the
    first forward immediately after a CPU -> GPU round-trip can occasionally
    fail a small allocation even while ``mem_get_info`` reports many GiB free.
    Once the failed-forward tensors are released, the same forward succeeds.
    Keep this recovery narrow: CUDA OOM only, CUDA device only, one retry.
    """
    try:
        return sample_once()
    except Exception as exc:
        if torch.device(device).type != "cuda" or not _is_cuda_oom(exc):
            raise
        logger.warning(
            "transient CUDA OOM during generation; clearing allocator state "
            "and retrying once with seed=%d",
            seed,
        )

    if callable(before_retry):
        before_retry()
    _recover_cuda_allocator(device)
    torch.manual_seed(seed)
    random.seed(seed)
    return sample_once()


def _te_auto_safety_margin(total_bytes: int) -> int:
    """Keep residency headroom to avoid WDDM paging before CUDA OOM.

    ``mem_get_info`` describes allocatable CUDA memory, not the point where
    Windows starts evicting GPU allocations. A measured 32 GiB run remained
    below CUDA OOM yet made a 16 second TE load take more than three minutes.
    Reserving 40% avoids that cliff; larger cards can still use co-residency.
    """
    return max(_GIB, int(total_bytes * 0.40))


def _should_yield_dit_for_te(
    policy: str,
    device: str,
    te_increment_bytes: int,
) -> tuple[bool, dict[str, int] | None]:
    """Return task-level DiT yield decision and its auditable memory budget."""
    normalized = str(policy or "auto")
    if normalized == "performance":
        return False, None
    if normalized == "save_vram":
        return True, None
    info = _cuda_mem_info(device)
    if info is None:
        # auto 的目标是保护峰值；读不到预算时保守顺序化。
        return True, None
    free, total = info
    margin = _te_auto_safety_margin(total)
    required = max(0, int(te_increment_bytes)) + margin
    return free < required, {
        "free": free,
        "total": total,
        "te_increment": max(0, int(te_increment_bytes)),
        "margin": margin,
        "required": required,
    }


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
        self.family_id: Optional[str] = None
        self.family: Any = None
        self.transformer_path: Optional[str] = None
        self.vae_path: Optional[str] = None
        self.text_encoder_path: Optional[str] = None
        self.t5_tokenizer_path: Optional[str] = None
        self.attention_backend: Optional[str] = None
        self.mixed_precision: Optional[str] = None
        self.vae_precision: Optional[str] = None
        self.vae_tiling: str = "auto"
        # 只控制 fp8 底模 LoRA merge 临时 delta；不属于底模 identity，切换时
        # detach + 重 merge 即可，不应触发 DiT/VAE 全重载。
        self.lora_merge_precision: str = "fp32"
        self.text_encoder_backend: Optional[str] = None
        self.t5_tokenizer_backend: Optional[str] = None
        self.ram_guard: bool = True
        #: TE 先行栈的身份键（family_id, text_encoder_path）——ensure_text_ready
        #: 据此复用/重建；_load 据此避免重复加载
        self._text_ready_key: Optional[tuple] = None
        # 首次安全的 TE 先行加载用 CUDA allocator 实测校准。value =
        # (load+encode 峰值增量, TE 已驻留时仅 encode 的额外增量)。
        self._te_peak_calibration: dict[tuple, tuple[int, int]] = {}
        self.device: Optional[str] = None
        self.dtype: Any = None
        self.lora_dtype: Any = torch.float32
        self.model: Any = None
        #: block swap（training.block_swap.PinnedBlockSwap）。属于底模 identity：
        #: blocks_to_swap 改变要重载 DiT（换出层是在 loader 里就不上卡的）。
        self.blocks_to_swap: int = 0
        self.block_swap: Any = None
        self.vae: Any = None
        # 族 opaque 文本栈（anima=(qwen_model, qwen_tok, t5_tok) 三元组，
        # krea2=Krea2TextStack）——只经 family.sample_image 消费，daemon 不拆包
        self.text_stack: Any = None
        # adapters 必须保持引用，否则 forward hook 失效（lycoris closure）
        self.adapters: list[Any] = []
        self.last_lora_specs: list[LoRASpec] = []
        self.last_lora_metas: list[LoRAMeta] = []
        self.last_lora_merge_precision: Optional[str] = None
        # 中间步预览用 latent2rgb 线性投影（见 _decode_latent2rgb_preview）——
        # 无外部模型 / 无下载，故 CACHE 不再持任何 preview decoder 状态。

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def ensure_text_ready(self, cfg: dict[str, Any]) -> None:
        """krea2 两段加载第一段：先就绪 TE 栈（不动 DiT）。

        TE 先行编排（任务驱动分阶段，训练两段式的推理版）：任务开始先让
        文本栈就绪 → precache 编码 → 彻底释放 → 才加载 13GB DiT——任一
        时刻 GPU 上只有一个大模型（受控实测预编码期三者同驻峰值 24.1GB
        → 错开后 ~15GB，16GB 卡免让位）。TE 参数变化只重建文本栈（在线
        LRU 随之作废），不再触发全家桶重载。anima 族 no-op（TE 常驻语义
        走 _load 全家桶）。
        """
        cfg = force_comfy_parity_runtime_config(
            cfg, force_exact_ksampler_backend=False,
        )
        family_id = str(cfg.get("model_family") or "anima")
        if family_id != "krea2":
            return
        repo_root = _T.find_diffusion_pipe_root()
        bases = [Path.cwd(), _THIS_DIR, repo_root]
        text_encoder_path = _T.resolve_path_best_effort(
            cfg["text_encoder_path"], bases,
        )
        key = (family_id, text_encoder_path)
        if self.text_stack is not None and self._text_ready_key == key:
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = _torch_dtype_from_precision(cfg.get("mixed_precision", "bf16"))
        family = _T.get_family(family_id)
        logger.info("loading text encoders (TE 先行) %s", text_encoder_path)
        self.text_stack = family.load_text(
            text_encoder_path, device, dtype,
            purpose="generate",
            cache_enabled=False,
        )
        self._text_ready_key = key
        self.text_encoder_path = text_encoder_path

    def ensure_loaded(self, cfg: dict[str, Any]) -> None:
        """按 cfg 决定是否需要 (重新) 加载。路径或后端变了 → 全重载。"""
        cfg = force_comfy_parity_runtime_config(
            cfg,
            force_exact_ksampler_backend=False,
        )
        family_id = str(cfg.get("model_family") or "anima")
        backend = cfg.get("attention_backend", "none")
        precision = cfg.get("mixed_precision", "bf16")
        vae_precision = cfg.get("vae_precision", precision)
        vae_tiling = str(cfg.get("vae_tiling") or "auto")
        blocks_to_swap = int(cfg.get("blocks_to_swap") or 0)
        self.lora_merge_precision = str(
            cfg.get("lora_merge_precision") or "fp32"
        ).lower()
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

        self.ram_guard = bool(cfg.get("ram_guard", True))

        # 比较是否需要 reload（换族 = 换整套模型栈，全重载）
        needs_reload = (
            not self.loaded
            or self.family_id != family_id
            or self.transformer_path != transformer_path
            or self.vae_path != vae_path
            or self.text_encoder_path != text_encoder_path
            or self.t5_tokenizer_path != t5_tokenizer_path
            or self.attention_backend != backend
            or self.mixed_precision != precision
            or self.vae_precision != vae_precision
            or self.vae_tiling != vae_tiling
            # 换出哪些层是在 loader 里决定的（换出层根本不上卡），改了必须重载
            or self.blocks_to_swap != blocks_to_swap
            or self.text_encoder_backend != text_encoder_backend
            or self.t5_tokenizer_backend != t5_tokenizer_backend
        )

        if needs_reload:
            from training.sysmem import check_load_budget

            check_load_budget(
                self.ram_guard,
                weight_paths=[transformer_path, vae_path],
                stage="模型加载",
                vram_discount_ratio=_swap_discount_ratio(
                    family_id, blocks_to_swap,
                ),
            )
            # keep_text：TE 先行栈刚编码完（LRU 已填充），重载不清它
            self.unload(keep_text=True)
            self._load(
                family_id=family_id,
                transformer_path=transformer_path,
                vae_path=vae_path,
                text_encoder_path=text_encoder_path,
                t5_tokenizer_path=t5_tokenizer_path,
                backend=backend,
                precision=precision,
                vae_precision=vae_precision,
                text_encoder_backend=text_encoder_backend,
                t5_tokenizer_backend=t5_tokenizer_backend,
                vae_tiling=vae_tiling,
                blocks_to_swap=blocks_to_swap,
            )
            _emit_evt("loaded")

    def _load(
        self,
        *,
        family_id: str,
        transformer_path: str,
        vae_path: str,
        text_encoder_path: str,
        t5_tokenizer_path: str,
        backend: str,
        precision: str,
        vae_precision: str,
        text_encoder_backend: str,
        t5_tokenizer_backend: str,
        vae_tiling: str = "auto",
        blocks_to_swap: int = 0,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = _torch_dtype_from_precision(precision)
        vae_dtype = _torch_dtype_from_precision(vae_precision)
        repo_root = _T.find_diffusion_pipe_root()
        use_flash = backend == "flash_attn"
        use_xformers = backend == "xformers"

        family = _T.get_family(family_id)
        logger.info("loading transformer [%s] %s", family_id, transformer_path)
        swap_extra: dict[str, Any] = {}
        if blocks_to_swap > 0:
            if "block_swap" not in family.spec.capabilities:
                raise RuntimeError(
                    f"model_family='{family_id}' 不支持 block swap，但 "
                    f"blocks_to_swap={blocks_to_swap}。请置 0。"
                )
            # 换出层由 loader 直接落 CPU pinned，不经显存
            swap_extra["blocks_to_swap"] = blocks_to_swap
            logger.info("block swap: last %d blocks stay in RAM", blocks_to_swap)
        model = family.load_dit(
            transformer_path, device, dtype,
            attention_backend=("flash_attn" if use_flash else "none"), repo_root=repo_root,
            purpose="generate",
            **swap_extra,
        )
        if use_xformers and not _T.enable_xformers(model):
            raise RuntimeError(
                "Exact ComfyUI KSampler parity is guaranteed only with xformers, "
                "but xformers could not be enabled"
            )

        # VAE is deliberately absent from VRAM during model load, LoRA merge,
        # prompt encoding and diffusion sampling.  The proxy loads it on the
        # first decode-side attribute lookup inside family.sample_image().
        vae = DeferredVAE(
            lambda: family.load_vae(
                vae_path, device, vae_dtype, tiling=vae_tiling,
            ),
            device=device,
            label=f"VAE {vae_path}",
        )

        # 族 opaque 文本栈，不拆包。krea2 的 TE 先行栈（ensure_text_ready）
        # 已就绪且身份匹配时复用——保住预编码 LRU，不重复加载。
        if (
            family_id == "krea2"
            and self.text_stack is not None
            and self._text_ready_key == (family_id, text_encoder_path)
        ):
            text_stack = self.text_stack
        else:
            logger.info("loading text encoders %s", text_encoder_path)
            text_stack = family.load_text(
                text_encoder_path, device, dtype,
                t5_tokenizer_path=t5_tokenizer_path or None,
                comfy_qwen=text_encoder_backend == "comfy_qwen3",
                t5_fast=t5_tokenizer_backend == "fast",
                purpose="generate",
                cache_enabled=False,
            )

        # block swap：换出层此刻已在 CPU pinned（loader 直接落的），这里建管理
        # 对象并挂前向钩子。LoRA 在 apply_loras 里按任务注入/merge，那边会先
        # restore_masters() 保证 merge 落在主副本上（doc §9.6）。
        self.block_swap = None
        self.blocks_to_swap = blocks_to_swap
        if blocks_to_swap > 0:
            from training.block_swap import PinnedBlockSwap

            self.block_swap = PinnedBlockSwap(model.blocks, blocks_to_swap, device)
            self.block_swap.attach()

        self.family_id = family_id
        self.family = family
        self.model = model
        self.vae = vae
        self.text_stack = text_stack
        self.transformer_path = transformer_path
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path
        self.t5_tokenizer_path = t5_tokenizer_path
        self.attention_backend = backend
        self.mixed_precision = precision
        self.vae_precision = vae_precision
        self.vae_tiling = vae_tiling
        self.text_encoder_backend = text_encoder_backend
        self.t5_tokenizer_backend = t5_tokenizer_backend
        self.device = device
        self.dtype = dtype
        self.lora_dtype = torch.float32
        self.adapters = []
        self.last_lora_specs = []
        self.last_lora_metas = []
        self.last_lora_merge_precision = None
        # 大权重加载完：mmap 文件缓存页（DiT 13-26GB + VAE）归还系统，
        # 防物理内存紧张机器换页卡死（TE lazy 加载后由 ensure_model 再 trim）
        from training.sysmem import trim_working_set

        trim_working_set()

    def apply_loras(self, lora_configs: list[dict[str, Any]]) -> list[Any]:
        """按 lora_configs inject adapters；同结构 checkpoint 切换时只热换权重。"""
        self._move_runtime_to_device()
        if self.block_swap is not None:
            # 关键顺序（doc §9.6）：fp8 底模的 LoRA 走**权重 merge**，而跑过前向后
            # 换出层的 .data 指向的是会被下一层换入覆盖的 GPU 槽位——直接 merge
            # 会静默丢失。先把参数指回 CPU pinned 主副本，delta 落在主副本上，
            # 之后每次换入带的都是 merged 权重。
            # bf16 底模走 lycoris hook 不改权重，restore 对它是无害的 no-op 语义。
            self.block_swap.restore_masters()

        specs = [
            LoRASpec(path=str(lc.get("path", "")), scale=float(lc.get("scale", 1.0)))
            for lc in lora_configs
        ]
        # 动态 adapter 不受 merge 精度影响；fp8 merge 句柄则在设置改变时
        # 必须 detach 后按新精度重 merge。
        adapters_are_dynamic = bool(self.adapters) and all(
            getattr(a, "supports_hot_reload", True) for a in self.adapters
        )
        if (
            specs == self.last_lora_specs
            and self.adapters
            and (
                adapters_are_dynamic
                or self.last_lora_merge_precision == self.lora_merge_precision
            )
        ):
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
            # fp8 merge 句柄无常驻 network 权重可热换，必须 detach（还原
            # 原始 fp8 权重）后重 merge
            and all(getattr(a, "supports_hot_reload", True) for a in self.adapters)
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
                self.last_lora_merge_precision = self.lora_merge_precision
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
                self.t5_tokenizer_backend, self.vae_tiling,
            )
            # family_id 在 unload() 里被清空，必须先存——漏传曾是隐性
            # TypeError（_load 的必需参数，P4-4 引入时本路径漏改）
            saved_family = self.family_id or "anima"
            self.unload()
            self._load(
                family_id=saved_family,
                transformer_path=saved_paths[0],
                vae_path=saved_paths[1],
                text_encoder_path=saved_paths[2],
                t5_tokenizer_path=saved_paths[3] or "",
                backend=saved_paths[4],
                precision=saved_paths[5],
                vae_precision=saved_paths[6] or saved_paths[5],
                text_encoder_backend=saved_paths[7] or "hf",
                t5_tokenizer_backend=saved_paths[8] or "slow",
                vae_tiling=saved_paths[9] or "auto",
            )
            _emit_evt("loaded")
        self.last_lora_specs = []
        self.last_lora_metas = []
        self.last_lora_merge_precision = None

        self.adapters = apply_loras(
            self.model, specs, self.device, self.lora_dtype,
            family_id=self.family_id or "anima",
            lora_merge_precision=self.lora_merge_precision,
        )
        self.last_lora_specs = specs
        self.last_lora_metas = current_metas
        self.last_lora_merge_precision = self.lora_merge_precision
        self.model.eval()
        return self.adapters

    def _move_runtime_to_device(self) -> None:
        if not self.device:
            return
        if self.block_swap is not None:
            # 一刀切 .to(device) 会把换出层的 CPU pinned 主副本一起搬上卡，
            # swap 白做且瞬时占用等于完整模型（小卡直接 OOM）——跳过被管理的张量
            from training.block_swap import move_module_excluding

            move_module_excluding(self.model, self.device, self.block_swap)
        else:
            _move_module_to_device(self.model, self.device)
        # anima 文本栈是 (qwen_model, qwen_tok, t5_tok) 三元组——采样内部的
        # decode offload 会把 TE 挪去 CPU，这里搬回。自管 device 的栈
        # （krea2 的 Krea2TextStack）没有裸模块成员，循环自然 no-op。
        if isinstance(self.text_stack, (tuple, list)):
            for member in self.text_stack:
                if isinstance(member, torch.nn.Module):
                    _move_module_to_device(member, self.device)
        for adapter in self.adapters:
            _move_adapter_to_device(adapter, self.device, self.lora_dtype)

    def _move_dit_and_adapters(self, device: str) -> None:
        """Move only the sampling model side; leave TE/VAE orchestration alone."""
        _move_module_to_device(self.model, device)
        for adapter in self.adapters:
            _move_adapter_to_device(adapter, device, self.lora_dtype)
        if torch.cuda.is_available():
            # ``Module.to(cuda)`` schedules thousands of FP8 parameter copies.
            # Establish an explicit hand-off boundary before the first large
            # attention allocation; this is especially important under WDDM.
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(torch.device(device))
            torch.cuda.empty_cache()

    def unload(self, *, keep_text: bool = False) -> None:
        """卸载模型栈。``keep_text``：保留 TE 先行栈（含预编码 LRU）——
        ensure_loaded 的重载路径用，避免刚编码完的结果被清掉。"""
        had_text = self.text_stack is not None
        if not keep_text:
            self.text_stack = None
            self._text_ready_key = None
        if not self.loaded:
            if had_text and not keep_text:
                try:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            return
        logger.info("unloading model")
        if self.block_swap is not None:
            # 摘钩子并丢弃管理对象——pinned 主副本（可达 11GB+）随之释放。
            # 页锁定内存不会被 GC 之外的任何机制回收，漏掉这步等于常驻泄漏。
            try:
                self.block_swap.detach()
            except Exception:  # noqa: BLE001
                pass
            self.block_swap = None
            self.blocks_to_swap = 0
        self.model = None
        self.vae = None
        self.family_id = None
        self.family = None
        self.adapters = []
        self.last_lora_specs = []
        self.last_lora_metas = []
        self.last_lora_merge_precision = None
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                try:
                    # cuBLAS workspace 是 C++ 级常驻分配（Python gc 不可见，
                    # 仅 ~10MB），但会把所在 allocator segment 整段钉住——
                    # 实测 fp8 采样后 8GB+ reserved 无法被 empty_cache 释放
                    # （tmp/diag_vram_leak.py 复现）。ComfyUI soft_empty_cache
                    # 同款处理；内部 API，失败可忽略（下轮加载会复用 cache）。
                    torch._C._cuda_clearCublasWorkspaces()
                except Exception:
                    pass
                torch.cuda.empty_cache()
        except Exception:
            pass


CACHE = ModelCache()


# ---------------------------------------------------------------------------
# Generate 实现（复用 anima_generate.py 的循环逻辑）
# ---------------------------------------------------------------------------


def _precache_prompts_and_release(
    prompts: list[str],
    vram_policy: str,
    phase_callback: Any = None,
) -> None:
    """krea2 任务级 prompt 预编码 + TE 彻底释放（训练两段式的推理版）。

    XY / 多 prompt generate 的 prompt 集合在任务开始前就封闭——先把全部
    prompt 编进在线 LRU，再**彻底释放** TE（release，非 offload：不留
    CPU 权重副本；conditioning 已在 LRU，下个任务 LRU miss 再从盘载）。
    与 ensure_text_ready 组成 TE 先行编排：首次任务在 DiT 加载前编码；后续
    cache miss 时，auto/save_vram 可先让 DiT 到 CPU，避免 TE/DiT 叠峰。

    - performance 档不释放（用户显式要求全同驻零搬运）。
    - anima 文本栈（tuple）无此 API，安全跳过。
    - 预编码失败不阻塞任务：逐格惰性编码路径兜底。
    """
    precache = getattr(CACHE.text_stack, "precache_online_prompts", None)
    if not callable(precache):
        return
    # TE 在此 lazy 加载（fp8 5GB / bf16 8.9GB 的 mmap 读盘）——按 TE 文件
    # 大小预算 RAM/VRAM。必须在兜底 try 之外：护栏错误要中止任务，不能被
    # 「退回逐格编码」吞掉后照样加载 TE 卡死。TE 已在卡上时零预算直通。
    from training.sysmem import check_load_budget

    te_paths = (
        [] if getattr(CACHE.text_stack, "is_model_loaded", False)
        else [CACHE.text_encoder_path]
    )
    check_load_budget(CACHE.ram_guard, weight_paths=te_paths, stage="文本编码器加载")
    captions = [str(p) for p in prompts]
    cached = getattr(CACHE.text_stack, "online_conditions_cached", None)
    needs_encoding = not (callable(cached) and cached(captions))
    te_resident = bool(getattr(CACHE.text_stack, "is_model_on_device", False))
    calibration = CACHE._te_peak_calibration.get(CACHE._text_ready_key)
    if calibration is not None:
        te_increment = calibration[1] if te_resident else calibration[0]
        estimate_source = "calibrated"
    else:
        if te_resident:
            te_increment = _TE_ENCODE_FALLBACK
        elif getattr(CACHE.text_stack, "is_fp8_storage", False):
            te_increment = _TE_LOAD_FALLBACK_FP8
        else:
            te_increment = _TE_LOAD_FALLBACK_FULL
        estimate_source = "fallback"

    yielded_dit = False
    if needs_encoding and CACHE.model is not None and CACHE.device:
        yielded_dit, budget = _should_yield_dit_for_te(
            vram_policy, CACHE.device, te_increment,
        )
        if budget is not None:
            logger.info(
                "TE/DiT auto 预算：free=%.1fGiB, TE=%s %.1fGiB, "
                "margin=%.1fGiB → %s",
                budget["free"] / _GIB,
                estimate_source,
                budget["te_increment"] / _GIB,
                budget["margin"] / _GIB,
                "DiT 让位" if yielded_dit else "允许同驻",
            )
        elif yielded_dit:
            logger.info("TE 预编码：%s 策略先将 DiT 移到 CPU", vram_policy)
        if yielded_dit:
            CACHE._move_dit_and_adapters("cpu")

    measure_cuda = bool(
        needs_encoding
        and CACHE.device
        and torch.device(CACHE.device).type == "cuda"
        and torch.cuda.is_available()
    )
    start_allocated = 0
    if measure_cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(CACHE.device)
        start_allocated = int(torch.cuda.memory_allocated(CACHE.device))
        torch.cuda.reset_peak_memory_stats(CACHE.device)

    encoded = 0
    success = False
    try:
        if phase_callback is not None:
            phase_callback("clip")
        encoded = precache(captions)
        success = True
        if measure_cuda:
            torch.cuda.synchronize(CACHE.device)
            peak_delta = max(
                0,
                int(torch.cuda.max_memory_allocated(CACHE.device)) - start_allocated,
            )
            resident_delta = max(
                0,
                int(torch.cuda.memory_allocated(CACHE.device)) - start_allocated,
            )
            encode_delta = max(0, peak_delta - resident_delta)
            previous = CACHE._te_peak_calibration.get(CACHE._text_ready_key, (0, 0))
            CACHE._te_peak_calibration[CACHE._text_ready_key] = (
                max(previous[0], peak_delta),
                max(previous[1], encode_delta),
            )
            logger.info(
                "TE 峰值校准：load+encode=%.1fGiB，resident encode=%.1fGiB",
                peak_delta / _GIB,
                encode_delta / _GIB,
            )
    except Exception:
        logger.exception("prompt 预编码失败；退回逐格惰性编码")
    finally:
        if vram_policy != "performance" and success:
            release = getattr(CACHE.text_stack, "release_model", None)
            if callable(release):
                release()
        elif yielded_dit and not success:
            # 避免失败后 TE 与即将恢复的 DiT 再次叠峰；保留 CPU 副本给
            # family.sample_image 的逐格兜底路径。
            offload = getattr(CACHE.text_stack, "offload_model", None)
            if callable(offload):
                offload()
        if yielded_dit:
            CACHE._move_dit_and_adapters(CACHE.device or "cuda")

    if not success:
        return
    if encoded:
        if vram_policy == "performance":
            logger.info("krea2 预编码 %d 条 prompt；performance 保持 TE 驻留", encoded)
        else:
            logger.info("krea2 预编码 %d 条 prompt；TE 已释放，采样期零占用", encoded)


def _begin_initial_te_peak_calibration(cfg: dict[str, Any]) -> int | None:
    """Start measuring the first Krea2 TE load before ``CACHE.device`` exists."""
    if (
        str(cfg.get("model_family") or "anima") != "krea2"
        or CACHE.model is not None
        or not torch.cuda.is_available()
    ):
        return None
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize("cuda")
        start_allocated = int(torch.cuda.memory_allocated("cuda"))
        torch.cuda.reset_peak_memory_stats("cuda")
        return start_allocated
    except Exception:
        logger.exception("failed to start initial TE peak calibration")
        return None


def _finish_initial_te_peak_calibration(start_allocated: int | None) -> None:
    """Persist first-load TE peak for later task-level residency decisions."""
    if start_allocated is None or CACHE._text_ready_key is None:
        return
    try:
        torch.cuda.synchronize("cuda")
        peak_delta = max(
            0,
            int(torch.cuda.max_memory_allocated("cuda")) - start_allocated,
        )
        previous = CACHE._te_peak_calibration.get(CACHE._text_ready_key, (0, 0))
        CACHE._te_peak_calibration[CACHE._text_ready_key] = (
            max(previous[0], peak_delta),
            previous[1],
        )
        logger.info("initial TE peak calibrated: load+encode=%.1fGiB", peak_delta / _GIB)
    except Exception:
        logger.exception("failed to finish initial TE peak calibration")


def _set_lora_multiplier(adapter: Any, scale: float) -> None:
    if adapter.network is None:
        # 含 fp8 merge 句柄（network=None）：scale 已烘进权重，逐格设值
        # 安全跳过（fp8 的 lora_scale 轴由 _cell_lora_configs →
        # CACHE.apply_loras 重 merge 生效）
        return
    adapter.network.multiplier = float(scale)
    for lora in getattr(adapter.network, "loras", []):
        if hasattr(lora, "multiplier"):
            lora.multiplier = float(scale)


def _swap_ckpt_for_axis(spec: dict[str, Any], val: Any,
                        lora_configs: list[dict[str, Any]]) -> None:
    """axis=lora_ckpt 时把 lora_configs[lora_index].path 改成 val。"""
    if spec.get("axis") != "lora_ckpt":
        return
    idx = int(spec.get("lora_index") or 0)
    if 0 <= idx < len(lora_configs):
        lora_configs[idx]["path"] = str(val)


def _cell_lora_configs(
    x_spec: dict[str, Any],
    y_spec: dict[str, Any] | None,
    xv: Any,
    yv: Any,
    base_paths: list[str],
    base_scales: list[float],
    *,
    fp8_scale_axes: bool,
) -> list[dict[str, Any]] | None:
    """组装本格需要重挂载的 lora_configs；无需重挂载时返回 None。

    - lora_ckpt 轴：按 lora_index 换单条 path（bf16/fp8 都走这里）。
    - lora_scale 轴仅在 fp8 底模（``fp8_scale_axes=True``）时在这里生效：
      merge 无常驻 network，改强度必须 detach 还原 + 重 merge——
      CACHE.apply_loras 对 supports_hot_reload=False 的句柄自动走该路径
      （lora_ckpt 轴同款），specs 相同的格子被去重零成本跳过。bf16 走
      _apply_axis 的 multiplier 热换，不进这里。
      全局轴语义：所有条目 scale=cell 值；x/y 都是 scale 轴时 y 后写赢，
      与 _apply_axis 的 x→y 调用顺序一致。
    """
    x_axis = x_spec.get("axis")
    y_axis = y_spec.get("axis") if y_spec is not None else None
    needs = x_axis == "lora_ckpt" or y_axis == "lora_ckpt" or (
        fp8_scale_axes and "lora_scale" in (x_axis, y_axis)
    )
    if not needs:
        return None
    configs = [
        {"path": p, "scale": s} for p, s in zip(base_paths, base_scales)
    ]
    _swap_ckpt_for_axis(x_spec, xv, configs)
    if y_spec is not None and yv is not None:
        _swap_ckpt_for_axis(y_spec, yv, configs)
    if fp8_scale_axes:
        if x_axis == "lora_scale":
            for lc in configs:
                lc["scale"] = float(xv)
        if y_axis == "lora_scale" and yv is not None:
            for lc in configs:
                lc["scale"] = float(yv)
    return configs


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


# latent→RGB 线性投影系数按**当前加载族的 spec** 取（D17 收编进 ModelSpec，
# 单一来源在 families/latent_spaces.py）。Anima 与 Krea2 同为 Wan2.1 16-ch
# latent 空间（Qwen-Image VAE：ComfyUI supported_models.py
# `QwenImage.latent_format = latent_formats.Wan21`）；未来不同 latent 空间的族
# 接入时预览自动跟随。之前误用 TAEFlux（Flux 解码器）→ 颜色反相，已废弃。
from training.families.latent_spaces import WAN21_F8C16 as _WAN21_F8C16  # noqa: E402

# 预览放大目标（最长边像素）。latent 是 1/8 分辨率（1024²→128²），latent2rgb 直出
# 128²；放大到 512 让前端铺满时不至于过糊，JPEG 仍很小。
_PREVIEW_TARGET_PX = 512


def _preview_latent_spec():
    """当前加载族的 LatentSpec；未加载（单测 / 启动早期）回退 Wan21 共享空间。"""
    family = getattr(CACHE, "family", None)
    return family.spec.latent if family is not None else _WAN21_F8C16


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
        latent_spec = _preview_latent_spec()
        with torch.no_grad():
            x = latent[0, :, 0].float()  # [16, H, W]
            factors = torch.tensor(
                [list(r) for r in latent_spec.rgb_factors],
                device=x.device, dtype=x.dtype,
            )  # [16, 3]
            bias = torch.tensor(
                list(latent_spec.rgb_bias), device=x.device, dtype=x.dtype,
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
    # 蒸馏推理底模（Krea2 Turbo）：studio 按 catalog variant purpose 检测后注入；
    # anima family 接受并忽略
    distilled: bool = bool(cfg.get("distilled", False))

    # phase 上报：加载模型/LoRA 阶段（clip/sample/vae 由 sample_image 内部报）→ 进度条覆盖全流程
    _emit_for(req_id, "phase", name="load")
    # TE 先行编排（krea2）：首次任务在 DiT 加载前编码并记录 TE 实际峰值；
    # 后续 cache miss 由显存策略决定 DiT 是否先让位。anima 族两步均 no-op。
    initial_te_measurement = _begin_initial_te_peak_calibration(cfg)
    CACHE.ensure_text_ready(cfg)
    _precache_prompts_and_release(
        [*prompts, negative_prompt],
        str(cfg.get("vram_policy") or "auto"),
        _phase_cb,
    )
    _finish_initial_te_peak_calibration(initial_te_measurement)
    CACHE.ensure_loaded(cfg)
    adapters = CACHE.apply_loras(cfg.get("lora_configs", []))

    # 进度推送：永远建 callback 推 preview_step（含 step/total）；
    # preview_every_n_steps>0 时附 image_b64 中间预览图（commit 14）。
    preview_every = int(cfg.get("preview_every_n_steps", 0) or 0)
    preview_callback = _build_preview_callback(req_id, preview_every, cancel_event)

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
                vram_policy = str(cfg.get("vram_policy") or "auto")
                try:
                    def _sample_once():
                        return CACHE.family.sample_image(
                            CACHE.model, CACHE.vae, CACHE.text_stack,
                            prompt,
                            height=height,
                            width=width,
                            steps=steps,
                            cfg_scale=cfg_scale,
                            negative_prompt=negative_prompt,
                            sampler_name=sampler_name,
                            scheduler=scheduler,
                            distilled=distilled,
                            device=CACHE.device,
                            dtype=CACHE.dtype,
                            step_callback=preview_callback,
                            phase_callback=_phase_cb,
                            seed=seed,
                            vram_policy=vram_policy,
                        )

                    img = _sample_with_cuda_oom_retry(
                        _sample_once,
                        seed=seed,
                        device=CACHE.device,
                        before_retry=lambda: release_vae_after_decode(
                            CACHE.vae, vram_policy,
                        ),
                    )
                finally:
                    release_vae_after_decode(CACHE.vae, vram_policy)
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
    distilled: bool = bool(cfg.get("distilled", False))

    # fp8 底模的 LoRA 是 merge 进权重的（无常驻 network），lora_scale 轴
    # 不能 multiplier 热换——逐格走 detach 还原 + 重 merge
    # （_cell_lora_configs 组装 → CACHE.apply_loras，lora_ckpt 轴同款）。
    # 每个不同 scale 值一次全模型重 merge：scale 放 Y 轴时每行只 merge
    # 一次（specs 去重），放 X 轴则每格一次。
    fp8_model = False
    if CACHE.model is not None:
        from training.families.krea2.quant_fp8 import model_has_fp8_layers

        fp8_model = model_has_fp8_layers(CACHE.model)

    if base_seed == 0:
        base_seed = random.randint(0, 2**31 - 1)
        logger.info("XY 共享种子（cfg.seed=0 随机化）: %d", base_seed)

    base_scales = [float(s.scale) for s in CACHE.last_lora_specs]
    base_lora_paths = [str(s.path) for s in CACHE.last_lora_specs]
    total = len(x_values) * len(y_values)
    _emit_for(req_id, "started", task_id=task_id, total=total)

    # XY 无 prompt 轴——prompt/negative 已由 _run_generate 的 TE 先行编排
    # 统一预编码并释放 TE，此处逐格全 LRU 命中，无需重复。

    img_idx = 0
    image_done_count = 0
    image_errors: list[str] = []
    for yi, yv in enumerate(y_values):
        for xi, xv in enumerate(x_values):
            _raise_if_canceled(cancel_event)
            # lora_ckpt 换文件 /（fp8 时）lora_scale 换强度：组装本格
            # lora_configs 调 CACHE.apply_loras —— detach 还原 + 重挂载
            # （bf16 reinject / fp8 重 merge）。base_paths/base_scales 是
            # 循环外快照，格间互不污染。
            lora_configs = _cell_lora_configs(
                x_spec, y_spec, xv, yv, base_lora_paths, base_scales,
                fp8_scale_axes=fp8_model,
            )
            if lora_configs is not None:
                adapters = CACHE.apply_loras(lora_configs)

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
                vram_policy = str(cfg.get("vram_policy") or "auto")
                try:
                    def _sample_once():
                        return CACHE.family.sample_image(
                            CACHE.model, CACHE.vae, CACHE.text_stack,
                            prompt,
                            height=height,
                            width=width,
                            steps=cur_steps,
                            step_callback=preview_callback,
                            phase_callback=phase_callback,
                            cfg_scale=cur_cfg_scale,
                            negative_prompt=negative_prompt,
                            sampler_name=base_sampler,
                            scheduler=scheduler,
                            distilled=distilled,
                            device=CACHE.device,
                            dtype=CACHE.dtype,
                            seed=cur_seed,
                            vram_policy=vram_policy,
                        )

                    img = _sample_with_cuda_oom_retry(
                        _sample_once,
                        seed=cur_seed,
                        device=CACHE.device,
                        before_retry=lambda: release_vae_after_decode(
                            CACHE.vae, vram_policy,
                        ),
                    )
                finally:
                    release_vae_after_decode(CACHE.vae, vram_policy)
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
