"""推理核心 — 多 LoRA 加载 / 合并的统一实现。

服务对象：
  - runtime/anima_generate.py（独立测试出图，多 LoRA 叠加）
  - runtime/anima_train.py 训练期 sample（PR-9 commit 7 切过来）
  - runtime/anima_reg_ai.py（先验生成 — 不调 apply_loras，base 模型直出）

PR #17 作者在 anima_generate.py / anima_reg_ai.py 各 copy 了一份 LoRA 加载，
有两个 P0 bug：
  1. rank/alpha 硬编码 32/32，不从顶层 ss_network_dim/ss_network_alpha 读 ——
     训练 dim≠32 的 LoRA 会 shape 错或 alpha 缩放错。
  2. 多 LoRA 把不同 LoRA 的 tensor 直接 add 到一份 state_dict 然后灌进
     一个 LycorisNetwork —— LoKr 的 lokr_w1/lokr_w2 是子矩阵，
     子矩阵相加 ≠ 权重 delta 相加，出图错。

本模块统一修这两条：
  - read_lora_meta()：从顶层 metadata 读 rank/alpha，从 ss_network_args 读 algo/factor
  - apply_loras()：每份 LoRA 单独 inject 一份 AnimaLycorisAdapter，靠
    LycorisNetwork.multiplier=scale 控制贡献权重；forward 时多份 hook
    自然累加 delta，等价于权重 delta 加和。
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# 测试出图 task 的临时输出目录前缀。每个 task 一个 anima_gen_{task_id}/。
# 用户决策：测试页面出图不保存，task 结束 supervisor 清掉整个目录；
# studio 启动时扫一遍清遗留（防 supervisor crash 时 leak）。
GENERATE_TEMP_PREFIX = "anima_gen_"


class DeferredVAE:
    """Load a test-generation VAE only when sampling reaches decode.

    The family sampling functions intentionally keep their existing API: they
    receive this proxy in place of the real VAE.  No VAE attribute is touched
    during diffusion sampling, so the first ``decode``-side lookup is the exact
    lazy-load boundary.  Non-performance policies park the loaded wrapper on
    CPU after every image, keeping it in RAM while releasing its VRAM.
    """

    def __init__(self, loader, *, device: str, label: str = "VAE") -> None:
        self._loader = loader
        self._device = str(device)
        self._label = label
        self._value: Any = None
        self._resident_device: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._value is not None

    @property
    def resident_device(self) -> Optional[str]:
        return self._resident_device

    @staticmethod
    def _move(value: Any, device: str) -> None:
        mover = getattr(value, "to", None)
        if callable(mover):
            mover(device)
            return
        model = getattr(value, "model", None)
        model_mover = getattr(model, "to", None)
        if callable(model_mover):
            model_mover(device)
            return
        raise TypeError(f"{type(value).__name__} 不支持在 CPU/GPU 间移动")

    def _ready(self) -> Any:
        if self._value is None:
            logger.info("采样完成，开始加载 %s", self._label)
            self._value = self._loader()
            self._resident_device = self._device
        elif self._resident_device != self._device:
            logger.info("将 %s 从 CPU 移回 %s 用于 decode", self._label, self._device)
            self._move(self._value, self._device)
            self._resident_device = self._device
        return self._value

    def release_after_decode(self, vram_policy: str) -> None:
        """Park the VAE in RAM unless the user selected performance mode."""
        if self._value is None or str(vram_policy or "auto") == "performance":
            return
        if self._resident_device == "cpu":
            return
        self._move(self._value, "cpu")
        self._resident_device = "cpu"
        logger.info("%s decode 完成，已按 %s 策略移到 CPU RAM", self._label, vram_policy)
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.warning("VAE decode 后释放 CUDA cache 失败", exc_info=True)

    def __getattr__(self, name: str) -> Any:
        # Only called after normal proxy attributes fail, so the loader itself
        # is never triggered by introspection of ``is_loaded``/policy state.
        return getattr(self._ready(), name)


def release_vae_after_decode(vae: Any, vram_policy: str) -> None:
    """Release a deferred VAE when present; remain compatible with test fakes."""
    release = getattr(vae, "release_after_decode", None)
    if callable(release):
        release(vram_policy)


# 缺 metadata 时的回退值。与 AnimaLycorisAdapter 默认对齐。
_DEFAULT_RANK = 32
_DEFAULT_ALPHA = 16.0
_DEFAULT_ALGO = "lokr"
_DEFAULT_FACTOR = 8


@dataclass
class LoRASpec:
    """单个 LoRA 的加载参数。"""
    path: str
    scale: float = 1.0


@dataclass
class LoRAMeta:
    """从 safetensors metadata 解析出来的 LoRA 训练参数。

    weight_decompose / rs_lora 必须忠实回放训练侧设置 —— 否则推理网络结构
    与文件不匹配：DoRA 漏 dora_scale 张量（unexpected keys），RS-LoRA 把
    effective alpha 从 α/√rank 错算成 α/rank，强度被砍 √rank 倍。

    lora_reg_dims（正则 → rank）同理：训练时按 pattern 把部分层的 rank 覆盖
    成自定义值；推理重建网络若用全局 rank 实例化，被覆盖层的 lokr_w2_a/b
    形状跟 checkpoint 立刻 mismatch。
    """
    rank: int
    alpha: float
    algo: str
    factor: int
    weight_decompose: bool = False
    rs_lora: bool = False
    lora_reg_dims: Optional[dict[str, int]] = None
    #: 产物所属模型族（D13 标记）；无标记的存量产物 grandfather 为 anima
    model_family: str = "anima"
    #: model_family 是否来自显式 metadata 标记。外部生态文件（civitai /
    #: musubi / comfy 系）不带我们的标记——grandfather 值只用于展示，
    #: 跨族硬拒绝只对显式标记生效，无标记靠注入/merge 的键匹配兜底。
    family_explicit: bool = False


def read_lora_meta(path: str) -> LoRAMeta:
    """从 safetensors 顶层 metadata 读 LoRA 训练参数。

    AnimaLycorisAdapter.save() 写入约定（utils/lycoris_adapter.py）：
      - 顶层 metadata: ss_network_dim (rank), ss_network_alpha (alpha)
      - ss_network_args JSON 内: algo, factor, weight_decompose, rs_lora, ...

    缺字段或解析失败时回退到默认值（rank=32, alpha=rank, algo=lokr, factor=8,
    weight_decompose=False, rs_lora=False）。
    """
    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
    except Exception as e:
        logger.warning(f"读 LoRA metadata 失败 {path}: {e}; 用默认参数")
        return LoRAMeta(_DEFAULT_RANK, _DEFAULT_ALPHA, _DEFAULT_ALGO, _DEFAULT_FACTOR)

    try:
        ss_args = json.loads(meta.get("ss_network_args", "{}"))
        if not isinstance(ss_args, dict):
            ss_args = {}
    except (ValueError, TypeError):
        ss_args = {}

    rank = _DEFAULT_RANK
    if "ss_network_dim" in meta:
        try:
            rank = int(meta["ss_network_dim"])
        except (ValueError, TypeError):
            pass

    # 没显式 alpha 时常见约定是 alpha=rank（保留 1.0 倍率）
    alpha = float(rank)
    if "ss_network_alpha" in meta:
        try:
            alpha = float(meta["ss_network_alpha"])
        except (ValueError, TypeError):
            pass

    algo = str(ss_args.get("algo", _DEFAULT_ALGO))
    factor = _DEFAULT_FACTOR
    if "factor" in ss_args:
        try:
            factor = int(ss_args["factor"])
        except (ValueError, TypeError):
            pass

    weight_decompose = bool(ss_args.get("weight_decompose", False))
    rs_lora = bool(ss_args.get("rs_lora", False))

    # lora_reg_dims：训练时按正则覆盖部分层 rank；推理必须按同样 pattern 重建，
    # 否则被覆盖层 shape 与 checkpoint 不匹配。校验是 dict[str, int]，其他形态丢弃。
    lora_reg_dims: Optional[dict[str, int]] = None
    raw_reg = ss_args.get("lora_reg_dims")
    if isinstance(raw_reg, dict) and raw_reg:
        parsed: dict[str, int] = {}
        for k, v in raw_reg.items():
            try:
                parsed[str(k)] = int(v)
            except (ValueError, TypeError):
                logger.warning(f"lora_reg_dims 条目跳过（rank 非整数）: {k!r}={v!r}")
        if parsed:
            lora_reg_dims = parsed

    return LoRAMeta(
        rank=rank,
        alpha=alpha,
        algo=algo,
        factor=factor,
        weight_decompose=weight_decompose,
        rs_lora=rs_lora,
        lora_reg_dims=lora_reg_dims,
        model_family=str(ss_args.get("model_family") or "anima"),
        family_explicit=bool(ss_args.get("model_family")),
    )


_PEFT_LORA_PREFIX = "diffusion_model."
_PEFT_SUFFIX_MAP = {
    "lora_A.weight": "lora_down.weight",
    "lora_B.weight": "lora_up.weight",
    "alpha": "alpha",
}


def _normalize_peft_lora_sd(
    sd: dict,
) -> Optional[tuple[dict, int, Optional[dict[str, int]]]]:
    """PEFT/comfy 键格式（civitai 生态）归一到 kohya/lycoris 约定。

    ``diffusion_model.{点分层名}.lora_A/lora_B`` → ``lora_unet_{下划线层名}.
    lora_down/lora_up.weight``；无 alpha 键 = comfy 缩放 1.0 语义 → 补
    per-layer ``alpha = rank``（alpha/rank 式 loader 得 1.0，数值一致）；
    rank 从 lora_A 张量形状推断（此类文件无 ss_* metadata，header 读不到），
    混秩层进 lora_reg_dims。返回 (归一 sd, max_rank, reg_dims)；非纯 PEFT
    形态返回 None（kohya/lycoris 文件原样走）。
    """
    import torch  # noqa: PLC0415

    if not sd or not all(key.startswith(_PEFT_LORA_PREFIX) for key in sd):
        return None
    grouped: dict[str, dict[str, Any]] = {}
    for key, tensor in sd.items():
        rest = key[len(_PEFT_LORA_PREFIX):]
        for peft_suffix, kohya_suffix in _PEFT_SUFFIX_MAP.items():
            if rest.endswith("." + peft_suffix):
                layer = rest[: -len(peft_suffix) - 1]
                grouped.setdefault(layer, {})[kohya_suffix] = tensor
                break
        else:
            if rest.endswith(".dora_scale"):
                raise ValueError(
                    "PEFT 形态的 DoRA LoRA 不支持（dora_scale 非线性）；"
                    "请改用 kohya 格式导出的版本。"
                )
            raise ValueError(f"无法识别的 PEFT LoRA 键：{key}")

    normalized: dict[str, Any] = {}
    ranks: dict[str, int] = {}
    for layer, tensors in grouped.items():
        down = tensors.get("lora_down.weight")
        up = tensors.get("lora_up.weight")
        if down is None or up is None:
            raise ValueError(f"PEFT LoRA 层缺 lora_A/lora_B 对：{layer}")
        rank = int(down.shape[0])
        ranks[layer] = rank
        prefix = f"lora_unet_{layer.replace('.', '_')}"
        normalized[f"{prefix}.lora_down.weight"] = down
        normalized[f"{prefix}.lora_up.weight"] = up
        alpha = tensors.get("alpha")
        normalized[f"{prefix}.alpha"] = (
            alpha if alpha is not None else torch.tensor(float(rank))
        )

    max_rank = max(ranks.values())
    reg_dims = {
        f"lora_unet_{layer.replace('.', '_')}": rank
        for layer, rank in ranks.items()
        if rank != max_rank
    } or None
    return normalized, max_rank, reg_dims


def apply_loras(
    model: Any,
    specs: Sequence[LoRASpec],
    device: str,
    dtype: Any,
    family_id: str = "anima",
    lora_merge_precision: str = "fp32",
    lora_merge_chunk_rows: int = 1024,
    keep_merge_backup: bool = True,
) -> list[Any]:
    """对每个 LoRA 单独 inject 一份 AnimaLycorisAdapter；forward 时 hook 累加 delta。

    dtype 是动态 LoRA network / tensor 的计算 dtype。FP8 底模不用 hook，改走
    权重 merge；其 delta 临时精度由 lora_merge_precision 控制（fp32/bf16）。
    普通 Linear LoRA 默认按 1024 输出行分块，避免物化整层 dense delta；
    LoHa/LoKr 暂时保留原算法。

    multiplier 字段控制每份 LoRA 贡献权重（用户传的 scale）：
      - LycorisNetwork.multiplier 是 forward 内取的全局倍率
      - per-lora module 也设一份兜底（lycoris 不同版本取值路径有差异）

    返回 adapter 列表 — caller **必须保持引用**，否则 Python GC 触发后
    AnimaLycorisAdapter 内的 LycorisNetwork 也会被 GC，model 上的 forward
    hook 跟着失效（lycoris 通过 closure 持有 network）。
    """
    from safetensors import safe_open

    from utils.lycoris_adapter import AnimaLycorisAdapter

    # fp8 量化底模走 ComfyUI merge 语义（dequant → 加 delta → stochastic
    # rounding 回写，seed=层名 CRC32）——lycoris hook 直接注入 fp8 权重会因
    # dtype 崩或产生与 Comfy 不一致的数值。目前只有 krea2 loader 会产出
    # fp8 权重（Anima loader 拒绝 fp8）。
    fp8_merge = False
    if specs:
        from training.families.krea2.quant_fp8 import model_has_fp8_layers  # noqa: PLC0415

        fp8_merge = model_has_fp8_layers(model)

    merge_sources: list[tuple[dict, float, str]] = []
    adapters: list[Any] = []
    for spec in specs:
        path = spec.path or ""
        if not path or not Path(path).exists():
            logger.warning(f"LoRA 路径不存在，跳过: {path!r}")
            continue

        meta = read_lora_meta(path)
        # 跨族 fail-fast（A5，与训练侧 resume_lora 检查同款）：krea2 LoRA 配
        # anima 底模（或反之）用错 preset 注入 = 键全 miss 的静默坏结果。
        # 只对**显式标记**硬拒——外部生态文件（civitai/musubi/comfy 系）没有
        # 我们的 model_family 标记，grandfather 值不可作拒绝依据；无标记文件
        # 放行，由下方注入/merge 的键匹配兜底（全 miss 报错，不静默）。
        if meta.family_explicit and meta.model_family != family_id:
            raise ValueError(
                f"LoRA 跨模型族被拒绝：{Path(path).name} 属于 "
                f"'{meta.model_family}'，当前底模族为 '{family_id}'。"
                f"请换用同族 LoRA 或切换底模。"
            )
        sd_raw: dict = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd_raw[k] = f.get_tensor(k)

        # 外部生态 PEFT 键格式归一（civitai 常见）：转 kohya 键 + 从张量
        # 形状推断 rank/reg_dims（这类文件零 ss_* metadata，meta 里的
        # rank=32 只是回退默认，碰运气不可用）
        rank, alpha, algo = meta.rank, meta.alpha, meta.algo
        reg_dims = meta.lora_reg_dims
        peft = _normalize_peft_lora_sd(sd_raw)
        if peft is not None:
            sd_raw, rank, reg_dims = peft
            alpha = float(rank)   # per-layer alpha 已补进 sd，全局值仅建网用
            algo = "lora"         # PEFT 双矩阵 = plain LoRA

        if fp8_merge:
            # merge 是权重级线性操作，只认 per-layer alpha/dim 缩放。
            # rs_lora 产物无需特判：lycoris 保存时把 √rank 校正烘进
            # per-layer alpha 键（register_buffer("alpha", α·dim/√dim)，
            # locon/loha/lokr 同款），标准 alpha/dim merge 得到的正是
            # α/√rank——与 bf16 注入路径（建网 scale=α/√r）数值一致。
            # DoRA（列范数归一化，非线性）merge 需要 comfy
            # weight_decompose 语义，尚未实现，保持拒绝。
            if meta.weight_decompose:
                raise ValueError(
                    f"fp8 量化底模不支持挂载 DoRA（weight_decompose）训练的 "
                    f"LoRA：{Path(path).name}。请改用 bf16 版本底模。"
                )
            merge_sources.append((sd_raw, float(spec.scale), Path(path).name))
            continue
        from training.families import get_family  # noqa: PLC0415

        adapter = AnimaLycorisAdapter(
            preset=get_family(family_id).lora_preset(),
            algo=algo,
            rank=rank,
            alpha=alpha,
            factor=meta.factor,
            weight_decompose=meta.weight_decompose,
            rs_lora=meta.rs_lora,
            lora_reg_dims=reg_dims,
        )
        adapter.inject(model)
        if adapter.network is not None:
            adapter.network.to(device=device, dtype=dtype)

        if adapter.network is not None:
            adapter.network.multiplier = float(spec.scale)
            for lora in getattr(adapter.network, "loras", []):
                if hasattr(lora, "multiplier"):
                    lora.multiplier = float(spec.scale)

        sd = {k: v.to(device=device, dtype=dtype) for k, v in sd_raw.items()}

        result = adapter.load_state_dict(sd, strict=False)
        missing = len(getattr(result, "missing_keys", []) or [])
        unexpected = len(getattr(result, "unexpected_keys", []) or [])
        if sd and unexpected >= len(sd):
            # 键全部没被 LoRA 网络吃掉 = 异族文件或本路径不支持的键格式
            # （无标记文件放行后的内容匹配兜底，防静默出无 LoRA 效果的图）
            raise ValueError(
                f"LoRA 与当前底模不匹配：{Path(path).name} 的键全部无法对应"
                f"（可能属于其他模型族，或是本路径尚不支持的键格式）。"
            )
        logger.info(
            f"已加载 LoRA: {Path(path).name} "
            f"(algo={algo}, rank={rank}, alpha={alpha}, "
            f"scale={spec.scale}; missing={missing}, unexpected={unexpected})"
        )
        adapters.append(adapter)

    if merge_sources:
        import torch

        from training.families.krea2.lora_fp8_merge import (  # noqa: PLC0415
            merge_loras_into_fp8_model,
        )

        precision = str(lora_merge_precision or "fp32").lower()
        if precision == "fp32":
            merge_dtype = torch.float32
        elif precision == "bf16":
            merge_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"LoRA merge 精度仅支持 fp32/bf16，收到：{lora_merge_precision!r}"
            )

        # 单个句柄对应全部 LoRA（merge 一次完成）；daemon 换 LoRA / 变
        # scale 时 detach() 从备份还原后重 merge
        merged = merge_loras_into_fp8_model(
            model,
            merge_sources,
            compute_dtype=merge_dtype,
            chunk_rows=lora_merge_chunk_rows,
            keep_backup=keep_merge_backup,
        )

        # The merge dequantizes fp8 weights to temporary fp16 tensors one layer
        # at a time.  Trim only after the merge frame (and its last w16/qdata
        # references) is gone, otherwise the allocator can retain several GiB
        # until the first decode.  This shared boundary covers every caller.
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.warning("failed to trim CUDA cache after fp8 LoRA merge", exc_info=True)
        return [merged]
    return adapters


# ---------------------------------------------------------------------------
# Generate 测试出图：临时目录管理
# ---------------------------------------------------------------------------


def generate_tempdir(task_id: int) -> Path:
    """单个 generate task 的临时输出目录路径。

    位于系统 tempdir 下（与 studio_data 隔离），task 完成清掉。
    """
    return Path(tempfile.gettempdir()) / f"{GENERATE_TEMP_PREFIX}{task_id}"


def cleanup_generate_tempdir(task_id: int) -> None:
    """task 结束时清单个 tempdir。目录不存在视为 noop（非 generate task 也安全调）。"""
    d = generate_tempdir(task_id)
    if not d.exists():
        return
    try:
        shutil.rmtree(d)
        logger.info(f"cleaned generate tempdir: {d}")
    except OSError as e:
        logger.warning(f"failed to clean {d}: {e}")


def cleanup_stale_generate_tempdirs() -> None:
    """启动时扫清所有 anima_gen_* 遗留目录（防 supervisor crash 泄漏）。"""
    parent = Path(tempfile.gettempdir())
    if not parent.exists():
        return
    for d in parent.glob(f"{GENERATE_TEMP_PREFIX}*"):
        if not d.is_dir():
            continue
        try:
            shutil.rmtree(d)
            logger.info(f"cleaned stale generate tempdir: {d}")
        except OSError as e:
            logger.warning(f"failed to clean stale {d}: {e}")
