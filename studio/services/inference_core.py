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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# 测试出图 task 的临时输出目录前缀。每个 task 一个 anima_gen_{task_id}/。
# 用户决策：测试页面出图不保存，task 结束 supervisor 清掉整个目录；
# studio 启动时扫一遍清遗留（防 supervisor crash 时 leak）。
GENERATE_TEMP_PREFIX = "anima_gen_"


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
    """从 safetensors metadata 解析出来的 LoRA 训练参数。"""
    rank: int
    alpha: float
    algo: str
    factor: int


def read_lora_meta(path: str) -> LoRAMeta:
    """从 safetensors 顶层 metadata 读 LoRA 训练参数。

    AnimaLycorisAdapter.save() 写入约定（utils/lycoris_adapter.py:178）：
      - 顶层 metadata: ss_network_dim (rank), ss_network_alpha (alpha)
      - ss_network_args JSON 内: algo, factor, dropout, ...

    缺字段或解析失败时回退到默认值（rank=32, alpha=rank, algo=lokr, factor=8）。
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

    return LoRAMeta(rank=rank, alpha=alpha, algo=algo, factor=factor)


def apply_loras(
    model: Any,
    specs: Sequence[LoRASpec],
    device: str,
    dtype: Any,
) -> list[Any]:
    """对每个 LoRA 单独 inject 一份 AnimaLycorisAdapter；forward 时 hook 累加 delta。

    multiplier 字段控制每份 LoRA 贡献权重（用户传的 scale）：
      - LycorisNetwork.multiplier 是 forward 内取的全局倍率
      - per-lora module 也设一份兜底（lycoris 不同版本取值路径有差异）

    返回 adapter 列表 — caller **必须保持引用**，否则 Python GC 触发后
    AnimaLycorisAdapter 内的 LycorisNetwork 也会被 GC，model 上的 forward
    hook 跟着失效（lycoris 通过 closure 持有 network）。
    """
    from safetensors import safe_open

    from utils.lycoris_adapter import AnimaLycorisAdapter

    adapters: list[Any] = []
    for spec in specs:
        path = spec.path or ""
        if not path or not Path(path).exists():
            logger.warning(f"LoRA 路径不存在，跳过: {path!r}")
            continue

        meta = read_lora_meta(path)
        adapter = AnimaLycorisAdapter(
            algo=meta.algo,
            rank=meta.rank,
            alpha=meta.alpha,
            factor=meta.factor,
        )
        adapter.inject(model)

        if adapter.network is not None:
            adapter.network.multiplier = float(spec.scale)
            for lora in getattr(adapter.network, "loras", []):
                if hasattr(lora, "multiplier"):
                    lora.multiplier = float(spec.scale)

        sd: dict = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k).to(device=device, dtype=dtype)

        result = adapter.load_state_dict(sd, strict=False)
        missing = len(getattr(result, "missing_keys", []) or [])
        unexpected = len(getattr(result, "unexpected_keys", []) or [])
        logger.info(
            f"已加载 LoRA: {Path(path).name} "
            f"(algo={meta.algo}, rank={meta.rank}, alpha={meta.alpha}, "
            f"scale={spec.scale}; missing={missing}, unexpected={unexpected})"
        )
        adapters.append(adapter)

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
