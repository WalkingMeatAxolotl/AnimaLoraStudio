"""LoRA ↔ 底模兼容契约：写（元数据）/ 读（元数据优先、键扫描兜底）/ 判（一个纯函数）。

背景：同族底模可以有不同层数（Anima 官方 28 层、第三方插层扩展版 40 层……）。
LoRA 按模块名 ``blocks.N.xxx`` 套用，28 层底模上训的 LoRA 挂到 40 层底模，
``blocks.0..27`` 全按名字对上、``blocks.28..39`` 缺 —— 不报错不警告，出的图是
错位的垃圾。族标记（``model_family``）分不出这个差别，所以再加一层「底模架构」
契约：

- **写**：训练保存 LoRA 时把底模的层数 / 通道 / 文件名写进 ``ss_network_args``
  （:func:`base_arch_network_args`）；三处写盘（lycoris / ortho / lora_merge）
  共用 :func:`build_lora_metadata` 拼顶层 metadata。
- **读**：:func:`lora_num_blocks_from_keys` 从 LoRA 键名数 ``blocks.N`` 最大下标
  +1，给没有元数据的存量 / 外部文件兜底（注意只是**下界**——只挂部分层的 LoRA
  也会小于底模层数）。
- **判**：:func:`check_lora_compat` 是唯一的判定函数，出图 apply / 训练 resume_lora
  / 恢复点 / 前端预检全部消费它的结论，不各自再算一遍：

  | 情形 | 结论 |
  |---|---|
  | 元数据有层数且 ≠ 底模层数 | reject（一定错，与「跨族拒绝」同级） |
  | 无元数据，键最大下标 ≥ 底模层数 | reject（键必然 unexpected，一定错） |
  | 无元数据，键最大下标 +1 < 底模层数 | warn（可能是老 28 层 LoRA，也可能只挂部分层） |
  | 其余 | ok |

与 ``core.py`` / ``checkpoint_arch.py`` 同款定位：runtime 与 studio 服务端共用，只用标准库。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

#: ``ss_network_args`` 里的底模架构键（写读两侧唯一出处）
KEY_BASE_NUM_BLOCKS = "base_num_blocks"
KEY_BASE_MODEL_CHANNELS = "base_model_channels"
KEY_BASE_MODEL_FILE = "base_model_file"

#: LoRA 键里的 block 下标：kohya 键把模块路径的 ``.`` 换成 ``_``
#: （``lora_unet_blocks_12_self_attn_q_proj.lora_down.weight``），PEFT / 外部键
#: 保留 ``.``（``diffusion_model.blocks.12.self_attn.q_proj.lora_A.weight``）。
_LORA_BLOCK_RE = re.compile(r"(?:^|[._])blocks[._](\d+)[._]")
_ADAPTER_MARK = "llm_adapter"

ArchSource = Literal["metadata", "keys", "unknown"]
CompatLevel = Literal["ok", "warn", "reject"]


# ── 写 ──────────────────────────────────────────────────────────────────────

def base_arch_network_args(
    *,
    num_blocks: Optional[int],
    model_channels: Optional[int] = None,
    base_model_file: Optional[str] = None,
) -> dict[str, Any]:
    """底模架构 → 要并进 ``ss_network_args`` 的键。None 的项不写。"""
    out: dict[str, Any] = {}
    if num_blocks is not None:
        out[KEY_BASE_NUM_BLOCKS] = int(num_blocks)
    if model_channels is not None:
        out[KEY_BASE_MODEL_CHANNELS] = int(model_channels)
    if base_model_file:
        out[KEY_BASE_MODEL_FILE] = str(base_model_file)
    return out


def base_arch_network_args_from_model(model: Any, transformer_path: Any = None) -> dict[str, Any]:
    """从已加载的 DiT 取架构事实（``len(model.blocks)`` / ``model_channels``）。

    loader 会在模型上挂 ``checkpoint_arch``（Anima）；没有的族（krea2）退回
    数 ``model.blocks``。``transformer_path`` 只取文件名做溯源展示。
    """
    num_blocks = model_num_blocks(model)
    model_channels: Optional[int] = None
    arch = getattr(model, "checkpoint_arch", None)
    if arch is not None:
        model_channels = _positive_int(getattr(arch, "model_channels", None))
    if model_channels is None:
        model_channels = _positive_int(getattr(model, "model_channels", None))
    base_file = None
    if transformer_path:
        base_file = str(transformer_path).replace("\\", "/").rsplit("/", 1)[-1]
    return base_arch_network_args(
        num_blocks=num_blocks, model_channels=model_channels, base_model_file=base_file,
    )


def build_lora_metadata(
    *,
    rank: Any,
    alpha: Any,
    network_args: dict[str, Any],
    extra: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """LoRA safetensors 顶层 metadata（kohya / ComfyUI 兼容形态）。

    三处写盘共用：``ss_network_dim/alpha/module`` + ``ss_network_args``（JSON）。
    ``extra`` 是额外的顶层键（如 lora_merge 的 provenance）。
    """
    meta = {
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(alpha),
        "ss_network_module": "lycoris.kohya",
        "ss_network_args": json.dumps(network_args),
    }
    if extra:
        meta.update({str(k): str(v) for k, v in extra.items()})
    return meta


# ── 读 ──────────────────────────────────────────────────────────────────────

def lora_num_blocks_from_keys(keys: Iterable[str]) -> Optional[int]:
    """LoRA 键名里 ``blocks.N`` 最大下标 +1；没有 block 键返回 None。

    只是下界：LoRA 若只挂了部分层（自定义 target / reg_dims 之类），数出来会
    小于真实底模层数。llm_adapter 的 blocks 排除。
    """
    top = -1
    for k in keys:
        if _ADAPTER_MARK in k:
            continue
        m = _LORA_BLOCK_RE.search(k)
        if m:
            top = max(top, int(m.group(1)))
    return top + 1 if top >= 0 else None


@dataclass(frozen=True)
class LoraBaseArch:
    """一份 LoRA 声明 / 推断出的底模架构。"""

    num_blocks: Optional[int]
    source: ArchSource
    model_channels: Optional[int] = None
    base_model_file: Optional[str] = None

    @property
    def explicit(self) -> bool:
        return self.source == "metadata"


def read_lora_base_arch(path: Any) -> LoraBaseArch:
    """只读 header 取 LoRA 的底模架构（studio 列表用：不 import torch、毫秒级）。

    读失败 / 不是 safetensors → unknown（列表不因单个坏文件失败）。
    """
    from .checkpoint_arch import read_safetensors_header

    try:
        header = read_safetensors_header(path)
    except Exception:  # noqa: BLE001
        return LoraBaseArch(num_blocks=None, source="unknown")
    meta = header.get("__metadata__") or {}
    try:
        ss_args = json.loads(meta.get("ss_network_args") or "{}") if isinstance(meta, dict) else {}
        if not isinstance(ss_args, dict):
            ss_args = {}
    except (TypeError, ValueError):
        ss_args = {}
    return lora_base_arch(ss_args, (k for k in header if k != "__metadata__"))


def lora_base_arch(network_args: dict[str, Any], keys: Iterable[str] = ()) -> LoraBaseArch:
    """元数据优先（``ss_network_args``），否则扫键兜底。"""
    raw = network_args.get(KEY_BASE_NUM_BLOCKS)
    if raw is not None:
        try:
            nb = int(raw)
        except (TypeError, ValueError):
            nb = None
        if nb is not None and nb > 0:
            mc_raw = network_args.get(KEY_BASE_MODEL_CHANNELS)
            try:
                mc = int(mc_raw) if mc_raw is not None else None
            except (TypeError, ValueError):
                mc = None
            bf = network_args.get(KEY_BASE_MODEL_FILE)
            return LoraBaseArch(
                num_blocks=nb, source="metadata", model_channels=mc,
                base_model_file=str(bf) if bf else None,
            )
    nb = lora_num_blocks_from_keys(keys)
    if nb is not None:
        return LoraBaseArch(num_blocks=nb, source="keys")
    return LoraBaseArch(num_blocks=None, source="unknown")


# ── 判 ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CompatVerdict:
    level: CompatLevel
    #: 事实句（用户可读，中文），ok 时为空串
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.level == "ok"


def check_lora_compat(
    lora: LoraBaseArch,
    model_num_blocks: Optional[int],
    *,
    lora_name: str = "LoRA",
) -> CompatVerdict:
    """LoRA 声明/推断的底模层数 vs 当前底模层数（见模块 docstring 的规则表）。

    ``model_num_blocks`` 为 None（底模层数未知）时不判，返回 ok。
    """
    if model_num_blocks is None or lora.num_blocks is None:
        return CompatVerdict("ok")
    if lora.source == "metadata":
        if lora.num_blocks != model_num_blocks:
            return CompatVerdict(
                "reject",
                f"{lora_name} 训练自 {lora.num_blocks} 层底模，当前底模 {model_num_blocks} 层，"
                f"层与层对不上，不能挂载",
            )
        return CompatVerdict("ok")
    # 无元数据：键扫描只是下界
    if lora.num_blocks > model_num_blocks:
        return CompatVerdict(
            "reject",
            f"{lora_name} 含 blocks.{lora.num_blocks - 1} 的权重，当前底模只有 {model_num_blocks} 层，"
            f"层与层对不上，不能挂载",
        )
    if lora.num_blocks < model_num_blocks:
        return CompatVerdict(
            "warn",
            f"{lora_name} 只覆盖前 {lora.num_blocks} 层（无底模层数元数据），当前底模 {model_num_blocks} 层："
            f"若它训练自层数更少的底模，挂上去层与层对不上",
        )
    return CompatVerdict("ok")


def _positive_int(v: Any) -> Optional[int]:
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None


def model_num_blocks(model: Any) -> Optional[int]:
    """当前已加载底模的 DiT 层数（Anima loader 挂的 ``checkpoint_arch`` 优先，否则数 ``blocks``）。

    只认真正的正整数（替身 / mock 对象一律 None → 不判）。
    """
    arch = getattr(model, "checkpoint_arch", None)
    nb = _positive_int(getattr(arch, "num_blocks", None)) if arch is not None else None
    if nb is not None:
        return nb
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return None
    try:
        return _positive_int(len(blocks))
    except TypeError:
        return None
