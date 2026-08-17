"""Anima 族底模 checkpoint 架构探测 —— 只读 safetensors header，不碰 payload。

Anima 的层数由 checkpoint 决定（官方 2B=28 层、14B=36 层、第三方插层扩展版
如 Anima-2.9B=40 层……），代码里**不允许再出现任何写死的层数**：loader 建模型、
block swap 算换出比例、LoRA 元数据记底模架构、studio 目录行展示层数，全部
从这里的 :class:`AnimaCheckpointArch` 取值（单一真相）。

放在 studio/services/inference/ 而不是 runtime/：与 ``core.py``（LoRA 元数据 /
apply）同款定位 —— 被 runtime 训练/出图进程与 studio 服务端两边共用，依赖方向
runtime → studio 已是既有约定；且 studio 服务端列目录时不能为此 import torch
（``runtime.training.families`` 包顶层会拉起 torch），本模块只用标准库。

header 解析自己做（8 字节小端长度 + JSON），不走 ``safetensors.safe_open``：
后者 ``framework="pt"`` 会 import torch。
"""
from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: model_channels → 注意力头数。这是本模块唯一允许的「查表」——头数不出现在
#: 权重形状里（q_proj 是 [C, C]），只能靠通道数对应；层数一律数出来。
HEADS_BY_MODEL_CHANNELS: dict[int, int] = {2048: 16, 5120: 40}

#: DiT block 键（键可能带 net. / model. / module. 等前缀，按子串匹配）。
#: llm_adapter 自己也有 ``blocks.N``（6 层小 transformer），必须排除，否则会
#: 把 adapter 参数算进 DiT block 0-5、层数推断也可能被污染。
_BLOCK_KEY_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
_ADAPTER_MARK = "llm_adapter."
_X_EMBEDDER_SUFFIX = "x_embedder.proj.1.weight"

#: safetensors header 长度上限：真实 checkpoint 的 header 在几十 KB ~ 几百 KB，
#: 超过这个数基本是读到了别的文件 / 损坏文件，直接拒绝而不是分配巨量内存。
_MAX_HEADER_BYTES = 256 * 1024 * 1024


class CheckpointInspectError(ValueError):
    """checkpoint 不是（可识别的）Anima 族 transformer 权重。"""


def read_safetensors_header(path: str | os.PathLike[str]) -> dict[str, Any]:
    """读 safetensors 文件头（含 ``__metadata__``），不读任何张量数据。

    格式：前 8 字节 = 小端 uint64 header 长度 N，随后 N 字节 UTF-8 JSON。
    """
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise CheckpointInspectError(f"不是 safetensors 文件（不足 8 字节）: {path}")
        (n,) = struct.unpack("<Q", head)
        if n <= 0 or n > _MAX_HEADER_BYTES:
            raise CheckpointInspectError(f"safetensors header 长度异常（{n}）: {path}")
        raw = f.read(n)
    if len(raw) != n:
        raise CheckpointInspectError(f"safetensors header 被截断: {path}")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CheckpointInspectError(f"safetensors header 不是合法 JSON: {path}") from exc
    if not isinstance(header, dict):
        raise CheckpointInspectError(f"safetensors header 结构异常: {path}")
    return header


def _numel(shape: Any) -> int:
    n = 1
    for d in shape or ():
        n *= int(d)
    return n


@dataclass(frozen=True)
class AnimaCheckpointArch:
    """从 header 推断出的 Anima 底模架构。

    只含「决定怎么建模型 / LoRA 能否互换 / 资源量级」的事实；不含任何人类命名
    （2B / 2.9B 之类只在下载目录的 label 里）。
    """

    #: ``x_embedder.proj.1.weight.shape[0]``
    model_channels: int
    #: ``x_embedder.proj.1.weight.shape[1] // 4 - 1``（concat_padding_mask=True）
    in_channels: int
    #: DiT block 数 = ``blocks.N`` 最大下标 + 1（不含 llm_adapter.blocks）
    num_blocks: int
    #: 按 :data:`HEADS_BY_MODEL_CHANNELS` 查表；未知通道数为 None（由建模方决定报错）
    num_heads: int | None
    has_llm_adapter: bool
    #: header 数出的全模型参数量（dtype 无关）
    param_count: int
    #: 每个 DiT block 的参数量，下标 = block 序号
    block_param_counts: tuple[int, ...]
    file_bytes: int

    @property
    def dit_param_count(self) -> int:
        return sum(self.block_param_counts)

    def swapped_param_ratio(self, blocks_to_swap: int) -> float:
        """换出末尾 ``blocks_to_swap`` 层占全模型参数的比例（超界按全部换出）。

        显存折扣必须按比例乘文件实际大小（dtype 无关），见 krea2 loader 同名函数说明。
        """
        n = min(int(blocks_to_swap), self.num_blocks)
        if n <= 0 or self.param_count <= 0:
            return 0.0
        return sum(self.block_param_counts[self.num_blocks - n:]) / self.param_count

    def as_dict(self) -> dict[str, Any]:
        """给 API / 元数据用的扁平字典（不含 per-block 明细）。"""
        return {
            "model_channels": self.model_channels,
            "in_channels": self.in_channels,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "has_llm_adapter": self.has_llm_adapter,
            "param_count": self.param_count,
            "file_bytes": self.file_bytes,
        }


def arch_from_header(header: dict[str, Any], *, file_bytes: int = 0) -> AnimaCheckpointArch:
    """纯函数：safetensors header（tensor 名 → {dtype, shape, data_offsets}）→ 架构。"""
    x_key = next((k for k in header if k != "__metadata__" and k.endswith(_X_EMBEDDER_SUFFIX)), None)
    if x_key is None:
        raise CheckpointInspectError(
            "不是 Anima / Cosmos-Predict2 transformer 权重（缺 x_embedder.proj.1.weight）"
        )
    x_shape = header[x_key].get("shape") or []
    if len(x_shape) != 2:
        raise CheckpointInspectError(f"x_embedder.proj.1.weight 形状异常: {x_shape}")
    model_channels = int(x_shape[0])
    in_channels = int(x_shape[1]) // 4 - 1

    per_block: dict[int, int] = {}
    total = 0
    has_adapter = False
    for key, ent in header.items():
        if key == "__metadata__" or not isinstance(ent, dict):
            continue
        numel = _numel(ent.get("shape"))
        total += numel
        if _ADAPTER_MARK in key:
            has_adapter = True
            continue
        m = _BLOCK_KEY_RE.search(key)
        if m:
            idx = int(m.group(1))
            per_block[idx] = per_block.get(idx, 0) + numel
    if not per_block:
        raise CheckpointInspectError("权重里没有任何 blocks.N.* 张量")
    num_blocks = max(per_block) + 1
    missing = [i for i in range(num_blocks) if i not in per_block]
    if missing:
        raise CheckpointInspectError(f"DiT block 下标不连续，缺 {missing[:8]}（共 {len(missing)} 层）")

    return AnimaCheckpointArch(
        model_channels=model_channels,
        in_channels=in_channels,
        num_blocks=num_blocks,
        num_heads=HEADS_BY_MODEL_CHANNELS.get(model_channels),
        has_llm_adapter=has_adapter,
        param_count=total,
        block_param_counts=tuple(per_block[i] for i in range(num_blocks)),
        file_bytes=int(file_bytes),
    )


def inspect_anima_checkpoint(path: str | os.PathLike[str]) -> AnimaCheckpointArch:
    """读文件头并推断架构（毫秒级；不读 payload、不 import torch）。"""
    p = Path(path)
    header = read_safetensors_header(p)
    try:
        file_bytes = p.stat().st_size
    except OSError:
        file_bytes = 0
    return arch_from_header(header, file_bytes=file_bytes)
