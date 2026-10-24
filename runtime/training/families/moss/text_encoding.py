"""MOSS-Image 文本条件：Qwen3.5-9B last_hidden_state + 变长缓存（MOSS 设计稿 §5.5）。

- 模板：``<|im_start|>user\\n{caption}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n``。
  官方 ``apply_chat_template(..., enable_thinking=True)`` 与 ComfyUI
  ``MOSSImageTokenizer`` 的字面模板逐 token 一致（两边都会 strip caption）。
- 每条 caption 单独、不 padding 地前向（官方：避免 GDN 线性注意力受 padding
  影响），取 last_hidden_state，整段保留（含模板 token），上限 4096 token。
- 只加载语言模型（``Qwen3_5TextModel``），跳过视觉塔、MTP 头与 lm_head。

TE 目录布局（``text_encoder_path``）：``config.json``（Qwen3.5 配置）+ 权重，权重
二选一：

- HF 分片（``model.safetensors.index.json``，键 ``language_model.*``）——官方
  ``OpenMOSS-Team/MOSS-Image-*`` 的 ``text_encoder/``，bf16
- ComfyUI 单文件（键 ``model.language_model.*``）——``int8_tensorwise + convrot``，
  int8 原样常驻（Linear 与 embed_tokens 都是 int8），逐层反量化

tokenizer 取目录本身，没有时取同级 ``tokenizer/``（HF 仓库的 text_encoder/ 与
tokenizer/ 并列布局）。TE 生命周期与缓存协议在 ``training.text_stack``。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from studio.infrastructure.log_messages import msg
from ...text_stack import VarlenTextStack
from .forward import MossTextCondition
from .quant_int8 import parse_comfy_quant, patch_int8_embedding, patch_int8_linear

logger = logging.getLogger(__name__)

MOSS_TEXT_WIDTH = 4096
#: 官方 pipeline 与 ComfyUI 都截到 4096 token（含模板）
MOSS_MAX_TOKENS = 4096
#: 指纹 = TE 身份 + 取层 + 模板版本；存储形态另加后缀（设计稿 M8）
MOSS_TEXT_FINGERPRINT = "qwen3.5-9b-moss-last-think-v1"
#: int8 TE 走 ComfyUI 的「逐层反量化到 compute dtype + 普通 linear」（``-dq``）。
#: 曾用 W8A16 前向（后缀无 ``-dq``），两者嵌入有舍入级差异，换后缀让旧缓存重编
_INT8_FINGERPRINT_SUFFIX = "-int8convrot-dq"
#: 官方要求（模型卡 `transformers>=5.3`）；实际判据是 qwen3_5 模型类能否导入
MOSS_MIN_TRANSFORMERS = "5.3"
_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
#: 源键前缀 → Qwen3_5TextModel 键（HF 分片 / ComfyUI 单文件）
_LANGUAGE_PREFIXES = ("model.language_model.", "language_model.")


def check_transformers_supports_qwen35():
    """MOSS TE 需要 transformers 的 qwen3_5 实现；全局依赖下限不动，这里按需检查。"""
    try:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    except ImportError as exc:
        try:
            import transformers

            current = transformers.__version__
        except ImportError:  # pragma: no cover - 环境相关
            current = "未安装"
        raise RuntimeError(msg(
            "train.moss_transformers_too_old",
            required=MOSS_MIN_TRANSFORMERS, current=current,
        )) from exc
    return Qwen3_5Config, Qwen3_5TextModel


def moss_prompt(caption: str) -> str:
    return _TEMPLATE.format(str(caption).strip())


def _weights_layout(model_path: Path) -> tuple[str, list[Path]]:
    """``("hf", 分片列表)`` 或 ``("comfy", [单文件])``。"""
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return "hf", [model_path / name for name in sorted(set(weight_map.values()))]
    candidates = sorted(model_path.glob("*.safetensors"))
    if len(candidates) == 1:
        return "comfy", candidates
    raise ValueError(
        f"MOSS 文本编码器目录 {model_path} 里应有 HF 分片（model.safetensors.index.json）"
        f"或恰好一个 ComfyUI 单文件，实际找到 {len(candidates)} 个 .safetensors"
    )


def is_int8_text_encoder(model_path: str | Path) -> bool:
    """只看布局与 header，不读权重：Comfy 单文件且含 I8 张量即 int8 TE。"""
    path = Path(model_path)
    layout, files = _weights_layout(path)
    if layout != "comfy":
        return False
    from safetensors import safe_open

    with safe_open(str(files[0]), framework="pt", device="cpu") as handle:
        return any(
            handle.get_slice(key).get_dtype() == "I8"
            for key in handle.keys() if key.startswith(_LANGUAGE_PREFIXES)
        )


def _target_key(key: str) -> str | None:
    for prefix in _LANGUAGE_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return None  # visual / mtp / lm_head：不需要


def load_moss_text_model(model_path: Path, device: torch.device, dtype: torch.dtype):
    """meta 构造 ``Qwen3_5TextModel`` 后按键 assign 加载；int8 层挂反量化前向。"""
    Qwen3_5Config, Qwen3_5TextModel = check_transformers_supports_qwen35()
    from safetensors import safe_open

    try:
        from accelerate import init_empty_weights
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError("MOSS 文本编码器加载需要 accelerate（transformers 伴生依赖）") from exc

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"MOSS 文本编码器目录缺少 config.json：{model_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = Qwen3_5Config(**{k: v for k, v in raw.items() if k != "architectures"}).text_config
    if text_config.hidden_size != MOSS_TEXT_WIDTH:
        raise ValueError(f"MOSS 文本编码器 hidden_size 应为 {MOSS_TEXT_WIDTH}，实际 {text_config.hidden_size}")

    _layout, files = _weights_layout(model_path)
    logger.info(msg("train.moss_loading_text_encoder", path=model_path))
    # init_empty_weights 只把 parameters 放 meta；rotary inv_freq 等 buffer 在 CPU
    # 真实构造（krea2 单文件 TE 同款处理）
    with init_empty_weights():
        model = Qwen3_5TextModel(text_config)
    model.requires_grad_(False)

    state: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    quant: dict[str, int] = {}
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                name = _target_key(key)
                if name is None:
                    continue
                if name.endswith(".comfy_quant"):
                    layer = name[: -len(".comfy_quant")]
                    quant[layer] = parse_comfy_quant(handle.get_tensor(key))["convrot_groupsize"]
                    continue
                tensor = handle.get_tensor(key)
                if name.endswith(".weight_scale"):
                    scales[name[: -len(".weight_scale")]] = tensor.to(device=device)
                elif tensor.dtype == torch.int8:
                    state[name] = tensor.to(device=device)
                elif tensor.is_floating_point():
                    state[name] = tensor.to(device=device, dtype=dtype)
                else:
                    raise ValueError(f"MOSS 文本编码器 {name} 的类型 {tensor.dtype} 不受支持")

    int8_layers = {name[: -len(".weight")] for name, t in state.items() if t.dtype == torch.int8}
    if int8_layers != set(scales) or int8_layers != set(quant):
        raise ValueError("MOSS 文本编码器的 int8 层与 weight_scale / comfy_quant 不一一对应")
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"MOSS 文本编码器键不匹配：缺少 {list(result.missing_keys)[:5]}，"
            f"多出 {list(result.unexpected_keys)[:5]}"
        )
    del state
    modules = dict(model.named_modules())
    for layer in sorted(int8_layers):
        module = modules[layer]
        if isinstance(module, torch.nn.Embedding):
            patch_int8_embedding(module, scales[layer], quant[layer], dtype)
        else:
            # ComfyUI 编码时 TE 不走 W8A8（full_precision_mm）：每次前向反量化
            patch_int8_linear(module, scales[layer], quant[layer], forward="dequant")
    # buffers 构造在 CPU（init_empty_weights 只 meta 化参数）——搬到目标 device
    model.to(device)
    leftover = [n for n, t in (*model.named_parameters(), *model.named_buffers()) if t.device.type == "meta"]
    if leftover:
        raise ValueError(f"MOSS 文本编码器加载后仍有未物化参数：{leftover[:5]}")
    if int8_layers:
        logger.debug("moss_te: int8 convrot layers=%d", len(int8_layers))
    return model.eval()


def _load_tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    for candidate in (model_path, model_path.parent / "tokenizer"):
        if (candidate / "tokenizer_config.json").exists():
            return AutoTokenizer.from_pretrained(str(candidate), local_files_only=True)
    raise ValueError(
        f"找不到 MOSS tokenizer：{model_path} 与同级 tokenizer/ 下都没有 tokenizer_config.json"
    )


def pad_moss_conditions(contexts: Sequence[Tensor], *, device, dtype) -> MossTextCondition:
    """把变长 ``[L, 4096]`` 右 pad 成 ``[B, S, 4096]`` + mask。"""
    if not contexts:
        raise ValueError("MOSS 文本 batch 不能为空")
    target = torch.device(device)
    longest = max(item.shape[0] for item in contexts)
    padded = torch.zeros(len(contexts), longest, MOSS_TEXT_WIDTH, device=target, dtype=dtype)
    mask = torch.zeros(len(contexts), longest, device=target, dtype=torch.bool)
    for index, item in enumerate(contexts):
        padded[index, :item.shape[0]].copy_(item.to(device=target, dtype=dtype))
        mask[index, :item.shape[0]] = True
    return MossTextCondition(context=padded, attention_mask=mask)


class MossTextStack(VarlenTextStack):
    """Qwen3.5-9B 变长条件栈（生命周期 / 缓存 / 在线 LRU 见 VarlenTextStack）。"""

    _label = "MOSS"

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        cache_enabled: bool = True,
        tokenizer=None,
        model_loader=None,
        text_fingerprint: str = MOSS_TEXT_FINGERPRINT,
        max_tokens: int = MOSS_MAX_TOKENS,
        is_int8: bool | None = None,
    ) -> None:
        model_path = Path(model_path)
        check_transformers_supports_qwen35()
        #: int8 TE 编码的嵌入与 bf16 有量化级差异——指纹区分，换 TE 即全量重编（M8）
        self.is_int8_storage = is_int8_text_encoder(model_path) if is_int8 is None else bool(is_int8)
        if self.is_int8_storage:
            text_fingerprint = f"{text_fingerprint}{_INT8_FINGERPRINT_SUFFIX}"
        #: TE 上卡的显存增量估计（daemon 让位判据的兜底；首次加载后改用实测）：
        #: int8 权重 9.75GB 实测编码峰值 ≈8.1GB（不含视觉塔）；bf16 权重 ≈16.8GB
        self.te_load_bytes_estimate = (9 if self.is_int8_storage else 18) * 1024 ** 3
        super().__init__(
            model_path,
            device=device,
            dtype=dtype,
            compute_dtype=None,
            cache_enabled=cache_enabled,
            tokenizer=tokenizer if tokenizer is not None else _load_tokenizer(model_path),
            model_loader=model_loader or load_moss_text_model,
            text_fingerprint=text_fingerprint,
            cache_batch_size=1,
        )
        self.max_tokens = int(max_tokens)
        if self.max_tokens <= 0:
            raise ValueError("MOSS max_tokens 必须为正数")

    def _tokenize(self, caption: str):
        # 逐条、不 padding（官方 encode_prompt 同口径）；上限作用于套完模板的整串
        return self.tokenizer(
            moss_prompt(caption), return_tensors="pt",
            truncation=True, max_length=self.max_tokens,
        )

    def token_count(self, caption: str) -> int:
        """caption 编码后的文本 token 数（= 缓存条件的长度；只 tokenize，不加载 TE）。"""
        return int(self._tokenize(caption)["input_ids"].shape[-1])

    def _encode_many(self, captions: Sequence[str]) -> list[Tensor]:
        if not captions:
            return []
        model = self.ensure_model()
        contexts = []
        with torch.inference_mode():
            for caption in captions:
                inputs = self._tokenize(caption)
                outputs = model(
                    input_ids=inputs["input_ids"].to(self.device),
                    attention_mask=inputs["attention_mask"].to(self.device),
                    use_cache=False,
                    return_dict=True,
                )
                contexts.append(self._validate_context(
                    outputs.last_hidden_state[0], source="Qwen3.5 输出",
                ))
        return contexts

    def _validate_context(self, context: object, *, source: str) -> Tensor:
        if (
            not isinstance(context, Tensor)
            or context.ndim != 2
            or context.shape[0] == 0
            or context.shape[1] != MOSS_TEXT_WIDTH
            or not context.is_floating_point()
        ):
            shape = tuple(context.shape) if isinstance(context, Tensor) else type(context).__name__
            raise ValueError(f"{source} context 必须为非空浮点 (seq, {MOSS_TEXT_WIDTH})，实际 {shape}")
        return context

    def _pad_conditions(self, contexts, *, device, dtype) -> MossTextCondition:
        return pad_moss_conditions(contexts, device=device, dtype=dtype)
