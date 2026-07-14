"""Anima 族结构定义：Anima(MiniTrainDIT) + LLMAdapter + attention backend 状态机。"""

from modeling.anima.anima_modeling import (  # noqa: F401
    Anima,
    set_attention_backend,
    set_flash_attn_enabled,
    set_xformers_enabled,
)
