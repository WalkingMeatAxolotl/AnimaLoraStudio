# Krea2 接入 flash-attn / xformers 注意力后端

**创建于** 2026-10-19
**触发** MOSS-Image 接入 `modeling/attention.py`（按模型实例选后端）之后，Krea2 仍固定 SDPA
**计划** 下个 release 处理

---

## 为什么要做

Windows 版 PyTorch 没把 FlashAttention 编进 SDPA（torch 2.11 + RTX 5090 实测强制 FLASH 报
`Torch was not compiled with flash attention`），SDPA 自动落到 EFFICIENT 内核。1024² 单层注意力
fwd+bwd（B=1，bf16，48 q 头 / 12 kv 头，head_dim 128）：

| 路径 | 耗时 |
|---|---|
| 现状：SDPA + bool padding mask + `repeat_interleave` 展开 kv | 74.1 ms |
| 去掉 pad 文本 token 后 SDPA（无 mask） | 49.4 ms |
| 去掉 pad 后 SDPA `enable_gqa=True` | 133.8 ms（更慢） |
| 去掉 pad 后 flash-attn 包（原生 GQA） | **16.6 ms** |
| 去掉 pad 后 xformers（kv `repeat_interleave` 成 4D） | **16.4 ms** |

28 层 DiT 按此估算，每步每张 1024² 图注意力约省 1.6 s（开梯度检查点时更多）。

## 难点：padding mask

Krea2 文本按 512 定长下限右 pad（`KREA2_MAX_LENGTH`，文本缓存兼容依据，不能改），前向总带
`(B, text_len)` 的 bool mask（文本精炼块 `txtfusion` 与联合注意力都用）。flash-attn / xformers
的稠密接口不吃这种 mask，必须先去 pad：

- batch=1：直接切掉 pad 的文本 token（被 mask 的 key 不影响有效 token；pad 位置的输出本来就不用）。
- batch>1：varlen——flash-attn `flash_attn_varlen_func(cu_seqlens)`，或 xformers
  `BlockDiagonalMask.from_seqlens`（Anima NaViT 已有同款工具）。
- 要回归：去 pad 路径与「SDPA + mask」在有效 token 上数值一致（容差内），以及 GQA 头映射正确。
- xformers 的 5D GQA 布局没有 backward（`NotImplementedError: memory_efficient_attention_backward`），
  只能把 kv 展开成 4D。

## 顺带要修的现存问题

daemon 出图选 xformers 时，`_load` 对模型调 `enable_xformers`，没匹配到钩子就抛
`RuntimeError("Exact ComfyUI KSampler parity is guaranteed only with xformers ...")`。Krea2 模型
没有钩子，所以「设置 → 出图 attention backend = xformers」（或 `auto` 且只装了 xformers）时
Krea2 出图会直接失败。MOSS 已随 `modeling/attention.py` 接入修好（模型实现
`enable_xformers_memory_efficient_attention` 钩子）；Krea2 接入时实现同一钩子即可。

## 做法

照 MOSS：模型实现 `set_attention_backend` + `enable_xformers_memory_efficient_attention`，族的
`load_dit` 调 `training.model_loading.apply_attention_backend`，去掉 `train.krea2_sdpa_only` 文案；
去 pad / varlen 工具放进 `modeling/attention.py`。
