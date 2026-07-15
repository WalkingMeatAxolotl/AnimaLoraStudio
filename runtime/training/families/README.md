# families/ —— 模型族 registry（第 8 套 plugin registry，架构级）

设计权威口径：`docs/design/multi-model/04-synthesis.md`（接口冻结面在 03 §4.1 + 04 §3）。

## 一个族的三个居所

| 层 | 位置 | 内容 |
|---|---|---|
| 结构定义 | `modeling/<fam>/` | DiT/包装层（只依赖 torch/einops） |
| 行为适配 | `runtime/training/families/<fam>/` | ModelFamily 实现：loader / forward / preset / sampling / text_encoding |
| 资产清单 | `studio/services/models/families/<fam>.py` | 权重 repo / 下载 target / 默认路径（PR-4 落地） |

族名字符串（`anima` / `krea2`）是贯穿三层的唯一 join key。

## 边界纪律

- 共享循环只消费 `(latents, noise, t, pred, target, loss, mask, loss_weight)`；
  凡这些之外的模型知识（文本编码、pad_mask、检查点展开、采样栈）归族内。
- **共享代码禁止 `if family == "..."` 分支**——一律查 `spec.capabilities` 或 spec 字段。
- 缓存指纹是 latent 空间身份（`wan21-f8c16`）而非族名：同空间的族自动共享缓存。
- 演化：方法追加参数一律 keyword-only 带默认值；禁止 `**kwargs`。

## 加第 3 个族的步骤

1. `modeling/<fam>/` 放结构定义（模块命名对齐 ComfyUI 内部命名——kohya 键名编码模块路径）。
2. 本目录建 `<fam>/`：`__init__.py`（SPEC）+ `family.py` + `loader.py` + `preset.py` + `sampling.py`（+ 文本缓存族加 `text_encoding.py`）。
3. 注册：`families/__init__.py` 的 `_register(SPEC)` + `get_family()` 分支各一行。
4. schema：`model_family` Literal 加值 + 能力门控 `show_when`（PR-3 机制）。
5. studio：`FAMILY_ASSETS` 加清单（PR-4 机制）。
6. 测试：`tests/test_families_<fam>_*.py`（spec 常量 / preset / 扁平键样本）。

共享循环（masked loss / InfoNoise / losses / optimizer / eval / 暂停恢复）零修改。
