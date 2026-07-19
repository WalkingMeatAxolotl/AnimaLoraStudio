# 多模型支持 · 04 综合裁决

- 状态：三视角（01 代码归属 / 02 生态对标 / 03 接口契约）收敛后的最终设计。与 `00-decisions.md` 一起构成本改造的权威口径；两者冲突时以本文为准（本文对 00 的修订在 §4 逐条列出）。
- 日期：2026-07-14
- 输入：`01-code-layout.md`、`02-ecosystem-survey.md`、`03-interface-evolution.md`

## 1. TL;DR

- 三视角高度收敛：**「共享循环 + ModelSpec（声明）/ ModelFamily（行为）两分 + registry 派发」**被代码现状（01）、全行业实践（02）、压力测试（03）三方独立确认。02 的结论尤其有分量：8 家对标项目全部收敛到这一形态，没有一家靠复制训练脚本活得好。
- 目录方案采 01：**按层切 + 族聚合纪律**，一个族 = `modeling/<fam>/` + `runtime/training/families/<fam>/` + `studio/services/models/families/<fam>.py` 三处同名单元，族名字符串是唯一 join key，启动期校验三处一致（02-P2：这一点做到位即超越所有对标项目）。exec-load 退役改正常 import。
- 接口冻结面采 03 §4.1 全文为规范文本（八方法 + 七不变量 + 两个拒绝位 `objective`/`temporal`），外加本文 §3 的四处裁决修正（最主要：新增第九方法 `convert_lora_state_dict`，采纳 02 的生态证据）。
- **D8 修订**（03 的最重要实地发现）：派发不能只落在 `phases/models.py`——loader 三件套 + `sample_image` 有 5 个循环外调用方，必须经 `anima_train` 公共命名空间收口（supervisor/cmd_builder/入口脚本仍然不动）。
- 反模式纪律（02）全部采纳为执行红线：共享循环禁止族名 if、禁止向共享文件堆积横切功能、PR-2 必须把 Anima 完整迁入不留双世代过渡态、缓存指纹失配自动重算不抛 shape error。

## 2. 三视角一致确认的主干（不再讨论，直接执行）

1. **形态**：ModelFamily 适配层 + 第 8 套 registry；加第 3 个族 = 新建约 8 文件 + 4 处一行注册，共享循环零修改（01 §3.2；02-P2 生态黄金标准 <100 行共享改动；03 §3.1 Qwen-Image 压测零裂验证）。
2. **ModelSpec / ModelFamily 两分**：声明常量与行为方法分离（02-P1：ComfyUI `LatentFormat`+`sampling_settings`、SimpleTuner 类属性、ai-toolkit 能力 flag 三家印证；03 §2.1/§2.2 给出精确形状）。
3. **缓存指纹 = latent 空间身份 + layout 版本，族名不入键**（02-P3 ai-toolkit `latent_space_version` 方案；03 §2.5 协议细节 + grandfather 规则）。D6 的跨族共享由此免费兑现。
4. **timestep 双旋钮双归属**：训练端 t 采样留在共享 `timestep_samplers/` registry（K2 分辨率感知 shift 实现为新策略，如 `krea2_shift`，仅实现一次——02-P4/A2，diffusion-pipe 每族复制是反面教材）；推理端 sigma 时刻表归 family `sample_image` + `spec.sampling.shift_policy`（03-⑨）。两者不与现有 `timestep_shift_resolution_aware`（SD3 sqrt 修正）合并——公式与语义都不同。
5. **入口脚本不改名**，`anima_` 前缀定为产品名口径（01 §5；02 §10.3-5a 生态支持；改名成本清单存档于 01 §5.2 备查）。
6. **键名向 ComfyUI 生态看齐**是行业事实标准，且保存键名转换是 per-family 窄钩子、不写通用映射（02-P5，diffusers 3000 行转换账单为戒）。
7. **能力门控走 schema 管道**：`cap_gate()` 作者写时展开成 `model_family==...` 表达式，三个 show_when 求值器零改动；pydantic validator + bootstrap 双保险；交叉不变量 `cached_varlen ⇒ ¬caption_tag_ops` 在 registry 注册期校验（03 §2.4；02-P6 确认 schema 驱动 UI 是我们对全部对标项目的差异化优势）。
8. **utils/ 切法**：算法族无关、preset 族相关——`ANIMA_PRESET` 迁 `families/anima/preset.py`，`AnimaLycorisAdapter` 改 preset 注入并更名；与延后的 utils 完整重构是衔接关系，仅修订「preset 不进 lora/ 包」一项（01 §7）。
9. **磁盘 models root 不按族分**，唯一改动：新族 TE 落 `text_encoders/<safe_dir_name(repo)>/` 子目录，Anima 存量扁平布局零迁移（01 §8.2）。
10. **测试布局**：flat `tests/` 不动；新增 `test_model_spec.py` / `test_model_families.py`（registry ↔ schema Literal ↔ catalog 三方一致性断言）+「`phases/models.py` 不得再出现 `load_anima_model(` 字面量」防回归断言（01 §9）。

## 3. 冲突点裁决

三视角仅四处真实分歧，逐条裁决如下。

### C1 · K2 采样器代码放共享 `inference_samplers/` 还是族内？—— 裁：族内（从 03）

01 目标树曾把 `flow_euler.py` 画进共享 `inference_samplers/`；03 §4.3 论证采样器数值代码不跨族抽象（Comfy parity 栈与 musubi/diffusers 参考实现不同源，共享基类只会制造伪共性），`inference_samplers/` registry 保留为 **Anima 内部**组织方式。**裁决**：K2 的 FlowMatchEuler 动态 shift 栈整体落 `families/krea2/sampling.py`；01 目标树相应修订（删去 `inference_samplers/flow_euler.py` 一行）。第 3 个 flow-matching 族落地时若发现 solver 真重复，再上提——那时有两个已证实的实例，符合 03 §4.4 的抽象时点纪律。

### C2 · `forward_train` 返回 pred 还是 pred+target 数据类？—— 裁：只返回 v_pred（从 03）

02 §10.3-2 建议对齐 musubi `call_dit → DiTOutput`（返回 pred+target），理由是给未来非 v-pred 族留自由度。03 的 SDXL 压测恰好否决了这个自由度：target 代数 `ε−x₀` 是共享循环的心脏（InfoNoise record、loss_weighting、leap 全部依赖），下放 family 等于「共享循环」名存实亡，非 RF 族的正确答案是 `objective` 拒绝位 + 将来付 per-objective loop 全价。**裁决**：`forward_train → v_pred`（形状同 noisy），target 留循环，`spec.objective` v1 仅 `"rectified_flow"`。musubi 需要 DiTOutput 是因为它同时支持 I2V/多模态目标，我们的产品边界不含这些。

### C3 · 能力位要不要 frozenset 之外再加布尔 flag 面？—— 裁：单一 frozenset（收窄 02）

02-P1(b) 建议学 ai-toolkit 同时提供能力集合（管 UI）与布尔 flag（管代码路径）。**裁决**：只保留 `capabilities: frozenset[str]` 单一事实源——UI 经 `cap_gate()` 编译查询，代码经 `"navit" in spec.capabilities` 查询，两个消费面共享同一数据。ai-toolkit 的双面并存是它 legacy 迁移不彻底的产物（02-A4 恰好点名），不是值得复制的设计。

### C4 · LoRA 保存键名转换钩子要不要进 v1 冻结面？—— 裁：要，作为第九方法（从 02）

03 冻结了 `lora_preset()` + `lora_metadata()` 两方法，键名约定藏在 preset 的 `lora_prefix` 里；02 的生态证据（musubi `convert_weight_keys` 默认恒等、ai-toolkit `convert_lora_weights_before_save`、OneTrainer 四格式导出、diffusion-pipe per-model 保存格式）表明「target 选择」与「保存键名变换」是两个独立关注点，且 K2 的 Comfy 键名核对（open question）很可能需要前缀之外的结构性变换。**裁决**：v1 冻结面增补第九方法：

```python
def convert_lora_state_dict(self, sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """保存前的键名/结构变换钩子，默认恒等。Comfy 侧 K2 键名核对结论的落点。"""
```

默认恒等实现放 protocol 层，Anima 不覆写。现在加成本为零，Phase 3 才发现要加则是契约破坏。

## 4. 对 `00-decisions.md` 的修订与增补

| # | 内容 | 来源 |
|---|---|---|
| **D8′（修订）** | supervisor/cmd_builder/入口脚本不动**维持**；但派发收口点从「`phases/models.py`」改为「`anima_train` 公共命名空间 re-export `get_family/resolve_family`，覆盖 phases + `anima_generate`/`anima_daemon`/`anima_reg_ai`/`eval_samples`/`sample_runner` 全部 6 个调用面」 | 03 §1.3 |
| D10 | 目录方案：按层切 + 族聚合纪律，目标树以 01 §4.1 为准（含 C1 修订）；exec-load 退役、`find_diffusion_pipe_root` 塌缩为 shim（sister 契约 7 名不减） | 01 §3/§6 |
| D11 | 接口规范：03 §4.1 冻结面 + §4.2 扩展点 + §4.3 不抽象清单整体采纳，外加本文 C1-C4 修订（九方法版） | 03 |
| D12 | 缓存协议：npz 增 `latent_fingerprint`+`layout_version` 两键；无键存量 grandfather 为 `wan21-f8c16`；失配走既有删除重编码路径（自动、无用户动作） | 03 §2.5、02-P3/A7 |
| D13 | 恢复与产物防串：resume state 与 LoRA metadata 各加 `model_family` 标记，跨族加载 fail-fast（修 `strict=False` 静默冷启动隐患）；存量无标记 grandfather 为 anima | 03 §2.6/§4.5-⑦ |
| D14 | 训练端 K2 shift = `timestep_samplers/` 新策略（共享实现一次）；推理端 = `spec.sampling.shift_policy` typed union；不与 `timestep_shift_resolution_aware` 合并 | 02-P4、03-⑨ |
| D15 | 执行红线：共享循环禁止 `if family == ...`（一律能力位/spec 字段）；横切功能禁止堆进 family 基类或共享大文件；PR-2b 后 `TrainingContext.qwen_model/qwen_tok/t5_tok` 必须退役，不留 Anima 旧路径 | 02-A1/A4/A5 |
| D16 | family 加载器必须对「checkpoint 与所选 family 不匹配」fail-fast 并给可操作文案（消费级用户拿 Anima 权重配 K2 版本的错误必然发生） | 02-P7 |
| D17 | `latent_rgb_factors`（latent2rgb 预览投影系数）进 `LatentSpec`——`anima_daemon` 硬编码的 Wan21 系数收编，K2 同 latent 空间直接复用 | 02-P1(a) |
| D18 | K2 v1 训练中预览用 Raw 权重 + CFG 采样（模型已在显存里）；Turbo 热切换（musubi `--turbo_dit` 模式）不进 v1，Turbo 作为 Generate 页可选底模留给 Phase 4 评估 | 02 §10.3-6 |
| D19 | text cache 与 latent npz 缓存同域：放数据集 train/ 目录随图 sidecar。理由：caption 本身在 train/ 下、与 VAE 缓存一致、直接被既有项目导出 npz bundle 机制（PR #391）覆盖。同 caption 跨图重复存储的代价接受。失效判据仍按 D3/D12（文件内嵌 caption hash + TE 指纹，失配自动重编码）；非 caption prompt（sample/negative）的聚合缓存文件**修订（2026-07-18）**：改放 task 档案根（`tasks/<id>/.text-cache/`，纯 CLI 退回 output_dir），不落 train/——放 train/ 会被数据集扫描当 concept 文件夹误触；bundle 不再打包聚合缓存 | 用户裁定 2026-07-14（取代本文原「集中式内容寻址」倾向）；聚合缓存位置 2026-07-18 修订 |

## 5. 更新后的 Phase 1 执行序列

在 00 §5 基础上合入三文档的范围增补（PR-2 拆 2a/2b 采 01 §10）：

- **PR-1**：ModelSpec 常量收敛（含 D17 的 rgb factors）+ npz 指纹/layout 版本 + grandfather（D12）。新增 `tests/test_model_spec.py`。
- **PR-2a（机械刀）**：`modeling/anima/` 归位（git mv）+ exec-load 退役 + attention backend 别名表塌缩 + `find_diffusion_pipe_root` shim 化。验证：用户真卡跑训练 + 出图（PR #303 先例）。
- **PR-2b（适配刀）**：`families/` registry + protocol（九方法版）+ AnimaFamily 完整迁入（loader/forward/text_encoding/comfy_qwen/sampling/navit/leap/sra/preset 全部归位，`loop.py:222-254` 文本块与 pad_mask 构造下沉）+ **派发经 `anima_train` 公共命名空间收口 6 个调用面（D8′）** + resume/metadata family 标记（D13）+ lycoris preset 注入改造 + `TrainingContext` 字段退役（D15）。硬门禁：`test_anima_generate_xy.py` / `test_anima_train_migration.py` 全绿。
- **PR-3**：schema `model_family` Literal + `cap_gate()` 三层接线 + `t5_tokenizer_path` show_when + validator（含 `cached_varlen ⇒ ¬caption_tag_ops`）。
- **PR-4**：studio `FAMILY_ASSETS` registry + `ModelsConfig.selected` per-family 迁移 + catalog/downloader 遍历化 + TE 子目录布局。

Phase 2（text cache 基建）、Phase 3（Krea2Family，含 D16 指纹校验、`krea2_shift` 策略、`families/krea2/sampling.py`）、Phase 4（UI + Turbo 评估）不变。

## 6. 拍板结果（2026-07-14，全部闭环）

| # | 事项 | 结论 |
|---|---|---|
| 1 | `modeling/anima/` 下沉 | **做**，随 PR-2a（与 exec-load 退役同刀） |
| 2 | `DIFFUSION_PIPE_ROOT` 退役节奏 | 保留一个 release 的弃用 warning，下版删 |
| 3 | `models.py` 瘦身后改名 `vae.py` | 做，随 PR-2b |
| 4 | text cache 存放位置 | **train/ 目录随图 sidecar，与 latent 缓存同域**（用户裁定，见 D19；本文原「集中式内容寻址」倾向作废） |
| 5 | K2 LoRA Comfy 键名核对 | Phase 3 前闭环；落点已备好（`KREA2_PRESET.lora_prefix` + `convert_lora_state_dict`） |
| 6 | Phase 0（真卡验证） | 已执行，结果见 §7 |

## 7. Phase 0 结果（2026-07-14 记录，用户真卡实测）

实际用 **ai-toolkit**（而非 musubi）完成 K2 LoRA 训练验证：

1. **32GB 可训但余量≈0**（「刚刚好，差一点 OOM」）。D2 的 32GB 下限成立，但无安全边际。推论：
   - Phase 3 的「K2 专属小规模 block swap 兜底」从备选**提升为计划内可选项**（默认关、撞显存开）；fp8 基建仍按 D2 搁置。
   - K2 v1 默认强制 grad checkpointing；TE 预缓存后必须释放（03-⑩ 已定）；训练中预览的采样期峰值需专门关注。
   - 风险提示：近满载峰值在 Windows WDDM 有换页崖先例（VAE decode 190s 事件，PR #281）——「差一点 OOM」的工况正是崖区，实现时优先保峰值而非平均占用。
   - 已确认（2026-07-14 补）：当次训练**底模无 fp8 量化（bf16）**——与我们计划的 bf16 实现直接可比，32GB 结论原样采信。
2. **ComfyUI 升级后可加载并跑通该 LoRA** → K2 推理链路验证通过。

### 7.1 键名核对结论（拍板项 5 闭环；实物：`chenbin_style_krea2_v2.safetensors`）

对 ai-toolkit 产物 safetensors 头部的实测（518 tensors = 259 个 Linear × lora_A/lora_B）：

- **格式为 ComfyUI 原生 / PEFT 风格**：`diffusion_model.<模块路径>.lora_A.weight` / `.lora_B.weight`，bf16，**无 `.alpha` 键**，rank 32。非 kohya `lora_unet_*` 风格。
- target 覆盖实证：每 block `attn.{wq,wk,wv,wo,gate}` + `mlp.{up,down}`（7×28=196），外加 `txtfusion.refiner_blocks.*`、`txtmlp.*` 等文本融合段共 259 个 Linear——**`attn.gate`（gated sigmoid attention 的输出门）与文本融合层都在 target 内**，与 musubi「全模型 Linear」口径一致，`KREA2_PRESET` 照此定。wq [6144]/wk·wv [1536] 的形状差同时实证了 GQA 48Q/12KV。
- **saver 决策（2026-07-14 修订：与 Anima 统一为 kohya 风格）**：经对本地 ComfyUI（`G:\ComfyUI-aki-v1.6\ComfyUI`，已含 Krea2 支持）源码核实，三条链路全部走通，K2 产物**直接沿用 Anima 的 kohya 格式**（`lora_unet_*` + `lora_up/lora_down` + `.alpha`，`lora_prefix="lora_unet"`），`convert_lora_state_dict` 两族都保持恒等（第九方法保留为逃生舱）：
  1. `comfy/lora.py:191-196`：对模型 state dict 中每个 `diffusion_model.*.weight` 自动生成 `lora_unet_{path 下划线扁平}` 映射——由 state dict 动态构建，对含 Krea2 在内的所有族生效；
  2. `comfy/weight_adapter/lora.py:160,47`：kohya `lora_up/lora_down` 命名与 `.alpha` 键（scale=alpha/rank）原生支持，与 PEFT `lora_A/B` 并列识别；
  3. `comfy/ldm/krea2/model.py`：内部模块命名 `blocks.N.attn.{wq,wk,wv,gate,wo}` / `blocks.N.mlp.{gate,up,down}`（SwiGLU 三 Linear）/ `txtfusion.refiner_blocks.*` / `txtmlp.N`，与 ai-toolkit 产物键路径一致。
- **由此产生的硬约束（Phase 3 移植要求）**：我们的 `modeling/krea2/` 模块命名必须与 ComfyUI 内部命名逐字一致（kohya 键编码的就是模块路径），移植时以 `comfy/ldm/krea2/model.py` 的命名为准（结构可参考 diffusers，命名跟 Comfy）；并加单测钉死若干扁平键样本（如 `lora_unet_blocks_0_attn_wq`、`lora_unet_txtmlp_1`）防止重命名破坏产物兼容。
