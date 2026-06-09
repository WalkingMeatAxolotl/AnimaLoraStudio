# Concept Slider POC — Saturation Tweaker

**创建于** 2026-06-09
**分支** `feat/concept-slider-poc`（fork 自 dev `bade65c`）
**当前状态** 🟡 v3 训练中；待评估
**触发** 用户希望在 Anima 上做类似 SDXL 社区"saturation tweaker"的连续滑块 LoRA：固定 seed 下，weight ∈ ℝ 连续插值控制图像饱和度，**理想状态：不改变其他维度**。

---

## TL;DR — Cold pickup 用

| 项 | 当前状态 |
|---|---|
| 方法 | image-pair concept slider（Gandikota+ 2023），4 或 6 forward / step |
| Adapter | 复用 `AnimaLycorisAdapter`；ComfyUI 兼容输出 |
| 数据 | `tmp/slider_data/` 188 张 booru 风格图，统一 caption `"a photo"` |
| 最新跑 | v3：rank=1, lr=3e-5, eta=0.5, Lab pair, bidirectional, 500 steps |
| 已知问题 | v2 时：+1 强 / -1 弱（不对称）、跨 weight 亮度漂；v3 算法侧已修，效果待验 |
| 决定性发现 | 数据集 `chroma↔L*` 实际 ρ=-0.07（HSV S 那个 -0.51 是颜色空间数学伪相关）；亮度漂主因是 VAE latent 耦合，不是数据 |

---

## 目标 + 验证标准

**目标**：训一个 saturation tweaker LoRA，达到：

1. weight=0 时 ≡ 不加 LoRA（已验证 ✓，lycoris bypass+multiplier=0 路径干净）
2. weight=+w 时饱和度提升 w 比例，**风格 / 构图 / 亮度尽量不变**
3. weight=-w 时饱和度降低 w 比例，对称程度跟 +w 类似
4. LoRA 文件可被现有 Generate 页 load（已验证 ✓，复用 `AnimaLycorisAdapter.save()` 输出 `lora_unet_*` 前缀 + `ss_*` metadata）

**实测验证**：训练中 `--sample-steps N` 每 N 步出 3 张图（fixed seed × {-1, 0, +1}），肉眼比对。

---

## 方法：image-pair concept slider

### 4-forward 训练循环（单向，默认）

每个 micro-batch：

```
对每张图，PIL 生成 (pos, neg) pair：
  pos = 原图（高饱和）
  neg = ImageEnhance.Color 或 Lab 缩 chroma 出的低饱和版

VAE encode 两份 → latents_pos, latents_neg
同 noise、同 t 计算 noisy_pos / noisy_neg

# Base 前向（LoRA off, no_grad）
pred_pos_base = model(noisy_pos, t, cross)
pred_neg_base = model(noisy_neg, t, cross)

# LoRA 前向（grad）
pred_lora = model(noisy_pos, t, cross)
target = pred_pos_base + eta * (pred_pos_base - pred_neg_base)
loss = MSE(pred_lora, target)
```

LoRA 学到的方向 = `eta * (pos - neg)` 在 velocity 空间的偏移。
推理时 `weight=+w` 等价应用这个偏移 ×w；`weight=-w` 反向应用。

### 6-forward 双向训练（`--slider_bidirectional`）

在 4-forward 基础上追加：

```
pred_lora_neg = model(noisy_neg, t, cross)  # LoRA on, grad
target_neg = pred_neg_base - eta * (pred_pos_base - pred_neg_base)
loss += MSE(pred_lora_neg, target_neg)
```

动机：单向训练 LoRA 只见过 `noisy_pos` 输入，`weight=-1` 推理是外推；双向强制对两端都有训练信号。代价 +50% wall-clock。

---

## 迭代历史

### v1 — 默认 baseline（**失败：mode collapse**）

| 参数 | 值 |
|---|---|
| rank | 4 |
| alpha | 1 |
| lr | 1e-4 |
| eta | 1.0 |
| pair op | PIL ImageEnhance.Color (neg=enhance(0.5)) |
| data | 80 张 booru reg 集 |
| steps | 3000 |
| 时间 | ~70 min |

**结果**：mode collapse。step 150 时 LoRA 已经把 `weight=+1` 锁死成"一个固定的动漫女角色"，`weight=-1` 锁死成"另一个固定真人女"。step 3000 跟 step 150 的 +1/-1 视觉几乎不变——LoRA 把 rank=4 的容量用来记忆两个端点的具体图像，而不是学连续方向轴。

**根因诊断**：
- rank=4 + lr=1e-4 容量充裕 + 步长大 → 跳进局部最优
- 80 张数据全是 booru 风格 → "饱和度差异 + 动漫风格" 在数据里强相关 → LoRA 走捷径
- sample prompt 是真人写实，跟训练分布不匹配 → +1 看起来像"切到动漫"

### v2 — 收紧容量 + 拓数据（**改善但仍有问题**）

| 参数 | 值 |
|---|---|
| rank | 1 |
| alpha | 1 |
| lr | 3e-5 |
| eta | 1.0 |
| pair op | PIL ImageEnhance.Color |
| data | 188 张（80 → 188，用户扩） |
| steps | 500 |
| 时间 | ~12 min |

**结果**：mode collapse 消失，weight 连续控制开始工作。但 3 个新问题：

1. **+1 强 / -1 弱不对称**：weight=+1 饱和度提升明显，weight=-1 降低几乎看不出
2. **跨 weight 亮度漂**：weight={-1, 0, +1} 三张图主体亮度不一致
3. **±1 影响太大**：用户反馈 ±0.5 才是好用的强度

### v3 — 算法三件套（**当前训练中**）

| 参数 | 值 | 改动理由 |
|---|---|---|
| rank | 1 | 同 v2 |
| alpha | 1 | 同 v2 |
| lr | 3e-5 | 同 v2 |
| eta | **0.5** | 修"±1 太强" |
| pair op | **lab_chroma** | 理论上更纯（Lab a*/b* 跟 L* 数学正交）|
| bidirectional | **on** | 修 +1/-1 不对称（双向监督）|
| data | 188 张（未 prune）| 同 v2 |
| steps | 500 | 同 v2 |
| 时间预估 | ~18 min（+50% 因 bidirectional） |

**待验**：上述 3 个问题各改善多少。结果填写到下面这里：

> v3 训练命令在 `tmp/slider_out_v3/`；评估结果待补

---

## 数据集统计分析（2026-06-09，188 张）

工具：`tools/analyze_slider_dataset.py`
产物：`tmp/slider_analysis/{stats.csv, health_hist.png, correlation.png, outliers.txt, REPORT.md}`

### 决定性发现 1：HSV S ↔ L\* 强负相关是数学伪相关

观察：
- `sat_hsv ↔ L_star`：ρ = **-0.514**（看起来像 confounder）
- `chroma ↔ L_star`：ρ = **-0.067**（真实色彩量跟亮度几乎无关）

原因：HSV S = (max - min) / max，亮度↑ → max↑ → S 数学性下降。这是颜色空间属性，不是数据集的内在偏差。

**结论**：v2 出现的跨 weight 亮度漂**不是数据集 confounder 造成的**。真正主因：
- VAE latent 空间里 saturation 和 brightness 维度耦合（无法在 pixel/pair 层修）
- PIL Color enhance 跟 Lab chroma scale 都有 HSV V ~4% 漂（无解，颜色空间局限）
- 模型自己的 anime saturation 先验

**别再追这条**。v3 的 Lab pair op 是边际改善，但不期望大变化。

### 决定性发现 2：饱和度分布严重右偏

- `sat_hsv` mean=0.32, median=0.29
- `chroma` mean=18, median=16
- 大头集中在 chroma < 20 的中低饱和区

含义：LoRA 的 saturation direction 是在低饱和区学的。推理时面对 anime 高饱和图（典型 chroma 30+），`weight=-1` 外推困难——**部分解释了不对称**。修法：补 30-50 张 chroma > 25 的高饱和图（暂未做）。

### 决定性发现 3：19/188 张图是 dead-signal（10%）

判定：`chroma < 5` OR `chroma_std < 5` OR `L* < 5` OR `L* > 95`。

清单：见 `tmp/slider_analysis/outliers.txt`
工具：`tools/prune_slider_dataset.py`（移动到 `tmp/slider_data_dead/`，可逆）

**未执行**——v3 正在用 tmp/slider_data 训练，运行中移动文件会触发 FileNotFoundError。v3 跑完后是否做 prune 由 v4 比对决定。

### 其他指标分布良好

- HSV V / CIE L\* / L\* spatial std / chroma_std / RGB 通道：均跨度好、无明显偏置
- Edge density 偏简单（mean=0.046），但不影响 slider 任务

---

## 待办（v3 跑完后决定）

按"如果 v3 改善程度"分支：

**情况 A — v3 三件套都明显改善**
- ✅ POC 实质成功
- 选打磨方向：v4 = v3 + prune 19 张 dead-signal，看是否进一步改善
- 之后讨论 Studio UI 集成 / yaml / 量产 axes

**情况 B — v3 部分改善但仍有问题**
- 优先级 1：扩数据集到 ~240 张，补 30-50 张高饱和图（chroma > 25）
- 优先级 2：再训 v4，eta=0.3 进一步收
- 优先级 3：评估是否需要更激进改动（如训练时 brightness 桶平衡采样）

**情况 C — v3 完全不改善**
- 算法侧基本到顶
- 需要考虑：Anima 的 anime 先验对 slider 任务可能是根本性阻碍（model 本身就把 saturation 和 style 耦合在训练中学过）
- 备选路径：直接走 SDXL base 训 slider，套用到 SDXL anime finetune 上（论文社区标准流程）

---

## 代码结构

```
runtime/training/concept_slider/
├── __init__.py
├── data.py          SaturationPairDataset + _lab_chroma_scale + collate_pair + build_dataloader
└── loop.py          run() — 4-forward 或 6-forward 训练循环 + multi-weight 采样

utils/lycoris_adapter.py
└── AnimaLycorisAdapter.set_multiplier(s)    运行时切 LoRA scale
└── AnimaLycorisAdapter.disabled()           context manager 临时关 LoRA

runtime/training/cli.py
└── 5 个新 flag（见下）

runtime/anima_train.py
└── main() 里加 mode 分支：concept_slider 走自定义 build_dataloader + loop

tools/
├── analyze_slider_dataset.py    数据集 9 指标分析 + 直方图 + 相关矩阵
└── prune_slider_dataset.py      按 stats.csv 移除 dead-signal 图
```

砍掉的范围（POC 不做）：
- Studio UI 集成
- yaml 配置
- Cached latent 路径（pair op 现场算）
- ARB 分桶（pair shape 必须一致 → 固定单分辨率）
- reg 集（POC 路径直接绕开 MergedDataset）
- kv_trim / loss_weighting / huber / InfoNoise record
- wandb upload / 断点续训

---

## CLI flags（concept slider 专用）

| flag | 默认 | 含义 |
|---|---|---|
| `--training_mode {lora, concept_slider}` | `lora` | mode dispatch |
| `--slider_eta FLOAT` | 1.0 | target = pos_base + eta·(pos_base - neg_base)；论文标 1.0，POC 推 0.5 |
| `--slider_caption STR` | `"a photo"` | 统一 caption；零文本信号最干净 |
| `--slider_neg_strength FLOAT` | 0.5 | neg 的饱和度缩放 factor；0.0=全灰度（不推荐），1.0=原图 |
| `--slider_pair_op {lab_chroma, pil_color}` | `lab_chroma` | pair 生成算子；lab 严格保 L*，pil 保 BT.601 L |
| `--slider_bidirectional` | off | 6-forward 双向训练；+50% wall-clock；修对称性 |

复用现有 flag（不新增）：`--data-dir / --transformer / --vae / --qwen / --t5-tokenizer / --output-dir / --output-name / --resolution / --batch-size / --grad-accum / --lr / --lora-rank / --lora-alpha / --lora-type / --epochs / --max-steps / --save-every-steps / --sample-steps / --sample-seed / --mixed-precision`

---

## 经验教训（写给未来的 Claude）

1. **HSV S 跟亮度的相关是数学伪相关**：高亮度像素的 S 由公式 (max-min)/max 数学性受压，跟数据集本身没关系。**判数据集 confounder 必须用 Lab chroma**，不要看 HSV S。

2. **PIL ImageEnhance.Color 实际保 BT.601 L 比 Lab chroma scale 保 CIE L\* 还好**——我之前以为 Lab 是 dominantly better，实测 PIL 的 L\* delta 0.0017 vs Lab 的 0.2443（Lab round-trip 量化 + gamut 截断引入 0.001 量级残漂）。**Lab 的好处主要在理论清晰、不是实测精度**。

3. **rank=1 + 低 lr 是 concept slider 的基础**：v1 rank=4 + lr=1e-4 直接 mode collapse，把容量用来记忆端点而非学方向。**rank 越小、lr 越小，越能强迫 LoRA 学连续轴**。

4. **统一 caption 比 cltagger 标好**：concept slider 论文实测发现"文本侧零信号"训练出的 LoRA 隔离度最高。给每张图标 booru tag 反而会让 LoRA 把 slider 方向跟特定 tag 关联。

5. **`weight=0` 必须 ≡ no-LoRA 是基础诊断**：训练完先在 Generate 页测这一条；不通过说明 adapter 路径有 bug，比看 +1/-1 效果优先。我们验证过 lycoris bypass + multiplier=0 路径干净。

6. **训练数据里所有同质的 secondary 特征都会被 LoRA 学进去**：80 张 booru 全是 1girl portrait → LoRA 学到的"高饱和方向" = "动漫脸方向"。多样化（不同主体 / 场景 / 画风）比"更多数据"更重要。**多样化在风格层，不是数量层**。

7. **`docs/todo/concept-slider-poc.md`（本文件）作为 session handoff doc**：每次 v_n 迭代完更新结果。允许新 session cold pickup 而不必读完整段对话。

---

## Commit 历史（feat/concept-slider-poc 分支）

```
ba20c52  feat(concept-slider): POC image-pair saturation tweaker training
74f3019  feat(concept-slider): Lab pair op + 双向训练修不对称
6344b14  tools(concept-slider): 数据集统计分析 + dead-signal prune 脚本
f4d81ea  fix(tools): prune dest 默认放 data_dir 同级而非子目录
```

跟 dev 分叉点：`bade65c`

---

## 关键引用文件

- 训练：`runtime/training/concept_slider/loop.py` + `data.py`
- Adapter：`utils/lycoris_adapter.py:284-323`（`set_multiplier` + `disabled()`）
- CLI：`runtime/training/cli.py:32-58`
- main() dispatch：`runtime/anima_train.py:118-150`
- 数据分析：`tools/analyze_slider_dataset.py`
- 分析报告：`tmp/slider_analysis/REPORT.md`
- 当前最佳运行 output：待补 v3
