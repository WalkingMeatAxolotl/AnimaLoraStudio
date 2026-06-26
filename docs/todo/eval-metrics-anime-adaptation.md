# LoRA eval 指标改进调研：baseline + 动漫域适配

> 状态：调研 / 待拍板。本文档先把现状、三个优化方向、社区/论文调研写清楚，
> 不含实现。决策后再开实现 PR（届时把采纳项写进 ADR 0011 Addendum）。
>
> 相关：[ADR 0011](../adr/0011-lora-eval-metrics.md)（eval 指标体系）、PR #138 /
> #319（held-out 验证集）/ #323（评估可见性）。

## 1. 现状：三个指标怎么算、用什么模型

每张 held-out 验证图会用「它自己的 caption 当 prompt + 固定 seed + 底模 + LoRA(scale 1.0)」出一张图，得到三元组 `(生成图, 参考图=验证图, prompt=caption)`。三个指标都是把图/文编码成向量后算**余弦相似度**（向量先 L2 归一化，余弦 = 点积），再对所有样本取平均。

| 指标 | 模型 | 比什么 | 测什么 | 代码 |
|---|---|---|---|---|
| **CLIP-T** | `openai/clip-vit-base-patch32`（图+文双塔） | 生成图 emb ↔ prompt 文本 emb | prompt following | `eval_clip.py:380-383` |
| **CLIP-I** | 同上（仅图塔） | 生成图 emb ↔ 参考图 emb | 整体语义相似 | `eval_clip.py:390` |
| **DINO-I** | `facebook/dinov2-small`（CLS 特征） | 生成图 ↔ 参考图 | 主体/结构保真 | `eval_dino.py:342` |

三个共同前提：**生成永远带 LoRA、scale 固定 1.0、没有 baseline**（`eval_samples.py:577` `apply_loras(..., scale=1.0)`）。

### 三个根本局限

1. **无 baseline → 绝对值难解读。** 底模（Anima 本身就是动漫底模）对动漫参考图天然有不低的相似度，绝对分里「底模本来就有的」和「LoRA 学到的」混在一起。CLIP-I=0.72 是好是坏，没有对照就不知道。
2. **CLIP-T 的 tag-caption 问题。** openai CLIP 文本塔在自然英文上训练，**不认 booru tag**（`1girl, blue_hair, ohwx`）。tag 风格 caption 对它是 out-of-distribution → 分数被压得极低、噪声大（实测 CLIP-T≈0.0467）。
3. **CLIP-I / DINO-I 非动漫适配。** openai CLIP 与 DINOv2 都是自然图（LAION / ImageNet-ish）训练，**对二次元角色身份、画风的区分力弱**——没被训过细分动漫角色 / 风格。用在动漫 LoRA 评测上抓不准「像不像这个角色 / 这个画风」。

## 2. 三个优化方向（详解：「方案是什么意思」）

### A. no-LoRA baseline + Δ

**是什么**：每次评估时，**同样的 prompt + 同样的 seed**，再用**纯底模**（不挂 LoRA，等价 scale=0）出一组对照图，对它们算同样的三个指标。最终展示 **Δ = LoRA − baseline**，而不只是 LoRA 的绝对分。

**为什么**：把「LoRA 的效果」从「底模 + prompt 本身的效果」里分离出来。例：CLIP-I baseline 0.61 → LoRA 0.72，Δ=+0.11，才说明 LoRA 真的把生成往参考拉近了；如果 Δ≈0，说明这个分是底模白送的。也能更早暴露「LoRA 没学到东西 / 学过头」。

**开销**：baseline 对**同一 version 是不变的**（同底模、同 prompt、同 seed、与 checkpoint 无关）→ **只算一次、缓存复用**，跨该 version 的所有 checkpoint 共享。所以新增成本 ≈ 一次额外出图 + 编码，不随 checkpoint 数增长。

**改动面**：出图层加「baseline run」（scale=0 / 不挂 adapter）；run.json / metrics.json 存 baseline 分；`listEvalMetrics` 返回 Δ；前端指标卡显示「LoRA 值 + Δ」。

**取舍**：纯加法、不依赖任何外部模型，**性价比最高、风险最低**。唯一成本是首个 checkpoint 评估时多出一组 baseline 出图。

### B. 动漫域图像相似度（替代 / 补充 CLIP-I·DINO-I）

**是什么**：用**动漫域训练的模型**算图像相似度，而不是自然图的 CLIP/DINO。三条可选路（见 §3 调研）：

- **B1. CCIP**（anime 角色身份相似度）：`deepghs/ccip`，专门判「两张图是不是同一个动漫角色」。对**角色 LoRA**，这是 DINO-I 的动漫域正确替代。ONNX，走项目已有的 onnxruntime（同 WD14/CLTagger）。**局限：只适用单角色图**（多角色 / 场景图不适用）。
- **B2. WD14 embedding 相似度**（anime 通用视觉相似度）：用 WD14 tagger 的视觉特征（penultimate embedding）算余弦。**项目本来就下载 WD14**，复用现成模型 + onnxruntime。比 CLIP-I 更懂动漫语义。
- **B3. tag-vector 相似度**（动漫原生、可解释）：对生成图 + 参考图各跑一遍 tagger，比 **tag 概率向量**（cosine / 加权 Jaccard）。最可解释——能落到「学没学到 `blue_hair`、`twintails` 这些标签」。

**为什么**：自然图 CLIP/DINO 对动漫角色 / 画风分辨率不够；动漫域模型在「同角色 vs 不同角色」「这个画风 vs 那个画风」上敏感得多。

**改动面**：新增一个 metric runner（类似 `eval_clip.py`/`eval_dino.py` 的 job + scorer 结构），加新指标 key（如 `ccip_i` / `wd_i` / `tag_sim`）到 metric_states / 前端指标卡；模型走统一下载中心（§参考 #321 CLIP/DINO 接入方式）。

**取舍**：B1（CCIP）对角色 LoRA 信号最强但限单角色；B2 通用、复用 WD14、零新模型族；B3 最可解释但 tag 阈值 / 黑名单影响结果。建议 **B2 或 B3 做通用项，B1 作为「检测到单角色」时的加分项**。

### C. CLIP-T → tag-recall

**是什么**：放弃「caption 喂 CLIP 文本塔」，改成：**对生成图跑 tagger，看它能不能召回 prompt 里的关键 tag**。即 recall = (生成图 tag ∩ prompt tag) / prompt tag（或按 tag 置信度加权）。

**为什么**：tag-caption 对 CLIP 文本塔 OOD（局限 2）。而「生成图能被 tagger 标出 prompt 里要求的 tag」直接、动漫原生地衡量了 prompt following，且可解释（哪些 tag 没召回）。

**改动面**：CLIP-T 旁边加一个 `tag_recall` 指标（或在无自然语言 caption 时替代 CLIP-T）；复用 WD14。

**取舍**：依赖 tagger 质量与阈值；对自然语言 caption（LLM tagger 产出）不如对 booru-tag caption 合适——可按 caption 形态择一。

## 3. 社区 / 论文调研：有没有动漫特调

### 角色身份：CCIP（强推，DINO 的动漫替代）

- **CCIP = Contrastive Anime Character Image Pre-training**（deepghs / narugo1992、7eu7d7）。专门做「两张图是不是同一个动漫角色」的视觉相似度。`ccip_extract_feature()` 出特征向量、`ccip_difference()` / `ccip_same()` 比对，阈值取 F1 曲线最大点。
- 训练数据 `deepghs/character_similarity`：~24 万图 / 3982 个角色。ONNX 模型（caformer 变体），通过 `dghs-imgutils` 或直接 onnxruntime 加载。
- **局限**：限单角色图；多角色 / 场景不适用。
- **与本项目契合**：纯 ONNX，复用现有 onnxruntime 基建（同 WD14/CLTagger），可走统一下载中心。**license 待确认**（imgutils 为 Apache-2.0，CCIP 权重需逐个核对）。

### 通用 anime 视觉相似度：WD14 embeddings

- SmilingWolf 的 `danbooru2022_image_similarity`（HF Space）直接用 WD14 系视觉特征做 anime 图像相似度检索 —— 证明 WD14 特征做相似度是社区成熟做法。
- `PrometheusProject/wd14_tagger_with_embeddings`：在 WD14 ONNX 上加了 embedding 第二输出，专门给相似度用。
- **项目已经下载 WD14**，复用零新模型族，是 B2 的现成落点。

### CLIP-T 的动漫替代

- **没有**干净的「anime CLIP 图文相似度模型」适合给 booru-tag prompt 算 CLIP-T。Waifu-Diffusion / Animagine 等是**带 finetuned CLIP 文本编码器的扩散模型**，不是独立的图文相似度打分器。
- 结论：CLIP-T 走 §2-C 的 **tag-recall（WD14）** 比找 anime CLIP 现实。

### DINO 的动漫替代

- **没有**广泛使用的 anime-finetuned DINOv2（DINO 是自监督、社区没出动漫版）。**CCIP 实质填了「anime 图像身份相似度」这个位**。

### 学术参照

- CLIP-I + DINO 做主体保真，源自 **DreamBooth**（Ruiz et al., 2022, arXiv 2208.12242）—— 当前实现照搬了这套自然图范式。本调研的核心论点正是：**动漫域应换成域内模型**。
- 注：搜到的 `arXiv 2305.14672`（Quantifying Character Similarity with ViT）是**历史文字字形**相似度，与动漫角色无关，**不引为依据**。

## 4. 推荐组合与优先级

1. **A（baseline + Δ）**：先做，独立、低风险、性价比最高，解决「绝对值没法解读」。任何后续指标都受益于有 baseline 对照。
2. **B2 或 B3（WD14 通用 anime 相似度 / tag-vector）**：复用项目已有 WD14，零新模型族，补上「CLIP-I/DINO 非动漫适配」。
3. **B1（CCIP）**：作为「检测到单角色」时的角色身份加分项（DINO-I 的动漫域替代）；需引入 CCIP 模型 + license 核对。
4. **C（tag-recall）**：与 B 共用 WD14，顺带把 CLIP-T 的 tag-caption 问题解决；按 caption 形态（booru tag vs 自然语言）择一启用。

> 取向：保留现有 CLIP-T/CLIP-I/DINO-I 不删（向后兼容 + 跟自然图基线对照），**新增** baseline-Δ 与动漫域指标，让用户同时看到「自然图视角」和「动漫域视角」。

## 5. 开放问题（待拍板）

- 动漫域相似度选哪条：B1(CCIP) / B2(WD14 emb) / B3(tag-vector)，还是组合？
- 新指标是**替代**还是**并列新增**（默认倾向并列新增，保留兼容）。
- baseline 出图的缓存粒度（version 级？随 prompt/seed/底模 指纹失效）。
- CCIP / WD14-embedding 模型的下载来源 + license（统一下载中心接入，参考 #321）。
- CCIP「限单角色」如何门控（先检测人数 / 角色数，再决定是否算 ccip）。

## 参考

- CCIP：<https://huggingface.co/deepghs/ccip> · <https://deepghs.github.io/waifuc/main/advanced_guides/ccip/index.html> · imgutils metrics：<https://dghs-imgutils.deepghs.org/main/api_doc/metrics/ccip.html>
- imgutils：<https://github.com/deepghs/imgutils>
- WD14 anime 相似度：<https://huggingface.co/spaces/SmilingWolf/danbooru2022_image_similarity> · <https://huggingface.co/PrometheusProject/wd14_tagger_with_embeddings>
- DreamBooth（CLIP-I/DINO 来源）：<https://arxiv.org/abs/2208.12242>
