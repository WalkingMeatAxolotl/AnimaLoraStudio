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
- **与本项目契合**：纯 ONNX，复用现有 onnxruntime 基建（同 WD14/CLTagger），可走统一下载中心。**license（已核，见 §6）**：imgutils 库 = MIT；CCIP 权重 = OpenRAIL（含使用行为限制、需向下游传递；按需下载、不打包再分发可接受）。**不要 `pip install dghs-imgutils`** —— 它 import 时会自动 `pip install onnxruntime`，撞项目的 onnxruntime 选 build 策略；正解是把 CCIP 的 ONNX 直接下到下载中心、用现成 `OnnxTaggerBase` 跑（`ccip_difference` 走一个 learned metric ONNX，不是裸 cosine，自实现要照搬才对得上阈值）。

### 通用 anime 视觉相似度：WD14 embeddings

- SmilingWolf 的 `danbooru2022_image_similarity`（HF Space）直接用 WD14 系视觉特征做 anime 图像相似度检索 —— 证明 WD14 特征做相似度是社区成熟做法。
- `deepghs/wd14_tagger_with_embeddings`（Apache-2.0）：在 WD14 ONNX 上加了 embedding 第二输出（子目录布局 + `tags_info.csv`），专门给相似度用。（注：先前误引的 `PrometheusProject/wd14_tagger_with_embeddings` 已 401 不可访问，用 deepghs 版替代。）
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
- WD14 anime 相似度：<https://huggingface.co/spaces/SmilingWolf/danbooru2022_image_similarity> · <https://huggingface.co/deepghs/wd14_tagger_with_embeddings>
- DreamBooth（CLIP-I/DINO 来源）：<https://arxiv.org/abs/2208.12242>

---

## 6. 交叉验证综合：可行方案汇总（2026-06，待并入 ADR 0011 Addendum）

> 在 §2–§4 的 A / B1 / B2 / B3 / C 框架上，叠加 8 路调研 + 逐候选交叉验证的结论。每个组件都核到了存在性 / license / 运行时契合 / 局限。**结论先行：§4 的推荐组合（A → B2/B3 → B1 → C）经交叉验证全部成立、可直接进实现**；下面补每条的精确落地方式、被验证修正的事实、几处文档更正点，以及本轮未覆盖、值得二轮的方法族。

### ① 交叉验证后的关键结论

1. **baseline 是必需的，且是最高 ROI 的一步**（回答"需不需要 baseline"=需要）。CCIP 自带 F1 标定阈值只能把"裸 cosine"变成"同/异角色判定"，部分缓解可解读性，但**不等于 base-vs-LoRA 的 Δ**——"无 baseline"这一根本缺口只有 §2-A 的 no-LoRA Δ 能补。统计依据：同 seed 配对差（Common Random Numbers）的方差被两路噪声协方差抵消，少量验证图即稳定；且 base 与 LoRA 吃同一条 booru prompt、同为动漫域，CLIP 文本塔对 tag 的系统误读与编码器域失配在相减时大部分抵消（仅"部分"——编码器整体不敏感时 Δ 也会变钝）。

2. **deepghs 生态（CCIP / WD14-emb / tag / ai_corrupt / dbaesthetic）全部核实为真、纯 ONNX、可走项目现有 onnxruntime + 统一下载中心**，与已有 WD14/CLTagger 同栈。但**不要 `pip install dghs-imgutils` / `sdeval`**：(a) imgutils 的 `utils/onnxruntime.py:_ensure_onnxruntime()` 在 import 时若没装 ort 会**自动 `pip install onnxruntime-gpu`**，与本项目 `studio/services/runtime/onnxruntime.py`（三互斥包、按 torch CUDA 大版本 cu12/cu13 选 build、用户在 Settings 主动触发）**直接冲突**；(b) imgutils 基础安装还拉 opencv-contrib-python(可能与项目的 opencv-python 撞)/pandas/scipy/sklearn/shapely 等约 20 个传递依赖（无 torch 但不轻）。**正解：把对应 ONNX 直接下到下载中心、用现成 `OnnxTaggerBase` 跑**——CCIP/dbaesthetic 的 384/448 预处理与阈值都公开，自己写十几行前后处理即可。

3. **CCIP 只解一根轴：单角色身份保真**。对画风/概念 LoRA、多角色图无效，也完全不碰 CLIP-T 的 booru-tag 问题；偏发型/轮廓、**弱发色/肤色**（发色画错的角色 LoRA 可能仍拿高分）。它是"角色 LoRA 子集的 DINO-I 替代"，不是三联指标的通用替代——必须门控（先判单角色）+ 与 baseline-Δ、tag-recall 组合。CCIP 训练数据是 zerochan.net 爬取（**不是 booru**，别和局限2 的 booru-tag 问题混为一谈）。

4. **license 已核清**：imgutils=MIT、sdeval=Apache-2.0、cyberharem=**AGPL-3.0**（网络 copyleft，**勿 vendor 其 eval.py**，只复刻算法）；WD14 系列（含 `deepghs/wd14_tagger_with_embeddings`）=Apache-2.0；但**模型权重 CCIP / ai_image_corrupted / anime_aesthetic = OpenRAIL**（含使用行为限制、需向下游传递），非宽松许可。本项目按需下载、不打包再分发权重 → OpenRAIL 可接受，进 THIRD_PARTY_NOTICES 标注即可。

5. **tag-recall / tag-vector 几乎零成本**：`studio/services/tagging/wd14.py:_postprocess_one(scores)` 收到的就是**阈值前的全量 tag 概率向量**（9083(v2)/10861(v3) 维），图↔prompt 的 P/R/F1 与图↔图的 tag-cosine 都只要在这一步抓住 `scores`，**无需新模型、无需 imgutils**。

6. **WD14 penultimate embedding ≠ 现成可读**：项目下载的 `model.onnx` 只暴露 tag 预测输出（`wd14.py:_download_model` 的 `allow_patterns=["model.onnx","selected_tags.csv"]`），且 `onnx_base.py:_tag_loop` 只取 `run(...)[0]`。要拿 emb 得换 `deepghs/wd14_tagger_with_embeddings`（**不同 repo、不同布局**：子目录 + `tags_info.csv`）+ 改 `_tag_loop` 捕获 `output[1]`——是小工程量，不是"免费读现有输出"。且 WD14-emb 是**属性相似度不是身份**（两个蓝发双马尾不同角色会高分），与 tag-vector(B3) 高度相关，别当两个独立指标。

7. **CCIP 的 `ccip_difference` 不是裸 cosine**：它跑一个学习度量头 `model_metrics.onnx`，公开阈值是按该头标定的——自实现时必须把 768d 特征过 metric head 才能对上阈值语义（这反而比裸 cosine 更可解释）。注意别张冠李戴：默认 `ccip-caformer-24-randaug-pruned` 阈值 0.178，而 F1 0.9409 属于另一变体 `ccip-caformer_b36-24`（阈值 0.213）。

8. **被剔除项**：`PrometheusProject/wd14_tagger_with_embeddings`（401 不可访问、不可核 license，被 deepghs Apache-2.0 版取代）= not-viable，**项目 §3 行77 与 §参考 行115 仍引用它，应改 `deepghs/wd14_tagger_with_embeddings`**；`sdeval` 包（v0.0.4 / 2024-01 停更）与 `tagger_embedding_aligner`（emb 去归一化基础设施，本项目对生成图直接跑 WD14 即可，用不上）= 借原语 / 跳过，勿作运行时依赖。

### ② 可行方案表

| 方案 | 采纳 | 怎么落地（本项目） | 开销 | 主要风险 | 来源 |
|---|---|---|---|---|---|
| **A. no-LoRA baseline + Δ**（同 seed 配对） | adopt | `eval_samples.py` 增 scale=0 baseline run（同 prompt/seed/底模，version 级缓存）；`eval_metrics.py` 存 baseline 分 + 算 Δ；前端指标卡显示"LoRA 值 + Δ"。三个现有指标全部加 Δ，复用现有 CLIP/DINO 编码器，零新模型 | 首个 checkpoint 多一组 baseline 出图+编码，version 内缓存复用、不随 checkpoint 增长 | 仅降随机方差、不修编码器域偏差；LoRA scale 极大时配对相关性下降 | [diffusers eval](https://huggingface.co/docs/diffusers/conceptual/evaluation) · [CRN 2409.02086](https://arxiv.org/abs/2409.02086) · [DreamBooth 2208.12242](https://arxiv.org/abs/2208.12242) |
| **C. WD14 tag-recall**（CLIP-T 动漫替代） | adopt | 新 `eval_tag` runner（仿 `eval_dino.py`）；对生成图跑现有 WD14（`_postprocess_one` 已有全量概率），算 prompt booru-tag 的 P/R/F1，按 category 加权；仅 booru-tag caption 启用 | 零新模型，一次 WD14 推理/图 | tagger 阈值敏感；自然语言 caption 不适用；训练 caption 同源时量的是"与 tagger 一致性" | [Civitai #2591](https://civitai.com/articles/2591/evaluating-anime-models-systematically-basics) |
| **B3. WD14 tag-vector 相似度**（图↔图，最可解释） | adopt | 同次 WD14 对参考图也跑，比 tag 概率向量 cosine / 加权 Jaccard（高频 tag 做 IDF）；key 如 `tag_i` | 零新模型，参考图侧可随 baseline 缓存 | 粒度粗、测属性非身份；与 B2 相关勿双计 | [SmilingWolf 相似度 Space](https://huggingface.co/spaces/SmilingWolf/danbooru2022_image_similarity) |
| **B2. WD14 penultimate emb cosine**（图↔图，anime dense） | trial | 下 `deepghs/wd14_tagger_with_embeddings`（Apache-2.0；注意子目录+`tags_info.csv` 布局，需新 catalog 项）；改 `onnx_base.py:_tag_loop` 捕获 `output[1]`；L2 归一后 cosine，key `wd_i`。画风/概念 LoRA 比 CCIP 更适用 | 新下一个同族 ONNX + 小改 _tag_loop | 属性非身份；与 B3 相关；固定单一 backbone 不可混维度 | [deepghs/wd14_tagger_with_embeddings](https://huggingface.co/deepghs/wd14_tagger_with_embeddings) · [discussions/2](https://huggingface.co/spaces/SmilingWolf/danbooru2022_image_similarity/discussions/2) |
| **B1. CCIP**（单角色身份保真，DINO-I 动漫替代） | trial | 把 `deepghs/ccip_onnx` 默认变体 `model_feat.onnx`+`model_metrics.onnx`+`metrics.json` 下到 `models/eval/ccip/`（**不 pip imgutils**）；新 `eval_ccip` runner 复用 `OnnxTaggerBase`；`ccip_difference` 走 metric head + 变体自带阈值判同/异；**先门控单角色**；取"被参考集判同角色比例"∈[0,1] | 新下 ~150MB ONNX + 单角色检测门控 | 仅单角色/仅身份；弱发色；权重 OpenRAIL；阈值为真人画作标定、迁生成图是强信号非保证 | [deepghs/ccip](https://huggingface.co/deepghs/ccip) · [imgutils ccip 文档](https://dghs-imgutils.deepghs.org/main/api_doc/metrics/ccip.html) · [Illustrious 2409.19946](https://arxiv.org/abs/2409.19946) · [Kohaku-XL-Epsilon](https://huggingface.co/KBlueLeaf/Kohaku-XL-Epsilon) |
| **CCIP upbound 归一 + F-β best-epoch**（方法学） | trial | B1 之上：held-out 参考集 `ccip_merge` 成原型→`ccip_upbound` 把分归一 [0,1]；F-β 合"像角色×没崩"自动选 epoch。**自实现，勿 vendor cyberharem(AGPL)** | 纯算法、无新依赖 | upbound 需干净单角色参考；阈值/权重需按 Anima 重标定；出图链不可移植 | [cyberharem eval.py](https://raw.githubusercontent.com/deepghs/cyberharem/main/cyberharem/eval/eval.py) · [sdeval](https://github.com/deepghs/sdeval) |
| **质量门：AICorrupt + anime_dbaesthetic**（正交轴） | trial(可选) | 直接下 `deepghs/ai_image_corrupted`(崩坏二分类) + `deepghs/anime_aesthetic`(swinv2pv3) ONNX，走现有 onnxruntime；独立轴并列，**不并入保真总分** | 各一个轻 ONNX | 正交于保真、可被"高糖风格"刷分、需 baseline；AICorrupt 训练于 SD1.5 对 Anima DiT 域偏移；裸 ONNX 分数方向(corrupted 高=坏)与 sdeval wrapper 相反易写反；权重 OpenRAIL | [ai_image_corrupted](https://huggingface.co/deepghs/ai_image_corrupted) · [dbaesthetic 文档](https://dghs-imgutils.deepghs.org/main/api_doc/metrics/dbaesthetic.html) |
| **skip：sdeval 包 / tagger_embedding_aligner / PrometheusProject 镜像** | skip | sdeval 停更→借原语不依赖包；aligner 是 emb 去归一化基础设施，本项目用不上；PrometheusProject 401 不可用→改用 deepghs 版 | — | 误装 sdeval/imgutils 会触发自动 pip ort + 依赖膨胀，撞项目 onnxruntime 管理 | [deepghs/wd14_tagger_with_embeddings](https://huggingface.co/deepghs/wd14_tagger_with_embeddings) |

### ③ 推荐优先级

1. **A baseline-Δ（先做）**——零依赖、零新模型，唯一补"无 baseline"(局限1)，且后续每个指标都因有对照而可解读。
2. **C tag-recall + B3 tag-vector（与 A 并行）**——复用已下载 WD14、一次推理同时拿"prompt 跟随(解局限2)"+"anime 图-图相似(解局限3)"，落地成本最低（`_postprocess_one` 已暴露全量概率向量）。
3. **B1 CCIP（trial）**——角色 LoRA（多数用例）身份保真信号最强、anime 域、带标定阈值；限单角色需门控 + 新模型 + OpenRAIL，排零成本项之后；配 A + C 才完整。
4. **B2 WD14-emb cosine（trial）**——补画风/概念 LoRA（CCIP 不适用）的 anime 域 dense 相似；需新下 ONNX + 改 `_tag_loop` 取第二输出，且与 B3 相关，B2/B3 不重复计。
5. **CCIP upbound + F-β（trial）**——B1 落地后顺势加，归一 0-1 + 自动选 epoch。
6. **质量门 AICorrupt + dbaesthetic（可选）**——正交"没崩/美学"轴，注意域偏移与分数方向坑。

> 取向延续 §4：**保留现有 CLIP-T/CLIP-I/DINO-I 不删（兼容+自然图基线），新增 baseline-Δ + 动漫域指标**。角色 LoRA 跑全套(含 CCIP)；画风/概念/多角色 LoRA 自动跳过 CCIP，靠 A + B2/B3 + C + dbaesthetic。

### ④ 仍存疑点

- **单角色门控**用什么：CCIP 仅单角色有效，算前需判"恰好一个角色"。是引入 imgutils person 检测（又一个新 ONNX），还是用"WD14 含 `1girl`+无 `2girls/multiple_*`"近似（零成本但糙）。
- **baseline 缓存失效粒度**：按 底模+prompt+seed+采样器 指纹缓存，换 variant / 改 prompt 集 / 改 seed 时如何失效，防脏复用。
- **替代 vs 并列**：默认并列新增、保留旧三指标兼容；但指标卡变多，是否按"角色/画风 LoRA"分档只显示相关指标（CCIP 仅角色档）。
- **自然语言/LLM caption 的 prompt 轴**：tag-recall(C) 只对 booru-tag caption 成立；NL caption 用什么测 prompt-following（保留 CLIP-T？还是不计该轴）。
- **B2 与 B3 是否都上**：二者同源高度相关，别双计；建议先 B3（零成本），B2 看是否有额外收益再上。
- **OpenRAIL 合规拍板**：CCIP / ai_image_corrupted / anime_aesthetic 权重 OpenRAIL（使用限制需传递）。按需下载、不打包再分发应可接受，进 THIRD_PARTY_NOTICES，但需确认"download-on-demand 不算再分发"。
- **文档更正**：§3 行72 imgutils 应为 **MIT**（非 Apache-2.0）、CCIP 权重应为 **OpenRAIL**（非"待确认"）；§3 行77 / §参考 行115 的 PrometheusProject 改 **deepghs/wd14_tagger_with_embeddings**。
- **本轮未交叉验证的方法族**（来自调研角度4/6/7/8，剔除原因=本轮未逐候选交叉验证、非判否，建议二轮再核）：directional-CLIP ΔI·ΔT、DINOv2 升级+SCR 阈值化、prdc(precision/density + recall/coverage)、Vendi/LPIPS 多样性、SSCD 或"生成图→训练集 top-1 NN"记忆/复制护栏、CSD 画风描述子、CMMD 分布距离。其中两条是当前指标集真正的盲区且近乎零依赖（可建在已算的 CLIP/DINO/CCIP emb 上）：**①copy/记忆护栏**（角色 LoRA 训练集小、易背诵训练图，而 DINO/CCIP 对 copy-paste 反给高分）；**②多样性/塌缩**（同 prompt 换 seed 不变=过拟合）。值得专门补一轮。
- **质量门先标定**：AICorrupt(SD1.5 域)、dbaesthetic(百分位参考分布未公开) 上线前应在 Anima DiT 实际出图上抽样校准阈值。

### 参考（本轮新增/核实）

- CCIP：<https://huggingface.co/deepghs/ccip> · <https://huggingface.co/deepghs/ccip_onnx> · <https://dghs-imgutils.deepghs.org/main/api_doc/metrics/ccip.html> · <https://huggingface.co/datasets/deepghs/character_similarity>
- WD14 embeddings（取代 PrometheusProject）：<https://huggingface.co/deepghs/wd14_tagger_with_embeddings> · <https://huggingface.co/spaces/SmilingWolf/danbooru2022_image_similarity/discussions/2>
- tag-recall / 系统化 anime 评测：<https://civitai.com/articles/2591/evaluating-anime-models-systematically-basics>
- 质量门：<https://huggingface.co/deepghs/ai_image_corrupted> · <https://dghs-imgutils.deepghs.org/main/api_doc/metrics/dbaesthetic.html>
- best-epoch 方法学：<https://github.com/deepghs/cyberharem>（AGPL，仅借算法）· <https://github.com/deepghs/sdeval>
- baseline/相对化方法：<https://huggingface.co/docs/diffusers/conceptual/evaluation> · <https://arxiv.org/abs/2409.02086> · <https://arxiv.org/abs/2108.00946>
- anime 评测范式背书：<https://arxiv.org/abs/2409.19946>（Illustrious，用 CCIP + Elo、弃 FID/CLIP）· <https://huggingface.co/KBlueLeaf/Kohaku-XL-Epsilon>（CCIP score 做角色保真基准）
- imgutils（MIT，注意启动期自动 pip 与项目 onnxruntime 管理冲突）：<https://github.com/deepghs/imgutils>
