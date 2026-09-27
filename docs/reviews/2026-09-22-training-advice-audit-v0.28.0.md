# v0.28.0 训练数据量、repeat 与总步数建议审计

日期：2026-09-22

状态：**审计完成；用于修订训练指南，不是模型质量排行榜。**

审计基线：`origin/dev @ ea01e774` 之后的 v0.28 指南草稿

## 1. 结论

旧表格不应继续作为训练建议：

| 旧说法 | 结论 |
|---|---|
| 单角色 LoRA：30 张起步、50–100 张扩展、repeat 10–20 | 张数和 repeat 都不是硬门槛；10–30 张高质量角色图可以成为有效小规模基线，但需要 checkpoint 对照，不能保证一次成功 |
| 画风 LoRA：50 张起步、100–300 张扩展、repeat 5–10 | 与本仓库历史配置明显不符；本地完整画风任务主要使用 repeat 1 或 2，29–100 张也能完整运行 |
| 多角色 LoKr：200 张起步、500+ 张扩展、repeat 1–3 | 缺少本地受控证据；多角色首先是类别覆盖与平衡问题，不能用总张数代替覆盖矩阵 |

更可靠的指南应同时说明：

1. 图片的目标一致性、变化覆盖和 caption 质量；
2. 每个文件夹的 repeat（相对采样权重）；
3. epoch、batch、梯度累积、分辨率档数和正则样本；
4. 当前 UI 推导出的 optimizer 更新步数；
5. 固定条件下的 checkpoint 视觉比较。

## 2. 旧建议从哪里来

`git blame` 显示旧数据量表来自仓库初始文档提交 `c6bd6b64`（2026-02-05），后来只被搬迁到 `docs/user-guide/training-tips.md`。仓库中没有与这组三档数字对应的实验报告、任务 ID 或模型族限定。

因此它属于早期通用经验摘录，不是 AnimaLoraStudio、Anima 或 Krea 2 的验证结果。

## 3. 本地训练记录怎么统计

### 3.1 数据源

只读检查了：

- `studio_data/studio.db` 的训练任务状态与项目／版本关联；
- `studio_data/tasks/<id>/run.log` 的训练文件夹、图片数、repeat、有效样本和步数计划；
- `studio_data/tasks/<id>/snapshot/config.yaml` 的模型族、epoch、batch、梯度累积与 adapter；
- `studio_data/tasks/<id>/monitor/state.json` 的实际结束步数。

没有读取或公开 caption、prompt、图片内容和凭据。

### 3.2 去重与纳入条件

- 97 次训练有日志，其中 93 次可解析出训练文件夹统计。
- “完整任务”要求 DB 状态为 `done`，且监控 `step >= total_steps`；共有 57 次。
- 同一版本可能重跑多次。按版本只保留最后一次完整任务后有 44 组，避免把重试当独立数据集。
- 画风谱系通过版本／配置／采样中的显式 `style` 标记及同项目版本沿袭识别；排除测试项目后有 38 组。
- 此分类可以支持“这些配置用于画风项目”，不能替代图片视觉检查。

### 3.3 画风谱系汇总

| 指标 | 38 个版本的历史分布 |
|---|---:|
| 训练图片 | 29–100；中位数 52；四分位约 44–59 |
| 文件夹 repeat | 26 组只用 2；11 组只用 1；1 组混用 1/2 |
| epoch | 2–100；中位数 40；四分位约 40–60 |
| 记录的总更新步数 | 60–6480；中位数 1820；四分位约 1160–3720 |

其中 2 epoch / 60 步属于短任务，不能作为成品训练建议。按较新的显式模型族拆分：

| 模型族 | 版本数 | 图片范围 / 中位数 | repeat | 总更新步数范围 / 中位数 |
|---|---:|---:|---|---:|
| Anima | 9 | 29–88 / 46 | 8 组为 2，1 组为 1 | 1260–3520 / 1680 |
| Krea 2 | 9 | 30–85 / 52 | 6 组为 1，3 组为 2 | 480–1740 / 900 |
| 未显式声明族的旧任务 | 20 | 30–100 / 53 | 以 2 为主 | 跨多个历史实现，不宜与当前版本直接合并定标 |

这些数字支持两个有限结论：

- 旧文档的 repeat 5–20 不是本仓库实际训练的常态；
- 画风任务不需要先达到 100–300 张才可以开始。

它们**不支持**以下说法：

- 29 张一定足以得到好画风；
- 900 或 1820 步是最优值；
- Anima 必然需要比 Krea 2 更多步；
- 任务完成或 loss 下降就等于模型视觉质量好。

历史任务跨实现、分辨率、正则集、adapter、优化器和数据内容，也没有统一的人工评分。上述范围只能描述“实际使用过并完整结束的配置”。

## 4. 当前版本如何计算训练量

当前实现位于：

- `studio/web/src/pages/project/steps/train/TrainPlanPanel.tsx`
- `runtime/training/phases/optimizer.py`

常规路径的近似关系是：

```text
文件夹有效样本 = 图片数 × repeat × 分辨率档数
总有效样本 = 训练文件夹有效样本 + 正则文件夹有效样本
每 epoch optimizer 更新步数 ≈ ceil(总有效样本 ÷ (batch_size × grad_accum))
自然总步数 ≈ 每 epoch 更新步数 × epochs
最终总步数 = min(自然总步数, max_steps)  # max_steps > 0 时
```

NaViT Packing 不使用这条简单的 batch 公式；UI 和训练端按 token pack 估算，并用 `≈` 标识跨 epoch 的小幅波动。

单张图片的计划曝光次数约为：

```text
repeat × epochs × 分辨率档数
```

曝光次数和 optimizer 更新步数不是同一指标。repeat 还会改变某个文件夹相对其它训练／正则文件夹的权重，因此不能把 repeat 当作孤立的“质量档位”。

## 5. 角色 LoRA 与 10–30 张社区经验

维护者提出的“社区用 10、20、30 张也能训练出很好角色 LoRA”与公开社区案例的总体方向一致：小而干净、角度和构图覆盖充分的数据集可以成功，更多图片并非必要前提。但社区案例跨 SDXL、Illustrious、Flux、Qwen 等模型，caption、rank、学习率和训练器也不同，不能直接变成 Anima / Krea 2 的效果保证。

本次抽查的社区入口包括：

- [The Right Number of Images for Training a LoRA](https://www.reddit.com/r/StableDiffusion/comments/1puzohv/the_right_number_of_images_for_training_a_lora/)：讨论本身就反映对固定张数没有共识；
- [Z-Image character LoRA on 29 real photos](https://www.reddit.com/r/StableDiffusion/comments/1p9e8g3/z_image_character_lora_on_29_real_photos_trained/)：是小数据角色训练案例，不可跨模型直接套参；
- [Basic Guide to Creating Character LoRAs for Klein 9B](https://www.reddit.com/r/StableDiffusion/comments/1ri65uz/basic_guide_to_creating_character_loras_for_klein/)：给出另一模型族的小数据经验；
- [Anima LoRA training settings for sd-scripts](https://civitai.com/articles/31972/anima-lora-training-settings-for-sd-scripts) 与 [A Complete Beginner's Guide to Local Anima LoRA Training](https://civitai.com/articles/31678/a-complete-beginners-guide-to-local-anima-lora-training)：作为 Anima 社区实践入口。

这些是社区经验样本，不是同一数据集上的受控实验。本报告只用它们支持“较小数据集存在成功案例、固定最低张数不可靠”这一弱结论，不从中派生统一 repeat 或总步数。

本地项目没有可靠的结构化“角色／画风”任务类型和统一视觉评分，因此本次无法用数据库独立验证角色 LoRA 的 10／20／30 张质量差异。指南将其写成**可开始做对照的小规模基线**，而不是最低门槛或推荐终值，并要求：

- 10–30 张内部覆盖正面／侧面、表情、构图、光照以及希望独立控制的服装变化；
- 避免同一原图的近邻裁剪或轻微变体虚增数量；
- 用更密的 checkpoint 比较判断欠拟合和过拟合；
- 若身份不稳，先补缺失角度和错误 caption，再决定增加曝光量；
- 若服装或背景被绑定，优先补变化和修 caption，而不是只提高 repeat。

## 6. 新建议

### 6.1 图片数量

- **单角色／单一服装**：10–30 张高质量图可以开始，不必先凑到 30、50 或 100；扩充优先补角度、表情、构图、光照和需要独立控制的属性。
- **画风／单一视觉概念**：先覆盖内容与构图变化。本地记录表明约 30–100 张是实际常见范围，但不是最低值或最佳区间。
- **多角色／复合概念**：先列每个身份或子概念的覆盖矩阵并检查平衡；目标纠缠时应考虑拆分 LoRA，而不是直接套用 200／500 张门槛。

### 6.2 repeat

- 默认从 `1` 开始。
- 只有希望某个文件夹提高相对采样权重时才使用 `2` 或更高。
- 改 repeat 后必须重新检查 epoch、有效样本、总更新步数和 checkpoint 间隔。
- 不应把 `repeat=10/20` 与 40–60 epoch 机械叠加。
- 不应复制相同图片来命中某个步数；这只增加重复曝光，不增加覆盖面。

### 6.3 总步数与停止点

- 不给角色、画风或某模型族规定一个通用总步数。
- 第一次运行应较密保存 checkpoint 和固定 seed 样图，先定位“尚未学到 → 合适 → 开始过拟合”的区间。
- 后续版本围绕该区间减少或增加训练量；不要仅凭 loss、epoch 编号或 `done` 状态选最终权重。
- 数据、模型族、adapter、batch 或梯度累积改变后，应重新检查 UI 的步数推导，不能沿用上一次的 epoch 数。

## 7. 仍缺什么证据

若要把当前“实验起点”升级为模型族专属推荐值，还需要：

1. 为任务增加角色／画风／服装／复合概念等结构化标签；
2. 记录数据集唯一图片数、近重复率、caption 策略和验证划分；
3. 固定底模、adapter、rank、优化器、采样 prompt / seed；
4. 对 checkpoint 做统一盲评，记录身份相似度、画风一致性、可控性和非触发偏移；
5. 分别比较图片数、repeat 与总更新步数，避免同时改变多个变量。

在这些证据齐备前，指南应帮助用户正确设计对照，而不是提供看似精确的万能配方。
