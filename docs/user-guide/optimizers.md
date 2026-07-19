# 优化器选型与起步参数

各优化器的推荐起点 lr / weight_decay、以及从 AdamW 切换时的换算关系。schema 字段描述只写"参数是什么"，**调参建议在这里**。

## 总览

| 优化器 | 推荐起点 lr | weight_decay | scheduler | state 显存 vs AdamW fp32 | 适用场景 |
|---|---|---|---|---|---|
| **adamw** | 1e-4 | 0.01 | cosine / cosine_with_warmup | 100%（基线） | 默认基线，几乎不踩坑 |
| **lion** | ≈ AdamW lr / 3（1e-4 → 3e-5）| AdamW wd × 3-10（0.01 → 0.03-0.1）| cosine / cosine_with_warmup | **≈ 50%**（只 exp_avg）| 显存吃紧但又想固定 lr |
| **automagic** | **1e-6**（必须；UI 切换时自动改）| 0（一般不开）| **none**（内部 per-param 自适应）| ≈ 50%（factored 2nd moment + int8 lr_mask）| 不想调 lr 又不想 Prodigy |
| **came** | ≈ AdamW 量级（1e-4 起步，可略高）| 0-0.01 | cosine / cosine_with_warmup | **≈ 50-60%**（exp_avg + 行/列分解二阶矩与 instability）| 想省 state 显存但保留 AdamW 式动量稳定性 |
| **prodigy** | 1.0（固定，UI 锁定）| 0.01 | constant 或 cosine | 比 AdamW 略大（多一个 d 状态）| 通用自适应，最稳的"不调 lr" |
| **prodigy_plus_schedulefree** | 1.0（固定）| 0.0 | **none**（Schedule-Free 内部 averaging）| 比 Prodigy 大一些（averaged weights）| 解决 Prodigy mutation ep / 风格突变 |
| **soap** | AdamW 量级（1e-4 ~ 3e-4）| 0.01 | cosine / cosine_with_warmup | **> AdamW**（exp_avg + exp_avg_sq + 每矩阵轴 Shampoo GG/Q）| 矩阵型 adapter（LoRA/LoKr）想更快拟合 |
| **soap_sf** | AdamW 量级（1e-4 ~ 3e-4）| 0.01 | **none**（Schedule-Free averaging）| ≈ soap（z 替掉 exp_avg）| 要 SOAP 提速 + 不想调 LR 调度；**短训练 ≤ ~100 步改用 soap** |

> 显存说明：AdamW8bit（bitsandbytes）才是真省显存基准（≈ AdamW fp32 的 25%）。Lion / Automagic 比 fp32 AdamW 省一半，但**不比 AdamW8bit 省**。

## Lion — 从 AdamW 切换

Lion 论文（Chen et al. 2023, [arxiv 2302.06675](https://arxiv.org/abs/2302.06675) §4.3）经验：

> "Lion needs a smaller learning rate than AdamW, e.g. 3-10× smaller, and a larger weight decay, e.g. 3-10× larger, to maintain similar effective weight decay strength."

| AdamW 参数 | Lion 推荐换算 |
|---|---|
| lr = 1e-4 | **lr ≈ 3e-5**（× 1/3）|
| lr = 1e-5 | lr ≈ 3e-6 |
| weight_decay = 0.01 | **weight_decay ≈ 0.03-0.1**（× 3-10）|

**为什么**：Lion 的 update 是 `sign()` 后的固定大小（`±lr`），不像 AdamW 按梯度幅度缩放。同样的 lr 在 Lion 上每步走得更猛，所以要降。weight_decay 的解耦更新公式里有 lr 相乘，lr 降了就要把 wd 提起来才能维持等效衰减强度。

如果直接把 AdamW 1e-4 拿来用：训练初期 loss 大概率发散或卡死。AnimaLoraStudio 在 `create_lion` 检测到 lr ≥ 1e-4 时会打 warning。

## Automagic — 必须 1e-6 起步

Automagic（[Ostris](https://github.com/ostris/ai-toolkit)）走 per-parameter 自适应 lr，全程不需要 scheduler。**`lr` 字段是每个参数的初始学习率**，不是常规优化器那种全局 step size。

- 上游 ostris / tdrussell 默认都是 `lr=1e-6`
- `[automagic_min_lr, automagic_max_lr]` 默认 `[1e-7, 1e-3]`，每个参数自己在这个区间里靠 sign-agreement 自适应
- 起点 lr 太高（如 AdamW 量级 1e-4）→ sign-agreement 调度需要很多 step 才能把 per-param lr 拉回工作区间，前期等价于 100× 跑飞

**UI 切换**：用户从其他优化器切到 Automagic 时，前端自动把 `learning_rate` 改写为 1e-6（仍可手动调）。保存配置 / CLI 直接传超过 1e-5 的值，训练启动期 `create_automagic` 打 warning，不强制改。

**已知行为**：`automagic_min_lr` / `automagic_max_lr` / `automagic_lr_bump` 是 instance global，**多 param group 时全局共享，不走 per-group**。当前 trainer 单组训练不受影响；未来若引入 LoRA+（B 矩阵 16× lr 类）多 group lr 调度，min/max/bump 仍是单值。这是上游 ostris/ai-toolkit + tdrussell/diffusion-pipe 一致的行为。

## CAME — 省 state 显存的置信度引导优化器

CAME（Luo et al. 2023, [arxiv 2307.02047](https://arxiv.org/abs/2307.02047)，ACL 2023 Outstanding Paper）解决的问题：Adafactor 式行/列分解二阶矩虽省显存，但分解近似会给 update 引入噪声、损失收敛质量。CAME 的做法是**置信度引导**——追踪 update 与其动量 exp_avg 的残差平方 EMA（instability），对动量更新做逆方差加权：残差大（近似不可信）的坐标步子自动收小，残差小的坐标全速走。

- **lr**：AdamW 量级真实值（**不像 Prodigy 填 1.0**），LoRA 起步 1e-4；论文经验 CAME 可承受比 Adafactor 略大的 lr
- **scheduler**：常规优化器，cosine / cosine_with_warmup 都可配
- **state 显存**：exp_avg（全尺寸）+ 4 条行/列分解向量 ≈ AdamW 的一半多一点；比 Lion 大（多了分解统计），比 AdamW / SOAP 小
- `came_beta3`（默认 0.9999）是置信度 EMA 衰减，一般不动；`came_eps2` 是 instability 的下限正则：调大会压平逐坐标的置信度差异，同时整体步长随之缩小（不是中性地"关闭"置信度加权）

实现派生自官方 [yangluo7/CAME](https://github.com/yangluo7/CAME)（MIT），做了 bf16 训练工程适配（state 固定 fp32 + stochastic rounding 写回 + resume fixup），算法公式与官方逐行对齐。

## Prodigy / PPSF — lr 锁 1.0

Prodigy 系列内部估计步长 `d`，**`lr` 字段必须为 1.0**（工厂会强制覆盖）。调参重点：

- `prodigy_d_coef` / `ppsf_d_coef`：估出 d 的整体缩放系数。欠拟合调到 2.0+，过拟合 / 小数据集调到 0.5。
- PPSF 比 Prodigy 多一个 `prodigy_steps` 字段：训练后期冻结 d 估计避免跳档，建议设为总步数的 1/4 ~ 1/2。

PPSF 用 Schedule-Free averaging，sample / save 前必须 `optimizer.eval()`，事后 `optimizer.train()`。Studio 内部用 `optimizer_eval_mode` context manager 自动处理，CLI 用户参考 `utils/optimizer_utils.py:optimizer_eval_mode`。

## SOAP / SOAP-SF — 二阶预条件提拟合速度

SOAP（Vyas et al. 2024, [arxiv 2409.11321](https://arxiv.org/abs/2409.11321)）= **Adam 跑在 Shampoo 的特征基里**：用梯度协方差的特征基旋转梯度，在该基里做标准 Adam，再旋转回来。对矩阵型参数（LoRA / LoKr 的低秩因子）拟合更快；相比纯 Shampoo，靠 `soap_precondition_frequency` 少刷新特征基省算力。**动机是拟合速度**，不要指望它改善纹理 / 画质本身——换 SOAP 是用显存换速度。

`soap_sf` 在 SOAP 外面套 Schedule-Free（Defazio et al. 2024, *The Road Less Scheduled*, [arxiv 2405.15682](https://arxiv.org/abs/2405.15682)）：丢一阶动量，用 base 序列 z 与 Polyak 平均 x 的插值取代 LR 调度，所以 **`lr_scheduler` 固定 none**（启动期校验 fatal），sample / save 自动走 averaged x（`optimizer_eval_mode` 统一处理，跟 PPSF 一样）。

**lr**：SOAP 系用 AdamW 量级真实 lr（**不像 Prodigy 填 1.0**）。LoRA/LoKr 起步 1e-4 ~ 3e-4。

**提速关键 = `soap_max_precond_dim`**（逐维阈值）：

- 某轴维度 ≤ 阈值 → 该轴建满秩二阶预条件；> 阈值 → 该轴退化为 Adam。
- 设大（如 `10000`）让大特征维也做二阶 = **提速主来源**；设小（如 `256`）只预条件 rank 维 = SOAP-lite，省显存但丢掉大部分提速。
- 配 `soap_precond_in_state: false` 把可重算的 GG/Q 剔出 ckpt 保持 state 小（从零训练不 resume 时零代价；resume 会冷重建特征基，有几步过渡）。

**短训练注意**：Schedule-Free 的 Polyak 平均在极短训练（≤ ~100 步）严重滞后（x ≈ 轨迹质心 = 欠拟合），那种 regime 用纯 `soap` 不要用 `soap_sf`；千步级训练 SF 正常，判图建议在 ~880 步以后。

## 选哪个

- **没头绪，想稳的**：AdamW + cosine_with_warmup，跟着默认 preset 走（两族通用）
- **显存吃紧 + 不想动 lr**：Lion，按上面换算把 lr 降 3×
- **省 state 显存但想保住 AdamW 式稳定收敛**：came，lr 直接用 AdamW 量级
- **不想调 lr + 不想踩 Schedule-Free 坑**：Prodigy
- **风格 LoRA 怕 mutation ep**：prodigy_plus_schedulefree
- **per-param 细粒度自适应**：Automagic，记得起点 1e-6
- **想要更快拟合（有显存预算）**：soap（带 scheduler）或 soap_sf（免调度），`soap_max_precond_dim` 设大；短训练用 soap
