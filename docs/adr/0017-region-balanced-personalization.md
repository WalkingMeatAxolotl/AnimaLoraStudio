# 0017 — 以可退火空间权重实现区域平衡个性化训练

**状态**：Accepted

**日期**：2026-08-04
**决策者**：ROCm 个性化训练维护者

## 背景

少量人物图训练容易先记住脸部和固定位置，也容易把衣服、姿势、背景与主体标识词绑定。
社区工作流常在前期强化局部、后期恢复整图；论文分别给出了自适应过拟合控制、选择性描述、
多概念区域监督和先验保持的可解释组件。项目需要一条可审查、可关闭、兼容既有训练数据的实现，
而不是复制视频中未公开的算法细节。

## 候选方案

1. 把主区域裁成额外图片并增加 repeat。实现简单，但会改变数据集步数、构图分布和缓存身份；
   后半程也无法严格恢复原始整图目标。
2. 把主区域写进现有 ignore mask。两者语义相反；复用后无法同时表达“不学这里”和“先强化这里”。
3. 保留原图，在 latent loss 上施加独立正空间权重，并按全局进度退火；另以冻结底模参考前向
   计算 APT 过拟合指标。

## 决策

采用方案 3：

- 每张图最多一个 `*.regions.json` 主区域，矩形使用归一化坐标；resize、crop、flip 与 latent
  cache 都保持几何一致。
- 前 45% 训练保持区域强化，45%–55% 余弦退火，55% 后区域权重严格退出，回到原来的整图
  mean loss。空间 reduction 除以有效权重和，避免小框改变整张图在 batch 中的总权重。
- ignore mask 与主区域独立，组合时相乘。
- APT-inspired 控制器按连续 flow 时间分 10 个 bin，比较临时关闭 adapter 的冻结底模误差与
  当前 LoRA 误差，使用论文的 `gamma = 1-exp(-T*(EMA_base-EMA_tuned))`、`1-gamma` loss
  权重和自适应仿射概率。正则集不参与指标学习或仿射增广。
- 主体 caption 使用 SID 风格结构化预设；DreamBooth 式先验保持继续复用现有 `reg_data_dir`
  与 `reg_weight`，不另建第二套正则数据层。

## 理由

空间权重不裁图，训练步数和图像上下文都保持稳定；显式退火让后半程回到基础模型熟悉的整图
目标。独立 sidecar 可被 UI、bundle 和缓存可靠追踪，也不会改变已有 caption/mask 格式。
APT 额外前向只保存轻量分箱状态，不复制一份底模权重，适合 24GB Windows ROCm 环境。

## 后果

- 两个功能默认关闭，且 v1 只支持 Anima 标准网格训练；Leap、NaViT 与 APT/区域路径按 schema
  fail-fast 互斥。
- APT 每步增加一次无梯度底模前向，计算时间接近增加一倍；冻结前向不会增加同等 activation
  显存。
- Anima 是 continuous rectified flow，增广发生在 latent 空间，因此命名为 APT-inspired，
  不宣称逐位复现 SDXL 论文。v1 未实现论文的中间表征统计稳定和 cross-attention alignment。
- `*.regions.json` 是训练数据，版本 fork、预处理变换和可选 bundle 导出都必须保留它。

## 参考

- [APT: Adaptive Personalized Training for Diffusion Models with Limited Data](https://openaccess.thecvf.com/content/CVPR2025/html/Chae_APT_Adaptive_Personalized_Training_for_Diffusion_Models_with_Limited_Data_CVPR_2025_paper.html)
- [Selectively Informative Description can Reduce Undesired Embedding Entanglements](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_Selectively_Informative_Description_can_Reduce_Undesired_Embedding_Entanglements_in_Text-to-Image_CVPR_2024_paper.html)
- [Break-A-Scene](https://arxiv.org/abs/2305.16311)
- [DreamBooth](https://arxiv.org/abs/2208.12242)
