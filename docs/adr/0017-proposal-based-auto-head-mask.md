# 0017 — 用提案式 ONNX 检测生成可撤销的头部空间遮罩

**状态**：Accepted  
**日期**：2026-09-06  
**决策者**：AnimaLoraStudio maintainers

## 背景

服装、姿势或画风 LoRA 的训练集常含多个卡通人物。用户需要让 masked loss 忽略人物头部，避免模型把脸、发色、耳朵或头饰等身份特征一起学入。手工逐图涂遮罩准确但成本高；直接由检测器覆盖 `.mask` 又会破坏用户已经完成的手工区域。

## 候选方案

1. 在训练时动态做人脸检测。配置简单，但结果不可预览、难以复现，也把外部模型耦合进训练热路径。
2. 直接调用 `dghs-imgutils` 或 Ultralytics。开发量较小，但会引入 OpenCV 与额外推理栈，和现有 ONNX Runtime 重复。
3. 预处理阶段用现有 ONNX Runtime 生成提案，用户确认后合并 `.mask`。多一步确认，但结果可审查、可复用，训练侧无需修改。

## 决策

采用方案 3：

- 固定使用 `deepghs/anime_head_detection` 的 `head_detect_v2.0_s/model.onnx` revision `06604feee81983792a57c21081e539c0ae229833`，输入 640×640。
- 权重由下载中心按需获取，不随程序分发；下载后同时校验 44,585,386 字节和 SHA-256 `6679f9b71192298bbf174d82e9e5581c3237b0c3dc67deace7cdbf686b070a00`。
- 推理沿用 ONNX Runtime，执行提供程序顺序为 CUDA、DirectML、CPU；CUDA 推理失败时按任务回退 CPU。
- 检测只生成位于 `studio_data/tasks/<job_id>/head-mask/result.json` 的提案。用户可取消单个区域，然后通过 `pixelwise min(existing_mask, auto_mask)` 合并已有 `.mask`。
- 应用前验证所有源图快照并生成全部临时文件；全部成功后才原子替换。任务目录保存应用前快照和应用后摘要，用于一次受保护撤销。
- 源图在检测后改变，或应用后的遮罩被手工修改时，拒绝对应应用或撤销。

## 理由

提案与训练解耦让误检能在写盘前修正，也保证检测模型更新不会改变已经确认的训练数据。复用现有运行时减少安装体积和 Windows 环境冲突。最小值合并符合 `.mask` 的既有语义：255 参与学习，0 忽略，因此自动区域不会恢复任何手工忽略区域。

## 后果

- 检测结果需要额外的用户确认步骤，并占用任务目录保存提案与撤销快照。
- v1 只处理像素空间；caption 中的人名、眼睛、发色等标签不会自动删除，用户仍须在打标页处理。
- 训练时必须启用 masked loss。训练页在发现 `.mask` 且该选项关闭时提示并提供一键启用，同时保留 Leap / NaViT Packing 的互斥校验。
- 模型许可可能变化；分发前必须重新核对。按需下载降低再分发风险，但不替用户判断用途是否合规。

## 参考

- [anime_head_detection 模型与指标](https://huggingface.co/deepghs/anime_head_detection/commit/1b784b384f81b5018bc0516536af1adba7e95b68)
- [imgutils 商业再分发许可讨论](https://github.com/deepghs/imgutils/issues/179)
- [自动头部遮罩用户指南](../user-guide/auto-head-mask.md)
