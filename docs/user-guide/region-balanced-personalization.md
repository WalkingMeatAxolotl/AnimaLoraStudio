# 区域平衡个性化训练

这套功能用于少量人物/主体图：先把有限梯度更多分配到关键主体区域，再平滑退回整图训练；
同时用选择性 caption 减少主体、服装、姿势和背景的错误绑定。它借鉴论文中的组件，但不是对
某个社区视频私有算法的复刻。

## 1. 用 Ollama JoyCaption 做 SID 标注

先确认 Ollama 已运行且模型名存在：

```powershell
ollama list
```

在「打标」页选择 `LLM`，再选择：

- `JoyCaption（Ollama 本地）`：生成普通自然语言 caption；
- `SID 主体解耦 JSON（Ollama JoyCaption）`：推荐人物 LoRA，输出可分类 shuffle 的 JSON。

内置预设使用 `http://localhost:11434/v1` 和模型名
`llama-joycaption-beta-one-hf-llava`。如果本机 Ollama 显示的名字带 tag（例如 `:latest`），
在预设编辑器里改成 `ollama list` 的准确名称。类别词填 `1girl`，触发词填唯一标识（本数据集
为 `yuemeng`）。worker 会把触发词放在 caption 第一位；类别词只提供给 SID 提示词，不能用
唯一触发词代替。

SID 预设把稳定、可见的身份特征放入 `appearance`，把服装、表情、姿势和构图放入 `tags`，
把场景、物件和光线放入 `environment`。打标后仍应在人审页删除模型猜测出的姓名、作品名、
不可见特征和重复词。

## 2. 标主区域

进入「预处理 → 主区域」，每张图拖一个矩形。人物 LoRA 通常框脸和标志性发型；不要把整件
固定服装和大面积背景一起框入。每张图只有一个主框，保存为同目录的
`{图片名}.regions.json`。坐标是 0–1 归一化值，普通 resize 会保留；crop 会取相交区域并
重新归一化；水平翻转和 latent cache 会同步变换。

主区域与涂抹页的 Mask 不同：主区域是前期的正权重，Mask 是始终不学习的区域。两者可同时用。
没有主区域的图片自动按整图训练，不会报错。

## 3. 推荐训练开关

Train 页 Loss 组启用：

```yaml
region_balance_enabled: true
region_max_weight: 3.0
region_hold_ratio: 0.45
region_end_ratio: 0.55
apt_enabled: true
apt_identifier_token: yuemeng
apt_class_token: 1girl
apt_bins: 10
apt_ema_alpha: 0.1
apt_p_max: 0.8
apt_zoom_max: 3.0
apt_rotation_degrees: 15.0
```

前 45% 主区域保持最高权重，45%–55% 余弦降到零，之后严格使用整图 loss。APT 会增加一次冻结
底模参考前向，训练明显变慢；先用 256px、`max_steps: 1` 确认链路，再做正式实验。

若有合适的同类别正则图，在 Dataset 组设置 `reg_data_dir`、`reg_caption: 1girl` 和
`reg_weight`（常从 0.5–1.0 试起），这是项目现有的 DreamBooth 式 prior preservation 路径。
不要用训练主体的近重复图充当正则集。

## 4. A/B 评估

固定 5 张不参与训练的图片和至少三类 prompt：近训练构图、新姿势/新服装、复杂背景。用同一 seed、
采样器、步数和 LoRA 强度比较：baseline、仅区域平衡、区域平衡+APT。关注身份相似度、提示词遵循、
背景多样性和构图崩坏，而不是只看训练 loss。

`examples/rocm/anima-yuemeng-region-balance.example.yaml` 给出本机路径的可复制配置。原始
`D:\5_yuemeng3_V01\5_yuemeng3_V01` 不应直接写 sidecar；先导入 Studio 或复制到独立实验目录。

## 限制

- 当前只支持 Anima 标准训练路径，不支持 NaViT/Leap；APT 还与 InfoNoise 互斥。
- APT-inspired v1 实现自适应调整部分，未实现论文的中间层表征稳定和 cross-attention alignment。
- 主区域 caption 当前是标注审阅元数据；训练 conditioning 仍使用图片的完整 TXT/JSON caption。
