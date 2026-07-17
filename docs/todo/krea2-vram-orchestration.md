# Krea2 生成显存编排：按需让位 + prompt 缓存 + 显存策略配置

- 状态：方案已拍板（2026-07-16），待实施——排在 fp8 parity 真卡验收之后，独立 PR
- 前置：codex/krea2-fp8-inference 分支（fp8 推理 + LoRA merge + TE offload 现状）

## 动机（真卡实测 + 讨论结论）

现状编排是**无条件固定**的：加载时 DiT/VAE/TE 全上 GPU → 编码 → TE 无条件卸
CPU → 采样。三个问题：

1. **fp8 + 32GB**：三者同驻仅 ~24GB，无显存压力，无条件卸 TE = 每个 prompt
   白付 8.9GB×2 的 PCIe 搬运（~2-4s）。
2. **16GB 卡跑不了 fp8**：fp8 DiT 13.7GB 单独装得下，但固定编排在加载期
   就三者同驻 OOM。ComfyUI 16GB 能跑，靠的是 free_memory **按需**让位。
3. **bf16 + 32GB**：加载/首编码期 35GB 瞬时超额，WDDM 换页颠簸（现状 TE
   offload 只保住了采样期稳态，见 bad6a480）。

核心认识：ComfyUI 的 free_memory 是按需 LRU——显存装得下就同驻（快），装不
下才让位（能跑）。我们把它做成了无条件，两头都不讨好。

## 决策

### D1 按需让位编排（comfy free_memory 轻量版）

- 编码前：free VRAM < TE 上卡所需（~11GB 含 embed cast fp32 瞬时）→ DiT 先
  撤 CPU；否则 TE 直接上（DiT 留着）。
- 采样前：free VRAM < 采样余量 → 卸 TE；否则不卸（32GB fp8 从此零搬运）。
- **判据用粗估不做记账**：采样余量 = activation 估算（分辨率相关，1024²
  约 3-4GB）+ VAE 缓冲，留 5GB 安全边际。不复刻 comfy 的 per-model 显存
  记账（复杂度即 bug 面）。用户已确认接受粗估。
- TE 初始 device 改 CPU-lazy（加载到 CPU，首次 ensure 才上 GPU）——16GB
  卡加载期不再 OOM。
- RAM 下限：DiT + TE 同驻系统 RAM ≈ 23GB，机器需 32GB+ RAM；不满足时报错
  文案写清（DomainError 规范）。

### D2 prompt conditioning LRU 缓存（无条件开，不做配置）

- key = caption 文本；缓存 **pad 之前的 per-caption varlen context**
  （`_encode_many` 查询层），一条 ≈30MB（512×12×2560 fp16），LRU 8-16 条。
- 命中时 TE 完全不动——按需编排下省掉整个 TE↔DiT 交换回合；测 LoRA 固定
  prompt 刷 seed / 换 scale 的主场景每图省 2-3s。
- 这也是 Comfy 既有行为（conditioning 节点输出缓存），属 parity 语义。
- 无 tradeoff（正确性由 key 完备性保证）→ 不进配置。

### D3 配置项：单个「显存策略」三档（对齐 vae_tiling 先例）

- **auto**（默认）：D1 按需判据
- **省显存**：强制顺序化 load TE→encode→unload TE→load DiT（auto 对碎片
  误判的逃生门 + 想留显存给其他程序的用户）
- **性能优先**：强制全同驻，禁止一切 offload
- schema/Settings/i18n 抄 vae_tiling 模板；描述按用户视角写（写显存/速度
  差别，不写指令型建议）。

## 显存账（fp8 Turbo 官方文件）

| 编排 | 编码期 | 采样期 | 峰值 |
|---|---|---|---|
| 现状（32GB） | 13.7+8.9+VAE+瞬时 ≈24GB | ~14GB | 24GB |
| auto @32GB | 同上（不让位，零搬运） | ~24GB（TE 不卸） | 24GB |
| auto @16GB（=顺序化） | 8.9+瞬时 ≈11GB | 13.7+act ≈16-17GB | ~17GB |

16GB 卡 1024² 贴边可跑——PR 卖点：解锁 16GB 卡 fp8 krea2 生成。

## 与 parity 的关系

编排不影响数值（出图内容与模型在哪个 device 无关）——故排在 parity 验收
之后：验收环境越接近现状越好归因。

## 实施面（single PR）

daemon/family.sample_image 编排触点 + TE 初始 device + Krea2TextStack 在线
LRU + 配置三件套（schema/Settings/i18n）+ 测试（判据分支/缓存命中/三档）。
