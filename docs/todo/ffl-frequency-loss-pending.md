# ① 频域 loss（FFL）—— 已实现，待判（未合入）

**状态**：⏸️ 实现完成并提交在 `feat/freq-loss-ffl`（commit `785b2fc6`），**未合 dev/master**。合入门槛未达，去留决定排在 `timestep_shift` 实验之后。
**日期**：2026-07-08
**相关**：论文 arXiv:2012.12821（Focal Frequency Loss, Jiang et al. ICCV 2021）；`dit-lora-detail-texture-notes.md` 方案①；LPL 证伪记录 `docs/todo/lpl-gate0-null.md`（同源「细节质感」调研的姊妹项）

> FFL 在 latent 空间对预测的干净 latent x̂₀ 与真实 x₀ 做频率域 focal 对齐，自适应加权「难合成/高频」频率，弥补 MSE 磨平高频细节。与 SRA 同构（附加项、零可训练参数、标准路径专属、训完丢弃），纯 latent、无 VAE decode、≈0 显存——是 LPL 证伪后更便宜的对照臂（FINAL 落地顺序 P0）。

## 已交付（都在 `feat/freq-loss-ffl`）

- `runtime/training/ffl.py`：`FocalFrequencyLoss`（公式照官方 EndlessSora 实现）。
- loop / context / phases/models / schema（4 字段 `ffl_enabled/weight/alpha/t_threshold`）集成，经 argparse bridge 自动流到 args。
- `tests/test_ffl.py`：9 测（官方 stack 风格独立参考交叉校验、degenerate、grad、schema）全绿；loop 集成块 + 全链路验证过。

## 为什么没合：一次真 A/B 是 null

38-kupa 完整 20 epoch @ RTX 5090，`ffl_weight=1.0`（默认）：**采样出图无可见变化。** 诊断（数据支持，非玄学）：

- **量级太弱**：实测 ffl 值多 0.002–0.015 vs denoise 0.05–0.2 → ffl/denoise 中位 ~7%；且 **~40% 的步整批是 reg（reg 单独分桶）FFL 零贡献**。加进 loss 的平均贡献个位数百分比。
- **x̂₀ 的 t² 抑制（结构性，非调参问题）**：`x̂₀ − x₀ = t·(target − pred)` → FFL 量级与梯度 ∝ t²。**低 t（细节区）FFL≈0，信号偏高 t（结构区），与目标相反。** 后果：把 `ffl_weight` 拉大，放大的是高 t 结构信号，不是低 t 细节——「weight=8 就能救」这个赌注本身可疑，可能需 v 空间重构（FFL 作用 pred vs target，去 t² 但语义变追噪声）。
- **latent 高频是小目标**：`tmp/lpl/latent_spectrum_probe.py` 实测——83% 能量在 DC，高频半区仅 4.3%（模糊抹掉 43%，说明有细节但占比小）。

**非 bug，FFL 忠实但太温柔。** 与 LPL 的区别：LPL 被 Gate-0 **证伪**（前提不成立）；FFL 只是**未充分测试**（只测了 weight=1.0），不删。

## 合入门槛（满足才经 dev 合入）

一次像样的 A/B 赢，且**前提是先修好采样**：

1. **先做 `timestep_shift` 实验**（见下），让低 t 被正常采到——现在 40% 空步 + 高 t 偏置就是「细节区没样本」的证据，单独判 FFL 不公平（它一直在没被喂到的低 t 上空转）。
2. 然后测 FFL **叠在改好的采样之上**（`ffl_weight=8` 或 v 空间版）。
3. **赢 → 经 dev 合入；仍 null → 和 LPL 一起 park**（本文改状态为「park」）。

## 关联：`timestep_shift` 默认偏高（更高优先级的头号根因）

修「细节差」的头号杠杆不是加 loss 项，是根因 A（低 t 被采样饿死）。已核实（2026-07-08）：

- 我们 `timestep_shift` 默认 **3.0**（logit_normal + Möbius shift，t=1 是噪声）→ 采样中位 t≈**0.75**，重偏高噪声/粗结构。方向已在 `timestep_sampling.py:70` + loop noisy 公式核实：>1 高噪声、<1 细节；1.0→中位 0.5、0.7→0.41、0.5→0.33。
- **kohya 自己的 Anima 训练器默认 `sigmoid`+`discrete_flow_shift=1.0`（中性）——我们比参考 Anima 更偏高噪声。** Flux/Wan 默认 3–7（SD3 分辨率相关 shift，为对齐底模 SNR/整体保真，非细节优化）。
- **下一步 A/B（零算力、比 FFL 对症）**：baseline(`timestep_shift=3.0`) vs `0.7`，一次只动这一个数；有效但不够可叠容量（`lokr_factor=4` / `lora_rank=32`）。

## 一句话给未来的自己

先跑 `timestep_shift=0.7` 那版。若它就解决细节 → FFL 大概率直接 park；若有效但不够 → 拿 FFL 叠上去测,那时才有公平的合入判据。别在 timestep_shift 之前单独给 FFL 判死或合入。
