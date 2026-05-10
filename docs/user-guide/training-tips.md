# Anima LoRA 训练技巧

## 数据准备

### 数据量建议

| 场景 | 最少图片数 | 推荐图片数 | repeats |
|------|-----------|-----------|---------|
| 单角色 LoRA | 30 | 50-100 | 10-20 |
| 画风 LoRA | 50 | 100-300 | 5-10 |
| 多角色 LoKr | 200 | 500+ | 1-3 |

### 图片质量要求

- **分辨率**：建议 1024×1024 或更高
- **裁剪**：尽量保留完整构图，避免截断重要部位
- **多样性**：包含不同角度、表情、服装、光照
- **一致性**：如果是角色 LoRA，确保同一角色的外观一致

### 标签质量

- 使用 VLM 打标时，检查输出是否准确
- 删除明显错误的标签
- 保持标签风格一致（全小写，空格分隔）

---

## 参数调优

### 学习率

| 场景 | 推荐学习率 | 说明 |
|------|-----------|------|
| 小数据集 (<100 张) | 5e-5 ~ 1e-4 | 防止过拟合 |
| 中等数据集 (100-500 张) | 1e-4 ~ 2e-4 | 标准范围 |
| 大数据集 (500+ 张) | 1e-4 ~ 3e-4 | 可以激进一些 |

**调试技巧**：
- 如果 loss 下降太慢 → 提高学习率
- 如果 loss 震荡剧烈 → 降低学习率
- 如果过拟合（采样图变差）→ 降低学习率或减少 epoch

### LoRA Rank

| Rank | 参数量 | 适用场景 |
|------|--------|----------|
| 8 | ~1MB | 简单画风微调 |
| 16 | ~2MB | 画风 LoRA |
| 32 | ~4MB | 单角色 LoRA |
| 64 | ~8MB | 复杂角色/多角色 |
| 128 | ~16MB | 极复杂场景（很少用） |

**经验法则**：从低 rank 开始，如果效果不够再提高。

### LoRA vs LoKr

| 类型 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| LoRA | 简单稳定，兼容性好 | 表达力有限 | 单角色、简单画风 |
| LoKr | 表达力强，参数高效 | 需要调参 | 多角色、复杂画风 |

---

## 常见问题

### 过拟合

**症状**：
- 训练 loss 很低，但采样图质量下降
- 生成的图和训练集几乎一样
- 无法响应新的提示词变化

**解决方案**：
1. 减少 epochs
2. 降低学习率
3. 增加 tag_dropout（5-15%）
4. 降低 LoRA rank
5. 增加数据多样性

### 欠拟合

**症状**：
- 训练 loss 居高不下
- 采样图完全没有学到特征
- 角色/画风不像目标

**解决方案**：
1. 增加 epochs
2. 提高学习率
3. 提高 LoRA rank
4. 检查标签是否正确
5. 检查数据是否正确加载

### 角色崩坏

**症状**：
- 角色特征不稳定
- 有时正确有时错误
- 多角色混淆

**解决方案**：
1. 确保每个角色的标签一致
2. 增加角色名标签的权重（推理时）
3. 使用 keep_tokens 保护角色名
4. 增加训练数据

### 显存不足

**症状**：
- CUDA out of memory
- 训练中断

**解决方案**：
1. 启用 `grad_checkpoint: true`
2. 减小 `batch_size`（改用 `grad_accum` 补偿）
3. 降低 `resolution`
4. 关闭 `cache_latents`（会变慢）
5. 使用 `mixed_precision: bf16`

---

## 监控训练

### Loss 曲线解读

```
理想曲线：
  快速下降 → 缓慢下降 → 趋于平稳
  
过拟合曲线：
  快速下降 → 继续下降 → 非常低（接近 0）
  
欠拟合曲线：
  缓慢下降 → 停滞 → 居高不下
```

### 采样图检查

每隔几个 epoch 检查采样图：

1. **早期** (1-5 epoch)：应该开始出现目标特征的雏形
2. **中期** (5-15 epoch)：特征应该越来越明显
3. **后期** (15+ epoch)：质量应该稳定，注意过拟合

### 使用训练监控

走 Studio 的监控页：启动训练后打开 <http://127.0.0.1:8765/studio/tools/monitor>，
或在 ⑥ 训练 / 队列页里点任务进入 **任务详情 → 监控** 标签。

监控面板显示：
- 实时 loss 曲线
- 学习率变化
- 采样图预览
- 训练速度

> 旧的 `python train_monitor.py` 自带 HTTP server 已在 PP6.1 删除（详见
> `tools/train_monitor.py` 顶部 docstring）；现在它只是个状态写入器，由
> `anima_train` 调用，不需要单独启动。

---

## 最佳实践

### 训练前

1. ✅ 验证模型文件完整
   ```bash
   python tools/validate_local_models.py
   ```

2. ✅ 检查数据集
   - 图片是否正确加载
   - 标签文件是否存在
   - 标签格式是否正确

3. ✅ 小批量测试
   ```bash
   python scripts/anima_train.py --config config.yaml --epochs 3 --save_every 1
   ```

### 训练中

1. ✅ 监控 loss 曲线
2. ✅ 定期检查采样图
3. ✅ 保存多个 checkpoint（便于回退）

### 训练后

1. ✅ 在 ComfyUI 中测试
2. ✅ 测试不同提示词
3. ✅ 测试与其他 LoRA 的兼容性

---

## ComfyUI 使用

### 加载 LoRA

使用 `LoraLoader` 或 `LoraLoaderModelOnly` 节点：

```
模型路径：models/loras/my_lora.safetensors
strength_model: 0.8-1.0
strength_clip: 0.8-1.0
```

### 推荐参数

| 参数 | 推荐值 |
|------|--------|
| Steps | 25-50 |
| CFG | 4-5 |
| Sampler | er_sde |
| Scheduler | simple |

### 提示词格式

```
masterpiece, best quality, newest, safe, 
1girl, [角色名], [作品名], @[画师], 
[外观标签], [动作标签], [环境标签]
```

---

## 硬件优化

### RTX 3090/4090 (24GB)

```yaml
batch_size: 1
grad_accum: 4
resolution: 1024
grad_checkpoint: true
mixed_precision: "bf16"
cache_latents: true
```

### RTX 5090 (32GB)

```yaml
batch_size: 2
grad_accum: 2
resolution: 1024
grad_checkpoint: true
mixed_precision: "bf16"
xformers: false  # 用 PyTorch SDPA
cache_latents: true
```

### 多 GPU

目前脚本不支持多 GPU 并行，建议单卡训练。
