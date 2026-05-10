# Anima LoRA 打标指南

> 基于 Anima Model Card 和 ComfyUI 官方实现总结的打标最佳实践

## 核心规则

### 1. 标签格式：使用空格，不用下划线

根据 Anima Model Card 官方示例和 ComfyUI 实现：

```
✅ 正确: oomuro sakurako, yuru yuri, brown hair, long hair
❌ 错误: oomuro_sakurako, yuru_yuri, brown_hair, long_hair
```

**原因**：ComfyUI 的 tokenizer 直接把文本传给 Qwen2Tokenizer/T5Tokenizer，不做下划线转换。Anima 训练数据使用空格分隔。

### 2. 标签顺序（官方推荐）

```
质量/安全 → 人数 → 角色 → 作品 → 画师 → 外观 → 标签 → 环境. 自然语言描述
```

| 位置 | 字段 | 示例 |
|------|------|------|
| 1 | quality | `newest, safe` |
| 2 | count | `1girl`, `2boys`, `no humans` |
| 3 | character | `hatsune miku` |
| 4 | series | `vocaloid` |
| 5 | artist | `@wlop` |
| 6 | appearance | `long hair, blue eyes, twintails` |
| 7 | tags | `smile, standing, looking at viewer` |
| 8 | environment | `concert stage, spotlight, crowd` |
| 9 | nl | `.` 句号后接自然语言描述 |

### 3. 画师标签必须带 `@` 前缀

```
✅ 正确: @wlop, @sakimichan, @torino aqua
❌ 错误: wlop, sakimichan, torino aqua
```

**重要**：没有 `@` 前缀的画师标签几乎不起作用！

### 4. 质量标签建议

训练 LoRA 时，**不建议**使用复杂质量标签，只保留：

```
newest, safe
```

让 LoRA 专注学习画风和角色，质量标签在推理时由用户自己添加。

**完整质量标签体系**（推理时使用）：
- 人工评分：`masterpiece` > `best quality` > `good quality` > `normal quality`
- 美学评分：`score_9` > `score_8` > ... > `score_1`
- 年份：`newest`, `recent`, `mid`, `early`, `old` 或 `year 2024`
- 安全：`safe`, `sensitive`, `nsfw`, `explicit`

---

## 角色变体命名

使用 **空格 + 括号** 表示变体：

| 变体类型 | 标签格式 |
|----------|----------|
| 基础角色 | `hatsune miku` |
| 特定服装 | `hatsune miku (racing)` |
| 年龄变体 | `hatsune miku (adult)` |
| 世界线/形态 | `hatsune miku (append)` |

---

## 固定字段 vs 动态字段

### 固定字段（根据项目/目录自动填充）

对于特定项目的 LoRA 训练，建议固定以下字段：

| 字段 | 说明 | 示例 |
|------|------|------|
| quality | 统一质量标签 | `newest, safe` |
| series | 作品/项目名 | `my project` |
| artist | 画师/画风标签 | `@my artist` |
| character | 角色名（可从目录映射） | `character a` |

### 动态字段（VLM 打标）

以下字段需要根据每张图片内容动态生成：

| 字段 | 描述 |
|------|------|
| count | 人物数量 (`1girl`, `2boys`, `no humans`) |
| appearance | 角色外观（发型、发色、瞳色、服装、配饰） |
| tags | 动作、表情、构图、手持物品 |
| environment | 背景、场景、光影、氛围 |
| nl | 1-2 句自然语言描述（放在最后，句号分隔） |

---

## VLM 打标

### System Prompt 模板

```
You are an anime image tagging expert. Output ONLY valid JSON.

JSON fields (tag fields are arrays of lowercase strings):
1. count: string - Character count ("1girl", "2boys", "1girl, 1boy", "no humans")
2. appearance: string[] - Visual features (hair color, eye color, hairstyle, clothing, accessories)
3. tags: string[] - Actions, expressions, poses, composition, objects
4. environment: string[] - Background, location, lighting, atmosphere
5. nl: string - One sentence natural language description

Rules:
- Use lowercase English booru-style tags
- Each tag is a separate array element
- Only describe what is clearly visible
- Be detailed but don't repeat tags
- Output ONLY the JSON object, no markdown or explanation

Example:
{"count": "1girl", "appearance": ["long hair", "blue eyes", "school uniform"], "tags": ["smile", "standing", "looking at viewer"], "environment": ["classroom", "window", "sunlight"], "nl": "A cheerful girl stands by the window in a sunny classroom."}
```

### API 调用参数（Gemini 推荐）

```python
{
    "generationConfig": {
        "temperature": 0.2,
        "topP": 0.8,
        "maxOutputTokens": 512,
        "thinkingConfig": {
            "thinkingBudget": 128
        }
    }
}
```

**关键参数**：
- `thinkingBudget: 128` - 限制思考 token，让输出更干净
- **不要**在 Prompt 中加 `SPECIAL INSTRUCTION` 或 `Danbooru` 等词，可能触发安全过滤

---

## 目录结构自动映射

### 创建角色映射

为你的项目创建角色映射字典：

```python
# 示例：角色目录名 → 英文标签
CHAR_MAP = {
    "角色A": "character a",
    "角色A-变体": "character a (variant)",
    "角色B": "character b",
    # ... 添加你的角色
}
```

### 变体/服装映射

子文件夹名可以自动解析并添加标签：

| 类型 | 目录名示例 | 英文 tag | 添加到 |
|------|-----------|----------|--------|
| **年龄** | `成年` | `(adult)` | 角色名后缀 |
| **发色** | `金发` | `blonde hair` | appearance |
| **服装** | `和服` | `kimono` | appearance |
| **服装** | `泳装` | `swimsuit` | appearance |
| **服装** | `校服` | `school uniform` | appearance |
| **状态** | `战斗` | `fighting stance` | tags |

**示例路径解析**：
- `角色A/金发-便装/xxx.png` → `character a, blonde hair, casual clothes`
- `角色B/和服/xxx.png` → `character b, kimono`

---

## 最终 Caption 示例

**输入图片**：`character/角色A/和服/001.png`

**VLM 输出（JSON 格式）**：
```json
{
  "count": "1girl",
  "appearance": ["long hair", "black hair", "red eyes", "hair ornament"],
  "tags": ["standing", "smile", "looking at viewer", "upper body"],
  "environment": ["indoors", "traditional room", "soft lighting"],
  "nl": "A graceful girl in traditional attire smiles warmly in a serene room."
}
```

**固定字段**：
```python
FIXED = {
    "quality": "newest, safe",
    "series": "my project",
    "artist": "@my artist",
}
```

**路径自动添加**：`kimono`（来自子文件夹 `和服`）

**最终 Caption**：
```
newest, safe, 1girl, character a, my project, @my artist, long hair, black hair, red eyes, hair ornament, kimono, standing, smile, looking at viewer, upper body, indoors, traditional room, soft lighting. A graceful girl in traditional attire smiles warmly in a serene room.
```

---

## 训练参数建议

### TXT 模式

```yaml
shuffle_caption: true   # 打乱标签
keep_tokens: 6          # 保护前 6 个 tag 不被打乱
                        # newest, safe, 1girl, character, series, artist
```

### JSON 模式（推荐）

```yaml
prefer_json: true       # 使用 JSON 文件
shuffle_caption: true   # 分类内部打乱（appearance/tags/environment）
keep_tokens: 0          # JSON 模式下固定字段自动在前
```

---

## 常见问题

### Q: 为什么画师标签不起作用？

A: 检查是否加了 `@` 前缀。`@wlop` 有效，`wlop` 无效。

### Q: 角色名用下划线还是空格？

A: **空格**。Anima 训练数据使用空格，下划线会被当作普通字符。

### Q: 需要加很多质量标签吗？

A: 训练时只用 `newest, safe`。推理时再加 `masterpiece, best quality` 等。

### Q: 自然语言描述放哪里？

A: 放在最后，用 `.`（句号）分隔：`..., environment tags. Natural language description here.`

---

## 参考资料

- [Anima Model Card](https://huggingface.co/circlestone-labs/Anima)
- [ComfyUI-AnimaTool Prompt Guide](https://github.com/Moeblack/ComfyUI-AnimaTool/wiki/Prompt-Guide)
- [ComfyUI 源码 - comfy/text_encoders/anima.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/text_encoders/anima.py)
