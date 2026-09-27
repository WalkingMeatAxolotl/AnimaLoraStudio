# Anima LoRA 打标指南

> 参考 Anima Model Card 和 ComfyUI 实现整理的格式约定与起步建议；格式一致不等于训练效果已获验证。
>
> **适用范围**：本文的 booru tag 体系（tag 顺序 / 质量标签 / `@画师` 前缀等）是 **Anima 专属**。
> Krea 2 走 Qwen3-VL 自然语言 caption——打标推荐用 LLM 打标器写长描述，tag
> shuffle / keep_tokens / tag_dropout 等 tag 生态操作对 Krea 2 不适用（UI 已按族隐藏）；
> 触发词机制两族通用。

## 核心规则

### 1. 标签格式：使用空格，不用下划线

根据 Anima Model Card 官方示例和 ComfyUI 实现：

```
✅ 正确: oomuro sakurako, yuru yuri, brown hair, long hair
❌ 错误: oomuro_sakurako, yuru_yuri, brown_hair, long_hair
```

**建议**：采用空格形式并保持训练/推理一致，不依赖 tokenizer 自动把下划线变成空格。示例说明推荐写法，不意味着使用下划线就必然无法学习。

### 2. 标签组织顺序

以下为分类 caption 的组织方式；有 `meta.trigger` 时，触发词会在最前。普通 TXT 或标签编辑后的扁平列表，以最终实际文本顺序为准。

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

### 3. 画师标签使用 `@` 前缀

```
✅ 正确: @wlop, @sakimichan, @torino aqua
❌ 错误: wlop, sakimichan, torino aqua
```

使用 `@` 是这里采用的画师标签约定；具体响应仍取决于底模是否学过该标签，不能保证任意画师名有效，也不应宣称无前缀一定无效。

### 4. 质量标签建议

质量标签按素材实际情况填写，避免堆叠互相矛盾的评分。对符合条件的较新、安全内容，可用以下简洁示例起步；**不要把所有图片都标成 `safe` 或 `newest`**：

```
newest, safe
```

是否保留更多质量信息应做对照，不能由固定两个标签保证更专注或更高质量。推理时也需与训练约定相符。

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

### API 调用参数示例（按具体 Gemini 模型核对）

这是调用结构示例，不是通用最佳参数。不同模型版本可能不支持相同的 thinking 参数或取值；先小范围检查输出完整性，再调整预算与温度。

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

**注意**：输出预算过低可能截断JSON；thinking预算不保证标签更准确。遵守服务的内容政策，如遇拒绝或参数不支持，应查看错误与所用模型文档，而不是依靠特定词语规避过滤。

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

下例没有额外前置触发词，前六个逗号分隔tag恰好是所列六项。`keep_tokens` 按最终文本位置保护前缀，不按语义字段识别；若prepend了trigger或某字段含多个tag，重新数实际项数，不要一律照抄6或改成7。

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

A: 先核对是否使用 `@` 约定，以及底模是否能识别该画师。固定seed比较有无标签，不能仅凭前缀断定效果；自定义触发词还需在训练caption中一致使用。

### Q: 角色名用下划线还是空格？

A: 本指南建议用**空格**，并保持训练/推理一致；不要依赖自动下划线转换。

### Q: 需要加很多质量标签吗？

A: 不必堆叠；按实际素材填写。`newest, safe` 只是适用内容的示例，不是所有图片的强制值。

### Q: 自然语言描述放哪里？

A: 放在最后，用 `.`（句号）分隔：`..., environment tags. Natural language description here.`

---

## 参考资料

- [Anima Model Card](https://huggingface.co/circlestone-labs/Anima)
- [ComfyUI-AnimaTool Prompt Guide](https://github.com/Moeblack/ComfyUI-AnimaTool/wiki/Prompt-Guide)
- [ComfyUI 源码 - comfy/text_encoders/anima.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/text_encoders/anima.py)
