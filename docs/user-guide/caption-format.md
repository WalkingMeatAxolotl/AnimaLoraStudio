# JSON Caption 格式规范

AnimaLoraStudio 支持结构化的 JSON 标签文件，相比传统 TXT 文件有以下优势：

- **分类 Shuffle**：appearance/tags/environment 各自内部打乱，保持语义结构
- **固定字段**：quality/character/series/artist 始终在前，不被打乱
- **易于管理**：结构化数据便于批量修改和版本控制

## 文件结构

```
dataset/
├── image001.jpg
├── image001.json    # 与图片同名的 JSON 文件
├── image002.png
├── image002.json
└── ...
```

## JSON Schema

### 完整格式

```json
{
  "fixed": {
    "quality": "newest, safe",
    "series": "project name",
    "artist": "@artist name"
  },
  "character": {
    "name": "character name",
    "variant": ""
  },
  "from_path": {
    "appearance": ["blonde hair", "casual clothes"]
  },
  "ai_output": {
    "count": "1girl",
    "appearance": ["long hair", "blue eyes", "smile"],
    "tags": ["standing", "looking at viewer", "upper body"],
    "environment": ["outdoors", "sky", "sunlight"],
    "nl": "A cheerful girl stands under the bright sky."
  }
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `fixed.quality` | string | 质量标签，固定在最前 |
| `fixed.series` | string | 作品/项目名 |
| `fixed.artist` | string | 画师标签（必须带 @） |
| `character.name` | string | 角色名 |
| `character.variant` | string | 角色变体（如 adult, alternate costume） |
| `from_path.appearance` | string[] | 从目录路径自动提取的外观标签 |
| `ai_output.count` | string | VLM 识别的人物数量 |
| `ai_output.appearance` | string[] | VLM 识别的外观特征 |
| `ai_output.tags` | string[] | VLM 识别的动作/表情/构图 |
| `ai_output.environment` | string[] | VLM 识别的环境/背景 |
| `ai_output.nl` | string | 自然语言描述 |

## 简化格式

如果不需要复杂的分层，可以使用简化格式：

```json
{
  "quality": "newest, safe",
  "count": "1girl",
  "character": "hatsune miku",
  "series": "vocaloid",
  "artist": "@wlop",
  "appearance": ["long hair", "blue hair", "twintails", "blue eyes"],
  "tags": ["singing", "microphone", "concert", "dynamic pose"],
  "environment": ["stage", "spotlight", "crowd", "night"],
  "nl": "Miku performs energetically on stage."
}
```

## 渲染顺序

JSON 会按以下顺序渲染为最终 caption：

```
quality → count → character → series → artist → appearance → tags → environment. nl
```

**示例输出**：
```
newest, safe, 1girl, hatsune miku, vocaloid, @wlop, long hair, blue hair, twintails, blue eyes, singing, microphone, concert, dynamic pose, stage, spotlight, crowd, night. Miku performs energetically on stage.
```

## 分类 Shuffle

启用 `shuffle_caption: true` 时：

| 字段 | 是否打乱 |
|------|----------|
| quality | ❌ 固定 |
| count | ❌ 固定 |
| character | ❌ 固定 |
| series | ❌ 固定 |
| artist | ❌ 固定 |
| appearance | ✅ 内部打乱 |
| tags | ✅ 内部打乱 |
| environment | ✅ 内部打乱 |
| nl | ❌ 固定在最后 |

**打乱示例**：
```
# 原始
appearance: ["long hair", "blue eyes", "school uniform"]
tags: ["smile", "standing", "looking at viewer"]

# 打乱后（示例）
appearance: ["blue eyes", "school uniform", "long hair"]
tags: ["looking at viewer", "smile", "standing"]
```

## 如何生成

落盘格式跟着打标产物走，不再单独选择输出格式：

- **LLM 打标器 + JSON 输出预设**（如 `general_json` / `style_json`）→ 生成符合此格式的 `.json` 文件
- **本地打标器**（wd14 / CLTagger）与 **LLM 文本预设** → 生成 `.txt` 文件（触发词 prepend 在第一位）
- 图片已有 `.json` 时，重新打标会更新其中的 `tags` 数组并保留其余字段（含 `meta.trigger`），不改变文件格式

生成的 JSON 包含：
- 从目录结构提取的 character/variant
- LLM 打标的 count/appearance/tags/environment/nl
- 配置中的 fixed 字段

### Legacy 扁平格式

早期版本的本地打标器可选输出 `{"tags": ["tag1", "tag2", ...], "meta": {"trigger": "..."}}`
形式的扁平 JSON（tags 为列表而非分类对象）。该格式**读取永久兼容**——训练时
`meta.trigger` 在第一位、其余 tags 作为可打乱区处理——但不再新生成。

## 配置示例

```yaml
# config/my_training.yaml

data_dir: "./dataset"
prefer_json: true        # 优先使用 JSON 文件
shuffle_caption: true    # 启用分类 shuffle
keep_tokens: 0           # JSON 模式下不需要
tag_dropout: 0.05        # 可选：5% 标签随机丢弃
```

## 回退机制

同名 caption 文件按扩展名取优先级（文件级，扫描时决定）：

```
优先级: image001.json > image001.txt
```

注意：优先级是文件级的（扫描时按扩展名决定）——选中 `.json` 后不会再逐样本
回退同名 `.txt`。

TXT 模式下 `tag_dropout` 同样生效：`keep_tokens` 前缀不参与打乱与 dropout，
其余 tag 逐个独立丢弃（与 kohya 语义一致）。

## 迁移指南

### 从 TXT 迁移到 JSON

1. 在 Studio 打标页用 LLM 打标器、选 JSON 输出预设重新打标
2. 或手动转换：

```python
# txt_to_json.py
import json
from pathlib import Path

def convert(txt_path):
    tags = txt_path.read_text().strip()
    # 简单解析（假设已按顺序排列）
    parts = [t.strip() for t in tags.split(",")]
    
    json_data = {
        "quality": "newest, safe",
        "count": parts[2] if len(parts) > 2 else "1girl",
        "character": parts[3] if len(parts) > 3 else "",
        "series": parts[4] if len(parts) > 4 else "",
        "artist": parts[5] if len(parts) > 5 else "",
        "appearance": parts[6:10] if len(parts) > 6 else [],
        "tags": parts[10:15] if len(parts) > 10 else [],
        "environment": parts[15:] if len(parts) > 15 else [],
        "nl": ""
    }
    
    json_path = txt_path.with_suffix(".json")
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2))

# 批量转换
for txt in Path("./dataset").glob("*.txt"):
    convert(txt)
```

## 验证工具

检查 JSON 文件是否符合格式：

```python
# validate_json.py
import json
from pathlib import Path

REQUIRED_FIELDS = ["count"]
ARRAY_FIELDS = ["appearance", "tags", "environment"]

def validate(json_path):
    data = json.loads(json_path.read_text())
    
    # 检查必需字段
    for field in REQUIRED_FIELDS:
        if field not in data and field not in data.get("ai_output", {}):
            print(f"Warning: {json_path} missing {field}")
    
    # 检查数组字段
    for field in ARRAY_FIELDS:
        value = data.get(field) or data.get("ai_output", {}).get(field)
        if value and not isinstance(value, list):
            print(f"Warning: {json_path} {field} should be array")
    
    return True

for json_file in Path("./dataset").glob("*.json"):
    validate(json_file)
```
