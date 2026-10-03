# JSON Caption 格式规范

> **适用范围**：分类 JSON caption（fixed / character / series / artist + 分类 shuffle）服务
> Anima 的 booru tag 生态。Krea 2 用自然语言 caption（程序不按512 token截断，仍受上下文与资源限制），
> 一般直接用 TXT / 扁平文本即可，不需要本格式的分类结构。

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
| `meta.trigger` | string | 可选触发词，渲染时放在最前 |
| `fixed.quality` | string | 质量标签，位于可选触发词之后 |
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

## 标签编辑器保存后的优先级

标签编辑器会在完整格式中写入顶层扁平 `tags` 列表，并保留 `ai_output` / `fixed` / `from_path` 等原始结构作为来源信息。此时**顶层列表代表用户确认后的全部标签**：重新读取时不会把旧结构中的已删除标签合并回来，自然语言 `nl` 仍保留。扁平编辑后的标签不再沿用旧分类字段的 shuffle 分组。

这个优先级只适用于同时含上述完整格式标记的编辑结果。下方简化格式中的 `tags` 仍是与 `appearance`、`environment` 并列的一个分类，并不覆盖其他分类。

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

未被顶层扁平列表覆盖的分类 JSON，按以下顺序渲染为最终 caption：

```
可选 meta.trigger → quality → count → character → series → artist → appearance → tags → environment. nl
```

触发词缺省时直接从quality开始。标签编辑后的完整格式若含顶层扁平`tags`，遵循上面的编辑优先级，不再恢复旧分类顺序。

**示例输出（未设置 `meta.trigger`）**：
```
newest, safe, 1girl, hatsune miku, vocaloid, @wlop, long hair, blue hair, twintails, blue eyes, singing, microphone, concert, dynamic pose, stage, spotlight, crowd, night. Miku performs energetically on stage.
```

## 分类 Shuffle

启用 `shuffle_caption: true` 时：

| 字段 | 是否打乱 |
|------|----------|
| meta.trigger（若有） | ❌ 固定在最前 |
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
- 图片已有 `.json` 且选择了实际重打标时，会更新其中的 `tags` 数组并保留其余字段（含 `meta.trigger`），不改变文件格式

页面默认跳过已有caption，因此跳过的文件也不会补写或替换触发词。导入人工caption后，先在标签编辑检查trigger；不要为补trigger无意覆盖整份描述。

“已有 caption”按**是否能渲染出有效训练文本**判断，不是只看同名文件是否存在。空白、无有效内容或无法解析的 caption 会在“跳过已有”模式下重新生成；仅有自然语言描述或 `meta.trigger` 的结构化 JSON 仍属于有效 caption。历史遗留的空文件不需要先手工删除。

在标签编辑中把最后一个可编辑标签清空后，改动仍先暂存在页面。点击页面级“保存”时，Studio 会自动创建还原点，将空内容写回 sidecar，**不删除文件**，也不增加删除确认。未保存时磁盘文件与打标覆盖率不变，可以继续编辑恢复内容。

图片是否参与训练由选定训练集的扫描范围决定，与是否已打标无关。缺失或空白 TXT 都视为“未打标”，但图片仍以空文本 `""` 参与训练；“跳过已有”会为这些图片生成新 Caption。仅浏览没有 Caption 的图片不会自动补写空文件。

JSON 保持文件级优先级：空顶层标签列表不会恢复完整格式中的旧标签，也不会回退同名 TXT；未编辑的自然语言描述与 `meta.trigger` 保留，若仍有有效训练文本则继续视为已打标。因此“清空标签”不等于“清空完整 JSON Caption”。合法但无有效内容的 JSON 使用空文本训练，无法解析的首选 JSON 仍会在训练前明确报错，不能静默视为空内容。

覆盖率仅表示有有效 Caption 的图片数量，不要求 100% 才能推进打标/编辑阶段。桶分布、样本数和 NaViT 步数预估均包含无 Caption 图片；训练日志汇总使用空文本的源图片与展开样本数量，不包含随机标签丢弃导致的空文本。显式 `caption_override` 同样适用于没有 sidecar 的图片。

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
其余 tag 逐个独立丢弃（与 kohya 语义一致）。`keep_tokens` 计数按最终文本的逗号分隔项，前置触发词也占位置；不要把“前N项”理解成固定保护某N个语义字段。

## 迁移指南

### 从 TXT 迁移到 JSON

1. 在 Studio 打标页用 LLM 打标器、选 JSON 输出预设重新打标
2. 或手动转换（下例仅演示**无额外前置trigger、字段位置固定**的TXT；先备份并抽样检查。若字段缺失、顺序不同或有trigger，需要改映射，不能直接批量套用）：

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
