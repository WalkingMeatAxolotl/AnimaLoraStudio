"""Built-in LLM caption preset loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# PR-7：本文件搬到 studio/infrastructure/ 后，json 数据资源仍留 studio/llm_presets/
# （不是 python module），所以 parent 要上跳一层到 studio/。
PRESETS_DIR = Path(__file__).resolve().parent.parent / "llm_presets"
BUILTIN_PRESET_ORDER = (
    "style_json",
    "general_json",
    "txt_tags",
    "joycaption",
    "assist_json",
    "assist_text",
)


def builtin_llm_presets() -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in PRESETS_DIR.glob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        raw["builtin"] = True
        preset_id = str(raw.get("id") or path.stem).strip()
        if not preset_id:
            continue
        raw["id"] = preset_id
        by_id[preset_id] = raw

    items: list[dict[str, Any]] = []
    for preset_id in BUILTIN_PRESET_ORDER:
        item = by_id.pop(preset_id, None)
        if item is not None:
            items.append(item)
    items.extend(item for _, item in sorted(by_id.items(), key=lambda pair: pair[0]))
    return items
