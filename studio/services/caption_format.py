"""Caption JSON helpers shared by taggers, workers, and editor views."""
from __future__ import annotations

from typing import Any


def split_tags(value: Any) -> list[str]:
    """Normalize a string/list-ish value to a deduped tag list."""
    if value is None:
        raw: list[Any] = []
    elif isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raw = [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item or "").strip().strip("#")
        if not text:
            continue
        text = " ".join(text.split())
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(split_tags(value))
    return " ".join(str(value).strip().split())


def _character_full(value: Any) -> str:
    if isinstance(value, dict):
        full = _as_text(value.get("full"))
        if full:
            return full
        name = _as_text(value.get("name"))
        variant = _as_text(value.get("variant"))
        return ", ".join(t for t in (name, variant) if t)
    return _as_text(value)


def normalize_caption_json(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Return the repo's train-time standard JSON shape.

    Accepted inputs:
    - standard shape: {"meta": ..., "tags": {...}}
    - documented full shape: {"fixed": ..., "character": ..., "ai_output": ...}
    - documented simplified shape: {"quality": ..., "appearance": [...], ...}
    - legacy simplified editor shape: {"tags": ["tag", ...]}
    """
    data = raw if isinstance(raw, dict) else {}
    tags_obj = data.get("tags")
    if isinstance(tags_obj, dict):
        return {
            "meta": data.get("meta") if isinstance(data.get("meta"), dict) else {},
            "tags": {
                "quality": split_tags(tags_obj.get("quality")),
                "count": _as_text(tags_obj.get("count")),
                "character": _as_text(tags_obj.get("character")),
                "series": _as_text(tags_obj.get("series")),
                "artist": _as_text(tags_obj.get("artist")),
                "appearance": split_tags(tags_obj.get("appearance")),
                "tags": split_tags(tags_obj.get("tags")),
                "environment": split_tags(tags_obj.get("environment")),
                "nl": _as_text(tags_obj.get("nl")),
            },
        }

    if "ai_output" in data or "fixed" in data or "from_path" in data:
        fixed = data.get("fixed") if isinstance(data.get("fixed"), dict) else {}
        character = data.get("character") if isinstance(data.get("character"), dict) else data.get("character")
        ai = data.get("ai_output") if isinstance(data.get("ai_output"), dict) else {}
        from_path = data.get("from_path") if isinstance(data.get("from_path"), dict) else {}
        appearance = split_tags(ai.get("appearance"))
        appearance.extend(t for t in split_tags(from_path.get("appearance")) if t.lower() not in {x.lower() for x in appearance})
        appearance.extend(t for t in split_tags(from_path.get("extra_appearance")) if t.lower() not in {x.lower() for x in appearance})
        tags = split_tags(ai.get("tags"))
        tags.extend(t for t in split_tags(from_path.get("tags")) if t.lower() not in {x.lower() for x in tags})
        tags.extend(t for t in split_tags(from_path.get("extra_tags")) if t.lower() not in {x.lower() for x in tags})
        return {
            "meta": data.get("meta") if isinstance(data.get("meta"), dict) else {},
            "tags": {
                "quality": split_tags(fixed.get("quality")),
                "count": _as_text(ai.get("count")),
                "character": _character_full(character),
                "series": _as_text(fixed.get("series")),
                "artist": _as_text(fixed.get("artist")),
                "appearance": split_tags(appearance),
                "tags": split_tags(tags),
                "environment": split_tags(ai.get("environment")),
                "nl": _as_text(ai.get("nl")),
            },
        }

    return {
        "meta": data.get("meta") if isinstance(data.get("meta"), dict) else {},
        "tags": {
            "quality": split_tags(data.get("quality")),
            "count": _as_text(data.get("count")),
            "character": _as_text(data.get("character")),
            "series": _as_text(data.get("series")),
            "artist": _as_text(data.get("artist")),
            "appearance": split_tags(data.get("appearance")),
            "tags": split_tags(tags_obj),
            "environment": split_tags(data.get("environment")),
            "nl": _as_text(data.get("nl")),
        },
    }


def caption_json_to_tags(caption: dict[str, Any] | None) -> list[str]:
    data = normalize_caption_json(caption)
    tags = data.get("tags", {})
    out: list[str] = []
    for key in ("quality",):
        out.extend(split_tags(tags.get(key)))
    for key in ("count", "character", "series", "artist"):
        text = _as_text(tags.get(key))
        if text:
            out.append(text)
    for key in ("appearance", "tags", "environment"):
        out.extend(split_tags(tags.get(key)))
    return split_tags(out)


def caption_json_to_text(caption: dict[str, Any] | None) -> str:
    data = normalize_caption_json(caption)
    tags = data.get("tags", {})
    text = ", ".join(caption_json_to_tags(data))
    nl = _as_text(tags.get("nl"))
    if nl:
        return f"{text}. {nl}" if text else nl
    return text


def standard_to_documented_full(caption: dict[str, Any] | None) -> dict[str, Any]:
    """Write the documented full JSON caption shape used by local tagging."""
    data = normalize_caption_json(caption)
    tags = data["tags"]
    character = _as_text(tags.get("character"))
    return {
        "fixed": {
            "quality": ", ".join(split_tags(tags.get("quality"))),
            "series": _as_text(tags.get("series")),
            "artist": _as_text(tags.get("artist")),
        },
        "character": {
            "name": character,
            "variant": "",
            "full": character,
        },
        "from_path": {},
        "ai_output": {
            "count": _as_text(tags.get("count")),
            "appearance": split_tags(tags.get("appearance")),
            "tags": split_tags(tags.get("tags")),
            "environment": split_tags(tags.get("environment")),
            "nl": _as_text(tags.get("nl")),
        },
    }
