"""全局服务凭证 + 配置 —— 集中存到 studio_data/secrets.json。

`studio_data/` 已被 .gitignore，本文件即可放真实 token / api key。
对外通过 `to_masked_dict()` 把敏感字段以 "***" 返回；前端 PUT
时若回传 "***" 表示「保持不变」，由 `update()` 的 deep-merge 处理。
"""
from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from .paths import STUDIO_DATA

SECRETS_FILE = STUDIO_DATA / "secrets.json"
MASK = "***"
SENSITIVE_FIELDS: tuple[str, ...] = (
    "gelbooru.api_key",
    "danbooru.api_key",
    "huggingface.token",
)


class GelbooruConfig(BaseModel):
    user_id: str = ""
    api_key: str = ""
    save_tags: bool = False
    convert_to_png: bool = True
    remove_alpha_channel: bool = False


class DanbooruConfig(BaseModel):
    """Danbooru 用 HTTP Basic auth：username + api_key（可匿名跑，但有速率限制）。"""
    username: str = ""
    api_key: str = ""


class HuggingFaceConfig(BaseModel):
    token: str = ""


class DownloadConfig(BaseModel):
    """全局下载偏好（跨渠道共享）。"""
    # 全局排除 tag：搜索时自动追加 -tag1 -tag2（gelbooru / danbooru 语法一致）
    exclude_tags: list[str] = Field(default_factory=list)


class JoyCaptionConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    model: str = "fancyfeast/llama-joycaption-beta-one-hf-llava"
    prompt_template: str = "Descriptive Caption"


class WD14Config(BaseModel):
    model_id: str = "SmilingWolf/wd-vit-tagger-v3"
    local_dir: Optional[str] = None
    threshold_general: float = 0.35
    threshold_character: float = 0.85
    blacklist_tags: list[str] = Field(default_factory=list)


class Secrets(BaseModel):
    gelbooru: GelbooruConfig = Field(default_factory=GelbooruConfig)
    danbooru: DanbooruConfig = Field(default_factory=DanbooruConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)
    joycaption: JoyCaptionConfig = Field(default_factory=JoyCaptionConfig)
    wd14: WD14Config = Field(default_factory=WD14Config)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load() -> Secrets:
    """读 secrets.json；缺失或损坏时返回默认实例（不抛错）。"""
    if not SECRETS_FILE.exists():
        return Secrets()
    try:
        return Secrets.model_validate_json(SECRETS_FILE.read_text(encoding="utf-8"))
    except Exception:
        # 文件损坏不应阻断 Studio 启动；用默认值覆盖
        return Secrets()


def save(s: Secrets) -> None:
    SECRETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SECRETS_FILE.write_text(s.model_dump_json(indent=2), encoding="utf-8")


def get(path: str) -> Any:
    """点路径取值，例：`get('wd14.threshold_general')`。"""
    cur: Any = load()
    for seg in path.split("."):
        cur = getattr(cur, seg)
    return cur


def update(partial: dict[str, Any]) -> Secrets:
    """deep-merge `partial` 进当前持久化值；返回新 Secrets 并落盘。

    - `partial` 里 leaf 值为 MASK ("***") 时，表示「保持原值不变」。
    - 未提及的字段沿用旧值。
    """
    current_dict = load().model_dump()
    merged = _deep_merge(current_dict, partial)
    new = Secrets.model_validate(merged)
    save(new)
    return new


def to_masked_dict(s: Secrets) -> dict[str, Any]:
    """GET /api/secrets 返回此结构；敏感字段非空时替换为 MASK。"""
    d = s.model_dump()
    for path in SENSITIVE_FIELDS:
        segs = path.split(".")
        cur: Any = d
        for seg in segs[:-1]:
            cur = cur[seg]
        if cur.get(segs[-1]):
            cur[segs[-1]] = MASK
    return d


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """把 patch 合并到 base：嵌套 dict 递归合并；leaf 值为 MASK 则丢弃。"""
    out = dict(base)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        elif val == MASK:
            # 保持旧值
            continue
        else:
            out[key] = val
    return out


def has_gelbooru_credentials() -> bool:
    """便捷：用于前端 / 端点判断是否已经配好 Gelbooru。"""
    g = load().gelbooru
    return bool(g.user_id and g.api_key)


def has_credentials_for(api_source: str) -> bool:
    """各下载渠道的「能不能跑」判定：
    - gelbooru: 必须有 user_id + api_key（API 强制要求）
    - danbooru: 匿名也能跑（仅速率受限），所以始终 True
    """
    if api_source == "gelbooru":
        return has_gelbooru_credentials()
    if api_source == "danbooru":
        return True
    return False
