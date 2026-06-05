"""Tag 翻译词典 —— 给前端 chip 显示中文 + autocomplete 提供数据。

存储布局：
    studio_data/tag_dictionary/
        active.json   ← 解析后的词典 + meta（前端 GET /api/tag-dictionary/data 读这个）
        source.csv    ← 原始上传 / 下载文件留底（排查用）

默认数据源（Physton/sd-webui-prompt-all-in-one-assets）格式：
    english_tag,zh1 zh2 zh3
    （第二列空白分隔多个中文别名；无 header；'#' 起头视为注释）
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

import requests

from .paths import STUDIO_DATA

logger = logging.getLogger(__name__)

TAG_DICT_DIR = STUDIO_DATA / "tag_dictionary"
ACTIVE_JSON = TAG_DICT_DIR / "active.json"
SOURCE_FILE = TAG_DICT_DIR / "source.csv"

DEFAULT_URL = (
    "https://raw.githubusercontent.com/Physton/"
    "sd-webui-prompt-all-in-one-assets/main/tags/danbooru.zh_CN.csv"
)
DEFAULT_SOURCE_NAME = "physton/danbooru.zh_CN.csv"

MAX_BYTES = 10 * 1024 * 1024  # 10MB
MAX_ENTRIES = 200_000

_CJK_RE = re.compile(r"[一-鿿]")


def parse_csv(text: str) -> dict[str, list[str]]:
    """解析 csv/txt 内容 → `{english_tag: [zh_alias, ...]}`。

    规则：
    - 每行按第一个 ',' 切两段；只有一列时 value 为 `[]`（让英文 tag 仍参与 autocomplete）
    - english 列：strip + 把 '_' 换成空格（用户/caption 形态约定）
    - zh 列：按任意空白切分；过滤纯英文 token？—— **不**，原始翻译里可能混罗马音
      （如 `breasts,胸部 乳房 oppai`），全部保留让反向索引覆盖度高
    - 空行 / '#' 起头注释跳过
    - 超 MAX_ENTRIES 截断 + 日志告警；返回字典（最后一次出现的 tag 覆盖前面）
    """
    entries: dict[str, list[str]] = {}
    truncated = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if len(entries) >= MAX_ENTRIES:
            truncated = True
            break
        if "," in line:
            head, _, tail = line.partition(",")
            tag = head.strip().replace("_", " ")
            aliases = [seg for seg in tail.split() if seg]
        else:
            tag = line.replace("_", " ")
            aliases = []
        if not tag:
            continue
        entries[tag] = aliases
    if truncated:
        logger.warning(
            "tag_dictionary: 超过 %d 条上限，截断；丢弃了后续行", MAX_ENTRIES
        )
    return entries


def _meta(source_name: str, source_url: str, kind: str, count: int) -> dict[str, Any]:
    return {
        "source_name": source_name,
        "source_url": source_url,
        "entry_count": count,
        "downloaded_at": int(time.time()),
        "kind": kind,  # "default" | "user"
    }


def _write_active(entries: dict[str, list[str]], meta: dict[str, Any]) -> None:
    TAG_DICT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "entries": entries}
    ACTIVE_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def load_active() -> Optional[tuple[dict[str, list[str]], dict[str, Any]]]:
    """读 active.json；未初始化或损坏返回 None。"""
    if not ACTIVE_JSON.exists():
        return None
    try:
        raw = json.loads(ACTIVE_JSON.read_text(encoding="utf-8"))
        entries = raw.get("entries") or {}
        meta = raw.get("meta") or {}
        if not isinstance(entries, dict) or not isinstance(meta, dict):
            return None
        return entries, meta
    except Exception:
        logger.exception("tag_dictionary: active.json 损坏，视作未加载")
        return None


def get_meta() -> Optional[dict[str, Any]]:
    """只读 meta 字段（GET /api/tag-dictionary/meta 用，避免 3MB 全量解析）。"""
    loaded = load_active()
    if loaded is None:
        return None
    _, meta = loaded
    return meta


def download_default() -> dict[str, Any]:
    """从 GitHub 拉默认词典并 commit 成 active.json，返回新 meta。

    失败抛 RuntimeError（带原因），上层 router 转 502。
    """
    TAG_DICT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(DEFAULT_URL, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"download failed: {exc}") from exc
    content = resp.content
    if len(content) > MAX_BYTES:
        raise RuntimeError(
            f"downloaded file too large: {len(content)} bytes > {MAX_BYTES}"
        )
    text = content.decode("utf-8", errors="replace")
    entries = parse_csv(text)
    if not entries:
        raise RuntimeError("downloaded file parsed to zero entries")
    SOURCE_FILE.write_bytes(content)
    meta = _meta(DEFAULT_SOURCE_NAME, DEFAULT_URL, "default", len(entries))
    _write_active(entries, meta)
    logger.info("tag_dictionary: 默认词典已下载 (%d 条)", len(entries))
    return meta


def apply_uploaded(content: bytes, filename: str) -> dict[str, Any]:
    """处理用户上传的 csv/txt：校验大小 → 解析 → 写盘 → 返回新 meta。

    超 MAX_BYTES 或解析后 0 条都抛 ValueError（上层转 400）。
    """
    if len(content) > MAX_BYTES:
        raise ValueError(
            f"文件过大：{len(content)} bytes，上限 {MAX_BYTES} bytes（10MB）"
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"文件编码必须是 UTF-8：{exc}") from exc
    entries = parse_csv(text)
    if not entries:
        raise ValueError("解析后 0 条；文件格式可能不对（期望 `english_tag,zh1 zh2`）")
    TAG_DICT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_FILE.write_bytes(content)
    meta = _meta(filename or "user-upload", "", "user", len(entries))
    _write_active(entries, meta)
    logger.info("tag_dictionary: 用户上传词典已加载 (%d 条, %s)", len(entries), filename)
    return meta


def reset_to_default() -> dict[str, Any]:
    """"恢复默认词典"按钮调用 —— 重新下载并替换 active.json。"""
    return download_default()


def has_cjk(s: str) -> bool:
    """是否含中文字符（前端反向匹配判定也是同一规则；这里仅给测试 / 内部用）。"""
    return bool(_CJK_RE.search(s))
