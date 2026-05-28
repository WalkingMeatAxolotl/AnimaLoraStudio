"""v6 → v7: versions.trigger_word — 项目级触发词由 Step 4 (Tagging) 填写。

加列 `versions.trigger_word TEXT DEFAULT ''`。空串语义 = 不启用触发词。
Tag worker 用它在写 caption 时 prepend 第一个 tag；version_config 把它注入
私有 yaml，runtime bootstrap_phase 顺便把它注入 sample_prompt/sample_prompts。
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(
        conn, "versions", "trigger_word",
        "trigger_word TEXT NOT NULL DEFAULT ''"
    )
