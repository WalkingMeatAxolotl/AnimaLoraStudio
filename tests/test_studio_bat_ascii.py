"""studio.bat 必须保持纯 ASCII 字节。

为什么：cmd.exe 在 `chcp 65001` 生效**之前**用系统 ANSI codepage 解析 .bat 文件
（cp936 中文 / cp932 日文 / cp1252 西欧）。任何 UTF-8 多字节序列在这些 codepage
下会被错位解析，常见结果是把字节边界中的子串当作命令名 → 报「'XXX' is not
recognized」一连串错误。

历史教训：PR-S1bd 提交时在 REM 注释里加了一行中文「内容 hash 与 marker 不一致」，
在 cp936 系统上整段脚本崩溃（用户报告 'udio'/'all'/'frontend' 等乱命令错误）。
本测试守住这条约束，让以后 LLM / 人误加非 ASCII 字符立刻被 pytest 拦下。

studio.sh 不在守护范围 —— bash/zsh 默认 LANG=en_US.UTF-8 / zh_CN.UTF-8，
脚本里中文不会引发同类问题；echo 出 stderr 给非 UTF-8 locale 的用户看到乱码
最多是 cosmetic，不会让脚本本身解析失败。
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDIO_BAT = REPO_ROOT / "studio.bat"


def test_studio_bat_exists() -> None:
    assert STUDIO_BAT.exists(), "studio.bat 不该消失"


def test_studio_bat_is_pure_ascii() -> None:
    """studio.bat 字节流必须全部 < 128（纯 ASCII）。"""
    data = STUDIO_BAT.read_bytes()
    bad_offsets = [(i, b) for i, b in enumerate(data) if b > 127]
    if bad_offsets:
        # 给出前 5 处定位 + 上下文，方便诊断
        snippets = []
        for off, b in bad_offsets[:5]:
            start = max(0, off - 20)
            end = min(len(data), off + 20)
            ctx = data[start:end].decode("utf-8", errors="replace")
            snippets.append(f"  offset {off} (byte 0x{b:02x}): ...{ctx!r}...")
        pytest.fail(
            f"studio.bat 含 {len(bad_offsets)} 个非 ASCII 字节（详情前 5 处）：\n"
            + "\n".join(snippets)
            + "\n\nstudio.bat 必须保持纯 ASCII —— cmd.exe 在 chcp 65001 之前"
            "用系统 ANSI 解析，UTF-8 中文会被错位拆出乱命令。"
            "中文消息请放 echo 里走 PYTHONUTF8 / chcp 65001 处理后的 Python 进程。"
        )
