#!/usr/bin/env python
"""Bootstrap helper: 按检测到的 NVIDIA 驱动版本输出 PyTorch wheel index URL。

studio.bat / studio.sh 在 venv **首装**时调本脚本，先按 GPU 装对的 torch，再装
requirements.txt。约束 `torch>=2.0.0` 已被首步满足，pip 不会再覆盖成 PyPI 默认 CPU 版。

Stdlib only —— 在 venv 刚装好（只有 pip + setuptools）时也能跑。

输出：
- 检测到合适驱动 → stdout 一行 URL，如 `https://download.pytorch.org/whl/cu128`
- 没装 nvidia-smi / 解析失败 / 驱动太旧 → 静默无输出（caller 用 PyPI 默认）
- 永远 exit 0，不让 bootstrap 因这一项 fail

驱动→cu wheel 映射：与 studio/services/torch_setup.py:_DRIVER_TO_BEST_CU 同步。
单独 duplicate 是为了 bootstrap 阶段不依赖 studio.services 子模块加载链。
"""
from __future__ import annotations

import re
import subprocess
import sys

# 注：维护此表时同步更新 studio/services/torch_setup.py:_DRIVER_TO_BEST_CU
_DRIVER_TO_CU: list[tuple[int, str]] = [
    (555, "cu128"),
    (550, "cu126"),
    (545, "cu124"),
    (470, "cu118"),
]
_PYPI_BASE = "https://download.pytorch.org/whl"


def detect_driver_major() -> int | None:
    """跑 nvidia-smi 拿驱动版本主号；失败 / 不存在返回 None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    line = (out.stdout or "").strip().split("\n")[0]
    m = re.match(r"^(\d+)\.", line)
    if not m:
        return None
    return int(m.group(1))


def select_index_url(driver_major: int | None) -> str | None:
    """driver 主号 → PyTorch wheel index URL；驱动太旧 / None → None。"""
    if driver_major is None:
        return None
    for threshold, tag in _DRIVER_TO_CU:
        if driver_major >= threshold:
            return f"{_PYPI_BASE}/{tag}"
    return None


def main() -> int:
    url = select_index_url(detect_driver_major())
    if url:
        # 不带换行 —— shell `for /f` / `$()` 处理时少踩坑
        sys.stdout.write(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
