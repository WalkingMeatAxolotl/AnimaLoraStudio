"""Backward-compat shim — 真正的 updater 已搬到 studio.services.runtime.updater。

**保留此文件的目的（不要删！）**：
0.10.2 及更早的 client 在 preflight 阶段调 `target_has_self_update()` 只检查
一条 `_SELF_UPDATE_MARKER = "studio/services/updater.py"` 路径。v0.11.0 PR #143
services/ 重构把 updater.py 搬到 `services/runtime/` 之后，v0.10.2 用户的
client 跑 `git cat-file -e origin/master:studio/services/updater.py` 永远失败
→ preflight 误报"目标版本早于 webui 自更新 feature" → 阻断确认按钮 → 用户
没法用 webui 升级。保留此 shim 让旧 client 的存在性检查能通过。

只要还有 0.10.2- 用户存在升级路径，就**不要删**此文件。删之前 grep 一遍
发布版历史里 `_SELF_UPDATE_MARKER` / `_SELF_UPDATE_MARKERS` 的值，确认所有
有人可能装的旧版本 client 都已经识别新路径，再考虑废弃。

新代码请直接 `from studio.services.runtime.updater import ...`。
"""
from .runtime.updater import *  # noqa: F401, F403
