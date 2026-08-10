"""v19 → v20: tasks 加 generate_images(出图时间线 DB 单源)+ 最小列回填。

出图时间线切 DB 单源(替代 disk 扫盘 ∪ cache index 双源):tasks 行是列表
唯一来源,generate_images 记录任务全部产出图,图不在时行显示「已释放」。

- generate_images:JSON 数组,元素二选一:
    {"file": "<date>/single/single image 5.png"}          落盘图(相对 test/,正斜杠)
    {"file": "<date>/xy/xy plot 3/cell x0 y0.png", "xi": 0, "yi": 0}   XY cell
    {"cache": "<daemon filename>"}                         temp 图(会话级加密 cache)
  XY composite(xy plot.png)不入列表——它是文件夹附件(外站上传用),
  回看用 cells 渲网格。

回填(拍板决策 3):**只 UPDATE 绝不 INSERT** —— 过去 forward-write 期间
(v14 起)的行已有 generate_cover,从 cover 推导 images;无 task 行的更老
落盘图不造合成行(从应用列表消失,盘上文件保留)。migration user_version
门槛保证一次性,不会每次启动重跑;UPDATE-only 保证零重复行。

- cover 指 single PNG → images=[{"file": cover}]
- cover 指 xy composite → glob 同文件夹 `cell x* y*.png` parse xi/yi;
  文件夹已被手删 → images=[](行显示已释放)
- temp 行(cover NULL)不动 → images NULL → 已释放
"""
from __future__ import annotations

import json
import re
import sqlite3

from ._v2_projects import _add_column_if_missing

_CELL_RE = re.compile(r"^cell x(\d+) y(\d+)\.png$")


def _images_from_cover(cover: str) -> list[dict[str, object]]:
    """generate_cover(相对 test/ 的路径,可能含反斜杠)→ generate_images 值。"""
    from ...paths import STUDIO_DATA

    rel = cover.replace("\\", "/")
    parts = rel.split("/")
    if "xy" not in parts:
        return [{"file": rel}]
    # xy composite:<date>/xy/<folder>/xy plot.png → glob 文件夹 cells
    folder = (STUDIO_DATA / "test" / rel).parent
    if not folder.is_dir():
        return []
    folder_rel = "/".join(parts[:-1])
    out: list[dict[str, object]] = []
    for p in sorted(folder.iterdir()):
        m = _CELL_RE.match(p.name)
        if m:
            out.append({
                "file": f"{folder_rel}/{p.name}",
                "xi": int(m.group(1)),
                "yi": int(m.group(2)),
            })
    out.sort(key=lambda c: (c["yi"], c["xi"]))
    return out


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "generate_images", "generate_images TEXT")
    rows = conn.execute(
        "SELECT id, generate_cover FROM tasks "
        "WHERE task_type = 'generate' AND generate_cover IS NOT NULL "
        "AND generate_images IS NULL"
    ).fetchall()
    for task_id, cover in rows:
        try:
            images = _images_from_cover(str(cover))
        except OSError:
            images = [{"file": str(cover).replace("\\", "/")}]
        conn.execute(
            "UPDATE tasks SET generate_images = ? WHERE id = ?",
            (json.dumps(images, ensure_ascii=False), task_id),
        )
