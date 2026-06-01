"""采样图代理（PR-6 commit 1 从 server.py 抽出）。

1 route：
    GET /samples/{filename}    带 task_id 时按 monitor_state_path 多候选目录解析；
                               不给走全局 OUTPUT_DIR/samples/ 兜底；可选 ?w=N 缩略图
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import errors as _errors
from ..responses import _thumb_response
from ... import db
from ...paths import OUTPUT_DIR, task_samples_dir

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/samples/{filename}")
def get_sample(
    filename: str,
    task_id: Optional[int] = None,
    w: Optional[int] = None,
) -> FileResponse:
    """采样图代理。

    `?task_id=N` 给了 → 按多个候选目录查找：
    - **新（task-scoped）** `studio_data/tasks/<task_id>/samples/`
    - `monitor_state.json` 同级 `samples/`（PP6.1 v0.5.0+ 老 task 兼容；
      state file 在 versions/<v>/monitor/task_<id>/ 时，samples 也在那）
    - `monitor_state.json` 同级 `output/samples/`（pre-PP6.1 老 task；
      sample_dir = output_dir/samples，output_dir 通常是 versions/{label}/output）
    - 同级 `output/<任意子目录>/samples/`（兜底防 anima_train 用别的 output 名）

    没给 task_id → 兜底全局 OUTPUT_DIR/samples/（旧训练直接命令行的兼容）。

    `?w=N` 给了 → 走 thumb_cache 生成 N px 缩略图（用于监控页缩略图条）；
    不给 → 返回原图。两种都走 _thumb_response 的弱 etag + no-cache，浏览器
    304 命中即可，避免「重启窗口期失败响应被永久缓存」问题。
    """
    _errors._validate_component_or_400(filename)

    resolved: Optional[Path] = None
    if task_id is not None:
        with db.connection_for() as conn:
            row = conn.execute(
                "SELECT monitor_state_path FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        if not row or not row["monitor_state_path"]:
            raise HTTPException(404)
        monitor_dir = Path(row["monitor_state_path"]).parent
        candidates = [
            # task-scoped 档案 —— 新 task 写在这（migration 之后唯一目标）
            task_samples_dir(task_id) / filename,
            # 老 task 兼容：state.json 同级 samples/（PP6.1 v0.5.0+ +
            # c3961d9 unreleased dev 期间的过渡布局）
            monitor_dir / "samples" / filename,
            # 老 task 兼容：output/samples/（pre-PP6.1）
            monitor_dir / "output" / "samples" / filename,
        ]
        # 再扫一层 output/<sub>/samples/ 兜底（用户改 output_dir 名字时仍能找到）
        output_root = monitor_dir / "output"
        if output_root.is_dir():
            for sub in output_root.iterdir():
                if sub.is_dir():
                    candidates.append(sub / "samples" / filename)
        for p in candidates:
            if p.exists():
                resolved = p
                break
        if resolved is None:
            logger.info(
                "sample 404: task_id=%s file=%s tried=%s",
                task_id, filename, [str(p) for p in candidates],
            )
            raise HTTPException(404)
    else:
        path = OUTPUT_DIR / "samples" / filename
        if not path.exists():
            raise HTTPException(404)
        resolved = path

    # w 给了走缩略图；w<=0 或没给 → 原图。复用 thumb_cache，盘上落 .jpg；
    # 浏览器走弱 etag + no-cache，304 命中很轻。
    if w is not None and w > 0:
        return _thumb_response(resolved, w)
    return _thumb_response(resolved, 0)  # size=0 内部直接返回 src，不缩
