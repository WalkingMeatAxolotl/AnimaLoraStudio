"""测试出图 + daemon 控制 + TAEFlux（PR-6 commit 5 从 server.py 抽出）。

8 routes：
    POST /api/generate                          启动出图 task（daemon 跑）
    GET  /api/generate/{task_id}                查询测试 task 状态
    GET  /api/generate/taeflux/status           中间步预览模型是否就绪
    POST /api/generate/taeflux/install          同步下载 TAEFlux（~1.6MB 秒级）
    GET  /api/generate/daemon/status            daemon state / model_loaded / busy
    GET  /api/generate/daemon/logs              ring buffer 日志（since_seq / limit）
    POST /api/generate/daemon/unload            手动卸载（busy 时 409）
    GET  /api/generate/{task_id}/sample/{filename}  从 generate_cache 取 PNG bytes

测试出图不持久化（commit 10 起）：daemon 把 PNG bytes base64 推回 server 入
generate_cache（内存 dict），HTTP 这里从 cache 取。tempdir 仅装 config.json，
task 结束 supervisor 仍调 cleanup_generate_tempdir 清掉空目录。server 重启 →
内存 cache 自动没；强杀也不残留。
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..deps import _resolve_anima_model_paths
from ..errors import _validate_component_or_400
from ..schemas.generate import GenerateRequest
from ... import db, secrets
from ...domain import GenerateConfig
from ...infrastructure.event_bus import bus

router = APIRouter()


@router.post("/api/generate")
def enqueue_generate(body: GenerateRequest) -> dict[str, Any]:
    """启动测试出图 task。"""
    from ...services.inference_core import generate_tempdir

    model_paths = _resolve_anima_model_paths()

    with db.connection_for() as conn:
        task_id = db.create_task(
            conn, name="generate", config_name="generate", priority=0,
        )
        db.update_task(conn, task_id, task_type="generate")

    tempdir = generate_tempdir(task_id)
    tempdir.mkdir(parents=True, exist_ok=True)

    # attention_backend：secrets 读默认；body 给值则覆盖（兼容旧客户端）
    # secrets 默认 'auto' → 调 detect_attention_backend 按"装了什么用什么"决定
    try:
        gen_cfg = secrets.load().generate
        attn_default = gen_cfg.attention_backend
        preview_n = int(gen_cfg.preview_every_n_steps or 0)
    except Exception:
        attn_default = "auto"
        preview_n = 0
    attn = body.attention_backend or attn_default
    if attn == "auto":
        from ...services.xformers_setup import detect_attention_backend
        attn = detect_attention_backend()

    cfg = GenerateConfig(
        **model_paths,
        output_dir=str(tempdir),
        prompts=body.prompts,
        negative_prompt=body.negative_prompt,
        width=body.width,
        height=body.height,
        steps=body.steps,
        cfg_scale=body.cfg_scale,
        sampler_name=body.sampler_name,
        scheduler=body.scheduler,
        count=body.count,
        seed=body.seed,
        lora_configs=[lc.model_dump() for lc in body.lora_configs],
        mixed_precision=body.mixed_precision,
        attention_backend=attn,
        xy_matrix=body.xy_matrix.model_dump() if body.xy_matrix else None,
    )

    # commit 14：注入 daemon 端用的 preview 节流参数（settings 全局开关）
    cfg_dict = cfg.model_dump()
    cfg_dict["preview_every_n_steps"] = preview_n

    cfg_path = tempdir / "config.json"
    cfg_path.write_text(json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    with db.connection_for() as conn:
        db.update_task(conn, task_id, config_path=str(cfg_path))
        task = db.get_task(conn, task_id)

    bus.publish({"type": "task_state_changed", "task_id": task_id, "status": "pending"})
    return task or {"id": task_id}


@router.get("/api/generate/{task_id}")
def get_generate_task(task_id: int) -> dict[str, Any]:
    """查询测试 task 状态。"""
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task or task.get("task_type") != "generate":
        raise HTTPException(404)
    return task


# ---------------------------------------------------------------------------
# /api/generate/daemon — 测试 daemon 状态查询 + 手动卸载（commit 13）
# ---------------------------------------------------------------------------


@router.get("/api/generate/taeflux/status")
def get_taeflux_status() -> dict[str, Any]:
    """commit 14：查询 TAEFlux 模型是否就绪（中间步预览依赖）。"""
    from ...services import model_downloader as _md
    d = _md.taeflux_dir()
    return {
        "available": _md.taeflux_available(),
        "dir": str(d),
        "files": _md.TAEFLUX_FILES,
    }


@router.post("/api/generate/taeflux/install")
def install_taeflux() -> dict[str, Any]:
    """同步下载 TAEFlux（~1.6MB，秒级）。已存在直接返回 OK。"""
    from ...services import model_downloader as _md
    if _md.taeflux_available():
        return {"ok": True, "noop": True}
    ok = _md.download_taeflux()
    if not ok:
        raise HTTPException(500, "download failed; check server log")
    return {"ok": True}


@router.get("/api/generate/daemon/status")
def get_daemon_status() -> dict[str, Any]:
    """查询 daemon 当前状态。前端 DaemonControls 用。"""
    from ...services.inference_daemon import get_daemon
    daemon = get_daemon()
    return {
        "state": daemon.state,
        "model_loaded": daemon.is_model_loaded,
        "busy": daemon.is_busy,
        "alive": daemon.is_alive,
    }


@router.get("/api/generate/daemon/logs")
def get_daemon_logs(since_seq: int = 0, limit: int = 2000) -> dict[str, Any]:
    """读 daemon stderr ring buffer。前端日志抽屉打开时拉历史；增量靠 SSE。

    since_seq>0 时只返新于该 seq 的行。
    """
    from ...services.inference_daemon import get_daemon
    return get_daemon().read_logs(since_seq=since_seq, limit=limit)


@router.post("/api/generate/daemon/unload")
def unload_daemon() -> dict[str, Any]:
    """手动卸载 daemon 模型（释放 VRAM）。busy 时拒绝（409）。

    卸载完成后 supervisor 会推 daemon_state_changed SSE，前端按钮自动 disable。
    下次用户点「开始生成」daemon 按需重 load。
    """
    from ...services.inference_daemon import get_daemon
    daemon = get_daemon()
    if daemon.is_busy:
        raise HTTPException(409, "daemon is busy, cannot unload")
    if not daemon.is_model_loaded:
        return {"ok": True, "noop": True}
    daemon.request_unload()
    return {"ok": True}


@router.get("/api/generate/{task_id}/sample/{filename}")
def get_generate_sample(task_id: int, filename: str) -> Any:
    """读 generate task 的输出图（commit 10：从 server 内存 cache 取，无磁盘）。

    daemon 出图完成后把 PNG bytes 推回 server 入 generate_cache；HTTP 这里
    直接返回 bytes。LRU / 客户端断连清理在 commit 11 加 —— 在那之前 cache
    跟着 supervisor finalize 释放（一 task 一组 entry，task 终止时全清）。
    """
    _validate_component_or_400(filename)
    if not filename.lower().endswith(".png"):
        raise HTTPException(400, "only .png supported")
    from ...services import generate_cache
    data = generate_cache.get_image(task_id, filename)
    if data is None:
        raise HTTPException(404)
    # 用 no-store 不是 _thumb_response 那套 no-cache + ETag：
    # generate cache 同 (task_id, filename) 内容会随重跑覆盖（用户改 prompt 重生成），
    # 没有稳定 ETag 可发；用 no-store 让浏览器每次都重拉，永远拿到最新结果。
    # 带宽代价小：用户在测试出图页主动看才命中本 endpoint，QPS 低。
    # （Thumbnail / dataset 那种内容稳定的图，继续用 _thumb_response 的 ETag。）
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )
