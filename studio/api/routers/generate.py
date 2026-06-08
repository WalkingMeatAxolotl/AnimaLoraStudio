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

import io
import json
import re
import time
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from ..deps import _resolve_anima_model_paths
from ..errors import _validate_component_or_400
from ..schemas.generate import GenerateRequest
from ... import db, secrets
from ...domain import GenerateConfig
from ...infrastructure.event_bus import bus
from ...infrastructure.paths import STUDIO_DATA

router = APIRouter()

TEST_IMAGES_DIR = STUDIO_DATA / "test"
_IMAGE_NAME_RE = re.compile(r"^image_(\d+)\.png$")


def _next_image_index(dir_: Path) -> int:
    """扫描 dir 下 image_<N>.png，返回最大 N+1。找不到则 0。"""
    if not dir_.is_dir():
        return 0
    max_n = -1
    for p in dir_.iterdir():
        m = _IMAGE_NAME_RE.match(p.name)
        if m:
            try:
                max_n = max(max_n, int(m.group(1)))
            except ValueError:
                pass
    return max_n + 1


@router.post("/api/generate")
def enqueue_generate(body: GenerateRequest) -> dict[str, Any]:
    """启动测试出图 task。"""
    from ...services.inference.core import generate_tempdir

    model_paths = _resolve_anima_model_paths()

    with db.connection_for() as conn:
        task_id = db.create_task(
            conn, name="generate", config_name="generate", priority=0,
        )
        db.update_task(conn, task_id, task_type="generate")

    # create_task 已把 task 落成 pending+generate，但 config_path 还没写；supervisor
    # _dispatch_generate 会跳过 config_path=NULL 的 generate task（视为还在入队），等
    # 下面 config.json 落库后再派。这里任一步失败必须把 task 标 failed，否则它会以
    # config_path=NULL 永远 pending（dispatcher 永远跳过）。
    try:
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
            from ...services.runtime.xformers import detect_attention_backend
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
        cfg_path.write_text(
            json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        import time as _time
        with db.connection_for() as conn:
            now = _time.time()
            db.update_task(
                conn, task_id, status="failed",
                started_at=now, finished_at=now,
                error_msg=f"enqueue failed: {e}",
            )
        bus.publish({"type": "task_state_changed", "task_id": task_id, "status": "failed"})
        raise HTTPException(500, f"failed to enqueue generate task: {e}")

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
    from ...services import models as _md
    d = _md.taeflux_dir()
    return {
        "available": _md.taeflux_available(),
        "dir": str(d),
        "files": _md.TAEFLUX_FILES,
    }


@router.post("/api/generate/taeflux/install")
def install_taeflux() -> dict[str, Any]:
    """同步下载 TAEFlux（~1.6MB，秒级）。已存在直接返回 OK。"""
    from ...services import models as _md
    if _md.taeflux_available():
        return {"ok": True, "noop": True}
    ok = _md.download_taeflux()
    if not ok:
        raise HTTPException(500, "download failed; check server log")
    return {"ok": True}


@router.get("/api/generate/daemon/status")
def get_daemon_status() -> dict[str, Any]:
    """查询 daemon 当前状态。前端 DaemonControls 用。"""
    from ...services.inference.daemon import get_daemon
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
    from ...services.inference.daemon import get_daemon
    return get_daemon().read_logs(since_seq=since_seq, limit=limit)


@router.post("/api/generate/daemon/unload")
def unload_daemon() -> dict[str, Any]:
    """手动卸载 daemon 模型（释放 VRAM）。busy 时拒绝（409）。

    卸载完成后 supervisor 会推 daemon_state_changed SSE，前端按钮自动 disable。
    下次用户点「开始生成」daemon 按需重 load。
    """
    from ...services.inference.daemon import get_daemon
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
    from ...services.inference import cache as generate_cache
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


SIDECAR_SCHEMA_VERSION = 1

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DISK_MODES = ("single", "xy")


def _inject_png_params(raw: bytes, params: dict[str, Any]) -> bytes:
    """把 params 作为 tEXt `anima_params` 注入 PNG。失败返回原 bytes。

    单一 tEXt 块即可——用户拷走单文件，参数随图走。a1111 兼容文本块按需后加。
    """
    try:
        from PIL import Image, PngImagePlugin
        img = Image.open(io.BytesIO(raw))
        info = PngImagePlugin.PngInfo()
        info.add_text("anima_params", json.dumps(params, ensure_ascii=False))
        out = io.BytesIO()
        img.save(out, format="PNG", pnginfo=info)
        return out.getvalue()
    except Exception:
        return raw


@router.post("/api/generate/save")
async def save_test_image(
    mode: str = Form(...),
    image: UploadFile = File(...),
    params: str = Form(""),
) -> dict[str, Any]:
    """落盘测试出图到 studio_data/test/<YYYY-MM-DD>/<mode>/image_N.png。

    - mode ∈ {"single", "xy"}，其它值（含 "compare"）400
    - Settings 开关 generate.save_test_images=False → 403
    - N = 当前 <date>/<mode>/ 下已有 image_*.png 最大编号+1（找不到则 0）
    - 并发兜底：x-flag 写入，FileExistsError 则重扫一次
    - params 非空 → 注入 PNG `anima_params` tEXt + 同目录写 image_N.json sidecar
      （sidecar 是 disk-history 扫描入口；PNG metadata 给"拷走 PNG 还能恢复"用）
    """
    if mode not in ("single", "xy"):
        raise HTTPException(400, f"unsupported mode: {mode}")
    if not secrets.load().generate.save_test_images:
        raise HTTPException(403, "save_test_images is disabled")
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image body")

    params_obj: dict[str, Any] | None = None
    if params:
        try:
            decoded = json.loads(params)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"params: invalid JSON ({e})")
        if not isinstance(decoded, dict):
            raise HTTPException(400, "params: must be a JSON object")
        params_obj = decoded
        raw = _inject_png_params(raw, params_obj)

    target_dir = TEST_IMAGES_DIR / date.today().isoformat() / mode
    target_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        idx = _next_image_index(target_dir)
        target = target_dir / f"image_{idx}.png"
        try:
            with open(target, "xb") as f:
                f.write(raw)
        except FileExistsError:
            continue
        sidecar_path: str | None = None
        if params_obj is not None:
            sidecar = target_dir / f"image_{idx}.json"
            sidecar.write_text(
                json.dumps({
                    "schema_version": SIDECAR_SCHEMA_VERSION,
                    "mode": mode,
                    "created_at": time.time(),
                    "filename": f"image_{idx}.png",
                    "params": params_obj,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            sidecar_path = str(sidecar)
        return {"path": str(target), "index": idx, "sidecar": sidecar_path}
    raise HTTPException(500, "could not allocate filename")


# ---------------------------------------------------------------------------
# 磁盘历史浏览：扫 sidecar JSON 列 entries，按图片 URL 单独服务
# ---------------------------------------------------------------------------


def _disk_history_id(date_str: str, mode: str, stem: str) -> str:
    """前端 dedup / merge 用的稳定 id。同图不论几次扫描都映射同一 id。"""
    return f"disk:{date_str}:{mode}:{stem}"


def _scan_sidecars(limit: int) -> list[dict[str, Any]]:
    """扫 TEST_IMAGES_DIR 下所有 <date>/{single,xy}/image_*.json。

    没有 sidecar 的图不入列表（参数缺失，回填无意义；用户仍可去文件夹手动看）。
    """
    if not TEST_IMAGES_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for date_dir in TEST_IMAGES_DIR.iterdir():
        if not date_dir.is_dir() or not _DATE_RE.match(date_dir.name):
            continue
        for mode in _DISK_MODES:
            mode_dir = date_dir / mode
            if not mode_dir.is_dir():
                continue
            for sc in mode_dir.glob("image_*.json"):
                stem = sc.stem  # image_5
                img = mode_dir / f"{stem}.png"
                if not img.is_file():
                    continue  # sidecar 留着但图没了 → 跳过
                try:
                    data = json.loads(sc.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                params = data.get("params")
                if not isinstance(params, dict):
                    continue
                created_at = data.get("created_at")
                if not isinstance(created_at, (int, float)):
                    # 容错：旧 sidecar 缺时间 → fallback 文件 mtime
                    try:
                        created_at = img.stat().st_mtime
                    except OSError:
                        continue
                out.append({
                    "id": _disk_history_id(date_dir.name, mode, stem),
                    "date": date_dir.name,
                    "mode": mode,
                    "filename": f"{stem}.png",
                    "path": str(img),
                    "url": f"/api/generate/disk/image/{date_dir.name}/{mode}/{stem}.png",
                    "created_at": float(created_at),
                    "schema_version": data.get("schema_version", 1),
                    "params": params,
                })
    out.sort(key=lambda e: e["created_at"], reverse=True)
    return out[:limit]


@router.get("/api/generate/disk/history")
def list_disk_history(limit: int = 500) -> dict[str, Any]:
    """列出所有落盘测试图（按 sidecar JSON 扫），按 created_at desc 排。

    前端历史栏拉一次 merge 到 IndexedDB 视图；entry.id 稳定，前端按 id dedup。
    没有 sidecar 的图（老数据 / 客户端没传 params）不入列表。
    """
    limit = max(1, min(int(limit), 2000))
    return {"entries": _scan_sidecars(limit)}


@router.get("/api/generate/disk/image/{date_str}/{mode}/{filename}")
def get_disk_image(date_str: str, mode: str, filename: str) -> Any:
    """读落盘测试图（前端历史栏点击磁盘 entry 时大图来源）。"""
    if not _DATE_RE.match(date_str):
        raise HTTPException(400, "invalid date")
    if mode not in _DISK_MODES:
        raise HTTPException(400, "invalid mode")
    _validate_component_or_400(filename)
    if not filename.lower().endswith(".png"):
        raise HTTPException(400, "only .png supported")
    path = TEST_IMAGES_DIR / date_str / mode / filename
    if not path.is_file():
        raise HTTPException(404)
    # 落盘图内容稳定（不会被同名覆盖——image_N 编号递增），可缓存
    return FileResponse(path, media_type="image/png")
