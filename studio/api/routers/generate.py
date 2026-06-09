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

import hashlib
import io
import json
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
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

# v2 命名（决策 #6）：父目录区分 mode，文件名仅 "<label> N.png"
_DISPLAY_LABELS = {"single": "single image", "xy": "xy plot"}
_V2_SINGLE_RE = re.compile(r"^single image (\d+)\.png$")
_V2_XY_RE = re.compile(r"^xy plot (\d+)\.png$")
# v1 legacy：image_N.png（旧版命名），扫描时仍读取，但新写入只用 v2
_V1_NAME_RE = re.compile(r"^image_(\d+)\.png$")


def _next_image_index(dir_: Path, mode: str) -> int:
    """扫描 dir 下当前 mode 的 PNG 文件，返回下一个 1-based 序号。

    决策 #11：无并发跑图场景，不做 O_EXCL / 锁；序号扫 max+1 + atomic 写即可。
    决策 #6：v2 命名 1-based（"single image 1" 比 0 直观），v1 legacy `image_N`
    若同目录混存按合并扫一组取 max+1。
    """
    if not dir_.is_dir():
        return 1
    rx_v2 = _V2_SINGLE_RE if mode == "single" else _V2_XY_RE
    max_n = 0
    for p in dir_.iterdir():
        if not p.is_file():
            continue
        m_v2 = rx_v2.match(p.name)
        m_v1 = _V1_NAME_RE.match(p.name)
        if m_v2:
            max_n = max(max_n, int(m_v2.group(1)))
        elif m_v1:
            # v1 legacy 0-based；映射到 v2 编号空间 +1 避免冲突
            max_n = max(max_n, int(m_v1.group(1)) + 1)
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


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DISK_MODES = ("single", "xy")
_PNG_NAME_SAFE_RE = re.compile(r"^[a-zA-Z0-9 ._-]+\.png$")  # disk-image / thumb / delete 路径校验


SCHEMA_VERSION = 2


def _format_a1111_parameters(params: dict[str, Any]) -> str:
    """组装 a1111 兼容的 `parameters` tEXt 块（ComfyUI / WebUI / Civitai 等通用）。

    格式：
        <prompt> [<lora:name:scale> ...]
        Negative prompt: <neg>
        Steps: N, Sampler: ..., Schedule type: ..., CFG scale: N, Seed: N, Size: WxH

    LoRA 用 <lora:basename-without-ext:scale> 语法（a1111/ComfyUI 标准）。
    xy_draft / dataset_pick 不入此块（a1111 没标准字段；用 anima_params 取）。
    """
    prompts = params.get("prompts") or [""]
    prompt = prompts[0] if isinstance(prompts, list) else str(prompts)
    loras = params.get("loras") or []
    lora_tags: list[str] = []
    for lo in loras:
        if not isinstance(lo, dict):
            continue
        name = str(lo.get("name") or "").rsplit(".", 1)[0]  # 去 .safetensors
        if not name:
            continue
        scale = lo.get("scale", 1.0)
        lora_tags.append(f"<lora:{name}:{scale}>")
    if lora_tags:
        prompt = f"{prompt} {' '.join(lora_tags)}".strip()

    neg = params.get("negative_prompt", "")
    width = params.get("width", 0)
    height = params.get("height", 0)
    parts = [
        f"Steps: {params.get('steps', '')}",
        f"Sampler: {params.get('sampler_name', 'er_sde')}",
        f"Schedule type: {params.get('scheduler', 'simple')}",
        f"CFG scale: {params.get('cfg_scale', '')}",
        f"Seed: {params.get('seed', '')}",
        f"Size: {width}x{height}",
    ]
    return f"{prompt}\nNegative prompt: {neg}\n{', '.join(parts)}"


def _inject_png_metadata(raw: bytes, params: dict[str, Any], *, mode: str) -> bytes:
    """注入 PNG tEXt 块到图：
       - `anima_params` —— 结构化 JSON，**zTXt 压缩**（决策 #17），本程序回填用
       - `parameters`   —— a1111 兼容文本（决策 #7：xy **不写**，矩阵图单图拖
         进 a1111 参数语义对不上）；仅 single 模式写

    失败返回原 bytes（不阻塞落盘主流程）。
    """
    try:
        from PIL import Image, PngImagePlugin
        img = Image.open(io.BytesIO(raw))
        info = PngImagePlugin.PngInfo()
        # zip=True → zTXt 压缩块（PIL 9+），XY cells[] 时 anima_params 可能 6KB+，
        # 压缩后通常 1-2KB，a1111 不识别 anima_params 反正会跳过
        info.add_text("anima_params", json.dumps(params, ensure_ascii=False), zip=True)
        if mode == "single":
            info.add_text("parameters", _format_a1111_parameters(params))
        out = io.BytesIO()
        img.save(out, format="PNG", pnginfo=info)
        return out.getvalue()
    except Exception:
        return raw


def _read_png_anima_params(path: Path) -> dict[str, Any] | None:
    """从 PNG `anima_params` tEXt / zTXt 块解析 params；无 / 解析失败返 None。

    决策 #16：只读 PNG header chunk（PIL `Image.open` 已自动解析 tEXt/zTXt），
    **不调 `img.load()`**（不 decode 像素）。500 张 4K PNG mount 用时从 15-30s
    降到 1-2s。
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            # img.text 在 PIL 8+ 由 open() 阶段读 PNG chunks（含 tEXt/zTXt）填充；
            # 不需要 img.load() 触发像素解码
            text = img.text.get("anima_params") if hasattr(img, "text") else None
        if not text:
            return None
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _migrate_anima_params(meta: dict[str, Any]) -> dict[str, Any]:
    """v1 → v2 schema 迁移（决策 #18）。

    v1: `lora_configs[].path` 是绝对路径（旧 schema 直接存 path）
    v2: `loras[].name` basename + project_id/version_id；不存绝对路径

    迁移规则：v1 PNG → 把 `lora_configs[].path` 末段 basename 当 v2 `loras[].name`，
    保留 project_id/version_id/scale；旧 path 丢弃（隐私 + 跨机器死链）。
    """
    version = meta.get("schema_version", 1)
    if version >= 2:
        return meta
    if version == 1:
        legacy_loras = meta.pop("lora_configs", None)
        if isinstance(legacy_loras, list):
            new_loras: list[dict[str, Any]] = []
            for lc in legacy_loras:
                if not isinstance(lc, dict):
                    continue
                path = str(lc.get("path") or "")
                name = path.replace("\\", "/").rsplit("/", 1)[-1] if path else ""
                new_loras.append({
                    "name": name,
                    "scale": float(lc.get("scale", 1.0)),
                    "project_id": lc.get("project_id"),
                    "version_id": lc.get("version_id"),
                })
            meta["loras"] = new_loras
        meta["schema_version"] = 2
        return meta
    # 未知版本 → 当作 v2 透传（forward-compat）
    return meta


def _enrich_params_server_side(
    params: dict[str, Any], *, task_id: int | None, mode: str
) -> dict[str, Any]:
    """server 端补全 params 的服务端信息（避免前端伪造 / 漏字段）。

    - `schema_version` 强制覆盖为当前版本
    - `created_at` 落盘时刻（Unix 秒）
    - `task_id` 来自 enqueue（前端不传 / 不可信任）
    - `mode` 来自路由参数（前端不传）
    """
    params = dict(params)
    params["schema_version"] = SCHEMA_VERSION
    params["created_at"] = time.time()
    if task_id is not None:
        params["task_id"] = int(task_id)
    params["mode"] = mode
    return params


def _atomic_write_png(target: Path, raw: bytes) -> None:
    """原子写 PNG：写 tmp + os.replace（决策 #11 crash safety）。

    server 在写到一半挂掉时不会留半截 PNG 让 disk-history 扫到（半截 PNG 无
    PNG IEND chunk，PIL 解析失败，disk-history 会跳过这条；但用户在文件管理
    器里看到一半文件仍是噪音）。tmp + replace 让 target 出现的瞬间内容已完整。
    """
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, target)


@router.post("/api/generate/save")
async def save_test_image(
    mode: str = Form(...),
    image: UploadFile = File(...),
    params: str = Form(""),
    task_id: Optional[int] = Form(None),
) -> dict[str, Any]:
    """落盘测试出图到 `studio_data/test/<YYYY-MM-DD>/<mode>/<label> <N>.png`。

    - mode ∈ {"single", "xy"}，其它值（含 "compare"）400
    - Settings 开关 generate.save_test_images=False → 403
    - 文件名（决策 #6）：`single image 1.png` / `xy plot 1.png` 类型递增
    - 原子写（决策 #11）：tmp + os.replace；不做 EEXIST 循环
    - params 非空 → 注入 PNG `anima_params` (zTXt) + 仅 single 写 `parameters` (a1111 tEXt)
    - server 端 enrich：强制 `schema_version`/`created_at`/`task_id`/`mode`
    """
    if mode not in ("single", "xy"):
        raise HTTPException(400, f"unsupported mode: {mode}")
    if not secrets.load().generate.save_test_images:
        raise HTTPException(403, "save_test_images is disabled")
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image body")

    if params:
        try:
            decoded = json.loads(params)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"params: invalid JSON ({e})")
        if not isinstance(decoded, dict):
            raise HTTPException(400, "params: must be a JSON object")
        enriched = _enrich_params_server_side(decoded, task_id=task_id, mode=mode)
        raw = _inject_png_metadata(raw, enriched, mode=mode)

    target_dir = TEST_IMAGES_DIR / date.today().isoformat() / mode
    target_dir.mkdir(parents=True, exist_ok=True)
    idx = _next_image_index(target_dir, mode)
    target = target_dir / f"{_DISPLAY_LABELS[mode]} {idx}.png"
    _atomic_write_png(target, raw)
    return {"path": str(target), "index": idx, "filename": target.name}


# ---------------------------------------------------------------------------
# 磁盘历史浏览：扫 PNG `anima_params` tEXt 块列 entries，按图片 URL 单独服务
# ---------------------------------------------------------------------------


def _disk_history_id(date_str: str, mode: str, filename: str) -> str:
    """前端 dedup / merge 用的稳定 id。

    用 sha1 短哈希替代直接拼 filename —— filename 含空格（决策 #6 "single image 1"）
    塞进 React key / data-testid / URL fragment 会踩坑。哈希 12 位足够全局唯一。
    """
    h = hashlib.sha1(f"{date_str}/{mode}/{filename}".encode("utf-8")).hexdigest()[:12]
    return f"disk:{h}"


def _url_quote_filename(filename: str) -> str:
    """文件名内空格 / 中文等 URL encode（决策 #6 文件名带空格）。后端返 URL
    时直接 encode 好，前端拼接禁止。"""
    from urllib.parse import quote
    return quote(filename, safe="")


def _scan_png_metadata(limit: int) -> list[dict[str, Any]]:
    """扫 TEST_IMAGES_DIR 下所有 <date>/{single,xy}/*.png 的 anima_params。

    没有 anima_params tEXt 块的图（老数据 / 客户端没传 params）不入列表。
    决策 #16：_read_png_anima_params 只读 header（不 load 像素），500 张 1-2s。
    决策 #18：扫到 schema_version=1 PNG 用 _migrate_anima_params 适配。
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
            for img in mode_dir.glob("*.png"):
                if img.name.endswith(".tmp.png"):
                    continue  # atomic write tmp file 兜底
                params = _read_png_anima_params(img)
                if params is None:
                    continue
                params = _migrate_anima_params(params)
                try:
                    created_at = img.stat().st_mtime
                except OSError:
                    continue
                encoded = _url_quote_filename(img.name)
                out.append({
                    "id": _disk_history_id(date_dir.name, mode, img.name),
                    "date": date_dir.name,
                    "mode": mode,
                    "filename": img.name,
                    "path": str(img),
                    "image_url": f"/api/generate/disk/image/{date_dir.name}/{mode}/{encoded}",
                    "thumb_url": f"/api/generate/disk/thumb/{date_dir.name}/{mode}/{encoded}?w=128",
                    "created_at": float(created_at),
                    "schema_version": int(params.get("schema_version", SCHEMA_VERSION)),
                    "params": params,
                })
    out.sort(key=lambda e: e["created_at"], reverse=True)
    return out[:limit]


@router.get("/api/generate/disk/history")
def list_disk_history(limit: int = 500) -> dict[str, Any]:
    """列出所有落盘测试图（按 PNG `anima_params` tEXt 扫），按 created_at desc 排。

    前端历史栏拉一次 merge 到 IndexedDB 视图；entry.id 稳定，前端按 id dedup。
    没有 anima_params 的图（老数据 / 客户端没传 params）不入列表。
    """
    limit = max(1, min(int(limit), 2000))
    return {"entries": _scan_png_metadata(limit)}


def _resolve_disk_png(date_str: str, mode: str, filename: str) -> Path:
    """三种 endpoint（image / thumb / delete）共用的路径校验 + resolve。

    校验：date 格式 / mode 枚举 / filename 安全字符集（无 / \ .. 等）/ 扩展名 .png
    返回：实际磁盘 Path（不保证 exists，由调用方决定 404 时机）
    """
    if not _DATE_RE.match(date_str):
        raise HTTPException(400, "invalid date")
    if mode not in _DISK_MODES:
        raise HTTPException(400, "invalid mode")
    if not _PNG_NAME_SAFE_RE.match(filename):
        raise HTTPException(400, "invalid filename")
    # 二次防御：safe_join 反 traversal
    base = (TEST_IMAGES_DIR / date_str / mode).resolve()
    try:
        path = (base / filename).resolve()
    except OSError:
        raise HTTPException(400, "invalid filename")
    if not str(path).startswith(str(base)):
        raise HTTPException(400, "path escapes base dir")
    return path


@router.get("/api/generate/disk/image/{date_str}/{mode}/{filename}")
def get_disk_image(date_str: str, mode: str, filename: str) -> Any:
    """读落盘测试图（前端历史栏点击磁盘 entry 时大图来源）。"""
    path = _resolve_disk_png(date_str, mode, filename)
    if not path.is_file():
        raise HTTPException(404)
    # 落盘图内容稳定（序号递增不覆盖），可强 cache
    return FileResponse(
        path, media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/api/generate/disk/thumb/{date_str}/{mode}/{filename}")
def get_disk_thumb(
    date_str: str, mode: str, filename: str,
    w: int = Query(128, ge=32, le=512),
) -> Any:
    """PIL 在线生成缩略图（决策 Dev v1 / Arch v2）—— 替代前端 IDB dataURL cache。

    - ETag = sha1(file mtime + size + w)；304 命中直接返
    - Cache-Control: public, max-age=86400（落盘图内容稳定）
    - 失败 fallback 原图（避免缩略生成 bug 阻塞历史栏）
    """
    path = _resolve_disk_png(date_str, mode, filename)
    if not path.is_file():
        raise HTTPException(404)
    try:
        st = path.stat()
        etag = hashlib.sha1(
            f"{st.st_mtime}:{st.st_size}:{w}".encode("utf-8")
        ).hexdigest()[:16]
    except OSError:
        raise HTTPException(404)
    # 这里没有直接读 request header，由 FastAPI / Starlette 处理 304 略复杂；
    # 简化方案：返 ETag + Cache-Control，浏览器自管 304 转换（max-age 内不再请求）
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.thumbnail((w, w), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = buf.getvalue()
    except Exception:
        return FileResponse(path, media_type="image/png")
    return Response(
        content=data,
        media_type="image/png",
        headers={
            "ETag": f'"{etag}"',
            "Cache-Control": "public, max-age=86400",
        },
    )


@router.delete("/api/generate/disk/{date_str}/{mode}/{filename}")
def delete_disk_image(date_str: str, mode: str, filename: str) -> dict[str, Any]:
    """删除落盘测试图（前端历史栏单条删除）。

    返回 OK + 是否真删（noop=True 表示文件本不存在）。安全校验同 image / thumb。
    """
    path = _resolve_disk_png(date_str, mode, filename)
    if not path.is_file():
        return {"ok": True, "noop": True}
    try:
        path.unlink()
        return {"ok": True, "noop": False}
    except OSError as e:
        raise HTTPException(500, f"delete failed: {e}")
