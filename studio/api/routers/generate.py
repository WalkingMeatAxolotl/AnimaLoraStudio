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
import logging
import os
import re
import shutil
import time
import zlib
from datetime import date
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from ..deps import _resolve_model_paths
from ..errors import _validate_component_or_400
from ..schemas.generate import GenerateRequest
from ... import db, secrets
from ...domain import GenerateConfig
from ...domain.errors import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from ...domain.comfy_parity import force_comfy_parity_runtime_config
from ...infrastructure.event_bus import bus
from ...infrastructure.paths import STUDIO_DATA

router = APIRouter()
logger = logging.getLogger(__name__)

TEST_IMAGES_DIR = STUDIO_DATA / "test"


def _write_generate_cover(task_id: Optional[int], cover_path: Path) -> None:
    """0.17 P-I forward-write：落盘时把封面图（磁盘）相对地址写进 task.generate_cover
    （相对 TEST_IMAGES_DIR，_v14 列）。前端暂不读；未来 DB 驱动出图时间线据此定位/判
    存在。task_id 缺省（老前端/异常）或写失败时静默跳过——纯攒未来数据，不影响出图。"""
    if task_id is None:
        return
    try:
        rel = str(cover_path.relative_to(TEST_IMAGES_DIR))
    except ValueError:
        rel = str(cover_path)
    try:
        with db.connection_for() as conn:
            db.update_task(conn, task_id, generate_cover=rel)
    except Exception:
        logger.warning("write generate_cover for task %s failed", task_id, exc_info=True)

# v2 命名（决策 #6）：父目录区分 mode，文件名仅 "<label> N.png"
_DISPLAY_LABELS = {"single": "single image", "xy": "xy plot"}
_V2_SINGLE_RE = re.compile(r"^single image (\d+)\.png$")
_V2_XY_RE = re.compile(r"^xy plot (\d+)\.png$")
# v1 legacy：image_N.png（旧版命名），扫描时仍读取，但新写入只用 v2
_V1_NAME_RE = re.compile(r"^image_(\d+)\.png$")

# XY 文件夹布局（恢复 PreviewXYGrid 历史回看）：
#   <date>/xy/xy plot <N>/{xy plot.png, cell x<i> y<j>.png, ...}
# composite 是合成大图（导出 + 缩略图来源）；cell 是每格原图（PreviewXYGrid + 拖进 Comfy）
_XY_FOLDER_RE = re.compile(r"^xy plot (\d+)$")
_XY_TMP_FOLDER_RE = re.compile(r"^\.xy plot \d+\.tmp$")
_XY_CELL_RE = re.compile(r"^cell x(\d+) y(\d+)\.png$")
_XY_COMPOSITE_NAME = "xy plot.png"

# 路径校验（disk-image / thumb / delete 全套共用）
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DISK_MODES = ("single", "xy")
_PNG_NAME_SAFE_RE = re.compile(r"^[a-zA-Z0-9 ._-]+\.png$")


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


def _next_xy_folder_index(xy_dir: Path) -> int:
    """XY 模式下一个文件夹 1-based 序号。

    扫两个空间防撞：
    - 新格式子文件夹 `xy plot N/`（_XY_FOLDER_RE）
    - legacy 平铺文件 `xy plot N.png`（_V2_XY_RE） —— PR #245 早期落盘的，
      虽然不会出现在 history 但残留磁盘上时不能复用编号

    决策 #11：单用户无并发跑图，不做锁；扫描 + atomic mkdir 即可。
    """
    if not xy_dir.is_dir():
        return 1
    max_n = 0
    for p in xy_dir.iterdir():
        if p.is_dir():
            m = _XY_FOLDER_RE.match(p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
        elif p.is_file():
            m = _V2_XY_RE.match(p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _cleanup_xy_tmp_folders() -> None:
    """import-time 清理上次 server crash 留下的 `.xy plot N.tmp/` 半成品。

    save 流程：先写到 sibling tmp 文件夹，全部 cell 落盘后 os.replace 成
    正式名。中途 crash 会留 tmp 文件夹。每次模块 import 扫一遍清。
    """
    if not TEST_IMAGES_DIR.is_dir():
        return
    for date_dir in TEST_IMAGES_DIR.iterdir():
        if not date_dir.is_dir() or not _DATE_RE.match(date_dir.name):
            continue
        xy_dir = date_dir / "xy"
        if not xy_dir.is_dir():
            continue
        for p in xy_dir.iterdir():
            if p.is_dir() and _XY_TMP_FOLDER_RE.match(p.name):
                shutil.rmtree(p, ignore_errors=True)


# import-time 清理（上次 server crash 留下的 tmp 文件夹）
_cleanup_xy_tmp_folders()


@router.post("/api/generate")
def enqueue_generate(body: GenerateRequest) -> dict[str, Any]:
    """启动测试出图 task。"""
    from ...services.inference.core import generate_tempdir
    from ...services.models.families import get_assets

    model_paths = _resolve_model_paths(body.base_model, family=body.model_family)
    # Turbo 检测（A4/C9）：官方蒸馏 variant → daemon 走 8 步/guidance 0/固定 mu
    # 的采样时刻表默认；custom 权重无 purpose 元数据按非蒸馏处理
    distilled = bool(get_assets(body.model_family).is_distilled_path(
        model_paths.get("transformer_path", "")))

    with db.connection_for() as conn:
        task_id = db.create_task(
            conn, name="generate", config_name="generate", priority=0,
        )
        db.update_task(conn, task_id, task_type="generate")

    # create_task 已把 task 落成 pending+generate，但 config_path 还没写；supervisor
    # _dispatch_exclusive_tasks 会跳过 config_path=NULL 的 generate task（视为还在入队），等
    # 下面 config.json 落库后再派。这里任一步失败必须把 task 标 failed，否则它会以
    # config_path=NULL 永远 pending（dispatcher 永远跳过）。
    try:
        tempdir = generate_tempdir(task_id)
        tempdir.mkdir(parents=True, exist_ok=True)

        # 测试出图走 Comfy-style runtime。xformers backend 可提供 pinned oracle
        # 的 exact KSampler parity；flash_attn/none 可生成但不保证 exact parity。
        # preview 节流仍读 settings；训练 / RegAI 的 backend 选择不受影响。
        try:
            gen_cfg = secrets.load().generate
            attn_default = gen_cfg.attention_backend
            preview_n = int(gen_cfg.preview_every_n_steps or 0)
            vae_precision = str(getattr(gen_cfg, "vae_precision", "bf16") or "bf16")
            vram_policy = str(getattr(gen_cfg, "vram_policy", "auto") or "auto")
            te_precision = str(getattr(gen_cfg, "te_precision", "fp16") or "fp16")
        except Exception:
            attn_default = "auto"
            preview_n = 0
            vae_precision = "bf16"
            vram_policy = "auto"
            te_precision = "fp16"
        attn = body.attention_backend or attn_default
        if attn == "auto":
            from ...services.runtime.xformers import detect_attention_backend
            attn = detect_attention_backend()

        cfg = GenerateConfig(
            **model_paths,
            model_family=body.model_family,
            distilled=distilled,
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
            mixed_precision="bf16",
            vae_precision=vae_precision,
            attention_backend=attn,
            vram_policy=vram_policy,
            te_precision=te_precision,
            xy_matrix=body.xy_matrix.model_dump() if body.xy_matrix else None,
        )

        # commit 14：注入 daemon 端用的 preview 节流参数（settings 全局开关）
        cfg_dict = force_comfy_parity_runtime_config(
            cfg.model_dump(),
            force_exact_ksampler_backend=False,
        )
        cfg_dict["preview_every_n_steps"] = preview_n

        # 决策 #15：task 启动时冻结 save_test_images，避免用户中途切开关导致
        # 一 task 内一半图走 cache 一半落盘。daemon submit_task 读这个字段
        # 存到 _ActiveTask.save_to_disk，_handle_image_done 决定 SSE delivery
        try:
            cfg_dict["save_test_images_at_dispatch"] = bool(
                secrets.load().generate.save_test_images
            )
        except Exception:
            cfg_dict["save_test_images_at_dispatch"] = False

        # 前端 params snapshot 透传：路由 → config.json → supervisor →
        # daemon.submit_task → _ActiveTask.params_snapshot → cache.put 时
        # 跟 PNG bytes 一起塞进加密 payload header。下划线前缀提示 daemon
        # 子进程不读这个字段（cfg 透传不解析）。
        if body.params_snapshot:
            cfg_dict["_anima_params_snapshot_"] = body.params_snapshot

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

    # 0.17 P-I forward-write：把参数快照落 DB（generate_params 列，_v14）。前端暂不读，
    # 为未来「纯 DB 出图时间线」攒数据——那时按 params + generate_cover(出图完成时写)
    # 直接定位/回填，无需扫盘、无需迁移。参数就是前端随 body 发来的 params_snapshot。
    generate_params = (
        json.dumps(body.params_snapshot, ensure_ascii=False)
        if body.params_snapshot else None
    )
    with db.connection_for() as conn:
        db.update_task(
            conn, task_id, config_path=str(cfg_path), generate_params=generate_params,
        )
        task = db.get_task(conn, task_id)

    bus.publish({"type": "task_state_changed", "task_id": task_id, "status": "pending"})
    return task or {"id": task_id}


@router.get("/api/generate/{task_id}")
def get_generate_task(task_id: int) -> dict[str, Any]:
    """查询测试 task 状态。"""
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task or task.get("task_type") != "generate":
        raise NotFoundError(
            "Task not found", code="task.not_found",
            details={"task_id": task_id}, http_status=404,
        )
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
        raise ValidationError(
            "Failed to download the preview model; check the server log",
            code="generate.preview_model_download_failed", http_status=500,
        )
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
        raise ConflictError(
            "Inference service is busy; try again after the current task finishes",
            code="generate.daemon_busy", http_status=409,
        )
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
        raise ValidationError(
            "Select a .png file", code="file.ext_invalid",
            details={"types": ".png"}, http_status=400,
        )
    from ...services.inference import disk_cache as generate_cache
    data = generate_cache.get_image(task_id, filename)
    if data is None:
        raise NotFoundError(
            "Image not found", code="image.not_found",
            details={"task_id": task_id, "filename": filename}, http_status=404,
        )
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


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _decode_png_text_chunk(ctype: bytes, data: bytes) -> str | None:
    """从 PNG 文本 chunk 取出 keyword==`anima_params` 的文本；非该 keyword /
    解析失败返 None。

    - tEXt：keyword\\0 + latin-1 明文
    - zTXt：keyword\\0 + 压缩方法(1 字节) + zlib 压缩流（latin-1）
    - iTXt：keyword\\0 + 压缩 flag(1) + 压缩方法(1) + 语言\\0 + 翻译 keyword\\0 + 文本(utf-8)

    解码用 PIL 读 PNG 文本块的同一套规则（tEXt/zTXt → latin-1，iTXt → utf-8），
    保证与旧 PIL 实现逐值一致。
    """
    keyword, sep, rest = data.partition(b"\x00")
    if not sep or keyword != b"anima_params":
        return None
    try:
        if ctype == b"tEXt":
            return rest.decode("latin-1")
        if ctype == b"zTXt":
            return zlib.decompress(rest[1:]).decode("latin-1") if rest else None
        if ctype == b"iTXt":
            if len(rest) < 2:
                return None
            comp_flag = rest[0]
            body = rest[2:]
            _, _, body = body.partition(b"\x00")  # 跳语言 tag
            _, _, body = body.partition(b"\x00")  # 跳翻译 keyword
            return (zlib.decompress(body) if comp_flag else body).decode("utf-8")
    except Exception:
        return None
    return None


def _read_png_anima_params(path: Path) -> dict[str, Any] | None:
    """从 PNG `anima_params` tEXt / zTXt / iTXt 块解析 params；无 / 解析失败返 None。

    直接顺序扫 PNG chunk，读到第一个 IDAT（像素数据起始）前找 `anima_params`
    文本块即停。`anima_params` 由 `PngInfo.add_text(..., zip=True)` 写成 zTXt 且
    位于 IDAT 之前，必然在 header 区命中。

    原实现走 `PIL.Image.open` 只读 header（不 decode 像素），但实测 PIL open 每
    文件 ~30-40ms，disk-history 扫数百张落盘图要 10-15s（冷缓存可达 ~1min）。
    手写 chunk 扫描每文件 ~0.1ms（实测 356 张 15s → 0.04s，~350×），且对全部
    历史 PNG 与旧实现逐值一致。
    """
    try:
        with open(path, "rb") as f:
            if f.read(8) != _PNG_SIGNATURE:
                return None
            while True:
                head = f.read(8)
                if len(head) < 8:
                    return None
                length = int.from_bytes(head[:4], "big")
                ctype = head[4:8]
                if ctype == b"IDAT":
                    return None  # 到像素数据，header 区没有 anima_params
                if ctype in (b"tEXt", b"zTXt", b"iTXt"):
                    text = _decode_png_text_chunk(ctype, f.read(length))
                    f.read(4)  # CRC
                    if text is not None:
                        parsed = json.loads(text)
                        return parsed if isinstance(parsed, dict) else None
                else:
                    f.seek(length + 4, 1)  # 跳 chunk data + CRC（IHDR 等）
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


def _decode_params_field(raw: str, field: str) -> dict[str, Any]:
    """`params` / per-cell manifest 元素 → dict。失败抛 HTTPException 400."""
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValidationError(
            "Image parameters are not valid JSON",
            code="generate.params_invalid",
            details={"field": field, "reason": str(e)}, http_status=400,
        ) from e
    if not isinstance(decoded, dict):
        raise ValidationError(
            "Image parameters are not valid JSON",
            code="generate.params_invalid",
            details={"field": field}, http_status=400,
        )
    return decoded


@router.post("/api/generate/save")
async def save_test_image(
    mode: str = Form(...),
    image: UploadFile = File(...),
    params: str = Form(""),
    task_id: Optional[int] = Form(None),
    cells: list[UploadFile] = File(default=[]),
    cells_manifest: str = Form(""),
) -> dict[str, Any]:
    """落盘测试出图。

    **single mode** → `studio_data/test/<YYYY-MM-DD>/single/single image <N>.png`
    返回 `{path, index, filename}` —— `cells` / `cells_manifest` 必须空，否则 400。

    **xy mode** → `studio_data/test/<YYYY-MM-DD>/xy/xy plot <N>/{xy plot.png, cell x<i> y<j>.png ...}`
    - `image` = composite 大图（导出 + 缩略图来源），按 mode='xy' 注 anima_params，不写 a1111
    - `cells` = 每格原图 N 张；`cells_manifest` = JSON 数组 [{xi:int, yi:int, params:dict}]，
      与 `cells` 同序；每 cell 按 mode='single' 注 anima_params + a1111
    - 校验：len(cells)==len(manifest)，无重复 (xi,yi)
    - atomic：先写 sibling `.xy plot <N>.tmp/`，全部 cell 落盘后 `os.replace` 成正式名；
      任一步失败 → `shutil.rmtree(tmp)` 抛 500
    - 返回 `{folder, composite, cells: [path,...]}`

    其它（含 "compare"）→ 400. Settings.save_test_images=False → 403.
    server 端 enrich 强制 schema_version/created_at/task_id/mode。
    """
    if mode not in ("single", "xy"):
        raise ValidationError(
            f"Unsupported mode: {mode}", code="generate.mode_invalid",
            details={"mode": mode}, http_status=400,
        )
    if not secrets.load().generate.save_test_images:
        raise ForbiddenError(
            "Saving test images is disabled",
            code="generate.save_disabled", http_status=403,
        )
    raw = await image.read()
    if not raw:
        raise ValidationError(
            "The uploaded image is empty",
            code="generate.empty_image", http_status=400,
        )

    if mode == "single":
        if cells or cells_manifest:
            raise HTTPException(400, "single mode does not accept cells")
        if params:
            decoded = _decode_params_field(params, "params")
            enriched = _enrich_params_server_side(decoded, task_id=task_id, mode=mode)
            raw = _inject_png_metadata(raw, enriched, mode=mode)

        target_dir = TEST_IMAGES_DIR / date.today().isoformat() / mode
        target_dir.mkdir(parents=True, exist_ok=True)
        idx = _next_image_index(target_dir, mode)
        target = target_dir / f"{_DISPLAY_LABELS[mode]} {idx}.png"
        _atomic_write_png(target, raw)
        _write_generate_cover(task_id, target)  # 0.17 P-I forward-write
        return {"path": str(target), "index": idx, "filename": target.name}

    # ----- mode == "xy" -----
    if not cells_manifest:
        raise HTTPException(400, "xy mode requires cells_manifest")
    try:
        manifest = json.loads(cells_manifest)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"cells_manifest: invalid JSON ({e})")
    if not isinstance(manifest, list):
        raise HTTPException(400, "cells_manifest: must be a JSON array")
    if len(manifest) != len(cells):
        raise HTTPException(400, f"cells_manifest length {len(manifest)} != cells {len(cells)}")
    if not cells:
        raise HTTPException(400, "xy mode requires at least one cell")

    # 校验 manifest 条目 + 收集 (xi, yi) 防重
    seen_xy: set[tuple[int, int]] = set()
    cell_specs: list[tuple[int, int, dict[str, Any]]] = []
    for i, entry in enumerate(manifest):
        if not isinstance(entry, dict):
            raise HTTPException(400, f"cells_manifest[{i}]: must be an object")
        try:
            xi = int(entry["xi"])
            yi = int(entry["yi"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, f"cells_manifest[{i}]: missing xi/yi")
        if xi < 0 or yi < 0:
            raise HTTPException(400, f"cells_manifest[{i}]: xi/yi must be non-negative")
        if (xi, yi) in seen_xy:
            raise HTTPException(400, f"cells_manifest[{i}]: duplicate (xi={xi}, yi={yi})")
        seen_xy.add((xi, yi))
        cell_params = entry.get("params")
        if cell_params is not None and not isinstance(cell_params, dict):
            raise HTTPException(400, f"cells_manifest[{i}].params: must be a JSON object")
        cell_specs.append((xi, yi, cell_params or {}))

    # composite 注入 anima_params（mode='xy'，不写 a1111）
    composite_bytes = raw
    if params:
        composite_decoded = _decode_params_field(params, "params")
        composite_enriched = _enrich_params_server_side(composite_decoded, task_id=task_id, mode="xy")
        composite_bytes = _inject_png_metadata(composite_bytes, composite_enriched, mode="xy")

    # 读所有 cell bytes（在文件夹分配前，避免半写）
    cell_bytes_list: list[bytes] = []
    for i, cell_upload in enumerate(cells):
        cb = await cell_upload.read()
        if not cb:
            raise HTTPException(400, f"cells[{i}]: empty body")
        cell_bytes_list.append(cb)

    # 分配 folder + tmp 路径
    xy_dir = TEST_IMAGES_DIR / date.today().isoformat() / "xy"
    xy_dir.mkdir(parents=True, exist_ok=True)
    idx = _next_xy_folder_index(xy_dir)
    final_dir = xy_dir / f"{_DISPLAY_LABELS['xy']} {idx}"
    tmp_dir = xy_dir / f".{_DISPLAY_LABELS['xy']} {idx}.tmp"
    if final_dir.exists():
        raise HTTPException(500, f"folder collision: {final_dir} already exists")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    try:
        tmp_dir.mkdir(parents=False, exist_ok=False)
        # composite
        _atomic_write_png(tmp_dir / _XY_COMPOSITE_NAME, composite_bytes)
        # cells
        cell_paths: list[Path] = []
        for (xi, yi, cell_params), cb in zip(cell_specs, cell_bytes_list):
            cell_payload = cb
            if cell_params:
                enriched_cell = _enrich_params_server_side(cell_params, task_id=task_id, mode="single")
                cell_payload = _inject_png_metadata(cell_payload, enriched_cell, mode="single")
            cell_path = tmp_dir / f"cell x{xi} y{yi}.png"
            _atomic_write_png(cell_path, cell_payload)
            cell_paths.append(cell_path)
        # atomic rename tmp → final (Windows: target must not exist, 我们刚 _next_xy_folder_index 保证)
        os.replace(tmp_dir, final_dir)
    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, f"failed to write xy folder: {e}")

    _write_generate_cover(task_id, final_dir / _XY_COMPOSITE_NAME)  # 0.17 P-I forward-write
    return {
        "folder": str(final_dir),
        "index": idx,
        "composite": str(final_dir / _XY_COMPOSITE_NAME),
        "cells": [str(final_dir / p.name) for p in cell_paths],
    }


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
    return quote(filename, safe="")


def _scan_single_dir(single_dir: Path, date_str: str) -> list[dict[str, Any]]:
    """扫一个 `<date>/single/` 目录的所有 PNG，返回 disk-history entry 列表。"""
    out: list[dict[str, Any]] = []
    for img in single_dir.glob("*.png"):
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
            "id": _disk_history_id(date_str, "single", img.name),
            "date": date_str,
            "mode": "single",
            "filename": img.name,
            "path": str(img),
            "image_url": f"/api/generate/disk/image/{date_str}/single/{encoded}",
            "thumb_url": f"/api/generate/disk/thumb/{date_str}/single/{encoded}?w=128",
            "created_at": float(created_at),
            "schema_version": int(params.get("schema_version", SCHEMA_VERSION)),
            "params": params,
        })
    return out


def _build_xy_meta_from_folder(
    folder: Path, composite_params: dict[str, Any], date_str: str, folder_name: str,
) -> dict[str, Any] | None:
    """读 XY 文件夹下所有 cell 文件，按 composite 的 xy_draft 反查 xv/yv，
    拼成 disk-history entry 的 `xy_meta` 字段。

    决策：只读 composite 的 anima_params（一次 file open），cell 的 xi/yi
    从文件名 parse（regex），xv/yv 从 composite.xy_draft.x.raw/y.raw split
    后查表。**不**逐 cell 打开 anima_params —— 5×5 矩阵 1 次 open 而非 26 次。

    返回 None 表示 composite 缺 xy_draft（异常状态，前端兜底走 <img>）。
    """
    xy_draft = composite_params.get("xy_draft")
    if not isinstance(xy_draft, dict):
        return None
    x_axis_info = xy_draft.get("x")
    if not isinstance(x_axis_info, dict):
        return None
    x_raw = str(x_axis_info.get("raw", ""))
    x_values = [s.strip() for s in x_raw.split(",") if s.strip()]
    x_axis = x_axis_info.get("axis")

    y_axis_info = xy_draft.get("y") if xy_draft.get("y") else None
    y_values: list[str | None]
    y_axis: str | None
    if isinstance(y_axis_info, dict):
        y_raw = str(y_axis_info.get("raw", ""))
        y_values = [s.strip() for s in y_raw.split(",") if s.strip()]
        y_axis = y_axis_info.get("axis")
    else:
        y_values = [None]
        y_axis = None

    samples: list[dict[str, Any]] = []
    for cell_file in folder.glob("cell x*.png"):
        m = _XY_CELL_RE.match(cell_file.name)
        if not m:
            continue
        xi = int(m.group(1))
        yi = int(m.group(2))
        xv: str | None = x_values[xi] if 0 <= xi < len(x_values) else None
        yv: str | None = y_values[yi] if 0 <= yi < len(y_values) else None
        enc_folder = _url_quote_filename(folder_name)
        enc_file = _url_quote_filename(cell_file.name)
        samples.append({
            "path": cell_file.name,
            "xy": {"xi": xi, "yi": yi, "xv": xv, "yv": yv},
            "image_url": f"/api/generate/disk/image/{date_str}/xy/{enc_folder}/{enc_file}",
        })
    samples.sort(key=lambda s: (s["xy"]["yi"], s["xy"]["xi"]))
    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "x_values": x_values,
        "y_values": y_values,
        "samples": samples,
    }


def _scan_xy_dir(xy_dir: Path, date_str: str) -> list[dict[str, Any]]:
    """扫一个 `<date>/xy/` 目录 —— 只看子文件夹（新布局），跳所有平铺文件（legacy）。

    每个匹配 `_XY_FOLDER_RE` 的子文件夹：
    - 必须有 `xy plot.png` composite，否则跳过整个 folder
    - 读 composite 的 anima_params，调 `_build_xy_meta_from_folder` 出 cells
    """
    out: list[dict[str, Any]] = []
    for folder in xy_dir.iterdir():
        if not folder.is_dir():
            continue  # legacy 平铺 `xy plot N.png` 文件不再出现在 history（用户决策）
        if not _XY_FOLDER_RE.match(folder.name):
            continue
        composite = folder / _XY_COMPOSITE_NAME
        if not composite.is_file():
            continue  # 没 composite 的文件夹（半成品 / 用户手动 mkdir）跳过
        params = _read_png_anima_params(composite)
        if params is None:
            continue
        params = _migrate_anima_params(params)
        try:
            created_at = composite.stat().st_mtime
        except OSError:
            continue
        xy_meta = _build_xy_meta_from_folder(folder, params, date_str, folder.name)
        enc_folder = _url_quote_filename(folder.name)
        enc_composite = _url_quote_filename(_XY_COMPOSITE_NAME)
        out.append({
            "id": _disk_history_id(date_str, "xy", folder.name),
            "date": date_str,
            "mode": "xy",
            "folder": folder.name,
            "path": str(folder),
            "image_url": f"/api/generate/disk/image/{date_str}/xy/{enc_folder}/{enc_composite}",
            "thumb_url": f"/api/generate/disk/thumb/{date_str}/xy/{enc_folder}/{enc_composite}?w=128",
            "created_at": float(created_at),
            "schema_version": int(params.get("schema_version", SCHEMA_VERSION)),
            "params": params,
            "xy_meta": xy_meta,
        })
    return out


def _scan_png_metadata(limit: int) -> list[dict[str, Any]]:
    """扫 TEST_IMAGES_DIR 下所有 <date>/{single,xy}/ 的 anima_params。

    - single 模式：扫 `<date>/single/*.png`
    - xy 模式：扫 `<date>/xy/xy plot <N>/{composite + cells}` 子文件夹
      （legacy 平铺 `<date>/xy/xy plot N.png` 不入列表 —— 用户决策 hide）

    没 anima_params 的 PNG 不入列表（老数据 / 客户端没传 params）。
    决策 #16：composite 只读 header（不 load 像素）；cell 不读 PNG（用文件名 parse + composite xy_draft 反查 xv/yv）。
    决策 #18：v1→v2 migrate。
    """
    if not TEST_IMAGES_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for date_dir in TEST_IMAGES_DIR.iterdir():
        if not date_dir.is_dir() or not _DATE_RE.match(date_dir.name):
            continue
        single_dir = date_dir / "single"
        if single_dir.is_dir():
            out.extend(_scan_single_dir(single_dir, date_dir.name))
        xy_dir = date_dir / "xy"
        if xy_dir.is_dir():
            out.extend(_scan_xy_dir(xy_dir, date_dir.name))
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


@router.get("/api/generate/cache/index")
def list_cache_index() -> dict[str, Any]:
    """当前 session 加密磁盘 cache 里所有 entry 的索引（save_test_images=false
    时前端历史栏唯一来源）。

    server 进程 SessionCache 维护活跃 entry → 这里直接 dump；按 createdAt
    desc 排。entry 里的 params snapshot 是图入 cache 时跟 PNG bytes 一起塞
    进加密 payload header 的那份，进程死了一起没。

    刷新 / 切路由都拉这里 → 前端零持久化层，零脏数据可能。
    """
    from ...services.inference import disk_cache as generate_cache
    try:
        return {"entries": generate_cache.list_index()}
    except RuntimeError:
        # cache 尚未 init（理论上不该发生，lifespan startup 已建好）
        return {"entries": []}


def _resolve_disk_png(date_str: str, mode: str, filename: str) -> Path:
    r"""三种 endpoint（image / thumb / delete）共用的路径校验 + resolve。

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
        raise NotFoundError(
            "Image not found", code="image.not_found",
            details={"date": date_str, "mode": mode, "filename": filename},
            http_status=404,
        )
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
        raise NotFoundError(
            "Image not found", code="image.not_found",
            details={"date": date_str, "mode": mode, "filename": filename},
            http_status=404,
        )
    try:
        st = path.stat()
        etag = hashlib.sha1(
            f"{st.st_mtime}:{st.st_size}:{w}".encode("utf-8")
        ).hexdigest()[:16]
    except OSError as exc:
        raise NotFoundError(
            "Image not found", code="image.not_found",
            details={"date": date_str, "mode": mode, "filename": filename},
            http_status=404,
        ) from exc
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


# ---------------------------------------------------------------------------
# XY 文件夹专用 routes（新布局）
#
# 注：DELETE /api/generate/disk/<date>/xy/<folder> 必须在
# `delete_disk_image`（5 段通配 {date}/{mode}/{filename}）之前注册，
# 否则后者会先匹配（FastAPI 按注册顺序匹配，3 段通配会先吞 xy/<folder>）。
# ---------------------------------------------------------------------------


def _resolve_disk_xy_cell(date_str: str, folder: str, filename: str) -> Path:
    """XY 文件夹下 composite / cell 文件的路径校验 + resolve。

    校验：date / folder（必须匹配 `xy plot N`）/ filename（_PNG_NAME_SAFE_RE）。
    返回实际磁盘 Path（不保证 exists）。
    """
    if not _DATE_RE.match(date_str):
        raise HTTPException(400, "invalid date")
    if not _XY_FOLDER_RE.match(folder):
        raise HTTPException(400, "invalid folder")
    if not _PNG_NAME_SAFE_RE.match(filename):
        raise HTTPException(400, "invalid filename")
    base = (TEST_IMAGES_DIR / date_str / "xy" / folder).resolve()
    try:
        path = (base / filename).resolve()
    except OSError:
        raise HTTPException(400, "invalid filename")
    if not str(path).startswith(str(base)):
        raise HTTPException(400, "path escapes base dir")
    return path


@router.get("/api/generate/disk/image/{date_str}/xy/{folder}/{filename}")
def get_disk_xy_image(date_str: str, folder: str, filename: str) -> Any:
    """读 XY 文件夹下的 composite 或 cell PNG（PreviewXYGrid 回看 + 拖进 Comfy 复用）。"""
    path = _resolve_disk_xy_cell(date_str, folder, filename)
    if not path.is_file():
        raise NotFoundError(
            "Image not found", code="image.not_found",
            details={"date": date_str, "folder": folder, "filename": filename},
            http_status=404,
        )
    return FileResponse(
        path, media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/api/generate/disk/thumb/{date_str}/xy/{folder}/{filename}")
def get_disk_xy_thumb(
    date_str: str, folder: str, filename: str,
    w: int = Query(128, ge=32, le=512),
) -> Any:
    """XY 文件夹下文件的 PIL 缩略图（历史栏 thumb_url 用 composite）。"""
    path = _resolve_disk_xy_cell(date_str, folder, filename)
    if not path.is_file():
        raise NotFoundError(
            "Image not found", code="image.not_found",
            details={"date": date_str, "folder": folder, "filename": filename},
            http_status=404,
        )
    try:
        st = path.stat()
        etag = hashlib.sha1(
            f"{st.st_mtime}:{st.st_size}:{w}".encode("utf-8")
        ).hexdigest()[:16]
    except OSError as exc:
        raise NotFoundError(
            "Image not found", code="image.not_found",
            details={"date": date_str, "folder": folder, "filename": filename},
            http_status=404,
        ) from exc
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


@router.delete("/api/generate/disk/{date_str}/xy/{folder}")
def delete_disk_xy_folder(date_str: str, folder: str) -> dict[str, Any]:
    """删除整个 XY 文件夹（composite + 所有 cell）。

    历史栏点 × 时调；返回 OK + 是否真删（noop=True 表示文件夹本不存在）。
    """
    if not _DATE_RE.match(date_str):
        raise HTTPException(400, "invalid date")
    if not _XY_FOLDER_RE.match(folder):
        raise HTTPException(400, "invalid folder")
    base = (TEST_IMAGES_DIR / date_str / "xy" / folder).resolve()
    test_root = TEST_IMAGES_DIR.resolve()
    if not str(base).startswith(str(test_root)):
        raise HTTPException(400, "path escapes base dir")
    if not base.is_dir():
        return {"ok": True, "noop": True}
    try:
        shutil.rmtree(base)
        return {"ok": True, "noop": False}
    except OSError as e:
        raise HTTPException(500, f"delete failed: {e}")


@router.delete("/api/generate/disk/{date_str}/{mode}/{filename}")
def delete_disk_image(date_str: str, mode: str, filename: str) -> dict[str, Any]:
    """删除落盘单文件测试图（single 模式 / admin 清 legacy XY 平铺文件）。

    XY 模式新布局走 `delete_disk_xy_folder`；这条路由保留主要是 single
    与 legacy flat XY 清理。注册顺序在 XY folder DELETE 之后 —— 否则 3 段通配
    会先吞 `xy/<folder>` 路径（FastAPI 按注册顺序匹配）。
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
