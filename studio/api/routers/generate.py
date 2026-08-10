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

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import shutil
import time
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
from ...domain.common import supports_capability
from ...infrastructure.event_bus import bus
from ...infrastructure.paths import STUDIO_DATA
from ...services import generate_history_index as history_index
from ...services import generate_storage as storage
from ...services.generate_history_index import (
    DATE_RE as _DATE_RE,
    SCHEMA_VERSION,
    XY_COMPOSITE_NAME as _XY_COMPOSITE_NAME,
    XY_FOLDER_RE as _XY_FOLDER_RE,
)
from ...services.generation_metadata import (
    build_external_metadata,
    write_manifest as write_generation_metadata_manifest,
)

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
# _XY_FOLDER_RE / _XY_COMPOSITE_NAME / _DATE_RE 移到 services.generate_history_index
#（索引服务与本 router 共用一套布局约定），顶部 import 回来。
_XY_TMP_FOLDER_RE = re.compile(r"^\.xy plot \d+\.tmp$")

# 路径校验（disk-image / thumb / delete 全套共用）
_DISK_MODES = ("single", "xy")
_PNG_NAME_SAFE_RE = re.compile(r"^[a-zA-Z0-9 ._-]+\.png$")


# 命名序号 / 原子写 / PNG 注入 / server-side enrich 已下沉 services.generate_storage
#（出图时间线 DB 单源:server 直落闭环与 /save 端点共用同一份实现;/save 端点
# 前端已不再调用,PR-B 退役)。alias 保持本模块内既有调用点 / 测试引用不变。
_next_image_index = storage.next_image_index
_next_xy_folder_index = storage.next_xy_folder_index
_atomic_write_png = storage.atomic_write_png
_format_a1111_parameters = storage.format_a1111_parameters
_inject_png_metadata = storage.inject_png_metadata
_enrich_params_server_side = storage.enrich_params_server_side
_build_external_metadata_safe = storage._build_external_metadata_safe


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
    # TE variant 覆盖（krea2）：请求显式给 bf16/fp8 时覆盖 selected_te 默认
    # （default_paths 已按 selected_te 解析）；fp8 未下载给可操作报错。
    if body.text_encoder and body.model_family == "krea2":
        from ...services.models.families.krea2 import qwen3_vl_dir_for
        from ...services.models.paths import models_root

        te_dir = qwen3_vl_dir_for(models_root(), body.text_encoder)
        if body.text_encoder == "fp8" and not (te_dir / "config.json").exists():
            raise HTTPException(
                status_code=409,
                detail="Qwen3-VL fp8 文本编码器未下载——请到 设置 → 模型下载 "
                       "下载后重试。",
            )
        model_paths["text_encoder_path"] = str(te_dir)
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
            lora_merge_precision = str(
                getattr(gen_cfg, "lora_merge_precision", "fp32") or "fp32"
            )
            vram_policy = str(getattr(gen_cfg, "vram_policy", "auto") or "auto")
            ram_guard = bool(getattr(gen_cfg, "ram_guard", False))
            blocks_to_swap = int(getattr(gen_cfg, "blocks_to_swap", 0) or 0)
        except Exception:
            attn_default = "auto"
            preview_n = 0
            vae_precision = "bf16"
            lora_merge_precision = "fp32"
            vram_policy = "auto"
            ram_guard = False
            blocks_to_swap = 0
        attn = body.attention_backend or attn_default
        if attn == "auto":
            from ...services.runtime.xformers import detect_attention_backend
            attn = detect_attention_backend()

        # 族条件门控：blocks_to_swap 来自**全局**出图设置，是用户为某个模型调的，
        # 但这次请求可以是另一个族的底模。原样透传时 daemon 在加载 DiT 时 fail-fast
        # （"model_family=X 不支持 block swap"），整个出图直接崩。用户并没有为这个
        # 模型要求 block swap，所以忽略而不是报错。
        if blocks_to_swap and not supports_capability(body.model_family, "block_swap"):
            logger.info(
                "model_family=%s 不支持 block swap，本次出图忽略全局设置的 "
                "blocks_to_swap=%s", body.model_family, blocks_to_swap,
            )
            blocks_to_swap = 0

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
            lora_merge_precision=lora_merge_precision,
            attention_backend=attn,
            vram_policy=vram_policy,
            ram_guard=ram_guard,
            blocks_to_swap=blocks_to_swap,
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

        # Civitai/A1111 资源身份不能塞进可移植的 anima_params（其中禁止绝对
        # 路径），单独写 task 私有档案。task 结束只清 anima_gen_<id> 临时目录，
        # tasks/<id>/ 档案会保留到用户删任务；落盘接口凭 task_id 读取并算 hash。
        try:
            write_generation_metadata_manifest(
                task_id,
                prompts=list(body.prompts),
                model_family=body.model_family,
                model_path=str(model_paths.get("transformer_path") or ""),
                vae_path=str(model_paths.get("vae_path") or "") or None,
                text_encoder=body.text_encoder,
                loras=[lc.model_dump() for lc in body.lora_configs],
                xy_matrix=body.xy_matrix.model_dump() if body.xy_matrix else None,
            )
        except (OSError, TypeError, ValueError):
            # metadata 是附加能力，档案写入失败不能让已经校验通过的生成任务失败。
            logger.warning("write generation metadata manifest failed", exc_info=True)

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


_TOKENIZER_CACHE: dict[str, Any] = {}


@router.post("/api/generate/token_count")
def count_prompt_tokens(body: dict) -> dict[str, Any]:
    """prompt 的真实 token 数（前端角标用；tokenizer 与训练/推理同源）。

    krea2 的文本条件训练口径 512 token，超出部分模型没见过（不拦截、
    不警告——质量后果由用户掌握，前端只给中性计数）。tokenizer 惰性
    加载并缓存；不可用时返回 tokens=null，前端隐藏角标。
    """
    text = str(body.get("text") or "")
    family = str(body.get("model_family") or "anima")
    try:
        from ...services.models.paths import models_root

        if family == "krea2":
            from ...services.models.families.krea2 import (
                qwen3_vl_dir_for, selected_te_variant,
            )

            tok_dir = str(qwen3_vl_dir_for(models_root(), selected_te_variant()))
        else:
            from ...services.models.families.anima import qwen_dir

            tok_dir = str(qwen_dir(models_root()))
        tokenizer = _TOKENIZER_CACHE.get(tok_dir)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tok_dir, local_files_only=True,
            )
            _TOKENIZER_CACHE[tok_dir] = tokenizer
        tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        return {"tokens": tokens}
    except Exception:
        return {"tokens": None}


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
        # 落盘 fallback:save=on 时图落盘成功后 cache 中转副本即被 drop
        #（generate_storage 闭环),live 显示 / composite 拼图仍按 daemon
        # filename 走本端点 → 按台账 src 反查磁盘文件。
        disk_path = storage.find_disk_file(task_id, filename)
        if disk_path is not None:
            return FileResponse(
                disk_path, media_type="image/png",
                headers={"Cache-Control": "no-store"},
            )
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


# SCHEMA_VERSION 移到 services.generate_history_index（顶部 import 回来）


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
    source_filename: str = Form(""),
    cells: list[UploadFile] = File(default=[]),
    cells_manifest: str = Form(""),
) -> dict[str, Any]:
    """落盘测试出图。

    **single mode** → `studio_data/test/<YYYY-MM-DD>/single/single image <N>.png`
    返回 `{path, index, filename}` —— `cells` / `cells_manifest` 必须空，否则 400。

    **xy mode** → `studio_data/test/<YYYY-MM-DD>/xy/xy plot <N>/{xy plot.png, cell x<i> y<j>.png ...}`
    - `image` = composite 大图（导出 + 缩略图来源），按 mode='xy' 注 anima_params，不写 a1111
    - `cells` = 每格原图 N 张；`cells_manifest` = JSON 数组
      [{xi:int, yi:int, params:dict, source_filename:str}]，与 `cells` 同序；每 cell
      按 mode='single' 注 anima_params + a1111
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
            external = await asyncio.to_thread(
                _build_external_metadata_safe,
                task_id,
                enriched,
                source_filename=source_filename,
            )
            raw = _inject_png_metadata(raw, enriched, mode=mode, external=external)

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
    cell_specs: list[tuple[int, int, dict[str, Any], str]] = []
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
        cell_source_filename = str(entry.get("source_filename") or "")
        cell_specs.append((xi, yi, cell_params or {}, cell_source_filename))

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
        for (xi, yi, cell_params, cell_source_filename), cb in zip(
            cell_specs, cell_bytes_list
        ):
            cell_payload = cb
            if cell_params:
                enriched_cell = _enrich_params_server_side(cell_params, task_id=task_id, mode="single")
                external = await asyncio.to_thread(
                    _build_external_metadata_safe,
                    task_id,
                    enriched_cell,
                    source_filename=cell_source_filename,
                )
                cell_payload = _inject_png_metadata(
                    cell_payload, enriched_cell, mode="single", external=external,
                )
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
# 出图时间线（DB 单源）：tasks 表是唯一台账，行=一次图片任务。
# 蓝图 tmp/generate-timeline-db-refactor-plan.md；替代 disk 扫盘 ∪ cache index
# 双源（旧双源端点 PR-B 退役）。
# ---------------------------------------------------------------------------


def _disk_image_urls(rel: str) -> Optional[tuple[str, str]]:
    """generate_images 的 file 相对路径 → (image_url, thumb_url)。

    single: `<date>/single/<fn>`（3 段）；xy cell: `<date>/xy/<folder>/<fn>`
    （4 段）。段数对不上（台账被外部改坏）→ None 跳过该图。
    """
    parts = rel.split("/")
    if len(parts) == 3:
        d, m, fn = parts
        enc = quote(fn, safe="")
        return (
            f"/api/generate/disk/image/{d}/{m}/{enc}",
            f"/api/generate/disk/thumb/{d}/{m}/{enc}?w=128",
        )
    if len(parts) == 4 and parts[1] == "xy":
        d, _, folder, fn = parts
        enc_f = quote(folder, safe="")
        enc = quote(fn, safe="")
        return (
            f"/api/generate/disk/image/{d}/xy/{enc_f}/{enc}",
            f"/api/generate/disk/thumb/{d}/xy/{enc_f}/{enc}?w=128",
        )
    return None


def _timeline_entry(task: dict[str, Any]) -> Optional[dict[str, Any]]:
    """task 行 → 时间线 entry。failed/canceled 且无图的行不进时间线
    （与旧右栏行为一致：失败任务不留条目；canceled 的 XY 部分图保留）。"""
    from ...services.inference import disk_cache as generate_cache

    status = str(task.get("status") or "")
    raw_images = task.get("generate_images")
    try:
        images_raw = json.loads(raw_images) if raw_images else []
    except json.JSONDecodeError:
        images_raw = []
    if not isinstance(images_raw, list):
        images_raw = []
    if status in ("failed", "canceled") and not images_raw:
        return None

    params: Optional[dict[str, Any]] = None
    raw_params = task.get("generate_params")
    if raw_params:
        try:
            decoded = json.loads(raw_params)
            if isinstance(decoded, dict):
                params = decoded
        except json.JSONDecodeError:
            pass

    task_id = int(task["id"])
    try:
        cache_filenames = set(generate_cache.list_filenames(task_id))
    except RuntimeError:
        cache_filenames = set()

    images: list[dict[str, Any]] = []
    available = False
    first = True
    xy_folder: Optional[str] = None
    for it in images_raw:
        if not isinstance(it, dict):
            continue
        img: dict[str, Any] = {}
        rel = it.get("file")
        cache_fn = it.get("cache")
        if rel:
            urls = _disk_image_urls(str(rel))
            if urls is None:
                continue
            img["url"], img["thumb_url"] = urls
            parts = str(rel).split("/")
            if len(parts) == 4:
                xy_folder = parts[2]
            if first:
                available = (TEST_IMAGES_DIR / str(rel)).is_file()
        elif cache_fn:
            img["url"] = (
                f"/api/generate/{task_id}/sample/{quote(str(cache_fn), safe='')}"
            )
            if first:
                available = str(cache_fn) in cache_filenames
        else:
            continue
        if "xi" in it:
            img["xi"] = int(it.get("xi") or 0)
            img["yi"] = int(it.get("yi") or 0)
        images.append(img)
        first = False

    mode = str((params or {}).get("mode") or ("xy" if xy_folder else "single"))
    entry: dict[str, Any] = {
        "task_id": task_id,
        "status": status,
        "created_at": task.get("created_at"),
        "mode": mode,
        "storage": "disk" if any(i.get("file") for i in images_raw
                                 if isinstance(i, dict)) else "temp",
        "params": params,
        "images": images,
        "available": available,
    }
    if xy_folder is not None:
        entry["xy_folder"] = xy_folder
        composite_rel = None
        for it in images_raw:
            if isinstance(it, dict) and it.get("file"):
                composite_rel = "/".join(
                    str(it["file"]).split("/")[:3] + [_XY_COMPOSITE_NAME]
                )
                break
        if composite_rel and (TEST_IMAGES_DIR / composite_rel).is_file():
            urls = _disk_image_urls(composite_rel)
            if urls:
                entry["composite_url"] = urls[0]
    return entry


@router.get("/api/generate/timeline")
def generate_timeline(limit: int = 200, offset: int = 0) -> dict[str, Any]:
    """出图时间线：所有 generate 任务行，`id DESC` 分页。

    pending/running 行天然在内（enqueue 即有行，前端不再单独拉 live 队列合并）；
    done 行按 generate_images 拼图 URL；图不在（temp 会话结束 / 用户手删文件）
    → `available=false`，前端显示「已释放」，参数仍可回填。
    """
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))
    with db.connection_for() as conn:
        rows = db.list_tasks_page(
            conn, statuses=(), types=("generate",), limit=limit, offset=offset,
        )
        total = db.count_tasks(conn, statuses=(), types=("generate",))
    entries = [e for e in (_timeline_entry(t) for t in rows) if e is not None]
    return {"entries": entries, "total": total, "offset": offset}


@router.post("/api/generate/{task_id}/xy-composite")
async def attach_xy_composite(
    task_id: int, image: UploadFile = File(...),
) -> dict[str, Any]:
    """XY composite 补传（决策 1：盘上仍要有大图，外站上传用）。

    前端在 task done 后用 composeXYMatrix 现拼 POST 一张；server 写入该 task
    的 xy 文件夹（排 storage executor，天然序在所有 cell 落盘之后）。参数注入
    取 DB generate_params，不信前端传参。composite 不入 generate_images
    （应用内回看用 cells 渲网格）。
    """
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task or task.get("task_type") != "generate":
        raise NotFoundError(
            "Task not found", code="task.not_found",
            details={"task_id": task_id}, http_status=404,
        )
    raw = await image.read()
    if not raw:
        raise ValidationError(
            "The uploaded image is empty",
            code="generate.empty_image", http_status=400,
        )
    try:
        target = await asyncio.to_thread(storage.attach_xy_composite, task_id, raw)
    except LookupError:
        raise ConflictError(
            "Task has no xy folder on disk",
            code="generate.no_xy_folder",
            details={"task_id": task_id}, http_status=409,
        )
    return {"path": str(target)}


# ---------------------------------------------------------------------------
# 磁盘历史浏览：SQLite 索引（services.generate_history_index，sync-on-read）
# 列 entries；图片 URL 单独服务。扫描/解析/迁移逻辑全在索引服务里。
# ---------------------------------------------------------------------------


@router.get("/api/generate/disk/history")
def list_disk_history(limit: int = 2000) -> dict[str, Any]:
    """列出所有落盘测试图，按 created_at desc 排。

    数据来自 sync-on-read 的 SQLite 索引（PNG 仍是唯一 canonical，索引可
    随时删除重建）：请求先做 scandir 快照 diff，只解析新增/变化的 PNG ——
    落盘图上千张后每次进页面的全量重扫从秒级降到 ~10ms 级。

    entry.id 稳定，前端按 id dedup。没有 anima_params 的图（老数据 /
    客户端没传 params）不入列表。默认 limit 从 500 提到 2000 —— 索引化后
    500 截断没有存在意义，老历史应该列得出来。
    """
    limit = max(1, min(int(limit), 10000))
    return {"entries": history_index.sync_and_list(TEST_IMAGES_DIR, limit)}


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
    except OSError as e:
        raise HTTPException(500, f"delete failed: {e}")
    history_index.remove_entry(TEST_IMAGES_DIR, date_str, "xy", folder)
    return {"ok": True, "noop": False}


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
    except OSError as e:
        raise HTTPException(500, f"delete failed: {e}")
    history_index.remove_entry(TEST_IMAGES_DIR, date_str, mode, filename)
    return {"ok": True, "noop": False}
