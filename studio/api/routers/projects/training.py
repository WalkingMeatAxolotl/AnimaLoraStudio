"""tag + captions + reg + version_config + 入队训练 + version thumb
（PR-6.5 commit 5 从 server.py 抽出）。

23 routes：

  tagging (1)
    POST /api/projects/{pid}/versions/{vid}/tag

  captions (8)
    GET /captions    PUT/GET /captions/{folder}/{filename}
    POST /captions/snapshot  GET /captions/snapshots
    POST /captions/snapshots/{sid}/restore  DELETE /captions/snapshots/{sid}
    POST /captions/commit  POST /captions/batch

  reg (5)
    GET /reg/preview-tags   GET /reg   POST /reg/build
    GET /reg/caption   DELETE /reg

  reg_ai (2)
    POST /reg/generate-prior   GET /reg/generate-prior/{task_id}

  version_config (4)
    GET /config  PUT /config  POST /config/from_preset  POST /config/save_as_preset

  training launch (1)
    POST /api/projects/{pid}/versions/{vid}/queue

  version thumb (1)
    GET /api/projects/{pid}/versions/{vid}/thumb
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...deps import _resolve_anima_model_paths
from ...errors import _preset_err_code as _err_code, _safe_join_or_400
from ...responses import _thumb_response
from ...schemas.training import (
    BatchOp,
    CaptionEdit,
    CommitRequest,
    FromPresetRequest,
    RegAiRequest,
    RegBuildRequest,
    RegDeleteFilesRequest,
    SaveAsPresetRequest,
    TagJobRequest,
)
from ._shared import (
    _project_and_version_or_404,
    _publish_job_state,
    _publish_version_state,
    _reg_dir,
    _version_dir_or_404,
    _version_train_dir_or_404,
)
from .... import db
from ....services.presets import io as presets_io
from ....services.projects import jobs as project_jobs, projects, versions
from ....services.dataset import scan as datasets
from ....domain import RegAiConfig
from ....infrastructure.event_bus import bus
from ....paths import STUDIO_DATA, safe_join
from ....services import model_downloader, version_config
from ....services import presets as preset_flow
from ....services.tagging import caption_snapshot
from ....services.reg import builder as reg_builder, dedup as reg_dedup
from ....services.dataset import tagedit
from ....services.tagging.base import VALID_TAGGER_NAMES

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# /api/projects/{pid}/versions/{vid}/tag  (PP4)
# ---------------------------------------------------------------------------


@router.post("/api/projects/{pid}/versions/{vid}/tag")
def start_tag(pid: int, vid: int, body: TagJobRequest) -> dict[str, Any]:
    if body.tagger not in VALID_TAGGER_NAMES:
        raise HTTPException(400, f"unknown tagger: {body.tagger}")
    if body.output_format not in {"txt", "json"}:
        raise HTTPException(400, "output_format must be txt|json")
    if body.on_existing not in {"overwrite", "skip", "append"}:
        raise HTTPException(400, "on_existing must be overwrite|skip|append")
    _, v, _ = _version_train_dir_or_404(pid, vid)

    # 触发词：先 strip，落到 version 表（持久化，TagEdit / Train 都能读），再
    # 顺手放进 worker params。body.trigger_word=None 表示前端没传字段（不改
    # version 现有值）；空串 "" 表示用户主动清空。
    trigger_word = body.trigger_word.strip() if body.trigger_word is not None else None

    params: dict[str, Any] = {
        "tagger": body.tagger,
        "version_id": vid,
        "output_format": body.output_format,
    }
    # 默认值 "overwrite" 不写入 params（worker 端默认就是 overwrite），减小 payload。
    if body.on_existing != "overwrite":
        params["on_existing"] = body.on_existing
    if trigger_word:
        params["trigger_word"] = trigger_word
    # 通用：按 tagger 名取 `<name>_overrides` 字段并落到 params 同名键。
    # 仅保留用户实际填写的字段；空 dict 也不写。
    overrides_field = getattr(body, f"{body.tagger}_overrides", None)
    if overrides_field is not None:
        ov = overrides_field.model_dump(exclude_none=True)
        if ov:
            params[f"{body.tagger}_overrides"] = ov

    with db.connection_for() as conn:
        if trigger_word is not None and trigger_word != (v.get("trigger_word") or ""):
            updated = versions.update_version(conn, vid, trigger_word=trigger_word)
            _publish_version_state(updated)
            v = updated
        job = project_jobs.create_job(
            conn,
            project_id=pid,
            version_id=vid,
            kind="tag",
            params=params,
        )
    _publish_job_state(job)
    return job


# ---------------------------------------------------------------------------
# Captions read/write + snapshots + commit/batch
# ---------------------------------------------------------------------------


@router.get("/api/projects/{pid}/versions/{vid}/captions")
def list_captions_endpoint(
    pid: int, vid: int, folder: Optional[str] = None, full: bool = False,
) -> dict[str, Any]:
    _, _, train = _version_train_dir_or_404(pid, vid)
    if folder is None:
        return {"folder": None, "items": tagedit.list_all_captions(train, full=full)}
    _safe_join_or_400(train, folder)
    return {
        "folder": folder,
        "items": tagedit.list_captions_in_folder(train, folder, full=full),
    }


@router.get("/api/projects/{pid}/versions/{vid}/captions/{folder}/{filename}")
def get_caption_endpoint(
    pid: int, vid: int, folder: str, filename: str,
) -> dict[str, Any]:
    _, _, train = _version_train_dir_or_404(pid, vid)
    _safe_join_or_400(train, folder, filename)
    try:
        return tagedit.read_one(train, folder, filename)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.put("/api/projects/{pid}/versions/{vid}/captions/{folder}/{filename}")
def put_caption_endpoint(
    pid: int, vid: int, folder: str, filename: str, body: CaptionEdit,
) -> dict[str, Any]:
    _, _, train = _version_train_dir_or_404(pid, vid)
    _safe_join_or_400(train, folder, filename)
    try:
        return tagedit.write_one(train, folder, filename, body.tags)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/api/projects/{pid}/versions/{vid}/captions/snapshot")
def create_caption_snapshot(pid: int, vid: int) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    return caption_snapshot.create_snapshot(vdir)


@router.get("/api/projects/{pid}/versions/{vid}/captions/snapshots")
def list_caption_snapshots(pid: int, vid: int) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    return {"items": caption_snapshot.list_snapshots(vdir)}


@router.post("/api/projects/{pid}/versions/{vid}/captions/snapshots/{sid}/restore")
def restore_caption_snapshot(pid: int, vid: int, sid: str) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    try:
        return caption_snapshot.restore_snapshot(vdir, sid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except caption_snapshot.SnapshotError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/api/projects/{pid}/versions/{vid}/captions/snapshots/{sid}")
def delete_caption_snapshot(pid: int, vid: int, sid: str) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    try:
        caption_snapshot.delete_snapshot(vdir, sid)
        return {"deleted": sid}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except caption_snapshot.SnapshotError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/api/projects/{pid}/versions/{vid}/captions/commit")
def commit_captions(pid: int, vid: int, body: CommitRequest) -> dict[str, Any]:
    """一次性写入多个 caption；写之前自动生成快照作还原点。"""
    _, _, vdir = _version_dir_or_404(pid, vid)
    train = vdir / "train"
    snap = caption_snapshot.create_snapshot(vdir)
    written = 0
    skipped: list[str] = []
    for it in body.items:
        try:
            img = safe_join(train, it.folder, it.name)
        except ValueError:
            skipped.append(f"{it.folder}/{it.name}")
            continue
        if not img.exists():
            skipped.append(f"{it.folder}/{it.name}")
            continue
        tagedit.write_tags(img, it.tags)
        written += 1
    return {"snapshot": snap, "written": written, "skipped": skipped}


@router.post("/api/projects/{pid}/versions/{vid}/captions/batch")
def batch_caption_endpoint(
    pid: int, vid: int, body: BatchOp,
) -> dict[str, Any]:
    _, _, train = _version_train_dir_or_404(pid, vid)
    op = body.op
    scope = body.scope
    if op == "add":
        n = tagedit.add_tags(
            scope, train, body.tags or [],
            position="front" if body.position == "front" else "back",
        )
        return {"op": op, "affected": n}
    if op == "remove":
        return {"op": op, "affected": tagedit.remove_tags(scope, train, body.tags or [])}
    if op == "replace":
        if not body.old or not body.new:
            raise HTTPException(400, "replace 需要 old 和 new")
        return {"op": op, "affected": tagedit.replace_tag(scope, train, body.old, body.new)}
    if op == "dedupe":
        return {"op": op, "affected": tagedit.dedupe(scope, train)}
    if op == "stats":
        return {"op": op, "items": tagedit.stats(scope, train, top=max(1, body.top))}
    raise HTTPException(400, f"unknown op: {op}")


# ---------------------------------------------------------------------------
# /api/projects/{pid}/versions/{vid}/reg  (PP5)
# ---------------------------------------------------------------------------


@router.get("/api/projects/{pid}/versions/{vid}/reg/preview-tags")
def reg_preview_tags(pid: int, vid: int, top: int = 20) -> dict[str, Any]:
    """返回 train 的 tag 频率 top N（不真生成 reg）。给 UI「排除 tag」勾选用。"""
    _, _, vdir = _version_dir_or_404(pid, vid)
    train = vdir / "train"
    items = reg_builder.preview_train_tag_distribution(train, top=max(1, top))
    return {"items": [{"tag": t, "count": c} for t, c in items]}


@router.get("/api/projects/{pid}/versions/{vid}/reg")
def get_reg_status(pid: int, vid: int) -> dict[str, Any]:
    """返回 reg 集状态（meta + 图片数 + 文件名列表）。"""
    _, _, vdir = _version_dir_or_404(pid, vid)
    rdir = _reg_dir(vdir)
    if not rdir.exists():
        return {"exists": False, "meta": None, "image_count": 0, "files": []}
    images: list[str] = []
    for f in sorted(rdir.rglob("*")):
        if f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS:
            try:
                rel = f.relative_to(rdir).as_posix()
            except ValueError:
                continue
            images.append(rel)
    meta = reg_builder.read_meta(rdir)
    meta_dict = None
    if meta is not None:
        meta_dict = asdict(meta)
    return {
        "exists": bool(images) or meta is not None,
        "meta": meta_dict,
        "image_count": len(images),
        "files": images,
    }


# A3 — reg auto-tag 本轮只在 UI 暴露 wd14 / cltagger。底层 VALID_TAGGER_NAMES
# 含 LLM / JoyCaption，但它们对 reg 体积（>train）慢/贵，留单独 PR；422 校验
# 兜底防 contributor 误传。
_REG_TAGGER_ALLOWED = {"wd14", "cltagger"}


@router.post("/api/projects/{pid}/versions/{vid}/reg/build")
def start_reg_build(pid: int, vid: int, body: RegBuildRequest) -> dict[str, Any]:
    if body.api_source not in {"gelbooru", "danbooru"}:
        raise HTTPException(400, "api_source must be gelbooru|danbooru")
    if body.postprocess_method not in {"smart", "stretch", "crop"}:
        raise HTTPException(400, "postprocess_method must be smart|stretch|crop")
    if not (0.05 <= body.postprocess_max_crop_ratio <= 0.5):
        raise HTTPException(400, "postprocess_max_crop_ratio must be 0.05–0.5")
    if body.aspect_ratio_filter_enabled and not (
        0.0 < body.min_aspect_ratio < body.max_aspect_ratio
    ):
        raise HTTPException(400, "min_aspect_ratio must be < max_aspect_ratio (both > 0)")
    if body.auto_tag_kind not in _REG_TAGGER_ALLOWED:
        raise HTTPException(
            422,
            f"auto_tag_kind must be one of {sorted(_REG_TAGGER_ALLOWED)}",
        )
    # B1 — build_mode + target_count 校验
    if body.build_mode not in {"mirror", "flat"}:
        raise HTTPException(422, "build_mode must be 'mirror' or 'flat'")
    if body.target_count is not None and body.target_count <= 0:
        raise HTTPException(422, "target_count must be > 0 or null")
    _, v, vdir = _version_dir_or_404(pid, vid)
    train = vdir / "train"
    has_image = train.exists() and any(
        f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS
        for f in train.rglob("*")
    )
    if not has_image:
        raise HTTPException(400, "train 还没有图片，先去 ① 整理 / ② 下载")

    with db.connection_for() as conn:
        job = project_jobs.create_job(
            conn,
            project_id=pid,
            version_id=vid,
            kind="reg_build",
            params={
                "version_id": vid,
                "excluded_tags": list(body.excluded_tags),
                "auto_tag": bool(body.auto_tag),
                "auto_tag_kind": body.auto_tag_kind,
                "auto_dedup": bool(body.auto_dedup),
                "build_mode": body.build_mode,
                "target_count": body.target_count,
                "api_source": body.api_source,
                "incremental": bool(body.incremental),
                "skip_similar": bool(body.skip_similar),
                "aspect_ratio_filter_enabled": bool(body.aspect_ratio_filter_enabled),
                "min_aspect_ratio": float(body.min_aspect_ratio),
                "max_aspect_ratio": float(body.max_aspect_ratio),
                "postprocess_method": body.postprocess_method,
                "postprocess_max_crop_ratio": float(body.postprocess_max_crop_ratio),
            },
        )
    _publish_job_state(job)
    return job


@router.get("/api/projects/{pid}/versions/{vid}/reg/caption")
def get_reg_caption(pid: int, vid: int, path: str) -> dict[str, Any]:
    """读 reg 集中单张图的 caption。`path` 是相对 reg/ 的路径（含子文件夹）。"""
    if not path:
        raise HTTPException(400, "invalid path")
    _, _, vdir = _version_dir_or_404(pid, vid)
    rdir = _reg_dir(vdir)
    # path 允许含 `/` 子目录；按分隔符拆成片段交给 safe_join 做组件校验 + containment
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    img = _safe_join_or_400(rdir, *parts)
    if not img.exists() or img.suffix.lower() not in datasets.IMAGE_EXTS:
        raise HTTPException(404, "image not found")
    return {"path": path, "tags": tagedit.read_tags(img)}


@router.post("/api/projects/{pid}/versions/{vid}/reg/generate-prior")
def reg_generate_prior(pid: int, vid: int, body: RegAiRequest) -> dict[str, Any]:
    """启动先验生成 task —— base 模型给每张 train 图的 tag 反向出对照图。"""
    model_paths = _resolve_anima_model_paths()
    _, _, vdir = _version_dir_or_404(pid, vid)
    train = vdir / "train"
    has_image = train.exists() and any(
        f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS
        for f in train.rglob("*")
    )
    if not has_image:
        raise HTTPException(400, "train 还没有图片，请先完成 Step 1（下载）或 Step 2（筛选）")

    rdir = _reg_dir(vdir)
    rdir.mkdir(parents=True, exist_ok=True)

    from ....services.runtime.xformers import detect_attention_backend
    cfg = RegAiConfig(
        **model_paths,
        train_dir=str(train),
        reg_dir=str(rdir),
        excluded_tags=list(body.excluded_tags),
        negative_prompt=body.negative_prompt,
        width=body.width,
        height=body.height,
        steps=body.steps,
        cfg_scale=body.cfg_scale,
        sampler_name=body.sampler_name,
        scheduler=body.scheduler,
        seed=body.seed,
        incremental=body.incremental,
        mixed_precision=body.mixed_precision,
        attention_backend=detect_attention_backend(),
    )

    cfg_dir = STUDIO_DATA / "reg_ai_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    with db.connection_for() as conn:
        task_id = db.create_task(
            conn, name=f"reg-prior p{pid}v{vid}", config_name="reg_ai", priority=0,
        )
        db.update_task(
            conn, task_id, task_type="reg_ai", project_id=pid, version_id=vid,
        )

    cfg_path = cfg_dir / f"reg_ai_{task_id}.json"
    cfg_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

    with db.connection_for() as conn:
        db.update_task(conn, task_id, config_path=str(cfg_path))
        task = db.get_task(conn, task_id)

    bus.publish({"type": "task_state_changed", "task_id": task_id, "status": "pending"})
    return task or {"id": task_id}


@router.get("/api/projects/{pid}/versions/{vid}/reg/generate-prior/{task_id}")
def get_reg_prior_task(pid: int, vid: int, task_id: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task or task.get("task_type") != "reg_ai":
        raise HTTPException(404)
    return task


@router.delete("/api/projects/{pid}/versions/{vid}/reg")
def delete_reg(pid: int, vid: int) -> dict[str, Any]:
    """清空 reg/ 内容（含 meta.json + 所有子文件夹），保留空目录本身。

    `versions.create_version` 总会建空 reg/；判定「存在」= 有 meta 或图片。
    """
    import shutil as _shutil
    _, _, vdir = _version_dir_or_404(pid, vid)
    rdir = _reg_dir(vdir)
    has_content = rdir.exists() and (
        (rdir / "meta.json").exists()
        or any(
            f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS
            for f in rdir.rglob("*")
        )
    )
    if not has_content:
        return {"deleted": False, "reason": "reg empty"}
    try:
        for child in rdir.iterdir():
            if child.is_dir():
                _shutil.rmtree(child)
            else:
                child.unlink()
    except OSError as exc:
        raise HTTPException(500, f"删除失败: {exc}") from exc
    return {"deleted": True}


@router.post("/api/projects/{pid}/versions/{vid}/reg/delete-files")
def delete_reg_files(
    pid: int, vid: int, body: RegDeleteFilesRequest,
) -> dict[str, Any]:
    """按相对路径批量删 reg 集中的图（含同名 .txt caption），更新 meta.actual_count,
    把删除的 booru ID（文件名 stem）追加到 reg/.deleted_ids.json。

    增量补足时 builder 会读这个文件，把 ID 加进 search exclude，避免同一张
    booru post 又被拉回来。

    body.relative_paths 是相对 reg/ 的路径列表，跨子文件夹也可以；路径越界
    走 _safe_join_or_400 抛 400。
    """
    if not body.relative_paths:
        raise HTTPException(400, "relative_paths is empty")
    _, _, vdir = _version_dir_or_404(pid, vid)
    rdir = _reg_dir(vdir)
    if not rdir.exists():
        raise HTTPException(404, "reg dir not found")

    # 端点入口：每条路径走 _safe_join_or_400 防 traversal；合法的转回 rdir
    # 相对形式喂给 reg_dedup.purge_paths。worker 走 dedup.scan_for_dedup
    # 拿到的路径必合法，所以 dedup 模块内部不再做 traversal 校验。
    validated_rels: list[str] = []
    for rel in body.relative_paths:
        if not rel:
            continue
        parts = [p for p in rel.replace("\\", "/").split("/") if p]
        if not parts:
            continue
        target = _safe_join_or_400(rdir, *parts)
        validated_rels.append(target.relative_to(rdir).as_posix())

    return reg_dedup.purge_paths(rdir, validated_rels)


@router.post("/api/projects/{pid}/versions/{vid}/reg/dedup-purge")
def dedup_purge_reg(pid: int, vid: int) -> dict[str, Any]:
    """A4 — 用 preprocess dedup 默认参数扫 reg 集，自动删每组里"推荐删除"项
    （每组保留 group[0]，其余删 + 写 .deleted_ids.json + meta 递减）。

    用户手动入口；worker 在 auto_dedup=True 时 build 后会用同一套
    reg_dedup 模块自动跑。reg 集 quality bar 比 train 低 → 不弹 review panel。

    同步返回；图量大时慢（O(n^2)）。
    """
    _, _, vdir = _version_dir_or_404(pid, vid)
    rdir = _reg_dir(vdir)
    if not rdir.exists():
        raise HTTPException(404, "reg dir not found")

    to_delete = reg_dedup.scan_for_dedup(rdir)
    scanned = sum(
        1 for f in rdir.rglob("*")
        if f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS
    )
    if not to_delete:
        return {"scanned": scanned, "groups": 0, "deleted": [], "count": 0}

    result = reg_dedup.purge_paths(rdir, to_delete)
    return {
        "scanned": scanned,
        # groups 用「待删项数」近似（每组 >= 1 张被删）；scan_for_dedup 不
        # 暴露 group 总数，用户也只关心删了多少。
        "groups": len(to_delete),
        "deleted": result["deleted"],
        "count": result["count"],
    }


# ---------------------------------------------------------------------------
# /api/projects/{pid}/versions/{vid}/config  (PP6.2 训练配置 — version 私有)
# ---------------------------------------------------------------------------


@router.get("/api/projects/{pid}/versions/{vid}/config")
def get_version_config_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """读 version 私有 config；不存在返回 has_config=false / config=null。

    无论 has_config 与否都返回 `project_specific_defaults` —— fork preset 时
    后端将自动注入的项目预填值（项目路径 + 全局模型路径 + reg 检测结果）。
    前端「+ 新建预设」可以在 version 已有 config 的状态下被点（替换当前预设），
    所以这个 hint 跟 has_config 状态无关，永远要返回。
    """
    project, ver = _project_and_version_or_404(pid, vid)
    psf = sorted(version_config.PROJECT_SPECIFIC_FIELDS)
    psd = {
        **version_config.project_specific_overrides(project, ver),
        **model_downloader.default_paths_for_new_version(),
    }
    if not version_config.has_version_config(project, ver):
        return {
            "has_config": False,
            "config": None,
            "project_specific_fields": psf,
            "project_specific_defaults": psd,
        }
    try:
        cfg, dropped, defaulted = version_config.read_version_config_with_warnings(project, ver)
    except version_config.VersionConfigError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "has_config": True,
        "config": cfg,
        "project_specific_fields": psf,
        "project_specific_defaults": psd,
        "dropped_fields": dropped,
        "defaulted_fields": defaulted,
    }


@router.put("/api/projects/{pid}/versions/{vid}/config")
def put_version_config_endpoint(
    pid: int, vid: int, body: dict[str, Any],
) -> dict[str, Any]:
    """直接写 version 私有 config（全量替换）。

    PP10.4：项目特定字段（data_dir / output_dir / output_name 等）**不**强制
    覆盖。fork_preset 时已经预填好；用户在 Train 页可以自由改（例如
    `resume_lora` 接续训练、自定义 output_name）。改坏了再换一次预设回到
    默认。
    """
    project, ver = _project_and_version_or_404(pid, vid)
    try:
        version_config.write_version_config(
            project, ver, body, force_project_overrides=False,
        )
        cfg = version_config.read_version_config(project, ver)
    except version_config.VersionConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"has_config": True, "config": cfg}


@router.post("/api/projects/{pid}/versions/{vid}/config/from_preset")
def fork_preset_for_version_endpoint(
    pid: int, vid: int, body: FromPresetRequest,
) -> dict[str, Any]:
    """从全局 preset 复制一份进 version 私有 config（应用项目特定字段）。"""
    project, ver = _project_and_version_or_404(pid, vid)
    try:
        cfg, dropped, defaulted = preset_flow.fork_preset_for_version_with_warnings(
            body.name, project, ver
        )
    except presets_io.PresetError as exc:
        _err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    except version_config.VersionConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    # 同步 versions.config_name = 来源 preset 名（informational only）
    with db.connection_for() as conn:
        versions.update_version(conn, vid, config_name=body.name)
    return {
        "has_config": True,
        "config": cfg,
        "from_preset": body.name,
        "dropped_fields": dropped,
        "defaulted_fields": defaulted,
    }


@router.post("/api/projects/{pid}/versions/{vid}/config/save_as_preset")
def save_version_config_as_preset_endpoint(
    pid: int, vid: int, body: SaveAsPresetRequest,
) -> dict[str, Any]:
    """version 私有 config → 全局 preset（清掉项目特定字段）。"""
    project, ver = _project_and_version_or_404(pid, vid)
    try:
        cfg = preset_flow.save_version_config_as_preset(
            project, ver, body.name, overwrite=body.overwrite,
        )
    except presets_io.PresetError as exc:
        _err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    except version_config.VersionConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"saved_preset": body.name, "config": cfg}


@router.post("/api/projects/{pid}/versions/{vid}/queue")
def enqueue_version_training(pid: int, vid: int) -> dict[str, Any]:
    """PP6.3 — 把 version 入队训练。

    校验：
    - version 已配置训练参数（version_config 存在）
    - 该 version 没有 active task（pending / running）
    """
    project, ver = _project_and_version_or_404(pid, vid)
    if not version_config.has_version_config(project, ver):
        raise HTTPException(
            400, "请先在 ⑥ 训练页选预设并保存配置后再入队",
        )
    cfg_path = version_config.version_config_path(project, ver)

    with db.connection_for() as conn:
        # 该 version 当前是否已有 active task
        active = conn.execute(
            "SELECT id, status FROM tasks "
            "WHERE version_id = ? AND status IN ('pending', 'running') "
            "LIMIT 1",
            (vid,),
        ).fetchone()
        if active:
            raise HTTPException(
                409,
                f"该版本已有 active task #{active['id']}（{active['status']}），"
                "请等其完成或取消",
            )

        # 创建 task
        slug = project["slug"]
        label = ver["label"]
        task_name = f"{slug}_{label}"
        config_name = ver["config_name"] or f"proj_{pid}_{label}"  # informational
        # ADR-0009 PR-1 C6: 同 db.create_task — 存 ContextVar trace_id
        from studio.infrastructure.logging import get_trace_id, new_trace_id
        req_tid = get_trace_id() or f"bg-{new_trace_id()}"
        cur = conn.execute(
            "INSERT INTO tasks(name, config_name, status, priority, created_at, "
            "project_id, version_id, config_path, request_trace_id) "
            "VALUES (?, ?, 'pending', 0, ?, ?, ?, ?, ?)",
            (task_name, config_name, time.time(), pid, vid, str(cfg_path), req_tid),
        )
        tid = int(cur.lastrowid)
        conn.commit()
        # ADR-0007 PR-5: version.status 由 supervisor 在 _spawn_task 推到 training；
        # project 无 stage；这里不再 advance。
        task = db.get_task(conn, tid)
    bus.publish({
        "type": "task_state_changed",
        "task_id": tid,
        "status": "pending",
    })
    return task or {}


# version 级缩略图：bucket = train | reg | samples（PP3 加 train，reg/samples 留作 PP4-5）
@router.get("/api/projects/{pid}/versions/{vid}/thumb")
def version_thumb(
    pid: int,
    vid: int,
    bucket: str = "train",
    folder: str = "",
    name: str = "",
    size: int = 256,
) -> FileResponse:
    if bucket not in {"train", "reg", "samples"}:
        raise HTTPException(400, f"非法 bucket: {bucket}")
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        p = projects.get_project(conn, pid)
    if not v or not p or v["project_id"] != pid:
        raise HTTPException(404, "版本不存在")
    vdir = versions.version_dir(p["id"], p["slug"], v["label"]) / bucket
    if bucket in {"train", "reg"}:
        if not folder:
            raise HTTPException(400, "invalid folder")
        f = _safe_join_or_400(vdir, folder, name)
    else:
        f = _safe_join_or_400(vdir, name)
    if not f.exists() or f.suffix.lower() not in datasets.IMAGE_EXTS:
        logger.info(
            "version thumb 404: pid=%s vid=%s bucket=%s folder=%s name=%s -> %s",
            pid, vid, bucket, folder, name, f,
        )
        raise HTTPException(404)
    return _thumb_response(f, size)
