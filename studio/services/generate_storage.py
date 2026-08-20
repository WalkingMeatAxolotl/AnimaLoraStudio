"""测试出图的 server 端落盘/记账闭环(出图时间线 DB 单源)。

蓝图见 tmp/generate-timeline-db-refactor-plan.md:tasks 表是出图列表唯一
台账,`generate_images` 列记录任务全部产出图;本模块是唯一写入方。

daemon image_done 时(daemon.py 调 `handle_image_done`):
  - 图先入加密 session cache(live 显示走 `/api/generate/{task}/sample/{fn}`,
    这一步保持现状不动);
  - save=on  → 落盘任务排进单线程 executor:注入 PNG metadata → 原子写
    `test/<date>/{single,xy}/` → append `generate_images` → **drop cache 中转
    副本**(旧双源时代的中转残留正是「刷新后列表翻倍」的根源);
  - save=off → 同步 append `{"cache": <filename>}` 记账(图只活在 cache,
    session 结束后行显示「已释放」)。

为什么单线程 executor:注入前要算资源 hash(generation_metadata.file_sha256,
首次 hash 13-26GB 底模是分钟级),在 daemon reader 线程同步做会把 stdout 管道
堵死;单线程保证同 task 的命名序号扫描 / images append 串行无竞争。XY
composite 补传(路由端点)也排进同一 executor —— 天然序在所有 cell 之后。

落盘布局 / 命名(#245 / 决策 #6 不变,helper 从 api/routers/generate.py 搬来):
  single: test/<YYYY-MM-DD>/single/single image <N>.png
  xy:     test/<YYYY-MM-DD>/xy/xy plot <N>/{cell x<i> y<j>.png ..., xy plot.png}
composite 不入 generate_images(它是文件夹附件,外站上传用;应用内回看用
cells 渲网格)。XY 中途取消保留已落盘的 cells,不回滚。
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, Optional

from ..infrastructure import db
from ..paths import STUDIO_DATA
from .generation_metadata import build_external_metadata

logger = logging.getLogger(__name__)

TEST_IMAGES_DIR = STUDIO_DATA / "test"

# 落盘 PNG anima_params 的 schema 版本(server enrich 时强制写入;v1 是
# 2025 早期落盘格式 lora_configs[].path,现行工具链只产 v2)
SCHEMA_VERSION = 2

# 目录 / 文件名布局约定(#245 / 决策 #6;disk image 路由校验共用)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# v2 命名(决策 #6):父目录区分 mode,文件名仅 "<label> N.png"
DISPLAY_LABELS = {"single": "single image", "xy": "xy plot"}
V2_SINGLE_RE = re.compile(r"^single image (\d+)\.png$")
V2_XY_RE = re.compile(r"^xy plot (\d+)\.png$")
# v1 legacy:image_N.png(旧版命名),序号防撞仍扫,新写入只用 v2
V1_NAME_RE = re.compile(r"^image_(\d+)\.png$")
XY_FOLDER_RE = re.compile(r"^xy plot (\d+)$")
XY_COMPOSITE_NAME = "xy plot.png"

# 落盘串行 executor(见模块 docstring)。daemon 进程退出时排队中的任务丢弃
# —— crash 场景图本来也只在 cache,行会显示「已释放」,不做 atexit 编排。
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen-storage")


# ---------------------------------------------------------------------------
# 命名 / 原子写 / PNG metadata(从 api/routers/generate.py 原样搬迁)
# ---------------------------------------------------------------------------


def next_image_index(dir_: Path, mode: str) -> int:
    """扫描 dir 下当前 mode 的 PNG 文件,返回下一个 1-based 序号。

    决策 #11:无并发跑图场景,不做 O_EXCL / 锁;序号扫 max+1 + atomic 写即可
    (本模块所有写盘走单线程 executor,串行性由 executor 保证)。
    """
    if not dir_.is_dir():
        return 1
    rx_v2 = V2_SINGLE_RE if mode == "single" else V2_XY_RE
    max_n = 0
    for p in dir_.iterdir():
        if not p.is_file():
            continue
        m_v2 = rx_v2.match(p.name)
        m_v1 = V1_NAME_RE.match(p.name)
        if m_v2:
            max_n = max(max_n, int(m_v2.group(1)))
        elif m_v1:
            # v1 legacy 0-based;映射到 v2 编号空间 +1 避免冲突
            max_n = max(max_n, int(m_v1.group(1)) + 1)
    return max_n + 1


def next_xy_folder_index(xy_dir: Path) -> int:
    """XY 模式下一个文件夹 1-based 序号(新格式子文件夹 + legacy 平铺文件两个
    编号空间防撞)。"""
    if not xy_dir.is_dir():
        return 1
    max_n = 0
    for p in xy_dir.iterdir():
        if p.is_dir():
            m = XY_FOLDER_RE.match(p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
        elif p.is_file():
            m = V2_XY_RE.match(p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def atomic_write_png(target: Path, raw: bytes) -> None:
    """原子写 PNG:写 tmp + os.replace(决策 #11 crash safety)。target 出现
    的瞬间内容已完整,不会留半截 PNG。"""
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, target)


def format_a1111_parameters(
    params: dict[str, Any], external: dict[str, Any] | None = None
) -> str:
    """组装 a1111 兼容的 `parameters` tEXt 块(ComfyUI / WebUI / Civitai 等通用)。

    格式:
        <prompt> [<lora:name:scale> ...]
        Negative prompt: <neg>
        Steps: N, Sampler: ..., Schedule type: ..., CFG scale: N, Seed: N, Size: WxH

    LoRA 用 <lora:basename-without-ext:scale> 语法(a1111/ComfyUI 标准)。
    xy_draft / dataset_pick 的 UI 上下文不入此块(a1111 没标准字段)。
    """
    external = external or {}
    prompts = params.get("prompts") or [""]
    prompt = external.get("prompt")
    if prompt is None:
        prompt = prompts[0] if isinstance(prompts, list) else str(prompts)
    prompt = str(prompt)
    loras = external.get("loras") or params.get("loras") or []
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
    model_family = external.get("model_family") or params.get("model_family")
    if model_family:
        parts.append(f"Model family: {model_family}")
    text_encoder = external.get("text_encoder") or params.get("text_encoder")
    if text_encoder:
        parts.append(f"Text encoder: {text_encoder}")

    hashes: dict[str, str] = {}
    model = external.get("model")
    if isinstance(model, dict) and model.get("name"):
        parts.append(f"Model: {model['name']}")
        if model.get("hash"):
            model_hash = str(model["hash"])
            parts.append(f"Model hash: {model_hash}")
            hashes["model"] = model_hash
    vae = external.get("vae")
    if isinstance(vae, dict) and vae.get("name"):
        parts.append(f"VAE: {vae['name']}")
        if vae.get("hash"):
            vae_hash = str(vae["hash"])
            parts.append(f"VAE hash: {vae_hash}")
            hashes["vae"] = vae_hash

    lora_hashes: list[str] = []
    for lo in loras:
        if not isinstance(lo, dict) or not lo.get("hash"):
            continue
        name = str(lo.get("name") or "").rsplit(".", 1)[0]
        if not name:
            continue
        digest = str(lo["hash"])
        lora_hashes.append(f"{name}: {digest}")
        hashes[f"lora:{name}"] = digest
    if lora_hashes:
        parts.append(f'Lora hashes: "{", ".join(lora_hashes)}"')
    if hashes:
        parts.append(f"Hashes: {json.dumps(hashes, separators=(',', ':'))}")
    parts.append("Software: AnimaLoraStudio")
    return f"{prompt}\nNegative prompt: {neg}\n{', '.join(parts)}"


def inject_png_metadata(
    raw: bytes,
    params: dict[str, Any],
    *,
    mode: str,
    external: dict[str, Any] | None = None,
) -> bytes:
    """注入 PNG tEXt 块到图:
       - `anima_params` —— 结构化 JSON,**zTXt 压缩**(决策 #17),本程序回填用
       - `parameters`   —— a1111 兼容文本(决策 #7:xy **不写**,矩阵图单图拖
         进 a1111 参数语义对不上);仅 single 模式写

    失败返回原 bytes(不阻塞落盘主流程)。
    """
    try:
        from PIL import Image, PngImagePlugin
        img = Image.open(io.BytesIO(raw))
        info = PngImagePlugin.PngInfo()
        # zip=True → zTXt 压缩块(PIL 9+),XY cells[] 时 anima_params 可能 6KB+
        info.add_text("anima_params", json.dumps(params, ensure_ascii=False), zip=True)
        if mode == "single":
            info.add_text("parameters", format_a1111_parameters(params, external))
        out = io.BytesIO()
        img.save(out, format="PNG", pnginfo=info)
        return out.getvalue()
    except Exception:
        return raw


def enrich_params_server_side(
    params: dict[str, Any], *, task_id: int | None, mode: str
) -> dict[str, Any]:
    """server 端补全 params 的服务端信息(避免前端伪造 / 漏字段)。"""
    params = dict(params)
    params["schema_version"] = SCHEMA_VERSION
    params["created_at"] = time.time()
    if task_id is not None:
        params["task_id"] = int(task_id)
    params["mode"] = mode
    return params


def _build_external_metadata_safe(
    task_id: int | None, params: dict[str, Any], *, source_filename: str,
) -> dict[str, Any]:
    """资源 metadata 是 best-effort;失败时仍按旧格式保存 PNG。"""
    try:
        return build_external_metadata(
            task_id, params, source_filename=source_filename,
        )
    except Exception:
        logger.warning(
            "build external metadata failed: task_id=%s", task_id, exc_info=True,
        )
        return {}


# ---------------------------------------------------------------------------
# XY cell snapshot 物化(前端 paramsSnapshot.ts buildCellSnapshot 的 Python 移植)
# ---------------------------------------------------------------------------


def _lora_basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _apply_axis_to_cell(
    snap: dict[str, Any], axis: Any, value: Any, lora_index: Any,
) -> None:
    """轴语义与 daemon `_apply_axis` / 前端 applyAxisToCell 对齐:
    steps / cfg_scale 顶层标量;lora_scale 全 LoRA 共用;lora_ckpt 仅
    loras[loraIndex] 的 name 换 basename、ids 清空(ckpt 换了 ids 不再准)。
    值转换失败静默跳过(该轴保持 XY 快照原值)。"""
    try:
        if axis == "steps":
            snap["steps"] = int(float(value))
        elif axis == "cfg_scale":
            snap["cfg_scale"] = float(value)
        elif axis == "lora_scale":
            scale = float(value)
            snap["loras"] = [{**lo, "scale": scale} for lo in snap["loras"]]
        elif axis == "lora_ckpt":
            idx = int(lora_index) if lora_index is not None else None
            if idx is None or not (0 <= idx < len(snap["loras"])):
                return
            name = _lora_basename(str(value))
            snap["loras"] = [
                {**lo, "name": name, "project_id": None, "version_id": None}
                if i == idx else lo
                for i, lo in enumerate(snap["loras"])
            ]
    except (TypeError, ValueError):
        return


def build_cell_snapshot(
    xy: dict[str, Any], xy_info: dict[str, Any],
) -> dict[str, Any]:
    """XY snapshot 按 cell 物化成 single-snapshot(落盘 cell PNG metadata 用)。

    拖进 Comfy / A1111 能识别该格实际的 steps/cfg/seed/lora;本程序回填走
    mode='single' 主路径。轴类型 / loraIndex 取自快照 xy_draft,格值 xv/yv
    取 daemon image_done 事件携带的真值(daemon 实际跑的值,比反查 raw 可靠)。
    输出额外带 `xy_origin` 链回所属 XY plot。
    """
    out = dict(xy)
    out["mode"] = "single"
    out["xy_draft"] = None
    out["loras"] = [
        dict(lo) for lo in (xy.get("loras") or []) if isinstance(lo, dict)
    ]
    draft = xy.get("xy_draft") or {}
    x = draft.get("x") if isinstance(draft, dict) else None
    y = draft.get("y") if isinstance(draft, dict) else None
    xv = xy_info.get("xv")
    yv = xy_info.get("yv")
    if isinstance(x, dict) and xv is not None:
        _apply_axis_to_cell(out, x.get("axis"), xv, x.get("loraIndex"))
    if isinstance(y, dict) and yv is not None:
        _apply_axis_to_cell(out, y.get("axis"), yv, y.get("loraIndex"))
    out["xy_origin"] = {
        "xi": int(xy_info.get("xi", 0)),
        "yi": int(xy_info.get("yi", 0)),
        "xv": xv,
        "yv": yv,
        "x_axis": x.get("axis") if isinstance(x, dict) else None,
        "y_axis": y.get("axis") if isinstance(y, dict) else None,
    }
    return out


# ---------------------------------------------------------------------------
# generate_images 台账
# ---------------------------------------------------------------------------


def _rel_posix(target: Path) -> str:
    """落盘路径 → 相对 test/ 的正斜杠字符串(DB 存储统一形态,Windows 兼容)。"""
    return str(target.relative_to(TEST_IMAGES_DIR)).replace("\\", "/")


def load_images(task_id: int) -> list[dict[str, Any]]:
    """读该 task 的 generate_images(缺省 [])。"""
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    raw = (task or {}).get("generate_images")
    if not raw:
        return []
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []


def _append_image(task_id: int, item: dict[str, Any]) -> None:
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
        if not task:
            return
        raw = task.get("generate_images")
        try:
            images = json.loads(raw) if raw else []
        except json.JSONDecodeError:
            images = []
        if not isinstance(images, list):
            images = []
        images.append(item)
        db.update_task(
            conn, task_id,
            generate_images=json.dumps(images, ensure_ascii=False),
        )
    # 通知前端刷时间线:disk 落盘走 executor 异步(首次 hash 大模型时可达分钟
    # 级),done SSE 到达时 images 可能还没写全 —— 前端靠本事件增量刷新。
    try:
        from ..infrastructure.event_bus import bus
        bus.publish({"type": "generate_images_updated", "task_id": task_id})
    except Exception:
        pass


def find_disk_file(task_id: int, source_filename: str) -> Optional[Path]:
    """按 daemon filename 反查该 task 已落盘的文件(sample 端点 cache miss
    fallback:落盘后 cache 副本已 drop,live 显示 / composite 拼图仍按
    daemon filename 取图)。"""
    for item in load_images(task_id):
        if item.get("src") == source_filename and item.get("file"):
            p = TEST_IMAGES_DIR / str(item["file"])
            if p.is_file():
                return p
    return None


def xy_folder_for_task(task_id: int) -> Optional[Path]:
    """该 task 的 xy 文件夹(从台账里第一个 cell 的路径推导;无 → None)。"""
    for item in load_images(task_id):
        f = item.get("file")
        if f and "/xy/" in f:
            return (TEST_IMAGES_DIR / str(f)).parent
    return None


# ---------------------------------------------------------------------------
# daemon 入口:image_done 处置
# ---------------------------------------------------------------------------

# 落盘失败节流（T6）：磁盘满 / 目录被占用时 XY 一格一条黄行 = 几十上百条。
# 首条全文 WARNING（带文件名 + 原因）→ 同任务后续逐条 DEBUG → 收尾一条计数
# 汇总。executor 是单线程（max_workers=1），这份状态只被那一个线程碰。
_disk_store_failures: dict[int, dict[str, Any]] = {}


def _note_disk_store_failure(task_id: int, filename: str) -> None:
    st = _disk_store_failures.setdefault(task_id, {"n": 0, "first": ""})
    st["n"] += 1
    if st["n"] == 1:
        exc = sys.exc_info()[1]
        st["first"] = f"{type(exc).__name__}: {exc}" if exc is not None else "?"
        logger.warning(
            "disk store failed: task_id=%s file=%s; the image stays in the "
            "session cache only", task_id, filename, exc_info=True,
        )
    else:
        logger.debug(
            "disk store failed: task_id=%s file=%s", task_id, filename, exc_info=True,
        )


def flush_disk_store_summary(task_id: int, total: Optional[int] = None) -> None:
    """task 收尾：把该 task 的落盘失败计数收成一条 WARNING（没失败就 noop）。"""
    st = _disk_store_failures.pop(int(task_id), None)
    if not st or not st["n"]:
        return
    logger.warning(
        "disk store failed for %d/%s images: task_id=%s; first error: %s",
        st["n"], total if total is not None else "?", task_id, st["first"],
    )


def handle_image_done(
    task_id: int,
    filename: str,
    data: bytes,
    snapshot: dict[str, Any],
    *,
    mode: str,
    xy_info: Optional[dict[str, Any]],
    save_to_disk: bool,
) -> None:
    """daemon reader 线程调(cache put 之后)。快路径同步、慢路径入 executor。"""
    if not save_to_disk:
        # temp:图只活在加密 cache,台账记 filename(session 结束 → 已释放)
        item: dict[str, Any] = {"cache": filename}
        if mode == "xy" and xy_info is not None:
            item["xi"] = int(xy_info.get("xi", 0))
            item["yi"] = int(xy_info.get("yi", 0))
        _append_image(task_id, item)
        return
    _EXECUTOR.submit(
        _store_to_disk_safe, task_id, filename, data, dict(snapshot),
        mode, dict(xy_info) if xy_info else None,
    )


def _store_to_disk_safe(
    task_id: int,
    filename: str,
    data: bytes,
    snapshot: dict[str, Any],
    mode: str,
    xy_info: Optional[dict[str, Any]],
) -> None:
    """executor 线程:注入 → 落盘 → 记账 → drop cache 中转副本。

    失败降级:图仍在 cache(session 内可看),台账不记 → 行内该图重启后
    「已释放」;log warning 不炸 daemon。
    """
    try:
        if mode == "xy" and xy_info is not None:
            target = _write_xy_cell(task_id, filename, data, snapshot, xy_info)
        else:
            target = _write_single(task_id, filename, data, snapshot)
    except Exception:
        _note_disk_store_failure(task_id, filename)
        return
    # 落盘成功 → cache 中转副本没有存在意义了(列表/回看走磁盘;live 显示
    # 与 composite 拼图靠 sample 端点的 disk fallback 按 src 反查)
    try:
        from .inference import disk_cache as generate_cache
        generate_cache.drop_image(task_id, filename)
    except Exception:
        logger.warning(
            "drop cache copy failed: task_id=%s file=%s",
            task_id, filename, exc_info=True,
        )


def _write_single(
    task_id: int, filename: str, data: bytes, snapshot: dict[str, Any],
) -> Path:
    enriched = enrich_params_server_side(snapshot, task_id=task_id, mode="single")
    external = _build_external_metadata_safe(
        task_id, enriched, source_filename=filename,
    )
    payload = inject_png_metadata(data, enriched, mode="single", external=external)
    target_dir = TEST_IMAGES_DIR / date.today().isoformat() / "single"
    target_dir.mkdir(parents=True, exist_ok=True)
    idx = next_image_index(target_dir, "single")
    target = target_dir / f"{DISPLAY_LABELS['single']} {idx}.png"
    atomic_write_png(target, payload)
    _append_image(task_id, {"file": _rel_posix(target), "src": filename})
    return target


def _write_xy_cell(
    task_id: int,
    filename: str,
    data: bytes,
    snapshot: dict[str, Any],
    xy_info: dict[str, Any],
) -> Path:
    xi = int(xy_info.get("xi", 0))
    yi = int(xy_info.get("yi", 0))
    folder = xy_folder_for_task(task_id)
    if folder is None:
        xy_dir = TEST_IMAGES_DIR / date.today().isoformat() / "xy"
        xy_dir.mkdir(parents=True, exist_ok=True)
        idx = next_xy_folder_index(xy_dir)
        folder = xy_dir / f"{DISPLAY_LABELS['xy']} {idx}"
        folder.mkdir(exist_ok=True)
    cell_snap = build_cell_snapshot(snapshot, xy_info)
    enriched = enrich_params_server_side(cell_snap, task_id=task_id, mode="single")
    external = _build_external_metadata_safe(
        task_id, enriched, source_filename=filename,
    )
    payload = inject_png_metadata(data, enriched, mode="single", external=external)
    target = folder / f"cell x{xi} y{yi}.png"
    atomic_write_png(target, payload)
    _append_image(
        task_id, {"file": _rel_posix(target), "src": filename, "xi": xi, "yi": yi},
    )
    return target


# ---------------------------------------------------------------------------
# XY composite 补传(路由端点调;决策 1:盘上仍要有大图,前端拼好 POST 上来)
# ---------------------------------------------------------------------------


def attach_xy_composite(task_id: int, data: bytes) -> Path:
    """把前端拼好的 composite 写入该 task 的 xy 文件夹。

    排进 storage executor 串行执行 → 天然等所有 cell 落盘之后(前端在 task
    done 时 POST,此刻 executor 队列里可能还有 cell)。参数注入取 DB
    generate_params(单源,不再信前端传参)。

    Raises:
        LookupError: task 无 xy 文件夹(save 关着 / cells 全失败)。
    """
    return _EXECUTOR.submit(_attach_xy_composite_sync, task_id, data).result(
        timeout=120,
    )


def _attach_xy_composite_sync(task_id: int, data: bytes) -> Path:
    folder = xy_folder_for_task(task_id)
    if folder is None or not folder.is_dir():
        raise LookupError(f"task {task_id} has no xy folder on disk")
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    params: dict[str, Any] = {}
    raw = (task or {}).get("generate_params")
    if raw:
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                params = decoded
        except json.JSONDecodeError:
            pass
    payload = data
    if params:
        enriched = enrich_params_server_side(params, task_id=task_id, mode="xy")
        payload = inject_png_metadata(payload, enriched, mode="xy")
    target = folder / XY_COMPOSITE_NAME
    atomic_write_png(target, payload)
    return target
