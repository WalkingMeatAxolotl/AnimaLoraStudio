"""测试出图的外部生态 metadata 资源解析与文件哈希缓存。

PNG 内的 ``anima_params`` 是可移植的 UI 快照，刻意不含绝对路径；Civitai
要可靠识别底模 / LoRA 却需要文件 hash。两份数据不能混在一起，因此 enqueue
时把实际推理资源写进 task 私有档案，本模块在落盘 PNG 时读取档案并生成只供
``parameters`` 文本块使用的外部 metadata。绝对路径永不写入 PNG。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from ..infrastructure.paths import STUDIO_DATA, task_dir

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "generate-metadata.json"
HASH_CACHE_PATH = STUDIO_DATA / ".cache" / "resource-hashes.json"
_PROMPT_INDEX_RE = re.compile(r"_p(\d+)(?:_|\.)")
_HASH_LOCK = threading.Lock()
_HASH_CACHE: dict[str, dict[str, Any]] | None = None


def manifest_path(task_id: int) -> Path:
    return task_dir(task_id) / MANIFEST_FILENAME


def write_manifest(
    task_id: int,
    *,
    prompts: list[str],
    model_family: str,
    model_path: str,
    vae_path: str | None,
    text_encoder: str | None,
    loras: list[dict[str, Any]],
    xy_matrix: dict[str, Any] | None,
) -> None:
    """原子写 task 私有资源清单；清单与 task 档案同生命周期。"""
    path = manifest_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "prompts": list(prompts),
        "model_family": model_family,
        "model_path": model_path,
        "vae_path": vae_path,
        "text_encoder": text_encoder,
        "loras": loras,
        "xy_matrix": xy_matrix,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_manifest(task_id: int | None) -> dict[str, Any] | None:
    if task_id is None:
        return None
    try:
        value = json.loads(manifest_path(task_id).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _load_hash_cache() -> dict[str, dict[str, Any]]:
    global _HASH_CACHE
    if _HASH_CACHE is not None:
        return _HASH_CACHE
    try:
        value = json.loads(HASH_CACHE_PATH.read_text(encoding="utf-8"))
        _HASH_CACHE = value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        _HASH_CACHE = {}
    return _HASH_CACHE


def _write_hash_cache(cache: dict[str, dict[str, Any]]) -> None:
    HASH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = HASH_CACHE_PATH.with_suffix(HASH_CACHE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, HASH_CACHE_PATH)


def file_sha256(path_value: str | Path) -> str | None:
    """返回完整文件 SHA256；按 resolved path + size + mtime_ns 持久缓存。

    锁覆盖 hash 过程，避免 single 多图并发保存时重复顺序读取 13–26GB 底模。
    文件在读取过程中发生变化则不缓存也不返回，防止写入错误身份。
    """
    path = Path(path_value).expanduser()
    try:
        path = path.resolve(strict=True)
        before = path.stat()
        if not path.is_file():
            return None
    except OSError:
        return None

    key = str(path)
    with _HASH_LOCK:
        cache = _load_hash_cache()
        entry = cache.get(key)
        if (
            isinstance(entry, dict)
            and entry.get("size") == before.st_size
            and entry.get("mtime_ns") == before.st_mtime_ns
            and isinstance(entry.get("sha256"), str)
        ):
            return str(entry["sha256"])

        digest = hashlib.sha256()
        try:
            with path.open("rb") as f:
                while chunk := f.read(8 * 1024 * 1024):
                    digest.update(chunk)
            after = path.stat()
        except OSError:
            return None
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            logger.warning("resource changed while hashing; skip metadata hash: %s", path)
            return None

        result = digest.hexdigest()
        cache[key] = {
            "size": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "sha256": result,
        }
        try:
            _write_hash_cache(cache)
        except OSError:
            logger.warning("write generation resource hash cache failed", exc_info=True)
        return result


def _effective_prompt_from_snapshot(params: dict[str, Any]) -> str:
    """兼容无 task manifest 的旧客户端：按前端提交规则合并 picker tags。"""
    raw_prompts = params.get("prompts") or [""]
    prompts = [str(x).strip() for x in raw_prompts if str(x).strip()]
    pick = params.get("dataset_pick")
    tags = pick.get("tags") if isinstance(pick, dict) else None
    suffix = ", ".join(str(x) for x in tags if str(x).strip()) if isinstance(tags, list) else ""
    if prompts:
        return f"{prompts[0]}, {suffix}" if suffix else prompts[0]
    return suffix


def _select_prompt(manifest: dict[str, Any] | None, params: dict[str, Any], source_filename: str) -> str:
    prompts = manifest.get("prompts") if manifest else None
    if isinstance(prompts, list) and prompts:
        match = _PROMPT_INDEX_RE.search(source_filename)
        index = int(match.group(1)) if match else 0
        if 0 <= index < len(prompts):
            return str(prompts[index])
        return str(prompts[0])
    return _effective_prompt_from_snapshot(params)


def _apply_xy_resource_axis(
    loras: list[dict[str, Any]], axis: dict[str, Any] | None, value_index: int
) -> None:
    if not isinstance(axis, dict):
        return
    values = axis.get("values")
    if not isinstance(values, list) or not (0 <= value_index < len(values)):
        return
    value = values[value_index]
    kind = axis.get("axis")
    if kind == "lora_scale":
        try:
            scale = float(value)
        except (TypeError, ValueError):
            return
        for lora in loras:
            lora["scale"] = scale
    elif kind == "lora_ckpt":
        index = axis.get("lora_index")
        if isinstance(index, int) and 0 <= index < len(loras):
            loras[index]["path"] = str(value)


def build_external_metadata(
    task_id: int | None,
    params: dict[str, Any],
    *,
    source_filename: str = "",
) -> dict[str, Any]:
    """构造 Civitai/A1111 metadata 视图；返回值不含任何绝对路径。"""
    manifest = read_manifest(task_id)
    result: dict[str, Any] = {
        "prompt": _select_prompt(manifest, params, source_filename),
        "model_family": str((manifest or {}).get("model_family") or params.get("model_family") or ""),
        "text_encoder": (manifest or {}).get("text_encoder") or params.get("text_encoder"),
        "loras": [],
    }
    if not manifest:
        snapshot_loras = params.get("loras")
        if isinstance(snapshot_loras, list):
            result["loras"] = [dict(x) for x in snapshot_loras if isinstance(x, dict)]
        return result

    model_path = Path(str(manifest.get("model_path") or ""))
    if model_path.name:
        result["model"] = {
            "name": model_path.stem,
            "hash": file_sha256(model_path),
        }
    vae_path = Path(str(manifest.get("vae_path") or ""))
    if vae_path.name:
        result["vae"] = {
            "name": vae_path.stem,
            "hash": file_sha256(vae_path),
        }

    loras = [dict(x) for x in manifest.get("loras", []) if isinstance(x, dict)]
    origin = params.get("xy_origin")
    matrix = manifest.get("xy_matrix")
    if isinstance(origin, dict) and isinstance(matrix, dict):
        try:
            xi = int(origin.get("xi", 0))
            yi = int(origin.get("yi", 0))
        except (TypeError, ValueError):
            xi = yi = 0
        _apply_xy_resource_axis(loras, matrix.get("x"), xi)
        if matrix.get("y") is not None:
            _apply_xy_resource_axis(loras, matrix.get("y"), yi)

    external_loras: list[dict[str, Any]] = []
    for lora in loras:
        path = Path(str(lora.get("path") or ""))
        if not path.name:
            continue
        try:
            scale = float(lora.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        external_loras.append({
            "name": path.stem,
            "scale": scale,
            "hash": file_sha256(path),
        })
    result["loras"] = external_loras
    return result


def _reset_hash_cache_for_tests() -> None:
    global _HASH_CACHE
    with _HASH_LOCK:
        _HASH_CACHE = None
