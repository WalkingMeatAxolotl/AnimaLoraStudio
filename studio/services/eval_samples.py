"""Persistent eval sample runs for version-scoped LoRA validation.

This consumes the eval manifest from ADR-0010 and records generated samples,
but intentionally does not compute metrics. Heavy model imports stay inside the
default generator so API/tests can exercise the contract without loading torch.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from . import eval_manifest
from .projects import jobs as project_jobs
from .projects import versions

SCHEMA_VERSION = 1
JOB_KIND = "eval_samples"
RUNS_DIRNAME = "samples"
RUN_FILE = "run.json"
IMAGES_DIRNAME = "images"
DEFAULT_MAX_ITEMS = 64
MAX_ITEMS_LIMIT = 256

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_IMAGE_EXT = ".png"


class EvalSamplesError(Exception):
    """Business error for eval sample runs."""


SampleGenerator = Callable[[dict[str, Any], Path, Callable[[str], None]], None]


def samples_dir(version_dir: Path) -> Path:
    return eval_manifest.eval_dir(version_dir) / RUNS_DIRNAME


def run_dir(version_dir: Path, run_id: str) -> Path:
    _validate_run_id(run_id)
    return samples_dir(version_dir) / run_id


def run_path(version_dir: Path, run_id: str) -> Path:
    return run_dir(version_dir, run_id) / RUN_FILE


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _manifest_digest(manifest: dict[str, Any]) -> str:
    raw = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise EvalSamplesError(f"非法 eval sample run id: {run_id!r}")
    return run_id


def _now_run_id(now: float, checkpoint: dict[str, Any]) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    suffix = str(checkpoint.get("kind") or "ckpt")
    value = checkpoint.get("value")
    if value:
        suffix = f"{suffix}{value}"
    base = f"run-{stamp}-{suffix}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", base).strip("-")


def _clamp_max_items(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = DEFAULT_MAX_ITEMS
    return max(1, min(MAX_ITEMS_LIMIT, n))


def _rel_to_version(version_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(version_dir.resolve()).as_posix()
    except ValueError as exc:
        raise EvalSamplesError(f"路径不属于当前 version: {path}") from exc


def _resolve_checkpoint(version_dir: Path, raw_path: str | None) -> dict[str, Any]:
    ckpts = versions.list_lora_ckpts(version_dir)
    by_resolved: dict[str, dict[str, Any]] = {}
    for item in ckpts:
        try:
            by_resolved[str(Path(item["path"]).resolve())] = item
        except Exception:
            continue

    if raw_path:
        raw = str(raw_path).strip()
        if not raw:
            raise EvalSamplesError("checkpoint_path 不能为空")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = version_dir / "output" / raw.replace("\\", "/")
        path = candidate.resolve()
        output_dir = (version_dir / "output").resolve()
        try:
            path.relative_to(output_dir)
        except ValueError as exc:
            raise EvalSamplesError("checkpoint_path 必须位于 version output/ 内") from exc
        if path.suffix.lower() != ".safetensors" or not path.is_file():
            raise EvalSamplesError(f"checkpoint 不存在或不是 .safetensors: {raw_path}")
        item = by_resolved.get(str(path))
        if item is None:
            item = {
                "kind": "other",
                "value": 0,
                "label": path.stem,
                "path": str(path),
                "mtime": path.stat().st_mtime,
            }
    else:
        if not ckpts:
            raise EvalSamplesError("version output/ 下没有 LoRA checkpoint")
        item = ckpts[0]
        path = Path(item["path"]).resolve()

    return {
        "kind": str(item.get("kind") or "other"),
        "value": int(item.get("value") or 0),
        "label": str(item.get("label") or path.name),
        "path": _rel_to_version(version_dir, path),
        "mtime": float(item.get("mtime") or 0.0),
    }


def _prompt_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = manifest.get("prompts") if isinstance(manifest.get("prompts"), list) else []
    out: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            continue
        text = " ".join(str(prompt.get("text") or "").strip().split())
        if not text:
            continue
        out.append({
            "id": str(prompt.get("id") or f"prompt:{idx + 1}"),
            "text": text,
            "source": str(prompt.get("source") or "manifest"),
        })
    if out:
        return out

    heldout = manifest.get("heldout") if isinstance(manifest.get("heldout"), list) else []
    for idx, item in enumerate(heldout):
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("prompt") or "").strip().split())
        if text:
            out.append({
                "id": str(item.get("id") or f"heldout:{idx + 1}"),
                "text": text,
                "source": "heldout",
            })
    return out


def _seed_records(manifest: dict[str, Any]) -> list[int]:
    raw = manifest.get("seeds") if isinstance(manifest.get("seeds"), list) else []
    seeds: list[int] = []
    for value in raw:
        try:
            seed = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= seed <= 2**32 - 1 and seed not in seeds:
            seeds.append(seed)
    if seeds:
        return seeds

    heldout = manifest.get("heldout") if isinstance(manifest.get("heldout"), list) else []
    for item in heldout:
        if not isinstance(item, dict):
            continue
        try:
            seed = int(item.get("seed"))
        except (TypeError, ValueError):
            continue
        if 0 <= seed <= 2**32 - 1 and seed not in seeds:
            seeds.append(seed)
    return seeds or [12345]


def _planned_items(manifest: dict[str, Any], max_items: int) -> list[dict[str, Any]]:
    prompts = _prompt_records(manifest)
    if not prompts:
        raise EvalSamplesError("eval manifest 没有可用 prompt")
    seeds = _seed_records(manifest)
    items: list[dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(prompts):
        for seed in seeds:
            if len(items) >= max_items:
                return items
            idx = len(items)
            filename = f"sample_{idx:04d}_p{prompt_idx:03d}_s{seed}.png"
            items.append({
                "id": f"{prompt['id']}:{seed}",
                "prompt_id": prompt["id"],
                "prompt": prompt["text"],
                "prompt_source": prompt["source"],
                "seed": seed,
                "filename": filename,
                "path": "",
                "status": "pending",
                "error": None,
            })
    return items


def _summary(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(items),
        "pending": sum(1 for item in items if item.get("status") == "pending"),
        "running": sum(1 for item in items if item.get("status") == "running"),
        "done": sum(1 for item in items if item.get("status") == "done"),
        "failed": sum(1 for item in items if item.get("status") == "failed"),
    }


def load_run(version_dir: Path, run_id: str) -> dict[str, Any] | None:
    path = run_path(version_dir, run_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalSamplesError(f"eval sample run 读取失败: {exc}") from exc
    if not isinstance(data, dict):
        raise EvalSamplesError("eval sample run 必须是 JSON object")
    return data


def save_run(version_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
    run_id = _validate_run_id(str(run.get("run_id") or ""))
    items = run.get("items") if isinstance(run.get("items"), list) else []
    run["summary"] = _summary(items)
    run["updated_at"] = time.time()
    _atomic_write(run_path(version_dir, run_id), run)
    return run


def list_runs(version_dir: Path) -> list[dict[str, Any]]:
    root = samples_dir(version_dir)
    if not root.exists():
        return []
    runs: list[dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        path = child / RUN_FILE
        if not path.exists():
            continue
        try:
            run = load_run(version_dir, child.name)
        except EvalSamplesError:
            continue
        if run:
            runs.append(run)
    runs.sort(key=lambda r: float(r.get("created_at") or 0.0), reverse=True)
    return runs


def create_run(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    *,
    checkpoint_path: str | None = None,
    max_items: int | None = None,
    auto_metrics: bool = False,
    auto_source: dict[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    ts = time.time() if now is None else float(now)
    manifest = eval_manifest.load_manifest(version_dir)
    if manifest is None:
        manifest = eval_manifest.save_default_manifest(project, version, version_dir, now=ts)
    checkpoint = _resolve_checkpoint(version_dir, checkpoint_path)
    items = _planned_items(manifest, _clamp_max_items(max_items))
    base_run_id = _now_run_id(ts, checkpoint)
    run_id = base_run_id
    root = run_dir(version_dir, run_id)
    suffix = 2
    while (root / RUN_FILE).exists():
        run_id = f"{base_run_id}-{suffix}"
        root = run_dir(version_dir, run_id)
        suffix += 1

    for item in items:
        item["path"] = (
            eval_manifest.EVAL_DIRNAME
            + f"/{RUNS_DIRNAME}/{run_id}/{IMAGES_DIRNAME}/{item['filename']}"
        )

    run = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "pending",
        "project_id": int(project["id"]),
        "project_slug": str(project.get("slug") or ""),
        "version_id": int(version["id"]),
        "version_label": str(version.get("label") or ""),
        "created_at": ts,
        "updated_at": ts,
        "started_at": None,
        "finished_at": None,
        "error": None,
        "manifest_path": eval_manifest.manifest_path(version_dir).relative_to(version_dir).as_posix(),
        "manifest_digest": _manifest_digest(manifest),
        "manifest_snapshot": manifest,
        "checkpoint": checkpoint,
        "auto_metrics": bool(auto_metrics),
        "auto_source": dict(auto_source) if auto_source else None,
        "generation": dict(manifest.get("generation") or {}),
        "items": items,
        "summary": _summary(items),
    }
    _atomic_write(root / RUN_FILE, run)
    (root / IMAGES_DIRNAME).mkdir(parents=True, exist_ok=True)
    return run


def start_job(
    conn,
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    *,
    checkpoint_path: str | None = None,
    max_items: int | None = None,
    auto_metrics: bool = False,
    auto_source: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = create_run(
        project,
        version,
        version_dir,
        checkpoint_path=checkpoint_path,
        max_items=max_items,
        auto_metrics=auto_metrics,
        auto_source=auto_source,
    )
    params: dict[str, Any] = {
        "version_id": int(version["id"]),
        "run_id": run["run_id"],
        "checkpoint_path": run["checkpoint"]["path"],
    }
    if auto_metrics:
        params["auto_metrics"] = True
    if auto_source:
        params["auto_source"] = dict(auto_source)
    job = project_jobs.create_job(
        conn,
        project_id=int(project["id"]),
        version_id=int(version["id"]),
        kind=JOB_KIND,
        params=params,
    )
    return job, run


def run_sample_job(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    run_id: str,
    *,
    generator: SampleGenerator | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    progress = on_progress or (lambda _line: None)
    run = load_run(version_dir, run_id)
    if run is None:
        raise EvalSamplesError(f"eval sample run 不存在: {run_id}")
    if int(run.get("project_id") or 0) != int(project["id"]):
        raise EvalSamplesError("eval sample run 不属于当前 project")
    if int(run.get("version_id") or 0) != int(version["id"]):
        raise EvalSamplesError("eval sample run 不属于当前 version")

    run["status"] = "running"
    run["started_at"] = run.get("started_at") or time.time()
    run["error"] = None
    save_run(version_dir, run)

    try:
        (generator or _default_generator)(run, version_dir, progress)
    except Exception as exc:
        run = load_run(version_dir, run_id) or run
        run["status"] = "failed"
        run["finished_at"] = time.time()
        run["error"] = str(exc)
        save_run(version_dir, run)
        raise

    run = load_run(version_dir, run_id) or run
    failed = [item for item in run.get("items", []) if item.get("status") == "failed"]
    pending = [
        item for item in run.get("items", [])
        if item.get("status") in {"pending", "running"}
    ]
    run["status"] = "failed" if failed or pending else "done"
    run["finished_at"] = time.time()
    run["error"] = (
        f"{len(failed)} sample(s) failed"
        if failed else ("sample generation incomplete" if pending else None)
    )
    save_run(version_dir, run)
    return run


def mark_item_running(version_dir: Path, run: dict[str, Any], idx: int) -> dict[str, Any]:
    run["items"][idx]["status"] = "running"
    save_run(version_dir, run)
    return run


def mark_item_done(version_dir: Path, run: dict[str, Any], idx: int) -> dict[str, Any]:
    run["items"][idx]["status"] = "done"
    run["items"][idx]["error"] = None
    save_run(version_dir, run)
    return run


def mark_item_failed(
    version_dir: Path, run: dict[str, Any], idx: int, error: str
) -> dict[str, Any]:
    run["items"][idx]["status"] = "failed"
    run["items"][idx]["error"] = error
    save_run(version_dir, run)
    return run


def sample_image_path(version_dir: Path, run_id: str, filename: str) -> Path:
    if "/" in filename or "\\" in filename or filename in {"", ".", ".."}:
        raise EvalSamplesError(f"非法 sample filename: {filename!r}")
    if not filename.lower().endswith(_IMAGE_EXT):
        raise EvalSamplesError("eval sample image 只支持 .png")
    path = run_dir(version_dir, run_id) / IMAGES_DIRNAME / filename
    base = (run_dir(version_dir, run_id) / IMAGES_DIRNAME).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise EvalSamplesError(f"sample image 路径逃逸: {filename!r}") from exc
    return resolved


def _default_generator(
    run: dict[str, Any],
    version_dir: Path,
    progress: Callable[[str], None],
) -> None:
    import random
    import sys

    import torch

    runtime_dir = Path(__file__).resolve().parents[2] / "runtime"
    repo_root = runtime_dir.parent
    for path in (runtime_dir, repo_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    import anima_train as _T  # noqa: WPS433
    from studio.schema import migrate_legacy_attention
    from studio.services import version_config
    from studio.services.inference.core import LoRASpec, apply_loras

    project = {"id": run["project_id"], "slug": run["project_slug"]}
    version = {"id": run["version_id"], "label": run["version_label"]}
    cfg = migrate_legacy_attention(version_config.read_version_config(project, version))

    generation = run.get("generation") if isinstance(run.get("generation"), dict) else {}
    width = int(generation.get("width") or cfg.get("sample_width") or cfg.get("resolution") or 1024)
    height = int(generation.get("height") or cfg.get("sample_height") or cfg.get("resolution") or 1024)
    width = max(16, (width // 16) * 16)
    height = max(16, (height // 16) * 16)
    steps = int(generation.get("steps") or cfg.get("sample_infer_steps") or 25)
    cfg_scale = float(
        generation.get("guidance_scale")
        or generation.get("cfg_scale")
        or cfg.get("sample_cfg_scale")
        or 4.0
    )
    negative_prompt = str(generation.get("negative_prompt") or cfg.get("sample_negative_prompt") or "")
    sampler_name = str(generation.get("sampler_name") or cfg.get("sample_sampler_name") or "er_sde")
    scheduler = str(generation.get("scheduler") or cfg.get("sample_scheduler") or "simple")
    lora_scale = float(generation.get("lora_scale") or 1.0)
    precision = str(cfg.get("mixed_precision") or "bf16")
    backend = str(cfg.get("attention_backend") or "flash_attn")
    use_flash = backend == "flash_attn"
    use_xformers = backend == "xformers"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32

    bases = [Path.cwd(), runtime_dir, repo_root]
    transformer_path = _T.resolve_path_best_effort(str(cfg["transformer_path"]), bases)
    vae_path = _T.resolve_path_best_effort(str(cfg["vae_path"]), bases)
    text_encoder_path = _T.resolve_path_best_effort(str(cfg["text_encoder_path"]), bases)
    t5_tokenizer_path = str(cfg.get("t5_tokenizer_path") or "")
    if t5_tokenizer_path:
        t5_tokenizer_path = _T.resolve_path_best_effort(t5_tokenizer_path, bases)

    progress("[eval-samples] loading base model")
    diffusion_root = _T.find_diffusion_pipe_root()
    model = _T.load_anima_model(
        transformer_path, device, dtype, diffusion_root, flash_attn=use_flash
    )
    if use_xformers:
        _T.enable_xformers(model)
    vae = _T.load_vae(vae_path, device, dtype, diffusion_root)
    qwen_model, qwen_tok, t5_tok = _T.load_text_encoders(
        text_encoder_path, t5_tokenizer_path or None, device, dtype,
    )

    checkpoint = version_dir / run["checkpoint"]["path"]
    adapters = apply_loras(
        model, [LoRASpec(path=str(checkpoint), scale=lora_scale)], device, dtype
    )
    _ = adapters  # keep adapter references alive for forward hooks
    model.eval()

    items = run.get("items") if isinstance(run.get("items"), list) else []
    for idx, item in enumerate(items):
        run = mark_item_running(version_dir, load_run(version_dir, run["run_id"]) or run, idx)
        seed = int(item["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        output = sample_image_path(version_dir, run["run_id"], item["filename"])
        output.parent.mkdir(parents=True, exist_ok=True)
        progress(
            f"[eval-samples] {idx + 1}/{len(items)} seed={seed} "
            f"prompt={str(item.get('prompt') or '')[:80]}"
        )
        try:
            img = _T.sample_image(
                model, vae, qwen_model, qwen_tok, t5_tok,
                prompt=str(item.get("prompt") or ""),
                height=height,
                width=width,
                steps=steps,
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt or None,
                sampler_name=sampler_name,
                scheduler=scheduler,
                device=device,
                dtype=dtype,
                seed=seed,
            )
            img.save(output)
            run = mark_item_done(version_dir, load_run(version_dir, run["run_id"]) or run, idx)
        except Exception as exc:  # noqa: BLE001
            run = mark_item_failed(
                version_dir, load_run(version_dir, run["run_id"]) or run, idx, str(exc)
            )
