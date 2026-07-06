"""Persistent eval sample runs for LoRA validation (ADR-0011 revised).

Reference + prompts come from the version's held-out ``validation/`` set (see
``eval_validation``): each generated sample carries its own ``reference_image``.
When ``validation/`` is empty the run falls back to the training config's sample
prompts and scores CLIP-T only (no reference image → no CLIP-I / DINO-I).
Generation params come from the version's ``sample_*`` config. There is no
separate manifest file — each ``run.json`` is self-contained. Heavy model
imports stay inside the default generator so API/tests can exercise the contract
without loading torch.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from . import eval_validation
from .projects import jobs as project_jobs
from .projects import versions

EVAL_DIRNAME = "eval"
# Reproducible default generation seed when sample_seed is 0 (=random for samples).
DEFAULT_GEN_SEED = 12345
DEFAULT_SAMPLE_PROMPT = "masterpiece, best quality"

SCHEMA_VERSION = 1
JOB_KIND = "eval_samples"
RUNS_DIRNAME = "samples"
RUN_FILE = "run.json"
IMAGES_DIRNAME = "images"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_IMAGE_EXT = ".png"


class EvalSamplesError(Exception):
    """Business error for eval sample runs."""


SampleGenerator = Callable[[dict[str, Any], Path, Callable[[str], None]], None]


def _eval_root(version_dir: Path, eval_root: Path | None = None) -> Path:
    # Eval output is task-scoped (eval_root = tasks/<id>/eval). The version-dir
    # fallback only serves legacy reads when no task scope is supplied.
    return eval_root if eval_root is not None else version_dir / EVAL_DIRNAME


def samples_dir(version_dir: Path, eval_root: Path | None = None) -> Path:
    return _eval_root(version_dir, eval_root) / RUNS_DIRNAME


def run_dir(version_dir: Path, run_id: str, eval_root: Path | None = None) -> Path:
    _validate_run_id(run_id)
    return samples_dir(version_dir, eval_root) / run_id


def run_path(version_dir: Path, run_id: str, eval_root: Path | None = None) -> Path:
    return run_dir(version_dir, run_id, eval_root) / RUN_FILE


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


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


def _rel_to_version(version_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(version_dir.resolve()).as_posix()
    except ValueError as exc:
        raise EvalSamplesError(f"路径不属于当前 version: {path}") from exc


def _storage_path(version_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(version_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


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


def _caption_text(image: Path) -> str:
    """Read an image's caption sidecar (.json via caption_format, else .txt)."""
    for caption in eval_validation.caption_sidecars(image):
        if caption.suffix.lower() == ".json":
            try:
                data = json.loads(caption.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            from .tagging.caption_format import caption_json_to_text
            text = caption_json_to_text(data if isinstance(data, dict) else None)
        else:
            try:
                text = caption.read_text(encoding="utf-8")
            except OSError:
                continue
        text = " ".join(str(text or "").strip().split())
        if text:
            return text
    return ""


def _read_config(project: dict[str, Any], version: dict[str, Any]) -> dict[str, Any]:
    """Version training config (for sample_* generation params). {} if unset."""
    from . import version_config
    try:
        return version_config.read_version_config(project, version)
    except Exception:
        return {}


def _generation_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    raw_res = cfg.get("resolution")
    if isinstance(raw_res, (list, tuple)) and raw_res:
        res = int(max(raw_res))
    elif isinstance(raw_res, (int, float)):
        res = int(raw_res)
    else:
        res = 1024
    seed = int(cfg.get("sample_seed") or 0) or DEFAULT_GEN_SEED
    return {
        "width": int(cfg.get("sample_width") or 0) or res,
        "height": int(cfg.get("sample_height") or 0) or res,
        "steps": int(cfg.get("sample_infer_steps") or 25),
        "guidance_scale": float(cfg.get("sample_cfg_scale") or 4.0),
        "sampler_name": str(cfg.get("sample_sampler_name") or "er_sde"),
        "scheduler": str(cfg.get("sample_scheduler") or "simple"),
        "negative_prompt": str(cfg.get("sample_negative_prompt") or ""),
        "lora_scale": 1.0,
        "seed": seed,
    }


def _config_prompts(cfg: dict[str, Any]) -> list[str]:
    raw = cfg.get("sample_prompts")
    prompts = [str(p).strip() for p in raw if str(p).strip()] if isinstance(raw, list) else []
    if prompts:
        return prompts
    single = str(cfg.get("sample_prompt") or "").strip()
    return [single] if single else [DEFAULT_SAMPLE_PROMPT]


def _planned_items(
    version_dir: Path, cfg: dict[str, Any], gen_seed: int
) -> list[dict[str, Any]]:
    """Build sample items from the held-out validation set.

    Each validation image → one item whose prompt is its caption and whose
    ``reference_image`` is the image (rel to version dir) for CLIP-I / DINO-I.
    The **entire** validation set is evaluated — its size is the eval scope, set
    via the training config's ``eval_validation_split_ratio``. When ``validation/``
    is empty, fall back to the config's sample prompts with no reference (CLIP-T
    only).
    """
    items: list[dict[str, Any]] = []
    val_images = list(eval_validation.iter_images(eval_validation.validation_dir(version_dir)))
    if val_images:
        for folder_name, image in val_images:
            rel = _rel_to_version(version_dir, image)
            prompt = _caption_text(image) or DEFAULT_SAMPLE_PROMPT
            idx = len(items)
            items.append({
                "id": f"val:{rel}:{gen_seed}",
                "prompt_id": f"val:{rel}",
                "prompt": prompt,
                "prompt_source": "caption" if _caption_text(image) else "empty",
                "seed": gen_seed,
                "filename": f"sample_{idx:04d}_s{gen_seed}.png",
                "path": "",
                "reference_image": rel,
                "status": "pending",
                "error": None,
            })
        return items

    for idx, prompt in enumerate(_config_prompts(cfg)):
        items.append({
            "id": f"prompt:{idx}:{gen_seed}",
            "prompt_id": f"prompt:{idx}",
            "prompt": prompt,
            "prompt_source": "sample_prompt",
            "seed": gen_seed,
            "filename": f"sample_{idx:04d}_s{gen_seed}.png",
            "path": "",
            "reference_image": None,
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


def load_run(version_dir: Path, run_id: str, eval_root: Path | None = None) -> dict[str, Any] | None:
    path = run_path(version_dir, run_id, eval_root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalSamplesError(f"eval sample run 读取失败: {exc}") from exc
    if not isinstance(data, dict):
        raise EvalSamplesError("eval sample run 必须是 JSON object")
    return data


def save_run(version_dir: Path, run: dict[str, Any], eval_root: Path | None = None) -> dict[str, Any]:
    run_id = _validate_run_id(str(run.get("run_id") or ""))
    items = run.get("items") if isinstance(run.get("items"), list) else []
    run["summary"] = _summary(items)
    run["updated_at"] = time.time()
    _atomic_write(run_path(version_dir, run_id, eval_root), run)
    return run


def list_runs(version_dir: Path, eval_root: Path | None = None) -> list[dict[str, Any]]:
    root = samples_dir(version_dir, eval_root)
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
            run = load_run(version_dir, child.name, eval_root)
        except EvalSamplesError:
            continue
        if run:
            runs.append(run)
    runs.sort(key=lambda r: float(r.get("created_at") or 0.0), reverse=True)
    return runs


def delete_all_runs(version_dir: Path, eval_root: Path | None = None) -> int:
    """删该 eval scope 下所有 sample run（run.json + 图 + metrics）。返回删除的 run 数。

    用于「清空评估、重新跑」：去掉该 task 的全部历史 run，下次评估从干净状态出图。
    """
    root = samples_dir(version_dir, eval_root)
    if not root.exists():
        return 0
    count = sum(1 for c in root.iterdir() if c.is_dir() and (c / RUN_FILE).exists())
    shutil.rmtree(root, ignore_errors=True)
    return count


def create_run(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    *,
    checkpoint_path: str | None = None,
    auto_metrics: bool = False,
    auto_source: dict[str, Any] | None = None,
    eval_root: Path | None = None,
    baseline: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    ts = time.time() if now is None else float(now)
    cfg = _read_config(project, version)
    generation = _generation_from_cfg(cfg)
    # baseline run = 纯底模对照（同 prompt/seed，lora_scale=0 → LoRA 不生效），
    # 给各 checkpoint 算 Δ = checkpoint − baseline，解决「绝对值难解读」。
    if baseline:
        generation["lora_scale"] = 0.0
    checkpoint = _resolve_checkpoint(version_dir, checkpoint_path)
    items = _planned_items(version_dir, cfg, int(generation["seed"]))
    if not items:
        raise EvalSamplesError("没有可评估的样本：validation/ 为空且未配置 sample prompt")
    base_run_id = _now_run_id(ts, checkpoint)
    run_id = base_run_id
    root = run_dir(version_dir, run_id, eval_root)
    suffix = 2
    while (root / RUN_FILE).exists():
        run_id = f"{base_run_id}-{suffix}"
        root = run_dir(version_dir, run_id, eval_root)
        suffix += 1

    for item in items:
        item["path"] = _storage_path(
            version_dir,
            run_dir(version_dir, run_id, eval_root) / IMAGES_DIRNAME / item["filename"],
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
        "checkpoint": checkpoint,
        "baseline": bool(baseline),
        "auto_metrics": bool(auto_metrics),
        "auto_source": dict(auto_source) if auto_source else None,
        "storage_scope": "task" if eval_root is not None else "version",
        "eval_root": str(eval_root.resolve()) if eval_root is not None else None,
        "generation": generation,
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
    auto_metrics: bool = False,
    auto_source: dict[str, Any] | None = None,
    eval_root: Path | None = None,
    baseline: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = create_run(
        project,
        version,
        version_dir,
        checkpoint_path=checkpoint_path,
        auto_metrics=auto_metrics,
        auto_source=auto_source,
        eval_root=eval_root,
        baseline=baseline,
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
    if eval_root is not None:
        task_id = int((auto_source or {}).get("task_id") or 0)
        if task_id:
            params["task_id"] = task_id
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
    eval_root: Path | None = None,
) -> dict[str, Any]:
    progress = on_progress or (lambda _line: None)
    run = load_run(version_dir, run_id, eval_root)
    if run is None:
        raise EvalSamplesError(f"eval sample run 不存在: {run_id}")
    if int(run.get("project_id") or 0) != int(project["id"]):
        raise EvalSamplesError("eval sample run 不属于当前 project")
    if int(run.get("version_id") or 0) != int(version["id"]):
        raise EvalSamplesError("eval sample run 不属于当前 version")

    run["status"] = "running"
    run["started_at"] = run.get("started_at") or time.time()
    run["error"] = None
    save_run(version_dir, run, eval_root)

    try:
        (generator or _default_generator)(run, version_dir, progress)
    except Exception as exc:
        run = load_run(version_dir, run_id, eval_root) or run
        run["status"] = "failed"
        run["finished_at"] = time.time()
        run["error"] = str(exc)
        save_run(version_dir, run, eval_root)
        raise

    run = load_run(version_dir, run_id, eval_root) or run
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
    save_run(version_dir, run, eval_root)
    return run


def mark_item_running(version_dir: Path, run: dict[str, Any], idx: int, eval_root: Path | None = None) -> dict[str, Any]:
    run["items"][idx]["status"] = "running"
    save_run(version_dir, run, eval_root)
    return run


def mark_item_done(version_dir: Path, run: dict[str, Any], idx: int, eval_root: Path | None = None) -> dict[str, Any]:
    run["items"][idx]["status"] = "done"
    run["items"][idx]["error"] = None
    save_run(version_dir, run, eval_root)
    return run


def mark_item_failed(
    version_dir: Path, run: dict[str, Any], idx: int, error: str, eval_root: Path | None = None
) -> dict[str, Any]:
    run["items"][idx]["status"] = "failed"
    run["items"][idx]["error"] = error
    save_run(version_dir, run, eval_root)
    return run


def sample_image_path(version_dir: Path, run_id: str, filename: str, eval_root: Path | None = None) -> Path:
    if "/" in filename or "\\" in filename or filename in {"", ".", ".."}:
        raise EvalSamplesError(f"非法 sample filename: {filename!r}")
    if not filename.lower().endswith(_IMAGE_EXT):
        raise EvalSamplesError("eval sample image 只支持 .png")
    path = run_dir(version_dir, run_id, eval_root) / IMAGES_DIRNAME / filename
    base = (run_dir(version_dir, run_id, eval_root) / IMAGES_DIRNAME).resolve()
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
    from studio.services import version_config
    from studio.services.inference.core import LoRASpec, apply_loras

    project = {"id": run["project_id"], "slug": run["project_slug"]}
    version = {"id": run["version_id"], "label": run["version_label"]}
    cfg = version_config.read_version_config(project, version)

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
    # baseline run 用 lora_scale=0（纯底模对照）。不能写 `or 1.0`——0.0 是 falsy 会被
    # 当成「没设」回退到 1.0，baseline 就变成正常 LoRA 跑、Δ 恒为 0。
    _raw_scale = generation.get("lora_scale")
    lora_scale = float(_raw_scale) if _raw_scale is not None else 1.0
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
    scoped_root = (
        Path(str(run.get("eval_root")))
        if str(run.get("storage_scope") or "") == "task" and run.get("eval_root")
        else None
    )
    for idx, item in enumerate(items):
        run = mark_item_running(version_dir, load_run(version_dir, run["run_id"], scoped_root) or run, idx, scoped_root)
        seed = int(item["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        output = sample_image_path(version_dir, run["run_id"], item["filename"], scoped_root)
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
            run = mark_item_done(version_dir, load_run(version_dir, run["run_id"], scoped_root) or run, idx, scoped_root)
        except Exception as exc:  # noqa: BLE001
            run = mark_item_failed(
                version_dir, load_run(version_dir, run["run_id"], scoped_root) or run, idx, str(exc), scoped_root
            )
