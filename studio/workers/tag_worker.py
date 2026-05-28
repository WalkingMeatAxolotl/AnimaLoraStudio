"""打标 worker（PP4）。

`python -m studio.workers.tag_worker --job-id N`。读 `project_jobs.params`：
    {
      "tagger": "wd14" | "cltagger" | "joycaption",
      "version_id": int,
      "output_format": "txt"|"json",  # 默认 "txt"，已存在的 .json 仍按 .json 写
      "<tagger>_overrides": {...}     # 可选；本次任务对全局 settings 的覆盖
    }

打标永远覆盖 train/ 下全部 repeat 子目录（不再支持按 folder 划分）。

日志只走 stdout：见 `download_worker.py` 顶部的说明。
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from ._base import reconfigure_console_utf8

reconfigure_console_utf8()

# PP9.5 — 必须在任何 `import onnxruntime` 之前 import 本模块，触发顶层 preload
# （Linux: RTLD_GLOBAL 加载 torch 自带 CUDA so；Windows: os.add_dll_directory）。
# cli.py / server.py 已覆盖各自进程；worker 是独立 subprocess，靠 get_tagger
# 懒加载链触发太晚（懒加载在 main() 里，某些路径下来不及）—— worker 顶层显式 import。
from studio.services.runtime import onnxruntime as onnxruntime_setup  # noqa: F401

from studio import db
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.services.dataset.scan import IMAGE_EXTS
from studio.services.tagging.caption_format import (
    caption_json_to_text,
    standard_to_documented_full,
)
from studio.services.dataset import tagedit
from studio.services.tagging.base import get_tagger


def _collect_images(train_dir: Path) -> list[Path]:
    if not train_dir.exists():
        return []
    out: list[Path] = []
    for d in (sub for sub in train_dir.iterdir() if sub.is_dir()):
        out.extend(
            sorted(
                f for f in d.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            )
        )
    return out


def run(job_id: int) -> int:
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        print(f"[error] job {job_id} not found", flush=True)
        return 1
    if job["kind"] != "tag":
        print(f"[error] wrong kind: {job['kind']}", flush=True)
        return 1

    params: dict[str, Any] = job.get("params_decoded") or {}

    def progress(line: str) -> None:
        print(line, flush=True)

    try:
        tagger_name = params.get("tagger", "wd14")
        version_id = int(params["version_id"])
        fmt = str(params.get("output_format", "txt"))
        # 触发词：worker 端 prepend 到 caption 第一位。空串 / 缺省 = 不启用。
        trigger_word = str(params.get("trigger_word") or "").strip()
        # 约定：每个支持本次覆盖的 tagger 都把 overrides 存在 `<name>_overrides` 键下
        overrides = params.get(f"{tagger_name}_overrides") or None

        with db.connection_for() as conn:
            v = versions.get_version(conn, version_id)
            if not v or v["project_id"] != job["project_id"]:
                progress(f"[error] version {version_id} not in project {job['project_id']}")
                return 1
            p = projects.get_project(conn, v["project_id"])
        assert p is not None
        train_dir = versions.version_dir(p["id"], p["slug"], v["label"]) / "train"

        images = _collect_images(train_dir)
        if not images:
            progress("[done] 没有图可打标（train/ 是空的）")
            return 0

        progress(
            f"[start] tagger={tagger_name} version={v['label']} "
            f"images={len(images)} format={fmt}"
        )
        if trigger_word:
            progress(f"[trigger] '{trigger_word}' 将作为第一个 tag prepend 到每张图")

        tagger = get_tagger(tagger_name, overrides=overrides)
        tagger.prepare()
        if overrides:
            progress(
                f"[overrides] {', '.join(f'{k}={v}' for k, v in overrides.items())}"
            )
        progress(f"[ready] {tagger_name} 已就绪")

        ok = 0
        errs = 0
        for r in tagger.tag(
            images,
            on_progress=lambda d, t: progress(f"[progress] {d}/{t}"),
        ):
            if r.get("error"):
                progress(f"[err] {r['image'].name}: {r['error']}")
                errs += 1
                continue
            _write_caption(
                r["image"],
                r.get("tags") or [],
                fmt,
                caption_text=r.get("caption"),
                caption_json=r.get("caption_json"),
                trigger_word=trigger_word,
            )
            ok += 1
        progress(f"[done] tagged {ok}/{len(images)} (errors={errs})")
        return 0 if ok > 0 or errs == 0 else 1
    except Exception as exc:  # noqa: BLE001
        progress(f"[error] {exc}")
        print(traceback.format_exc(), flush=True)
        return 1


def _prepend_trigger_to_tags(tags: list[str], trigger_word: str) -> list[str]:
    """trigger 作为第一个 tag 注入；已存在（case-insensitive）则跳过。"""
    if not trigger_word:
        return list(tags)
    lower = trigger_word.lower()
    if any((t or "").strip().lower() == lower for t in tags):
        return list(tags)
    return [trigger_word, *tags]


def _prepend_trigger_to_text(text: str, trigger_word: str) -> str:
    """trigger prepend 到逗号分隔的 caption 字符串；已存在则跳过。"""
    if not trigger_word:
        return text
    lower = trigger_word.lower()
    tokens = [t.strip() for t in text.split(",")]
    if any(t.lower() == lower for t in tokens if t):
        return text
    return f"{trigger_word}, {text}" if text else trigger_word


def _write_caption(
    image: Path,
    tags: list[str],
    fmt: str,
    *,
    caption_text: str | None = None,
    caption_json: dict[str, Any] | None = None,
    trigger_word: str = "",
) -> None:
    """fmt 仅决定「不存在 caption 时」用什么格式；已存在的 .json 仍走 .json。

    trigger_word 非空时：
      - 标签列表：作为第 0 项 prepend（去重）
      - 字符串 caption：prepend 到最前
      - JSON：写入 ``meta.trigger`` 字段；caption_utils.build_caption_from_json
        会把它作为输出的第一个 token，不参与 shuffle —— 与 .txt 路径行为一致。
    """
    if caption_json is not None:
        if fmt == "json":
            doc = standard_to_documented_full(caption_json)
            if trigger_word:
                meta = doc.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                meta["trigger"] = trigger_word
                doc["meta"] = meta
            image.with_suffix(".json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            image.with_suffix(".txt").unlink(missing_ok=True)
            return
        text = caption_text if caption_text is not None else caption_json_to_text(caption_json)
        text = _prepend_trigger_to_text(text, trigger_word)
        image.with_suffix(".txt").write_text(text, encoding="utf-8")
        image.with_suffix(".json").unlink(missing_ok=True)
        return
    if fmt == "json" and not image.with_suffix(".txt").exists():
        # 强制写 json（即使没有现成 json 文件）
        data: dict[str, Any] = {"tags": _prepend_trigger_to_tags(tags, trigger_word)}
        if trigger_word:
            data["meta"] = {"trigger": trigger_word}
        image.with_suffix(".json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return
    # 否则交给 tagedit 决定（已有 .json 就写 .json，否则 .txt）。
    # 已有 .json 时 tagedit 保留其他字段（包括 meta.trigger 如有），只覆盖 tags 数组；
    # 这里我们把 trigger prepend 进 tags list，并保证 .json 走 tagedit 的同时也补 meta.trigger。
    new_tags = _prepend_trigger_to_tags(tags, trigger_word)
    written = tagedit.write_tags(image, new_tags)
    if trigger_word and written.suffix == ".json":
        try:
            existing = json.loads(written.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        meta = existing.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta["trigger"] = trigger_word
        existing["meta"] = meta
        written.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    from ._base import worker_main
    worker_main(run)
