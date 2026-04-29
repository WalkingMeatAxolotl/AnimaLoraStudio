"""打标 worker（PP4）。

`python -m studio.workers.tag_worker --job-id N`。读 `project_jobs.params`：
    {
      "tagger": "wd14" | "joycaption",
      "version_id": int,
      "output_format": "txt"|"json"  # 默认 "txt"，已存在的 .json 仍按 .json 写
    }

打标永远覆盖 train/ 下全部 repeat 子目录（不再支持按 folder 划分）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Windows 控制台默认 cp932/cp936，写中文 / emoji 会 UnicodeEncodeError。
# 强制 stdout/stderr 用 UTF-8 + 替换不可编码字符，让 progress 永远不抛。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from studio import db, project_jobs, projects, versions
from studio.datasets import IMAGE_EXTS
from studio.services import tagedit
from studio.services.tagger import get_tagger


def _open_log(log_path: str):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    return open(log_path, "a", encoding="utf-8", buffering=1)


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
    log_path = job.get("log_path") or str(project_jobs.log_path_for(job_id))

    with _open_log(log_path) as log_fp:
        def progress(line: str) -> None:
            log_fp.write(line + "\n")
            print(line, flush=True)

        try:
            tagger_name = params.get("tagger", "wd14")
            version_id = int(params["version_id"])
            fmt = str(params.get("output_format", "txt"))
            wd14_overrides = params.get("wd14_overrides") or None

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

            tagger = get_tagger(tagger_name, overrides=wd14_overrides)
            tagger.prepare()
            if wd14_overrides:
                progress(
                    f"[overrides] {', '.join(f'{k}={v}' for k, v in wd14_overrides.items())}"
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
                _write_caption(r["image"], r.get("tags") or [], fmt)
                ok += 1
            progress(f"[done] tagged {ok}/{len(images)} (errors={errs})")
            return 0 if ok > 0 or errs == 0 else 1
        except Exception as exc:  # noqa: BLE001
            progress(f"[error] {exc}")
            import traceback
            log_fp.write(traceback.format_exc())
            return 1


def _write_caption(image: Path, tags: list[str], fmt: str) -> None:
    """fmt 仅决定「不存在 caption 时」用什么格式；已存在的 .json 仍走 .json。"""
    if fmt == "json" and not image.with_suffix(".txt").exists():
        # 强制写 json（即使没有现成 json 文件）
        data = {"tags": list(tags)}
        image.with_suffix(".json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return
    # 否则交给 tagedit 决定（已有 .json 就写 .json，否则 .txt）
    tagedit.write_tags(image, tags)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", type=int, required=True)
    args = p.parse_args()
    sys.exit(run(args.job_id))


if __name__ == "__main__":
    main()
