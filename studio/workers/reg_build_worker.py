"""正则集构建 worker（PP5）。

`python -m studio.workers.reg_build_worker --job-id N`。读 `project_jobs.params`：
    {
      "version_id": int,
      "target_count": int | null,        # null = 用 train 总图数
      "excluded_tags": [str, ...],
      "auto_tag": bool,
      "api_source": "gelbooru" | "danbooru",  # 可选，默认 gelbooru
    }

凭据从 `secrets.gelbooru` / `secrets.danbooru` 拉。

工作流：
1. reg_builder.build(opts) 落图 + 写 meta.json（auto_tagged=False）
2. 若 auto_tag，内联调 WD14 给 reg/1_general/ 全图打标
3. 失败 catch → meta.auto_tagged 仍 false；reg 集本体保留

不开子进程：把 WD14 直接 import 进来，progress 走同一 log_path。
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any

# Windows console cp932/cp936 → 强制 UTF-8 + replace（同 tag_worker）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from studio import db, project_jobs, projects, secrets, versions
from studio.datasets import IMAGE_EXTS
from studio.services import reg_builder, tagedit


def _open_log(log_path: str):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    return open(log_path, "a", encoding="utf-8", buffering=1)


def _collect_reg_images(reg_dir: Path) -> list[Path]:
    """递归收 reg 目录下所有图片（含子文件夹镜像）。"""
    if not reg_dir.exists():
        return []
    out: list[Path] = []
    for f in reg_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            out.append(f)
    return sorted(out)


def _run_auto_tag(reg_dir: Path, progress) -> bool:
    """内联跑 WD14 给 reg 集打标，失败返回 False。"""
    images = _collect_reg_images(reg_dir)
    if not images:
        progress("[auto-tag] 没有图，跳过")
        return False
    progress(f"[auto-tag] 启动 WD14，{len(images)} 张图")
    try:
        from studio.services.tagger import get_tagger
        tagger = get_tagger("wd14")
        tagger.prepare()
        progress("[auto-tag] WD14 模型就绪")
        ok = 0
        errs = 0
        for r in tagger.tag(
            images,
            on_progress=lambda d, t: progress(f"[auto-tag] {d}/{t}"),
        ):
            if r.get("error"):
                progress(f"[auto-tag err] {r['image'].name}: {r['error']}")
                errs += 1
                continue
            tagedit.write_tags(r["image"], r.get("tags") or [])
            ok += 1
        progress(f"[auto-tag] done {ok}/{len(images)} (errors={errs})")
        return ok > 0
    except Exception as exc:
        progress(f"[auto-tag] 失败: {exc}")
        import traceback
        progress(traceback.format_exc())
        return False


def run(job_id: int) -> int:
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        print(f"[error] job {job_id} not found", flush=True)
        return 1
    if job["kind"] != "reg_build":
        print(f"[error] wrong kind: {job['kind']}", flush=True)
        return 1

    params: dict[str, Any] = job.get("params_decoded") or {}
    log_path = job.get("log_path") or str(project_jobs.log_path_for(job_id))

    cancel_event = threading.Event()  # supervisor 走 SIGTERM；这里只为 API 完整性

    with _open_log(log_path) as log_fp:
        def progress(line: str) -> None:
            log_fp.write(line + "\n")
            print(line, flush=True)

        try:
            version_id = int(params["version_id"])
            with db.connection_for() as conn:
                v = versions.get_version(conn, version_id)
                if not v or v["project_id"] != job["project_id"]:
                    progress(f"[error] version {version_id} not in project {job['project_id']}")
                    return 1
                p = projects.get_project(conn, v["project_id"])
            assert p is not None

            vdir = versions.version_dir(p["id"], p["slug"], v["label"])
            train_dir = vdir / "train"
            output_dir = vdir / "reg" / "1_general"

            sec = secrets.load()
            api_source = str(params.get("api_source", "gelbooru"))
            if api_source == "danbooru":
                user_id = ""
                username = sec.danbooru.username
                api_key = sec.danbooru.api_key
            else:
                user_id = sec.gelbooru.user_id
                username = ""
                api_key = sec.gelbooru.api_key

            opts = reg_builder.RegBuildOptions(
                train_dir=train_dir,
                output_dir=output_dir,
                api_source=api_source,
                user_id=user_id,
                api_key=api_key,
                username=username,
                target_count=params.get("target_count") or None,
                excluded_tags=list(params.get("excluded_tags") or []),
                blacklist_tags=list(sec.download.exclude_tags or []),
                auto_tag=bool(params.get("auto_tag", True)),
                based_on_version=v["label"],
                save_tags=sec.gelbooru.save_tags,
                convert_to_png=sec.gelbooru.convert_to_png,
                remove_alpha_channel=sec.gelbooru.remove_alpha_channel,
            )
            progress(
                f"[start] version={v['label']} api={api_source} "
                f"target={opts.target_count or '<train-total>'} "
                f"auto_tag={opts.auto_tag}"
            )

            meta = reg_builder.build(
                opts, on_progress=progress, cancel_event=cancel_event
            )
            progress(f"[reg-done] actual={meta.actual_count}/{meta.target_count}")

            # auto_tag：拉完后内联跑 WD14
            auto_ok = False
            if opts.auto_tag and meta.actual_count > 0:
                auto_ok = _run_auto_tag(output_dir, progress)
                reg_builder.update_meta_auto_tagged(output_dir, auto_ok)

            return 0 if meta.actual_count > 0 else 1
        except Exception as exc:
            progress(f"[error] {exc}")
            import traceback
            log_fp.write(traceback.format_exc())
            return 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", type=int, required=True)
    args = p.parse_args()
    sys.exit(run(args.job_id))


if __name__ == "__main__":
    main()
