"""正则集构建 worker（PP5 + PP5.1 + PP5.5）。

`python -m studio.workers.reg_build_worker --job-id N`。读 `project_jobs.params`：
    {
      "version_id": int,
      "target_count": int | null,        # null = 用 train 总图数
      "excluded_tags": [str, ...],
      "auto_tag": bool,
      "api_source": "gelbooru" | "danbooru",  # 可选，默认 gelbooru
      "incremental": bool,                    # PP5.1，可选，默认 False
    }

凭据从 `secrets.gelbooru` / `secrets.danbooru` 拉。

工作流：
1. reg_builder.build(opts) 落图 + 写 meta.json（auto_tagged=False，postprocessed_at=None）
2. PP5.5 — reg_postprocess.postprocess(reg_dir) 分辨率聚类 + smart resize 到统一分辨率
   - 失败 / 找不到满足 max_crop 的 K → catch，meta.postprocessed_at 仍 None；reg 集保留
3. 若 auto_tag，内联调 WD14 给 reg/ 全图打标
4. 失败 catch → meta.auto_tagged 仍 false；reg 集本体保留

不开子进程：把 WD14 / postprocess 直接 import 进来，progress 走同一 log_path。

日志只走 stdout：见 `download_worker.py` 顶部的说明。
"""
from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any

from ._base import reconfigure_console_utf8

reconfigure_console_utf8()

# PP9.5 — 必须在任何 `import onnxruntime` 之前 import 本模块，触发顶层 preload。
# auto_tag 路径会内联调 wd14_tagger（line ~105 `get_tagger("wd14")`），worker 是独立
# subprocess，必须自己 import；否则 CUDA EP 静默降级到 CPU，用户看不到任何信号。
from studio.services import onnxruntime_setup  # noqa: F401

from studio import db, project_jobs, projects, secrets, versions
from studio.datasets import IMAGE_EXTS
from studio.services import reg_builder, reg_postprocess, tagedit


def _collect_reg_images(reg_dir: Path) -> list[Path]:
    """递归收 reg 目录下所有图片（含子文件夹镜像）。"""
    if not reg_dir.exists():
        return []
    out: list[Path] = []
    for f in reg_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            out.append(f)
    return sorted(out)


def _run_postprocess(
    reg_dir: Path, progress, cancel_event,
    *, method: str = "smart", max_crop_ratio: float = 0.1,
) -> None:
    """PP5.5 — 分辨率聚类后处理。失败不算 fatal；meta 字段反映结果。"""
    import time as _time
    try:
        result = reg_postprocess.postprocess(
            reg_dir,
            method=method,
            max_crop_ratio=max_crop_ratio,
            on_progress=progress,
            cancel_event=cancel_event,
        )
    except Exception as exc:
        progress(f"[postprocess] 失败: {exc}")
        progress(traceback.format_exc())
        reg_builder.update_meta_postprocess(
            reg_dir, when=None, clusters=None, method=None, max_crop_ratio=None
        )
        return
    if result.get("clusters") is None:
        # 找不到满足 max_crop 的 K → 不动文件
        reg_builder.update_meta_postprocess(
            reg_dir, when=None, clusters=None,
            method=result.get("method"),
            max_crop_ratio=result.get("max_crop_ratio"),
        )
        return
    reg_builder.update_meta_postprocess(
        reg_dir,
        when=_time.time(),
        clusters=int(result["clusters"]),
        method=str(result.get("method") or method),
        max_crop_ratio=float(result.get("max_crop_ratio") or max_crop_ratio),
    )


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

    cancel_event = threading.Event()  # supervisor 走 SIGTERM；这里只为 API 完整性

    def progress(line: str) -> None:
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
        output_dir = vdir / "reg"  # 与源脚本一致：直接镜像 train 子文件夹

        sec = secrets.load()
        api_source = str(params.get("api_source", "gelbooru"))
        if api_source == "danbooru":
            user_id = ""
            username = sec.danbooru.username
            api_key = sec.danbooru.api_key
            # account_type 影响 max_search_tags 上限
            account_type = (sec.danbooru.account_type or "free").lower()
            max_search_tags = {
                "free": 2, "gold": 6, "platinum": 12,
            }.get(account_type, 2)
        else:
            user_id = sec.gelbooru.user_id
            username = ""
            api_key = sec.gelbooru.api_key
            max_search_tags = 20  # gelbooru 默认 20

        opts = reg_builder.RegBuildOptions(
            train_dir=train_dir,
            output_dir=output_dir,
            api_source=api_source,
            user_id=user_id,
            api_key=api_key,
            username=username,
            target_count=None,  # 永远镜像 train 总数（与源脚本一致）
            max_search_tags=max_search_tags,
            # batch_size = 搜索循环内部「每批下多少张后重算缺失 tag」，
            # 与 train 子文件夹镜像无关；走源脚本默认 5，UI 不暴露
            skip_similar=bool(params.get("skip_similar", True)),
            aspect_ratio_filter_enabled=bool(
                params.get("aspect_ratio_filter_enabled", False)
            ),
            min_aspect_ratio=float(params.get("min_aspect_ratio", 0.5)),
            max_aspect_ratio=float(params.get("max_aspect_ratio", 2.0)),
            excluded_tags=list(params.get("excluded_tags") or []),
            blacklist_tags=list(sec.download.exclude_tags or []),
            auto_tag=bool(params.get("auto_tag", True)),
            based_on_version=v["label"],
            save_tags=sec.gelbooru.save_tags,
            convert_to_png=sec.gelbooru.convert_to_png,
            remove_alpha_channel=sec.gelbooru.remove_alpha_channel,
        )
        incremental = bool(params.get("incremental", False))
        pp_method = str(params.get("postprocess_method", "smart"))
        pp_max_crop = float(params.get("postprocess_max_crop_ratio", 0.1))
        progress(
            f"[start] version={v['label']} api={api_source} "
            f"max_tags={max_search_tags} auto_tag={opts.auto_tag} "
            f"incremental={incremental} pp={pp_method}/{pp_max_crop}"
        )

        meta = reg_builder.build(
            opts,
            on_progress=progress,
            cancel_event=cancel_event,
            incremental=incremental,
        )
        progress(f"[reg-done] actual={meta.actual_count}/{meta.target_count}")

        # PP5.5 — 分辨率聚类后处理（auto_tag 之前，因为打标基于最终图）
        if meta.actual_count > 0:
            _run_postprocess(
                output_dir, progress, cancel_event,
                method=pp_method, max_crop_ratio=pp_max_crop,
            )

        # auto_tag：拉完后内联跑 WD14
        auto_ok = False
        if opts.auto_tag and meta.actual_count > 0:
            auto_ok = _run_auto_tag(output_dir, progress)
            reg_builder.update_meta_auto_tagged(output_dir, auto_ok)

        return 0 if meta.actual_count > 0 else 1
    except Exception as exc:
        progress(f"[error] {exc}")
        print(traceback.format_exc(), flush=True)
        return 1


if __name__ == "__main__":
    from ._base import worker_main
    worker_main(run)
