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

import logging
import threading
import traceback
from pathlib import Path
from typing import Any

# PR-1 C4: setup_logging 内已统一调 reconfigure_console_utf8。

logger = logging.getLogger(__name__)

# PP9.5 — 必须在任何 `import onnxruntime` 之前 import 本模块，触发顶层 preload。
# auto_tag 路径会内联调 wd14_tagger（line ~105 `get_tagger("wd14")`），worker 是独立
# subprocess，必须自己 import；否则 CUDA EP 静默降级到 CPU，用户看不到任何信号。
from studio.services.runtime import onnxruntime as onnxruntime_setup  # noqa: F401

from studio import db, secrets
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.services.dataset.scan import IMAGE_EXTS
from studio.services.reg import (
    builder as reg_builder,
    dedup as reg_dedup,
    postprocess as reg_postprocess,
)
from studio.services.dataset import tagedit


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


def _run_auto_tag(reg_dir: Path, progress, kind: str = "wd14") -> bool:
    """内联跑 tagger 给 reg 集打标，失败返回 False。

    A3 — `kind` 走 `studio.services.tagging.base.get_tagger`；目前 UI 暴露
    wd14 / cltagger，但底层支持 VALID_TAGGER_NAMES 全集。LLM / JoyCaption 后续
    PR 加，注意它们对 reg 图量（可能比 train 大）的体感是慢/贵。
    """
    images = _collect_reg_images(reg_dir)
    if not images:
        progress("[auto-tag] 没有图，跳过")
        return False
    progress(f"[auto-tag] 启动 {kind}，{len(images)} 张图")
    try:
        from studio.services.tagging.base import get_tagger
        tagger = get_tagger(kind)
        tagger.prepare()
        progress(f"[auto-tag] {kind} 模型就绪")
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
            target_count=params.get("target_count"),  # B1: None = 用 train 总数 / mirror 模式忽略
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
            auto_tag_kind=str(params.get("auto_tag_kind") or "wd14"),
            auto_dedup=bool(params.get("auto_dedup", True)),
            build_mode=str(params.get("build_mode") or "flat"),
            based_on_version=v["label"],
            save_tags=sec.download.save_tags,
            convert_to_png=sec.download.convert_to_png,
            remove_alpha_channel=sec.download.remove_alpha_channel,
        )
        incremental = bool(params.get("incremental", True))
        pp_method = str(params.get("postprocess_method", "smart"))
        pp_max_crop = float(params.get("postprocess_max_crop_ratio", 0.1))
        progress(
            f"[start] version={v['label']} api={api_source} "
            f"max_tags={max_search_tags} auto_tag={opts.auto_tag} "
            f"incremental={incremental} auto_dedup={opts.auto_dedup} "
            f"pp={pp_method}/{pp_max_crop}"
        )

        # full mode：先清掉 reg/（图、子文件夹、meta、.deleted_ids.json），
        # 用户语义是「从零开始」。incremental mode 保留所有已有内容。
        if not incremental and output_dir.exists():
            progress("[start] full mode：清空 reg/ 已有内容")
            reg_builder.clear_reg_dir(output_dir)

        meta = reg_builder.build(
            opts,
            on_progress=progress,
            cancel_event=cancel_event,
            incremental=incremental,
        )
        progress(f"[reg-done] actual={meta.actual_count}/{meta.target_count}")

        # A4 — auto_dedup：build 后扫重复 → 每组留 1 张其余删 → 不够则
        # incremental 补足。最多 MAX_DEDUP_ROUNDS 轮，每轮删数为 0 提前退出。
        if opts.auto_dedup and meta.actual_count > 0:
            MAX_DEDUP_ROUNDS = 3
            for r in range(MAX_DEDUP_ROUNDS):
                if cancel_event.is_set():
                    progress("[dedup] 用户中止")
                    break
                progress(f"[dedup r{r + 1}/{MAX_DEDUP_ROUNDS}] 扫描重复…")
                to_delete = reg_dedup.scan_for_dedup(output_dir)
                if not to_delete:
                    progress(f"[dedup r{r + 1}] 无可删项，结束")
                    break
                purged = reg_dedup.purge_paths(output_dir, to_delete)
                progress(f"[dedup r{r + 1}] 删 {purged['count']} 张")
                if purged["count"] == 0:
                    break  # scan 给了但全部 unlink 失败 / 不存在 → 防死循环
                meta = reg_builder.read_meta(output_dir) or meta
                shortfall = meta.target_count - meta.actual_count
                if shortfall <= 0:
                    progress(f"[dedup r{r + 1}] 已达目标，结束")
                    break
                progress(
                    f"[dedup r{r + 1}] 缺 {shortfall} 张，自动 incremental 补足"
                )
                meta = reg_builder.build(
                    opts,
                    on_progress=progress,
                    cancel_event=cancel_event,
                    incremental=True,
                )
                progress(
                    f"[dedup r{r + 1}] 补足后 actual={meta.actual_count}/{meta.target_count}"
                )

        # PP5.5 — 分辨率聚类后处理（auto_tag 之前，因为打标基于最终图）。
        # A4 顺序：必须在 dedup 之后 —— postprocess 会 resize 图，phash 会变。
        if meta.actual_count > 0:
            _run_postprocess(
                output_dir, progress, cancel_event,
                method=pp_method, max_crop_ratio=pp_max_crop,
            )

        # auto_tag：拉完 + 后处理后内联跑选定 tagger
        auto_ok = False
        if opts.auto_tag and meta.actual_count > 0:
            auto_ok = _run_auto_tag(output_dir, progress, kind=opts.auto_tag_kind)
            reg_builder.update_meta_auto_tagged(
                output_dir, auto_ok, kind=opts.auto_tag_kind,
            )

        return 0 if meta.actual_count > 0 else 1
    except Exception as exc:
        # PR-1 C7: 同 tag_worker — logger.exception 带 trace_id 进 stderr，
        # progress 给人读短摘要。
        logger.exception("reg_build worker crashed (job_id=%s)", job_id)
        progress(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    from ._base import worker_main
    worker_main(run)
