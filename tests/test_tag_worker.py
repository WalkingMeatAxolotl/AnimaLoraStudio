"""PP4 — tag worker: 用 mock tagger 跑通 worker 主流程，验证 caption 落盘。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio import db
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.services.tagging import base as tagger_mod
from studio.workers import tag_worker


class _FakeTagger:
    name = "wd14"
    requires_service = False

    def is_available(self):
        return True, "ok"

    def prepare(self):
        pass

    def tag(self, paths, on_progress=lambda d, t: None):
        for i, p in enumerate(paths):
            on_progress(i + 1, len(paths))
            yield {"image": p, "tags": [f"tag_{p.stem}", "common"]}


class _FakeLLMTagger:
    name = "llm"
    requires_service = True

    def is_available(self):
        return True, "ok"

    def prepare(self):
        pass

    def tag(self, paths, on_progress=lambda d, t: None):
        for i, p in enumerate(paths):
            on_progress(i + 1, len(paths))
            yield {
                "image": p,
                "tags": ["1girl", "watercolor", "blue background"],
                "caption": "1girl, watercolor, blue background. Soft watercolor style.",
                "caption_json": {
                    "tags": {
                        "quality": [],
                        "count": "1girl",
                        "character": "",
                        "series": "",
                        "artist": "",
                        "appearance": ["watercolor"],
                        "tags": ["soft shading"],
                        "environment": ["blue background"],
                        "nl": "Soft watercolor style.",
                    }
                },
            }


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    # tag_worker 内部 import 用的是 studio.services.tagger.get_tagger；
    # 接受 overrides=... 的新签名（由 PP4 后续 wd14 per-job 覆盖功能引入）。
    monkeypatch.setattr(
        tagger_mod, "get_tagger", lambda name, overrides=None: _FakeTagger()
    )
    monkeypatch.setattr(
        "studio.workers.tag_worker.get_tagger",
        lambda name, overrides=None: _FakeTagger(),
    )
    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="v1")
        # seed train/1_data with one image (1_data is the default folder created
        # by versions.create_version)
        train = versions.version_dir(p["id"], p["slug"], v["label"]) / "train" / "1_data"
        (train / "a.png").write_bytes(b"x")
        (train / "b.png").write_bytes(b"x")
        job = project_jobs.create_job(
            conn,
            project_id=p["id"],
            version_id=v["id"],
            kind="tag",
            params={
                "tagger": "wd14",
                "version_id": v["id"],
                "output_format": "txt",
            },
        )
    return {"db": dbfile, "p": p, "v": v, "job_id": job["id"], "train": train}


def test_run_creates_txt_captions(env) -> None:
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == "tag_a, common"
    assert (env["train"] / "b.txt").read_text(encoding="utf-8") == "tag_b, common"


def test_run_with_json_format(env, monkeypatch: pytest.MonkeyPatch) -> None:
    # 改 job 的 output_format 为 json
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, '$.output_format', 'json') "
            "WHERE id = ?",
            (env["job_id"],),
        )
        conn.commit()
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    data = json.loads((env["train"] / "a.json").read_text(encoding="utf-8"))
    assert data["tags"] == ["tag_a", "common"]


def test_run_passes_wd14_overrides_through(env, monkeypatch) -> None:
    """worker 应把 params['wd14_overrides'] 透传到 get_tagger(overrides=...)。"""
    captured: dict = {}

    def _factory(name: str, overrides=None):
        captured["name"] = name
        captured["overrides"] = overrides
        return _FakeTagger()

    monkeypatch.setattr("studio.workers.tag_worker.get_tagger", _factory)

    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, "
            "'$.wd14_overrides', json(?)) WHERE id = ?",
            (json.dumps({"threshold_general": 0.2}), env["job_id"]),
        )
        conn.commit()

    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert captured["name"] == "wd14"
    assert captured["overrides"] == {"threshold_general": 0.2}


def test_run_passes_cltagger_overrides_through(env, monkeypatch) -> None:
    """worker 应把 params['cltagger_overrides'] 透传到 get_tagger(overrides=...)。"""
    captured: dict = {}

    def _factory(name: str, overrides=None):
        captured["name"] = name
        captured["overrides"] = overrides
        return _FakeTagger()

    monkeypatch.setattr("studio.workers.tag_worker.get_tagger", _factory)

    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, "
            "'$.tagger', 'cltagger', "
            "'$.cltagger_overrides', json(?)) WHERE id = ?",
            (json.dumps({"threshold_character": 0.55}), env["job_id"]),
        )
        conn.commit()

    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert captured["name"] == "cltagger"
    assert captured["overrides"] == {"threshold_character": 0.55}


def test_run_writes_llm_json_caption(env, monkeypatch) -> None:
    monkeypatch.setattr(
        "studio.workers.tag_worker.get_tagger",
        lambda name, overrides=None: _FakeLLMTagger(),
    )
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, "
            "'$.tagger', 'llm', '$.output_format', 'json') WHERE id = ?",
            (env["job_id"],),
        )
        conn.commit()

    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    data = json.loads((env["train"] / "a.json").read_text(encoding="utf-8"))
    assert data["ai_output"]["count"] == "1girl"
    assert data["ai_output"]["appearance"] == ["watercolor"]
    assert data["ai_output"]["environment"] == ["blue background"]
    assert data["ai_output"]["nl"] == "Soft watercolor style."
    assert not (env["train"] / "a.txt").exists()


def test_run_writes_llm_txt_caption(env, monkeypatch) -> None:
    monkeypatch.setattr(
        "studio.workers.tag_worker.get_tagger",
        lambda name, overrides=None: _FakeLLMTagger(),
    )
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, '$.tagger', 'llm') "
            "WHERE id = ?",
            (env["job_id"],),
        )
        conn.commit()

    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == (
        "1girl, watercolor, blue background. Soft watercolor style."
    )


def test_run_unknown_job(env) -> None:
    assert tag_worker.run(99999) == 1


# ---------------------------------------------------------------------------
# trigger_word (PR #102): tag worker 把 trigger 写为 caption 第一个 tag。
# ---------------------------------------------------------------------------


def _set_trigger(env, trigger: str) -> None:
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, '$.trigger_word', ?) "
            "WHERE id = ?",
            (trigger, env["job_id"]),
        )
        conn.commit()


def test_trigger_word_prepended_to_txt(env) -> None:
    _set_trigger(env, "ohwx")
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    # trigger 是第一个 tag；原始 tagger 给的 ["tag_a", "common"] 跟在后面
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == "ohwx, tag_a, common"
    assert (env["train"] / "b.txt").read_text(encoding="utf-8") == "ohwx, tag_b, common"


def test_trigger_word_writes_meta_in_simple_json(env) -> None:
    """fmt=json + tagger 只给 tags list（无 caption_json）→ meta.trigger 写入。"""
    _set_trigger(env, "ohwx")
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, '$.output_format', 'json') "
            "WHERE id = ?",
            (env["job_id"],),
        )
        conn.commit()
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    data = json.loads((env["train"] / "a.json").read_text(encoding="utf-8"))
    assert data["tags"] == ["ohwx", "tag_a", "common"]
    assert data["meta"]["trigger"] == "ohwx"


def test_trigger_word_writes_meta_in_llm_json(env, monkeypatch) -> None:
    """LLM tagger + fmt=json → documented_full 增加顶层 meta.trigger。"""
    _set_trigger(env, "ohwx")
    monkeypatch.setattr(
        "studio.workers.tag_worker.get_tagger",
        lambda name, overrides=None: _FakeLLMTagger(),
    )
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, "
            "'$.tagger', 'llm', '$.output_format', 'json') WHERE id = ?",
            (env["job_id"],),
        )
        conn.commit()
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    data = json.loads((env["train"] / "a.json").read_text(encoding="utf-8"))
    # documented_full 主体不变
    assert data["ai_output"]["count"] == "1girl"
    # 顶层 meta.trigger 注入
    assert data["meta"]["trigger"] == "ohwx"


def test_trigger_word_prepended_to_llm_txt(env, monkeypatch) -> None:
    """LLM tagger + fmt=txt → caption_text 前 prepend trigger。"""
    _set_trigger(env, "ohwx")
    monkeypatch.setattr(
        "studio.workers.tag_worker.get_tagger",
        lambda name, overrides=None: _FakeLLMTagger(),
    )
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, '$.tagger', 'llm') "
            "WHERE id = ?",
            (env["job_id"],),
        )
        conn.commit()
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == (
        "ohwx, 1girl, watercolor, blue background. Soft watercolor style."
    )


def test_trigger_word_idempotent_when_already_present(env) -> None:
    """tagger 已经返回了 trigger（边界情况）→ 不重复 prepend（case-insensitive）。"""

    class _TaggerWithTrigger:
        name = "wd14"
        requires_service = False

        def is_available(self):
            return True, "ok"

        def prepare(self):
            pass

        def tag(self, paths, on_progress=lambda d, t: None):
            for p in paths:
                yield {"image": p, "tags": ["OHWX", "common"]}  # 大小写不同

    _set_trigger(env, "ohwx")
    import studio.workers.tag_worker as _tw
    orig = _tw.get_tagger
    _tw.get_tagger = lambda name, overrides=None: _TaggerWithTrigger()
    try:
        rc = tag_worker.run(env["job_id"])
    finally:
        _tw.get_tagger = orig
    assert rc == 0
    # 原 "OHWX" 保留（不替换大小写），不再前置一份小写的 "ohwx"
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == "OHWX, common"


def test_trigger_word_empty_string_no_op(env) -> None:
    """trigger_word="" → 与不传等效，输出和老行为一致。"""
    _set_trigger(env, "")
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == "tag_a, common"


def test_trigger_word_dataset_loader_sees_meta(env, tmp_path: Path) -> None:
    """端到端：worker 写 meta.trigger → utils.caption_utils.build_caption_from_json
    把 trigger 输出在第一位。验证 dataset 训练时确实拿到 trigger。"""
    import sys

    # 把仓库根加进 sys.path 让 `import utils.caption_utils` 工作
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from utils.caption_utils import load_and_build_caption

    _set_trigger(env, "ohwx")
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, '$.output_format', 'json') "
            "WHERE id = ?",
            (env["job_id"],),
        )
        conn.commit()
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    caption = load_and_build_caption(env["train"] / "a.json", shuffle=False)
    assert caption is not None
    # trigger 在第一位，后面跟原 tags
    assert caption.startswith("ohwx,")


# ---------------------------------------------------------------------------
# on_existing: skip / overwrite / append（已有 caption 时的策略）
# ---------------------------------------------------------------------------


def _set_on_existing(env, mode: str) -> None:
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, '$.on_existing', ?) "
            "WHERE id = ?",
            (mode, env["job_id"]),
        )
        conn.commit()


def test_on_existing_skip_preserves_existing_txt(env) -> None:
    """skip：已有 .txt 完全保留，新 tagger 输出被丢弃。"""
    (env["train"] / "a.txt").write_text("manual_a, kept", encoding="utf-8")
    _set_on_existing(env, "skip")
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == "manual_a, kept"
    # b 没有 existing，正常写
    assert (env["train"] / "b.txt").read_text(encoding="utf-8") == "tag_b, common"


def test_on_existing_skip_preserves_existing_json(env) -> None:
    """skip：已有 .json 也不动；不应写出 .txt。"""
    (env["train"] / "a.json").write_text(
        json.dumps({"tags": ["manual_a", "kept"], "extra": "x"}),
        encoding="utf-8",
    )
    _set_on_existing(env, "skip")
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    data = json.loads((env["train"] / "a.json").read_text(encoding="utf-8"))
    assert data["tags"] == ["manual_a", "kept"]
    assert data["extra"] == "x"
    assert not (env["train"] / "a.txt").exists()


def test_on_existing_skip_filters_before_tagger(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """skip：已有 caption 的图片不应进入 tagger，避免 LLM/ONNX 白跑。"""
    seen: list[str] = []

    class _SpyTagger(_FakeTagger):
        def tag(self, paths, on_progress=lambda d, t: None):
            seen.extend(p.name for p in paths)
            yield from super().tag(paths, on_progress=on_progress)

    (env["train"] / "a.txt").write_text("manual_a, kept", encoding="utf-8")
    _set_on_existing(env, "skip")
    monkeypatch.setattr(
        "studio.workers.tag_worker.get_tagger",
        lambda name, overrides=None: _SpyTagger(),
    )

    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert seen == ["b.png"]
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == "manual_a, kept"
    assert (env["train"] / "b.txt").read_text(encoding="utf-8") == "tag_b, common"


def test_on_existing_skip_all_does_not_prepare_tagger(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """skip：全量已有 caption 时不实例化 tagger，尤其避免 LLM 请求。"""
    (env["train"] / "a.txt").write_text("manual_a", encoding="utf-8")
    (env["train"] / "b.txt").write_text("manual_b", encoding="utf-8")
    _set_on_existing(env, "skip")

    def _factory(name: str, overrides=None):
        raise AssertionError("tagger should not be created when all images are skipped")

    monkeypatch.setattr("studio.workers.tag_worker.get_tagger", _factory)

    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == "manual_a"
    assert (env["train"] / "b.txt").read_text(encoding="utf-8") == "manual_b"


def test_on_existing_append_merges_and_dedupes_txt(env) -> None:
    """append：现有 tag 在前，tagger 新 tag 追加在后；重复去重。"""
    (env["train"] / "a.txt").write_text("existing, common", encoding="utf-8")
    _set_on_existing(env, "append")
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    # existing, common 在前；tag_a 追加；common 重复被去重
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == "existing, common, tag_a"
    # b 没有 existing，正常写
    assert (env["train"] / "b.txt").read_text(encoding="utf-8") == "tag_b, common"


def test_on_existing_append_merges_json_preserves_other_fields(env) -> None:
    """append + 已有 .json：merge tags 数组，保留其他顶层字段。"""
    (env["train"] / "a.json").write_text(
        json.dumps({"tags": ["existing"], "extra": "x"}),
        encoding="utf-8",
    )
    _set_on_existing(env, "append")
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    data = json.loads((env["train"] / "a.json").read_text(encoding="utf-8"))
    assert data["tags"] == ["existing", "tag_a", "common"]
    assert data["extra"] == "x"


def test_on_existing_append_with_trigger_word(env) -> None:
    """append + trigger：trigger 已在 existing 里 → 不重复加在末尾。"""
    (env["train"] / "a.txt").write_text("ohwx, manual_a", encoding="utf-8")
    _set_on_existing(env, "append")
    _set_trigger(env, "ohwx")
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    # existing 已有 ohwx，append 段也会 prepend ohwx，但 merge dedupe 保留位置
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == "ohwx, manual_a, tag_a, common"


def test_on_existing_overwrite_is_default(env) -> None:
    """不设 on_existing → 仍是覆盖（保留 PR 前行为）。"""
    (env["train"] / "a.txt").write_text("old_a", encoding="utf-8")
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == "tag_a, common"


def test_on_existing_invalid_falls_back_to_overwrite(env) -> None:
    """非法 on_existing → 当 overwrite 处理（防 worker crash）。"""
    (env["train"] / "a.txt").write_text("old_a", encoding="utf-8")
    _set_on_existing(env, "bogus")
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == "tag_a, common"


# ---------------------------------------------------------------------------
# scope: all / validation / 单个 train 文件夹（held-out 验证集打标）
# ---------------------------------------------------------------------------


def _val_dir(env, folder: str = "1_data") -> Path:
    d = (
        versions.version_dir(env["p"]["id"], env["p"]["slug"], env["v"]["label"])
        / "validation"
        / folder
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def _set_scope(env, scope: str) -> None:
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, '$.scope', ?) "
            "WHERE id = ?",
            (scope, env["job_id"]),
        )
        conn.commit()


def test_scope_default_all_tags_train_and_validation(env) -> None:
    """默认（不设 scope）train + validation 一起打。"""
    val = _val_dir(env)
    (val / "v1.png").write_bytes(b"x")
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert (env["train"] / "a.txt").exists()  # train 打了
    assert (val / "v1.txt").read_text(encoding="utf-8") == "tag_v1, common"  # validation 也打了


def test_scope_validation_only(env) -> None:
    """scope=validation：只打 validation/，train 不动。"""
    val = _val_dir(env)
    (val / "v1.png").write_bytes(b"x")
    _set_scope(env, "validation")
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert (val / "v1.txt").exists()
    assert not (env["train"] / "a.txt").exists()
    assert not (env["train"] / "b.txt").exists()


def test_scope_single_train_folder(env) -> None:
    """scope=<train 文件夹>：只打那个文件夹，validation 不动。"""
    val = _val_dir(env)
    (val / "v1.png").write_bytes(b"x")
    _set_scope(env, "1_data")
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert (env["train"] / "a.txt").exists()
    assert not (val / "v1.txt").exists()


def test_scope_validation_empty_is_noop(env) -> None:
    """scope=validation 但 validation/ 没图 → 不报错、train 不被波及。"""
    _set_scope(env, "validation")
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert not (env["train"] / "a.txt").exists()


def test_worker_imports_onnxruntime_setup_at_module_level() -> None:
    """worker 是独立 subprocess —— 必须在任何 `import onnxruntime` 之前触发
    onnxruntime_setup 的顶层 preload。回归测试：防止有人删掉那行 import 后
    Linux 上又出 CUDA EP 静默降级，UI 看不到任何信号。"""
    import re
    import sys
    src = Path(tag_worker.__file__).read_text(encoding="utf-8")
    assert "from studio.services.runtime import onnxruntime" in src, (
        "tag_worker.py 顶层必须 import onnxruntime 触发 preload；"
        "见 onnxruntime.py 顶部 PP9.5 注释。"
    )
    # worker 文件本身不应出现 `^import onnxruntime` / `^from onnxruntime`（行首真 import）
    bad = re.findall(r"^\s*(?:import onnxruntime|from onnxruntime\b)", src, re.MULTILINE)
    assert not bad, f"worker 不应直接 import onnxruntime；命中: {bad}"
    # 模块加载本身把 onnxruntime_setup 拉进 sys.modules
    assert "studio.services.runtime.onnxruntime" in sys.modules
