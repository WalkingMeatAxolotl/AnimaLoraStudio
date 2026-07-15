"""PR-6 — model_downloader._on_log per-line print。
PR-S3 — _resolve_endpoint env > secrets > None 优先级。
MS-1  — _get_download_source / download_flat_ms rename+cleanup 逻辑。
"""
from __future__ import annotations

import threading
import time

import pytest

from studio.services import models as model_downloader


@pytest.fixture
def reset_downloads():
    """每个测试用例独立，避免 _DOWNLOADS 全局状态污染。"""
    with model_downloader._LOCK:
        before = dict(model_downloader._DOWNLOADS)
        model_downloader._DOWNLOADS.clear()
    yield
    with model_downloader._LOCK:
        model_downloader._DOWNLOADS.clear()
        model_downloader._DOWNLOADS.update(before)


def _wait_done(key: str, timeout: float = 2.0) -> None:
    """轮询等任务结束（避免依赖 bus / 线程加入）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with model_downloader._LOCK:
            ds = model_downloader._DOWNLOADS.get(key)
        if ds and ds.status in ("done", "failed"):
            return
        time.sleep(0.01)
    raise AssertionError(f"download '{key}' didn't complete in {timeout}s")


def test_on_log_writes_to_ring_buffer_and_stdout(
    reset_downloads, capfd: pytest.CaptureFixture
) -> None:
    """on_log 同时写：(1) ring buffer ds.log，(2) stdout（print(line, flush=True)）。"""
    lines_to_emit = ["downloading file 1", "downloading file 2", "✓ done"]

    def fake_fn(on_log):
        for line in lines_to_emit:
            on_log(line)
        return True

    model_downloader.start_download_async("test-key", fake_fn)
    _wait_done("test-key")

    # ring buffer 完整保留
    with model_downloader._LOCK:
        ds = model_downloader._DOWNLOADS["test-key"]
        assert ds.status == "done"
        assert ds.log == lines_to_emit

    # stdout 也都拿到（用 capfd 抓 fd 级 stdout，覆盖跨线程 print）
    out = capfd.readouterr().out
    for line in lines_to_emit:
        assert line in out


# ---------------------------------------------------------------------------
# PR-S3 — _resolve_endpoint
# ---------------------------------------------------------------------------


def test_resolve_endpoint_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """HF_ENDPOINT 环境变量优先于 secrets。"""
    monkeypatch.setenv("HF_ENDPOINT", "https://my-mirror.example/")
    # secrets 读到不同值，但应被 env 覆盖
    from studio import secrets
    monkeypatch.setattr(
        secrets, "load",
        lambda: secrets.Secrets(huggingface=secrets.HuggingFaceConfig(
            token="", endpoint="https://hf-mirror.com",
        )),
    )
    assert model_downloader._resolve_endpoint() == "https://my-mirror.example/"


def test_resolve_endpoint_falls_back_to_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无 env 时读 secrets.huggingface.endpoint。"""
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    from studio import secrets
    monkeypatch.setattr(
        secrets, "load",
        lambda: secrets.Secrets(huggingface=secrets.HuggingFaceConfig(
            token="", endpoint="https://hf-mirror.com",
        )),
    )
    assert model_downloader._resolve_endpoint() == "https://hf-mirror.com"


def test_resolve_endpoint_returns_none_for_empty_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """secrets.huggingface.endpoint='' → None（让 huggingface_hub 用默认 huggingface.co）。"""
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    from studio import secrets
    monkeypatch.setattr(
        secrets, "load",
        lambda: secrets.Secrets(huggingface=secrets.HuggingFaceConfig(
            token="", endpoint="",
        )),
    )
    assert model_downloader._resolve_endpoint() is None


def test_resolve_endpoint_handles_corrupt_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """secrets.load() 抛异常 → 静默回退 None，不阻断下载。"""
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    from studio import secrets

    def boom():
        raise RuntimeError("simulated corrupt secrets")

    monkeypatch.setattr(secrets, "load", boom)
    assert model_downloader._resolve_endpoint() is None


def test_resolve_endpoint_env_with_whitespace_treated_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF_ENDPOINT='  ' (空白) 视作未设；走 secrets 路径。"""
    monkeypatch.setenv("HF_ENDPOINT", "   ")
    from studio import secrets
    monkeypatch.setattr(
        secrets, "load",
        lambda: secrets.Secrets(huggingface=secrets.HuggingFaceConfig(
            token="", endpoint="https://hf-mirror.com",
        )),
    )
    assert model_downloader._resolve_endpoint() == "https://hf-mirror.com"


def test_on_log_does_not_hold_lock_during_print(
    reset_downloads,
) -> None:
    """print 在锁外：on_log 调用本身不应让 _LOCK 在 I/O 期间被占着。

    检测方式：让 print 阻塞（替成 sleep 兼带计时），同时另一线程尝试拿锁；
    若锁外执行，并发 acquire 应当能立刻成功。
    """
    import builtins

    print_started = threading.Event()
    can_finish_print = threading.Event()

    real_print = builtins.print

    def slow_print(*args, **kwargs):
        print_started.set()
        can_finish_print.wait(timeout=2.0)
        return real_print(*args, **kwargs)

    def fake_fn(on_log):
        builtins.print = slow_print
        try:
            on_log("first")
        finally:
            builtins.print = real_print
        return True

    model_downloader.start_download_async("test-lock", fake_fn)

    assert print_started.wait(timeout=2.0), "fake_fn 没进 print"

    # 此时 _on_log 应已离开 with _LOCK 块（先写 ring buffer 再 print），
    # 主线程能在 100ms 内拿到锁
    acquired = model_downloader._LOCK.acquire(timeout=0.1)
    assert acquired, "_on_log 在 print 期间持锁，违反 PR-6 设计"
    model_downloader._LOCK.release()

    can_finish_print.set()
    _wait_done("test-lock")


# ---------------------------------------------------------------------------
# MS-1 — ModelScope 下载源选择 + download_flat_ms rename/cleanup 逻辑
# ---------------------------------------------------------------------------


def test_get_download_source_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """MODELSCOPE_SOURCE 环境变量优先于 secrets。"""
    monkeypatch.setenv("MODELSCOPE_SOURCE", "modelscope")
    from studio import secrets

    monkeypatch.setattr(
        secrets, "load",
        lambda: secrets.Secrets(download_source="huggingface"),
    )
    assert model_downloader._get_download_source() == "modelscope"


def test_get_download_source_falls_back_to_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """无 env 时读 secrets.download_source。"""
    monkeypatch.delenv("MODELSCOPE_SOURCE", raising=False)
    from studio import secrets

    monkeypatch.setattr(
        secrets, "load",
        lambda: secrets.Secrets(download_source="modelscope"),
    )
    assert model_downloader._get_download_source() == "modelscope"


def test_get_download_source_default_huggingface(monkeypatch: pytest.MonkeyPatch) -> None:
    """secrets 空串时回退 'huggingface'。"""
    monkeypatch.delenv("MODELSCOPE_SOURCE", raising=False)
    from studio import secrets

    monkeypatch.setattr(
        secrets, "load",
        lambda: secrets.Secrets(download_source=""),
    )
    assert model_downloader._get_download_source() == "huggingface"


def test_source_for_reads_per_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """_source_for 按类型读 download_sources；各类型独立。"""
    from studio import secrets
    from studio.services.models import sources

    monkeypatch.delenv("MODELSCOPE_SOURCE", raising=False)
    monkeypatch.setattr(
        secrets, "load",
        lambda: secrets.Secrets(download_sources={"training": "modelscope", "wd14": "huggingface"}),
    )
    assert sources._source_for("training") == "modelscope"
    assert sources._source_for("wd14") == "huggingface"
    assert sources._source_for("upscaler") == "huggingface"  # 种子默认


def test_source_for_env_overrides_all_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """MODELSCOPE_SOURCE env 仍是全局强制覆盖（CLI / CI）。"""
    from studio import secrets
    from studio.services.models import sources

    monkeypatch.setenv("MODELSCOPE_SOURCE", "modelscope")
    monkeypatch.setattr(
        secrets, "load",
        lambda: secrets.Secrets(download_sources={"training": "huggingface"}),
    )
    assert sources._source_for("training") == "modelscope"
    assert sources._source_for("wd14") == "modelscope"


def test_per_type_source_routing_is_isolated(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """training=modelscope 只让训练前置走 MS；wd14（=hf）仍走 HF。"""
    from studio import secrets

    monkeypatch.delenv("MODELSCOPE_SOURCE", raising=False)
    monkeypatch.setattr(
        secrets, "load",
        lambda: secrets.Secrets(
            download_sources={"training": "modelscope", "wd14": "huggingface"}
        ),
    )
    hf: list[str] = []
    ms: list[str] = []

    def fake_hf(repo_id, subpath, target, *, on_log=print):
        hf.append(repo_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return True

    def fake_ms(repo_id, subpath, target, *, on_log=print):
        ms.append(repo_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return True

    monkeypatch.setattr("studio.services.models.sources.download_flat", fake_hf)
    monkeypatch.setattr("studio.services.models.sources.download_flat_ms", fake_ms)

    model_downloader.download_anima_vae(tmp_path, on_log=lambda _l: None)
    assert ms and not hf  # 训练组 → MS

    hf.clear()
    ms.clear()
    model_downloader.download_wd14("SmilingWolf/wd-vit-tagger-v3", tmp_path, on_log=lambda _l: None)
    assert hf and not ms  # WD14 → HF


def test_build_catalog_exposes_download_source_options(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio import secrets

    monkeypatch.delenv("MODELSCOPE_SOURCE", raising=False)
    monkeypatch.setattr(
        secrets, "load",
        lambda: secrets.Secrets(download_sources={"training": "modelscope"}),
    )
    opts = model_downloader.build_catalog(tmp_path)["download_source_options"]
    assert opts["training"] == {"current": "modelscope", "available": ["huggingface", "modelscope"]}
    assert opts["wd14"]["current"] == "huggingface"
    assert opts["cltagger"] == {"current": "huggingface", "available": ["huggingface"]}
    assert opts["taeflux"]["available"] == ["huggingface"]


def test_download_flat_ms_skips_existing(tmp_path: "Path") -> None:
    """target 已存在时跳过，不调 modelscope API。"""
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"dummy")

    logs: list[str] = []
    ok = model_downloader.download_flat_ms(
        "some/repo", "split_files/foo.safetensors", target, on_log=logs.append
    )
    assert ok
    assert any("已存在" in l for l in logs)


def test_download_flat_ms_rename_and_cleanup(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """download_flat_ms 把 model_file_download 落盘的深路径文件移到 target，
    并清理掉空的中间目录。"""
    target = tmp_path / "model.safetensors"
    repo_subpath = "split_files/text_encoders/qwen_3_06b_base.safetensors"

    def fake_download(model_id, file_path, local_dir, **kwargs):
        # 模拟 modelscope 在 local_dir/repo_subpath 落盘
        dest = tmp_path / file_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"weights")

    import types
    fake_mod = types.ModuleType("modelscope.hub.file_download")
    fake_mod.model_file_download = fake_download  # type: ignore[attr-defined]

    import sys
    monkeypatch.setitem(sys.modules, "modelscope", types.ModuleType("modelscope"))
    monkeypatch.setitem(sys.modules, "modelscope.hub", types.ModuleType("modelscope.hub"))
    monkeypatch.setitem(sys.modules, "modelscope.hub.file_download", fake_mod)

    logs: list[str] = []
    ok = model_downloader.download_flat_ms(
        "circlestone-labs/Anima", repo_subpath, target, on_log=logs.append
    )
    assert ok, logs
    assert target.exists()
    assert target.read_bytes() == b"weights"
    # 中间目录应已被清理
    assert not (tmp_path / "split_files").exists()


def test_download_qwen3_modelscope_builds_complete_directory(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """ModelScope 源下载权重后，仍会从 HF 补齐 tokenizer/config 文件，
    使 text_encoders/ 成为 transformers 可直接加载的完整目录。"""
    monkeypatch.setenv("MODELSCOPE_SOURCE", "modelscope")

    ms_calls: list[tuple[str, str, str]] = []
    hf_calls: list[tuple[str, str, str]] = []

    def fake_download_flat_ms(repo_id, repo_subpath, target, *, on_log=print):
        ms_calls.append((repo_id, repo_subpath, str(target)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"ms-weights")
        return True

    def fake_download_flat(repo_id, repo_subpath, target, *, on_log=print):
        hf_calls.append((repo_id, repo_subpath, str(target)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(repo_subpath, encoding="utf-8")
        return True

    monkeypatch.setattr("studio.services.models.sources.download_flat_ms", fake_download_flat_ms)
    monkeypatch.setattr(
        "studio.services.models.sources.download_flat",
        fake_download_flat,
    )

    logs: list[str] = []
    ok = model_downloader.download_qwen3(tmp_path, on_log=logs.append)

    assert ok, logs
    qwen_dir = tmp_path / "text_encoders"
    assert (qwen_dir / "model.safetensors").read_bytes() == b"ms-weights"

    assert ms_calls == [(
        model_downloader.ANIMA_REPO,
        model_downloader.MS_ANIMA_TEXT_ENCODER_PATH,
        str(qwen_dir / "model.safetensors"),
    )]

    expected_hf_files = [f for f in model_downloader.QWEN_FILES if f != "model.safetensors"]
    assert [repo_subpath for _, repo_subpath, _ in hf_calls] == expected_hf_files
    for f in expected_hf_files:
        assert (qwen_dir / f).read_text(encoding="utf-8") == f


def test_download_krea2_main_uses_modelscope_comfy_org_mirror(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """训练源选 MS 时从 Comfy-Org/Krea-2 下载对应 bf16 单文件。"""
    monkeypatch.setenv("MODELSCOPE_SOURCE", "modelscope")
    hf_calls: list[tuple[str, str, str]] = []
    ms_calls: list[tuple[str, str, str]] = []

    def fake_hf(repo_id, repo_subpath, target, *, on_log=print):
        hf_calls.append((repo_id, repo_subpath, str(target)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"checkpoint")
        return True

    def fake_ms(repo_id, repo_subpath, target, *, on_log=print):
        ms_calls.append((repo_id, repo_subpath, str(target)))
        return True

    monkeypatch.setattr("studio.services.models.sources.download_flat", fake_hf)
    monkeypatch.setattr("studio.services.models.sources.download_flat_ms", fake_ms)
    logs: list[str] = []

    assert model_downloader.download_krea2_main(
        tmp_path, "raw", on_log=logs.append,
    )
    assert hf_calls == []
    assert ms_calls == [(
        "Comfy-Org/Krea-2",
        "diffusion_models/krea2_raw_bf16.safetensors",
        str(tmp_path / "diffusion_models" / "krea2-raw-bf16.safetensors"),
    )]


def test_download_qwen3_vl_modelscope_builds_isolated_complete_directory(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODELSCOPE_SOURCE", "modelscope")
    calls: list[tuple[str, str, str]] = []

    def fake_ms(repo_id, repo_subpath, target, *, on_log=print):
        calls.append((repo_id, repo_subpath, str(target)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(repo_subpath, encoding="utf-8")
        return True

    monkeypatch.setattr("studio.services.models.sources.download_flat_ms", fake_ms)
    assert model_downloader.download_qwen3_vl(
        tmp_path, on_log=lambda _line: None,
    )

    target_dir = tmp_path / "text_encoders" / "Qwen_Qwen3-VL-4B-Instruct"
    assert [subpath for _, subpath, _ in calls] == model_downloader.QWEN3_VL_FILES
    assert all(repo == model_downloader.QWEN3_VL_REPO for repo, _, _ in calls)
    assert all((target_dir / filename).exists() for filename in model_downloader.QWEN3_VL_FILES)


def test_trigger_routes_krea2_assets_to_async_downloaders(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[str, object]] = []

    def fake_start(key, fn):
        captured.append((key, fn))
        return None

    calls: list[tuple[str, str]] = []

    def fake_main(root, variant, *, on_log=print):
        assert root == tmp_path
        calls.append(("main", variant))
        return True

    def fake_text(root, *, on_log=print):
        assert root == tmp_path
        calls.append(("text", ""))
        return True

    monkeypatch.setattr("studio.services.models.downloader.models_root", lambda: tmp_path)
    monkeypatch.setattr(
        "studio.services.models.downloader.start_download_async", fake_start,
    )
    monkeypatch.setattr(
        "studio.services.models.downloader.download_krea2_main", fake_main,
    )
    monkeypatch.setattr(
        "studio.services.models.downloader.download_qwen3_vl", fake_text,
    )

    assert model_downloader.trigger("krea2_main", "raw") == "krea2_main:raw"
    assert model_downloader.trigger("krea2_text_encoder") == "krea2_text_encoder"
    assert [key for key, _ in captured] == [
        "krea2_main:raw", "krea2_text_encoder",
    ]
    assert all(fn(lambda _line: None) for _, fn in captured)
    assert calls == [("main", "raw"), ("text", "")]


# ---------------------------------------------------------------------------
# 预处理放大器
# ---------------------------------------------------------------------------


def test_upscaler_path_helpers(tmp_path: "Path") -> None:
    """upscaler_dir / upscaler_target / find_upscaler 路径布局与存在性判断。"""
    assert model_downloader.upscaler_dir(tmp_path) == tmp_path / "upscalers"

    target = model_downloader.upscaler_target("4x-AnimeSharp", tmp_path)
    assert target == tmp_path / "upscalers" / "4x-AnimeSharp.pth"

    # 未下载
    assert model_downloader.find_upscaler("4x-AnimeSharp", tmp_path) is None

    # 已下载
    target.parent.mkdir(parents=True)
    target.write_bytes(b"weights")
    assert model_downloader.find_upscaler("4x-AnimeSharp", tmp_path) == target


def test_upscaler_target_unknown_label() -> None:
    """非法 label 抛 ValueError，避免拼错落到错误路径。"""
    with pytest.raises(ValueError, match="unknown upscaler"):
        model_downloader.upscaler_target("4x-RealNotALabel")


def test_download_upscaler_unknown_label_returns_false() -> None:
    logs: list[str] = []
    ok = model_downloader.download_upscaler("nope", on_log=logs.append)
    assert not ok
    assert any("未知放大器" in l for l in logs)


def test_download_upscaler_delegates_to_download_flat(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """download_upscaler 应走 HF download_flat，参数从 UPSCALER_VARIANTS 取。"""
    calls: list[tuple] = []

    def fake_download_flat(repo_id, repo_subpath, target, *, on_log=print):
        calls.append((repo_id, repo_subpath, str(target)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"esrgan")
        return True

    monkeypatch.setattr("studio.services.models.sources.download_flat", fake_download_flat)
    # 钉死下载源：_get_download_source 读真实 secrets.json，开发机若配了
    # modelscope 会走 download_flat_ms 分支（甚至真实下载），patch 落空。
    # env 优先级最高，保证任何机器上都走 HF 分支。
    monkeypatch.setenv("MODELSCOPE_SOURCE", "huggingface")

    ok = model_downloader.download_upscaler("4x-AnimeSharp", tmp_path, on_log=lambda _l: None)
    assert ok
    assert calls == [(
        "Kim2091/AnimeSharp",
        "4x-AnimeSharp.pth",
        str(tmp_path / "upscalers" / "4x-AnimeSharp.pth"),
    )]


def test_build_catalog_includes_upscalers(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """build_catalog 把 upscalers 段加上；未下载 exists=False；新 schema 字段齐。"""
    from studio import secrets

    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets())
    cat = model_downloader.build_catalog(tmp_path)
    assert "upscalers" in cat
    section = cat["upscalers"]
    assert section["default"] == "4x-AnimeSharp"
    assert section["current"] == "4x-AnimeSharp"  # 默认从 selected_upscaler 来
    labels = [v["label"] for v in section["variants"]]
    assert "4x-AnimeSharp" in labels
    # 新预设也要在
    assert "R-ESRGAN_4x+Anime6B" in labels
    sharp = next(v for v in section["variants"] if v["label"] == "4x-AnimeSharp")
    assert sharp["exists"] is False
    assert sharp["kind"] == "preset"
    assert sharp["hf_repo"] == "Kim2091/AnimeSharp"
    assert sharp["ms_repo"] == "libfishopen/upscaler"
    assert sharp["size_mb"] == 64
    assert sharp["filename"] == "4x-AnimeSharp.pth"
    assert sharp["target_path"].endswith("4x-AnimeSharp.pth")


def test_build_catalog_picks_up_custom_upscaler(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """upscalers/ 下不在预设里的 .pth 文件被列为 kind='custom'。"""
    from studio import secrets

    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets())
    (tmp_path / "upscalers").mkdir()
    (tmp_path / "upscalers" / "my-custom.pth").write_bytes(b"x" * 1024)

    cat = model_downloader.build_catalog(tmp_path)
    variants = cat["upscalers"]["variants"]
    custom = next(v for v in variants if v["label"] == "my-custom.pth")
    assert custom["kind"] == "custom"
    assert custom["filename"] == "my-custom.pth"
    assert custom["exists"] is True
    assert custom["size"] == 1024
    assert custom["hf_repo"] is None
    assert custom["ms_repo"] is None


def test_download_upscaler_uses_modelscope_when_configured(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """download_sources['upscaler']='modelscope' → download_upscaler 走 download_flat_ms。"""
    monkeypatch.setenv("MODELSCOPE_SOURCE", "modelscope")
    ms_calls: list[tuple] = []
    hf_calls: list[tuple] = []

    def fake_ms(repo_id, subpath, target, *, on_log=print):
        ms_calls.append((repo_id, subpath, str(target)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"ms")
        return True

    def fake_hf(repo_id, subpath, target, *, on_log=print):
        hf_calls.append((repo_id, subpath, str(target)))
        return True

    monkeypatch.setattr("studio.services.models.sources.download_flat_ms", fake_ms)
    monkeypatch.setattr("studio.services.models.sources.download_flat", fake_hf)

    ok = model_downloader.download_upscaler("4x-AnimeSharp", tmp_path, on_log=lambda _l: None)
    assert ok
    assert ms_calls == [(
        "libfishopen/upscaler",
        "4x-AnimeSharp.pth",
        str(tmp_path / "upscalers" / "4x-AnimeSharp.pth"),
    )]
    assert hf_calls == []


def test_download_upscaler_fallback_to_ms_when_hf_missing(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-ESRGAN_4x+Anime6B 没 HF 镜像；source=hf 时也应回退到 MS。"""
    monkeypatch.setenv("MODELSCOPE_SOURCE", "huggingface")
    ms_calls: list[tuple] = []

    def fake_ms(repo_id, subpath, target, *, on_log=print):
        ms_calls.append((repo_id, subpath))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"ms")
        return True

    monkeypatch.setattr("studio.services.models.sources.download_flat_ms", fake_ms)
    monkeypatch.setattr(
        "studio.services.models.sources.download_flat",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("HF not expected"))
    )

    ok = model_downloader.download_upscaler(
        "R-ESRGAN_4x+Anime6B", tmp_path, on_log=lambda _l: None
    )
    assert ok
    assert ms_calls and ms_calls[0][0] == "libfishopen/upscaler"


def test_download_upscaler_custom_hf(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """自定义 HF 下载落到 upscalers/{filename}。"""
    calls: list[tuple] = []

    def fake_hf(repo_id, subpath, target, *, on_log=print):
        calls.append((repo_id, subpath, str(target)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return True

    monkeypatch.setattr("studio.services.models.sources.download_flat", fake_hf)

    ok = model_downloader.download_upscaler_custom(
        "hf", "Kim2091/UltraSharp", "4x-UltraSharp.pth",
        tmp_path, on_log=lambda _l: None,
    )
    assert ok
    assert calls == [(
        "Kim2091/UltraSharp",
        "4x-UltraSharp.pth",
        str(tmp_path / "upscalers" / "4x-UltraSharp.pth"),
    )]


def test_download_upscaler_custom_rejects_bad_ext(
    tmp_path: "Path",
) -> None:
    """非 .pth/.safetensors 扩展名直接拒绝（防穿越 / 误传）。"""
    logs: list[str] = []
    ok = model_downloader.download_upscaler_custom(
        "hf", "foo/bar", "evil.sh", tmp_path, on_log=logs.append
    )
    assert not ok
    assert any("扩展名" in l for l in logs)


def test_download_upscaler_custom_rejects_bad_source(
    tmp_path: "Path",
) -> None:
    logs: list[str] = []
    ok = model_downloader.download_upscaler_custom(
        "ftp", "foo/bar", "a.pth", tmp_path, on_log=logs.append
    )
    assert not ok
    assert any("未知下载源" in l for l in logs)


def test_upscaler_target_accepts_custom_filename(tmp_path: "Path") -> None:
    """非预设但合法扩展名的 label 视作 custom 文件名。"""
    target = model_downloader.upscaler_target("my-custom.pth", tmp_path)
    assert target == tmp_path / "upscalers" / "my-custom.pth"


def test_upscaler_target_blocks_path_traversal() -> None:
    for bad in ("../foo.pth", "a/b.pth", "..\\x.pth"):
        with pytest.raises(ValueError):
            model_downloader.upscaler_target(bad)


def test_selected_upscaler_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """selected_upscaler 字段值为空 / 非法 / 不存在文件时回退 DEFAULT_UPSCALER。"""
    from studio import secrets

    class Fake:
        class models:
            selected_upscaler = ""
    monkeypatch.setattr(secrets, "load", lambda: Fake())
    assert model_downloader.selected_upscaler() == model_downloader.DEFAULT_UPSCALER

    Fake.models.selected_upscaler = "totally-not-a-preset.pth"  # 不存在文件
    assert model_downloader.selected_upscaler() == model_downloader.DEFAULT_UPSCALER


def test_download_cltagger_v2_downloads_external_data(
    tmp_path: "Path",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLTagger v2 official ONNX uses a sibling model.onnx.data file."""
    from studio import secrets

    cfg = secrets.CLTaggerConfig(
        model_id="cella110n/cl_tagger_v2",
        model_path="v2_01a/model.onnx",
        tag_mapping_path="v2_01a/model_vocabulary.json",
    )
    calls: list[str] = []

    def fake_download_flat(repo_id, repo_subpath, target, *, on_log=print):
        calls.append(repo_subpath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return True

    monkeypatch.setattr("studio.services.models.sources.download_flat", fake_download_flat)

    ok = model_downloader.download_cltagger(tmp_path, cfg, on_log=lambda _l: None)

    assert ok
    assert calls == [
        "v2_01a/model.onnx",
        "v2_01a/model.onnx.data",
        "v2_01a/model_metadata.json",
        "v2_01a/model_vocabulary.json",
    ]


def test_download_cltagger_v2_normalizes_legacy_root_paths(
    tmp_path: "Path",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Earlier v2 support briefly saved root file paths; keep those configs usable."""
    from studio import secrets

    cfg = secrets.CLTaggerConfig(
        model_id="cella110n/cl_tagger_v2",
        model_path="model.onnx",
        tag_mapping_path="model_vocabulary.json",
    )
    calls: list[str] = []

    def fake_download_flat(repo_id, repo_subpath, target, *, on_log=print):
        calls.append(repo_subpath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return True

    monkeypatch.setattr(
        "studio.services.models.sources.download_flat",
        fake_download_flat,
    )

    ok = model_downloader.download_cltagger(tmp_path, cfg, on_log=lambda _l: None)

    assert ok
    assert calls == [
        "v2_01a/model.onnx",
        "v2_01a/model.onnx.data",
        "v2_01a/model_metadata.json",
        "v2_01a/model_vocabulary.json",
    ]


def test_build_catalog_lists_cltagger_v2_variant_with_external_data(
    tmp_path: "Path",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studio import secrets

    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets())

    cat = model_downloader.build_catalog(tmp_path)
    v2 = next(
        v for v in cat["cltagger"]["variants"]
        if v["label"] == "cl_tagger_v2_v2_01a"
    )

    assert v2["model_id"] == "cella110n/cl_tagger_v2"
    assert v2["model_path"] == "v2_01a/model.onnx"
    assert v2["tag_mapping_path"] == "v2_01a/model_vocabulary.json"
    assert [f["name"] for f in v2["files"]] == [
        "v2_01a/model.onnx",
        "v2_01a/model.onnx.data",
        "v2_01a/model_metadata.json",
        "v2_01a/model_vocabulary.json",
    ]


def test_trigger_cltagger_v2_uses_dedicated_repo_and_target(
    tmp_path: "Path",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The v2 preset lives in cella110n/cl_tagger_v2, not cella110n/cl_tagger."""
    from studio import secrets

    captured = {}

    def fake_start_download_async(key, fn):
        captured["key"] = key
        captured["ok"] = fn(lambda _line: None)
        return None

    def fake_download_cltagger(target_root, cfg, *, on_log=print):
        captured["target_root"] = target_root
        captured["cfg"] = cfg
        return True

    monkeypatch.setattr(
        "studio.services.models.downloader.models_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets())
    monkeypatch.setattr(
        "studio.services.models.downloader.start_download_async",
        fake_start_download_async,
    )
    monkeypatch.setattr(
        "studio.services.models.downloader.download_cltagger",
        fake_download_cltagger,
    )

    key = model_downloader.trigger("cltagger", "cl_tagger_v2_v2_01a")

    assert key == "cltagger:cl_tagger_v2_v2_01a"
    assert captured["key"] == key
    assert captured["ok"] is True
    assert captured["cfg"].model_id == "cella110n/cl_tagger_v2"
    assert captured["cfg"].model_path == "v2_01a/model.onnx"
    assert captured["cfg"].tag_mapping_path == "v2_01a/model_vocabulary.json"
    assert captured["target_root"] == tmp_path / "cltagger" / "cella110n_cl_tagger_v2"


def test_failure_summary_surfaces_gated_hint() -> None:
    """下载失败 message 必须给出可操作原因（token / 授权），而非通用串。"""
    from studio.services.models import downloader as dl

    log = [
        "📥 CLTagger → /models/cltagger",
        "   ✗ model.onnx: 401 Client Error: Cannot access gated repo",
        "   ↳ 该仓库可能是 gated/private：请到 设置→密钥 填 HuggingFace token 后重试。",
    ]
    msg = dl._failure_summary(log)
    assert "↳" in msg
    assert "token" in msg.lower()
    assert "✗" in msg  # 同时带上原始错误


def test_failure_summary_falls_back_to_last_error() -> None:
    from studio.services.models import downloader as dl

    log = ["📥 ...", "   ✗ model.onnx: connection reset"]
    assert dl._failure_summary(log) == "✗ model.onnx: connection reset"


def test_failure_summary_generic_when_no_error_line() -> None:
    from studio.services.models import downloader as dl

    assert dl._failure_summary([]) == "下载失败，详见下载日志"


def test_download_flat_passes_configured_hf_token(
    tmp_path: "Path",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gated/private 仓库需 token：download_flat 应把 secrets 里的 HF token 透传给 hf_hub_download。"""
    import huggingface_hub
    from pathlib import Path
    from studio import secrets
    from studio.services.models import sources

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    fake = secrets.Secrets()
    fake.huggingface.token = "hf_test123"
    monkeypatch.setattr(secrets, "load", lambda: fake)

    captured: dict = {}

    def fake_hf_hub_download(**kwargs):
        captured.update(kwargs)
        p = Path(kwargs["local_dir"]) / kwargs["filename"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    ok = sources.download_flat(
        "cella110n/cl_tagger_v2",
        "v2_01a/model.onnx",
        tmp_path / "model.onnx",
        on_log=lambda _l: None,
    )

    assert ok
    assert captured["token"] == "hf_test123"


def test_download_flat_hints_on_gated_auth_error(
    tmp_path: "Path",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gated 仓库无授权时，除裸错误外应追加可操作提示（提到 token/授权）。"""
    import huggingface_hub
    from studio import secrets
    from studio.services.models import sources

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets())

    def fake_hf_hub_download(**kwargs):
        raise RuntimeError("401 Client Error: Cannot access gated repo for url ...")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    logs: list[str] = []
    ok = sources.download_flat(
        "cella110n/cl_tagger_v2",
        "v2_01a/model.onnx",
        tmp_path / "model.onnx",
        on_log=logs.append,
    )

    assert ok is False
    assert any(("token" in m.lower() or "gated" in m.lower()) for m in logs)


# ---------------------------------------------------------------------------
# eval 指标模型（CLIP / DINO）接入统一下载中心
# ---------------------------------------------------------------------------


def test_build_catalog_includes_eval_metrics(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio import secrets

    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets())
    cat = model_downloader.build_catalog(tmp_path)
    em = cat["eval_metrics"]
    assert em["id"] == "eval_metrics"
    kinds = {v["kind"]: v for v in em["variants"]}
    assert set(kinds) == {"clip", "dino", "ccip"}
    assert kinds["ccip"]["model_id"] == "ccip-caformer-24-randaug-pruned"
    assert kinds["ccip"]["exists"] is False
    assert kinds["ccip"]["size_estimate"] > 0
    assert kinds["clip"]["model_id"] == "openai/clip-vit-base-patch32"
    assert kinds["clip"]["exists"] is False
    assert kinds["clip"]["size_estimate"] > 0  # 已知模型给下载前预估
    assert kinds["clip"]["target_path"].replace("\\", "/").endswith(
        "eval/clip/openai_clip-vit-base-patch32"
    )
    assert cat["download_source_options"]["eval"] == {
        "current": "huggingface", "available": ["huggingface", "modelscope"]
    }
    # 指标 registry（Settings 复选框列表）也随 catalog 暴露
    keys = [m["key"] for m in cat["eval_metric_catalog"]]
    assert keys == ["clip_t", "clip_i", "dino_i", "ccip_i", "tag_recall"]
    assert all({"label", "default", "desc", "note"} <= set(m) for m in cat["eval_metric_catalog"])


def test_trigger_eval_clip_dispatches_to_download(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio import secrets

    captured: dict = {}

    def fake_start(key, fn):
        captured["key"] = key
        captured["ok"] = fn(lambda _l: None)

    def fake_download_eval(kind, model_id, root, *, on_log=print):
        captured["args"] = (kind, model_id, root)
        return True

    monkeypatch.setattr("studio.services.models.downloader.models_root", lambda: tmp_path)
    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets())
    monkeypatch.setattr("studio.services.models.downloader.start_download_async", fake_start)
    monkeypatch.setattr("studio.services.models.downloader.download_eval_model", fake_download_eval)

    key = model_downloader.trigger("eval_dino", "facebook/dinov2-small")
    assert key == "eval_dino:facebook/dinov2-small"
    assert captured["args"] == ("dino", "facebook/dinov2-small", tmp_path)
    assert captured["ok"] is True


def test_download_ccip_model_calls_snapshot_with_variant_patterns(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_snapshot(repo, target, *, allow_patterns=None, on_log=print):
        captured["repo"] = repo
        captured["target"] = str(target)
        captured["patterns"] = allow_patterns
        return True

    monkeypatch.setattr("studio.services.models.sources.download_snapshot", fake_snapshot)
    ok = model_downloader.download_ccip_model("ccip-x", tmp_path)
    assert ok is True
    assert captured["repo"] == "deepghs/ccip_onnx"
    assert captured["target"].replace("\\", "/").endswith("eval/ccip")
    assert set(captured["patterns"]) == {
        "ccip-x/model_feat.onnx", "ccip-x/model_metrics.onnx", "ccip-x/metrics.json",
    }


def test_trigger_eval_ccip_dispatches_to_ccip_download(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_start(key, fn):
        captured["key"] = key
        captured["ok"] = fn(lambda _l: None)

    def fake_ccip(variant, root, *, on_log=print):
        captured["args"] = (variant, root)
        return True

    monkeypatch.setattr("studio.services.models.downloader.models_root", lambda: tmp_path)
    monkeypatch.setattr("studio.services.models.downloader.start_download_async", fake_start)
    monkeypatch.setattr("studio.services.models.downloader.download_ccip_model", fake_ccip)
    key = model_downloader.trigger("eval_ccip", "ccip-caformer-24-randaug-pruned")
    assert key == "eval_ccip:ccip-caformer-24-randaug-pruned"
    assert captured["args"] == ("ccip-caformer-24-randaug-pruned", tmp_path)
    assert captured["ok"] is True


def test_ensure_ccip_model_skips_when_files_present(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("studio.services.models.downloader.models_root", lambda: tmp_path)
    d = model_downloader.ccip_model_dir(tmp_path, "ccip-x")
    d.mkdir(parents=True)
    for f in ("model_feat.onnx", "model_metrics.onnx", "metrics.json"):
        (d / f).write_bytes(b"x")
    called: list = []
    monkeypatch.setattr(
        "studio.services.models.downloader.download_ccip_model",
        lambda *a, **k: called.append(1) or True,
    )
    got = model_downloader.ensure_ccip_model("ccip-x")
    assert got == d
    assert not called  # 三文件齐全，不触发下载


def test_download_eval_model_routes_modelscope_when_mapped(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio import secrets

    monkeypatch.delenv("MODELSCOPE_SOURCE", raising=False)
    monkeypatch.setattr(
        secrets, "load",
        lambda: secrets.Secrets(download_sources={"eval": "modelscope"}),
    )
    ms: list = []
    hf: list = []
    monkeypatch.setattr(
        "studio.services.models.sources.download_snapshot_ms",
        lambda repo, target, *, on_log=print: ms.append((repo, str(target))) or True,
    )
    monkeypatch.setattr(
        "studio.services.models.sources.download_snapshot",
        lambda repo, target, *, on_log=print, allow_patterns=None: hf.append(repo) or True,
    )
    ok = model_downloader.download_eval_model(
        "dino", "facebook/dinov2-small", tmp_path, on_log=lambda _l: None
    )
    assert ok
    assert ms and ms[0][0] == "AI-ModelScope/dinov2-small"
    assert not hf


def test_download_eval_model_falls_back_to_hf_without_ms_map(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio import secrets

    monkeypatch.delenv("MODELSCOPE_SOURCE", raising=False)
    monkeypatch.setattr(
        secrets, "load",
        lambda: secrets.Secrets(download_sources={"eval": "modelscope"}),
    )
    hf: list = []
    monkeypatch.setattr(
        "studio.services.models.sources.download_snapshot",
        lambda repo, target, *, on_log=print, allow_patterns=None: hf.append(repo) or True,
    )
    # 自定义 repo 无魔搭映射 → 回退 HuggingFace
    ok = model_downloader.download_eval_model(
        "clip", "some/custom-clip", tmp_path, on_log=lambda _l: None
    )
    assert ok
    assert hf == ["some/custom-clip"]


def test_ensure_eval_model_returns_existing_without_download(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    target = model_downloader.eval_model_target_dir(
        tmp_path, "clip", "openai/clip-vit-base-patch32"
    )
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    called: list = []
    monkeypatch.setattr(
        "studio.services.models.downloader.download_eval_model",
        lambda *a, **k: called.append(a) or True,
    )
    got = model_downloader.ensure_eval_model(
        "clip", "openai/clip-vit-base-patch32", tmp_path
    )
    assert got == target
    assert not called


def test_ensure_eval_model_downloads_when_missing(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    target = model_downloader.eval_model_target_dir(tmp_path, "dino", "facebook/dinov2-small")
    called: list = []

    def fake_download(kind, model_id, root, *, on_log=print):
        called.append((kind, model_id))
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        return True

    monkeypatch.setattr(
        "studio.services.models.downloader.download_eval_model", fake_download
    )
    got = model_downloader.ensure_eval_model("dino", "facebook/dinov2-small", tmp_path)
    assert got == target
    assert called == [("dino", "facebook/dinov2-small")]


def test_ensure_eval_model_uses_user_local_dir(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """model_id 是用户填的本地已有目录 → 直接用，不当 repo id 下载。"""
    local = tmp_path / "my-local-clip"
    local.mkdir()
    (local / "config.json").write_text("{}", encoding="utf-8")
    called: list = []
    monkeypatch.setattr(
        "studio.services.models.downloader.download_eval_model",
        lambda *a, **k: called.append(a) or True,
    )
    got = model_downloader.ensure_eval_model("clip", str(local), tmp_path)
    assert got == local
    assert not called
