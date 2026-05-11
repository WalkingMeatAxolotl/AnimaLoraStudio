"""PR-6 — model_downloader._on_log per-line print。
PR-S3 — _resolve_endpoint env > secrets > None 优先级。
MS-1  — _get_download_source / download_flat_ms rename+cleanup 逻辑。
"""
from __future__ import annotations

import threading
import time

import pytest

from studio.services import model_downloader


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
    monkeypatch.setattr(model_downloader, "_get_download_source", lambda: "modelscope")

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

    monkeypatch.setattr(model_downloader, "download_flat_ms", fake_download_flat_ms)
    monkeypatch.setattr(model_downloader, "download_flat", fake_download_flat)

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
