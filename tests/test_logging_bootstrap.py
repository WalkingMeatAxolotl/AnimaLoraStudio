"""PR-1 C4 — webui / cli / worker 三处入口接入 setup_logging 验证。

不真起 server / cli / worker；只验证 callable 的代码路径有正确装。
跟 test_logging_setup.py 不同：那个测 setup_logging 自身行为，本文件测
"3 个 caller 用了正确的 args 调"。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def reset(monkeypatch: pytest.MonkeyPatch):
    """unset NO_BOOTSTRAP 让 setup_logging 真跑；测后 reset。"""
    from studio.infrastructure.logging import _reset_for_tests
    monkeypatch.delenv("ANIMA_LOGGING_NO_BOOTSTRAP", raising=False)
    _reset_for_tests()
    saved = list(logging.getLogger().handlers)
    saved_level = logging.getLogger().level
    saved_excepthook = sys.excepthook
    yield
    _reset_for_tests()
    logging.getLogger().handlers = saved
    logging.getLogger().level = saved_level
    sys.excepthook = saved_excepthook


# ── lifespan ────────────────────────────────────────────────────────────


def test_webui_lifespan_calls_setup_logging(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """触发 lifespan startup 应调 setup_logging("webui")，装 file handler。"""
    monkeypatch.setenv("ANIMA_LOG_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from studio import server

    # 触发 lifespan
    with TestClient(server.app) as client:
        client.get("/api/health")

    # studio.log 应该被创建
    assert (tmp_path / "studio.log").exists(), (
        "lifespan 应该调 setup_logging('webui')，会装 studio.log file handler"
    )


# ── cli.main ────────────────────────────────────────────────────────────


def test_cli_main_calls_setup_logging_without_file_handler() -> None:
    """cli.main 调 setup_logging("cli:<subcmd>", file=False) — 不装 file handler。"""
    from studio import cli
    from studio.infrastructure import logging as _slog

    captured = {}
    real_setup = _slog.setup_logging

    def spy(process, **kwargs):
        captured["process"] = process
        captured["kwargs"] = kwargs
        # 不真装 — 直接 mark sentinel 让幂等返回
        _slog._CONFIGURED_PROCESSES.add(process)

    with patch.object(_slog, "setup_logging", side_effect=spy):
        # build cli 上下文执行到 setup_logging 行，但 args.func 是 cmd_build 立即 return
        # 让 cmd_build 也被 mock 避免真 npm install
        with patch.object(cli, "cmd_build", return_value=0):
            rc = cli.main(["build"])

    assert rc == 0
    assert captured["process"] == "cli:build", f"unexpected process: {captured}"
    assert captured["kwargs"].get("file") is False, (
        "CLI 必须 file=False（不写 studio.log，round 2 §1.3 决策）"
    )
    assert captured["kwargs"].get("console") is True


# ── workers/_base.worker_main ────────────────────────────────────────────


def test_worker_main_calls_setup_logging_without_file_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """worker_main 调 setup_logging("worker:<kind>/<job_id>", file=False)。"""
    from studio.workers import _base
    from studio.infrastructure import logging as _slog

    captured = {}

    def spy(process, **kwargs):
        captured["process"] = process
        captured["kwargs"] = kwargs
        _slog._CONFIGURED_PROCESSES.add(process)

    monkeypatch.setattr(sys, "argv", ["tag_worker.py", "--job-id", "42"])

    with patch.object(_slog, "setup_logging", side_effect=spy):
        with pytest.raises(SystemExit) as excinfo:
            _base.worker_main(lambda job_id: 0)
    assert excinfo.value.code == 0

    assert captured["process"] == "worker:tag/42", f"unexpected process: {captured}"
    assert captured["kwargs"].get("file") is False
    assert captured["kwargs"].get("console") is True


def test_worker_main_respects_process_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """supervisor C6 后会设 ANIMA_PROCESS_NAME；worker 优先用它而不是 argv 推断。"""
    from studio.workers import _base
    from studio.infrastructure import logging as _slog

    captured = {}

    def spy(process, **kwargs):
        captured["process"] = process
        _slog._CONFIGURED_PROCESSES.add(process)

    monkeypatch.setattr(sys, "argv", ["download_worker.py", "--job-id", "7"])
    monkeypatch.setenv("ANIMA_PROCESS_NAME", "worker:custom/99")

    with patch.object(_slog, "setup_logging", side_effect=spy):
        with pytest.raises(SystemExit):
            _base.worker_main(lambda job_id: 0)

    assert captured["process"] == "worker:custom/99", (
        "ANIMA_PROCESS_NAME env 应优先于 argv 推断"
    )


def test_worker_kind_from_argv_strips_worker_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.workers import _base
    for argv, expected in [
        (["tag_worker.py", "--job-id", "1"], "tag"),
        (["preprocess_worker.py"], "preprocess"),
        (["download_worker.py"], "download"),
        (["reg_build_worker.py"], "reg_build"),
        (["weird_script.py"], "weird_script"),  # 不含 _worker 后缀也不抛
    ]:
        monkeypatch.setattr(sys, "argv", argv)
        assert _base._worker_kind_from_argv() == expected


# ── conftest fixture self-check ────────────────────────────────────────


def test_anima_logging_no_bootstrap_env_blocks_setup_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """设了 env 后 setup_logging 应 noop（业务调用全跳过）。"""
    from studio.infrastructure.logging import _CONFIGURED_PROCESSES, setup_logging
    monkeypatch.setenv("ANIMA_LOGGING_NO_BOOTSTRAP", "1")
    setup_logging("test-process-name", log_dir=tmp_path, console=False)
    assert "test-process-name" not in _CONFIGURED_PROCESSES, (
        "env 设了应 noop，不进 sentinel"
    )
    assert not (tmp_path / "studio.log").exists(), "应不写文件"


def test_anima_log_dir_env_overrides_paths_logs_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ANIMA_LOG_DIR env 优先于 paths.LOGS_DIR。"""
    from studio.infrastructure.logging import _resolve_log_dir
    monkeypatch.setenv("ANIMA_LOG_DIR", str(tmp_path / "custom"))
    assert _resolve_log_dir() == tmp_path / "custom"

    monkeypatch.delenv("ANIMA_LOG_DIR", raising=False)
    from studio.infrastructure.paths import LOGS_DIR
    assert _resolve_log_dir() == LOGS_DIR
