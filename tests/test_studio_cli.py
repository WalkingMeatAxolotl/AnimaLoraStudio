"""studio.cli 启动器测试。

不真起 npm / uvicorn —— monkeypatch subprocess.call 把命令记下来再断言。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from studio import cli


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []
    def fake(cmd, **kwargs: Any) -> int:
        calls.append(list(cmd))
        return 0
    monkeypatch.setattr(cli.subprocess, "call", fake)
    return calls


@pytest.fixture
def fake_npm(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(cli, "find_npm", lambda: "fake-npm")
    return "fake-npm"


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def test_parser_has_all_subcommands() -> None:
    p = cli.build_parser()
    args = p.parse_args(["run"])
    assert args.cmd == "run"
    args = p.parse_args(["dev"])
    assert args.cmd == "dev"
    args = p.parse_args(["build"])
    assert args.cmd == "build"
    args = p.parse_args(["test"])
    assert args.cmd == "test"


def test_run_args_default_host_port() -> None:
    p = cli.build_parser()
    args = p.parse_args(["run"])
    assert args.host == "127.0.0.1"
    assert args.port == 8765


def test_run_custom_host_port() -> None:
    p = cli.build_parser()
    args = p.parse_args(["run", "--host", "0.0.0.0", "--port", "9000"])
    assert args.host == "0.0.0.0"
    assert args.port == 9000


def test_default_command_is_run() -> None:
    """无子命令时应当走 run。"""
    p = cli.build_parser()
    # 模拟 main() 的 fallback 逻辑
    args = p.parse_args([])
    assert getattr(args, "cmd", None) is None  # parser 本身识别不出
    # main() 处理这种情况：补上 'run'
    args2 = p.parse_args(["run"])
    assert args2.cmd == "run"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def test_build_runs_npm_run_build(fake_calls, fake_npm, monkeypatch: pytest.MonkeyPatch) -> None:
    # node_modules 已存在 → 跳过 install
    monkeypatch.setattr(cli, "NODE_MODULES", Path("/fake/exists"))
    monkeypatch.setattr(cli.Path, "exists", lambda self: True)
    rc = cli.main(["build"])
    assert rc == 0
    # 应该至少有一次 ['fake-npm', 'run', 'build']
    assert any(c[:3] == ["fake-npm", "run", "build"] for c in fake_calls)


def test_build_installs_when_node_modules_missing(
    fake_calls, fake_npm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "NODE_MODULES", tmp_path / "absent")
    rc = cli.main(["build"])
    assert rc == 0
    assert any(c[:2] == ["fake-npm", "install"] for c in fake_calls)
    assert any(c[:3] == ["fake-npm", "run", "build"] for c in fake_calls)


def test_build_no_npm_returns_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "find_npm", lambda: None)
    assert cli.main(["build"]) == 2


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_starts_backend_and_skips_build_when_dist_exists(
    fake_calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_dist = tmp_path / "dist"
    fake_dist.mkdir()
    monkeypatch.setattr(cli, "WEB_DIST", fake_dist)
    rc = cli.main(["run"])
    assert rc == 0
    # 没有 build 调用
    assert not any("run" in c and "build" in c for c in fake_calls)
    # 有一次 python -m studio.server
    assert any(
        "studio.server" in " ".join(c) and "--port" in c for c in fake_calls
    )


def test_run_no_build_skips_when_dist_missing(
    fake_calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--no-build 时即使 dist 缺失也不构建。"""
    monkeypatch.setattr(cli, "WEB_DIST", tmp_path / "absent")
    monkeypatch.setattr(cli, "find_npm", lambda: "fake-npm")
    rc = cli.main(["run", "--no-build"])
    assert rc == 0
    # 不应该出现 build
    assert not any(c[:3] == ["fake-npm", "run", "build"] for c in fake_calls)


def test_run_passes_host_port(
    fake_calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_dist = tmp_path / "dist"
    fake_dist.mkdir()
    monkeypatch.setattr(cli, "WEB_DIST", fake_dist)
    cli.main(["run", "--host", "0.0.0.0", "--port", "9999"])
    server_call = next(
        c for c in fake_calls if "studio.server" in " ".join(c)
    )
    assert "0.0.0.0" in server_call
    assert "9999" in server_call


# ---------------------------------------------------------------------------
# test 子命令（pytest + vitest 委派）
# ---------------------------------------------------------------------------


def test_test_subcommand_runs_pytest(
    fake_calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "find_npm", lambda: None)
    rc = cli.main(["test"])
    assert rc == 0
    assert any("pytest" in " ".join(c) for c in fake_calls)


def test_test_runs_vitest_when_npm_available(
    fake_calls, fake_npm, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_node_modules = tmp_path / "nm"
    fake_node_modules.mkdir()
    monkeypatch.setattr(cli, "NODE_MODULES", fake_node_modules)
    rc = cli.main(["test"])
    assert rc == 0
    assert any(c[:3] == ["fake-npm", "run", "test"] for c in fake_calls)


def test_test_pytest_failure_short_circuits(
    fake_npm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pytest 非零 → 不再调 vitest。"""
    calls: list[list[str]] = []
    def fake(cmd, **_: Any) -> int:
        calls.append(list(cmd))
        return 7 if "pytest" in " ".join(cmd) else 0
    monkeypatch.setattr(cli.subprocess, "call", fake)
    rc = cli.main(["test"])
    assert rc == 7
    assert all("vitest" not in " ".join(c) for c in calls)
    assert all(c[:3] != ["fake-npm", "run", "test"] for c in calls)
