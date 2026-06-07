"""studio.cli 启动器测试。

不真起 npm / uvicorn —— monkeypatch subprocess.call 把命令记下来再断言。
"""
from __future__ import annotations

import hashlib
import os
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

    # _npm_call 用 subprocess.Popen 而不是 subprocess.call；fake_calls fixture
    # 只监 call，install 路径会真起 npm 子进程。本测试拦截 Popen——
    # 同时支持 subprocess.run（_write_build_marker 调 git 走 run，内部 with Popen）。
    class FakePopen:
        def __init__(self, cmd, **kwargs):
            fake_calls.append(list(cmd))
            self.args = cmd
            self.returncode = 0
            self.stdin = None
            self.stdout = None
            self.stderr = None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

        def poll(self):
            return 0

        def communicate(self, input=None, timeout=None):
            return ("", "")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(cli.subprocess, "Popen", FakePopen)

    rc = cli.main(["build"])
    assert rc == 0
    assert any(c[:2] == ["fake-npm", "install"] for c in fake_calls)
    assert any(c[:3] == ["fake-npm", "run", "build"] for c in fake_calls)


def test_build_installs_when_package_file_newer_than_node_modules(
    fake_calls, fake_npm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web_dir = tmp_path / "web"
    node_modules = web_dir / "node_modules"
    bin_dir = node_modules / ".bin"
    bin_dir.mkdir(parents=True)
    marker = node_modules / ".package-lock.json"
    marker.write_text("{}", encoding="utf-8")
    (bin_dir / ("eslint.cmd" if cli.os.name == "nt" else "eslint")).write_text("", encoding="utf-8")
    package_json = web_dir / "package.json"
    package_json.write_text('{"dependencies":{"i18next":"latest"}}', encoding="utf-8")
    package_lock = web_dir / "package-lock.json"
    package_lock.write_text("{}", encoding="utf-8")
    os.utime(marker, (100, 100))
    os.utime(package_json, (200, 200))
    os.utime(package_lock, (100, 100))

    monkeypatch.setattr(cli, "WEB_DIR", web_dir)
    monkeypatch.setattr(cli, "NODE_MODULES", node_modules)

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            fake_calls.append(list(cmd))
            self.args = cmd
            self.returncode = 0
            self.stdin = None
            self.stdout = None
            self.stderr = None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

        def poll(self):
            return 0

        def communicate(self, input=None, timeout=None):
            return ("", "")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(cli.subprocess, "Popen", FakePopen)

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
    # PP9.1 — dist 新鲜度按 dist/index.html 与 src/ 树 mtime 比较；测试里直接屏蔽 stale 检测
    (fake_dist / "index.html").write_text("<html/>")
    monkeypatch.setattr(cli, "WEB_DIST", fake_dist)
    monkeypatch.setattr(cli, "_web_dist_is_stale", lambda: False)
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


# ---------------------------------------------------------------------------
# PR-4 — _check_torch_cuda
# ---------------------------------------------------------------------------


class _FakeTorchVersion:
    def __init__(self, cuda):
        self.cuda = cuda


class _FakeTorch:
    """最小 torch 替身：cuda.is_available / get_device_name + version.cuda + __version__。"""
    def __init__(self, *, available: bool, cuda_build, version: str = "2.5.0", device_name: str = "RTX 5090"):
        self._available = available
        self._device_name = device_name
        self.__version__ = version
        self.version = _FakeTorchVersion(cuda_build)
        # 简化 cuda 命名空间
        outer = self
        class _Cuda:
            @staticmethod
            def is_available():
                return outer._available
            @staticmethod
            def get_device_name(_idx):
                return outer._device_name
        self.cuda = _Cuda()


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, torch_module) -> None:
    """把 _FakeTorch 注入 sys.modules，让 `_check_torch_cuda` 内部 `import torch` 拿到它。"""
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "torch", torch_module)


def test_check_torch_cuda_silent_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """torch 没装时静默返回，让 _ensure_python_deps 接手。"""
    import sys as _sys
    monkeypatch.delitem(_sys.modules, "torch", raising=False)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _fake_import(name, *a, **k):
        if name == "torch":
            raise ImportError("simulated")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    cli._check_torch_cuda()
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err == ""


def test_check_torch_cuda_prints_ok_when_available(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """CUDA 可用 → stdout 一行 OK，无 stderr 噪声。"""
    _install_fake_torch(
        monkeypatch,
        _FakeTorch(available=True, cuda_build="12.8", version="2.5.0", device_name="RTX 5090"),
    )
    cli._check_torch_cuda()
    out = capsys.readouterr()
    assert "RTX 5090" in out.out
    assert "2.5.0" in out.out
    assert out.err == ""  # 一切正常不该写 stderr


def test_check_torch_cuda_warns_on_cpu_only_with_gpu(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """CPU-only torch + 检测到 NVIDIA GPU → 大警告 + 重装命令（最常见误装）。"""
    _install_fake_torch(
        monkeypatch,
        _FakeTorch(available=False, cuda_build=None, version="2.5.0+cpu"),
    )
    from studio.services.runtime import onnxruntime as onnxruntime_setup
    monkeypatch.setattr(
        onnxruntime_setup,
        "detect_cuda",
        lambda: {"available": True, "driver_version": "551.86", "gpu_name": "RTX 5090"},
    )
    cli._check_torch_cuda()
    out = capsys.readouterr()
    assert out.out == ""  # 警告全走 stderr
    assert "CPU-only" in out.err
    assert "pip install torch" in out.err
    assert "--index-url" in out.err
    # 不该再用 emoji
    assert "✓" not in out.err
    assert "⚠" not in out.err


def test_check_torch_cuda_info_on_cpu_only_without_gpu(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """CPU-only torch + 没 NVIDIA GPU → benign info（用户确实在 CPU 机器）。"""
    _install_fake_torch(
        monkeypatch,
        _FakeTorch(available=False, cuda_build=None, version="2.5.0+cpu"),
    )
    from studio.services.runtime import onnxruntime as onnxruntime_setup
    monkeypatch.setattr(
        onnxruntime_setup,
        "detect_cuda",
        lambda: {"available": False, "driver_version": None, "gpu_name": None},
    )
    cli._check_torch_cuda()
    out = capsys.readouterr()
    assert "CPU-only build" in out.out
    assert "未检测到 NVIDIA GPU" in out.out
    assert out.err == ""


def test_check_torch_cuda_warns_on_cuda_build_but_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """CUDA build + cuda.is_available()=False → 驱动 / WSL 警告。"""
    _install_fake_torch(
        monkeypatch,
        _FakeTorch(available=False, cuda_build="12.8", version="2.5.0+cu128"),
    )
    cli._check_torch_cuda()
    out = capsys.readouterr()
    assert "CUDA 12.8 build" in out.err
    assert "is_available()=False" in out.err
    # 该路径不应给出 pip install 重装建议（torch 装得没问题，是驱动 / WSL 问题）
    assert "pip install torch" not in out.err


# ---------------------------------------------------------------------------
# PR-5 — _print_npm_install_hint
# ---------------------------------------------------------------------------


def test_npm_hint_windows(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(cli.os, "name", "nt")
    cli._print_npm_install_hint()
    err = capsys.readouterr().err
    assert "Node.js 18+" in err
    assert "winget install" in err
    assert "nodejs.org" in err
    # 不应该泄漏 Linux-only 内容
    assert "nodesource" not in err
    assert "nvm" not in err


def test_npm_hint_linux_non_root(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli.os, "getuid", lambda: 1000, raising=False)
    cli._print_npm_install_hint()
    err = capsys.readouterr().err
    assert "nodesource.com" in err
    assert "sudo bash" in err  # 非 root → 命令带 sudo 前缀
    assert "nvm" in err
    # 不应该有 Windows-only 内容
    assert "winget" not in err


def test_npm_hint_linux_root_drops_sudo(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli.os, "getuid", lambda: 0, raising=False)
    cli._print_npm_install_hint()
    err = capsys.readouterr().err
    # root 环境直接跑命令，sudo 字样不应出现（避免误导：sudo 在容器里常无）
    assert " sudo " not in err
    assert "sudo bash" not in err
    assert "nodesource.com" in err


def test_cmd_build_no_npm_prints_install_hint(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """cmd_build 找不到 npm 时应通过 _print_npm_install_hint 输出（不只是「找不到 npm」一行）。"""
    monkeypatch.setattr(cli, "find_npm", lambda: None)
    monkeypatch.setattr(cli.os, "name", "nt")
    rc = cli.main(["build"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "winget" in err
    assert "Node.js 18+" in err


# ---------------------------------------------------------------------------
# PR-D — installer 自检（sha256 + exit 42）
# ---------------------------------------------------------------------------


def test_installer_hashes_sha256_per_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """三个 installer 文件存在 → sha256；缺失 → None。键是文件名。"""
    cli_py = tmp_path / "cli.py"
    sh = tmp_path / "studio.sh"
    bat = tmp_path / "studio.bat"   # 故意不创建，模拟跨平台一边没有
    cli_py.write_bytes(b"print('hello')\n")
    sh.write_bytes(b"#!/bin/sh\n")
    monkeypatch.setattr(cli, "_INSTALLER_FILES", (cli_py, sh, bat))
    h = cli._installer_hashes()
    assert h == {
        "cli.py": hashlib.sha256(b"print('hello')\n").hexdigest(),
        "studio.sh": hashlib.sha256(b"#!/bin/sh\n").hexdigest(),
        "studio.bat": None,
    }


def test_installer_hashes_changes_when_content_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "cli.py"
    p.write_bytes(b"v1")
    monkeypatch.setattr(cli, "_INSTALLER_FILES", (p,))
    before = cli._installer_hashes()
    p.write_bytes(b"v2")
    assert cli._installer_hashes() != before


def _stub_run_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """共享 fixture：把 cmd_run 的所有 bootstrap helper 短路，返回伪 dist 目录。

    cmd_run 在每轮 loop 里调一堆 helper（_ensure_python_deps / _apply_*
    / _check_torch_cuda / etc）；本身这些都已经在别处测过，PR-D 的测试
    只关心 server 退出后的 dispatch 逻辑。"""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html/>")
    monkeypatch.setattr(cli, "WEB_DIST", dist)
    monkeypatch.setattr(cli, "_web_dist_is_stale", lambda: False)
    monkeypatch.setattr(cli, "_ensure_python_deps", lambda: 0)
    monkeypatch.setattr(cli, "_apply_update_pending", lambda: None)
    monkeypatch.setattr(cli, "_apply_pending_install", lambda: None)
    monkeypatch.setattr(cli, "_check_torch_cuda", lambda: None)
    monkeypatch.setattr(cli, "_try_enable_flash_attn", lambda: None)
    monkeypatch.setattr(cli, "_check_onnxruntime", lambda: None)
    monkeypatch.setattr(cli, "_spawn_browser_opener", lambda *a, **k: None)
    return dist


def test_cmd_run_returns_42_when_installer_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """server 退出 + restart flag 在 + installer hash 变了 → return 42, flag 保留。

    保留 flag 是关键：studio.sh / studio.bat wrapper 看到 (exit_code==42 且
    tmp/restart 存在) 才会触发 exec self；只剩 flag 而 exit!=42 走普通 restart。
    """
    _stub_run_bootstrap(monkeypatch, tmp_path)

    fake_flag = tmp_path / "restart"
    fake_flag.touch()
    monkeypatch.setattr(cli, "_RESTART_FLAG", fake_flag)

    # 第一次 (startup snapshot) 返 A；第二次 (server 退出后) 返 B
    call_count = [0]
    def fake_hashes() -> dict[str, str]:
        call_count[0] += 1
        return {"cli.py": "B"} if call_count[0] > 1 else {"cli.py": "A"}
    monkeypatch.setattr(cli, "_installer_hashes", fake_hashes)

    # subprocess.call 直接返 0（不真起 server）
    monkeypatch.setattr(cli.subprocess, "call", lambda *a, **k: 0)

    rc = cli.main(["run", "--no-browser"])
    assert rc == cli._INSTALLER_RELOAD_EXIT_CODE == 42
    assert fake_flag.exists(), "flag 必须保留给 wrapper 走 exec-self 路径"
    assert "launcher 文件更新" in capsys.readouterr().out


def test_cmd_run_normal_restart_when_installer_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """server 退出 + restart flag 在 + installer hash 没变 → 删 flag, 重新 loop。"""
    _stub_run_bootstrap(monkeypatch, tmp_path)

    fake_flag = tmp_path / "restart"
    monkeypatch.setattr(cli, "_RESTART_FLAG", fake_flag)
    monkeypatch.setattr(cli, "_installer_hashes", lambda: {"cli.py": "stable"})

    # 第一轮 server 退出时写 flag（模拟 /api/system/restart），第二轮不写
    server_calls = [0]
    def fake(cmd, **_: Any) -> int:
        if "studio.server" in " ".join(cmd):
            if server_calls[0] == 0:
                fake_flag.touch()
            server_calls[0] += 1
        return 0
    monkeypatch.setattr(cli.subprocess, "call", fake)

    rc = cli.main(["run", "--no-browser"])
    assert rc == 0
    assert not fake_flag.exists(), "normal restart 走完后 flag 应被删"
    assert server_calls[0] == 2, "正常 restart 应当 loop 两次"


def test_cmd_run_no_flag_no_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """server 退出但 tmp/restart 不存在 → 直接 return rc，不查 installer hash。

    这条保证 PR-D 的 dispatch 不会拦截"用户 Ctrl+C 退出 server"这种正常下班路径。
    """
    _stub_run_bootstrap(monkeypatch, tmp_path)

    fake_flag = tmp_path / "restart"  # 不 touch
    monkeypatch.setattr(cli, "_RESTART_FLAG", fake_flag)

    # 如果 dispatch 看了 installer hash，这里会因为返回值变化导致 rc==42；
    # 测试期望根本不查
    hash_calls = [0]
    def fake_hashes() -> dict[str, str]:
        hash_calls[0] += 1
        return {"cli.py": "v"} if hash_calls[0] == 1 else {"cli.py": "v-changed"}
    monkeypatch.setattr(cli, "_installer_hashes", fake_hashes)

    monkeypatch.setattr(cli.subprocess, "call", lambda *a, **k: 7)

    rc = cli.main(["run", "--no-browser"])
    assert rc == 7  # 直接 return server 退出码
    # startup 时调过一次；server 退出后 flag 不在，不应再调
    assert hash_calls[0] == 1
