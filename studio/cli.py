"""跨平台启动器：替代 studio.bat 用 Python 管理前后端进程。

子命令：
    run    构建前端（如缺）+ 起后端（默认）
    dev    前后端开发模式（Vite 5173 + uvicorn 8765 --reload，并行）
    build  仅构建前端
    test   依次跑 pytest + vitest

入口：
    python -m studio                       # 等同 run
    python -m studio dev
    python -m studio build
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "studio" / "web"
WEB_DIST = WEB_DIR / "dist"
NODE_MODULES = WEB_DIR / "node_modules"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def find_npm() -> Optional[str]:
    """Windows 上 npm 是 .cmd / .ps1，需要找全名。"""
    for candidate in ("npm", "npm.cmd", "npm.ps1"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def find_python() -> str:
    """优先用当前解释器（venv 已激活则自然指对）。"""
    return sys.executable


def npm_install_if_missing(npm: str) -> int:
    if NODE_MODULES.exists():
        return 0
    try:
        rel = NODE_MODULES.relative_to(REPO_ROOT)
    except ValueError:
        rel = NODE_MODULES
    print(f"[studio] {rel} 不存在，运行 npm install...")
    return subprocess.call([npm, "install"], cwd=str(WEB_DIR))


def npm_build(npm: str) -> int:
    print("[studio] 构建前端 (npm run build)...")
    return subprocess.call([npm, "run", "build"], cwd=str(WEB_DIR))


# ---------------------------------------------------------------------------
# 子进程协调
# ---------------------------------------------------------------------------


class ProcGroup:
    """同时管理多个子进程；任一进程退出或收到信号都把全部干掉。"""

    def __init__(self) -> None:
        self.procs: list[tuple[str, subprocess.Popen]] = []
        self._stopping = False

    def spawn(
        self,
        label: str,
        cmd: list[str],
        cwd: Optional[Path] = None,
    ) -> subprocess.Popen:
        creationflags = 0
        preexec_fn = None
        if os.name == "nt":
            # CREATE_NEW_PROCESS_GROUP 让我们能给整个组发 CTRL_BREAK_EVENT
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            # POSIX 下放进新进程组，杀的时候用 killpg
            preexec_fn = os.setsid  # type: ignore[assignment]
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )
        print(f"[studio] {label} pid={proc.pid}: {' '.join(cmd)}")
        self.procs.append((label, proc))
        return proc

    def wait_any(self) -> int:
        """阻塞到任一进程退出，返回该进程的 exit code。"""
        while True:
            for label, p in self.procs:
                rc = p.poll()
                if rc is not None:
                    print(f"[studio] {label} 退出 (rc={rc})")
                    return rc
            try:
                # 让 KeyboardInterrupt 有机会触发
                threading.Event().wait(0.5)
            except KeyboardInterrupt:
                return 130

    def stop_all(self, grace: float = 10.0) -> None:
        if self._stopping:
            return
        self._stopping = True
        for label, p in self.procs:
            if p.poll() is not None:
                continue
            print(f"[studio] 停止 {label}...")
            try:
                if os.name == "nt":
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        for label, p in self.procs:
            try:
                p.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                print(f"[studio] {label} 超时未退出，强杀")
                p.kill()


# ---------------------------------------------------------------------------
# 命令实现
# ---------------------------------------------------------------------------


def cmd_build(_args: argparse.Namespace) -> int:
    npm = find_npm()
    if not npm:
        print("[studio] 错误：找不到 npm。请安装 Node 18+", file=sys.stderr)
        return 2
    rc = npm_install_if_missing(npm)
    if rc != 0:
        return rc
    return npm_build(npm)


def _spawn_browser_opener(url: str, *, delay: float = 1.0) -> None:
    """后台等服务起来后用默认浏览器打开 url；失败静默。"""

    def _wait_and_open() -> None:
        deadline = time.monotonic() + 30.0
        time.sleep(delay)
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.5) as resp:
                    if 200 <= resp.status < 500:
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(0.5)
                continue
            except Exception:
                break
        try:
            webbrowser.open(url)
        except Exception:
            pass

    t = threading.Thread(target=_wait_and_open, name="studio-browser", daemon=True)
    t.start()


def _bootstrap_onnxruntime() -> None:
    """PP8 — 启动期检测 GPU 后按需装 onnxruntime / onnxruntime-gpu。

    requirements.txt 不写死它，避免用户机器 CUDA 与硬编码包不匹配踩坑。
    失败不致命（log + 让用户从 Settings 页手动装）。
    """
    try:
        from studio.services import onnxruntime_setup
        state = onnxruntime_setup.bootstrap()
        if state.get("error"):
            print(f"[studio] onnxruntime 自动安装失败: {state['error']}", file=sys.stderr)
        elif state.get("cuda_available"):
            ver = state.get("version") or "?"
            print(f"[studio] onnxruntime: {state.get('installed')}=={ver} (CUDA EP available)")
        elif state.get("cuda_detect", {}).get("available"):
            print(
                f"[studio] 警告：检测到 NVIDIA GPU 但 onnxruntime 只有 CPU EP "
                f"(installed={state.get('installed')})。WD14 会跑 CPU。"
                f"去 Settings → WD14 点「重装为 GPU 版」。",
                file=sys.stderr,
            )
        else:
            print(f"[studio] onnxruntime: {state.get('installed')} (CPU only - no NVIDIA GPU detected)")
    except Exception as exc:  # noqa: BLE001
        print(f"[studio] onnxruntime bootstrap 异常（已忽略）: {exc}", file=sys.stderr)


WEB_SRC = WEB_DIR / "src"


def _web_dist_is_stale() -> bool:
    """dist 是否落后于 src（云上 git pull 新代码后旧 dist 还在的常见坑）。

    比较 dist/index.html mtime vs src/ 树最新 mtime。dist 不存在 → True。
    """
    dist_index = WEB_DIST / "index.html"
    if not dist_index.exists():
        return True
    try:
        dist_mtime = dist_index.stat().st_mtime
        src_latest = max(
            (p.stat().st_mtime for p in WEB_SRC.rglob("*") if p.is_file()),
            default=0.0,
        )
        # package.json / vite.config 改了也算
        for f in (WEB_DIR / "package.json", WEB_DIR / "vite.config.ts", WEB_DIR / "tsconfig.json"):
            if f.exists():
                src_latest = max(src_latest, f.stat().st_mtime)
        return src_latest > dist_mtime
    except OSError:
        return False


def cmd_run(args: argparse.Namespace) -> int:
    if not args.no_build:
        if not WEB_DIST.exists():
            print("[studio] studio/web/dist 不存在，先构建前端...")
            rc = cmd_build(args)
            if rc != 0:
                return rc
        elif _web_dist_is_stale():
            print("[studio] studio/web/dist 比 src 旧（git pull 后未重建？），重新构建前端...")
            rc = cmd_build(args)
            if rc != 0:
                return rc
    _bootstrap_onnxruntime()
    url = f"http://{args.host}:{args.port}/studio/"
    print(f"[studio] 启动后端 → {url}")
    if not args.no_browser:
        _spawn_browser_opener(url)
    return subprocess.call(
        [find_python(), "-m", "studio.server", "--host", args.host, "--port", str(args.port)]
    )


def cmd_dev(args: argparse.Namespace) -> int:
    npm = find_npm()
    if not npm:
        print("[studio] 错误：找不到 npm", file=sys.stderr)
        return 2
    rc = npm_install_if_missing(npm)
    if rc != 0:
        return rc
    _bootstrap_onnxruntime()

    pg = ProcGroup()
    try:
        pg.spawn("frontend", [npm, "run", "dev"], cwd=WEB_DIR)
        pg.spawn(
            "backend",
            [
                find_python(),
                "-m",
                "studio.server",
                "--host", args.host,
                "--port", str(args.port),
                "--reload",
            ],
        )
        frontend_url = "http://127.0.0.1:5173/studio/"
        print(
            f"[studio] frontend → {frontend_url}  "
            f"backend → http://{args.host}:{args.port}/studio/"
        )
        if not args.no_browser:
            # dev 模式打开 Vite 端口（HMR 能用），不开 backend 端口
            _spawn_browser_opener(frontend_url, delay=2.0)
        rc = pg.wait_any()
    finally:
        pg.stop_all()
    return rc


def cmd_test(_args: argparse.Namespace) -> int:
    """跑 pytest + vitest。任一失败 → 非零退出。"""
    print("[studio] pytest...")
    rc = subprocess.call([find_python(), "-m", "pytest", "tests/"], cwd=str(REPO_ROOT))
    if rc != 0:
        return rc
    npm = find_npm()
    if not npm:
        print("[studio] 跳过 vitest (未安装 npm)")
        return 0
    if not NODE_MODULES.exists():
        print("[studio] 跳过 vitest (node_modules 缺失，先 npm install)")
        return 0
    print("[studio] vitest...")
    return subprocess.call([npm, "run", "test"], cwd=str(WEB_DIR))


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="studio", description="AnimaStudio 启动器")
    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="构建前端（如缺）+ 起后端")
    p_run.add_argument("--host", default="127.0.0.1")
    p_run.add_argument("--port", type=int, default=8765)
    p_run.add_argument("--no-build", action="store_true",
                       help="即使 dist 不存在也不自动 build")
    p_run.add_argument("--no-browser", action="store_true",
                       help="启动后不自动打开浏览器")
    p_run.set_defaults(func=cmd_run)

    p_dev = sub.add_parser("dev", help="前后端开发模式")
    p_dev.add_argument("--host", default="127.0.0.1")
    p_dev.add_argument("--port", type=int, default=8765)
    p_dev.add_argument("--no-browser", action="store_true",
                       help="启动后不自动打开浏览器")
    p_dev.set_defaults(func=cmd_dev)

    p_build = sub.add_parser("build", help="仅构建前端")
    p_build.set_defaults(func=cmd_build)

    p_test = sub.add_parser("test", help="跑 pytest + vitest")
    p_test.set_defaults(func=cmd_test)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        # 默认 run
        args = parser.parse_args(["run", *(argv or [])])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
