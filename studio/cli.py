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
import re
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


_NPM_MIRROR = "https://registry.npmmirror.com"
_PIP_MIRROR = "https://mirrors.aliyun.com/pypi/simple/"


def _npm_call(npm: str, args: list[str], cwd: str, timeout: int = 180) -> int:
    """运行 npm 命令；超时则 kill 并返回 1。"""
    proc = subprocess.Popen([npm] + args, cwd=cwd)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return 1


def npm_install_if_missing(npm: str) -> int:
    # 检查关键 bin 而不只是目录，避免安装不完整时跳过重装
    _bin = "eslint.cmd" if os.name == "nt" else "eslint"
    if NODE_MODULES.exists() and (NODE_MODULES / ".bin" / _bin).exists():
        return 0
    try:
        rel = NODE_MODULES.relative_to(REPO_ROOT)
    except ValueError:
        rel = NODE_MODULES
    print(f"[studio] {rel} 不完整或不存在，运行 npm install（3 分钟超时）...")
    rc = _npm_call(npm, ["install"], str(WEB_DIR), timeout=180)
    if rc != 0:
        print(f"[studio] npm install 失败或超时，切换国内源 ({_NPM_MIRROR}) 重试...")
        rc = subprocess.call(
            [npm, "install", "--registry", _NPM_MIRROR],
            cwd=str(WEB_DIR),
        )
    return rc


def _pip_install(args: list[str]) -> int:
    """运行 pip install；失败时切换阿里云镜像重试。"""
    rc = subprocess.call([find_python(), "-m", "pip", "install"] + args)
    if rc != 0:
        print(f"[studio] pip install 失败，切换国内源 ({_PIP_MIRROR}) 重试...")
        rc = subprocess.call(
            [find_python(), "-m", "pip", "install"] + args
            + ["-i", _PIP_MIRROR],
        )
    return rc


def _ensure_python_deps() -> int:
    """检查关键包（fastapi）是否安装，缺失时自动补装 requirements.txt。"""
    req = REPO_ROOT / "requirements.txt"
    if not req.exists():
        return 0
    try:
        import importlib.util
        if importlib.util.find_spec("fastapi") is not None:
            return 0
    except Exception:
        pass
    print("[studio] 检测到 fastapi 缺失，重新安装 Python 依赖（requirements.txt）...")
    return _pip_install(["-r", str(req)])


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
        print("[studio] 错误：找不到 npm。请安装 Node.js 18+", file=sys.stderr)
        if os.name == "nt":
            print("  Windows：前往 https://nodejs.org 下载安装包，或用 winget install OpenJS.NodeJS.LTS", file=sys.stderr)
        else:
            print("  Ubuntu/Debian：curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs", file=sys.stderr)
            print("  或使用 nvm：curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash && nvm install --lts", file=sys.stderr)
        print("  安装后重新运行本命令。", file=sys.stderr)
        return 2
    rc = npm_install_if_missing(npm)
    if rc != 0:
        return rc
    rc = npm_build(npm)
    if rc == 0:
        _write_build_marker()
    return rc


def _current_git_head() -> Optional[str]:
    """当前仓库 HEAD commit hash；非 git 仓 / 没有 git 命令 → None。"""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _write_build_marker() -> None:
    """build 成功后把 HEAD 写到 dist/.built-from。云上下次启动直接对比 HEAD
    决定是否重建，绕开「git pull 不更新 mtime」的坑。"""
    head = _current_git_head()
    if not head:
        return
    try:
        (WEB_DIST / ".built-from").write_text(head, encoding="utf-8")
    except OSError:
        pass


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


def _check_torch_cuda() -> None:
    """启动期检查 torch 是否带 CUDA 支持；CPU-only 版会让训练/生成极慢。"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[studio] torch {torch.__version__}  GPU: {name}  ✓")
            return
        # CUDA 不可用：区分 CPU-only build 与真无 GPU
        cuda_build = getattr(torch.version, "cuda", None)
        if cuda_build is None:
            # CPU-only wheel（最常见的误装情况）
            try:
                from studio.services import onnxruntime_setup
                has_gpu = onnxruntime_setup.detect_cuda().get("available", False)
            except Exception:
                has_gpu = False
            if has_gpu:
                print(
                    f"\n[studio] ⚠  警告：检测到 NVIDIA GPU，但当前安装的是 CPU-only 版 PyTorch"
                    f"（{torch.__version__}）。\n"
                    f"         训练和图片生成将在 CPU 上运行，速度极慢（单步可能需要数十秒）。\n"
                    f"         请卸载后重装 CUDA 版：\n"
                    f"           pip uninstall torch torchvision -y\n"
                    f"           # 根据你的 CUDA 版本选择，例如 CUDA 12.8：\n"
                    f"           pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n",
                    file=sys.stderr,
                )
            else:
                print(f"[studio] torch {torch.__version__}（CPU-only build，未检测到 NVIDIA GPU）")
        else:
            # CUDA build 但 GPU 不可用（驱动问题、WSL 等）
            print(
                f"\n[studio] ⚠  警告：torch {torch.__version__}（CUDA {cuda_build} build）"
                f" 但 torch.cuda.is_available()=False。\n"
                f"         可能原因：NVIDIA 驱动未安装 / 版本过低 / WSL 缺少 CUDA 支持。\n",
                file=sys.stderr,
            )
    except ImportError:
        pass  # torch 未装，_ensure_python_deps 会处理


def _bootstrap_onnxruntime() -> None:
    """PP8 — 启动期检测 GPU 后按需装 onnxruntime / onnxruntime-gpu。

    requirements.txt 不写死它，避免用户机器 CUDA 与硬编码包不匹配踩坑。

    安装路径（onnxruntime 未装时）：
    - 直接调 cli._pip_install()，终端实时可见进度，失败自动切国内镜像重试
    - 失败不致命：打印警告后继续启动，WD14 打标不可用但其余功能正常

    EP 检查路径（onnxruntime 已装时）：
    - 委托 onnxruntime_setup.bootstrap()（仅做 GPU/EP 匹配警告，不重装）
    """
    try:
        from studio.services import onnxruntime_setup

        # ── 步骤 1：检查是否已安装（不 import .pyd，只读 dist-info）────────
        rt = onnxruntime_setup.current_runtime()
        if rt["installed"] is None:
            # 未安装：用能显示进度且有镜像回退的 _pip_install() 安装
            cuda = onnxruntime_setup.detect_cuda()
            if cuda["available"]:
                pkg = f"{onnxruntime_setup.GPU_PACKAGE}{onnxruntime_setup.GPU_VERSION_SPEC}"
                gpu_hint = f"（检测到 NVIDIA GPU: {cuda.get('gpu_name', '?')}，装 GPU 版）"
            else:
                pkg = f"{onnxruntime_setup.CPU_PACKAGE}{onnxruntime_setup.CPU_VERSION_SPEC}"
                gpu_hint = "（未检测到 NVIDIA GPU，装 CPU 版）"
            print(f"[studio] ONNX Runtime 未安装，正在安装 {pkg} {gpu_hint}")
            print("[studio] 提示：首次下载可能需要几分钟，进度会实时显示在下方...")
            rc = _pip_install([pkg])
            if rc != 0:
                print(
                    f"[studio] 警告：ONNX Runtime 安装失败（见上方输出）。\n"
                    f"         WD14 打标功能暂不可用；其余功能正常。\n"
                    f"         可稍后在 Settings → WD14 页面手动重装。",
                    file=sys.stderr,
                )
                return
            print(f"[studio] ONNX Runtime 安装完成：{pkg}")
            # 刷新状态（供后续 EP 检查）
            rt = onnxruntime_setup.current_runtime()

        # ── 步骤 2：已装（或刚装完）→ 检查 GPU/EP 匹配，打印状态 ────────
        cuda = onnxruntime_setup.detect_cuda()
        ver = rt.get("version") or "?"
        installed = rt.get("installed") or "?"
        cuda_available = rt.get("cuda_available", False)

        if cuda_available:
            print(f"[studio] onnxruntime: {installed}=={ver}（CUDA EP 可用）")
        elif cuda.get("available"):
            print(
                f"[studio] 警告：检测到 NVIDIA GPU 但 onnxruntime 只有 CPU EP "
                f"（installed={installed}）。WD14 打标会跑 CPU（较慢）。"
                f"可在 Settings → WD14 点「重装为 GPU 版」。",
                file=sys.stderr,
            )
        else:
            print(f"[studio] onnxruntime: {installed}=={ver}（CPU only，未检测到 NVIDIA GPU）")

    except Exception as exc:  # noqa: BLE001
        print(f"[studio] onnxruntime bootstrap 异常（已忽略）: {exc}", file=sys.stderr)


WEB_SRC = WEB_DIR / "src"


def _web_dist_is_stale() -> bool:
    """dist 是否落后于 src。两道检查并联，任一说 stale 就重建。

    1) git HEAD 比对：build 时把 HEAD 写到 dist/.built-from，启动时对比当前
       HEAD。云上 git pull 之后 HEAD 一定变，触发重建——这条是为了兜底
       "git pull 不更新文件 mtime" 在某些 git 版本下不可靠的坑。
    2) mtime 比对：dist/index.html 旧于 src/ 树或 package.json 等关键文件。
       这条是为了兜底本地未 commit 的修改——HEAD 不变但磁盘上的文件确实
       新过 dist，应当重建。

    曾经把 mtime 降级成 fallback（HEAD 一致就跳过），导致本地编辑后
    `studio run` 看不到变化。改并联后云上 git pull 行为不受影响，本地 dev
    iteration 也不需要每改完都 commit。
    """
    dist_index = WEB_DIST / "index.html"
    if not dist_index.exists():
        return True

    # 第一道：git HEAD 比对
    marker = WEB_DIST / ".built-from"
    head = _current_git_head()
    if head and marker.exists():
        try:
            built_from = marker.read_text(encoding="utf-8").strip()
            if built_from != head:
                return True
        except OSError:
            pass

    # 第二道：mtime 比对
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
        if src_latest > dist_mtime:
            return True
    except OSError:
        pass

    return False


def cmd_run(args: argparse.Namespace) -> int:
    rc = _ensure_python_deps()
    if rc != 0:
        return rc
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
    _check_torch_cuda()
    _bootstrap_onnxruntime()
    url = f"http://{args.host}:{args.port}/studio/"

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"studio_{time.strftime('%Y%m%d')}.log"
    print(f"[studio] 后端日志: {log_file}")
    print(f"[studio] 启动后端 → {url}")

    if not args.no_browser:
        _spawn_browser_opener(url)

    cmd = [find_python(), "-u", "-m", "studio.server",
           "--host", args.host, "--port", str(args.port)]
    # uvicorn access log 行：INFO:     IP:PORT - "METHOD /path HTTP/1.x" STATUS
    _access_log_re = re.compile(rb'INFO:\s+[\d.]+:\d+ - "(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS) ')

    with open(log_file, "ab") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        fd = proc.stdout.fileno()
        line_buf = b""
        try:
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                # 终端：原样输出
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                # 日志文件：按行过滤掉 access log
                line_buf += chunk
                while b"\n" in line_buf:
                    line, line_buf = line_buf.split(b"\n", 1)
                    if not _access_log_re.search(line):
                        lf.write(line + b"\n")
                        lf.flush()
        except (KeyboardInterrupt, OSError):
            pass
        # 剩余不完整行
        if line_buf and not _access_log_re.search(line_buf):
            lf.write(line_buf)
            lf.flush()
        proc.wait()
    return proc.returncode


def cmd_dev(args: argparse.Namespace) -> int:
    rc = _ensure_python_deps()
    if rc != 0:
        return rc
    npm = find_npm()
    if not npm:
        print("[studio] 错误：找不到 npm。请安装 Node.js 18+", file=sys.stderr)
        if os.name == "nt":
            print("  Windows：前往 https://nodejs.org 下载安装包，或用 winget install OpenJS.NodeJS.LTS", file=sys.stderr)
        else:
            print("  Ubuntu/Debian：curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs", file=sys.stderr)
            print("  或使用 nvm：curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash && nvm install --lts", file=sys.stderr)
        print("  安装后重新运行本命令。", file=sys.stderr)
        return 2
    rc = npm_install_if_missing(npm)
    if rc != 0:
        return rc
    _check_torch_cuda()
    _bootstrap_onnxruntime()

    pg = ProcGroup()
    try:
        pg.spawn("frontend", [npm, "run", "dev"], cwd=WEB_DIR)
        pg.spawn(
            "backend",
            [
                find_python(), "-u",
                "-m", "studio.server",
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
