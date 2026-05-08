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


def _print_npm_install_hint() -> None:
    """`find_npm()` 返回 None 时打印平台相关安装提示。

    放 stderr，与 `[studio] 错误：找不到 npm` 同流；root 环境去掉 sudo（直接 root 跑装包）。
    """
    print("[studio] 错误：找不到 npm。请安装 Node.js 18+", file=sys.stderr)
    if os.name == "nt":
        print(
            "  Windows：前往 https://nodejs.org 下载安装包，"
            "或用 winget install OpenJS.NodeJS.LTS",
            file=sys.stderr,
        )
    else:
        sudo = "" if (hasattr(os, "getuid") and os.getuid() == 0) else "sudo "
        print(
            f"  Ubuntu/Debian：curl -fsSL https://deb.nodesource.com/setup_22.x "
            f"| {sudo}bash - && {sudo}apt-get install -y nodejs",
            file=sys.stderr,
        )
        print(
            "  或使用 nvm（无需 sudo）："
            "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh "
            "| bash && nvm install --lts",
            file=sys.stderr,
        )
    print("  安装后重新运行本命令。", file=sys.stderr)


def cmd_build(_args: argparse.Namespace) -> int:
    npm = find_npm()
    if not npm:
        _print_npm_install_hint()
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


def _apply_pending_install() -> None:
    """启动期处理 server 进程不能完成的 pip 安装请求（torch 重装）。

    必须在 `_check_torch_cuda` 之前跑：那里会 import torch，之后 .pyd 被锁就装不动了。
    失败不抛 —— pending_install.apply_pending 内部已打印错误，让 launcher 继续起。
    """
    try:
        from studio.services import pending_install  # noqa: PLC0415
        pending_install.apply_pending()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[studio] 警告：处理 pending 安装请求时异常（{exc}），跳过",
            file=sys.stderr,
        )


def _try_enable_flash_attn() -> None:
    """启动期检查 flash_attn 是否装好；装好就开 cosmos / anima 状态机。

    没装就 silently skip（_check_torch_cuda 不重复提示，flash_attn 是 nice-to-have）。
    动态 import 避免拖慢 cli import 时间（cosmos_predict2_modeling 加载触发 torch import）。
    """
    try:
        from studio.services import flash_attention_setup  # noqa: PLC0415
        if not flash_attention_setup.current_status()["installed"]:
            return
        from models.cosmos_predict2_modeling import set_flash_attn_enabled  # noqa: PLC0415
        if set_flash_attn_enabled(True):
            print("[studio] flash_attn 启用")
        else:
            # 装了 flash_attn 但 set_flash_attn_enabled 拒绝（_FLASH_ATTN_AVAILABLE=False）
            # 通常意味着 import 时挂了（CUDA 版本不匹配等）；不噪声只 stderr 警告
            print(
                "[studio] 警告：flash_attn 已安装但模型层 import 失败，"
                "继续走 SDPA fallback",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001
        # Studio 启动不能为这一项加速 fail；记 warn 但放行
        print(
            f"[studio] 警告：flash_attn 启用时异常（{exc}），跳过加速",
            file=sys.stderr,
        )


def _check_torch_cuda() -> None:
    """启动期检查 torch 是否能用 CUDA；CPU-only torch 跑训练 / 出图会极慢。

    四种状态：
    - CUDA 可用                       → 一行 OK
    - torch 是 CPU-only build + 有 GPU → 大警告 + 重装命令（最常见误装）
    - torch 是 CPU-only build + 无 GPU → 一行 info（用户确实在 CPU 机器上）
    - torch 是 CUDA build 但 cuda 不可用 → 警告（驱动 / WSL 问题）

    `torch.version.cuda` 在 CPU-only wheel 上是 None；在 cu* wheel 上是 "12.8" 等。
    用它区分误装与驱动问题。
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return  # _ensure_python_deps 会在更早路径处理

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = "?"
        print(f"[studio] torch {torch.__version__}（GPU: {name}）")
        return

    cuda_build = getattr(torch.version, "cuda", None)
    if cuda_build is None:
        # CPU-only wheel：进一步判断本机是否其实有 NVIDIA GPU（误装）
        try:
            from studio.services import onnxruntime_setup  # noqa: PLC0415
            has_gpu = bool(onnxruntime_setup.detect_cuda().get("available"))
        except Exception:  # noqa: BLE001
            has_gpu = False
        if has_gpu:
            print(
                f"[studio] 警告：检测到 NVIDIA GPU，但当前安装的是 CPU-only 版 PyTorch "
                f"({torch.__version__})。\n"
                f"        训练 / 出图将跑在 CPU 上，速度极慢（单步常需数十秒）。\n"
                f"        请卸载后重装 CUDA 版：\n"
                f"          pip uninstall torch torchvision -y\n"
                f"          # 按你的 CUDA 版本选；如 CUDA 12.8：\n"
                f"          pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128",
                file=sys.stderr,
            )
        else:
            print(
                f"[studio] torch {torch.__version__}（CPU-only build，未检测到 NVIDIA GPU）"
            )
        return

    # CUDA build 但运行时不可用：驱动 / WSL / 容器问题
    print(
        f"[studio] 警告：torch {torch.__version__}（CUDA {cuda_build} build），"
        f"但 torch.cuda.is_available()=False。\n"
        f"        可能原因：NVIDIA 驱动未安装 / 版本过低 / WSL 缺 CUDA 支持。",
        file=sys.stderr,
    )


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
    _apply_pending_install()
    _check_torch_cuda()
    _try_enable_flash_attn()
    _bootstrap_onnxruntime()
    url = f"http://{args.host}:{args.port}/studio/"
    print(f"[studio] 启动后端 → {url}")
    if not args.no_browser:
        _spawn_browser_opener(url)
    return subprocess.call(
        [find_python(), "-m", "studio.server", "--host", args.host, "--port", str(args.port)]
    )


def cmd_dev(args: argparse.Namespace) -> int:
    rc = _ensure_python_deps()
    if rc != 0:
        return rc
    npm = find_npm()
    if not npm:
        _print_npm_install_hint()
        return 2
    rc = npm_install_if_missing(npm)
    if rc != 0:
        return rc
    _apply_pending_install()
    _check_torch_cuda()
    _try_enable_flash_attn()
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
