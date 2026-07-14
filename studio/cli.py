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
import hashlib
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
    """Windows 优先 .cmd（CreateProcess 可直接跑），.ps1 兜底；Linux/Mac 走裸名。

    注意不要把裸 ``npm`` 放在 Windows 候选首位：Node.js 官方安装包在
    ``C:\\Program Files\\nodejs\\`` 同时铺了 ``npm`` (Git Bash 用的 bash 脚本)
    / ``npm.cmd`` / ``npm.ps1`` 三份，``shutil.which("npm")`` 在 Windows 上
    会先吃裸名那份，subprocess 直接报 WinError 193（不是有效 Win32 应用）。
    """
    candidates = ("npm.cmd", "npm.ps1", "npm") if os.name == "nt" else ("npm",)
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    return None


def find_python() -> str:
    """优先用当前解释器（venv 已激活则自然指对）。"""
    return sys.executable


_NPM_MIRROR = "https://mirrors.cloud.tencent.com/npm/"
_PIP_MIRROR = "https://mirrors.cloud.tencent.com/pypi/simple/"


def _say(msg: str, level: str = "info") -> None:
    """统一 CLI 用户输出入口（ADR-0009 PR-3 C4）。

    保 print 路径（ADR-0009 round 2 §1.3 决策 — CLI 5s 短命周期落盘价值低；
    用户终端看 `[studio] ...` 比 logger 默认 format 清爽；capsys 测试 UX 优先）。
    本 wrapper 给未来加 verbose 控制 / 着色留单一入口；现在等价于带前缀的 print。

    level:
      - "info" / "success" → stdout，`[studio] ` 前缀
      - "warning" / "error" → stderr，`[studio] ` 前缀
    """
    file = sys.stderr if level in ("warning", "error") else sys.stdout
    # 注意：不能用 f"[studio] {msg}"，否则被批量 _say 替换正则误伤。
    print("[studio] " + str(msg), file=file, flush=True)


def _npm_argv(npm: str, args: list[str]) -> list[str]:
    """拼出真正可被 subprocess 执行的 argv。

    ``.ps1`` 无法被 ``CreateProcess`` 直接拉起，必须包 ``powershell.exe -File``；
    ``.cmd`` 和裸名（Linux）直接拼即可。
    """
    if npm.lower().endswith(".ps1"):
        return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", npm, *args]
    return [npm, *args]


def _npm_call(npm: str, args: list[str], cwd: str, timeout: int = 180) -> int:
    """运行 npm 命令；超时则 kill 并返回 1。"""
    proc = subprocess.Popen(_npm_argv(npm, args), cwd=cwd)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return 1


def _frontend_package_files_changed_since_install() -> bool:
    marker = NODE_MODULES / ".package-lock.json"
    if not marker.exists():
        return False
    try:
        marker_mtime = marker.stat().st_mtime
        for f in (WEB_DIR / "package.json", WEB_DIR / "package-lock.json"):
            if f.exists() and f.stat().st_mtime > marker_mtime:
                return True
    except OSError:
        return False
    return False


def npm_install_if_missing(npm: str) -> int:
    _bin = "eslint.cmd" if os.name == "nt" else "eslint"
    deps_complete = NODE_MODULES.exists() and (NODE_MODULES / ".bin" / _bin).exists()
    package_files_changed = deps_complete and _frontend_package_files_changed_since_install()
    if deps_complete and not package_files_changed:
        return 0
    try:
        rel = NODE_MODULES.relative_to(REPO_ROOT)
    except ValueError:
        rel = NODE_MODULES
    if package_files_changed:
        _say("studio/web/package.json 或 package-lock.json 比 node_modules 新，运行 npm install...")
    else:
        _say(f"{rel} 不完整或不存在，运行 npm install（3 分钟超时）...")
    rc = _npm_call(npm, ["install"], str(WEB_DIR), timeout=180)
    if rc != 0:
        _say(f"npm install 失败或超时，切换国内源 ({_NPM_MIRROR}) 重试...")
        rc = subprocess.call(
            _npm_argv(npm, ["install", "--registry", _NPM_MIRROR]),
            cwd=str(WEB_DIR),
        )
    return rc


def _pip_install(args: list[str]) -> int:
    """运行 pip install；失败时切换阿里云镜像重试。"""
    rc = subprocess.call([find_python(), "-m", "pip", "install"] + args)
    if rc != 0:
        _say(f"pip install 失败，切换国内源 ({_PIP_MIRROR}) 重试...")
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
    _say("检测到 fastapi 缺失，重新安装 Python 依赖（requirements.txt）...")
    return _pip_install(["-r", str(req)])


def npm_build(npm: str) -> int:
    _say("构建前端 (npm run build)...")
    return subprocess.call(_npm_argv(npm, ["run", "build"]), cwd=str(WEB_DIR))


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
        _say(f"{label} pid={proc.pid}: {' '.join(cmd)}")
        self.procs.append((label, proc))
        return proc

    def wait_any(self) -> int:
        """阻塞到任一进程退出，返回该进程的 exit code。"""
        while True:
            for label, p in self.procs:
                rc = p.poll()
                if rc is not None:
                    _say(f"{label} 退出 (rc={rc})")
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
            _say(f"停止 {label}...")
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
                _say(f"{label} 超时未退出，强杀")
                p.kill()


# ---------------------------------------------------------------------------
# 命令实现
# ---------------------------------------------------------------------------


def _print_npm_install_hint() -> None:
    """`find_npm()` 返回 None 时打印平台相关安装提示。

    放 stderr，与 `[studio] 错误：找不到 npm` 同流；root 环境去掉 sudo（直接 root 跑装包）。
    """
    _say("错误：找不到 npm。请安装 Node.js 18+", "error")
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
        from studio.services.runtime import pending_install  # noqa: PLC0415
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
        from studio.services.runtime import flash_attention as flash_attention_setup  # noqa: PLC0415
        if not flash_attention_setup.current_status()["installed"]:
            return
        from modeling.cosmos_predict2_modeling import set_flash_attn_enabled  # noqa: PLC0415
        if set_flash_attn_enabled(True):
            _say("flash_attn 启用")
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
        _say(f"torch {torch.__version__}（GPU: {name}）")
        return

    cuda_build = getattr(torch.version, "cuda", None)
    if cuda_build is None:
        # CPU-only wheel：进一步判断本机是否其实有 NVIDIA GPU（误装）
        try:
            from studio.services.runtime import onnxruntime as onnxruntime_setup  # noqa: PLC0415
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


def _check_onnxruntime() -> None:
    """启动期 onnxruntime 状态检查（仅 detect，不装包）。

    对齐 xformers / flash-attention：未装时 silent skip（Tagging 页选 WD14 /
    CLTagger 会有徽章 + 引导按钮）。已装则打一行状态；CPU 包 + 有 GPU 走
    warn，提醒用户去 Settings 切 GPU 版。
    """
    try:
        from studio.services.runtime import onnxruntime as onnxruntime_setup

        rt = onnxruntime_setup.current_runtime()
        if rt["installed"] is None:
            return

        installed = rt.get("installed") or "?"
        ver = rt.get("version") or "?"
        if rt.get("cuda_available"):
            _say(f"onnxruntime: {installed}=={ver}（CUDA EP 可用）")
            return

        cuda = onnxruntime_setup.detect_cuda()
        if cuda.get("available"):
            _say(
                f"检测到 NVIDIA GPU 但 onnxruntime 只有 CPU EP（installed={installed}）。"
                f"WD14 / CLTagger 打标会跑 CPU（较慢）。可在 Settings → ONNX Runtime 重装为 GPU 版。",
                "warning",
            )
        else:
            _say(f"onnxruntime: {installed}=={ver}（CPU only，未检测到 NVIDIA GPU）")

    except Exception as exc:  # noqa: BLE001
        _say(f"onnxruntime 状态检查异常（已忽略）: {exc}", "error")


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


_RESTART_FLAG = REPO_ROOT / "tmp" / "restart"

# PR-D — installer 自检（ADR 0002）。cmd_run 入口快照这三个文件的 sha256；
# 每次 server 退出 + 收到 restart 请求时再算一次，任一变化 → 返回退出码 42
# 让 wrapper（studio.sh / studio.bat）整体 exec 自己。原因：
#
# - cli.py 本身变更 → 旧 python 进程加载的是旧 cli.py，next-iteration 的 inner
#   loop 仍走老逻辑；只有让 wrapper 重新拉 `python -m studio` 才能拿到新 cli.py。
# - studio.sh / studio.bat 变更 → bash 已把 loop 体加载进内存，cmd.exe 也可能
#   缓存 .bat 解析结果；必须让 shell 进程 exec 自己拿到新 wrapper。
#
# 三个文件中任一变化都走同一协议（最简单 / 最稳）。
_INSTALLER_FILES: tuple[Path, ...] = (
    REPO_ROOT / "studio" / "cli.py",
    REPO_ROOT / "studio.sh",
    REPO_ROOT / "studio.bat",
)
_INSTALLER_RELOAD_EXIT_CODE = 42


def _installer_hashes() -> dict[str, Optional[str]]:
    """快照 installer 文件 sha256。文件不存在 → 值为 None（跨平台：Linux 上
    studio.bat 不存在，Windows 上 studio.sh 不存在；存在性也算入比对）。"""
    result: dict[str, Optional[str]] = {}
    for p in _INSTALLER_FILES:
        try:
            result[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            result[p.name] = None
    return result


def _apply_update_pending() -> None:
    """启动期处理 webui 触发的 update 请求（ADR 0002 / PR-B）。

    server 端 POST /api/system/update 写 studio_data/.update_pending 后通过
    SIGINT 退出。我们在 cli.py 主循环里下一轮 bootstrap 之前调这个函数完成：

    1. git fetch + git reset --hard {target}
    2. requirements.txt 变了 → pip install -r（增量）
    3. studio/web/package.json 变了 → npm install
    4. 清 update cache 让下次 check_update 重 fetch

    失败不抛 —— `apply_pending` 内部把错误写到 studio_data/.update_log，
    让 cli.py 继续走后面的 bootstrap 把 server 至少起回来（用户能在 UI
    看到"上次 update 失败"提示）。
    """
    try:
        from studio.services.runtime import updater  # noqa: PLC0415
        if not updater.has_pending():
            return
        updater.apply_pending(emit=print)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[studio] 警告：apply update pending 时异常（{exc}），跳过",
            file=sys.stderr,
        )


def _maybe_force_torch(args: argparse.Namespace) -> int:
    """--torch <tag> 指定时，检查当前安装是否匹配；不匹配则立即重装（流式输出）。
    仅在 launcher 启动期调一次，重装完由 restart 机制加载新 torch。"""
    tag = getattr(args, 'torch', None)
    if not tag:
        return 0
    from studio.services.runtime import torch as torch_setup  # noqa: PLC0415
    current = torch_setup.detect_torch()
    current_build = current.get('cuda_build') or ('未安装' if not current.get('installed') else 'unknown')
    if current.get('installed') and current.get('cuda_build') == tag:
        _say(f"torch 已是 {tag}，跳过重装")
        return 0
    _say(f"--torch {tag} 指定（当前: {current_build}），开始重装...")
    _say("提示：按 Ctrl+C 可跳过")
    try:
        res = torch_setup.reinstall(tag, stream=True)
        _say(f"torch 重装完成: {res.get('version')} ({res.get('tag')})")
        return 0
    except KeyboardInterrupt:
        print("\n[studio] 用户中断，跳过 torch 重装", file=sys.stderr)
        return 0
    except RuntimeError as exc:
        _say(f"torch 重装失败: {exc}", "error")
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """`run` 主循环。

    内层 loop：每次 server 退出后检查 `tmp/restart` 标志（由 server 端
    `/api/system/restart` 写）。存在则删除标志 + 重走 bootstrap + 重起 server；
    不存在则跳出，正常退出。

    重启协议详见 `docs/adr/0002-webui-self-update.md`。外层 shell wrapper
    (`studio.sh` / `studio.bat`) 也有同样的 loop 兜底（cli.py 异常退出但
    flag 还在的场景），并且响应退出码 42 把自己 exec 一遍（PR-D installer
    自检：当 cli.py / studio.sh / studio.bat 本身被 update 修改后，需要从
    磁盘重新加载 wrapper + Python 解释器）。

    冷启动只打开一次浏览器；重启时复用已存在的 webui 标签页（前端轮询
    `/api/health` 自动 reconnect），不重复弹新窗口。
    """
    opened_browser = False
    # --torch 强制重装（仅首次，不在 restart 循环里重复）
    rc = _maybe_force_torch(args)
    if rc != 0:
        return rc
    # PR-D：快照启动期 installer 文件 sha256；server 退出后重算，变化则
    # 退出码 42 让 wrapper 整体 exec 自己。
    startup_installer = _installer_hashes()
    while True:
        rc = _ensure_python_deps()
        if rc != 0:
            return rc

        # 检测 update pending（ADR 0002 / PR-B）：上一轮 server 写了
        # studio_data/.update_pending 并请求重启，这里在重新 bootstrap 之前
        # 先 git pull + 必要时 pip install / npm install，确保后续的 stale
        # 检查 / native module 加载用的都是新版代码 / 新版依赖。
        _apply_update_pending()

        if not args.no_build:
            if not WEB_DIST.exists():
                _say("studio/web/dist 不存在，先构建前端...")
                rc = cmd_build(args)
                if rc != 0:
                    return rc
            elif _web_dist_is_stale():
                _say("studio/web/dist 比 src 旧（git pull 后未重建？），重新构建前端...")
                rc = cmd_build(args)
                if rc != 0:
                    return rc
        if not getattr(args, 'skip_pending', False):
            _apply_pending_install()
        _check_torch_cuda()
        _try_enable_flash_attn()
        _check_onnxruntime()
        url = f"http://{args.host}:{args.port}/"
        _say(f"启动后端 → {url}")
        if not args.no_browser and not opened_browser:
            _spawn_browser_opener(url)
            opened_browser = True
        try:
            rc = subprocess.call(
                [find_python(), "-m", "studio.server", "--host", args.host, "--port", str(args.port)]
            )
        except KeyboardInterrupt:
            # 终端 Ctrl+C：CTRL_C_EVENT 同时广播给 server 子进程（它自己走
            # graceful shutdown）；父进程这边阻塞在 wait 的 KeyboardInterrupt
            # 要等子进程退干净后才抛出来。用户主动停机，不打 traceback。
            _say("已停止（Ctrl+C）")
            return 130

        if not _RESTART_FLAG.exists():
            return rc

        # PR-D：installer 自检。restart flag 存在的前提下，若 cli.py /
        # studio.sh / studio.bat 任一变化，**保留** flag 并返回 42，让 wrapper
        # 走 exec self 路径。flag 保留是关键 —— wrapper 检测到 (exit==42 &&
        # flag exists) 才会 re-exec；只剩 flag 而 exit!=42 则走普通 restart。
        if _installer_hashes() != startup_installer:
            _say("检测到 launcher 文件更新（cli.py / studio.sh / studio.bat），"
                  "退出码 42 让 wrapper 重新加载...")
            return _INSTALLER_RELOAD_EXIT_CODE

        # 收到重启请求：删除标志 + loop 回去重新 bootstrap
        try:
            _RESTART_FLAG.unlink()
        except OSError:
            pass
        _say("收到重启请求，重新启动...")


def cmd_dev(args: argparse.Namespace) -> int:
    rc = _maybe_force_torch(args)
    if rc != 0:
        return rc
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
    if not getattr(args, 'skip_pending', False):
        _apply_pending_install()
    _check_torch_cuda()
    _try_enable_flash_attn()
    _check_onnxruntime()

    pg = ProcGroup()
    try:
        pg.spawn("frontend", _npm_argv(npm, ["run", "dev", "--", "--port", str(args.fe_port)]), cwd=WEB_DIR)
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
        frontend_url = f"http://127.0.0.1:{args.fe_port}/"
        print(
            f"[studio] frontend → {frontend_url}  "
            f"backend → http://{args.host}:{args.port}/"
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
    _say("pytest...")
    rc = subprocess.call([find_python(), "-m", "pytest", "tests/"], cwd=str(REPO_ROOT))
    if rc != 0:
        return rc
    npm = find_npm()
    if not npm:
        _say("跳过 vitest (未安装 npm)")
        return 0
    if not NODE_MODULES.exists():
        _say("跳过 vitest (node_modules 缺失，先 npm install)")
        return 0
    _say("vitest...")
    return subprocess.call(_npm_argv(npm, ["run", "test"]), cwd=str(WEB_DIR))


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
    p_run.add_argument("--skip-pending", action="store_true",
                       help="跳过 pending pip 安装（torch 重装等），直接启动")
    p_run.add_argument("--torch", metavar="TAG",
                       help="强制指定 torch CUDA 版本（cu128/cu126/cu124/cu118/cpu），"
                            "与当前不符时自动重装。CPU 租赁机预装 GPU torch 时使用。")
    p_run.set_defaults(func=cmd_run)

    p_dev = sub.add_parser("dev", help="前后端开发模式")
    p_dev.add_argument("--host", default="127.0.0.1")
    p_dev.add_argument("--port", type=int, default=8765,
                       help="后端 uvicorn 端口（默认 8765）")
    p_dev.add_argument("--fe-port", type=int, default=5173,
                       help="前端 Vite 开发服务器端口（默认 5173）")
    p_dev.add_argument("--no-browser", action="store_true",
                       help="启动后不自动打开浏览器")
    p_dev.add_argument("--skip-pending", action="store_true",
                       help="跳过 pending pip 安装（torch 重装等），直接启动")
    p_dev.add_argument("--torch", metavar="TAG",
                       help="强制指定 torch CUDA 版本（cu128/cu126/cu124/cu118/cpu）")
    p_dev.set_defaults(func=cmd_dev)

    p_build = sub.add_parser("build", help="仅构建前端")
    p_build.set_defaults(func=cmd_build)

    p_test = sub.add_parser("test", help="跑 pytest + vitest")
    p_test.set_defaults(func=cmd_test)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args_list = list(argv) if argv is not None else sys.argv[1:]
    # 没有子命令时默认 run（如 studio.sh --port 6006 → run --port 6006）。
    # 找第一个不以 '-' 开头的参数，判断是否是已知子命令；不是则插入 run。
    _subcmds = {'run', 'dev', 'build', 'test'}
    _first_pos = next((a for a in args_list if not a.startswith('-')), None)
    if _first_pos not in _subcmds:
        args_list = ['run'] + args_list
    args = parser.parse_args(args_list)
    # PR-1 C4: 统一日志体系 (ADR-0009)。file=False — CLI 是 5s 短命周期，
    # 启动信息不进 studio.log（用户决定 — round 2 §1.3）。console=True 让
    # logger.x 调用走人读 stderr；现有 48 处 print() 不动（PR-3 _say() wrapper
    # 收编）。env ANIMA_LOGGING_NO_BOOTSTRAP=1 时 noop（测试态）。
    from .infrastructure.logging import setup_logging
    setup_logging(f"cli:{args.cmd}", file=False, console=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
