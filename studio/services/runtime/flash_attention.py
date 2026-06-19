"""Flash Attention wheel 查找与安装。

wheel 命名规律（mjun0812/flash-attention-prebuild-wheels）：
  flash_attn-{fa_ver}+{cuda}{torch}-{pyver}-{pyver}-{platform}.whl
  例：flash_attn-2.8.3+cu130torch2.11-cp312-cp312-win_amd64.whl

匹配策略：
- platform：必须精确一致
- torch：必须精确一致（2.11 ≠ 2.10）
- CUDA：精确 > 同大版本（cu132 → 接受 cu130，CUDA 小版本向下兼容）
- Python：必须精确（cp312 wheel 无法在 cp313 上运行，ABI 不同）

公开 API（也是 server 端点 / CLI 用的入口）：
- `detect_env()` — 当前 Python / CUDA / PyTorch / 平台
- `current_status()` — flash_attn 是否已装 + 版本
- `find_candidates(env)` — GitHub releases 列表（带 score / usable / notes）
- `find_best_wheel(env)` — 最优可用 wheel URL
- `install(url=None)` — 同步 pip install；url=None → 自动选
"""
from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import re
import subprocess
import sys
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

FA_RELEASES_URL = (
    "https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases"
)


def detect_env() -> dict[str, Any]:
    """检测当前 Python / CUDA / PyTorch / 平台。

    各字段在不可获取时为 None。`platform` 仅返回 `linux_x86_64` / `win_amd64`，
    其它平台（macOS arm64 / linux aarch64）目前没 prebuilt wheel。
    """
    vi = sys.version_info
    python_tag = f"cp{vi.major}{vi.minor}"

    syst = platform.system().lower()
    mach = platform.machine().lower()
    if syst == "linux" and mach == "x86_64":
        plat = "linux_x86_64"
    elif syst == "windows" and mach in ("amd64", "x86_64"):
        plat = "win_amd64"
    else:
        plat = None

    # cuda_tag 必须匹配 **PyTorch 编译时**使用的 CUDA runtime —— flash_attn ABI 跟着
    # torch 走，不是跟着 driver。优先从 `torch.__version__` 的 `+cuXXX` 后缀拿。
    #
    # 历史 bug（PR-7）：原实现从 nvidia-smi 拿 cuda_tag。但 `nvidia-smi` 输出的
    # "CUDA Version: X.Y" 是**驱动支持的最高 CUDA 版本**，与 PyTorch 编译时锁定的 CUDA
    # 不是一回事。例：5090 driver 报 13.0，venv 装 torch 2.11.0+cu128 → flash_attn 必须
    # 装 cu128 wheel；原实现却把 cuda_tag 设成 cu130 → 选了 cu130 wheel → ABI 不匹配
    # → flash_attn import 失败。
    cuda_tag: Optional[str] = None
    cuda_ver: Optional[str] = None
    torch_tag: Optional[str] = None
    torch_ver: Optional[str] = None
    # 'cu128' / 'cu130' / 'cpu' / None(未装) —— 区分 CPU 误装很重要：flash_attn
    # 是 CUDA C extension，CPU 版 torch 上压根装不了，UI 要给「先重装 CUDA 版 torch」
    # 而不是「找不到 wheel」这种误导。
    torch_cuda_build: Optional[str] = None
    # 历史 except 只挡 ImportError，但 Windows 上 torch DLL 加载失败抛 OSError
    # （WinError 126），新 torch 版本里 __version__ 解析也可能出 IndexError —— 任何
    # 异常都不该让 status endpoint 500。捕获到走 None。
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
        torch_ver = torch.__version__
        parts = torch_ver.split("+")[0].split(".")
        if len(parts) >= 2:
            torch_tag = f"torch{parts[0]}.{parts[1]}"
        m = re.search(r"\+(cu\d+|cpu)", torch_ver)
        if m:
            tag = m.group(1)
            torch_cuda_build = tag
            if tag.startswith("cu"):
                cuda_tag = tag
                num = tag[2:]
                # cu128 → 12.8、cu130 → 13.0（最后一位 minor，其余 major）
                if len(num) >= 2:
                    cuda_ver = f"{num[:-1]}.{num[-1]}"
        else:
            # 没 +cu/+cpu 后缀的老 build，靠 torch.version.cuda 判
            cuda_v = getattr(getattr(torch, "version", None), "cuda", None)
            if cuda_v is None:
                torch_cuda_build = "cpu"
            else:
                clean = str(cuda_v).replace(".", "")
                torch_cuda_build = f"cu{clean}"
                cuda_tag = torch_cuda_build
                cuda_ver = str(cuda_v)
    except Exception:  # noqa: BLE001
        pass

    # nvidia-smi 仍跑一下：torch 没 +cu 后缀（CPU-only build）→ fallback 给 cuda_tag；
    # driver 版本始终单独存到 `driver_cuda_ver` 供 UI 显示与排错（让用户看到「驱动支持
    # cu130，PyTorch 是 cu128」这种场景，立刻明白为什么 wheel 应该选 cu128）。
    driver_cuda_ver: Optional[str] = None
    # 历史只挡 (subprocess.SubprocessError, OSError)；Windows 中文 locale (cp936)
    # 下 text=True 解码 nvidia-smi 输出可能抛 UnicodeDecodeError（非上述子类），
    # 直接 500。改用 errors='replace' 避免，并把 except 兜底放宽到 Exception。
    try:
        r = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
            errors="replace",
        )
        if r.returncode == 0:
            m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", r.stdout)
            if m:
                driver_cuda_ver = f"{m.group(1)}.{m.group(2)}"
                if cuda_tag is None:
                    cuda_tag = f"cu{m.group(1)}{m.group(2)}"
                    cuda_ver = driver_cuda_ver
    except Exception:  # noqa: BLE001
        pass

    return {
        "python_tag": python_tag,
        "cuda_tag": cuda_tag,
        "cuda_ver": cuda_ver,
        "driver_cuda_ver": driver_cuda_ver,
        "torch_tag": torch_tag,
        "torch_ver": torch_ver,
        "torch_cuda_build": torch_cuda_build,
        "platform": plat,
    }


def _parse_wheel(name: str) -> Optional[dict[str, str]]:
    """从 wheel 文件名解析出 version / cuda / torch / python / platform 标签。

    例：flash_attn-2.8.3+cu130torch2.11-cp312-cp312-win_amd64.whl
    → {version: "2.8.3", cuda: "cu130", torch: "torch2.11", python: "cp312",
       platform: "win_amd64"}
    """
    m = re.search(
        r"flash_attn-([^+]+)\+(cu\d+)(torch[\d.]+)-(cp\d+)-cp\d+-([\w]+)\.whl$", name
    )
    if not m:
        return None
    return {
        "version": m.group(1),
        "cuda": m.group(2),
        "torch": m.group(3),
        "python": m.group(4),
        "platform": m.group(5),
    }


def _cuda_major(tag: str) -> int:
    """cu130 → 13，cu124 → 12；解析失败返回 -1。"""
    m = re.search(r"cu(\d+)", tag)
    return int(m.group(1)) // 10 if m else -1


def _version_key(version: str) -> tuple[int, ...]:
    """flash_attn 版本字符串 → 可比较的整数元组，用于评分相同的候选间偏向更新版本。

    例：'2.8.3' → (2, 8, 3)；'2.7.4.post1' → (2, 7, 4, 1)；解析不出 → ()。
    """
    return tuple(int(x) for x in re.findall(r"\d+", version or ""))


def find_candidates(
    env: dict[str, Any],
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """查询 GitHub Releases，返回 (candidates, fetch_error)。

    candidates 每项：`{url, name, score, notes, usable, tags}`，按 score 降序。
    `usable=True` 表示当前环境可以直接安装；False 表示 ABI 不兼容（典型 Python 不一致）。
    `fetch_error` 非 None 表示 GitHub API 请求失败（网络 / 限流 / 解析）；
    UI 应显示为「无法拉候选列表，可手动粘 URL」。
    """
    plat = env.get("platform")
    torch_tag = env.get("torch_tag")
    cuda_tag = env.get("cuda_tag")
    python_tag = env.get("python_tag")

    if not plat:
        return [], None

    try:
        req = urllib.request.Request(
            FA_RELEASES_URL + "?per_page=100",
            headers={"User-Agent": "AnimaLoraStudio"},
        )
        data = json.loads(urllib.request.urlopen(req, timeout=15).read())
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)

    # GitHub 频率限制时返回 dict（{"message": "API rate limit exceeded..."}），不是 list
    if not isinstance(data, list):
        msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)
        return [], f"GitHub API 错误: {msg}"

    candidates: list[dict[str, Any]] = []
    for release in data:
        for asset in release.get("assets", []):
            tags = _parse_wheel(asset["name"])
            if not tags:
                continue
            if tags["platform"] != plat:
                continue
            if torch_tag and tags["torch"] != torch_tag:
                continue

            score = 0
            notes: list[str] = []
            usable = True

            # Python ABI：严格匹配，不同版本无法使用
            if python_tag:
                if tags["python"] == python_tag:
                    score += 20
                else:
                    usable = False
                    notes.append(
                        f"Python 不兼容（wheel={tags['python']}，当前={python_tag}）"
                    )

            # CUDA：同大版本可用，但不如精确匹配
            if cuda_tag:
                if tags["cuda"] == cuda_tag:
                    score += 20
                elif _cuda_major(tags["cuda"]) == _cuda_major(cuda_tag):
                    score += 10
                    notes.append(
                        f"CUDA 小版本不同（wheel={tags['cuda']}，当前={cuda_tag}，"
                        f"同大版本应兼容）"
                    )
                else:
                    score -= 5
                    notes.append(
                        f"CUDA 大版本不同（wheel={tags['cuda']}，当前={cuda_tag}）"
                    )

            candidates.append({
                "url": asset["browser_download_url"],
                "name": asset["name"],
                "version": tags["version"],
                "score": score,
                "notes": notes,
                "usable": usable,
                "tags": tags,
            })

    # 评分相同（同 python/cuda/torch）时偏向更新的 flash 版本。旧实现只按 score 排，平手
    # 退回 GitHub asset 顺序（恰好最旧在前）→ 自动安装首选最旧 wheel；而比该 wheel 更新的
    # GPU 架构上（如 Blackwell sm_120）旧 wheel 没有对应 kernel，flash 全程回退 SDPA。把
    # 版本作为次级 key（高于 asset 顺序、低于 score）即可让 find_best_wheel 选到最新可用版本。
    return sorted(
        candidates,
        key=lambda x: (x["score"], _version_key(x["version"])),
        reverse=True,
    ), None


def find_best_wheel(env: dict[str, Any]) -> Optional[str]:
    """返回最优可用 wheel URL；无候选 / 全部 unusable → None。"""
    candidates, _ = find_candidates(env)
    for c in candidates:
        if c["usable"]:
            return c["url"]
    return None


def current_status() -> dict[str, Any]:
    """当前 flash_attn 安装状态。"""
    try:
        version = importlib.metadata.version("flash_attn")
        return {"installed": True, "version": version}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def install(url: Optional[str] = None) -> dict[str, Any]:
    """安装 flash_attn wheel；url=None 则自动从 GitHub 找最优匹配。

    同步 pip install，可能需要几分钟（远端 wheel ~150MB）。flash_attn 是 C extension，
    pip 重装后必须重启进程才能切换；返回 `restart_required=True` 让 UI 提示。
    """
    env = detect_env()

    if url is None:
        # CPU 版 torch 装不了 flash_attn —— flash_attn 是 CUDA C extension，必须配
        # CUDA 版 torch。auto 路径先 pre-check 给清楚错误，否则 find_best_wheel 会因
        # cuda_tag 来自 nvidia-smi（cu130）而误报「未找到 wheel」，让用户误以为是
        # 网络/仓库问题。显式 URL 路径不挡，留给强制安装。
        if env.get("torch_cuda_build") == "cpu":
            raise RuntimeError(
                "PyTorch 是 CPU 版（torch+cpu），无法安装 flash_attn。"
                "flash_attn 是 CUDA C extension，必须先把 PyTorch 重装为 CUDA 版。\n"
                "请到「设置 → 训练 → PyTorch」一键重装为 cu128 / cu130 等 CUDA 版本，"
                "重启 Studio 后再回来装 flash_attn。"
            )
        if not env.get("platform"):
            raise RuntimeError("不支持的平台（仅 linux_x86_64 / win_amd64）")
        if not env.get("torch_tag"):
            raise RuntimeError("未检测到 PyTorch，无法自动匹配 wheel")
        url = find_best_wheel(env)
        if not url:
            raise RuntimeError(
                f"未找到可用 wheel（Python={env.get('python_tag')}，"
                f"CUDA={env.get('cuda_tag')}，Torch={env.get('torch_tag')}）。\n"
                "请在下方候选列表中手动选择，或前往 "
                "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases 粘贴 URL"
            )

    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", url],
        capture_output=True,
        text=True,
    )
    stdout = r.stdout + r.stderr
    tail = "\n".join(stdout.splitlines()[-40:])

    if r.returncode != 0:
        raise RuntimeError(f"pip install 失败:\n{tail}")

    try:
        importlib.invalidate_caches()
        version = importlib.metadata.version("flash_attn")
    except Exception:  # noqa: BLE001
        version = None

    return {
        "installed": True,
        "version": version,
        "url": url,
        "stdout_tail": tail,
        "restart_required": True,
    }
