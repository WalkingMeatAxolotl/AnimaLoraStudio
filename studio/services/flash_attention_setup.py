"""Flash Attention wheel 查找与安装。

wheel 命名规律（mjun0812/flash-attention-prebuild-wheels）：
  flash_attn-{fa_ver}+{cuda}{torch}-{pyver}-{pyver}-{platform}.whl
  例：flash_attn-2.8.3+cu130torch2.11-cp312-cp312-win_amd64.whl

匹配策略：
- platform：必须精确一致
- torch：必须精确一致（2.11 ≠ 2.10）
- CUDA：精确 > 同大版本（cu132 → 接受 cu130，CUDA 小版本向下兼容）
- Python：必须精确（cp312 wheel 无法在 cp313 上运行，ABI 不同）
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import urllib.request
from typing import Any, Optional


FA_RELEASES_URL = (
    "https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases"
)


def detect_env() -> dict[str, Any]:
    """检测当前 Python / CUDA / PyTorch / 平台。"""
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

    cuda_tag: Optional[str] = None
    cuda_ver: Optional[str] = None
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", r.stdout)
            if m:
                cuda_ver = f"{m.group(1)}.{m.group(2)}"
                cuda_tag = f"cu{m.group(1)}{m.group(2)}"
    except Exception:
        pass

    torch_tag: Optional[str] = None
    torch_ver: Optional[str] = None
    try:
        import torch  # type: ignore
        v = torch.__version__.split("+")[0].split(".")
        torch_tag = f"torch{v[0]}.{v[1]}"
        torch_ver = torch.__version__
    except ImportError:
        pass

    return {
        "python_tag": python_tag,
        "cuda_tag": cuda_tag,
        "cuda_ver": cuda_ver,
        "torch_tag": torch_tag,
        "torch_ver": torch_ver,
        "platform": plat,
    }


def _parse_wheel(name: str) -> Optional[dict[str, str]]:
    """从 wheel 文件名解析出 cuda / torch / python / platform 标签。"""
    m = re.search(
        r"\+(cu\d+)(torch[\d.]+)-(cp\d+)-cp\d+-([\w]+)\.whl$", name
    )
    if not m:
        return None
    return {
        "cuda": m.group(1),
        "torch": m.group(2),
        "python": m.group(3),
        "platform": m.group(4),
    }


def _cuda_major(tag: str) -> int:
    """cu130 → 13, cu124 → 12"""
    m = re.search(r"cu(\d+)", tag)
    return int(m.group(1)) // 10 if m else -1


def find_candidates(env: dict[str, Any]) -> list[dict[str, Any]]:
    """查询 GitHub Releases，返回按匹配度排序的候选 wheel 列表。

    每项包含：url / name / score / notes / usable
    usable=True 表示当前环境可以直接安装。
    """
    plat = env.get("platform")
    torch_tag = env.get("torch_tag")
    cuda_tag = env.get("cuda_tag")
    python_tag = env.get("python_tag")

    if not plat:
        return []

    try:
        req = urllib.request.Request(
            FA_RELEASES_URL, headers={"User-Agent": "AnimaLoraStudio"}
        )
        data = json.loads(urllib.request.urlopen(req, timeout=15).read())
    except Exception:
        return []

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
                        f"CUDA 小版本不同（wheel={tags['cuda']}，当前={cuda_tag}，同大版本应兼容）"
                    )
                else:
                    score -= 5
                    notes.append(
                        f"CUDA 大版本不同（wheel={tags['cuda']}，当前={cuda_tag}）"
                    )

            candidates.append({
                "url": asset["browser_download_url"],
                "name": asset["name"],
                "score": score,
                "notes": notes,
                "usable": usable,
                "tags": tags,
            })

    return sorted(candidates, key=lambda x: -x["score"])


def find_best_wheel(env: dict[str, Any]) -> Optional[str]:
    """返回最优可用 wheel URL，无则返回 None。"""
    for c in find_candidates(env):
        if c["usable"]:
            return c["url"]
    return None


def current_status() -> dict[str, Any]:
    """当前 flash_attn 安装状态（包名 / 版本）。"""
    try:
        version = importlib.metadata.version("flash_attn")
        return {"installed": True, "version": version}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def install(url: Optional[str] = None) -> dict[str, Any]:
    """安装 flash_attn wheel。url=None 则自动从 GitHub 找最优匹配。

    同步 pip install，可能需要几分钟；前端按钮必须带 loading 状态。
    flash_attn 是 C extension，pip 重装后必须重启进程才能切换。
    """
    env = detect_env()

    if url is None:
        if not env.get("platform"):
            raise RuntimeError("不支持的平台（仅 linux_x86_64 / win_amd64）")
        if not env.get("torch_tag"):
            raise RuntimeError("未检测到 PyTorch，无法自动匹配 wheel")
        url = find_best_wheel(env)
        if not url:
            raise RuntimeError(
                f"未找到可用 wheel（Python={env.get('python_tag')}，"
                f"CUDA={env.get('cuda_tag')}，Torch={env.get('torch_tag')}）\n"
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
    except Exception:
        version = None

    return {
        "installed": True,
        "version": version,
        "url": url,
        "stdout_tail": tail,
        "restart_required": True,
    }
