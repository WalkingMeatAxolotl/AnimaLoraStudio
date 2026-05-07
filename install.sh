#!/usr/bin/env bash
# AnimaLoraStudio -- 首次安装脚本（Linux / macOS）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 颜色 ──────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
ok()   { echo -e "  ${GREEN}[OK]${RESET} $*"; }
info() { echo -e "  ${CYAN}[*]${RESET}  $*"; }
warn() { echo -e "  ${YELLOW}[~]${RESET}  $*"; }
err()  { echo -e "  ${RED}[!]${RESET}  $*"; }

echo ""
echo -e "${BOLD}=============================================================${RESET}"
echo -e "${BOLD}  AnimaLoraStudio  首次安装脚本${RESET}"
echo -e "${BOLD}=============================================================${RESET}"
echo ""

# ─────────────────────────────────────────────────────────────
# 1. Node.js
# ─────────────────────────────────────────────────────────────
echo -e "${BOLD}[1/4] 检测 Node.js...${RESET}"

if command -v node >/dev/null 2>&1; then
    ok "Node.js $(node --version)"
else
    err "未检测到 Node.js"
    read -r -p "  是否通过 nvm 自动安装最新 LTS 版 Node.js? [Y/N]: " WANT_NODE
    if [[ "${WANT_NODE^^}" == "Y" ]]; then
        info "安装 nvm..."
        # 获取最新 nvm 版本
        NVM_VER=$(curl -fsSL https://api.github.com/repos/nvm-sh/nvm/releases/latest \
            2>/dev/null | grep '"tag_name"' | sed 's/.*"tag_name": *"\(.*\)".*/\1/' || echo "v0.39.7")
        info "安装 nvm ${NVM_VER}..."
        curl -fsSL "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VER}/install.sh" | bash

        # 加载 nvm 到当前 shell
        export NVM_DIR="$HOME/.nvm"
        # shellcheck disable=SC1090
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

        if command -v nvm >/dev/null 2>&1; then
            info "安装 Node.js LTS..."
            nvm install --lts
            nvm use --lts
            ok "Node.js $(node --version)"
        else
            err "nvm 安装后无法加载，请重新打开终端后再运行此脚本"
            exit 1
        fi
    else
        warn "跳过 Node.js 安装"
    fi
fi

# ─────────────────────────────────────────────────────────────
# 2. Python
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[2/4] 检测 Python 环境...${RESET}"

PYTHON=""
if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
    ok "使用项目 venv: venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    err "未检测到 Python，请先安装 Python 3.10+"
    exit 1
fi

ok "$($PYTHON --version)"
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
CP_TAG="cp${PY_MAJOR}${PY_MINOR}"
ok "Python tag: ${CP_TAG}"

# ─────────────────────────────────────────────────────────────
# 3. CUDA + PyTorch
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[3/4] 检测 CUDA / PyTorch...${RESET}"

HAS_CUDA=0
CUDA_MAJOR=0
CUDA_MINOR=0
CUDA_TAG=""

if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi 2>/dev/null \
        | grep -oP '(?<=CUDA Version: )\d+\.\d+' || true)

    if [ -z "$CUDA_VER" ]; then
        warn "NVIDIA GPU 已检测，但无法解析 CUDA 版本，假设 12.x"
        CUDA_MAJOR=12; CUDA_MINOR=0; CUDA_VER="12.0"
    else
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        ok "CUDA ${CUDA_VER}"
    fi
    HAS_CUDA=1
    CUDA_TAG="cu${CUDA_MAJOR}${CUDA_MINOR}"
else
    warn "未检测到 NVIDIA GPU / nvidia-smi，跳过 GPU 组件"
fi

HAS_TORCH=0
TORCH_TAG=""
if $PYTHON -c "import torch" >/dev/null 2>&1; then
    TORCH_FULL=$($PYTHON -c "import torch; print(torch.__version__)")
    TORCH_TAG=$($PYTHON -c "
import torch
v = torch.__version__.split('+')[0].split('.')
print(f'torch{v[0]}.{v[1]}')
")
    ok "PyTorch ${TORCH_FULL}  tag: ${TORCH_TAG}"
    HAS_TORCH=1
else
    warn "未检测到 PyTorch（Flash Attention 自动匹配将不可用）"
fi

# ─────────────────────────────────────────────────────────────
# 4a. onnxruntime-gpu
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[4/5] 安装可选 GPU 组件${RESET}"
echo ""
echo -e "  ${BOLD}── onnxruntime-gpu ──${RESET}"

if [ "$HAS_CUDA" -eq 0 ]; then
    warn "无 CUDA，跳过 onnxruntime-gpu"
elif [ "$CUDA_MAJOR" -ge 13 ]; then
    info "CUDA ${CUDA_VER} → nightly CUDA 13.x 源"
    $PYTHON -m pip install coloredlogs flatbuffers numpy packaging protobuf sympy
    $PYTHON -m pip install --pre \
        --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ \
        onnxruntime-gpu
elif [ "$CUDA_MAJOR" -eq 12 ]; then
    info "CUDA ${CUDA_VER} → PyPI 正式版（默认 CUDA 12.x）"
    $PYTHON -m pip install onnxruntime-gpu
elif [ "$CUDA_MAJOR" -eq 11 ]; then
    info "CUDA ${CUDA_VER} → Azure DevOps CUDA 11 专用源"
    $PYTHON -m pip install coloredlogs flatbuffers numpy packaging protobuf sympy
    $PYTHON -m pip install onnxruntime-gpu \
        --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
else
    warn "无法识别 CUDA ${CUDA_MAJOR}.x，跳过 onnxruntime-gpu"
fi

# ─────────────────────────────────────────────────────────────
# 4b. Flash Attention
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "  ${BOLD}── Flash Attention（可选）──${RESET}"
echo "  参考: https://github.com/mjun0812/flash-attention-prebuild-wheels/releases"
echo ""

if [ "$HAS_CUDA" -eq 0 ]; then
    warn "无 CUDA，跳过"
elif [ "$HAS_TORCH" -eq 0 ]; then
    warn "未检测到 PyTorch，跳过"
else
    # 检测平台（仅支持 x86_64 Linux）
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        PLATFORM="linux_x86_64"
    else
        warn "不支持的架构 ${ARCH}，Flash Attention 请手动安装"
        PLATFORM=""
    fi

    if [ -n "$PLATFORM" ]; then
        FA_PATTERN="${CUDA_TAG}${TORCH_TAG}-${CP_TAG}-${CP_TAG}-${PLATFORM}"
        echo "  当前环境: ${CP_TAG} / ${CUDA_TAG} / ${TORCH_TAG} / ${PLATFORM}"
        echo "  自动匹配 pattern: ${FA_PATTERN}"
        echo ""
        echo "  请粘贴对应的 .whl 下载链接："
        echo "    留空  = 自动从 GitHub Releases 匹配上述 pattern"
        echo "    skip  = 跳过不安装"
        echo ""
        read -r -p "  URL> " FA_URL

        if [[ "${FA_URL,,}" == "skip" ]]; then
            warn "跳过 Flash Attention"
        else
            if [ -z "$FA_URL" ]; then
                info "查询 GitHub Releases..."

                # 用 Python 查询 API，避免 jq 依赖
                FA_URL=$($PYTHON - "$FA_PATTERN" <<'PYEOF'
import urllib.request, json, sys
pat = sys.argv[1]
req = urllib.request.Request(
    'https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases',
    headers={'User-Agent': 'AnimaLoraStudio-installer'}
)
try:
    data = json.loads(urllib.request.urlopen(req).read())
    for r in data:
        for a in r['assets']:
            if pat in a['name']:
                print(a['browser_download_url'])
                sys.exit(0)
except Exception as e:
    print(f"# error: {e}", file=sys.stderr)
PYEOF
                ) || true

                if [ -z "$FA_URL" ] || [[ "$FA_URL" == "#"* ]]; then
                    err "未找到匹配 wheel（pattern: ${FA_PATTERN}）"
                    echo "  请手动访问 https://github.com/mjun0812/flash-attention-prebuild-wheels/releases"
                    FA_URL=""
                else
                    ok "匹配到: ${FA_URL}"
                fi
            fi

            if [ -n "$FA_URL" ]; then
                info "安装中..."
                if $PYTHON -m pip install "$FA_URL"; then
                    ok "Flash Attention 安装成功"
                else
                    err "安装失败，请检查 URL 或网络连接"
                fi
            fi
        fi
    fi
fi


# ─────────────────────────────────────────────────────────────
# 5. ModelScope 模型下载（国内推荐）
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "  ${BOLD}── ModelScope 模型下载（可选，国内加速）──${RESET}"
echo "  https://www.modelscope.cn/models/circlestone-labs/Anima"
echo "  https://www.modelscope.cn/models/fireicewolf/wd-vit-large-tagger-v3"
echo ""
read -r -p "  是否使用 ModelScope 下载模型? [Y/N]: " WANT_MS

if [[ "${WANT_MS^^}" != "Y" ]]; then
    warn "跳过 ModelScope 下载"
else
    info "安装 modelscope..."
    if ! $PYTHON -m pip install modelscope; then
        err "modelscope 安装失败，跳过模型下载"
    else
        # ── Anima 模型 ────────────────────────────────────────
        echo ""
        echo "  ── Anima 模型（circlestone-labs/Anima）"
        echo "    [1] 主模型 anima-preview3-base  (~4 GB)"
        echo "    [2] VAE qwen_image_vae          (~250 MB)"
        echo "    [3] 主模型 + VAE（两项都下载）"
        echo "    [4] 跳过"
        echo ""
        read -r -p "  选择 [1/2/3/4]: " ANIMA_OPT

        if [[ "$ANIMA_OPT" =~ ^[123]$ ]]; then
            # 下载 + 打平路径的 Python 内联脚本
            $PYTHON - "$ANIMA_OPT" <<'PYEOF'
import subprocess, sys, shutil, pathlib
PYTHON = sys.executable
ROOT   = pathlib.Path('models')
REPO   = 'circlestone-labs/Anima'

def dl_file(repo_path: str, local_dir: pathlib.Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [PYTHON, '-m', 'modelscope', 'download',
         '--model', REPO, repo_path, '--local_dir', str(local_dir)],
        check=False
    )
    if r.returncode != 0:
        print(f'  [!] 下载失败: {repo_path}')
        return
    # modelscope 保留 repo 内部目录结构，把目标文件打平到 local_dir
    fname = pathlib.Path(repo_path).name
    want  = local_dir / fname
    if not want.exists():
        matches = list(local_dir.rglob(fname))
        if matches:
            shutil.move(str(matches[0]), str(want))
            for d in sorted(local_dir.rglob('*'), reverse=True):
                if d.is_dir() and d != local_dir:
                    try: d.rmdir()
                    except: pass
    if want.exists():
        print(f'  [+] {fname} -> {want}')
    else:
        print(f'  [!] 未找到 {fname}，请检查网络或手动放置')

opt = sys.argv[1]
if opt in ('1', '3'):
    dl_file('split_files/diffusion_models/anima-preview3-base.safetensors',
            ROOT / 'diffusion_models')
if opt in ('2', '3'):
    dl_file('split_files/vae/qwen_image_vae.safetensors',
            ROOT / 'vae')
PYEOF
        else
            warn "跳过 Anima 下载"
        fi

        # ── WD 打标模型 ───────────────────────────────────────
        echo ""
        echo "  ── WD 打标模型（fireicewolf / ModelScope）"
        echo "    [1] wd-eva02-large-tagger-v3  （推荐）"
        echo "    [2] wd-vit-large-tagger-v3"
        echo "    [3] wd-vit-tagger-v3"
        echo "    [4] wd-v1-4-convnext-tagger-v2"
        echo "    [5] 跳过"
        echo ""
        read -r -p "  选择 [1-5]: " WD_OPT

        case "$WD_OPT" in
            1) WD_NAME="wd-eva02-large-tagger-v3" ;;
            2) WD_NAME="wd-vit-large-tagger-v3" ;;
            3) WD_NAME="wd-vit-tagger-v3" ;;
            4) WD_NAME="wd-v1-4-convnext-tagger-v2" ;;
            *) WD_NAME="" ;;
        esac

        if [ -n "$WD_NAME" ]; then
            WD_MS_ID="fireicewolf/${WD_NAME}"
            # 本地路径按 HF model_id 命名（SmilingWolf_xxx），Studio 才能自动识别
            WD_LOCAL="models/wd14/SmilingWolf_${WD_NAME}"
            info "下载 ${WD_MS_ID} → ${WD_LOCAL}"
            mkdir -p "$WD_LOCAL"
            if $PYTHON -m modelscope download --model "$WD_MS_ID" --local_dir "$WD_LOCAL"; then
                ok "WD 模型下载完成: ${WD_LOCAL}"
            else
                err "下载失败，请检查网络或手动下载"
                echo "  https://www.modelscope.cn/models/${WD_MS_ID}"
            fi
        else
            warn "跳过 WD 打标模型下载"
        fi
    fi
fi

echo ""
echo -e "${BOLD}=============================================================${RESET}"
echo -e "${BOLD}  安装完成！运行 ./studio.sh 启动 AnimaLoraStudio${RESET}"
echo -e "${BOLD}=============================================================${RESET}"
echo ""
