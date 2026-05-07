#!/usr/bin/env bash
# =============================================================================
# start_vram_guard.sh — 启动 gpu_vram_guard.py 的包装脚本
#
# 用法：
#   chmod +x block.sh
#   ./block.sh
#
# 脚本会自动使用当前目录下的虚拟环境（./venv 或 ./.venv）。
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 【参数区】在这里修改所有配置，无需动其他地方
# -----------------------------------------------------------------------------

GPU=0                   # CUDA GPU 编号（多卡机器上改为 1、2…）
KEEP_FREE=2048          # 目标保留空闲显存（MiB）；该值以下才会停止抢占
HARD_MIN_FREE=512       # 强制释放触发线（MiB）；空闲低于此值立即让出一块
                        # 留空（""）则自动取 KEEP_FREE / 2
MAX_RESERVE=0           # 本进程最多占用的显存（MiB）；0 表示不限
BLOCK=256               # 稳态分配/释放的粒度（MiB）
STARTUP_BLOCK=2048      # 启动阶段快速抢占的粒度（MiB）
RELEASE_POLICY=hard-only  # 释放策略：hard-only（只在触发硬性下限时释放）
                          #           target（空闲低于 keep-free 时也释放）
INTERVAL=1.0            # 轮询间隔（秒）
NO_TOUCH=false          # true → 跳过写入（启动更快，但显存占用统计可能不准）
QUIET=false             # true → 只打印重要事件

# 目标脚本路径（默认与本 sh 脚本同目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/gpu_vram_guard.py"

# -----------------------------------------------------------------------------
# 虚拟环境自动检测（优先 ./venv，其次 ./.venv）
# -----------------------------------------------------------------------------

VENV_DIR=""
for candidate in "${SCRIPT_DIR}/venv" "${SCRIPT_DIR}/.venv"; do
    if [[ -f "${candidate}/bin/activate" ]]; then
        VENV_DIR="${candidate}"
        break
    fi
done

if [[ -z "${VENV_DIR}" ]]; then
    echo "[ERROR] 在 ${SCRIPT_DIR} 下未找到虚拟环境（venv 或 .venv）。" >&2
    echo "        请先创建：python -m venv venv && venv/bin/pip install torch" >&2
    exit 1
fi

echo "[INFO] 使用虚拟环境：${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# -----------------------------------------------------------------------------
# 检查目标脚本
# -----------------------------------------------------------------------------

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "[ERROR] 找不到 ${PYTHON_SCRIPT}" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# 组装命令行参数
# -----------------------------------------------------------------------------

ARGS=(
    --gpu           "${GPU}"
    --keep-free     "${KEEP_FREE}"
    --block         "${BLOCK}"
    --startup-block "${STARTUP_BLOCK}"
    --release-policy "${RELEASE_POLICY}"
    --interval      "${INTERVAL}"
    --max-reserve   "${MAX_RESERVE}"
)

if [[ -n "${HARD_MIN_FREE}" ]]; then
    ARGS+=(--hard-min-free "${HARD_MIN_FREE}")
fi

if [[ "${NO_TOUCH}" == "true" ]]; then
    ARGS+=(--no-touch)
fi

if [[ "${QUIET}" == "true" ]]; then
    ARGS+=(--quiet)
fi

# -----------------------------------------------------------------------------
# 启动
# -----------------------------------------------------------------------------

echo "[INFO] 启动命令：python ${PYTHON_SCRIPT} ${ARGS[*]}"
echo "       按 Ctrl+C 或发送 SIGTERM 优雅退出并释放显存。"
echo "---------------------------------------------------------------"

exec python "${PYTHON_SCRIPT}" "${ARGS[@]}"
