#!/usr/bin/env bash
# AnimaStudio Linux/macOS shortcut -- forwards to: python -m studio
# Usage:
#   ./studio.sh            same as: python -m studio run
#   ./studio.sh dev        frontend + backend dev mode
#   ./studio.sh build      build frontend only
#   ./studio.sh test       run pytest + vitest
#
# Safe to run with either ./studio.sh or `bash studio.sh`.
# Avoid `source studio.sh` -- not needed (we call venv python directly).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "studio.sh: cannot cd to $SCRIPT_DIR" >&2; exit 1; }

if [ -x "/opt/venv/bin/python" ]; then
    PYTHON="/opt/venv/bin/python"
elif [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    if command -v python3 >/dev/null 2>&1; then
        BOOTSTRAP_PY="python3"
    elif command -v python >/dev/null 2>&1; then
        BOOTSTRAP_PY="python"
    else
        echo "studio.sh: no python found (need python3 or python on PATH)" >&2
        exit 1
    fi
    echo "[studio] 未发现 venv，正在创建 venv/ 并安装依赖（首次运行，可能需要几分钟）..."
    "$BOOTSTRAP_PY" -m venv venv || { echo "studio.sh: 创建 venv 失败" >&2; exit 1; }
    PYTHON="venv/bin/python"
    "$PYTHON" -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ \
        || { echo "studio.sh: 升级 pip 失败" >&2; exit 1; }
    if [ -f requirements.txt ]; then
        echo "[studio] 安装 Python 依赖（如网络慢会自动切换国内源，可能需要几分钟）..."
        if ! "$PYTHON" -m pip install -r requirements.txt; then
            echo "[studio] pip install 失败，切换阿里云镜像重试..."
            "$PYTHON" -m pip install -r requirements.txt \
                -i https://mirrors.aliyun.com/pypi/simple/ \
                || { echo "studio.sh: pip install 失败" >&2; exit 1; }
        fi
    else
        echo "studio.sh: 找不到 requirements.txt，跳过依赖安装" >&2
    fi
fi

echo "studio.sh: using $PYTHON"
if [ -f "/.dockerenv" ] && [ $# -eq 0 ]; then
    # 容器环境（Docker / CNB）：监听全部接口，无需自动开浏览器
    exec "$PYTHON" -m studio run --host 0.0.0.0 --no-browser
else
    exec "$PYTHON" -m studio "$@"
fi
