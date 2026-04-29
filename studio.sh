#!/usr/bin/env bash
# AnimaStudio Linux/macOS shortcut -- forwards to: python -m studio
# Usage:
#   ./studio.sh            same as: python -m studio run
#   ./studio.sh dev        frontend + backend dev mode
#   ./studio.sh build      build frontend only
#   ./studio.sh test       run pytest + vitest

set -e
cd "$(dirname "$0")"

if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

exec "$PYTHON" -m studio "$@"
