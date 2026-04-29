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

if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "studio.sh: no python found (looked for venv/bin/python, .venv/bin/python, python3, python)" >&2
    exit 1
fi

echo "studio.sh: using $PYTHON"
"$PYTHON" -m studio "$@"
