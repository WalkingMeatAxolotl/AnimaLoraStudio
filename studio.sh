#!/usr/bin/env bash
# AnimaStudio Linux/macOS shortcut -- forwards to: python -m studio
# Usage:
#   ./studio.sh [--mirror] [--reinstall] [subcommand]
#
#   --mirror     Use Tencent pip mirror during first-run setup.
#                Without this flag, official PyPI is tried first; the mirror is
#                used as a fallback if the official source fails.
#
#   --reinstall  DELETE venv/ and rebuild from scratch (studio_data/ kept).
#                Use when venv is broken beyond repair (dep conflict / corrupt
#                wheels / etc). Asks for confirmation.
#
#   subcommand: run (default) | dev | build | test
#
# Safe to run with either ./studio.sh or `bash studio.sh`.
# Avoid `source studio.sh` -- not needed (we call venv python directly).
#
# NOTE: shell echo messages are kept in plain ASCII/English so non-UTF-8
#       locales don't render them as garbled bytes. Python-side messages are
#       UTF-8 (PYTHONUTF8=1 / PYTHONIOENCODING=utf-8 below).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "studio.sh: cannot cd to $SCRIPT_DIR" >&2; exit 1; }

# Force Python UTF-8 output so cli.py messages with non-ASCII characters are
# not mangled on non-UTF-8 locales.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# Parse our flags; collect remaining args to forward to Python.
_USE_MIRROR=0
_REINSTALL=0
_PASSTHROUGH=()
for _arg in "$@"; do
    case "$_arg" in
        --mirror)    _USE_MIRROR=1 ;;
        --reinstall) _REINSTALL=1 ;;
        *)           _PASSTHROUGH+=("$_arg") ;;
    esac
done

_TENCENT="https://mirrors.cloud.tencent.com/pypi/simple/"
_REQ_MARKER="venv/.studio-requirements.sha256"

_pip_install() {
    # Usage: _pip_install [pip args...]
    # Tries official PyPI first; falls back to Tencent mirror on failure.
    # With --mirror: goes straight to Tencent mirror.
    if [ "$_USE_MIRROR" = "1" ]; then
        echo "[studio] setup: using Tencent mirror for pip"
        "$PYTHON" -m pip install "$@" -i "$_TENCENT"
    else
        "$PYTHON" -m pip install "$@" || {
            echo "[studio] setup: pip failed, retrying via Tencent mirror..."
            "$PYTHON" -m pip install "$@" -i "$_TENCENT"
        }
    fi
}

# --reinstall: nuke venv before detection. studio_data/ is untouched.
if [ "$_REINSTALL" = "1" ] && [ -d venv ]; then
    echo "[studio] --reinstall: venv/ will be DELETED and rebuilt."
    echo "[studio]   - studio_data/ (your projects + LoRA weights) is NOT touched"
    echo "[studio]   - any user-installed pip packages outside requirements.txt will be lost"
    printf "Continue? [y/N] "
    read -r _ans
    case "$_ans" in
        [yY]*) ;;
        *)     echo "[studio] --reinstall aborted"; exit 0 ;;
    esac
    echo "[studio] removing venv/..."
    rm -rf venv || { echo "studio.sh: failed to remove venv" >&2; exit 1; }
fi

_check_venv_python_version() {
    # Warn if existing venv's Python is < 3.10. User can fix with --reinstall.
    if ! "$PYTHON" -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >/dev/null 2>&1; then
        echo "[studio] WARNING: venv/ uses Python < 3.10; some deps may fail to install" >&2
        echo "[studio] consider ./studio.sh --reinstall to recreate with a newer Python" >&2
    fi
}

if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
    _check_venv_python_version
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    _check_venv_python_version
else
    # PR-S0: iterate explicit versions first so users with multiple Python
    # installs (Ubuntu pre-22.04 has python3=3.8 even when python3.10 is also
    # installed) get the latest >= 3.10. Fall back to python3 / python.
    BOOTSTRAP_PY=""
    for _candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
        if command -v "$_candidate" >/dev/null 2>&1; then
            if "$_candidate" -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >/dev/null 2>&1; then
                BOOTSTRAP_PY="$_candidate"
                break
            fi
        fi
    done
    if [ -z "$BOOTSTRAP_PY" ]; then
        echo "studio.sh: no Python 3.10+ found on PATH (need one of python3.10/3.11/3.12/3.13)" >&2
        exit 1
    fi
    echo "[studio] No venv found. Creating venv/ via $BOOTSTRAP_PY ..."
    "$BOOTSTRAP_PY" -m venv venv || { echo "studio.sh: failed to create venv" >&2; exit 1; }
    PYTHON="venv/bin/python"
    _pip_install --upgrade pip || { echo "studio.sh: failed to upgrade pip" >&2; exit 1; }

    # GPU-aware torch first install (PR-S1a). Without this, requirements.txt's
    # bare `torch>=2.0.0` makes pip pull the CPU wheel from PyPI default. By
    # installing torch from PyTorch's CUDA index FIRST, the requirements.txt
    # constraint is already satisfied and pip won't replace it.
    _TORCH_INDEX="$("$PYTHON" tools/select_torch_index.py 2>/dev/null || true)"
    if [ -n "$_TORCH_INDEX" ]; then
        echo "[studio] setup: NVIDIA GPU detected; installing torch from $_TORCH_INDEX"
        if ! "$PYTHON" -m pip install torch torchvision --index-url "$_TORCH_INDEX"; then
            echo "[studio] setup: CUDA torch install failed; will fall back to PyPI default in requirements.txt"
            echo "[studio] setup: you can fix manually later via Studio Settings > PyTorch > Reinstall"
        fi
    fi

    if [ -f requirements.txt ]; then
        echo "[studio] Installing Python dependencies..."
        _pip_install -r requirements.txt || { echo "studio.sh: pip install failed" >&2; exit 1; }
    else
        echo "studio.sh: requirements.txt not found, skipping dependency install" >&2
    fi
    # PR-S1b: write hash marker after fresh install so future stale check is correct
    "$PYTHON" tools/check_requirements_changed.py --marker "$_REQ_MARKER" --update-marker >/dev/null 2>&1 || true
fi

# PR-S1b: stale check. If requirements.txt content hash differs from the marker
# (or no marker yet on an old venv), `pip install -r requirements.txt` to add
# missing packages. NO --upgrade -- existing torch+cu128 etc stays untouched.
_STALE="$("$PYTHON" tools/check_requirements_changed.py --marker "$_REQ_MARKER" 2>/dev/null || echo missing)"
if [ "$_STALE" = "stale" ]; then
    echo "[studio] requirements.txt changed since last sync; installing new deps (no upgrade)..."
    if _pip_install -r requirements.txt; then
        "$PYTHON" tools/check_requirements_changed.py --marker "$_REQ_MARKER" --update-marker >/dev/null 2>&1 || true
        echo "[studio] dep sync complete"
    else
        echo "[studio] WARNING: dep sync failed; existing venv still works but may miss new deps" >&2
        echo "[studio] try ./studio.sh --reinstall if errors persist" >&2
    fi
fi

echo "studio.sh: using $PYTHON"

# Restart loop (PR-A): if cli.py exits but tmp/restart is still present, loop
# back and re-run. cli.py's own inner loop handles the common case (server
# requests restart from /api/system/restart); this outer loop is the safety net.
#
# Special exit code 42 (PR-D, installer self-update): cli.py detected that
# cli.py / studio.sh / studio.bat itself was just replaced by `git reset`, and
# kept tmp/restart so we'd see it. We `exec` ourselves so the new wrapper code
# is loaded from disk (bash has the old loop body in memory; the new wrapper
# might have different bootstrap / dep-install logic). See ADR 0002.
# We exec with _PASSTHROUGH only (not original "$@") so --reinstall does not
# get re-triggered.
while true; do
    "$PYTHON" -m studio "${_PASSTHROUGH[@]}"
    EXIT_CODE=$?
    if [ ! -f tmp/restart ]; then
        break
    fi
    if [ "$EXIT_CODE" -eq 42 ]; then
        echo "[studio] launcher updated, re-exec wrapper"
        rm -f tmp/restart
        exec "$0" "${_PASSTHROUGH[@]}"
    fi
    echo "[studio] restart requested (wrapper loop)"
    rm -f tmp/restart
done

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[studio] Exit code $EXIT_CODE, see error messages above."
fi
exit $EXIT_CODE
