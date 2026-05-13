@echo off
chcp 65001 >nul
REM AnimaStudio Windows shortcut -- forwards to: python -m studio
REM Usage:
REM   .\studio.bat                same as: python -m studio run
REM   .\studio.bat --reinstall    DELETE venv\ and rebuild (studio_data\ kept)
REM   .\studio.bat dev            frontend + backend dev mode
REM   .\studio.bat build          build frontend only
REM   .\studio.bat test           run pytest + vitest
REM
REM Note: PowerShell needs the `.\` prefix; cmd.exe accepts either.
REM
REM NOTE: This file MUST stay pure ASCII. cmd.exe parses .bat files with the
REM system ANSI codepage BEFORE `chcp 65001` takes effect, so any non-ASCII
REM byte breaks line parsing on Japanese (cp932), Chinese (cp936), etc. hosts.

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Force Python to UTF-8 stdout/stderr so prints with non-ASCII (Chinese)
REM don't crash on non-UTF-8 system locales (e.g. cp932 Japanese).
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM Parse our flags; collect remaining args to forward to Python.
set REINSTALL=0
set PASSTHROUGH=
:argloop
if "%~1"=="" goto argdone
if /i "%~1"=="--reinstall" (
    set REINSTALL=1
) else (
    set PASSTHROUGH=!PASSTHROUGH! %1
)
shift
goto argloop
:argdone

set REQ_MARKER=venv\.studio-requirements.sha256

REM --reinstall: nuke venv before detection. studio_data\ is untouched.
if "%REINSTALL%"=="1" (
    if exist venv (
        echo [studio] --reinstall: venv\ will be DELETED and rebuilt.
        echo [studio]   - studio_data\ ^(your projects + LoRA weights^) is NOT touched
        echo [studio]   - any user-installed pip packages outside requirements.txt will be lost
        set /p ANS="Continue? [y/N] "
        if /i not "!ANS!"=="y" (
            echo [studio] --reinstall aborted
            exit /b 0
        )
        echo [studio] removing venv\...
        rmdir /s /q venv || (echo studio.bat: failed to remove venv 1>&2 & goto :fail)
    )
)

if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
    REM PR-S0: warn if existing venv is < 3.10 ^(user can `--reinstall` to fix^)
    venv\Scripts\python.exe -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo [studio] WARNING: venv\ uses Python ^< 3.10; some deps may fail to install. 1>&2
        echo [studio] consider .\studio.bat --reinstall to recreate with a newer Python. 1>&2
    )
) else if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
    .venv\Scripts\python.exe -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo [studio] WARNING: .venv\ uses Python ^< 3.10; some deps may fail to install. 1>&2
        echo [studio] consider .\studio.bat --reinstall to recreate with a newer Python. 1>&2
    )
) else (
    REM PR-S0: prefer `py -3` over `python`. Many Windows users keep an old
    REM system `python` on PATH (3.9 etc) for legacy compat and a newer Python
    REM via the `py` launcher (3.13 default). `py -3` picks the highest 3.x
    REM installed via python.org installers. Fall back to `python` for users
    REM without the launcher (uv / conda / scoop installs).
    set BOOTSTRAP_PY=
    where py >nul 2>nul
    if not errorlevel 1 set BOOTSTRAP_PY=py -3
    if not defined BOOTSTRAP_PY (
        where python >nul 2>nul
        if errorlevel 1 (
            echo studio.bat: neither `py` nor `python` found on PATH. Install Python 3.10+ from https://www.python.org/. 1>&2
            goto :fail
        )
        set BOOTSTRAP_PY=python
    )
    REM Version check: warn if < 3.10 ^(README requires 3.10+; some deps will fail to install on older^)
    !BOOTSTRAP_PY! -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo [studio] WARNING: bootstrap Python is older than 3.10; some deps may fail to install. 1>&2
        echo [studio] If you have a newer Python installed, set PY_PYTHON=3.13 ^(or your version^) and retry. 1>&2
    )
    echo [studio] No venv detected. Creating venv\ via `!BOOTSTRAP_PY!` -- first run may take a few minutes...
    !BOOTSTRAP_PY! -m venv venv || (echo studio.bat: failed to create venv 1>&2 & goto :fail)
    set PYTHON=venv\Scripts\python.exe
    !PYTHON! -m pip install --upgrade pip -i https://mirrors.cloud.tencent.com/pypi/simple/ || (echo studio.bat: failed to upgrade pip 1>&2 & goto :fail)

    REM GPU-aware torch first install (PR-S1a). Without this, requirements.txt's
    REM bare `torch>=2.0.0` makes pip pull the CPU wheel. Installing CUDA torch
    REM from PyTorch's index FIRST satisfies the constraint, pip won't replace.
    set TORCH_INDEX=
    for /f "delims=" %%i in ('!PYTHON! tools\select_torch_index.py 2^>nul') do set TORCH_INDEX=%%i
    if defined TORCH_INDEX (
        echo [studio] setup: NVIDIA GPU detected; installing torch from !TORCH_INDEX!
        !PYTHON! -m pip install torch torchvision --index-url !TORCH_INDEX!
        if errorlevel 1 (
            echo [studio] setup: CUDA torch install failed; will fall back to PyPI default in requirements.txt
            echo [studio] setup: you can fix manually later via Studio Settings ^> PyTorch ^> Reinstall
        )
    )

    if exist requirements.txt (
        echo [studio] Installing Python dependencies -- will retry via Tencent mirror if slow...
        !PYTHON! -m pip install -r requirements.txt
        if errorlevel 1 (
            echo [studio] pip install failed, retrying via Tencent mirror...
            !PYTHON! -m pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple/ || (echo studio.bat: pip install failed 1>&2 & goto :fail)
        )
    ) else (
        echo studio.bat: requirements.txt not found, skipping dependency install 1>&2
    )
    REM PR-S1b: write hash marker after fresh install
    !PYTHON! tools\check_requirements_changed.py --marker %REQ_MARKER% --update-marker >nul 2>nul
)

REM PR-S1b: stale check. If requirements.txt content hash differs from marker,
REM run `pip install -r requirements.txt` (no --upgrade) to add missing deps only.
set STALE=
for /f "delims=" %%i in ('!PYTHON! tools\check_requirements_changed.py --marker %REQ_MARKER% 2^>nul') do set STALE=%%i
if "!STALE!"=="stale" (
    echo [studio] requirements.txt changed since last sync; installing new deps ^(no upgrade^)...
    !PYTHON! -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [studio] WARNING: dep sync failed; existing venv still works but may miss new deps 1>&2
        echo [studio] try studio.bat --reinstall if errors persist 1>&2
    ) else (
        !PYTHON! tools\check_requirements_changed.py --marker %REQ_MARKER% --update-marker >nul 2>nul
        echo [studio] dep sync complete
    )
)

REM Restart loop (PR-A): if cli.py exits but tmp\restart still present, loop
REM back. cli.py's own inner loop handles the common case (server requests
REM restart); this outer loop is the safety net.
REM
REM Special exit code 42 (PR-D, installer self-update): cli.py detected that
REM cli.py / studio.sh / studio.bat itself was just replaced. cmd.exe has the
REM .bat parsed in memory; if we keep looping, we risk running stale wrapper
REM logic. So we `start` a fresh copy of ourselves and exit. Unlike POSIX
REM `exec`, this spawns a new process (different PID) -- /b keeps the same
REM console window, so the user-visible effect is the same. See ADR 0002.
:run_loop
!PYTHON! -m studio %PASSTHROUGH%
set STUDIO_ERR=%ERRORLEVEL%
if exist tmp\restart (
    if !STUDIO_ERR! EQU 42 (
        echo [studio] launcher updated, re-exec wrapper
        del /q tmp\restart 2>nul
        start "" /b "%~f0" %PASSTHROUGH%
        exit /b 0
    )
    echo [studio] restart requested ^(wrapper loop^)
    del /q tmp\restart 2>nul
    goto run_loop
)

if %STUDIO_ERR% NEQ 0 (
    echo.
    echo [studio] Exit code %STUDIO_ERR%. Press any key to close...
    pause >nul
)
exit /b %STUDIO_ERR%

:fail
echo.
echo [studio] setup failed. Press any key to close...
pause >nul
exit /b 1
