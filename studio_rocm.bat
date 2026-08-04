@echo off
chcp 65001 >nul
REM Windows ROCm launcher. Keep this file ASCII for cmd.exe compatibility.
REM Override the default runtime with: set ANIMA_ROCM_PYTHON=X:\path\python.exe

setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not defined ANIMA_ROCM_CACHE_DIR set "ANIMA_ROCM_CACHE_DIR=%~dp0.cache\miopen"
if not defined MIOPEN_USER_DB_PATH set "MIOPEN_USER_DB_PATH=%ANIMA_ROCM_CACHE_DIR%\db"
if not defined MIOPEN_CUSTOM_CACHE_DIR set "MIOPEN_CUSTOM_CACHE_DIR=%ANIMA_ROCM_CACHE_DIR%\kernels"
if not exist "%MIOPEN_USER_DB_PATH%" mkdir "%MIOPEN_USER_DB_PATH%"
if not exist "%MIOPEN_CUSTOM_CACHE_DIR%" mkdir "%MIOPEN_CUSTOM_CACHE_DIR%"

set "ROCM_PYTHON=E:\aiwork\python_embeded\python.exe"
if defined ANIMA_ROCM_PYTHON set "ROCM_PYTHON=%ANIMA_ROCM_PYTHON%"

if not exist "%ROCM_PYTHON%" (
    echo [rocm] Python not found: %ROCM_PYTHON% 1>&2
    echo [rocm] Set ANIMA_ROCM_PYTHON to your ROCm Python executable. 1>&2
    exit /b 1
)

if /i "%~1"=="--check" (
    shift
    "%ROCM_PYTHON%" tools\rocm_check.py %*
    exit /b %ERRORLEVEL%
)

"%ROCM_PYTHON%" tools\rocm_check.py --studio
if errorlevel 1 (
    echo [rocm] Install missing app dependencies without replacing torch: 1>&2
    echo [rocm]   "%ROCM_PYTHON%" -m pip install -r requirements-rocm.txt 1>&2
    exit /b 1
)

"%ROCM_PYTHON%" -m studio %*
exit /b %ERRORLEVEL%
