@echo off
chcp 65001 >nul
REM Usage: train_rocm.bat path\to\training.yaml

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

if "%~1"=="" (
    echo Usage: train_rocm.bat path\to\training.yaml 1>&2
    exit /b 2
)
if not exist "%ROCM_PYTHON%" (
    echo [rocm] Python not found: %ROCM_PYTHON% 1>&2
    exit /b 1
)

set "TRAIN_CONFIG=%~1"
"%ROCM_PYTHON%" tools\rocm_check.py --config "%TRAIN_CONFIG%"
if errorlevel 1 exit /b %ERRORLEVEL%

"%ROCM_PYTHON%" runtime\anima_train.py --config "%TRAIN_CONFIG%"
exit /b %ERRORLEVEL%
