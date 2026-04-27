@echo off
REM AnimaStudio Windows shortcut -- forwards to: python -m studio
REM Usage:
REM   studio.bat            same as: python -m studio run
REM   studio.bat dev        frontend + backend dev mode
REM   studio.bat build      build frontend only
REM   studio.bat test       run pytest + vitest

setlocal
cd /d "%~dp0"

set PYTHON=python
if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
) else if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
)

%PYTHON% -m studio %*
exit /b %ERRORLEVEL%
