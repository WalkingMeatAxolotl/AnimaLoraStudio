@echo off
REM AnimaStudio 一键启动脚本
REM 用途：
REM   studio.bat            构建前端（如有变更）+ 启动后端
REM   studio.bat dev        前后端开发模式（前端 5173 + 后端 8765）
REM   studio.bat build      只构建前端

setlocal enabledelayedexpansion
cd /d "%~dp0"

set MODE=%1
if "%MODE%"=="" set MODE=run

REM 选择 Python 解释器：优先 venv\Scripts\python.exe
set PYTHON=python
if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
) else if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
)

if "%MODE%"=="dev" goto :dev
if "%MODE%"=="build" goto :build
if "%MODE%"=="run" goto :run

echo Unknown mode: %MODE%
echo Usage: studio.bat [run^|dev^|build]
exit /b 1

:build
echo [1/1] Building frontend...
pushd studio\web
if not exist node_modules (
    echo   ^> npm install
    call npm install
    if errorlevel 1 (popd & exit /b 1)
)
call npm run build
if errorlevel 1 (popd & exit /b 1)
popd
echo Frontend built. Output in studio\web\dist\
exit /b 0

:run
REM 如果 dist 不存在，先 build
if not exist studio\web\dist (
    echo studio\web\dist not found, building first...
    call :build
    if errorlevel 1 exit /b 1
)
echo Starting AnimaStudio at http://127.0.0.1:8765/studio/
%PYTHON% -m studio.server
exit /b 0

:dev
echo Starting frontend dev server (port 5173) in new window...
start "AnimaStudio frontend" cmd /k "cd studio\web && npm run dev"
echo Starting backend at http://127.0.0.1:8765 ...
%PYTHON% -m studio.server --reload
exit /b 0
