@echo off
REM AnimaStudio Windows 快捷启动 —— 转发到 python -m studio。
REM 用法:
REM   studio.bat            等同 python -m studio run
REM   studio.bat dev        前后端开发模式
REM   studio.bat build      仅构建前端
REM   studio.bat test       跑 pytest + vitest

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
