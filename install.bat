@echo off
:: AnimaLoraStudio -- First-run installer
:: Save this file as UTF-8 (BOM recommended) for Chinese output to work.
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo =============================================================
echo   AnimaLoraStudio  首次安装脚本
echo =============================================================
echo.

:: ─────────────────────────────────────────────────────────────
:: 1. Node.js
:: ─────────────────────────────────────────────────────────────
echo [1/4] 检测 Node.js...
where node >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%v in ('node --version') do echo   [OK] Node.js %%v
    goto :step2
)

echo   [!] 未检测到 Node.js
set /p "WANT_NODE=  是否自动下载安装最新版 Node.js? [Y/N]: "
if /i "!WANT_NODE!" neq "Y" (
    echo   [~] 跳过 Node.js 安装
    goto :step2
)

echo   [*] 获取最新版本号...
for /f "tokens=*" %%v in ('powershell -NoProfile -Command ^
    "(Invoke-RestMethod https://nodejs.org/dist/index.json)[0].version"') do set NODE_VER=%%v

if "!NODE_VER!"=="" (
    echo   [!] 获取版本失败，请手动安装: https://nodejs.org/
    goto :step2
)

set NODE_MSI=node-!NODE_VER!-x64.msi
set NODE_URL=https://nodejs.org/dist/!NODE_VER!/!NODE_MSI!
echo   [*] 下载 !NODE_URL!
powershell -NoProfile -Command "Invoke-WebRequest '!NODE_URL!' -OutFile '!NODE_MSI!'"

if not exist "!NODE_MSI!" (
    echo   [!] 下载失败，请手动安装: https://nodejs.org/
    goto :step2
)

echo   [*] 静默安装中，请稍候...
msiexec /i "!NODE_MSI!" /quiet /norestart
del "!NODE_MSI!" 2>nul
echo   [+] Node.js !NODE_VER! 安装完成
echo   [!] 请关闭此窗口，重新打开后再运行此脚本（PATH 需刷新）
pause
exit /b 0

:: ─────────────────────────────────────────────────────────────
:: 2. Python
:: ─────────────────────────────────────────────────────────────
:step2
echo.
echo [2/4] 检测 Python 环境...
set PYTHON=

if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
    echo   [OK] 使用项目 venv
) else (
    where python >nul 2>&1
    if %errorlevel% neq 0 (
        echo   [!] 未检测到 Python，请先安装 Python 3.10+
        echo       https://www.python.org/downloads/
        pause & exit /b 1
    )
    set PYTHON=python
)

for /f "tokens=*" %%v in ('!PYTHON! --version 2^>^&1') do echo   [OK] %%v
for /f "tokens=*" %%v in ('!PYTHON! -c "import sys;print(sys.version_info.major)"') do set PY_MAJOR=%%v
for /f "tokens=*" %%v in ('!PYTHON! -c "import sys;print(sys.version_info.minor)"') do set PY_MINOR=%%v
set CP_TAG=cp!PY_MAJOR!!PY_MINOR!
echo   [OK] Python tag: !CP_TAG!

:: ─────────────────────────────────────────────────────────────
:: 3. CUDA + PyTorch
:: ─────────────────────────────────────────────────────────────
echo.
echo [3/4] 检测 CUDA / PyTorch...
set HAS_CUDA=0
set CUDA_MAJOR=0
set CUDA_MINOR=0
set CUDA_TAG=
set HAS_TORCH=0
set TORCH_TAG=

nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    :: 用 PowerShell 正则从 nvidia-smi 输出中提取 CUDA 版本
    for /f "tokens=*" %%v in ('powershell -NoProfile -Command ^
        "((nvidia-smi) -join [char]10) -match \"CUDA Version: (?<v>[0-9]+\.[0-9]+)\" | Out-Null; $Matches['v']"') do set CUDA_VER=%%v

    if "!CUDA_VER!"=="" (
        echo   [!] NVIDIA GPU 已检测，但无法解析 CUDA 版本，假设 12.x
        set CUDA_MAJOR=12 & set CUDA_MINOR=0 & set CUDA_VER=12.0
    ) else (
        for /f "tokens=1 delims=." %%a in ("!CUDA_VER!") do set CUDA_MAJOR=%%a
        for /f "tokens=2 delims=." %%b in ("!CUDA_VER!") do set CUDA_MINOR=%%b
        echo   [OK] CUDA !CUDA_VER!
    )
    set HAS_CUDA=1
    set CUDA_TAG=cu!CUDA_MAJOR!!CUDA_MINOR!
) else (
    echo   [~] 未检测到 NVIDIA GPU / nvidia-smi，跳过 GPU 组件
)

!PYTHON! -c "import torch" >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%v in ('!PYTHON! -c ^
        "import torch;v=torch.__version__.split('+')[0].split('.');print(f'torch{v[0]}.{v[1]}')\"') do set TORCH_TAG=%%v
    for /f "tokens=*" %%v in ('!PYTHON! -c "import torch;print(torch.__version__)"') do set TORCH_FULL=%%v
    echo   [OK] PyTorch !TORCH_FULL!  tag: !TORCH_TAG!
    set HAS_TORCH=1
) else (
    echo   [~] 未检测到 PyTorch（Flash Attention 自动匹配将不可用）
)

:: ─────────────────────────────────────────────────────────────
:: 4a. onnxruntime-gpu
:: ─────────────────────────────────────────────────────────────
echo.
echo [4/5] 安装可选 GPU 组件
echo.
echo  ── onnxruntime-gpu ──────────────────────────────────────
if "!HAS_CUDA!"=="0" (
    echo   [~] 无 CUDA，跳过 onnxruntime-gpu
    goto :flash_attn
)

if !CUDA_MAJOR! GEQ 13 (
    echo   [*] CUDA !CUDA_VER! → nightly CUDA 13.x 源
    !PYTHON! -m pip install coloredlogs flatbuffers numpy packaging protobuf sympy
    !PYTHON! -m pip install --pre --index-url ^
        https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ ^
        onnxruntime-gpu
) else if !CUDA_MAJOR! EQU 12 (
    echo   [*] CUDA !CUDA_VER! → PyPI 正式版（默认 CUDA 12.x）
    !PYTHON! -m pip install onnxruntime-gpu
) else if !CUDA_MAJOR! EQU 11 (
    echo   [*] CUDA !CUDA_VER! → Azure DevOps CUDA 11 专用源
    !PYTHON! -m pip install coloredlogs flatbuffers numpy packaging protobuf sympy
    !PYTHON! -m pip install onnxruntime-gpu --index-url ^
        https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
) else (
    echo   [!] 无法识别 CUDA !CUDA_MAJOR!.x，跳过 onnxruntime-gpu
)

:: ─────────────────────────────────────────────────────────────
:: 4b. Flash Attention
:: ─────────────────────────────────────────────────────────────
:flash_attn
echo.
echo  ── Flash Attention（可选）───────────────────────────────
echo   参考: https://github.com/mjun0812/flash-attention-prebuild-wheels/releases
echo.

if "!HAS_CUDA!"=="0" (
    echo   [~] 无 CUDA，跳过
    goto :done
)
if "!HAS_TORCH!"=="0" (
    echo   [~] 未检测到 PyTorch，跳过
    goto :done
)

set PLATFORM=win_amd64
set FA_PATTERN=!CUDA_TAG!!TORCH_TAG!-!CP_TAG!-!CP_TAG!-!PLATFORM!
echo   当前环境: !CP_TAG! / !CUDA_TAG! / !TORCH_TAG! / !PLATFORM!
echo   自动匹配 pattern: !FA_PATTERN!
echo.
echo   请粘贴对应的 .whl 下载链接：
echo     留空  = 自动从 GitHub Releases 匹配上述 pattern
echo     skip  = 跳过不安装
echo.
set /p "FA_URL=  URL> "

if /i "!FA_URL!"=="skip" goto :done

if "!FA_URL!"=="" (
    echo   [*] 查询 GitHub Releases...
    :: 将 Python API 查询写入临时文件，避免 BAT 引号转义地狱
    (
        echo import urllib.request, json, sys
        echo req = urllib.request.Request(
        echo     'https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases',
        echo     headers={'User-Agent': 'AnimaLoraStudio-installer'}
        echo ^)
        echo data = json.loads(urllib.request.urlopen(req^).read(^)^)
        echo pat = sys.argv[1]
        echo for r in data:
        echo     for a in r['assets']:
        echo         if pat in a['name']:
        echo             print(a['browser_download_url']^)
        echo             sys.exit(0^)
    ) > _fa_find.py

    for /f "tokens=*" %%u in ('!PYTHON! _fa_find.py "!FA_PATTERN!" 2^>nul') do set FA_URL=%%u
    del _fa_find.py 2>nul

    if "!FA_URL!"=="" (
        echo   [!] 未找到匹配 wheel（pattern: !FA_PATTERN!）
        echo       请手动访问 https://github.com/mjun0812/flash-attention-prebuild-wheels/releases
        goto :done
    )
    echo   [+] 匹配到: !FA_URL!
)

echo   [*] 安装中...
!PYTHON! -m pip install "!FA_URL!"
if %errorlevel% equ 0 (
    echo   [+] Flash Attention 安装成功
) else (
    echo   [!] 安装失败，请检查 URL 或网络连接
)

:: ─────────────────────────────────────────────────────────────
:: 5. ModelScope 模型下载（国内推荐）
:: ─────────────────────────────────────────────────────────────
:modelscope
echo.
echo  ── ModelScope 模型下载（可选，国内加速）────────────────────
echo   https://www.modelscope.cn/models/circlestone-labs/Anima
echo   https://www.modelscope.cn/models/fireicewolf/wd-vit-large-tagger-v3
echo.
set /p "WANT_MS=  是否使用 ModelScope 下载模型? [Y/N]: "
if /i "!WANT_MS!" neq "Y" goto :done

:: 安装 modelscope
echo   [*] 安装 modelscope...
!PYTHON! -m pip install modelscope
if %errorlevel% neq 0 (
    echo   [!] modelscope 安装失败，跳过模型下载
    goto :done
)

:: ── Anima 主模型 ──────────────────────────────────────────────
echo.
echo   ── Anima 模型（circlestone-labs/Anima）
echo     [1] 主模型 anima-preview3-base  (~4 GB)
echo     [2] VAE qwen_image_vae          (~250 MB)
echo     [3] 主模型 + VAE（两项都下载）
echo     [4] 跳过
echo.
set /p "ANIMA_OPT=  选择 [1/2/3/4]: "

if "!ANIMA_OPT!"=="4" goto :wd_menu
if "!ANIMA_OPT!"=="" goto :wd_menu

:: 写下载+整理脚本到临时 .py，避免 BAT 多行引号问题
(
    echo import subprocess, sys, shutil, pathlib, tempfile
    echo PYTHON = sys.executable
    echo ROOT = pathlib.Path('models'^)
    echo REPO = 'circlestone-labs/Anima'
    echo.
    echo def dl_file(repo_path, local_dir^):
    echo     local_dir.mkdir(parents=True, exist_ok=True^)
    echo     r = subprocess.run(
    echo         [PYTHON, '-m', 'modelscope', 'download',
    echo          '--model', REPO, repo_path, '--local_dir', str(local_dir^)],
    echo         check=False
    echo     ^)
    echo     if r.returncode != 0:
    echo         print(f'  [!] 下载失败: {repo_path}'^)
    echo         return
    echo     # modelscope 保留 repo 内部目录结构，把目标文件打平到 local_dir
    echo     fname = pathlib.Path(repo_path^).name
    echo     want = local_dir / fname
    echo     if not want.exists(^):
    echo         matches = list(local_dir.rglob(fname^)^)
    echo         if matches:
    echo             shutil.move(str(matches[0]^), str(want^)^)
    echo             # 清理空的中间目录
    echo             for d in sorted(local_dir.rglob('*'^), reverse=True^):
    echo                 if d.is_dir(^) and d != local_dir:
    echo                     try: d.rmdir(^)
    echo                     except: pass
    echo     if want.exists(^):
    echo         print(f'  [+] {fname} -> {want}'^)
    echo     else:
    echo         print(f'  [!] 未找到 {fname}，请检查网络或手动放置'^)
    echo.
    echo opt = sys.argv[1]
    echo if opt in ('1','3'^):
    echo     dl_file('split_files/diffusion_models/anima-preview3-base.safetensors',
    echo             ROOT / 'diffusion_models'^)
    echo if opt in ('2','3'^):
    echo     dl_file('split_files/vae/qwen_image_vae.safetensors',
    echo             ROOT / 'vae'^)
) > _ms_anima.py

!PYTHON! _ms_anima.py "!ANIMA_OPT!"
del _ms_anima.py 2>nul

:: ── WD 打标模型 ───────────────────────────────────────────────
:wd_menu
echo.
echo   ── WD 打标模型（fireicewolf / ModelScope）
echo     [1] wd-eva02-large-tagger-v3  （推荐）
echo     [2] wd-vit-large-tagger-v3
echo     [3] wd-vit-tagger-v3
echo     [4] wd-v1-4-convnext-tagger-v2
echo     [5] 跳过
echo.
set /p "WD_OPT=  选择 [1-5]: "

if "!WD_OPT!"=="5" goto :done
if "!WD_OPT!"=="" goto :done

if "!WD_OPT!"=="1" set WD_NAME=wd-eva02-large-tagger-v3
if "!WD_OPT!"=="2" set WD_NAME=wd-vit-large-tagger-v3
if "!WD_OPT!"=="3" set WD_NAME=wd-vit-tagger-v3
if "!WD_OPT!"=="4" set WD_NAME=wd-v1-4-convnext-tagger-v2

if "!WD_NAME!"=="" (
    echo   [!] 无效选择，跳过
    goto :done
)

:: WD 模型文件在 repo 根目录（model.onnx + selected_tags.csv），
:: 本地路径按 HF model_id 命名，Studio 才能自动识别
set WD_MS_ID=fireicewolf/!WD_NAME!
set WD_LOCAL=models\wd14\SmilingWolf_!WD_NAME!

echo   [*] 下载 !WD_MS_ID! → !WD_LOCAL!
mkdir "!WD_LOCAL!" 2>nul
!PYTHON! -m modelscope download --model "!WD_MS_ID!" --local_dir "!WD_LOCAL!"
if %errorlevel% equ 0 (
    echo   [+] WD 模型下载完成: !WD_LOCAL!
) else (
    echo   [!] 下载失败，请检查网络或手动下载
    echo       https://www.modelscope.cn/models/!WD_MS_ID!
)

:done
echo.
echo =============================================================
echo   安装完成！运行 studio.bat 启动 AnimaLoraStudio
echo =============================================================
echo.
pause
exit /b 0
