#!/bin/bash

# ==============================================================================
#  AnimaLoraStudio CNB 环境初始化脚本
#
#  CNB 容器每次重启都是全新环境，此脚本负责：
#  1. 拉取 Git LFS 大文件（模型权重 / 数据集）
#  2. 验证 CUDA / GPU 可用性
#  3. 创建必要的目录结构
#  4. 检查训练配置文件是否存在
#
#  用法:
#    chmod +x setup.sh
#    ./setup.sh
# ==============================================================================

set -e

echo "============================================"
echo " AnimaLoraStudio CNB 环境初始化"
echo "============================================"
echo ""

# --- 1. Git LFS 拉取 ---
echo "[1/4] 拉取 Git LFS 文件..."
if command -v git-lfs &> /dev/null; then
    git lfs pull
    echo "  Git LFS pull 完成。"
else
    echo "  [警告] git-lfs 未安装，跳过 LFS 拉取。"
    echo "         模型文件可能需要手动下载。"
fi
echo ""

# --- 2. 模型文件检查 ---
echo "[2/4] 检查模型文件..."
MISSING_MODELS=0

check_file() {
    local desc="$1"
    local path="$2"
    if [ -f "$path" ]; then
        local size=$(du -h "$path" | cut -f1)
        echo "  [OK] $desc ($size)"
    else
        echo "  [缺失] $desc -> $path"
        MISSING_MODELS=1
    fi
}

check_dir() {
    local desc="$1"
    local path="$2"
    if [ -d "$path" ] && [ -n "$(ls -A "$path" 2>/dev/null)" ]; then
        echo "  [OK] $desc (目录存在且非空)"
    else
        echo "  [缺失] $desc -> $path"
        MISSING_MODELS=1
    fi
}

check_file "Transformer"  "models/diffusion_models/anima-preview3-base.safetensors"
check_file "VAE"          "models/vae/qwen_image_vae.safetensors"
check_dir  "Text Encoder" "models/text_encoders"
check_dir  "T5 Tokenizer" "models/t5_tokenizer"

if [ "$MISSING_MODELS" -eq 1 ]; then
    echo ""
    echo "  [提示] 部分模型文件缺失。请确认："
    echo "    1. 模型文件已通过 git lfs track 并 push"
    echo "    2. 已运行 'git lfs install && git lfs pull'"
    echo "    3. 或手动将模型文件放入对应目录后用 'git add' 提交"
fi
echo ""

# --- 3. 必要目录 ---
echo "[3/4] 创建必要目录..."
mkdir -p output
mkdir -p logs
echo "  output/ logs/ 就绪。"
echo ""

# --- 4. 训练配置检查 ---
echo "[4/4] 检查训练配置..."
if [ -f "config/train_my.yaml" ]; then
    echo "  [OK] config/train_my.yaml 存在。"
else
    echo "  [警告] config/train_my.yaml 不存在，将使用默认配置。"
    echo "         请编辑此文件设置 data_dir、模型路径等参数。"
fi
echo ""

# --- GPU 检测 ---
echo "============================================"
echo " GPU 状态"
echo "============================================"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  nvidia-smi 运行失败，但 GPU 驱动可能已安装。"
else
    echo "  [警告] nvidia-smi 不可用，CUDA 可能未正确配置。"
fi
echo ""

echo "============================================"
echo " 初始化完成!"
echo "============================================"
echo ""
echo "  下一步:"
echo "    1. 编辑 config/train_my.yaml 设置 data_dir"
echo "    2. 运行: bash run.sh"
echo ""
