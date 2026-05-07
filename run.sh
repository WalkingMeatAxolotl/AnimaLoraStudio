#!/bin/bash

# ==============================================================================
#     AnimaLoraStudio 统一训练与管理脚本 (CNB Background Training Script)
#
# 功能:
# 1. 检查 Python 环境和相关脚本/配置是否存在。
# 2. 在后台安全启动 AnimaLoraStudio 训练过程。
# 3. 自动生成一个 'stop_anima_training.sh' 脚本，用于一键精准停止相关进程。
# ==============================================================================

# --- 用户配置区域 (User Configuration) ---

# 训练脚本路径
TRAIN_SCRIPT="./anima_train.py"

# 训练配置文件路径
CONFIG_FILE="./config/train_my.yaml"

# 日志输出路径
LOG_FILE="./anima_training.log"


# ==============================================================================
#                  脚本主体 (DO NOT MODIFY BELOW THIS LINE)
# ==============================================================================

STOP_SCRIPT="./stop_anima_training.sh"

# --- 0. 检测 Python 环境（CNB 镜像内置 vs 本地 venv）---
echo "正在检测 Python 环境..."
PYTHON_BIN=""
if [ -f "/opt/venv/bin/python" ]; then
    PYTHON_BIN="/opt/venv/bin/python"
    echo "  使用 CNB 镜像内置环境: $PYTHON_BIN"
elif [ -f "./venv/bin/python" ]; then
    PYTHON_BIN="./venv/bin/python"
    echo "  使用本地虚拟环境: $PYTHON_BIN"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="$(command -v python3)"
    echo "  使用系统 Python: $PYTHON_BIN"
else
    echo "错误: 未找到可用的 Python 环境"
    exit 1
fi

# --- 1. 路径与文件检查 ---
echo "正在检查运行环境..."
if [ ! -f "$PYTHON_BIN" ]; then
    echo "错误: Python 不可用: $PYTHON_BIN"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "错误: 训练脚本未找到: $TRAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件未找到: $CONFIG_FILE"
    exit 1
fi
echo "环境检查通过。"
echo " "

# --- 2. 启动训练 ---
echo "正在后台启动 AnimaLoraStudio 训练脚本..."

# 使用 nohup 将进程挂起在后台
nohup "$PYTHON_BIN" "$TRAIN_SCRIPT" --config="$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "训练任务已成功提交至后台运行 (PID: $TRAIN_PID)。"
echo "您可以通过以下命令实时查看训练进度和日志:"
echo "tail -f $LOG_FILE"
echo " "

# --- 3. 动态生成停止脚本 ---
echo "正在生成一键停止脚本 'stop_anima_training.sh'..."

cat > "$STOP_SCRIPT" <<EOF
#!/bin/bash
echo "正在查找 AnimaLoraStudio 训练进程..."

# 查找匹配 anima_train.py 及其特定配置文件的进程
TRAIN_PIDS=\$(pgrep -f "anima_train.py.*$CONFIG_FILE")

if [ -z "\$TRAIN_PIDS" ]; then
    echo "未找到正在运行的 AnimaLoraStudio 训练进程。"
else
    PIDS_TO_KILL=\$(echo \$TRAIN_PIDS | tr '\\n' ' ')
    echo "将要终止以下进程 (PIDs): \$PIDS_TO_KILL"
    kill -9 \$PIDS_TO_KILL > /dev/null 2>&1
    echo "所有相关进程已被终止。"
fi

echo "脚本执行完毕，将自动删除自身..."
rm -- "\$0"
EOF

# 赋予停止脚本执行权限
chmod +x "$STOP_SCRIPT"

echo "停止脚本 '$STOP_SCRIPT' 已成功创建。"
echo "--------------------------------------------------------"
echo "当您需要停止训练时，只需在终端运行这个命令:"
echo "bash $STOP_SCRIPT"
echo "--------------------------------------------------------"
