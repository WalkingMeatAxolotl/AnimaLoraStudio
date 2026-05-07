# --- .ide/Dockerfile (AnimaLoraStudio CNB 适配版) ---

# 1. 使用 Nvidia CUDA 开发镜像作为基础
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

# 2. 设置环境变量
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
EXPOSE 8188

# 3. 安装系统依赖
#    - 包含 AnimaLoraStudio 所需的全部系统级包
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-tk \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    python3-venv \
    python3 \
    python-is-python3 \
    python3-pip \
    zip \
    unzip \
    wget \
    curl \
    git \
    python3-dev \
    build-essential \
    ca-certificates \
    gnupg \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. 设置系统级 Git LFS hooks
RUN git lfs install --system

# 5. 设置默认工作目录
WORKDIR /workspace
