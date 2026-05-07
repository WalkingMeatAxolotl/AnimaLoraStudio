# --- .ide/Dockerfile (AnimaLoraStudio CNB 适配版) ---

FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

EXPOSE 8188

# 1. 系统依赖
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

# 2. 安装 Node.js 20
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. 安装 Git LFS
RUN git lfs install --system

# 4. 创建持久化虚拟环境
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 5. 安装 PyTorch（CUDA 12.x 预编译包）
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# 6. 安装其余 Python 依赖
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# 7. 预安装前端 node_modules（避免每次新环境重跑 npm install）
COPY studio/web/package*.json /opt/studio-web/
RUN cd /opt/studio-web && npm ci

# 8. 工作目录（CNB 会将仓库代码挂载到这里）
WORKDIR /workspace
