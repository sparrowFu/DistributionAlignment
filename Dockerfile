# syntax=docker/dockerfile:1
# =============================================================================
# GaussianImageDistribution / Distribution Alignment
#   Python 3.10  +  CUDA 12.8 (cu128)  —— 对应 conda 环境 CudaVersion128Fuxp
#
# 设计要点:
#   1. WORKDIR 故意设成 config.py 里硬编码的 PROJECT_ROOT,
#      这样容器内路径与本地完全一致,代码无需任何改动。
#   2. checkpoints/ PreTrainedModels/ TrainDatasets/ outputs/ logs/
#      这些大目录(合计 ~49GB)不进镜像,运行时用 -v 挂载(见 .dockerignore)。
#   3. torch 用官方 cu128 索引单独安装,其余依赖见 requirements-docker.txt。
# =============================================================================

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8

# ---- 系统依赖 + Python 3.10(Ubuntu 22.04 自带 python3 == 3.10)-----------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv \
        curl ca-certificates git \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# ---- 与 config.py 的 PROJECT_ROOT 保持一致,代码零改动 ----------------------
WORKDIR /home/xpfu/WorkSpace/DistributionAlignment

# ---- 1) 先装依赖(利用层缓存,改代码不重装依赖)----------------------------
# torch / torchvision / torchaudio 必须走 cu128 官方索引
RUN pip install --no-cache-dir \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements-docker.txt ./requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# ---- 2) 拷贝源码(大目录已被 .dockerignore 排除)----------------------------
COPY . .

# ---- 3) 默认入口:运行时用 `--task <name>` 传参 -----------------------------
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
