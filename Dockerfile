# ============================================================
# GPU BASE IMAGE (CUDA 12.3 + cuDNN 9 | Ubuntu 22.04)
# ============================================================
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# ============================================================
# SYSTEM CONFIG
# ============================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=all

# ============================================================
# WORKDIR
# ============================================================
WORKDIR /app

# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# PYTHON SETUP
# ============================================================
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

# ============================================================
# REQUIREMENTS
# ============================================================
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# APPLICATION CODE
# ============================================================
COPY . /app/

# ============================================================
# PYTHON PATH
# ============================================================
ENV PYTHONPATH=/app

# ============================================================
# OPTIONAL: VERIFY GPU VISIBILITY (SAFE)
# ============================================================
RUN python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
EOF

# ============================================================
# DEFAULT COMMAND (CELERY WORKER)
# ============================================================
CMD ["celery", "-A", "app.app.celery", "worker", "--pool=solo", "--loglevel=info", "--imports=app.tasks"]
