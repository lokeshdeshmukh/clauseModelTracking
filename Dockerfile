# ============================================================
# Champ + VideoRetalking Pipeline - RunPod Docker Image
# Base: CUDA 11.8 + cuDNN 8 + Ubuntu 22.04
# ============================================================

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# ── System env ────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# ── System dependencies ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils python3-pip \
    git git-lfs wget curl unzip \
    ffmpeg libffmpeg-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libegl1 \
    build-essential cmake ninja-build \
    libgoogle-perftools-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip setuptools wheel

# ── PyTorch (CUDA 11.8) ────────────────────────────────────
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ── Clone Champ ────────────────────────────────────────────
WORKDIR /workspace
RUN git clone https://github.com/fudan-generative-vision/champ.git
WORKDIR /workspace/champ
RUN pip install -r requirements.txt

# ── Clone VideoRetalking ───────────────────────────────────
WORKDIR /workspace
RUN git clone https://github.com/OpenTalker/video-retalking.git
WORKDIR /workspace/video-retalking
RUN pip install -r requirements.txt

# ── Extra shared dependencies ──────────────────────────────
RUN pip install \
    gradio==4.19.2 \
    huggingface_hub==0.21.4 \
    omegaconf==2.3.0 \
    einops==0.7.0 \
    diffusers==0.27.2 \
    accelerate==0.27.2 \
    transformers==4.38.2 \
    controlnet-aux==0.0.7 \
    safetensors==0.4.2 \
    imageio==2.34.0 \
    imageio-ffmpeg==0.4.9 \
    mediapipe==0.10.9 \
    facexlib==0.3.0 \
    basicsr==1.4.2 \
    realesrgan==0.3.0 \
    gfpgan==1.3.8 \
    runpod==1.6.2

# ── Copy project files ─────────────────────────────────────
WORKDIR /workspace
COPY scripts/           /workspace/scripts/
COPY app/               /workspace/app/
COPY configs/           /workspace/configs/
COPY pipeline.py        /workspace/pipeline.py
COPY download_models.py /workspace/download_models.py
COPY runpod_handler.py  /workspace/runpod_handler.py

# ── Download model weights at build time ──────────────────
# (comment out if you want runtime download to keep image smaller)
RUN python /workspace/download_models.py --champ --retalking

# ── Ports & entrypoint ─────────────────────────────────────
EXPOSE 7860

CMD ["python", "/workspace/runpod_handler.py"]
