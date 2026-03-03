FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    PIPELINE_WORKSPACE=/workspace \
    PIPELINE_OUTPUT_DIR=/workspace/outputs \
    PIPELINE_TEMP_DIR=/workspace/temp \
    PIPELINE_INPUT_DIR=/workspace/inputs

ARG CHAMP_REPO=https://github.com/fudan-generative-vision/champ.git
ARG CHAMP_REF=main
ARG RETALKING_REPO=https://github.com/OpenTalker/video-retalking.git
ARG RETALKING_REF=main
ARG PRELOAD_MODELS=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-distutils python3-pip \
    git git-lfs wget curl unzip ca-certificates \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libegl1 \
    build-essential cmake ninja-build \
    libgoogle-perftools-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel

RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace
RUN git clone --depth 1 --branch ${CHAMP_REF} ${CHAMP_REPO} champ
WORKDIR /workspace/champ
RUN pip install -r requirements.txt

WORKDIR /workspace
RUN git clone --depth 1 --branch ${RETALKING_REF} ${RETALKING_REPO} video-retalking
WORKDIR /workspace/video-retalking
RUN pip install -r requirements.txt

RUN pip install \
    boto3==1.34.131 \
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

WORKDIR /workspace
RUN mkdir -p /workspace/scripts /workspace/configs /workspace/inputs /workspace/outputs /workspace/temp
COPY scripts/ /workspace/scripts/
COPY configs/ /workspace/configs/
COPY pipeline.py /workspace/pipeline.py
COPY download_models.py /workspace/download_models.py
COPY runpod_handler.py /workspace/runpod_handler.py

RUN if [ "${PRELOAD_MODELS}" = "1" ]; then python /workspace/download_models.py --champ --retalking; fi

CMD ["python", "-u", "/workspace/runpod_handler.py"]
