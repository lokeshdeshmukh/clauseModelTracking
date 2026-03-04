FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    PIPELINE_WORKSPACE=/workspace \
    PIPELINE_OUTPUT_DIR=/workspace/outputs \
    PIPELINE_TEMP_DIR=/workspace/temp \
    PIPELINE_INPUT_DIR=/workspace/inputs

ARG CHAMP_REPO=https://github.com/fudan-generative-vision/champ.git
ARG RETALKING_REPO=https://github.com/OpenTalker/video-retalking.git
ARG FOURD_HUMANS_REPO=https://github.com/shubham-goel/4D-Humans.git
ARG DWPose_REPO=https://github.com/IDEA-Research/DWPose.git
ARG PRELOAD_MODELS=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-distutils python3-pip \
    git git-lfs wget curl unzip ca-certificates \
    ffmpeg blender \
    pkg-config libx11-dev libjpeg-dev \
    libopenblas-dev liblapack-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libegl1 libosmesa6 libosmesa6-dev \
    gcc-12 g++-12 \
    build-essential cmake ninja-build \
    libgoogle-perftools-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install --retries 10 --timeout 120 \
    torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --trusted-host download.pytorch.org

# Triton pulls in the Python `cmake` wrapper into /usr/local/bin/cmake, which
# breaks dlib's build isolation. Force the system cmake binary instead.
RUN rm -f /usr/local/bin/cmake /usr/local/bin/ctest /usr/local/bin/cpack \
    && ln -sf /usr/bin/cmake /usr/local/bin/cmake \
    && ln -sf /usr/bin/ctest /usr/local/bin/ctest \
    && ln -sf /usr/bin/cpack /usr/local/bin/cpack

WORKDIR /workspace
RUN git clone --depth 1 ${CHAMP_REPO} champ
WORKDIR /workspace/champ
RUN grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/champ-requirements.txt \
    && pip install -r /tmp/champ-requirements.txt
RUN git clone --depth 1 ${DWPose_REPO} DWPose

WORKDIR /workspace
RUN git clone --depth 1 ${FOURD_HUMANS_REPO} 4D-Humans
WORKDIR /workspace/4D-Humans
RUN pip install --no-build-isolation git+https://github.com/mattloper/chumpy \
    && CC=gcc-12 CXX=g++-12 pip install --no-build-isolation -e .[all]

WORKDIR /workspace
RUN git clone --depth 1 ${RETALKING_REPO} video-retalking
WORKDIR /workspace/video-retalking
RUN grep -v '^dlib==' requirements.txt > /tmp/video-retalking-requirements.txt \
    && pip install -r /tmp/video-retalking-requirements.txt \
    && CMAKE_ARGS="-DDLIB_USE_CUDA=0" pip install --verbose dlib==19.24.0

RUN python -m pip install --force-reinstall --retries 10 --timeout 120 \
    torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --trusted-host download.pytorch.org

RUN pip install \
    boto3==1.34.131 \
    huggingface_hub==0.21.4 \
    omegaconf==2.2.3 \
    einops==0.4.1 \
    diffusers==0.24.0 \
    accelerate==0.21.0 \
    transformers==4.30.2 \
    controlnet-aux==0.0.7 \
    safetensors==0.4.2 \
    imageio==2.33.0 \
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
COPY runpod_preprocess_handler.py /workspace/runpod_preprocess_handler.py
COPY worker_entrypoint.py /workspace/worker_entrypoint.py

RUN if [ "${PRELOAD_MODELS}" = "1" ]; then python /workspace/download_models.py --champ --preprocess --retalking; fi

CMD ["python", "-u", "/workspace/worker_entrypoint.py"]
