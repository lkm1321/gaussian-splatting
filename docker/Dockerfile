FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base
# pip needed obviously, egl1, gl1 and gomp1 for open3d
RUN apt update && apt install -y \
    python3-pip \
    python-is-python3 \
    libegl1 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/**

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.7 8.9 9.0"

FROM base

RUN pip install opencv-python-headless tqdm fire joblib plyfile pyyaml open3d
# TODO: need to copy
COPY . /gaussian-splatting
WORKDIR /gaussian-splatting

# Needed for headless
RUN pip install submodules/diff-gaussian-rasterization
RUN pip install submodules/simple-knn
RUN pip install submodules/fused-ssim
