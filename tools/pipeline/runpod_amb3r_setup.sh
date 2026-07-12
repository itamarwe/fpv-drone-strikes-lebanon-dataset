#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
AMB3R_DIR="${AMB3R_DIR:-$WORKSPACE_DIR/amb3r}"
VENV_DIR="${VENV_DIR:-$WORKSPACE_DIR/venvs/amb3r}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE_DIR/pip-cache}"
TMPDIR="${TMPDIR:-$WORKSPACE_DIR/tmp}"
CHECKPOINT_ID="${CHECKPOINT_ID:-14x0WW2rUE_he2hUEouP6ywSRnlJDeLel}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
MAX_JOBS="${MAX_JOBS:-4}"

export TORCH_CUDA_ARCH_LIST MAX_JOBS FORCE_CUDA=1 PIP_CACHE_DIR TMPDIR

echo "[amb3r-setup] workspace=$WORKSPACE_DIR"
echo "[amb3r-setup] amb3r_dir=$AMB3R_DIR"
echo "[amb3r-setup] venv_dir=$VENV_DIR"
echo "[amb3r-setup] pip_cache_dir=$PIP_CACHE_DIR"
echo "[amb3r-setup] tmpdir=$TMPDIR"
echo "[amb3r-setup] torch_cuda_arch_list=$TORCH_CUDA_ARCH_LIST max_jobs=$MAX_JOBS"

mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  ca-certificates \
  cmake \
  curl \
  ffmpeg \
  git \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  ninja-build \
  python3.10-venv \
  wget

cd "$WORKSPACE_DIR"
if [[ ! -d "$AMB3R_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/HengyiWang/amb3r.git "$AMB3R_DIR"
else
  git -C "$AMB3R_DIR" pull --ff-only
fi

cd "$AMB3R_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel packaging ninja

python -m pip install \
  torch==2.5.0 \
  torchvision==0.20.0 \
  torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install \
  torch-scatter==2.1.2 \
  -f https://data.pyg.org/whl/torch-2.5.0+cu118.html

python -m pip install \
  "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8" \
  --no-build-isolation

python -m pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation
python -m pip install --ignore-installed blinker==1.9.0
python -m pip install -r requirements.txt

mkdir -p checkpoints
if [[ ! -s checkpoints/amb3r.pt ]]; then
  python -m gdown "$CHECKPOINT_ID" -O checkpoints/amb3r.pt
fi

python - <<'PY'
import importlib
import os
import torch

print("[amb3r-setup] torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[amb3r-setup] gpu", torch.cuda.get_device_name(0))

for name in [
    "open3d",
    "xformers",
    "spconv",
    "pytorch3d",
    "flash_attn",
    "torch_scatter",
    "omegaconf",
    "evo",
    "timm",
]:
    importlib.import_module(name)
    print("[amb3r-setup] imported", name)

assert os.path.getsize("checkpoints/amb3r.pt") > 0
print("[amb3r-setup] checkpoint bytes", os.path.getsize("checkpoints/amb3r.pt"))
PY

echo "[amb3r-setup] done"
