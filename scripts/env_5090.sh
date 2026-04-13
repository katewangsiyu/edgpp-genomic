#!/usr/bin/env bash
# 5090 production env — PyTorch 2.6 + CUDA 12.8 + flash-attn.
set -euo pipefail

ENV_NAME="edgpp_5090"
echo "[env_5090] Creating conda env: $ENV_NAME (Python 3.11, CUDA 12.8)"

conda create -y -n "$ENV_NAME" python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# gcc-12+ for flash-attn compile
conda install -y -c conda-forge gxx=12 ninja

# PyTorch for Blackwell (CUDA 12.8)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Project deps
pip install -r requirements_5090.txt

# flash-attn: recompile for Blackwell from source
echo "[env_5090] Installing flash-attn (may take 15-30 min to compile)..."
pip install flash-attn>=2.7.0 --no-build-isolation

# borzoi-pytorch from baselines/
if [[ -d baselines/borzoi-pytorch ]]; then
    pip install -e baselines/borzoi-pytorch/
else
    echo "[env_5090] WARNING: baselines/borzoi-pytorch/ not found."
    echo "  git clone https://github.com/johahi/borzoi-pytorch baselines/borzoi-pytorch"
    echo "  pip install -e baselines/borzoi-pytorch/"
fi

pip install -e .

echo ""
echo "[env_5090] Done."
echo "  Sanity check:"
echo "    conda activate $ENV_NAME"
echo "    python -c 'import torch; from torch.backends.cuda import sdp_kernel; print(\"flash available:\", sdp_kernel.is_flash_attention_available())'"
