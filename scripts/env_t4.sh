#!/usr/bin/env bash
# T4 debug env — no flash-attn, Phase 0 uses FakeTeacher.
set -euo pipefail

ENV_NAME="edgpp_t4"
echo "[env_t4] Creating conda env: $ENV_NAME (Python 3.11)"

conda create -y -n "$ENV_NAME" python=3.11
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# PyTorch with CUDA 12.4 wheels (T4 supports up to CUDA 12.x)
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Project deps
pip install -r requirements_t4.txt
pip install -e .

echo ""
echo "[env_t4] Done."
echo "  conda activate $ENV_NAME"
