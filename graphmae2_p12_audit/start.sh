#!/usr/bin/env bash
set -eo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/jovyan/work/envs/graphmae2
GRAPHMAE2_AUDIT_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GRAPHMAE2_NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
GRAPHMAE2_CUDA_LIBS="$GRAPHMAE2_NVIDIA_DIR/cuda_runtime/lib:$GRAPHMAE2_NVIDIA_DIR/cublas/lib:$GRAPHMAE2_NVIDIA_DIR/cusparse/lib"
export LD_LIBRARY_PATH="$GRAPHMAE2_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$GRAPHMAE2_AUDIT_PACKAGE"
sha256sum --check SHA256SUMS.txt
python audit_p12.py
