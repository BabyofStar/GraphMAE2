#!/usr/bin/env bash
set -eo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/jovyan/work/envs/graphmae2
GRAPHMAE2_C_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GRAPHMAE2_NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
GRAPHMAE2_CUDA_LIBS="$GRAPHMAE2_NVIDIA_DIR/cuda_runtime/lib:$GRAPHMAE2_NVIDIA_DIR/cublas/lib:$GRAPHMAE2_NVIDIA_DIR/cusparse/lib"
export LD_LIBRARY_PATH="$GRAPHMAE2_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DGLBACKEND=pytorch
export DGL_DOWNLOAD_DIR=/home/jovyan/work/GraphMAE2/data/dgl
export WANDB_MODE=disabled MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$GRAPHMAE2_C_PACKAGE"
sha256sum --check SHA256SUMS.txt
if [ -f launch_receipt.json ]; then
    python launch_c9.py
    exit 0
fi
python install_c.py --repo /home/jovyan/work/GraphMAE2
python run_stage1_c.py --repo /home/jovyan/work/GraphMAE2
python launch_c9.py
