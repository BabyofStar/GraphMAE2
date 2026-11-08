#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${1:-/home/jovyan/work/GraphMAE2}"
if [[ ! -d "$PROJECT" ]]; then
  printf 'Project directory not found: %s\n' "$PROJECT" >&2
  exit 2
fi
if [[ "${CONDA_PREFIX:-}" != /home/jovyan/work/envs/graphmae2 ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    printf 'Conda unavailable. Activate /home/jovyan/work/envs/graphmae2 and retry.\n' >&2
    exit 2
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/jovyan/work/envs/graphmae2
fi
GRAPHMAE2_NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
GRAPHMAE2_CUDA_LIBS="$GRAPHMAE2_NVIDIA_DIR/cuda_runtime/lib:$GRAPHMAE2_NVIDIA_DIR/cublas/lib:$GRAPHMAE2_NVIDIA_DIR/cusparse/lib"
export LD_LIBRARY_PATH="$GRAPHMAE2_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DGLBACKEND=pytorch
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
REPORT_DIR="$(mktemp -d "$PROJECT/outputs/arxiv_l0_XXXXXXXX")"
python -B "$SCRIPT_DIR/arxiv_l0_preflight.py" --project "$PROJECT" --reference "$SCRIPT_DIR/reference_identity.json" | tee "$REPORT_DIR/preflight.json"
printf '\nL0_REPORT=%s/preflight.json\n' "$REPORT_DIR"
printf 'L0 only. No training or data preparation has been started.\n'
