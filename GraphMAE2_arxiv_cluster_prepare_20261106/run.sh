#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT="$1"
PROJECT=/home/jovyan/work/GraphMAE2
exec 9>>"$PROJECT/outputs/.formal_record_v1.lock"
if ! flock -n 9; then
  printf 'Project execution lock is held; no dependency setup or preprocessing started.\n' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES="" DGLBACKEND=pytorch MPLBACKEND=Agg WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$OUT/matplotlib_cache"
export TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
GRAPHMAE2_NVIDIA_DIR=/home/jovyan/work/envs/graphmae2/lib/python3.10/site-packages/nvidia
GRAPHMAE2_CUDA_LIBS="$GRAPHMAE2_NVIDIA_DIR/cuda_runtime/lib:$GRAPHMAE2_NVIDIA_DIR/cublas/lib:$GRAPHMAE2_NVIDIA_DIR/cusparse/lib"
export LD_LIBRARY_PATH="$GRAPHMAE2_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
/home/jovyan/work/envs/graphmae2/bin/python -B "$PACKAGE/bootstrap.py" --project "$PROJECT" --output-dir "$OUT"
