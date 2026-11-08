#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${CONDA_PREFIX:-}" != /home/jovyan/work/envs/graphmae2 ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/jovyan/work/envs/graphmae2
fi
export PYTHONDONTWRITEBYTECODE=1
ARTIFACT_PARENT=/home/jovyan/work/GraphMAE2_artifacts
mkdir -p "$ARTIFACT_PARENT"
ASSET_ROOT="$(mktemp -d "$ARTIFACT_PARENT/arxiv_prepare_XXXXXXXX")"
python -B "$SCRIPT_DIR/prepare_data.py" --asset-root "$ASSET_ROOT"
