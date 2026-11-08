#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${1:-/home/jovyan/work/GraphMAE2}"
if [[ "${CONDA_PREFIX:-}" != /home/jovyan/work/envs/graphmae2 ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/jovyan/work/envs/graphmae2
fi
export PYTHONDONTWRITEBYTECODE=1
REPORT_DIR="$(mktemp -d "$PROJECT/outputs/arxiv_l0_followup_XXXXXXXX")"
python -B "$SCRIPT_DIR/collect_followup.py" --project "$PROJECT" --output-dir "$REPORT_DIR"
