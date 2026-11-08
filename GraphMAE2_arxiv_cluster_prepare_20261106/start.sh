#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=/home/jovyan/work/GraphMAE2
if [[ -e "$PACKAGE/launch_receipt.json" ]]; then
  printf 'This package already has a launch receipt. Inspect it and the previous log; no second run started.\n' >&2
  cat "$PACKAGE/launch_receipt.json"
  exit 2
fi
if ! mkdir "$PACKAGE/.launch_claim"; then
  printf 'This package already has a launch claim; inspect previous output.\n' >&2
  exit 2
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/jovyan/work/envs/graphmae2
OUT="$(mktemp -d "$PROJECT/outputs/arxiv_lc_prepare_XXXXXXXX")"
nohup bash "$PACKAGE/run.sh" "$OUT" > "$OUT/run.log" 2>&1 < /dev/null &
PID=$!
python -B - "$PACKAGE/launch_receipt.json" "$OUT" "$PID" <<'PY'
import json,sys
with open(sys.argv[1],'x',encoding='utf-8') as f:json.dump(dict(directory=sys.argv[2],pid=int(sys.argv[3])),f,indent=2)
PY
printf 'LC_PREPARATION_DIR=%s\nPID=%s\nLOG=%s/run.log\nRETURN_FILE=%s/clustering_return.zip\n' "$OUT" "$PID" "$OUT" "$OUT"
