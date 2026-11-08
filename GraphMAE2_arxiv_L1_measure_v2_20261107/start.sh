#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=/home/jovyan/work/GraphMAE2
if [[ -e "$PACKAGE/launch_receipt.json" ]]; then
  printf 'This package has an existing launch receipt; no second task started.\n' >&2
  cat "$PACKAGE/launch_receipt.json"
  exit 2
fi
if ! mkdir "$PACKAGE/.launch_claim"; then
  printf 'This package has an existing launch claim; inspect the previous log.\n' >&2
  exit 2
fi
OUT="$(mktemp -d "$PROJECT/outputs/arxiv_l1_measure_v2_XXXXXXXX")"
nohup bash "$PACKAGE/run.sh" "$OUT" > "$OUT/run.log" 2>&1 < /dev/null &
PID=$!
/home/jovyan/work/envs/graphmae2/bin/python -B - "$PACKAGE/launch_receipt.json" "$OUT" "$PID" <<'PY'
import json,sys
with open(sys.argv[1],'x',encoding='utf-8') as stream:
    json.dump(dict(directory=sys.argv[2],pid=int(sys.argv[3])),stream,indent=2)
PY
printf 'MEASUREMENT_DIR=%s\nPID=%s\nLOG=%s/run.log\nRETURN_FILE=%s/measurement_return.zip\n' "$OUT" "$PID" "$OUT" "$OUT"
