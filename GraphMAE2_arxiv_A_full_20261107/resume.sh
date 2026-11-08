#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=/home/jovyan/work/GraphMAE2
OUT="$(/home/jovyan/work/envs/graphmae2/bin/python -B - "$PACKAGE/launch_receipt.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['directory'])
PY
)"
if [[ "$OUT" != "$PROJECT"/outputs/arxiv_A_* || ! -d "$OUT" ]]; then
  printf 'Unexpected A run directory; no resume started.\n' >&2
  exit 2
fi
if [[ -e "$OUT/resume_requires_review.json" ]]; then
  printf 'Allocator failure needs evidence review; automatic resume stopped.\n' >&2
  cat "$OUT/resume_requires_review.json"
  exit 2
fi
nohup bash "$PACKAGE/run.sh" "$OUT" resume >> "$OUT/run.log" 2>&1 < /dev/null &
PID=$!
printf 'RESUME_PID=%s\nLOG=%s/run.log\nThe new RETURN_FILE will be printed at the end.\n' "$PID" "$OUT"
