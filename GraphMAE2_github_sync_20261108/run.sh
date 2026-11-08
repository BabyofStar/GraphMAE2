#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash -n "$PACKAGE/run.sh"
PROJECT=/home/jovyan/work/GraphMAE2
STAGING=/home/jovyan/work/GraphMAE2_backup_staging
PY=/home/jovyan/work/envs/graphmae2/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 GIT_OPTIONAL_LOCKS=0
mkdir -p "$STAGING"
exec 9>>"$PROJECT/outputs/.formal_record_v1.lock"
flock -n 9 || { printf 'Another GraphMAE2 task holds the project lock; stopped.\n' >&2; exit 2; }
exec 8>>"$STAGING/.github_sync.lock"
flock -n 8 || { printf 'Another GitHub backup is active; stopped.\n' >&2; exit 2; }
RECEIPT="$PACKAGE/.launch_receipt.json"
if [[ -f "$RECEIPT" ]]; then
  OUT="$("$PY" -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["directory"])' "$RECEIPT")"
else
  OUT="$(mktemp -d "$STAGING/sync_XXXXXXXX")"
  "$PY" -B -c 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);t=p.with_suffix(".tmp");t.write_text(json.dumps({"directory":sys.argv[2]})+"\n");t.replace(p)' "$RECEIPT" "$OUT"
fi
[[ "$OUT" == "$STAGING"/sync_* && -d "$OUT" ]] || { printf 'Invalid resume directory.\n' >&2; exit 2; }
printf 'BACKUP_DIR=%s\n' "$OUT"
OWN_AGENT=0
finish() {
  result=$?
  trap - EXIT INT TERM
  if [[ "$OWN_AGENT" == 1 ]]; then ssh-agent -k >/dev/null 2>&1 || true; fi
  "$PY" -B "$PACKAGE/package_return.py" --output-dir "$OUT" --exit-code "$result" || true
  if [[ "$result" != 0 ]]; then printf 'Stopped; completed work is preserved. Re-run this same run.sh to resume.\n' >&2; fi
  exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
phase() {
  printf 'PHASE=%s\n' "$1" | tee -a "$OUT/execution.log"
  shift
  "$@" 2>&1 | tee -a "$OUT/execution.log"
}
phase package_identity "$PY" -B -c 'import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from common import job;job(Path(sys.argv[2]));print("PACKAGE_IDENTITY=PASS")' "$PACKAGE" "$OUT"
phase archive_self_check "$PY" -B "$PACKAGE/self_check.py" --output-dir "$OUT"
if [[ -f "$OUT/final_sync_verified.json" ]]; then
  printf 'This job already completed. Existing receipts returned; no second backup launched.\n'
  exit 0
fi
printf 'Enter the existing SSH key passphrase if prompted.\n'
eval "$(ssh-agent -s 9>&- 8>&-)" >/dev/null
OWN_AGENT=1
ssh-add /home/jovyan/.ssh/id_ed25519_graphmae2
export GIT_SSH_COMMAND='ssh -i /home/jovyan/.ssh/id_ed25519_graphmae2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o BatchMode=yes'
if [[ ! -f "$OUT/source_synced.json" ]]; then
  phase source_sync "$PY" -B "$PACKAGE/sync_source.py" --output-dir "$OUT"
fi
phase weight_archive "$PY" -B "$PACKAGE/weights.py" --output-dir "$OUT"
phase release_login_helper "$PY" -B "$PACKAGE/github_release.py" install-cli --output-dir "$OUT"
GH_BIN="$("$PY" -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["path"])' "$OUT/gh_tool.json")"
export GH_CONFIG_DIR="$STAGING/.gh_private"
mkdir -p "$GH_CONFIG_DIR"
chmod 700 "$GH_CONFIG_DIR"
unset GH_TOKEN GITHUB_TOKEN
export GH_PROMPT_DISABLED=1 GH_HOST=github.com
if ! "$GH_BIN" api --hostname github.com user --jq .login >/dev/null 2>&1; then
  printf 'GitHub Release login: open the shown URL and enter its one-time code in your browser.\n'
  printf 'Complete login as BabyofStar. Do not send a token or password to chat.\n'
  env -u GH_PROMPT_DISABLED "$GH_BIN" auth login --hostname github.com --git-protocol ssh --web --skip-ssh-key
fi
phase release_upload_and_verify "$PY" -B "$PACKAGE/github_release.py" publish --output-dir "$OUT"
phase backup_metadata_sync "$PY" -B "$PACKAGE/sync_source.py" --output-dir "$OUT" --finalize
printf 'GITHUB_SYNC_AND_WEIGHT_BACKUP=PASS\n' | tee -a "$OUT/execution.log"
