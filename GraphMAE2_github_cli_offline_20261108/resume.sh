#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash -n "$PACKAGE/resume.sh"
ORIGINAL=/home/jovyan/work/GraphMAE2/GraphMAE2_github_sync_20261108
STAGING=/home/jovyan/work/GraphMAE2_backup_staging
PY=/home/jovyan/work/envs/graphmae2/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec 9>>/home/jovyan/work/GraphMAE2/outputs/.formal_record_v1.lock
flock -n 9 || { printf 'Another GraphMAE2 task is active; stopped.\n' >&2; exit 2; }
exec 8>>"$STAGING/.github_sync.lock"
flock -n 8 || { printf 'Another GitHub backup is active; stopped.\n' >&2; exit 2; }
"$PY" -B "$PACKAGE/install_offline.py"
flock -u 8
exec 8>&-
flock -u 9
exec 9>&-
exec bash "$ORIGINAL/run.sh"
