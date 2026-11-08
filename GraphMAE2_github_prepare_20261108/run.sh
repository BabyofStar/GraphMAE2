#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=/home/jovyan/work/GraphMAE2
STAGING=/home/jovyan/work/GraphMAE2_backup_staging
mkdir -p "$STAGING"
OUT="$(mktemp -d "$STAGING/prepare_XXXXXXXX")"
exec 9>>"$PROJECT/outputs/.formal_record_v1.lock"
if ! flock -n 9; then
  printf 'GraphMAE2 task is still running; preparation stopped without staging or uploading.\n' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export GIT_OPTIONAL_LOCKS=0
printf 'PREPARATION_DIR=%s\n' "$OUT"
if ! /home/jovyan/work/envs/graphmae2/bin/python -B "$PACKAGE/collect.py" --output-dir "$OUT"; then
  printf 'Preparation stopped; inspect the return file printed above.\n' >&2
  exit 1
fi
if [[ "$(git -C "$PROJECT" remote get-url --all course_backup 2>/dev/null || true)" == 'ssh://git@ssh.github.com:443/BabyofStar/GraphMAE2.git' ]]; then
  printf 'Checking course_backup main; enter the existing SSH key passphrase if prompted.\n'
  GRAPHMAE2_SSH='ssh -i /home/jovyan/.ssh/id_ed25519_graphmae2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o NumberOfPasswordPrompts=1'
  if GIT_SSH_COMMAND="$GRAPHMAE2_SSH" git -C "$PROJECT" ls-remote --heads course_backup refs/heads/main > "$OUT/remote_main.txt" 2> "$OUT/remote_main_stderr.txt"; then
    printf 'REMOTE_MAIN_READ=PASS\n'
  else
    printf 'REMOTE_MAIN_READ=FAILED; other preparation results preserved.\n'
  fi
  /home/jovyan/work/envs/graphmae2/bin/python -B - "$OUT" <<'PY'
from pathlib import Path
import json,sys
p=Path(sys.argv[1]);target=p/'github_prepare_return.zip'
raw=(p/'remote_main.txt').read_text('utf-8',errors='replace').strip()
valid=len(raw.split())==2 and len(raw.split()[0])==40 and raw.split()[1]=='refs/heads/main'
result=dict(remote='course_backup',target_branch='main',sha=raw.split()[0] if valid else None,
            status='PASS' if valid else 'UNCONFIRMED',private_key_or_token_contents_saved=False)
(p/'remote_main_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
PY
fi
/home/jovyan/work/envs/graphmae2/bin/python -B - "$OUT" <<'PY'
from pathlib import Path
import json,sys,zipfile,hashlib
p=Path(sys.argv[1]);target=p/'github_prepare_return.zip';temporary=p/'github_prepare_return.finalizing'
files=sorted(p.glob('*.json'))
manifest=dict(files=[dict(path=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files])
with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED) as archive:
    for f in files:archive.write(f,f.name)
    archive.writestr('return_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(temporary) as archive:assert archive.testzip() is None
temporary.replace(target)
print('RETURN_FILE='+str(target))
PY
