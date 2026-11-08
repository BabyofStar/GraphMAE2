import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

PACKAGE=Path(__file__).resolve().parent
CONTRACT=json.loads((PACKAGE/'contract.json').read_text('utf-8'))
PROJECT=Path(CONTRACT['project'])
ARTIFACTS=Path(CONTRACT['artifacts'])


def need(ok,message):
    if not ok:raise RuntimeError(message)


def now():return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def save(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'));temp.replace(path)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):value.update(block)
    return value.hexdigest()


def run(args,cwd=None,input=None,check=True):
    result=subprocess.run([str(a) for a in args],cwd=cwd,input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
    if check and result.returncode:
        message=result.stderr.decode('utf-8',errors='replace')[-12000:]
        raise RuntimeError('Command failed: '+str(args[0])+' '+str(args[1:3])+'\n'+message)
    return result


def git(cwd,*args,input=None,check=True):return run(['git',*args],cwd=cwd,input=input,check=check)
def text(result):return result.stdout.decode('utf-8',errors='strict').strip()


def job(out):
    out=Path(out).resolve();expected=Path('/home/jovyan/work/GraphMAE2_backup_staging')
    need(out.parent==expected and out.name.startswith('sync_'),'Unexpected backup staging path')
    identity_path=PACKAGE/'kit_identity.json'
    identity=json.loads(identity_path.read_text('utf-8'))
    for row in identity['files']:
        need(sha(PACKAGE/row['name'])==row['sha256'],'Execution package changed: '+row['name'])
    path=out/'job.json'
    if path.exists():
        value=json.loads(path.read_text('utf-8'));need(value['contract_sha256']==sha(PACKAGE/'contract.json'),'Backup contract changed')
        need(value['kit_identity_sha256']==sha(identity_path),'Execution package identity changed')
        return value
    value=dict(id=out.name,created_beijing=now(),repo=CONTRACT['repo'],contract_sha256=sha(PACKAGE/'contract.json'),
               kit_identity_sha256=sha(identity_path),
               tag=CONTRACT['release_tag_prefix']+out.name.removeprefix('sync_'),independent_backup_complete=False)
    save(path,value);return value
