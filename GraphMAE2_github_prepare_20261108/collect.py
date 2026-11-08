"""Read-only Git/source/weight inventory before authorized GitHub synchronization.

No training, install, staging, commit, merge, push, Release mutation or model load.
Only this preparation's output directory is written. Authentication values stay in
process memory; results record presence/identity/status, never token contents.
"""
import argparse
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

PROJECT=Path('/home/jovyan/work/GraphMAE2')
ARTIFACTS=Path('/home/jovyan/work/GraphMAE2_artifacts')
BACKUP_URL='ssh://git@ssh.github.com:443/BabyofStar/GraphMAE2.git'
REPO='BabyofStar/GraphMAE2'
EXCLUDED_DIRS={'.git','__pycache__','.ipynb_checkpoints','.venv','venv','preprocess_env',
               'node_modules','pip_cache','matplotlib_cache','native_source','.pytest_cache'}
WEIGHT_SUFFIXES={'.pt','.pth','.ckpt','.safetensors'}
TEXT_SUFFIXES={'.py','.sh','.json','.jsonl','.md','.txt','.csv','.yaml','.yml','.log','.patch','.ini','.cfg','.toml'}
SECRET_RE=re.compile(rb'github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----|https?://[^/@\s]+:[^/@\s]+@')


def require(ok,message):
    if not ok:raise RuntimeError(message)


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def command(argv,timeout=20,env=None):
    try:
        result=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,
                              env={**os.environ,'GIT_OPTIONAL_LOCKS':'0',**(env or {})})
        return dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
    except (OSError,subprocess.TimeoutExpired) as error:
        return dict(error=type(error).__name__)


def git(*argv,timeout=20):return command(['git','-C',str(PROJECT),*argv],timeout=timeout)
def decoded(result):return result.get('stdout',b'').decode('utf-8',errors='replace').strip()


def hash_file(path,git_blob=False):
    path=Path(path);before=path.stat();digest=hashlib.sha256()
    blob=hashlib.sha1(b'blob '+str(before.st_size).encode()+b'\0') if git_blob else None
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            digest.update(block)
            if blob is not None:blob.update(block)
    after=path.stat()
    require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'File changed during hashing: '+str(path))
    item=dict(bytes=after.st_size,sha256=digest.hexdigest(),mtime_ns=after.st_mtime_ns)
    if blob is not None:item['git_blob_sha1']=blob.hexdigest()
    return item


def redact_url(value):
    try:
        parts=urllib.parse.urlsplit(value)
        if parts.scheme in ['http','https']:
            host=parts.hostname or ''
            if parts.port:host+=':'+str(parts.port)
            return urllib.parse.urlunsplit((parts.scheme,host,parts.path,'',''))
    except ValueError:pass
    return value if value.startswith(('ssh://git@','git@')) else '<unexpected remote format; value withheld>'


def exclusion(relative,tracked):
    if relative in tracked:return None
    parts=Path(relative).parts
    for part in parts:
        if part in EXCLUDED_DIRS or part=='tmp':return 'generated environment/cache: '+part
    if parts and (parts[0].startswith('GraphMAE2_github_prepare_') or parts[0].startswith('GraphMAE2_github_sync_')):
        return 'this synchronization helper, separately versioned'
    if parts and parts[0].endswith('.zip') and parts[0].startswith('GraphMAE2_github_'):
        return 'this synchronization helper archive'
    return None


def suspicious_name(relative):
    name=Path(relative).name.lower()
    return name=='.env' or name.startswith('.env.') or name in ['.netrc','.npmrc','.pypirc','credentials.json'] or name.endswith(('.pem','.key')) or name.startswith(('id_rsa','id_ed25519'))


def source_inventory(project,tracked):
    candidates=[];excluded=[];issues=[];all_files=0
    for folder,dirs,files in os.walk(project,followlinks=False):
        parent=Path(folder)
        for directory in list(dirs):
            path=parent/directory;relative=path.relative_to(project).as_posix()
            if directory=='.git':
                if path!=project/'.git':issues.append(dict(path=relative,reason='nested Git repository'))
                dirs.remove(directory);continue
            if path.is_symlink():issues.append(dict(path=relative,reason='directory symlink not followed'));dirs.remove(directory);continue
            reason=exclusion(relative,tracked)
            # A directory containing tracked files is never silently omitted.
            if reason and not any(p.startswith(relative+'/') for p in tracked):
                excluded.append(dict(path=relative,reason=reason));dirs.remove(directory)
        for name in files:
            path=parent/name;relative=path.relative_to(project).as_posix();all_files+=1
            reason=exclusion(relative,tracked)
            if reason:excluded.append(dict(path=relative,reason=reason));continue
            if path.is_symlink():issues.append(dict(path=relative,reason='file symlink not dereferenced'));continue
            if not path.is_file():issues.append(dict(path=relative,reason='not a regular file'));continue
            require(path.resolve().is_relative_to(project.resolve()),'Candidate escaped project root')
            if suspicious_name(relative):
                issues.append(dict(path=relative,reason='credential-like filename; content not read'))
                excluded.append(dict(path=relative,reason='credential-like file requires review before public synchronization'))
                continue
            record=dict(path=relative,tracked=relative in tracked,mode='100755' if path.stat().st_mode&stat.S_IXUSR else '100644',**hash_file(path,True))
            if record['bytes']>100*1024**2:issues.append(dict(path=relative,reason='exceeds regular Git single-file size; keep out of source commit'))
            if path.suffix.lower() in TEXT_SUFFIXES and record['bytes']<=8*1024**2:
                if SECRET_RE.search(path.read_bytes()):issues.append(dict(path=relative,reason='credential-like text; matching value withheld'))
                record['text_pattern_scan']='complete'
            else:record['text_pattern_scan']='not_text_or_above_text_scan_limit'
            candidates.append(record)
            if len(candidates)%100==0:print(f'SOURCE_FILES_HASHED={len(candidates)}',flush=True)
    missing=sorted(p for p in tracked if not (project/p).exists())
    return dict(files=sorted(candidates,key=lambda r:r['path']),excluded=excluded,issues=issues,missing_tracked=missing,
                regular_candidate_bytes=sum(r['bytes'] for r in candidates),visited_files=all_files,
                scan_limit='Filename patterns and bounded plaintext scan only; not a guarantee for arbitrary binary/archive secrets')


def weight_inventory(artifacts,expected):
    files=[];issues=[];excluded=[];roots=[]
    require(artifacts.is_dir(),'Artifact root is absent')
    checkpoint=artifacts/'checkpoints'
    if checkpoint.is_symlink():issues.append(dict(path=str(checkpoint),reason='checkpoint root symlink not followed'))
    elif checkpoint.is_dir():roots.append((checkpoint,'all_checkpoint_directory_files'))
    else:issues.append(dict(path=str(checkpoint),reason='expected small-graph checkpoints directory absent'))
    for child in sorted(artifacts.iterdir()):
        if child.is_symlink():issues.append(dict(path=str(child),reason='artifact symlink not followed'));continue
        if child.is_dir() and child.name.startswith(('arxiv_A_','arxiv_l1_measurement_')):
            roots.append((child,'arxiv_model_files'))
        elif child.name!='checkpoints':excluded.append(dict(path=str(child),reason='data/clustering or unclassified artifact root; not assumed to be model weights'))
    seen=set()
    for root,kind in roots:
        for folder,dirs,names in os.walk(root,followlinks=False):
            for directory in list(dirs):
                target=Path(folder)/directory
                if target.is_symlink():issues.append(dict(path=str(target),reason='weight directory symlink'));dirs.remove(directory)
            for name in sorted(names):
                path=Path(folder)/name
                if path.is_symlink():issues.append(dict(path=str(path),reason='weight symlink not dereferenced'));continue
                if not path.is_file():continue
                if kind=='arxiv_model_files' and path.suffix.lower() not in WEIGHT_SUFFIXES:
                    excluded.append(dict(path=str(path),reason='arxiv embedding/cache rather than model weight'));continue
                require(path.resolve().is_relative_to(artifacts.resolve()),'Weight escaped asset root')
                if path in seen:continue
                seen.add(path);record=dict(path=str(path),archive_path=path.relative_to(artifacts).as_posix(),category=kind,**hash_file(path))
                if str(path) in expected:
                    require(record['sha256']==expected[str(path)]['sha256'],'Audited arxiv weight differs: '+str(path))
                    record['matches_A_receipt']=True
                files.append(record)
                if len(files)%20==0:print(f'WEIGHTS_HASHED={len(files)} BYTES={sum(r["bytes"] for r in files)}',flush=True)
    for path in expected:
        if path not in {r['path'] for r in files}:issues.append(dict(path=path,reason='audited A model weight missing from inventory'))
    return dict(files=sorted(files,key=lambda r:r['archive_path']),issues=issues,excluded=excluded,
                model_file_count=len(files),total_bytes=sum(r['bytes'] for r in files),
                scope='All files in checkpoints plus model files in arxiv_A_* and arxiv_l1_measurement_*; excludes raw data, clustering, full embeddings and environment binaries')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):return None


def api_get(path,token=None):
    require(path.startswith('/') and not path.startswith('//'),'Invalid API path')
    headers={'Accept':'application/vnd.github+json','User-Agent':'GraphMAE2-backup-preparation','X-GitHub-Api-Version':'2022-11-28'}
    if token:headers['Authorization']='Bearer '+token
    request=urllib.request.Request('https://api.github.com'+path,headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect()).open(request,timeout=15) as response:
            raw=response.read(2*1024**2+1);require(len(raw)<=2*1024**2,'API metadata response exceeded limit')
            return dict(ok=True,status=response.status,data=json.loads(raw))
    except urllib.error.HTTPError as error:return dict(ok=False,status=error.code)
    except (OSError,ValueError) as error:return dict(ok=False,error=type(error).__name__)


def github_auth_check():
    token=None;source=None;gh=shutil.which('gh')
    for name in ['GH_TOKEN','GITHUB_TOKEN']:
        if os.environ.get(name):token=os.environ[name];source='existing_'+name;break
    if token is None and gh:
        # Captured only in memory. Never record this command's stdout or stderr.
        value=command([gh,'auth','token','--hostname','github.com'],timeout=10)
        if value.get('exit_code')==0:
            token=value['stdout'].decode('utf-8',errors='strict').strip();source='existing_gh_login'
    result=dict(gh_available=bool(gh),existing_token_present=bool(token),credential_source=source,
                token_saved=False,token_printed=False,release_upload_authenticated=False)
    if token:
        user=api_get('/user',token)
        result['user_api_status']={k:v for k,v in user.items() if k!='data'}
        if user.get('ok'):result['authenticated_login']=user['data'].get('login')
        repository=api_get('/repos/'+REPO,token)
        result['repository_api_status']={k:v for k,v in repository.items() if k!='data'}
        if repository.get('ok'):
            data=repository['data'];result['repository']=dict(full_name=data.get('full_name'),private=data.get('private'),permissions=data.get('permissions'),default_branch=data.get('default_branch'))
            result['release_upload_authenticated']=bool(user.get('ok') and data.get('full_name','').lower()==REPO.lower() and data.get('permissions',{}).get('push'))
            result['write_permission_limit']='Read-only API checks cannot prove fine-grained Contents:write until an authorized upload is attempted'
    else:
        public=api_get('/repos/'+REPO)
        result['public_repository_api_status']={k:v for k,v in public.items() if k!='data'}
        result['next_auth_if_needed']='Use existing GitHub CLI login or a narrowly scoped local credential; never paste tokens/private keys into chat or result files'
    token=None
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();out=args.output_dir.resolve();require(out.is_dir(),'Missing unique preparation directory')
    report=dict(stage='GitHub source synchronization and Release weight-backup preparation',time_beijing=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
         destination='GitHub BabyofStar/GraphMAE2: main + separate Release assets',project=str(PROJECT),artifact_root=str(ARTIFACTS),
         git_mutated=False,github_mutated=False,weights_archived=False,independent_backup_complete=False)
    started=time.monotonic()
    try:
        require(PROJECT.resolve()==PROJECT and ARTIFACTS.resolve()==ARTIFACTS,'Unexpected project/artifact symlink root')
        root=git('rev-parse','--show-toplevel');require(root.get('exit_code')==0 and decoded(root)==str(PROJECT),'Unexpected Git project')
        branch=decoded(git('branch','--show-current'));head=decoded(git('rev-parse','HEAD'));report['branch']=branch;report['HEAD']=head
        require(branch=='feat/structure-curriculum','Development branch differs from the established guide')
        tracked_result=git('ls-files','-z');require(tracked_result.get('exit_code')==0,'Cannot inventory tracked paths')
        tracked={raw.decode('utf-8',errors='strict') for raw in tracked_result['stdout'].split(b'\0') if raw}
        status=git('status','--porcelain=v1','-z','--untracked-files=normal')
        save(out/'git_status.json',dict(status_records=[p.decode('utf-8',errors='replace') for p in status.get('stdout',b'').split(b'\0') if p],
             staged_diff_stat=decoded(git('diff','--cached','--stat')),unstaged_diff_stat=decoded(git('diff','--stat')),
             merge_in_progress=(PROJECT/'.git/MERGE_HEAD').exists(),unmerged_paths=decoded(git('diff','--name-only','--diff-filter=U'))))
        remotes={}
        for kind,args2 in [('fetch',['remote','get-url','--all','course_backup']),('push',['remote','get-url','--push','--all','course_backup'])]:
            value=git(*args2);urls=decoded(value).splitlines()
            remotes[kind]=dict(exit_code=value.get('exit_code'),urls_redacted=[redact_url(u) for u in urls],exact_expected_match=urls==[BACKUP_URL])
        report['course_backup']=remotes;report['target_remote_branch']='refs/heads/main'
        report['source_inventory_started']=True;save(out/'preparation.json',report)
        print('STEP=source_inventory',flush=True);sources=source_inventory(PROJECT,tracked);save(out/'source_inventory.json',sources)
        expected_source=json.loads((Path(__file__).parent/'expected_experiment_source.json').read_text('utf-8'))
        observed={row['path']:row['sha256'] for row in sources['files']}
        report['audited_source_differences']=[name for name,digest in expected_source.items() if observed.get(name)!=digest]
        expected=json.loads((Path(__file__).parent/'expected_A_weights.json').read_text('utf-8'))
        print('STEP=external_weight_inventory',flush=True);weights=weight_inventory(ARTIFACTS,expected);save(out/'weight_inventory.json',weights)
        print('STEP=existing_GitHub_API_login_check',flush=True);auth=github_auth_check();save(out/'github_auth_check.json',auth)
        report.update(source_candidate_count=len(sources['files']),source_candidate_bytes=sources['regular_candidate_bytes'],
                      source_issues=len(sources['issues']),missing_tracked_count=len(sources['missing_tracked']),
                      weight_count=weights['model_file_count'],weight_bytes=weights['total_bytes'],weight_issues=len(weights['issues']),
                      existing_API_login_ready=auth['release_upload_authenticated'],
                      staging_disk=shutil.disk_usage(out)._asdict(),
                      private_key_present=(Path('/home/jovyan/.ssh/id_ed25519_graphmae2').is_file()),
                      git_object_format=decoded(git('rev-parse','--show-object-format')),
                      status='PREPARATION_COLLECTED_REVIEW_REQUIRED')
    except Exception as error:report.update(status='STOPPED_PRESERVED',error_type=type(error).__name__,error=str(error))
    report['wall_seconds']=time.monotonic()-started;save(out/'preparation.json',report)
    target=out/'github_prepare_return.zip'
    manifest=[]
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.glob('*.json')):
            require(path.stat().st_size<32*1024**2,'Unexpected metadata size')
            archive.write(path,path.name);manifest.append(dict(path=path.name,bytes=path.stat().st_size,sha256=hash_file(path)['sha256']))
        archive.writestr('return_manifest.json',json.dumps(dict(files=manifest),ensure_ascii=False,indent=2)+'\n')
    print('PREPARATION_STATUS='+report['status'],flush=True)
    if report['status']=='STOPPED_PRESERVED':print('RETURN_FILE='+str(target),flush=True)
    else:print('METADATA_COLLECTED='+str(out),flush=True)
    return 1 if report['status']=='STOPPED_PRESERVED' else 0


if __name__=='__main__':raise SystemExit(main())
