"""Targeted L0 resource followup and existing J12/P12 evidence export.

Uses only Python standard library. No training, .pt loading, environment changes,
memory reclamation, process termination, or data download. Network probes use HEAD.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

J_BATCH='j_development12_k72xhtjv'
LEAVES={
    'batch_status.json','batch.log','plan.json','historical_anchors.json','validated_runs.json',
    'validation_summary.json','queue_identity.json','manifest.json','stage_receipt.json','launch_receipt.json',
    'status.json','metrics.json','effective_args.json','environment.json','resources.json',
    'source.tar','source_sha256.json','train.log','reload.log','mask_stats.jsonl','mask_feedback.jsonl',
    'mask_node_statistics.npz','pair_plan.json','joint_trace.npz','structural_gate.json','tests.log',
    'audit.json','audit_receipt.json','evidence_sha256.json','audit_failure.json','P12独立审计.md'}
CODE_TOOLS={'graphmae2_v5_joint_dev1','graphmae2_p12_audit'}
NETWORK_HOSTS={'snap.stanford.edu','cloud.tsinghua.edu.cn'}


def text_file(path, limit=512*1024):
    try:
        with Path(path).open('rb') as f:data=f.read(limit+1)
        if len(data)>limit:return {'error':'size_limit'}
        return data.decode('utf-8',errors='replace')
    except OSError as e:return {'error':type(e).__name__}


def cmd(argv, timeout=12, secret_error=False):
    if not shutil.which(argv[0]):return {'available':False}
    try:
        p=subprocess.run(argv,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0','GIT_TERMINAL_PROMPT':'0'},
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,errors='replace',timeout=timeout)
        return {'available':True,'exit_code':p.returncode,'stdout':p.stdout[:1000000],
                'stderr':('omitted_for_credential_safety' if p.stderr else '') if secret_error else p.stderr[:3000]}
    except (OSError,subprocess.TimeoutExpired) as e:return {'available':True,'error':type(e).__name__}


def numbers(value):
    if not isinstance(value,str):return {}
    result={}
    for line in value.splitlines():
        fields=line.split()
        if len(fields)==2:
            try:result[fields[0]]=int(fields[1])
            except ValueError:pass
    return result


def memory_snapshot():
    base=Path('/sys/fs/cgroup')
    names=['memory.current','memory.max','memory.stat','memory.events','memory.events.local',
           'memory.pressure','memory.peak','memory.swap.current','memory.swap.max','cpu.max','cpu.stat','cpu.pressure',
           'pids.current','pids.max','cgroup.procs']
    raw={name:text_file(base/name) for name in names}
    result={'time_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'raw':raw,
            'memory_stat':numbers(raw['memory.stat']),'memory_events':numbers(raw['memory.events'])}
    try:
        limit=int(raw['memory.max'].strip());current=int(raw['memory.current'].strip())
        result.update(limit_GiB=limit/1024**3,current_GiB=current/1024**3,
                      raw_accounting_gap_MiB=(limit-current)/1024**2)
    except (ValueError,AttributeError):pass
    try:
        quota,period=raw['cpu.max'].split()
        result['cpu_quota_equivalent_cores']=None if quota=='max' else int(quota)/int(period)
    except (ValueError,AttributeError):pass
    result['interpretation']='Raw accounting gap is not usable-memory capacity. file includes cache/tmpfs/shared memory; no automatic subtraction or cache dropping.'
    return result


def own_process_memory():
    result=[];proc=Path('/proc')
    if not proc.is_dir() or not hasattr(os,'getuid'):return result
    with os.scandir(proc) as iterator:
        for i,entry in enumerate(iterator):
            if i>20000:break
            if not entry.name.isdigit():continue
            fields={}
            raw=text_file(Path(entry.path)/'status',64*1024)
            if not isinstance(raw,str):continue
            for line in raw.splitlines():
                if ':' in line:
                    key,val=line.split(':',1)
                    if key in {'Name','Uid','VmRSS','RssAnon','RssFile','RssShmem','VmSwap','Threads'}:fields[key]=val.strip()
            if fields.get('Uid','').split()[:1]!=[str(os.getuid())]:continue
            fields['pid']=int(entry.name);result.append(fields)
    return sorted(result,key=lambda r:int(r.get('VmRSS','0 kB').split()[0]),reverse=True)[:50]


def existing_lock_evidence(project):
    path=project/'outputs/.formal_record_v1.lock'
    result={'path':str(path),'lock_acquired_by_this_tool':False,'idle_established':False}
    try:
        st=path.stat();result['inode']=st.st_ino
        token=f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}'
        raw=text_file('/proc/locks',1024*1024)
        result['matching_proc_lock_lines']=[line for line in raw.splitlines() if token in line.split()] if isinstance(raw,str) else raw
        result['note']='Absence of a visible lock does not establish GPU idleness or exclude jobs outside this namespace.'
    except OSError as e:result['error']=type(e).__name__
    return result


class AllowedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        if urllib.parse.urlsplit(newurl).hostname not in NETWORK_HOSTS:
            raise urllib.error.HTTPError(req.full_url,code,'redirect_host_not_allowed',headers,fp)
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def head_probe(url):
    parts=urllib.parse.urlsplit(url)
    if parts.hostname not in NETWORK_HOSTS or parts.username or parts.password or parts.scheme not in {'https','http'}:
        return {'error':'URL_not_in_official_allowlist'}
    if parts.scheme=='http':url=urllib.parse.urlunsplit(('https',parts.netloc,parts.path,parts.query,''))
    try:
        request=urllib.request.Request(url,method='HEAD',headers={'User-Agent':'GraphMAE2-L0-read-only-check'})
        with urllib.request.build_opener(AllowedRedirects()).open(request,timeout=12) as response:
            return {'url':url,'method':'HEAD','status':response.status,'final_url':response.url,
                    'content_length':response.headers.get('Content-Length'),'content_type':response.headers.get('Content-Type'),
                    'body_downloaded':False}
    except urllib.error.HTTPError as e:return {'url':url,'status':e.code,'error':'HTTPError','body_downloaded':False}
    except (OSError,urllib.error.URLError) as e:return {'url':url,'error':type(e).__name__,'body_downloaded':False}


def ogb_metadata():
    try:
        spec=importlib.util.find_spec('ogb')
        if spec is None or spec.origin is None:return {'error':'ogb_package_not_found'}
        path=Path(spec.origin).parent/'nodeproppred/master.csv'
        value=text_file(path)
        if not isinstance(value,str):return {'error':'metadata_not_readable'}
        rows=list(csv.reader(io.StringIO(value)));idx=rows[0].index('ogbn-arxiv')
        return {'metadata_path':str(path),'arxiv':{row[0]:row[idx] for row in rows[1:] if len(row)>idx}}
    except (OSError,ValueError) as e:return {'error':type(e).__name__}


def safe_file(project,path):
    rel=path.relative_to(project)
    if any((project.joinpath(*rel.parts[:i])).is_symlink() for i in range(1,len(rel.parts)+1)):return False
    return path.is_file() and path.resolve().is_relative_to(project.resolve())


def evidence_paths(project):
    roots=[project/'outputs'/J_BATCH,project/'graphmae2_v5_joint_dev1',project/'graphmae2_p12_audit']
    outputs=project/'outputs'
    if outputs.is_dir():
        roots.extend(p for p in outputs.iterdir() if p.is_dir() and p.name.startswith(('j_dev1_stage1_','p12_independent_audit_')))
    selected=set()
    for root in roots:
        if not root.is_dir() or root.is_symlink():continue
        for folder,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[name for name in dirs if name not in {'__pycache__','.git','checkpoints'} and not (Path(folder)/name).is_symlink()]
            for name in files:
                p=Path(folder)/name;rel=p.relative_to(project)
                if name=='test_metrics.json' or p.suffix in {'.pt','.pth','.pkl','.pickle'}:continue
                keep=name in LEAVES or name.startswith(('diagnostic_','reload_')) and name.endswith('.log')
                keep=keep or ('queue_snapshot' in rel.parts and p.suffix in {'.py','.json','.sh'})
                keep=keep or (rel.parts[0] in CODE_TOOLS and p.suffix in {'.py','.json'})
                if keep and safe_file(project,p):selected.add(p)
    return sorted(selected)


def export_evidence(project,out,tree_text='',max_total=128*1024**2):
    tree={}
    for line in tree_text.splitlines():
        if '\t' not in line:continue
        info,name=line.split('\t',1);fields=info.split()
        if len(fields)==3 and fields[1]=='blob':tree[name]=fields[2]
    files=evidence_paths(project);records=[];skipped=[];total=0
    archive=out/'existing_J12_P12_evidence.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in files:
            rel=p.relative_to(project).as_posix();stat=p.stat()
            if stat.st_size>16*1024**2 or total+stat.st_size>max_total:
                skipped.append({'path':rel,'bytes':stat.st_size,'reason':'bounded_export_size_budget'});continue
            sha=hashlib.sha256();blob=hashlib.sha1(b'blob '+str(stat.st_size).encode()+b'\0')
            with p.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):sha.update(block);blob.update(block)
            before=(stat.st_size,stat.st_mtime_ns)
            z.write(p,'evidence/'+rel,compress_type=zipfile.ZIP_STORED if p.suffix=='.npz' else zipfile.ZIP_DEFLATED)
            after=p.stat()
            if before!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Evidence changed during export: '+rel)
            expected=tree.get(rel)
            records.append({'path':rel,'bytes':stat.st_size,'sha256':sha.hexdigest(),'git_blob_sha1':blob.hexdigest(),
                            'HEAD_blob_sha1':expected,'matches_HEAD':blob.hexdigest()==expected if expected else None})
            total+=stat.st_size
        required=[]
        for method in ['S','NA','NR','NC']:
            for seed in range(3):
                prefix=f'outputs/{J_BATCH}/{J_BATCH}_cora_{method}_s{seed}/'
                for leaf in ['status.json','metrics.json','effective_args.json','environment.json','source.tar',
                             'source_sha256.json','pair_plan.json','joint_trace.npz','mask_node_statistics.npz','train.log']:
                    required.append(prefix+leaf)
        present={r['path'] for r in records}
        manifest={'purpose':'Existing evidence transfer; not an independent J12 audit','files':records,'skipped':skipped,
                  'uncompressed_bytes':total,'required_J12_members_not_in_export':[p for p in required if p not in present],
                  'test_score_contents_opened':False,'model_weights_opened':False,'training_started':False,
                  'HEAD_inventory_scope':'Selected existing source/tool/experiment paths; file tracking does not itself establish remote backup.'}
        z.writestr('export_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (out/'export_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    digest=hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    (out/'existing_J12_P12_evidence.zip.sha256').write_text(digest.hexdigest()+'  '+archive.name+'\n',encoding='ascii')
    return {'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest.hexdigest(),'exported_files':len(records),
            'skipped_files':len(skipped),'required_J12_members_missing':len(manifest['required_J12_members_not_in_export']),
            'independent_audit_pass':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='/home/jovyan/work/GraphMAE2')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--skip-evidence',action='store_true')
    args=parser.parse_args();project=Path(args.project).expanduser().resolve();out=Path(args.output_dir).expanduser().resolve()
    if not project.is_dir() or not out.is_dir():parser.error('Existing project and fresh output directory required')
    if (out/'followup.json').exists():parser.error('Output already contains followup.json; use a fresh directory')
    if out.is_relative_to(project/'outputs'/J_BATCH):parser.error('Do not write into an existing evidence batch')
    snapshots=[];gpu=[]
    for i in range(3):
        snapshots.append(memory_snapshot())
        gpu.append(cmd(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']))
        if i<2:time.sleep(1)
    meta=ogb_metadata();urls=['https://cloud.tsinghua.edu.cn/d/64f859f389ca43eda472/']
    data_url=meta.get('arxiv',{}).get('url')
    if data_url:urls.append(data_url)
    network={'skipped':True}
    if not args.offline:
        with ThreadPoolExecutor(max_workers=2) as pool:network={'probes':list(pool.map(head_probe,urls))}
    head=cmd(['git','-C',str(project),'rev-parse','HEAD'])
    branch=cmd(['git','-C',str(project),'symbolic-ref','--short','HEAD'])
    branch_name=branch.get('stdout','').strip()
    remote={'skipped':True}
    if not args.offline and branch_name:
        remote=cmd(['git','-C',str(project),'ls-remote','--heads','origin','refs/heads/'+branch_name],secret_error=True)
        if remote.get('exit_code')==0:
            rows=remote.get('stdout','').splitlines()
            remote['same_branch_remote_HEAD_matches_local']=any(line.split()[:1]==[head.get('stdout','').strip()] for line in rows)
    tree=cmd(['git','-C',str(project),'ls-tree','-r','--full-tree','HEAD','--','outputs/'+J_BATCH,
              'graphmae2_v5_joint_dev1','graphmae2_p12_audit','outputs/p12_independent_audit_oci0zhx7'])
    (out/'selected_HEAD_tree.txt').write_text(tree.get('stdout',''),encoding='utf-8')
    result={'stage':'L0 targeted followup','status':'COLLECTED_NOT_A_TRAINING_READINESS_PASS',
            'time_beijing':dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
            'memory_samples':snapshots,'process_memory_without_command_arguments':own_process_memory(),
            'gpu_samples':gpu,'gpu_process_snapshot':cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']),
            'common_lock_evidence':existing_lock_evidence(project),'ogb_metadata':meta,'network_HEAD_probes':network,
            'git_HEAD':head,'git_branch':branch,'same_branch_remote_check':remote,'selected_HEAD_tree':str(out/'selected_HEAD_tree.txt'),
            'personal_disk_quota_established':False,'training_started':False,'environment_upgraded':False,
            'cache_dropped':False,'processes_killed':False,
            'next_stage':'Review memory breakdown and data/cluster accessibility, then prepare dependencies/data in isolated paths as needed. No full training budget selected.'}
    if not args.skip_evidence:result['evidence_export']=export_evidence(project,out,tree.get('stdout',''))
    (out/'followup.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('FOLLOWUP_JSON='+str(out/'followup.json'))
    if not args.skip_evidence:print('EXISTING_EVIDENCE_ZIP='+result['evidence_export']['path'])
    print('Targeted collection only. No training, data download, cache clearing or dependency installation.')


if __name__=='__main__':main()
