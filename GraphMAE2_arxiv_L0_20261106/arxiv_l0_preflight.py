"""Collect server prerequisites for GraphMAE2 ogbn-arxiv. No training or downloads.

Standard library only. Torch is queried in an isolated subprocess for GPU properties;
no model is constructed and no tensor workload is run. Existing .pt files are not loaded.
"""
import argparse
import ast
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time


def read_text(path, limit=1024*1024):
    try:
        with Path(path).open('rb') as stream:
            raw=stream.read(limit+1)
        if len(raw)>limit:return {'error':'size_limit','path':str(path)}
        return raw.decode('utf-8',errors='replace')
    except OSError as error:return {'error':type(error).__name__,'path':str(path)}


def command(argv, timeout=10, cwd=None):
    if not shutil.which(argv[0]):return {'available':False,'program':argv[0]}
    try:
        result=subprocess.run(argv,cwd=cwd,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0','PYTHONDONTWRITEBYTECODE':'1'},
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,errors='replace',timeout=timeout)
        return {'available':True,'exit_code':result.returncode,'stdout':result.stdout[:14000], 'stderr':result.stderr[:3000]}
    except (OSError,subprocess.TimeoutExpired) as error:return {'available':True,'error':type(error).__name__}


def file_info(path, hash_limit=64*1024**2):
    path=Path(path)
    try:
        info=path.stat()
        result={'path':str(path),'bytes':info.st_size,'mtime_utc':dt.datetime.fromtimestamp(info.st_mtime,dt.timezone.utc).isoformat()}
        if path.is_file() and info.st_size<=hash_limit:
            digest=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
            result['sha256']=digest.hexdigest()
        else:result['sha256_status']='not_hashed_in_L0_size_budget'
        return result
    except OSError as error:return {'path':str(path),'error':type(error).__name__}


def scan_arxiv_files(roots, seconds=8):
    """Bounded metadata scan: no symlink traversal and no reading cache tensors."""
    start=time.monotonic();stack=[(p,0) for p in roots if p.is_dir() and not p.is_symlink()]
    found=[];seen=set();entries=0;truncated=False
    while stack:
        folder,depth=stack.pop()
        try:
            resolved=folder.resolve()
            if resolved in seen:continue
            seen.add(resolved)
            with os.scandir(folder) as iterator:
                for item in iterator:
                    entries+=1
                    if entries>12000 or time.monotonic()-start>seconds:
                        truncated=True;break
                    p=Path(item.path)
                    if item.is_symlink():continue
                    if item.is_dir(follow_symlinks=False) and depth<4:stack.append((p,depth+1))
                    elif item.is_file(follow_symlinks=False) and 'arxiv' in str(p).lower():
                        stat=item.stat(follow_symlinks=False)
                        found.append({'path':str(p),'bytes':stat.st_size})
                        if len(found)>=200:truncated=True;break
        except OSError:continue
        if truncated:break
    return {'files':found,'entries_examined':entries,'truncated':truncated,'contents_loaded':False,
            'note':'Presence/size only; data/split/cluster content and complete fingerprints require later validation.'}


def cgroup_memory():
    result=[]
    if sys.platform!='linux':return {'supported':False}
    memberships=read_text('/proc/self/cgroup')
    candidates=[Path('/sys/fs/cgroup')]
    if isinstance(memberships,str):
        for line in memberships.splitlines():
            if line.startswith('0::'):
                rel=line.split('::',1)[1].lstrip('/')
                candidate=Path('/sys/fs/cgroup')/rel
                if '..' not in Path(rel).parts and candidate.is_relative_to(Path('/sys/fs/cgroup')):
                    candidates.extend([candidate,*[p for p in candidate.parents if p.is_relative_to(Path('/sys/fs/cgroup'))]])
    candidates.extend([Path('/sys/fs/cgroup/memory')])
    for folder in dict.fromkeys(candidates):
        values={}
        for name in ['memory.max','memory.current','memory.high','memory.swap.max','cpu.max','pids.max',
                     'memory.limit_in_bytes','memory.usage_in_bytes','memory.soft_limit_in_bytes']:
            p=folder/name
            if p.is_file():values[name]=read_text(p)
        if values:result.append({'directory':str(folder),'values':values})
    return {'supported':True,'membership':memberships,'hierarchy_records':result,
            'note':'Host MemAvailable and df capacity do not establish a personal RAM/disk quota.'}


def static_configuration(project):
    result={'actual_namespace_executed':False}
    code=read_text(project/'utils.py')
    defaults={}
    if isinstance(code,str):
        try:
            for node in ast.walk(ast.parse(code)):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
                    names=[a.value for a in node.args if isinstance(a,ast.Constant) and isinstance(a.value,str)]
                    if not names:continue
                    for kw in node.keywords:
                        if kw.arg=='default':
                            try:defaults[names[0]]=ast.literal_eval(kw.value)
                            except (ValueError,TypeError):pass
        except SyntaxError as error:result['argument_ast_error']=str(error)
    result['selected_cli_defaults']={k:v for k,v in defaults.items() if k in {
        '--dataset','--data_dir','--sampling_method','--ego_graph_file_path','--batch_size','--batch_size_f','--seeds'}}
    result['arxiv_yaml_text']=read_text(project/'configs/ogbn-arxiv.yaml')
    for name,pattern in [
        ('lc_sampler_support_lines',r'loader_type|NotImplementedError|num_workers'),
        ('probe_internal_budget_lines',r'batch_size =|seeds =|setup_eval_dataloder'),
        ('large_pretrain_seed_lines',r'set_random_seed|model_name =|setup_training_data|prepare.*mask|curriculum')]:
        relative={'lc_sampler_support_lines':'datasets/lc_sampler.py','probe_internal_budget_lines':'models/finetune.py',
                  'large_pretrain_seed_lines':'main_large.py'}[name]
        text=read_text(project/relative)
        result[name]=[{'line':i,'text':line.strip()} for i,line in enumerate(text.splitlines(),1) if re.search(pattern,line)][:35] if isinstance(text,str) else text
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='/home/jovyan/work/GraphMAE2')
    parser.add_argument('--reference',default=str(Path(__file__).with_name('reference_identity.json')))
    parser.add_argument('--data-dir',default=None)
    parser.add_argument('--ego-graph-file',default=None)
    parser.add_argument('--skip-runtime-probe',action='store_true')
    args=parser.parse_args();project=Path(args.project).expanduser().resolve()
    if not project.is_dir():parser.error('Project directory does not exist: '+str(project))
    reference=json.loads(Path(args.reference).read_text('utf-8-sig')) if Path(args.reference).is_file() else {}
    packages={}
    for name in ['torch','dgl','numpy','ogb','scipy','scikit-learn','localgraphclustering','PyYAML','psutil','tqdm','wandb',
                 'nvidia-cuda-runtime-cu11','nvidia-cublas-cu11','nvidia-cusparse-cu11']:
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]=None
    expected=reference.get('J55_after_sha256',{})
    large=reference.get('historical_large_files',{})
    identities={name:file_info(project/name,4*1024**2) for name in sorted(set(expected)|set(large))}
    mismatch=[name for name,digest in expected.items() if identities[name].get('sha256')!=digest]
    identity={'files':identities,'reference_J_files':len(expected),'J_reference_mismatches':mismatch,
              'historical_large_differences':[name for name,meta in large.items() if identities[name].get('sha256')!=meta['sha256']],
              'note':'Differences are reported for review, never repaired or overwritten by this script.'}
    data_dir=Path(args.data_dir).expanduser().resolve() if args.data_dir else project/'data'
    roots=[data_dir,project/'dataset',project/'lc_ego_graphs',project/'data/lc_ego_graphs']
    if args.ego_graph_file:roots.append(Path(args.ego_graph_file).expanduser().resolve().parent)
    cache=scan_arxiv_files(roots)
    clusters=[f for f in cache['files'] if Path(f['path']).suffix=='.pt' and any(t in Path(f['path']).name.lower() for t in ['ego','lc','cluster'])]
    if args.ego_graph_file:
        clusters=[file_info(Path(args.ego_graph_file).expanduser().resolve())]
    else:
        clusters=[file_info(Path(f['path'])) for f in clusters]
    batches=[]
    output_root=project/'outputs'
    if output_root.is_dir():
        with os.scandir(output_root) as iterator:
            for i,item in enumerate(iterator):
                if i>=500:break
                if not item.is_dir(follow_symlinks=False):continue
                status_file=Path(item.path)/'batch_status.json'
                if status_file.is_file():
                    value=read_text(status_file,128*1024)
                    try:value=json.loads(value) if isinstance(value,str) else value
                    except json.JSONDecodeError:value={'error':'invalid_json'}
                    batches.append({'directory':item.name,'batch_status':value})
    disk=shutil.disk_usage(project)
    runtime={'skipped':True}
    if not args.skip_runtime_probe:
        probe="""import json,torch
result={'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'available':torch.cuda.is_available(),'devices':[]}
if result['available']:
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        result['devices'].append({'index':i,'name':p.name,'visible_total_bytes':p.total_memory})
print(json.dumps(result))
"""
        runtime=command([sys.executable,'-B','-c',probe],timeout=45,cwd=str(project.parent))
        if runtime.get('exit_code')==0:
            try:runtime['parsed']=json.loads(runtime['stdout'].strip().splitlines()[-1])
            except (ValueError,IndexError):pass
    missing=[]
    if Path(sys.prefix).resolve()!=Path('/home/jovyan/work/envs/graphmae2').resolve():
        missing.append('Interpreter prefix is not the frozen graphmae2 environment')
    for required in ['torch','dgl','numpy']:
        if packages[required] is None:missing.append(required+' dependency not installed in this interpreter')
    if packages['ogb'] is None:missing.append('OGB dependency not installed in this interpreter')
    if not args.skip_runtime_probe and not runtime.get('parsed',{}).get('available'):
        missing.append('CUDA property probe failed or CUDA is not available; inspect captured stderr')
    if not clusters:missing.append('No arxiv local-clustering .pt identified in bounded candidate paths')
    if not (project/'main_large.py').is_file():missing.append('main_large.py missing')
    if not any('ogbn_arxiv' in f['path'] or 'ogbn-arxiv' in f['path'] for f in cache['files']):missing.append('OGB arxiv cache not identified in candidate paths')
    if expected and mismatch:missing.append('Current source differs from the retained J55 reference; review exact files')
    report={
        'stage':'ogbn-arxiv L0','status':'COLLECTED_REVIEW_REQUIRED_NOT_A_TRAINING_READINESS_PASS',
        'time_beijing':dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        'project':str(project),'platform':platform.platform(),'python':sys.version,'python_executable':sys.executable,'sys_prefix':sys.prefix,
        'expected_environment':'/home/jovyan/work/envs/graphmae2',
        'environment_matches_expected':Path(sys.prefix).resolve()==Path('/home/jovyan/work/envs/graphmae2').resolve(),
        'selected_environment':{k:os.environ.get(k) for k in ['CONDA_PREFIX','CUDA_VISIBLE_DEVICES','DGLBACKEND','OMP_NUM_THREADS','MKL_NUM_THREADS']},
        'packages':packages,'torch_property_probe':runtime,
        'gpu_snapshot':command(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        'gpu_process_snapshot':command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']),
        'cpu_count':os.cpu_count(),'cpu_affinity_count':len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        'host_meminfo':read_text('/proc/meminfo',128*1024),'cgroup':cgroup_memory(),
        'disk':{'total_bytes':disk.total,'used_bytes':disk.used,'filesystem_free_bytes':disk.free,'personal_quota_established':False},
        'quota_command':command(['quota','-s'],timeout=8),'git_head':command(['git','-C',str(project),'rev-parse','HEAD']),
        'git_branch':command(['git','-C',str(project),'symbolic-ref','--short','HEAD']),
        'git_tracked_status':command(['git','-C',str(project),'status','--short','--untracked-files=no']),
        'source_identity':identity,'static_large_configuration':static_configuration(project),
        'cache_scan':cache,'cluster_candidates':clusters,'batch_statuses':batches,
        'queue_process_names':command(['ps','-u',str(os.getuid()),'-o','pid,ppid,etimes,comm']) if hasattr(os,'getuid') else {'supported':False},
        'queue_idle_established':False,'identified_missing_or_review_items':missing,
        'not_performed':['training','model forward/backward','linear probing','dependency installation','data download',
                         'loading .pt or test scores','source changes','remote Git push','file deletion'],
        'next_step':'Review actual server output, resolve prerequisite gaps, then L1/L1b measurement. A/B full-training budget remains unselected.'}
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
