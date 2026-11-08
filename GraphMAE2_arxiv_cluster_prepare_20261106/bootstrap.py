"""Isolated dependency preparation for upstream-compatible arxiv local clustering."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

PACKAGE=Path(__file__).resolve().parent
BASE_PY=Path('/home/jovyan/work/envs/graphmae2/bin/python')
CORE=['torch','dgl','numpy','scipy','scikit-learn']
PIP_PACKAGES=['wheel==0.45.1','networkx==2.8.8','matplotlib==3.8.4','pandas==2.2.3','plotly==5.24.1','joblib==1.4.2']


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def run(command,log,timeout=600):
    print('STEP_LOG='+str(log),flush=True)
    with Path(log).open('x',encoding='utf-8') as stream:
        result=subprocess.run([str(x) for x in command],stdout=stream,stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL,timeout=timeout,env=os.environ.copy())
    return result.returncode


def core_versions(python):
    code='import json,sys,importlib.metadata as m; print(json.dumps(dict(prefix=sys.prefix,versions={n:m.version(n) for n in '+repr(CORE)+'})))'
    return json.loads(subprocess.check_output([str(python),'-B','-c',code],text=True))


def download_release(entry,target,expected):
    url=entry['url'];require(url.startswith('https://files.pythonhosted.org/') and entry['digests']['sha256']==expected,'PyPI release identity differs')
    with urllib.request.urlopen(url,timeout=40) as response,target.open('xb') as out:
        copied=0
        while True:
            block=response.read(1024*1024)
            if not block:break
            copied+=len(block);require(copied<32*1024**2,'Dependency download size budget');out.write(block)
    require(sha(target)==expected,'Dependency download SHA256 mismatch')


def unpack_source(archive,target):
    with tarfile.open(archive,'r:gz') as tar:
        seen=set();total=0
        for item in tar:
            p=PurePosixPath(item.name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in item.name and item.name not in seen,'Unsafe dependency source path')
            require(p.parts and p.parts[0]=='localgraphclustering-0.6.1' and (item.isfile() or item.isdir()),'Dependency archive schema')
            total+=item.size;require(total<128*1024**2 and item.size<32*1024**2,'Dependency source size budget')
            seen.add(item.name);dest=target.joinpath(*p.parts);require(dest.resolve().is_relative_to(target.resolve()),'Source path escaped isolation')
            if item.isdir():dest.mkdir(parents=True,exist_ok=True)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(item) as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst)
    return target/'localgraphclustering-0.6.1'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--project',type=Path,default=Path('/home/jovyan/work/GraphMAE2'))
    args=parser.parse_args();out=args.output_dir.resolve();project=args.project.resolve()
    require(out.is_dir() and not (out/'bootstrap.json').exists(),'Use a fresh output directory')
    contract=json.loads((PACKAGE/'preprocess_contract.json').read_text('utf-8'))
    report=dict(stage='Isolated clustering preparation',time_beijing=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
                project=str(project),asset_root=contract['asset_root'],output_dir=str(out),training_started=False)
    before=None;started=time.monotonic()
    try:
        require(BASE_PY.is_file(),'Frozen graphmae2 interpreter missing')
        before=core_versions(BASE_PY);report['base_before']=before
        require(before['prefix']=='/home/jovyan/work/envs/graphmae2','Wrong base interpreter prefix')
        require(sha(project/'datasets/localclustering.py')==contract['localclustering_source_sha256'],'Upstream clustering source differs')
        asset=Path(contract['asset_root']).resolve()
        require(asset.is_dir() and sha(asset/'downloads/arxiv.zip')==contract['ogb_archive_sha256'],'Prepared raw archive identity differs')
        for item in contract['raw_files']:require(sha(asset/'dataset/ogbn_arxiv'/item['path'])==item['sha256'],'Prepared raw file differs: '+item['path'])
        venv=out/'preprocess_env';require(run([BASE_PY,'-m','venv','--system-site-packages',venv],out/'create_venv.log',120)==0,'Virtual environment creation failed')
        python=venv/'bin/python';isolated=core_versions(python)
        require(Path(isolated['prefix']).resolve()==venv and isolated['versions']==before['versions'],'Isolated core versions differ')
        constraints=out/'constraints.txt';constraints.write_text(''.join(name+'=='+version+'\n' for name,version in before['versions'].items()),encoding='utf-8')
        pip=[python,'-m','pip','--isolated','--disable-pip-version-check','--no-input','--require-virtualenv',
             '--cache-dir',out/'pip_cache','install','--index-url','https://pypi.org/simple','--timeout','30','--retries','1','-c',constraints]
        require(run(pip+PIP_PACKAGES,out/'isolated_dependencies.log',900)==0,'Isolated dependency preparation failed; inspect log')
        with urllib.request.urlopen('https://pypi.org/pypi/localgraphclustering/0.6.1/json',timeout=35) as response:
            meta=json.loads(response.read(1024*1024))
        save(out/'pypi_release_metadata.json',meta)
        wheel=next(e for e in meta['urls'] if e['filename']=='localgraphclustering-0.6.1-py2.py3-none-any.whl')
        wheel_path=out/wheel['filename'];download_release(wheel,wheel_path,contract['wheel_sha256'])
        require(run(pip+['--no-deps',wheel_path],out/'install_lgc.log',300)==0,'Isolated LocalGraphClustering install failed')
        probe="import numpy as n; from scipy.sparse import csr_matrix; from localgraphclustering import GraphLocal,approximate_PageRank; g=GraphLocal.from_sparse_adjacency(csr_matrix(n.array([[1.,1.],[1.,1.]]))); x,p=approximate_PageRank(g,[0],iterations=1000,method='acl',normalize=False); assert len(x)>0; print('NATIVE_ACL_PROBE=PASS')"
        probe_rc=run([python,'-B','-c',probe],out/'native_probe.log',120)
        if probe_rc:
            text=(out/'native_probe.log').read_text('utf-8',errors='replace')
            require(any(s in text for s in ['libgraph','shared object','undefined symbol','GLIBC','Exec format']),
                    'Dependency import/API failed; preserved isolated log without changing the frozen environment')
            require(shutil.which('make') and (shutil.which('g++') or shutil.which('c++')),'Native fallback requires existing make/C++ compiler; no system install attempted')
            source=next(e for e in meta['urls'] if e['filename']=='localgraphclustering-0.6.1.tar.gz')
            source_path=out/source['filename'];download_release(source,source_path,contract['source_sha256'])
            extracted=unpack_source(source_path,out/'native_source');make_dir=extracted/'localgraphclustering/src/lib/graph_lib_test'
            require(run(['make','-B','-C',make_dir,'-j2','libgraph.so'],out/'native_build.log',600)==0,'Native source build failed')
            code="import importlib.util; print(importlib.util.find_spec('localgraphclustering').origin)"
            package=Path(subprocess.check_output([str(python),'-B','-c',code],text=True).strip()).parent
            require(package.resolve().is_relative_to(venv),'Native replacement target escaped isolated environment')
            native=package/'src/lib/graph_lib_test/libgraph.so'
            if native.exists():native.rename(native.with_name('libgraph.so.prebuilt'))
            shutil.copyfile(make_dir/'libgraph.so',native)
            require(run([python,'-B','-c',probe],out/'native_probe_rebuilt.log',120)==0,'Rebuilt native ACL probe failed')
            report['native_implementation']='Pinned 0.6.1 source rebuilt in isolation'
        else:report['native_implementation']='Pinned PyPI 0.6.1 wheel'
        package_code="import importlib.util; print(importlib.util.find_spec('localgraphclustering').origin)"
        package_path=Path(subprocess.check_output([str(python),'-B','-c',package_code],text=True).strip()).parent.resolve()
        require(package_path.is_relative_to(venv),'Dependency identity path escaped isolation')
        report['clustering_implementation_files']=[dict(path=str(p),sha256=sha(p)) for p in sorted(package_path.rglob('*'))
            if p.is_file() and (p.suffix=='.so' or p.name in ['GraphLocal.py','approximate_PageRank.py','aclpagerank_cpp.py'])]
        require(core_versions(python)['versions']==before['versions'],'Isolated core versions changed')
        code=run([python,'-B',PACKAGE/'cluster_worker.py','--project',project,'--output-dir',out,
                  '--contract',PACKAGE/'preprocess_contract.json'],out/'cluster_worker.log',2400)
        require(code==0,'Clustering worker stopped; inspect report/log')
        worker=json.loads((out/'clustering.json').read_text('utf-8'));report['clustering_status']=worker['status']
        report['status']='CLUSTERS_READY_FOR_L1' if worker['status']=='COMPLETE_SELF_GENERATED_UPSTREAM_LC' else 'PILOT_COMPLETE_FULL_GENERATION_HELD'
    except Exception as error:report.update(status='STOPPED_PRESERVED',error_type=type(error).__name__,error=str(error))
    if before is not None:
        after=core_versions(BASE_PY);report['base_after']=after;report['base_core_versions_unchanged']=after==before
        if after!=before:report['status']='STOPPED_BASE_IDENTITY_CHANGED'
    report['wall_seconds']=time.monotonic()-started;save(out/'bootstrap.json',report)
    with zipfile.ZipFile(out/'clustering_return.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        names=['bootstrap.json','clustering.json','progress.json','constraints.txt','create_venv.log','isolated_dependencies.log',
               'install_lgc.log','native_probe.log','native_probe_rebuilt.log','native_build.log','cluster_worker.log','pypi_release_metadata.json','pilot_clusters.npz']
        for name in names:
            p=out/name
            if p.is_file() and p.stat().st_size<8*1024**2:z.write(p,p.name)
    print('CLUSTER_PREPARATION_STATUS='+report['status'],flush=True)
    print('RETURN_FILE='+str(out/'clustering_return.zip'),flush=True)
    return 1 if report['status'].startswith('STOPPED') else 0


if __name__=='__main__':raise SystemExit(main())
