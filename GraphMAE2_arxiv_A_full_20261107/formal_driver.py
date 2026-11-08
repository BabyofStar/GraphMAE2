"""Launch only selected A; preserve failed attempts and committed epoch resumes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from measure_common import require,save,sha,now,versions,resource_snapshot,nvidia_snapshot,assess_allocator_log
from formal_logic import summarize_probes

PACKAGE=Path(__file__).resolve().parent


def run_phase(stage,snapshot,out,assets,attempt,cap):
    attempt_dir=out/'attempts'/attempt;result=out/(stage+'.json');log=attempt_dir/(stage+'.log')
    if result.exists():shutil.copyfile(result,attempt_dir/(stage+'_previous_result.json'))
    command=[sys.executable,'-B',str(PACKAGE/'formal_worker.py'),'--stage',stage,'--snapshot',str(snapshot),
             '--output-dir',str(out),'--assets',str(assets),'--result',str(result),'--attempt',attempt]
    print(f'STAGE_START={stage} LOG={log}',flush=True);started=time.monotonic();timed_out=False
    with log.open('x',encoding='utf-8') as stream,(attempt_dir/(stage+'_resources.jsonl')).open('x',encoding='utf-8') as samples:
        process=subprocess.Popen(command,cwd=snapshot,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                 start_new_session=True,pass_fds=(8,9))
        save(attempt_dir/(stage+'_process.json'),dict(pid=process.pid,stage=stage,attempt=attempt,command=command,
                                                    inherited_job_lock_fds=[8,9],time_beijing=now()))
        try:
            while process.poll() is None:
                state=resource_snapshot();state['child_pid']=process.pid
                try:state['child_status']='\n'.join(line for line in Path(f'/proc/{process.pid}/status').read_text().splitlines() if line.startswith(('VmRSS:','VmHWM:','Threads:')))
                except OSError:pass
                samples.write(json.dumps(state,allow_nan=False)+'\n');samples.flush()
                if time.monotonic()-started>cap:
                    timed_out=True;break
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=15)
    # A prior attempt's result must never masquerade as this process's output.
    if result.exists():
        value=json.loads(result.read_text('utf-8'))
        if value.get('attempt')!=attempt or value.get('stage')!=stage:
            value=dict(status='STOPPED_PRESERVED',stage=stage,attempt=attempt,error='Worker exited before a current-attempt result was written')
    else:value=dict(status='STOPPED_PRESERVED',stage=stage,attempt=attempt,error='Worker exited without result JSON')
    if timed_out:value.update(status='STOPPED_TIME_CAP_PRESERVED',cap_seconds=cap)
    if process.returncode!=0 and value['status'].startswith('PASS_'):value['status']='STOPPED_EXIT_CODE_MISMATCH'
    assessed=assess_allocator_log(value,log.read_text('utf-8',errors='replace'))
    save(attempt_dir/(stage+'_assessment.json'),assessed)
    if 'allocator_oom_log_lines' in assessed:
        save(out/'resume_requires_review.json',dict(stage=stage,attempt=attempt,reason='HAMI OOM observed; do not accept or resume unchecked weights automatically',log=str(log)))
    print(f'STAGE_END={stage} STATUS={assessed["status"]}',flush=True)
    return assessed


def verify_pretrain(out,job):
    accepted=out/'pretrain_accepted.json'
    if not accepted.exists():return False
    data=json.loads(accepted.read_text('utf-8'))
    require(data['status']=='PASS_FULL_PRETRAIN' and data['completed_epochs']==60 and data['job_id']==job['job_id'],'Accepted pretrain differs')
    require(sha(Path(data['checkpoint']['path']))==data['checkpoint']['sha256'],'Final encoder identity changed')
    return True


def verify_embedding(out):
    accepted=out/'inference_accepted.json'
    if not accepted.exists():return False
    data=json.loads(accepted.read_text('utf-8'));record=data['embedding']
    require(data['status']=='PASS_FULL_EMBEDDING_EXTRACTION' and record['shape']==[169343,1024] and
            sha(Path(record['path']))==record['sha256'] and sha(Path(record['root_ids_path']))==record['root_ids_sha256'],'Accepted embedding identity changed')
    require(json.loads((out/'embedding_receipt.json').read_text('utf-8'))==record,'Embedding receipt differs')
    return True


def pack(out,attempt):
    candidates=[]
    for path in out.rglob('*'):
        relative=path.relative_to(out)
        if not path.is_file() or relative.parts[0] in ['frozen_source','tmp','matplotlib_cache']:continue
        if path.suffix not in ['.json','.jsonl','.log','.tar'] or path.name=='run.log':continue
        if path.name=='return_manifest.json':continue
        require(path.stat().st_size<48*1024**2,'One return evidence file exceeds its size budget')
        candidates.append(path)
    require(sum(p.stat().st_size for p in candidates)<160*1024**2,'Return evidence exceeds its size budget')
    manifest=out/'attempts'/attempt/'return_manifest.json'
    save(manifest,dict(files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(candidates)]))
    target=out/f'A_return_{attempt}.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in candidates+[manifest]:archive.write(path,path.relative_to(out).as_posix())
        for name in ['formal_contract.json','formal_worker.py','formal_driver.py','formal_logic.py','measure_common.py','runtime_core.py','kit_identity.json']:
            archive.write(PACKAGE/name,'formal_kit/'+name)
    print('RETURN_FILE='+str(target),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();out=args.output_dir.resolve()
    contract=json.loads((PACKAGE/'formal_contract.json').read_text('utf-8'));project=Path(contract['project']).resolve()
    require(out.parent==project/'outputs' and out.name.startswith('arxiv_A_'),'Unexpected A output directory')
    attempts=out/'attempts';attempts.mkdir(exist_ok=True)
    indexes=[int(p.name[1:]) for p in attempts.iterdir() if p.is_dir() and p.name.startswith('a') and p.name[1:].isdigit()]
    attempt=f'a{max(indexes,default=0)+1:04d}';(attempts/attempt).mkdir()
    report=dict(status='STARTED',attempt=attempt,time_beijing=now(),selected_budget='A',method='B0',pretrain_seed=0,
                pretrain_epochs=60,probe_seeds=list(range(10)),probe_epochs=1000,estimated_hours=contract['estimated_full_hours'],
                resources_before=resource_snapshot(),GPU_before=nvidia_snapshot())
    started=time.monotonic();before=None
    try:
        require(not (out/'resume_requires_review.json').exists(),'A previous allocator OOM requires evidence review before resuming')
        for old_log in attempts.glob('a*/*.log'):
            if old_log.name not in ['pretrain.log','inference.log','classifier.log']:continue
            checked=assess_allocator_log(dict(stage=old_log.stem,status='STOPPED_PRESERVED'),old_log.read_text('utf-8',errors='replace'))
            if 'allocator_oom_log_lines' in checked:
                save(out/'resume_requires_review.json',dict(reason='Allocator OOM found in historical phase log',log=str(old_log)))
                raise RuntimeError('Historical HAMI OOM requires evidence review before resuming')
        before=versions();report['base_before']=before
        require(before['prefix']==contract['base_prefix'] and before['packages']==contract['base_versions'],'Frozen environment differs')
        for name,digest in json.loads((PACKAGE/'kit_identity.json').read_text('utf-8')).items():require(sha(PACKAGE/name)==digest,'Formal kit differs: '+name)
        snapshot=out/'frozen_source';identity=out/'job_identity.json'
        if identity.exists():
            require(args.resume,'Existing run requires resume.sh, never start.sh')
            job=json.loads(identity.read_text('utf-8'));require(job['contract_sha256']==sha(PACKAGE/'formal_contract.json') and job['job_id']==out.name and job['kit_identity_sha256']==sha(PACKAGE/'kit_identity.json'),'Run/contract/kit identity changed')
            for name,digest in contract['source_sha256'].items():require(sha(snapshot/name)==digest,'Snapshot identity changed')
        else:
            require(not args.resume,'No initialized run exists to resume')
            snapshot.mkdir();(out/'traces').mkdir()
            for name,digest in contract['source_sha256'].items():
                source=project/name;require(source.is_file() and not source.is_symlink() and source.resolve().is_relative_to(project) and sha(source)==digest,'Project source differs: '+name)
                target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            assets=Path(contract['asset_root']).parent/out.name;assets.mkdir(exist_ok=False)
            job=dict(job_id=out.name,asset_dir=str(assets),contract_sha256=sha(PACKAGE/'formal_contract.json'),
                     kit_identity_sha256=sha(PACKAGE/'kit_identity.json'),
                     method='B0',pretrain_seed=0,pretrain_epochs=60,probe_seeds=list(range(10)),probe_epochs=1000,
                     actual_pretrain_batch_size=256,actual_inference_batch_size=256,probe_batch_size=5120,graph_workers=0,
                     measurement_checkpoint_reused=False,source_sha256=contract['source_sha256'])
            save(identity,job);save(out/'frozen_contract.json',contract)
            with tarfile.open(out/'source_snapshot.tar','x') as archive:
                for name in sorted(contract['source_sha256']):archive.add(snapshot/name,arcname=name,recursive=False)
        assets=Path(job['asset_dir']).resolve();require(assets.parent==Path(contract['asset_root']).parent and assets.name==out.name,'Run assets escaped their expected directory')
        for name,digest in contract['source_sha256'].items():require(sha(project/name)==digest,'Current project source differs: '+name)
        checks=[]
        for item in contract['raw_files']:
            path=Path(contract['asset_root'])/'dataset/ogbn_arxiv'/item['path'];require(sha(path)==item['sha256'] and path.stat().st_size==item['bytes'],'Raw data identity differs')
            checks.append(dict(**item,verified=True))
        record=contract['cluster_file'];require(sha(Path(record['path']))==record['sha256'],'Full cluster identity differs')
        save(attempts/attempt/'input_identity.json',dict(raw_files=checks,cluster=record))
        report['job_identity']=job;save(out/'formal_status.json',report)
        if not verify_pretrain(out,job):
            result=run_phase('pretrain',snapshot,out,assets,attempt,contract['full_pretrain_phase_cap_seconds'])
            require(result['status']=='PASS_FULL_PRETRAIN','Pretrain stopped; return evidence and resume only if appropriate')
            save(out/'pretrain_accepted.json',result)
        if not verify_embedding(out):
            result=run_phase('inference',snapshot,out,assets,attempt,contract['inference_cap_seconds']+180)
            require(result['status']=='PASS_FULL_EMBEDDING_EXTRACTION','Embedding extraction stopped; partial output is preserved')
            save(out/'inference_accepted.json',result)
        result=run_phase('classifier',snapshot,out,assets,attempt,contract['probe_phase_cap_seconds'])
        require(result['status']=='PASS_TEN_FULL_PROBES','Classifier stage stopped; completed fixed probes remain reusable')
        records=[json.loads((out/f'probe_s{seed}.json').read_text('utf-8')) for seed in range(10)]
        summary=summarize_probes(records)
        require(all((out/f'epoch_{epoch:04d}.json').exists() for epoch in range(1,61)),'Missing complete-epoch records')
        report.update(status='PASS_A_FULL_RUN',formal_pretrain_runs_completed=1,completed_pretrain_epochs=60,completed_probe_runs=10,result=summary)
    except Exception as error:report.update(status='STOPPED_PRESERVED',error=str(error),traceback=traceback.format_exc())
    try:
        report['base_after']=versions();report['base_versions_unchanged']=before==report['base_after']
        if before is not None and before!=report['base_after']:report['status']='STOPPED_BASE_CHANGED'
        differences=[name for name,digest in contract['source_sha256'].items() if not (project/name).is_file() or sha(project/name)!=digest]
        report['project_source_differences_at_end']=differences
        if differences:report['status']='STOPPED_PROJECT_SOURCE_CHANGED'
    except Exception as error:report.update(status='STOPPED_FINAL_CHECK_FAILED',final_check_error=str(error))
    report.update(resources_after=resource_snapshot(),GPU_after=nvidia_snapshot(),wall_seconds_this_attempt=time.monotonic()-started)
    save(out/'formal_status.json',report);save(attempts/attempt/'formal_status.json',report)
    print('FORMAL_A_STATUS='+report['status'],flush=True);pack(out,attempt)
    return 0 if report['status']=='PASS_A_FULL_RUN' else 1


if __name__=='__main__':raise SystemExit(main())
