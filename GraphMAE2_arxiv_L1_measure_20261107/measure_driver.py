"""Freeze inputs, run bounded measurements, and package small return evidence."""
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

from measure_common import (require,save,sha,now,versions,resource_snapshot,nvidia_snapshot,
                            choose_capacity,budget_estimate)

PACKAGE=Path(__file__).resolve().parent


def run_child(stage,batch,snapshot,out,assets,timeout):
    name=f'{stage}_{batch}' if stage.endswith('_capacity') else stage
    result=out/(name+'.json');log=out/(name+'.log')
    command=[sys.executable,'-B',str(PACKAGE/'measure_worker.py'),'--stage',stage,
             '--batch-size',str(batch),'--snapshot',str(snapshot),'--output-dir',str(out),
             '--assets',str(assets),'--result',str(result)]
    print(f'STAGE_START={name} LOG={log}',flush=True)
    began=time.monotonic();timed_out=False
    with log.open('x',encoding='utf-8') as stream, (out/(name+'_resources.jsonl')).open('x',encoding='utf-8') as samples:
        process=subprocess.Popen(command,cwd=snapshot,stdout=stream,stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL,start_new_session=True)
        deadline=began+timeout
        try:
            while process.poll() is None:
                state=resource_snapshot();state['child_pid']=process.pid
                try:
                    state['child_status']='\n'.join(line for line in Path(f'/proc/{process.pid}/status').read_text().splitlines()
                                                   if line.startswith(('Name:','VmRSS:','VmHWM:','Threads:')))
                except OSError:
                    pass
                samples.write(json.dumps(state,allow_nan=False)+'\n');samples.flush()
                if time.monotonic()>deadline:
                    timed_out=True
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
                    break
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
        finally:
            if process.poll() is None:
                # Only the process group created above belongs to this measurement.
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
    if timed_out:
        value=dict(stage=stage,batch_size=batch,status='HELD_FOR_ENGINEERING_COST',
                   error='Declared subprocess time cap reached; its own process group stopped',
                   cap_seconds=timeout,wall_seconds=time.monotonic()-began,partial_files_preserved=True)
        save(result,value)
    elif result.is_file():
        value=json.loads(result.read_text('utf-8'))
        require(value['stage']==stage and value['batch_size']==batch,'Worker result identity differs')
        require(not value['status'].startswith('PASS_') or process.returncode==0,'Worker exited nonzero after reporting PASS')
    else:
        value=dict(stage=stage,batch_size=batch,status='STOPPED_PRESERVED',exit_code=process.returncode,
                   error='Worker exited without a result JSON; inspect its log. No automatic retry.',wall_seconds=time.monotonic()-began)
        save(result,value)
    print(f'STAGE_END={name} STATUS={value["status"]}',flush=True)
    return value


def capacity_loop(stage,contract,snapshot,out,assets,report):
    attempts=[]
    for batch in contract['batch_size_candidates']:
        value=run_child(stage,batch,snapshot,out,assets,contract['capacity_attempt_cap_seconds'])
        attempts.append(value)
        report[stage+'_attempts']=[dict(batch_size=v['batch_size'],status=v['status']) for v in attempts]
        save(out/'measurement.json',report)
        if value['status']=='HELD_FOR_ENGINEERING_COST':return None
        selected=choose_capacity(attempts)
        if selected is not None:return selected
    return None


def package_return(out,contract):
    # No checkpoints, embeddings, dataset files, caches or environment trees.
    candidates=sorted(p for p in out.iterdir() if p.is_file() and
                      p.suffix in ['.json','.jsonl','.log','.tar'] and p.name!='run.log')
    manifest=[]
    for path in candidates:
        require(path.stat().st_size<32*1024**2,'Unexpectedly large return log; preserve it on server')
        manifest.append(dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path)))
    require(sum(item['bytes'] for item in manifest)<96*1024**2,'Return evidence exceeds declared budget')
    save(out/'return_manifest.json',dict(files=manifest,scope='Small engineering evidence only; large measured assets remain at recorded paths'))
    target=out/'measurement_return.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in candidates+[out/'return_manifest.json']:archive.write(path,path.name)
        for name in ['measure_common.py','measure_worker.py','measure_driver.py','measurement_contract.json','pilot_clusters.npz']:
            archive.write(PACKAGE/name,'measurement_kit/'+name)
    print('RETURN_FILE='+str(target),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True,type=Path);parser.add_argument('--output-dir',required=True,type=Path)
    request=parser.parse_args();project=request.project.resolve();out=request.output_dir.resolve()
    require(out.is_dir() and not (out/'measurement.json').exists(),'Use the new output directory from start.sh')
    contract=json.loads((PACKAGE/'measurement_contract.json').read_text('utf-8'))
    report=dict(stage='ogbn-arxiv B0 bounded L1/L1b measurement',status='STARTED',time_beijing=now(),
                project=str(project),output_dir=str(out),formal_pretrains_started=0,full_60_epoch_training_started=False,
                B2_training_started=False,full_budget_selected=False,test_accuracy_computed=False,
                resources_before=resource_snapshot(),GPU_before=nvidia_snapshot(),
                limitations=['GPU and cgroup observations may include shared workloads; no other task is stopped.',
                             'Batch candidates are selected only for engineering feasibility, never for classification scores.',
                             'No B2 batch adapter is included or validated here; option B requires a later mapping/curriculum check.'])
    began=time.monotonic();before=None;source_verified=False
    try:
        before=versions();report['base_before']=before
        require(before['prefix']==contract['base_prefix'] and before['packages']==contract['base_versions'],'Frozen base environment differs')
        require(project==Path(contract['project']),'Unexpected project root')
        expected_kit=json.loads((PACKAGE/'kit_identity.json').read_text('utf-8'))
        for relative,digest in expected_kit.items():
            require(sha(PACKAGE/relative)==digest,'Measurement kit differs: '+relative)
        snapshot=out/'frozen_source';snapshot.mkdir()
        for relative,expected in contract['source_sha256'].items():
            source=project/relative
            require(source.is_file() and not source.is_symlink() and source.resolve().is_relative_to(project),'Unsafe or missing source: '+relative)
            require(sha(source)==expected,'Project source differs: '+relative)
            target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
            require(sha(target)==expected,'Source copy differs')
        source_verified=True
        save(out/'source_identity.json',dict(files=contract['source_sha256'],count=len(contract['source_sha256']),source='Current verified J55 plus the two extra frozen upstream large-graph files',unchanged=True))
        with tarfile.open(out/'source_snapshot.tar','x') as archive:
            for relative in sorted(contract['source_sha256']):archive.add(snapshot/relative,arcname=relative,recursive=False)
        checks=[]
        asset=Path(contract['asset_root'])
        for item in contract['raw_files']:
            source=asset/'dataset/ogbn_arxiv'/item['path']
            require(source.is_file() and source.stat().st_size==item['bytes'] and sha(source)==item['sha256'],'Prepared raw asset differs: '+item['path'])
            checks.append(dict(**item,verified=True))
        cluster=contract['cluster_file'];path=Path(cluster['path'])
        require(path.is_file() and path.stat().st_size==cluster['bytes'] and sha(path)==cluster['sha256'],'Complete generated cluster identity differs')
        save(out/'input_identity.json',dict(raw_files=checks,cluster_file=cluster,complete_cluster_sha256_verified=True))
        save(out/'frozen_contract.json',contract)
        assets=asset.parent/('arxiv_l1_measurement_'+out.name.removeprefix('arxiv_l1_measure_'))
        require(assets.parent==asset.parent,'Measurement assets escaped their intended directory')
        assets.mkdir(exist_ok=False);report['asset_dir']=str(assets)
        train_batch=capacity_loop('train_capacity',contract,snapshot,out,assets,report)
        if train_batch is None:
            report['status']='HELD_NO_TRAIN_CAPACITY_WITHIN_CAP'
        else:
            report['selected_pretrain_batch_size']=train_batch
            report['batch_size_is_resource_adaptation']=train_batch!=512
            training=run_child('pretrain',train_batch,snapshot,out,assets,contract['pretrain_epoch_cap_seconds']+180)
            if training['status']!='PASS_ONE_FULL_PRETRAIN_EPOCH':
                report.update(status='HELD_OR_STOPPED_DURING_PRETRAIN',failed_stage_status=training['status'])
            else:
                infer_batch=capacity_loop('infer_capacity',contract,snapshot,out,assets,report)
                if infer_batch is None:report['status']='HELD_NO_INFERENCE_CAPACITY_WITHIN_CAP'
                else:
                    report['selected_inference_batch_size']=infer_batch
                    embedding=run_child('inference',infer_batch,snapshot,out,assets,contract['inference_cap_seconds']+180)
                    if embedding['status']!='PASS_FULL_EMBEDDING_EXTRACTION':
                        report.update(status='HELD_OR_STOPPED_DURING_INFERENCE',failed_stage_status=embedding['status'])
                    else:
                        probe=run_child('classifier',5120,snapshot,out,assets,contract['probe_cap_seconds']+180)
                        if probe['status']!='PASS_CLASSIFIER_TIMING':
                            report.update(status='HELD_OR_STOPPED_DURING_CLASSIFIER',failed_stage_status=probe['status'])
                        else:
                            require(training['context']['scaled_feature_sha256']==embedding['context']['scaled_feature_sha256']==probe['context']['scaled_feature_sha256'], 'Cross-stage scaled input identity differs')
                            estimate=budget_estimate(training['epoch_seconds'],embedding['embedding_seconds'],
                                                      probe['steady_probe_epoch_seconds'],training['checkpoint']['write_and_CPU_sample_seconds'],probe['test_forward_seconds'])
                            save(out/'budget_estimate.json',estimate)
                            report.update(status='PASS_L1_L1B_B0_MEASUREMENT',completed_measured_pretrain_epochs=1,
                                          completed_embedding_rows=169343,classifier_epochs_measured=probe['probe_epochs_measured'],
                                          budget_estimate=estimate)
    except Exception as error:
        report.update(status='STOPPED_PRESERVED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    try:
        after=versions();report['base_after']=after;report['base_versions_unchanged']=before==after
        if before is not None and before!=after:report['status']='STOPPED_BASE_IDENTITY_CHANGED'
        if source_verified:
            differences=[relative for relative,digest in contract['source_sha256'].items() if sha(project/relative)!=digest]
            report['project_source_differences_at_end']=differences
            if differences:report['status']='STOPPED_SOURCE_CHANGED_DURING_MEASUREMENT'
    except Exception as error:
        report.update(status='STOPPED_FINAL_IDENTITY_CHECK_FAILED',final_check_error=str(error))
    report.update(resources_after=resource_snapshot(),GPU_after=nvidia_snapshot(),wall_seconds=time.monotonic()-began)
    save(out/'measurement.json',report)
    print('L1_MEASUREMENT_STATUS='+report['status'],flush=True)
    package_return(out,contract)
    return 0 if report['status']=='PASS_L1_L1B_B0_MEASUREMENT' else 1


if __name__=='__main__':raise SystemExit(main())
