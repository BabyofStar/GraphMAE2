"""One DS smoke check, then nine fixed-budget DS interaction experiments."""
import argparse
import datetime
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

sys.dont_write_bytecode=True
import reference_r9 as support
from reference_r9 import require,read,write,sha,locked,check_package

PACKAGE=Path(__file__).resolve().parent
MODE='reconstruction_density_curriculum'


def preflight(repo,stage):
    identity,manifest,library,anchors=support.preflight(repo,stage)
    candidates=support.load_module(PACKAGE/'legacy_candidate.py','_ds_old_candidates',identity['queue_files_sha256']['legacy_candidate.py'])
    batch=repo/'outputs/r_development9_yy6rh7vj'
    state=read(batch/'batch_status.json')
    require((state['status'],state['completed'],state['total'])==('complete',9,9),'R9 must be complete before DS')
    for (dataset,seed),pair in anchors.items():
        directory=repo/'outputs/v2_candidates18_p3TNwS'/('v2_candidates18_p3TNwS_%s_D_s%d'%(dataset,seed))
        old_reference=dict(pair['B0'],environment_signature={key:pair['B0']['environment_signature'][key] for key in candidates.ENVIRONMENT_KEYS})
        pair['D']=candidates.validate_candidate(library,directory,dataset,'D',seed,old_reference)
        require(support.env_signature(directory)==pair['B0']['environment_signature'],'D reference environment differs')
        support.verify_snapshot(directory,manifest['before_sha256'])
        require('MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS' in (directory/'reload.log').read_text(encoding='utf-8'),'D reload evidence missing')
        directory=batch/('r_development9_yy6rh7vj_%s_R_s%d'%(dataset,seed))
        anchor=dict(pair['B0'],s_directory=pair['S']['run_directory'])
        pair['R']=support.check_r(directory,dataset,seed,anchor,manifest)
        require('MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS' in (directory/'reload.log').read_text(encoding='utf-8'),'R reload evidence missing')
    return identity,manifest,library,anchors


def command(repo,directory,checkpoint_root,dataset,seed,phase):
    return [sys.executable,'-u','record_full_batch_r.py','run','--dataset',dataset,'--method','DS',
            '--seed',str(seed),'--phase',phase,'--device','0','--output-dir',str(directory),
            '--checkpoint-root',str(checkpoint_root)]


def alpha(step,epochs):
    return .5*min(max((step/max(epochs-1,1)-.2)/.6,0.),1.)


def validate(directory,dataset,seed,phase,pair,manifest):
    status,metrics,args=[read(directory/name) for name in ('status.json','metrics.json','effective_args.json')]
    for record in (status,metrics):
        for key,value in dict(status='complete',dataset=dataset,method='DS',seed=seed,phase=phase,
                              official_epoch_budget=phase=='development',recorder_version='2.0.0-r-dev1').items():
            require(record.get(key)==value,'DS metadata differs: '+key)
    expected=dict(pair['B0']['nonmethod_args'])
    if phase=='smoke':
        expected.update(max_epoch=5,max_epoch_f=5,mask_log_interval=1)
    require(support.normalized_args(args)==expected,'DS nonmethod configuration differs')
    require(args['seeds']==[seed] and args['input_mask_mode']==MODE
            and args['random_group_seed']==support.SEEDS[dataset],'DS seed/mode differs')
    require(metrics['dataset_info']==pair['B0']['dataset_info'],'DS data/features/splits differ')
    require(support.env_signature(directory)==pair['B0']['environment_signature'],'DS runtime differs')
    support.verify_snapshot(directory,manifest['after_sha256'])
    n=metrics['dataset_info']['nodes']
    epochs=args['max_epoch']
    k=int(n*args['mask_rate'])
    sizes=[n//4+(i<n%4) for i in range(4)]
    quotas=support.quotas(sizes,k)
    feedback=[support.json.loads(line) for line in (directory/'mask_feedback.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    interval=1 if phase=='smoke' else 100
    steps=sorted(set(range(0,epochs,interval))|{epochs-1})
    require([x['step'] for x in feedback]==steps and metrics['feedback_records']==len(steps),'DS feedback budget differs')
    for row in feedback:
        require(row['mode']==MODE and row['nodes']==n and row['masked']==k
                and math.isclose(row['alpha'],alpha(row['step'],epochs),abs_tol=1e-12),'DS course/mask budget differs')
        require([x['layer'] for x in row['density_layers']]==list(range(4))
                and [x['nodes'] for x in row['density_layers']]==sizes
                and [x['masked'] for x in row['density_layers']]==quotas,'DS fixed density quota differs')
        require('sampling_groups' not in row,'DS must use real density strata, not random R groups')
        require(math.isclose(row['observation_count_min_mean_max'][1],k*(row['step']+1)/n,rel_tol=1e-9,abs_tol=1e-8),'DS observation budget differs')
    node_path=directory/metrics['mask_node_statistics_file']
    require(sha(node_path)==metrics['mask_node_statistics_sha256'],'DS NPZ hash differs')
    import numpy as np
    with np.load(node_path,allow_pickle=False) as nodes,np.load(Path(pair['S']['run_directory'])/'mask_node_statistics.npz',allow_pickle=False) as s:
        require(set(nodes.files)=={'node_id','nonself_in_degree','reconstruction_error_ema','observation_count','local_feature_density','density_layer'},'DS node schema differs')
        for name in ('node_id','nonself_in_degree','local_feature_density','density_layer'):
            require(np.array_equal(nodes[name],s[name]),'DS true graph/density arrays differ from S: '+name)
        counts=nodes['observation_count']
        require(counts.shape==(n,) and np.issubdtype(counts.dtype,np.integer) and (counts>=0).all() and (counts<=epochs).all()
                and int(counts.sum())==epochs*k,'DS exact observation budget differs')
        errors=nodes['reconstruction_error_ema']
        require(errors.shape==(n,) and np.isfinite(errors).all() and (errors>=0).all(),'DS invalid difficulty diagnostics')
    val,epoch=metrics['best_val_acc'],metrics['best_val_epoch']
    require(isinstance(val,(float,int)) and math.isfinite(val) and 0<=val<=1
            and isinstance(epoch,int) and 0<=epoch<args['max_epoch_f'],'Invalid DS validation metric')
    resources=read(directory/'resources.json')
    return dict(dataset=dataset,method='DS',seed=seed,best_val_acc=val,best_val_epoch=epoch,
                resources=resources,run_directory=str(directory),checkpoint_sha256=metrics['checkpoint_sha256'])


def summary(rows,anchors):
    groups=[]
    for dataset in support.SEEDS:
        ordered=sorted((x for x in rows if x['dataset']==dataset),key=lambda x:x['seed'])
        require([x['seed'] for x in ordered]==[0,1,2],'Incomplete DS group')
        values=[100*x['best_val_acc'] for x in ordered]
        group=dict(dataset=dataset,n=3,DS_val_percent_mean=statistics.mean(values),DS_val_percent_population_std=statistics.pstdev(values))
        for method in ('S','B2','D','B0'):
            diffs=[100*(x['best_val_acc']-anchors[dataset,x['seed']][method]['best_val_acc']) for x in ordered]
            group['DS_minus_'+method+'_paired_pp']=diffs
            group['DS_minus_'+method+'_mean_pp']=statistics.mean(diffs)
        groups.append(group)
    return dict(groups=groups,runs=rows,primary_comparison='DS_minus_S',
                equal_dataset_DS_minus_S_pp=statistics.mean(g['DS_minus_S_mean_pp'] for g in groups),
                equal_dataset_DS_minus_B2_pp=statistics.mean(g['DS_minus_B2_mean_pp'] for g in groups),
                test_metrics_opened=False,interpretation='Exploratory n=3 interaction test; no significance or guaranteed gain')


def run(cli):
    repo,batch,stage=cli.repo.resolve(),cli.batch_dir.resolve(),cli.stage1_dir.resolve()
    require(batch.parent==repo/'outputs' and batch.name.startswith('ds_development9_'),'Use a unique DS output directory')
    require(batch.is_dir() and {p.name for p in batch.iterdir()}<={'batch.log','batch.pid'},'Never overwrite a previous DS batch')
    root=cli.checkpoint_root.resolve()
    require(not root.is_relative_to(repo),'Weights must be outside project')
    with locked(repo) as fd:
        state=dict(status='running',completed=0,total=9,stage='preflight',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(batch/'batch_status.json',state)
        started=time.perf_counter()
        try:
            identity,manifest,library,anchors=preflight(repo,stage)
            for name in sorted(set(identity['queue_files_sha256'])|{'queue_identity.json','SHA256SUMS.txt'}):
                dest=batch/'queue_snapshot'/name
                dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(PACKAGE/name,dest)
            jobs=[(dataset,seed) for dataset in support.SEEDS for seed in (0,1,2)]
            write(batch/'plan.json',dict(method='DS',runs=jobs,official_configs=library.CONFIGS,
                  primary_comparison='DS_minus_S',secondary_comparisons=['DS_minus_B2','DS_minus_D','DS_minus_B0'],
                  source_sha256=manifest['after_sha256'],queue_identity=identity,stage1_dir=str(stage),
                  smoke_before_full=True,test_metrics_used_for_selection=False))
            write(batch/'historical_anchors.json',[dict(dataset=d,seed=s,methods=pair) for (d,s),pair in anchors.items()])
            state['stage']='smoke'
            write(batch/'batch_status.json',state)
            smoke=batch/(batch.name+'_smoke_cora_DS_s0')
            support.execute(command(repo,smoke,root,'cora',0,'smoke'),repo,batch/'smoke_DS_launch.log',fd)
            validate(smoke,'cora',0,'smoke',anchors['cora',0],manifest)
            support.execute([sys.executable,'-u','record_full_batch_r.py','verify','--output-dir',str(smoke)],repo,smoke/'reload.log',fd)
            require('MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS' in (smoke/'reload.log').read_text(encoding='utf-8'),'DS smoke reload failed')
            print('DS_SMOKE=PASS (5 pretrain / 5 probe; CPU reload)',flush=True)
            rows=[]
            state['stage']='development'
            write(batch/'batch_status.json',state)
            for dataset,seed in jobs:
                check_package()
                support.check_sources(repo,identity)
                name='%s_%s_DS_s%d'%(batch.name,dataset,seed)
                directory=batch/name
                print('START dataset=%s method=DS seed=%d'%(dataset,seed),flush=True)
                support.execute(command(repo,directory,root,dataset,seed,'development'),repo,batch/(name+'_launch.log'),fd)
                row=validate(directory,dataset,seed,'development',anchors[dataset,seed],manifest)
                support.execute([sys.executable,'-u','record_full_batch_r.py','verify','--output-dir',str(directory)],repo,directory/'reload.log',fd)
                require('MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS' in (directory/'reload.log').read_text(encoding='utf-8'),'DS CPU reload failed')
                rows.append(row)
                state['completed']=len(rows)
                write(batch/'validated_runs.json',rows)
                write(batch/'batch_status.json',state)
                print('DONE %d/9 dataset=%s seed=%d BestVal=%.4f'%(len(rows),dataset,seed,row['best_val_acc']),flush=True)
            check_package()
            support.check_sources(repo,identity)
            result=summary(rows,anchors)
            write(batch/'validation_summary.json',result)
            for g in result['groups']:
                print('VAL_COMPARE %s DS=%.3f +/- %.3f %% DS-minus-S=%+.3fpp DS-minus-B2=%+.3fpp'%(g['dataset'],g['DS_val_percent_mean'],g['DS_val_percent_population_std'],g['DS_minus_S_mean_pp'],g['DS_minus_B2_mean_pp']),flush=True)
            state.update(status='complete',stage='complete',wall_seconds=time.perf_counter()-started)
            write(batch/'batch_status.json',state)
            print('ALL_DONE 9/9',flush=True)
        except BaseException as error:
            state.update(status='failed',error_type=type(error).__name__,error=str(error))
            write(batch/'batch_status.json',state)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True,type=Path)
    parser.add_argument('--batch-dir',required=True,type=Path)
    parser.add_argument('--stage1-dir',required=True,type=Path)
    parser.add_argument('--checkpoint-root',type=Path,default=Path('/home/jovyan/work/GraphMAE2_artifacts/checkpoints'))
    cli=parser.parse_args()
    code=0
    try:
        run(cli)
    except BaseException as error:
        print('DS9_ERROR='+str(error),flush=True)
        code=1
    print('BATCH_EXIT_CODE='+str(code),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
