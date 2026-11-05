"""Nine C exploratory runs after 94-test and real 5/5 smoke acceptance."""
import argparse
import datetime
from pathlib import Path
import shutil
import statistics
import sys
import time

sys.dont_write_bytecode=True
import c_validation as protocol
from c_validation import PACKAGE,require,read,write,sha,locked,execute


def check_package():
    return protocol.identity()


def preflight(repo,stage):
    require(Path(sys.prefix).resolve()==Path('/home/jovyan/work/envs/graphmae2'),'Activate graphmae2')
    manifest=protocol.sources(repo)
    library,anchors=protocol.anchors(repo)
    require(protocol.check_stage(repo,manifest,anchors)==stage.resolve(),'C stage directory differs from accepted receipt')
    return manifest,library,anchors


def command(directory,checkpoint_root,dataset,seed):
    return [sys.executable,'-u','record_full_batch_c.py','run','--dataset',dataset,'--method','C',
            '--seed',str(seed),'--phase','development','--device','0','--output-dir',str(directory),
            '--checkpoint-root',str(checkpoint_root)]


def summary(rows,anchors):
    groups=[]
    for dataset in protocol.ROUTE_SEEDS:
        ordered=sorted((r for r in rows if r['dataset']==dataset),key=lambda r:r['seed'])
        require([r['seed'] for r in ordered]==[0,1,2],'Incomplete C group')
        values=[100*r['best_val_acc'] for r in ordered]
        group=dict(dataset=dataset,n=3,C_val_percent_mean=statistics.mean(values),C_val_percent_population_std=statistics.pstdev(values))
        for method in ('S','B2','B0'):
            diff=[100*(r['best_val_acc']-anchors[dataset,r['seed']][method]['best_val_acc']) for r in ordered]
            group['C_minus_'+method+'_paired_pp']=diff
            group['C_minus_'+method+'_mean_pp']=statistics.mean(diff)
        groups.append(group)
    return dict(groups=groups,runs=rows,primary_comparison='C_minus_S',
                equal_dataset_C_minus_S_pp=statistics.mean(g['C_minus_S_mean_pp'] for g in groups),
                equal_dataset_C_minus_B2_pp=statistics.mean(g['C_minus_B2_mean_pp'] for g in groups),
                route_order_control_runs=0,test_metrics_opened=False,
                interpretation='Exploratory n=3 full-scheme comparison; lower S exposure and ordering are confounded without dose-matched control')


def run(cli):
    repo,batch,stage=cli.repo.resolve(),cli.batch_dir.resolve(),cli.stage1_dir.resolve()
    require(batch.parent==repo/'outputs' and batch.name.startswith('c_development9_'),'Use a unique C batch directory')
    require(batch.is_dir() and {p.name for p in batch.iterdir()}<={'batch.log','batch.pid'},'Never overwrite an existing batch')
    weights=cli.checkpoint_root.resolve()
    require(not weights.is_relative_to(repo),'Weights must be outside project')
    with locked(repo) as fd:
        state=dict(status='running',completed=0,total=9,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(batch/'batch_status.json',state)
        started=time.perf_counter()
        rows=[]
        try:
            manifest,library,anchors=preflight(repo,stage)
            identity=check_package()
            for name in sorted(set(identity['queue_files_sha256'])|{'queue_identity.json','SHA256SUMS.txt'}):
                dest=batch/'queue_snapshot'/name
                dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(PACKAGE/name,dest)
            jobs=[(dataset,seed) for dataset in protocol.ROUTE_SEEDS for seed in (0,1,2)]
            write(batch/'plan.json',dict(method='C',runs=jobs,official_configs=library.CONFIGS,source_sha256=manifest['after_sha256'],
                  stage1_dir=str(stage),queue_identity=identity,route_seeds=protocol.ROUTE_SEEDS,
                  route_order_control_runs=0,primary_comparison='C_minus_S',test_metrics_used_for_selection=False))
            write(batch/'historical_anchors.json',[dict(dataset=d,seed=s,methods=p) for (d,s),p in anchors.items()])
            routes={}
            for dataset,seed in jobs:
                check_package()
                protocol.sources(repo)
                name='%s_%s_C_s%d'%(batch.name,dataset,seed)
                directory=batch/name
                print('START dataset=%s method=C seed=%d'%(dataset,seed),flush=True)
                execute(command(directory,weights,dataset,seed),repo,batch/(name+'_launch.log'),fd)
                reference=dict(anchors[dataset,seed]['B0'],s_directory=anchors[dataset,seed]['S']['run_directory'])
                row=protocol.validate(directory,dataset,seed,'development',reference,manifest)
                if dataset in routes:
                    require(routes[dataset]==row['route_plan'],'C route plan changed across training seeds')
                routes[dataset]=row['route_plan']
                execute([sys.executable,'-u','record_full_batch_c.py','verify','--output-dir',str(directory)],repo,directory/'reload.log',fd)
                require('MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS' in (directory/'reload.log').read_text(encoding='utf-8'),'C CPU reload marker missing')
                rows.append(row)
                state['completed']=len(rows)
                write(batch/'validated_runs.json',rows)
                write(batch/'batch_status.json',state)
                print('DONE %d/9 dataset=%s seed=%d BestVal=%.4f'%(len(rows),dataset,seed,row['best_val_acc']),flush=True)
            protocol.sources(repo)
            result=summary(rows,anchors)
            write(batch/'validation_summary.json',result)
            for g in result['groups']:
                print('VAL_COMPARE %s C=%.3f +/- %.3f %% C-minus-S=%+.3fpp C-minus-B2=%+.3fpp'%(g['dataset'],g['C_val_percent_mean'],g['C_val_percent_population_std'],g['C_minus_S_mean_pp'],g['C_minus_B2_mean_pp']),flush=True)
            state.update(status='complete',wall_seconds=time.perf_counter()-started)
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
    args=parser.parse_args()
    code=0
    try:
        run(args)
    except BaseException as error:
        print('C9_ERROR='+str(error),flush=True)
        code=1
    print('BATCH_EXIT_CODE='+str(code),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
