"""Exactly twelve complete Cora runs after the frozen measurement gate."""
import argparse
import datetime
from pathlib import Path
import shutil
import statistics
import sys
import time

sys.dont_write_bytecode=True
import protocol as p

def preflight(repo,stage):
    p.require(Path(sys.prefix).resolve()==Path('/home/jovyan/work/envs/graphmae2').resolve(),'Activate graphmae2')
    manifest=p.sources(repo); anchor=p.anchors(repo)
    p.check_stage(repo,stage,manifest,anchor)
    return manifest,anchor

def summary(rows):
    groups=[]
    for method in p.METHODS:
        ordered=sorted([r for r in rows if r['method']==method],key=lambda r:r['seed'])
        p.require([x['seed'] for x in ordered]==[0,1,2],'Incomplete paired group')
        v=[100*x['best_val_acc'] for x in ordered]
        groups.append(dict(dataset='cora',method=method,n=3,seed_order=[0,1,2],per_seed_val_percent=v,
                           mean_val_percent=statistics.mean(v),population_std_percent=statistics.pstdev(v)))
    lookup={(x['method'],x['seed']):x['best_val_acc'] for x in rows}; comparisons=[]
    for a,b in (('P','S'),('P','H'),('P','PS'),('H','S'),('PS','S')):
        differences=[100*(lookup[a,s]-lookup[b,s]) for s in range(3)]
        comparisons.append(dict(method=a,reference=b,differences_pp=differences,mean_pp=statistics.mean(differences),
                                wins=sum(x>1e-10 for x in differences),ties=sum(abs(x)<=1e-10 for x in differences),losses=sum(x< -1e-10 for x in differences),
                                leave_one_seed_out_mean_pp=[statistics.mean(differences[:s]+differences[s+1:]) for s in range(3)]))
    return dict(dataset='cora',groups=groups,comparisons=comparisons,runs=rows,primary_comparison='P_minus_S',
                secondary_comparisons=['P_minus_H','P_minus_PS'],test_metrics_opened=False,
                interpretation='Exploratory single-graph n=3; matched rank-weight multisets, separate online feedback histories; no significance/causal/generalization claim')

def run(cli):
    repo,stage,batch,weights=cli.repo.resolve(),cli.stage1_dir.resolve(),cli.batch_dir.resolve(),cli.checkpoint_root.resolve()
    p.require(batch.parent==repo/'outputs' and batch.name.startswith('p_development12_'),'Use unique pilot output')
    p.require(batch.is_dir() and {x.name for x in batch.iterdir()}<={'batch.log','batch.pid'},'Never overwrite an existing batch')
    p.require(not weights.is_relative_to(repo),'Checkpoint root must be outside project')
    with p.locked(repo) as fd:
        state=dict(status='running',completed=0,total=12,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        p.write(batch/'batch_status.json',state); started=time.perf_counter(); rows=[]
        try:
            manifest,anchor=preflight(repo,stage); identity=p.check_package()
            for name in sorted(set(identity['queue_files_sha256'])|{'queue_identity.json','SHA256SUMS.txt'}):
                target=batch/'queue_snapshot'/name; target.parent.mkdir(exist_ok=True); shutil.copyfile(p.PACKAGE/name,target)
            p.write(batch/'plan.json',dict(dataset='cora',jobs=p.JOBS,training_seeds=[0,1,2],source_sha256=manifest['after_sha256'],
                      primary_comparison='P_minus_S',secondary_comparisons=['P_minus_H','P_minus_PS'],queue_identity=identity,
                      stage1_directory=str(stage),test_metrics_used_for_selection=False,full_runs=12,score_measurement='observation-time EMA proxy'))
            p.write(batch/'input_protocol_anchor.json',anchor)
            for method,seed in p.JOBS:
                p.sources(repo)
                name=batch.name+'_cora_'+method+'_s'+str(seed); directory=batch/name
                print('START dataset=cora method=%s seed=%d'%(method,seed),flush=True)
                p.execute(p.command(directory,weights,method,seed,'development'),repo,batch/(name+'_launch.log'),fd)
                p.execute([sys.executable,'-u','record_full_batch_p.py','verify','--output-dir',str(directory)],repo,directory/'reload.log',fd)
                row=p.validate(directory,method,seed,'development',anchor,manifest,batch/(name+'_launch.log'),directory/'reload.log')
                rows.append(row); state['completed']=len(rows)
                p.write(batch/'validated_runs.json',rows); p.write(batch/'batch_status.json',state)
                print('DONE %d/12 dataset=cora method=%s seed=%d BestVal=%.4f'%(len(rows),method,seed,row['best_val_acc']),flush=True)
            p.sources(repo); result=summary(rows); p.write(batch/'validation_summary.json',result)
            for group in result['groups']:
                print('VAL_GROUP %s=%.3f +/- %.3f %%'%(group['method'],group['mean_val_percent'],group['population_std_percent']),flush=True)
            for c in result['comparisons']:
                print('VAL_COMPARE %s-minus-%s=%+.3fpp'%(c['method'],c['reference'],c['mean_pp']),flush=True)
            state.update(status='complete',wall_seconds=time.perf_counter()-started); p.write(batch/'batch_status.json',state)
            print('ALL_DONE 12/12',flush=True)
        except BaseException as error:
            state.update(status='failed',error_type=type(error).__name__,error=str(error)); p.write(batch/'batch_status.json',state); raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True); parser.add_argument('--batch-dir',type=Path,required=True)
    parser.add_argument('--stage1-dir',type=Path,required=True)
    parser.add_argument('--checkpoint-root',type=Path,default=Path('/home/jovyan/work/GraphMAE2_artifacts/checkpoints'))
    code=0
    try: run(parser.parse_args())
    except BaseException as error: print('P12_ERROR='+str(error),flush=True); code=1
    print('BATCH_EXIT_CODE='+str(code),flush=True); return code

if __name__=='__main__': raise SystemExit(main())
