"""Frozen Cora S/H/P/PS pilot, common-lock execution and artifact validation."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tarfile

PACKAGE=Path(__file__).resolve().parent
METHODS=('S','H','P','PS')
MODES=dict(S='density_stratified',H='rank_difficulty',P='rank_progress',PS='rank_progress_shuffled')
JOBS=[(m,s) for s,order in enumerate((('S','H','P','PS'),('H','P','PS','S'),('P','PS','S','H'))) for m in order]
ENV_KEYS=('versions','torch_cuda_runtime','torch_cpu_threads','deterministic_algorithms','cudnn_deterministic',
          'cudnn_benchmark','cuda_matmul_allow_tf32','cudnn_allow_tf32','gpu_name','visible_gpu_memory_bytes','runtime_environment')

def require(value,message):
    if not value: raise RuntimeError(message)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))

def write(path,value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')
    os.replace(temporary,path)

def check_package():
    identity=read(PACKAGE/'queue_identity.json')
    require(identity['approved_full_runs']==12 and identity['dataset']=='cora' and identity['jobs']==[list(x) for x in JOBS],'Pilot budget/order differs')
    for name,expected in identity['queue_files_sha256'].items():
        require(sha(PACKAGE/name)==expected,'Package file differs: '+name)
    return identity

def sources(repo):
    check_package(); manifest=read(PACKAGE/'manifest.json')
    for name,expected in manifest['after_sha256'].items():
        path=repo/name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(repo) and sha(path)==expected,'Source differs: '+name)
    return manifest

@contextlib.contextmanager
def locked(repo):
    import fcntl
    require(repo==Path('/home/jovyan/work/GraphMAE2').resolve(),'Use the fixed server project')
    path=repo/'outputs/.formal_record_v1.lock'; path.parent.mkdir(exist_ok=True)
    inherited=os.environ.get('GRAPHMAE2_R_LOCK_FD')
    if inherited is not None:
        fd=int(inherited); actual=os.fstat(fd); expected=path.stat()
        require((actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino),'Wrong inherited lock')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); yield fd
    else:
        with path.open('a') as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB); yield stream.fileno()

def execute(command,repo,log,fd):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',DGLBACKEND='pytorch',WANDB_MODE='disabled',
             MPLBACKEND='Agg',GRAPHMAE2_R_LOCK_FD=str(fd))
    print('RUN_LOG='+str(log),flush=True)
    with log.open('x',encoding='utf-8') as stream:
        process=subprocess.run(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(fd,))
    if process.returncode:
        print('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-60:]),flush=True)
        raise RuntimeError('Child failed: '+str(log))

def snapshot(directory,expected):
    require(read(directory/'source_sha256.json')==expected,'Snapshot manifest differs')
    found=set()
    with tarfile.open(directory/'source.tar','r:') as archive:
        for item in archive:
            path=PurePosixPath(item.name)
            require(item.isfile() and not path.is_absolute() and '..' not in path.parts and '\\' not in item.name
                    and item.name in expected and item.name not in found and item.size<=10*1024*1024,'Unexpected snapshot entry')
            with archive.extractfile(item) as stream:
                require(hashlib.sha256(stream.read()).hexdigest()==expected[item.name],'Snapshot byte mismatch')
            found.add(item.name)
    require(found==set(expected),'Incomplete snapshot')

def normal(args):
    return {k:v for k,v in args.items() if k not in {'seeds','input_mask_mode','random_group_seed','density_route_seed'}}

def anchors(repo):
    b=repo/'outputs/v2_reference18_5AaMEP/v2_reference18_5AaMEP_cora_B0_s0'
    s=repo/'outputs/v2_candidates18_p3TNwS/v2_candidates18_p3TNwS_cora_S_s0'
    for directory in (b,s):
        require(read(directory/'status.json')['status']=='complete','Historical reference incomplete')
        metrics=read(directory/'metrics.json')
        require(sha(directory/'mask_node_statistics.npz')==metrics['mask_node_statistics_sha256'],'Historical node hash differs')
    data=read(b/'metrics.json')['dataset_info']
    require(data==read(s/'metrics.json')['dataset_info'] and data['nodes']==2708 and data['splits']['val_mask']==500,'Historical Cora inputs differ')
    args=read(b/'effective_args.json')
    require((args['max_epoch'],args['max_epoch_f'],args['mask_rate'],args['num_hidden'])==(2000,300,.5,1024),'Official Cora anchor differs')
    env=read(b/'environment.json')
    return dict(dataset_info=data,arguments=normal(args),environment={k:env[k] for k in ENV_KEYS},s_directory=str(s))

def command(directory,weights,method,seed,phase):
    return [sys.executable,'-u','record_full_batch_p.py','run','--dataset','cora','--method',method,'--seed',str(seed),
            '--phase',phase,'--device','0','--output-dir',str(directory),'--checkpoint-root',str(weights)]

def validate(directory,method,seed,phase,anchor,manifest,launch,reload):
    import numpy as np
    import torch
    # Use the bundled byte-identical pure helper; the stage process starts in PACKAGE.
    from progress_math import rank_weights,shuffle_seed
    require(sha(PACKAGE/'progress_math.py')==manifest['after_sha256']['models/progress_math.py'],'Bundled math helper differs from runtime source')
    status,metrics,args,env=[read(directory/n) for n in ('status.json','metrics.json','effective_args.json','environment.json')]
    for record in (status,metrics):
        for key,value in dict(status='complete',dataset='cora',method=method,seed=seed,phase=phase,
                              official_epoch_budget=phase=='development',recorder_version='4.0.0-progress-dev1').items():
            require(record.get(key)==value,'Run identity differs: '+key)
    expected=dict(anchor['arguments'])
    if phase=='diagnostic': expected.update(max_epoch=100,max_epoch_f=5,mask_log_interval=10)
    elif phase=='smoke': expected.update(max_epoch=5,max_epoch_f=5,mask_log_interval=1)
    require(normal(args)==expected and args['input_mask_mode']==MODES[method] and args['seeds']==[seed],'Effective arguments differ')
    require(metrics['dataset_info']==anchor['dataset_info'] and {k:env[k] for k in ENV_KEYS}==anchor['environment'],'Data or environment fingerprint differs')
    snapshot(directory,manifest['after_sha256'])
    epochs=args['max_epoch']; n=2708; k=1354
    steps=sorted(set(range(0,epochs,args['mask_log_interval']))|{epochs-1})
    feedback=[json.loads(line) for line in (directory/'mask_feedback.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    masks=[json.loads(line) for line in (directory/'mask_stats.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    require([r['step'] for r in feedback]==steps and [r['step'] for r in masks]==steps and metrics['feedback_records']==metrics['mask_records']==len(steps),'Logged steps differ')
    require(metrics['intermediate_probe_calls']==epochs//200,'Intermediate probes differ')
    require(sha(directory/'mask_node_statistics.npz')==metrics['mask_node_statistics_sha256'],'Node artifact hash differs')
    with np.load(directory/'mask_node_statistics.npz',allow_pickle=False) as f:
        nodes={name:f[name].copy() for name in f.files}
    common={'node_id','nonself_in_degree','reconstruction_error_ema','observation_count','local_feature_density','density_layer'}
    extra={'slow_reconstruction_error_ema','progress_score','last_observed_step'} if method!='S' else set()
    require(set(nodes)==common|extra and all(v.shape==(n,) and np.isfinite(v).all() for v in nodes.values()),'Node schema/shape/finite check failed')
    with np.load(Path(anchor['s_directory'])/'mask_node_statistics.npz',allow_pickle=False) as ref:
        for name in ('node_id','nonself_in_degree','local_feature_density','density_layer'):
            require(np.array_equal(nodes[name],ref[name]),'Density/input arrays differ')
    membership=nodes['density_layer']; counts=nodes['observation_count']
    require(np.issubdtype(counts.dtype,np.integer) and (counts>=0).all() and (counts<=epochs).all() and int(counts.sum())==epochs*k,'Observation budget differs')
    require((nodes['reconstruction_error_ema']>=0).all(),'Negative EMA')
    quotas=[339,339,338,338]
    for i in range(4): require(int(counts[membership==i].sum())==epochs*quotas[i],'Cumulative layer quota differs')
    for row in feedback:
        alpha=0. if method=='S' else .5*min(max((row['step']/max(epochs-1,1)-.2)/.6,0.),1.)
        require(row['mode']==MODES[method] and row['nodes']==n and row['masked']==k and math.isclose(row['alpha'],alpha,abs_tol=1e-12),'Logged sampler differs')
        require([x['nodes'] for x in row['density_layers']]==[677]*4 and [x['masked'] for x in row['density_layers']]==quotas,'Actual layer quotas differ')
        require(math.isclose(row['observation_count_min_mean_max'][1],k*(row['step']+1)/n,abs_tol=1e-8),'Feedback clock budget differs')
    require(feedback[-1]['observation_count_min_mean_max']==[int(counts.min()),float(counts.mean()),int(counts.max())],'Final count summary differs')
    if method!='S':
        require(metrics['rank_protocol']=='fixed-rank-linear-v1' and metrics['fast_beta']==.9 and metrics['slow_beta']==.99
                and metrics['rank_tie_seed']==2026110504 and metrics['rank_shuffle_seed']==2026110505,'Rank protocol differs')
        require(metrics['rank_trace_file']=='rank_trace.npz' and sha(directory/'rank_trace.npz')==metrics['rank_trace_sha256'],'Rank trace hash differs')
        require(np.array_equal(nodes['progress_score'],np.maximum(nodes['slow_reconstruction_error_ema']-nodes['reconstruction_error_ema'],0)),'Progress export differs')
        require((nodes['slow_reconstruction_error_ema']>=0).all(),'Negative slow EMA')
        with np.load(directory/'rank_trace.npz',allow_pickle=False) as f:
            traces={name:f[name].copy() for name in f.files}
        require(set(traces)=={'steps','tie_priority','fast','slow','scores','weights','observation_count','last_observed_step','masked'},'Trace schema differs')
        require(traces['steps'].tolist()==steps and sorted(traces['tie_priority'].tolist())==list(range(n)),'Trace index/tie schema differs')
        generator=torch.Generator(device='cpu').manual_seed(2026110504)
        require(np.array_equal(traces['tie_priority'],torch.randperm(n,generator=generator,device='cpu').numpy()),'Private tie replay differs')
        for name in set(traces)-{'steps','tie_priority'}:
            require(traces[name].shape==(len(steps),n) and np.isfinite(traces[name]).all(),'Invalid trace matrix')
        require((traces['fast']>=0).all() and (traces['slow']>=0).all(),'Negative pre-feedback EMA')
        require(np.issubdtype(traces['observation_count'].dtype,np.integer)
                and np.issubdtype(traces['last_observed_step'].dtype,np.integer),'Trace integer schema differs')
        for j,step in enumerate(steps):
            raw=traces['fast'][j] if method=='H' else np.maximum(traces['slow'][j]-traces['fast'][j],0)
            require(np.array_equal(raw,traces['scores'][j]),'Trace used a different scoring signal')
            before=traces['observation_count'][j]; last=traces['last_observed_step'][j]; masked=traces['masked'][j]
            require(int(before.sum())==step*k and (before>=0).all() and (before<=step).all(),'Pre-feedback observation clock differs')
            require(((before==0)==(last==-1)).all() and (last<step).all() and (last>=-1).all(),'Trace uses current or future feedback')
            require(np.isin(masked,[0,1]).all() and int(masked.sum())==k,'Recorded actual mask differs')
            alpha=feedback[j]['alpha']
            ordered_groups=np.array_split(np.argsort(nodes['local_feature_density'],kind='stable'),4)
            for layer in range(4):
                group=ordered_groups[layer]
                expected=np.asarray(rank_weights(raw[group].tolist(),traces['tie_priority'][group].tolist(),alpha))
                if method=='PS':
                    generator=torch.Generator(device='cpu').manual_seed(shuffle_seed(step,layer))
                    expected=expected[torch.randperm(len(group),generator=generator,device='cpu').numpy()]
                require(np.array_equal(expected,traces['weights'][j,group]),'Actual sampling weight replay differs')
                require(int(masked[group].sum())==quotas[layer],'Trace mask violates density quota')
            require(hashlib.sha256(traces['weights'][j].astype('<f8',copy=False).tobytes()).hexdigest()==feedback[j]['sampling_weight_sha256'],'Trace/feedback weight hash differs')
            if j:
                delta=step-steps[j-1]; diff=before-traces['observation_count'][j-1]
                require((diff>=0).all() and (diff<=delta).all(),'Trace count trajectory impossible')
        require(np.array_equal(traces['observation_count'][-1]+traces['masked'][-1],counts),'Final feedback is not exactly one update after last trace')
    launch_text=launch.read_text(encoding='utf-8'); reload_text=reload.read_text(encoding='utf-8')
    for marker in ('RECORDED_RUN=PASS','CHECKPOINT_READBACK=PASS'): require(marker in launch_text,'Missing save/readback marker')
    for marker in ('MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS','CHECKPOINT_SHA256=PASS'): require(marker in reload_text,'Missing CPU reload marker')
    require(metrics['checkpoint_readback_equal'] is True,'Checkpoint readback mismatch')
    require(0<=metrics['best_val_acc']<=1 and type(metrics['best_val_epoch']) is int and 0<=metrics['best_val_epoch']<args['max_epoch_f'],'Invalid final validation result')
    return dict(dataset='cora',method=method,seed=seed,best_val_acc=metrics['best_val_acc'],best_val_epoch=metrics['best_val_epoch'],
                run_directory=str(directory),resources=read(directory/'resources.json'),checkpoint_sha256=metrics['checkpoint_sha256'])

def measurement_gate(directory):
    """Frozen, performance-independent screening; never reads classifier metrics."""
    import numpy as np
    with np.load(directory/'rank_trace.npz',allow_pickle=False) as f:
        traces={name:f[name].copy() for name in f.files}
    def corr(a,b):
        return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and a.std()>0 and b.std()>0 else None
    rows=[]
    for j,step in enumerate(traces['steps']):
        if step<50: continue
        scores=traces['scores'][j]; counts=traces['observation_count'][j]
        ages=step-traces['last_observed_step'][j]; seen=counts>=5
        rows.append(dict(step=int(step),seen_five_fraction=float(seen.mean()),positive_fraction=float((scores[seen]>0).mean()) if seen.any() else 0.,
                         score_std=float(scores[seen].std()) if seen.any() else 0.,age_pearson=corr(scores[seen],ages[seen]),
                         count_pearson=corr(scores[seen],counts[seen]),error_pearson=corr(scores[seen],traces['fast'][j,seen])))
    reasons=[]
    if len(rows)!=6: reasons.append('expected six late diagnostic snapshots')
    if not rows or any(x['seen_five_fraction']<.99 or x['positive_fraction']<.05 or x['score_std']<=1e-8 for x in rows):
        reasons.append('insufficient observation coverage or degenerate progress signal')
    for key in ('age_pearson','count_pearson'):
        values=[abs(x[key]) for x in rows if x[key] is not None]
        if values and sum(values)/len(values)>=.8: reasons.append('strong '+key+' measurement confounding')
    return dict(status='PASS' if not reasons else 'HOLD',passed=not reasons,reasons=reasons,rows=rows,
                thresholds=dict(min_seen_five_fraction=.99,min_positive_fraction=.05,min_std=1e-8,max_mean_abs_correlation=.8),
                test_scores_used=False,validation_scores_used=False,
                interpretation='Heuristic measurement screen only; PASS does not establish true learnability, causality or downstream improvement')

def check_stage(repo,stage,manifest,anchor):
    require(stage.parent==repo/'outputs' and stage.name.startswith('p_dev1_stage1_'),'Invalid stage location')
    receipt=read(PACKAGE/'stage_receipt.json')
    require(Path(receipt['directory']).resolve()==stage and receipt['source_sha256']==manifest['after_sha256'],'Stage receipt differs')
    state=read(stage/'status.json'); require(state['status']=='complete','Stage is not complete')
    test_lines=(stage/'tests.log').read_text(encoding='utf-8').splitlines()
    require('P_DEV1_TEST_COUNT=114' in test_lines and 'P_DEV1_SKIPPED_COUNT=0' in test_lines and 'OK' in test_lines,'114 no-skip test evidence missing')
    for m in METHODS:
        run=stage/(stage.name+'_cora_'+m+'_s0')
        validate(run,m,0,'diagnostic',anchor,manifest,stage/('diagnostic_'+m+'.log'),stage/('reload_'+m+'.log'))
    actual=measurement_gate(stage/(stage.name+'_cora_P_s0'))
    require(actual==read(stage/'measurement_gate.json') and actual['passed'] and sha(stage/'measurement_gate.json')==receipt['gate_sha256'],'Measurement gate differs or is on hold')
    return actual
