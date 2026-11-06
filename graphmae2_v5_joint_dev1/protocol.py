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
METHODS=('S','NA','NR','NC')
MODES=dict(S='pair_uniform_reference',NA='pair_semantic_antithetic',NR='pair_random_antithetic',NC='pair_semantic_comask')
JOBS=[(m,s) for s,order in enumerate((('S','NA','NR','NC'),('NA','NR','NC','S'),('NR','NC','S','NA'))) for m in order]
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
    return [sys.executable,'-u','record_full_batch_j.py','run','--dataset','cora','--method',method,'--seed',str(seed),
            '--phase',phase,'--device','0','--output-dir',str(directory),'--checkpoint-root',str(weights)]

def prior_audit(repo):
    manifest=read(PACKAGE/'manifest.json')
    tool=repo/'graphmae2_p12_audit'
    receipt=read(tool/'audit_receipt.json')
    require(receipt['status']=='PASS' and receipt['source_sha256']==manifest['before_sha256'], 'Prior P12 audit receipt/source mismatch')
    require(receipt['auditor_sha256']==EXPECTED_AUDITOR_SHA and sha(tool/'audit_p12.py')==EXPECTED_AUDITOR_SHA, 'Prior independent auditor differs')
    require(Path(receipt['batch_directory']).resolve()==repo/'outputs/p_development12_by02jzbr', 'Wrong P12 audited batch')
    directory=Path(receipt['audit_directory']).resolve()
    require(directory.parent==repo/'outputs' and directory.name.startswith('p12_independent_audit_'), 'Unexpected audit location')
    require(sha(directory/'audit.json')==receipt['audit_sha256'], 'Prior audit report changed')
    result=read(directory/'audit.json')
    require(result['status']=='PASS' and result['counters']['runs']==16 and result['counters']['source_file_bytes']==768
            and result['counters']['cross_arm_weight_multisets']==592 and not result['test_metrics_opened']
            and not result['model_weights_opened'], 'Incomplete prior audit evidence')
    return dict(directory=str(directory),audit_sha256=receipt['audit_sha256'],auditor_sha256=EXPECTED_AUDITOR_SHA)

EXPECTED_AUDITOR_SHA='002901d6b8b92665677b534d5e54091494c33d1fcbf44abbce7ba8932307c231'

_REFERENCE_CACHE={}

def reference_plans(repo):
    if _REFERENCE_CACHE: return _REFERENCE_CACHE
    sys.path.insert(0,str(repo))
    from datasets.data_proc import load_small_dataset
    from models.joint_masking import JointInputMask
    graph,_=load_small_dataset('cora')
    for kind,mode in (('semantic','pair_semantic_antithetic'),('random','pair_random_antithetic')):
        sampler=JointInputMask(mode,log_interval=0)
        sampler.prepare(graph,'cpu',graph.ndata['feat'])
        _REFERENCE_CACHE[kind]=sampler.pair_plan()
    return _REFERENCE_CACHE

def validate(directory,method,seed,phase,anchor,manifest,launch,reload):
    import numpy as np
    status,metrics,args,env=[read(directory/n) for n in ('status.json','metrics.json','effective_args.json','environment.json')]
    for record in (status,metrics):
        for key,value in dict(status='complete',dataset='cora',method=method,seed=seed,phase=phase,
                              official_epoch_budget=phase=='development',recorder_version='5.0.0-joint-dev1').items():
            require(record.get(key)==value,'Run identity differs: '+key)
    expected=dict(anchor['arguments'])
    if phase=='diagnostic': expected.update(max_epoch=100,max_epoch_f=5,mask_log_interval=10)
    elif phase=='smoke': expected.update(max_epoch=5,max_epoch_f=5,mask_log_interval=1)
    require(normal(args)==expected and args['input_mask_mode']==MODES[method] and args['seeds']==[seed],'Arguments differ')
    require(metrics['dataset_info']==anchor['dataset_info'] and {k:env[k] for k in ENV_KEYS}==anchor['environment'],'Input/environment mismatch')
    snapshot(directory,manifest['after_sha256'])
    require(metrics['joint_protocol']=='equal-marginal-pair-v1' and metrics['pair_plan_file']=='pair_plan.json'
            and metrics['joint_trace_file']=='joint_trace.npz','Joint protocol identity differs')
    require(sha(directory/'pair_plan.json')==metrics['pair_plan_sha256'] and sha(directory/'joint_trace.npz')==metrics['joint_trace_sha256'],
            'Pair plan or full mask trace hash differs')
    plan=read(directory/'pair_plan.json')
    repo=directory.parents[2]
    expected_plan=reference_plans(repo)['random' if method=='NR' else 'semantic']
    require(plan==expected_plan,'Actual pair plan does not match deterministic input-only reconstruction')
    require(sha(directory/'mask_node_statistics.npz')==metrics['mask_node_statistics_sha256'],'Node array hash differs')
    with np.load(directory/'mask_node_statistics.npz',allow_pickle=False) as data:
        nodes={name:data[name].copy() for name in data.files}
    schema={'node_id','nonself_in_degree','reconstruction_error_ema','observation_count','local_feature_density','density_layer','pair_partner','pair_from_graph_matching'}
    n=2708; k=1354; epochs=args['max_epoch']; quotas=[339,339,338,338]
    require(set(nodes)==schema and all(x.shape==(n,) and np.isfinite(x).all() for x in nodes.values()),'Node schema/shape/finite check failed')
    with np.load(Path(anchor['s_directory'])/'mask_node_statistics.npz',allow_pickle=False) as old:
        for key in ('node_id','nonself_in_degree','local_feature_density','density_layer'):
            require(np.array_equal(nodes[key],old[key]),'True input/density arrays differ')
    counts=nodes['observation_count']; membership=nodes['density_layer']
    require(np.issubdtype(counts.dtype,np.integer) and (counts>=0).all() and (counts<=epochs).all() and int(counts.sum())==k*epochs,'Final observation budget differs')
    require((nodes['reconstruction_error_ema']>=0).all(),'Negative error EMA')
    with np.load(directory/'joint_trace.npz',allow_pickle=False) as data:
        require(set(data.files)=={'steps','masked'},'Full trace schema differs')
        steps=data['steps'].copy(); mask=data['masked'].copy()
    require(steps.tolist()==list(range(epochs)) and mask.shape==(epochs,n) and np.isin(mask,[0,1]).all(),'Full trace shape/step/value mismatch')
    require(np.array_equal(mask.sum(axis=1),np.full(epochs,k)) and np.array_equal(mask.sum(axis=0),counts),'Per-epoch or per-node full trace budget mismatch')
    partner=np.full(n,-1,dtype=np.int64); from_matching=np.zeros(n,dtype=np.uint8)
    covered=[]; all_pairs=[]; all_flags=[]; mixed_by_layer=[]
    for layer,group in enumerate(plan['groups']):
        pairs=np.asarray(group['pairs'],dtype=np.int64).reshape(-1,2); singleton=group['singleton']
        local=pairs.flatten().tolist()+([singleton] if singleton is not None else [])
        require(len(local)==len(set(local))==677 and all(membership[i]==layer for i in local),'Pair coverage/stratum mismatch')
        require(np.array_equal(mask[:,local].sum(axis=1),np.full(epochs,quotas[layer])),'An epoch violates fixed stratum quota')
        covered+=local; flags=np.asarray(group['from_graph_matching'],dtype=bool)
        require(flags.shape==(len(pairs),),'Pair flag schema mismatch')
        partner[pairs[:,0]]=pairs[:,1]; partner[pairs[:,1]]=pairs[:,0]
        from_matching[pairs.flatten()]=np.repeat(flags.astype(np.uint8),2)
        differing=(mask[:,pairs[:,0]]!=mask[:,pairs[:,1]]).sum(axis=1)
        mixed_by_layer.append(differing)
        remaining=quotas[layer]-(mask[:,singleton] if singleton is not None else 0)
        if method in ('NA','NR'):
            expected_mixed=np.minimum(remaining,2*len(pairs)-remaining)
            require(np.array_equal(differing,expected_mixed),'Antithetic coupling failed in an epoch')
        elif method=='NC': require(np.array_equal(differing,remaining%2),'Comasking coupling failed in an epoch')
        all_pairs.append(pairs); all_flags.append(flags)
    require(sorted(covered)==list(range(n)) and np.array_equal(partner,nodes['pair_partner'])
            and np.array_equal(from_matching,nodes['pair_from_graph_matching']),'Pair membership/export mismatch')
    pairs=np.concatenate(all_pairs); flags=np.concatenate(all_flags)
    feedback=[json.loads(line) for line in (directory/'mask_feedback.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    masks=[json.loads(line) for line in (directory/'mask_stats.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    logged=sorted(set(range(0,epochs,args['mask_log_interval']))|{epochs-1})
    require([x['step'] for x in feedback]==logged and [x['step'] for x in masks]==logged
            and metrics['feedback_records']==metrics['mask_records']==len(logged),'Feedback/logged budget mismatch')
    require(metrics['intermediate_probe_calls']==epochs//200,'Intermediate probe budget mismatch')
    for m,row in zip(masks,feedback):
        step=row['step']; selected=mask[step]
        differing=selected[pairs[:,0]]!=selected[pairs[:,1]]
        require(row['mode']==MODES[method] and row['alpha']==0. and row['masked']==k and row['nodes']==n,'Sampler identity or mask budget differs')
        require([x['nodes'] for x in row['density_layers']]==[677]*4 and [x['masked'] for x in row['density_layers']]==quotas,'Logged layer quota mismatch')
        require(row['joint_protocol']=='equal-marginal-pair-v1' and row['pairing_seed']==plan['pairing_seed'],'Logged joint protocol mismatch')
        require(math.isclose(row['pair_disagreement_fraction'],float(differing.mean()),abs_tol=1e-12)
                and math.isclose(row['graph_matched_pair_fraction'],float(flags.mean()),abs_tol=1e-12),'Joint diagnostic mismatch')
        if flags.any(): require(math.isclose(row['graph_matched_pair_disagreement_fraction'],float(differing[flags].mean()),abs_tol=1e-12),'Semantic pair diagnostic mismatch')
        require(math.isclose(row['observation_count_min_mean_max'][1],k*(step+1)/n,abs_tol=1e-8),'Observation clock mismatch')
        for key in ('step','mode','alpha','masked','nodes','actual_mask_rate'): require(m[key]==row[key],'Mask/feedback mismatch')
    require(feedback[-1]['observation_count_min_mean_max']==[int(counts.min()),float(counts.mean()),int(counts.max())],'Final feedback mismatch')
    launch_text=launch.read_text(encoding='utf-8'); reload_text=reload.read_text(encoding='utf-8')
    require(all(marker in launch_text for marker in ('RECORDED_RUN=PASS','CHECKPOINT_READBACK=PASS')),'Missing readback marker')
    require(all(marker in reload_text for marker in ('MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS','CHECKPOINT_SHA256=PASS')),'Missing CPU reload evidence')
    require(metrics['checkpoint_readback_equal'] is True and 0<=metrics['best_val_acc']<=1 and 0<=metrics['best_val_epoch']<args['max_epoch_f'],'Weight/validation metadata mismatch')
    return dict(dataset='cora',method=method,seed=seed,best_val_acc=metrics['best_val_acc'],best_val_epoch=metrics['best_val_epoch'],
                run_directory=str(directory),resources=read(directory/'resources.json'),checkpoint_sha256=metrics['checkpoint_sha256'],
                pair_plan_sha256=metrics['pair_plan_sha256'],joint_trace_sha256=metrics['joint_trace_sha256'],
                empirical_pair_disagreement_fraction=float(np.concatenate(mixed_by_layer).sum()/(epochs*len(pairs))))

def structural_gate(stage):
    plans={m:read(stage/(stage.name+'_cora_'+m+'_s0')/'pair_plan.json') for m in METHODS}
    require(plans['S']==plans['NA']==plans['NC'],'S/NA/NC must share exactly the same semantic pairing')
    def stats(plan):
        values=[x for group in plan['groups'] for x in group['pair_feature_cosine']]
        flags=[x for group in plan['groups'] for x in group['from_graph_matching']]
        return dict(mean_pair_cosine=sum(values)/len(values),graph_matching_pair_fraction=sum(flags)/len(flags),pairs=len(values))
    semantic=stats(plans['NA']); random=stats(plans['NR']); reasons=[]
    if semantic['graph_matching_pair_fraction']<.1: reasons.append('less than 10% of semantic pairs came from graph matching')
    if semantic['mean_pair_cosine']-random['mean_pair_cosine']<=1e-6: reasons.append('semantic pairing did not increase mean feature cosine over random pairing')
    return dict(status='PASS' if not reasons else 'HOLD',passed=not reasons,reasons=reasons,semantic=semantic,random=random,
                thresholds=dict(min_graph_matching_fraction=.1,min_cosine_difference=1e-6),test_scores_used=False,validation_scores_used=False,
                interpretation='Input-only contrast/manipulation check; no downstream performance guarantee; independent P12 audit required before installation')

def check_stage(repo,stage,manifest,anchor):
    require(stage.parent==repo/'outputs' and stage.name.startswith('j_dev1_stage1_'),'Invalid stage directory')
    prior_audit(repo)
    receipt=read(PACKAGE/'stage_receipt.json')
    require(Path(receipt['directory']).resolve()==stage and receipt['source_sha256']==manifest['after_sha256'],'Stage receipt/source mismatch')
    require(read(stage/'status.json')['status']=='complete','Stage incomplete')
    test_lines=(stage/'tests.log').read_text(encoding='utf-8').splitlines()
    require('J_DEV1_TEST_COUNT=134' in test_lines and 'J_DEV1_SKIPPED_COUNT=0' in test_lines and 'OK' in test_lines,'134 no-skip tests evidence missing')
    for method in METHODS:
        run=stage/(stage.name+'_cora_'+method+'_s0')
        validate(run,method,0,'diagnostic',anchor,manifest,stage/('diagnostic_'+method+'.log'),stage/('reload_'+method+'.log'))
    gate=structural_gate(stage)
    require(gate==read(stage/'structural_gate.json') and gate['passed'] and sha(stage/'structural_gate.json')==receipt['gate_sha256'],'Structural gate differs/is on HOLD')
    return gate
