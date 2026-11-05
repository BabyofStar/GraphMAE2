"""C protocol guards; source/legacy helpers are preserved from approved packages."""
import hashlib
import math
from pathlib import Path

import reference_r9 as support
from reference_r9 import require,read,sha,write,locked,execute

PACKAGE=Path(__file__).resolve().parent
ROUTE_SEEDS={'cora':2026110501,'citeseer':2026110502,'pubmed':2026110503}


def identity():
    value=read(PACKAGE/'queue_identity.json')
    for name,expected in value['queue_files_sha256'].items():
        require(sha(PACKAGE/name)==expected,'C package file differs: '+name)
    return value


def sources(repo):
    identity()
    manifest=read(PACKAGE/'manifest.json')
    for name,expected in manifest['after_sha256'].items():
        path=repo/name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(repo)
                and sha(path)==expected,'C source differs: '+name)
    return manifest


def anchors(repo):
    value=identity()
    path=repo/'graphmae2_v2_r_dev1/manifest.json'
    require(sha(path)=='d9189ce6c30e5082eacdd60c2f0c2ae4d051b64fdd33db8e3a9f333356c982b5','Accepted R manifest differs')
    return support.load_anchors(repo,read(path),value)


def normal(args):
    return {k:v for k,v in args.items() if k not in ('seeds','input_mask_mode','random_group_seed','density_route_seed')}


def check_stage(repo,manifest,loaded_anchors=None):
    receipt=read(PACKAGE/'stage_receipt.json')
    directory=Path(receipt['directory']).resolve()
    require(directory.parent==repo/'outputs' and receipt['source_sha256']==manifest['after_sha256']
            and receipt['test_count']==94,'C stage receipt differs')
    require(read(directory/'status.json')['status']=='complete','C engineering acceptance is incomplete')
    lines=(directory/'tests.log').read_text(encoding='utf-8').splitlines()
    require('C_DEV1_TEST_COUNT=94' in lines and 'C_DEV1_SKIPPED_COUNT=0' in lines,'C 94-test/no-skip evidence missing')
    if loaded_anchors is None:
        _,loaded_anchors=anchors(repo)
    smoke=directory/(directory.name+'_cora_C_s0')
    reference=dict(loaded_anchors['cora',0]['B0'],s_directory=loaded_anchors['cora',0]['S']['run_directory'])
    validate(smoke,'cora',0,'smoke',reference,manifest)
    require('MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS' in (directory/'reload_C.log').read_text(encoding='utf-8'),'C smoke CPU reload evidence missing')
    return directory


def validate_plan(plan,dataset,epochs):
    require(plan['density_route_seed']==ROUTE_SEEDS[dataset] and plan['epochs']==epochs
            and plan['warmup_ratio']==.2 and plan['ramp_end_ratio']==.8 and plan['probability_max']==1.,'C route parameters differ')
    routes,prob=plan['routes'],plan['route_probabilities']
    require(len(routes)==len(prob)==epochs and all(type(x) is int and x in (0,1) for x in routes),'C route schema differs')
    expected=[min(max((t/max(epochs-1,1)-.2)/(.8-.2),0.),1.) for t in range(epochs)]
    require(all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(prob,expected)),'C route probability schedule differs')
    require(all((p!=0 or r==0) and (p!=1 or r==1) for p,r in zip(prob,routes)),'C forced endpoint routes differ')
    require(plan['density_epochs']==sum(routes) and plan['uniform_epochs']==epochs-sum(routes)
            and plan['routes_sha256']==hashlib.sha256(bytes(routes)).hexdigest(),'C route count/hash differs')


def validate(directory,dataset,seed,phase,reference,manifest):
    status,metrics,args=[read(directory/n) for n in ('status.json','metrics.json','effective_args.json')]
    for row in (status,metrics):
        for key,value in dict(status='complete',dataset=dataset,method='C',seed=seed,phase=phase,
                              official_epoch_budget=phase=='development',recorder_version='3.0.0-density-dev1').items():
            require(row.get(key)==value,'C run metadata differs: '+key)
    expected=dict(reference['nonmethod_args'])
    if phase=='smoke':
        expected.update(max_epoch=5,max_epoch_f=5,mask_log_interval=1)
    require(normal(args)==expected,'C nonmethod arguments differ')
    require(args['seeds']==[seed] and args['input_mask_mode']=='density_curriculum'
            and args['density_route_seed']==ROUTE_SEEDS[dataset]
            and args['random_group_seed']==support.SEEDS[dataset],'C mode/seeds differ')
    require(metrics['dataset_info']==reference['dataset_info'],'C graph/features/splits differ')
    require(support.env_signature(directory)==reference['environment_signature'],'C runtime differs')
    support.verify_snapshot(directory,manifest['after_sha256'])
    path=directory/metrics['density_route_plan_file']
    require(path.name=='density_route_plan.json' and sha(path)==metrics['density_route_plan_sha256'],'C route plan file differs')
    plan=read(path)
    epochs=args['max_epoch']
    validate_plan(plan,dataset,epochs)
    import torch
    generator=torch.Generator(device='cpu')
    generator.manual_seed(ROUTE_SEEDS[dataset])
    draws=torch.rand(epochs,generator=generator,device='cpu').tolist()
    require(plan['routes']==[int(u<p) for u,p in zip(draws,plan['route_probabilities'])],'C private seeded route replay differs')
    n=metrics['dataset_info']['nodes']
    k=int(n*args['mask_rate'])
    sizes=[n//4+(i<n%4) for i in range(4)]
    quotas=support.quotas(sizes,k)
    feedback=[support.json.loads(line) for line in (directory/'mask_feedback.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    interval=1 if phase=='smoke' else 100
    steps=sorted(set(range(0,epochs,interval))|{epochs-1})
    require([x['step'] for x in feedback]==steps and metrics['feedback_records']==metrics['mask_records']==len(steps),'C logged steps/counters differ')
    for row in feedback:
        step=row['step']
        require(row['mode']=='density_curriculum' and row['nodes']==n and row['masked']==k and row['alpha']==0.,'C mask budget/weight semantics differ')
        require(row['density_route_seed']==ROUTE_SEEDS[dataset] and row['density_routes_sha256']==plan['routes_sha256']
                and row['density_route_probability']==plan['route_probabilities'][step]
                and row['density_route']==('density' if plan['routes'][step] else 'uniform'),'C logged route differs')
        require([x['layer'] for x in row['density_layers']]==list(range(4)) and [x['nodes'] for x in row['density_layers']]==sizes
                and sum(x['masked'] for x in row['density_layers'])==k,'C true density diagnostics differ')
        if plan['routes'][step]:
            require([x['masked'] for x in row['density_layers']]==quotas,'C density-route quotas differ')
        require(math.isclose(row['observation_count_min_mean_max'][1],k*(step+1)/n,rel_tol=1e-9,abs_tol=1e-8),'C observation budget differs')
    node_path=directory/metrics['mask_node_statistics_file']
    require(sha(node_path)==metrics['mask_node_statistics_sha256'],'C NPZ hash differs')
    import numpy as np
    with np.load(node_path,allow_pickle=False) as nodes,np.load(Path(reference['s_directory'])/'mask_node_statistics.npz',allow_pickle=False) as s:
        require(set(nodes.files)=={'node_id','nonself_in_degree','reconstruction_error_ema','observation_count','local_feature_density','density_layer'},'C node schema differs')
        for name in ('node_id','nonself_in_degree','local_feature_density','density_layer'):
            require(np.array_equal(nodes[name],s[name]),'C true graph/density arrays differ: '+name)
        counts=nodes['observation_count']
        require(counts.shape==(n,) and np.issubdtype(counts.dtype,np.integer) and (counts>=0).all()
                and (counts<=epochs).all() and int(counts.sum())==k*epochs,'C observation array budget differs')
        errors=nodes['reconstruction_error_ema']
        require(errors.shape==(n,) and np.isfinite(errors).all() and (errors>=0).all(),'C error diagnostics invalid')
    val,epoch=metrics['best_val_acc'],metrics['best_val_epoch']
    require(isinstance(val,(float,int)) and math.isfinite(val) and 0<=val<=1 and isinstance(epoch,int) and 0<=epoch<args['max_epoch_f'],'C validation metric invalid')
    return dict(dataset=dataset,method='C',seed=seed,best_val_acc=val,best_val_epoch=epoch,
                resources=read(directory/'resources.json'),run_directory=str(directory),route_plan=plan,
                checkpoint_sha256=metrics['checkpoint_sha256'])
