"""Independent P12 evidence audit; reads numeric artifacts, never model weights/test scores."""
import argparse
from collections import Counter
import contextlib
import datetime
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import re
import statistics
import sys
import tarfile
import tempfile

EXPECTED_SOURCE = {'configs/citeseer.yaml': 'd9d9302aca6148a322861bb160300ff32e97e615f0638771ea1f5a2d9f3d7688',
 'configs/cora.yaml': '3a5a6b87a73fd986ed0bd94ce224f37059739a7df7defa3cc89e6ad78f8aaca6',
 'configs/pubmed.yaml': '19f7897bab0097f2a6f3a85f9a7b02914cbb0e6627701832ea33c07aeb48a50a',
 'datasets/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'datasets/data_proc.py': 'd43cc47fb81e3d021dbaa15438b7a117054a5172855372f7fc4e17475671f88f',
 'datasets/lc_sampler.py': '0aca1c668e93bd6842220396a216a098b10e29aa9b3eb416ddd1fac472fea2cb',
 'datasets/localclustering.py': 'a94c0b9f9c7db2656e75614aa66af6de2a2f8de82c0dab9609d7c490267aaf0c',
 'datasets/saint_sampler.py': '729cca3522d9927b65df00e8e6234ccf8f64140eddabe3332ee03c7eb4412a8e',
 'main_full_batch.py': '10567a6173362ca3aca12218fce59d30597b8145ecbbddb4b604340ac9f7d0d2',
 'models/__init__.py': '63046c62a7cb961a94ede1b383c5e83f9908a39205ca199fdb0403b991963532',
 'models/adaptive_masking.py': '136b4f2d9cd86fceed091417b64ba7aaba8bb8cea11794175e2c8cc8aa996420',
 'models/density_curriculum.py': 'b55aa35c4b843946b237e262ad45d02716e0246bbb936f8db1f70232dfa0ebbb',
 'models/density_route_math.py': 'e7e9f6397715008b55755e1bff4c8eb7990ab654f56325e03204fa55ee3164e5',
 'models/edcoder.py': 'fedd504b2fb44c7b28e67a9871e8d8d06ce364fd89ecd42227221dbc6acd0b77',
 'models/finetune.py': '3c3f6e8f1154035a9a6d62420dc964f600cf672116c4fb3b12e7bb10f71de248',
 'models/gat.py': '2e536f955793d4190bfae9ed4161bedab53928ed43d2012bbb9e5e126d0a4a3c',
 'models/gcn.py': '55e3f0fac56ce78d7ed4856f83a3a84d61edde00be454f2db885b66ab992ac70',
 'models/input_masking.py': '230a45cf29255bfc46272f8774e5650aea46e85e48b558f82b286019e7d5d4b9',
 'models/loss_func.py': 'b71a86e278b039ea77a2943b3ead804bb222ad28e437c379a1c22dd02e30140e',
 'models/masking_math.py': 'ff32da677a5600dcc2a812f5a76c1cbada66875295bcdff9a0388d01f099a9aa',
 'models/progress_masking.py': '1598de4964c93b1be748455dde5a43bb1c34dc9ba10428479c987ef75bf8ac4b',
 'models/progress_math.py': '2ffb9626651f045fe5f07b915f9e55227ca0b553d126773cfa34e488743eeb93',
 'models/random_group_masking.py': '57d9f8acccf18d74923a7dbd0cb085a636080184ea5617198979910b7d81f8be',
 'record_full_batch.py': '7744e2e3ed17305b5b8428bc04f771c3a64eaa3e53b4c6331427c9055f025c95',
 'record_full_batch_c.py': '91677c90bf6f1b7f534b916d8db7313411443b6271776be708abf06a028f8244',
 'record_full_batch_p.py': '56937375ec19dbbf043ec948a7c641093d10e37139365c79d944073a2aff7992',
 'record_full_batch_r.py': '2226412317b8658d5089cfc3db142d1f4d5123eb00364c0482ec254c029a3bc4',
 'record_full_batch_v2.py': '033012c493b602357e4fe55d3758c91586593b6b27210f5cf9dcfd9592ef8fc0',
 'test_record_full_batch.py': '2cb961fe1b54fb90495c01c0a8a01d5f4f2146a3b02909cb8e3811345f25fbd9',
 'test_record_full_batch_c.py': 'b6bd8d6ec12f5391a9d540a7e1c5eb7bdc500ab5272534ae3d280f94ee6488d9',
 'test_record_full_batch_p.py': '21226fa6039c7e32cb7df7cdbcccdbd55f1e450cbc1ca65afa7ae06321216bb4',
 'test_record_full_batch_r.py': 'a7aa57b6fc9a37f8c2007e2bb3145e50827e59415f6311a5f480b52920bea6b5',
 'test_record_full_batch_v2.py': '84870be035acbdef9116d18450954f9fee904df8edcf8c4f42f2ebc5d513afed',
 'tests/fixtures/adaptive_masking_v2_dev1.py': '1709e8a1ab5029ea75136253f93272c8fdb58d843a2755ec39f9a99023062939',
 'tests/fixtures/edcoder_c_dev1_reference.py': 'a7268e0fd0441cdb54cc468f1bc98b036cd15bf0274aefc45fbdbd57e37868b9',
 'tests/fixtures/edcoder_r_dev1_reference.py': '34c8cbda5449ff3f65aad86e17169f94e9b5dc0eb90481934b4b30668adab9ab',
 'tests/fixtures/edcoder_v1_2_reference.py': 'b7994fdce33baf76964afbad1fb5f25d0091321e5d4985c6998754f38f77af44',
 'tests/fixtures/edcoder_v2_dev1.py': '7268fb466da2a6a67cbd987ae35756933c1df7b61fd885445f0ec870e88c6d28',
 'tests/fixtures/input_masking_v1_1_reference.py': '1c73f7899c9338cae8c0420b6a8ce4223561309894f42f49be028e8ef9a06518',
 'tests/test_adaptive_masking_v2.py': 'c4ae7e18019b18bc5679bcf983ab25649e76118e2bf0b033ccf7cbd929f11335',
 'tests/test_density_curriculum.py': '09a38526fc253c8e4bc7f75166c30c7e4fc69736962ed04062c72d956d0c8bb7',
 'tests/test_density_route_math.py': '74cfc2b0ba4f692b53ca13391286999b2eb7e1b6e0af6dd863e1ad6949b2edac',
 'tests/test_input_masking.py': '574e801ca550f28cd7b4a8626e58654002cab2c0d9acd7506070538e8c7eea9a',
 'tests/test_masking_math_v2.py': 'ec1af1f30a4773bf5dec51778b92b17f39f5c1eac9615f5df11bcad9a4b89296',
 'tests/test_progress_masking.py': '8afd9b67c07e8d97d742085c928ed8924f211b8a31fe30af6542b14dadbcdf69',
 'tests/test_progress_math.py': 'bebff6e5477082e9198799d28386d8b41980c3a1479975ea3a2f9021adbd57cb',
 'tests/test_random_group_masking.py': '4c27d598a0fc921176bd450a4b03c8e485963112671afed334bbbf6cdf782d59',
 'utils.py': '27ebc33f1648a88e9a2a5e9a819007328855a5484a735e624e5fd1e742f23c25'}
EXPECTED_QUEUE = {'approved_full_runs': 12,
 'dataset': 'cora',
 'diagnostic_pretrain_epochs': 100,
 'diagnostic_probe_epochs': 5,
 'diagnostic_runs': 4,
 'jobs': [['S', 0],
          ['H', 0],
          ['P', 0],
          ['PS', 0],
          ['H', 1],
          ['P', 1],
          ['PS', 1],
          ['S', 1],
          ['P', 2],
          ['PS', 2],
          ['S', 2],
          ['H', 2]],
 'measurement_gate_uses_validation_scores': False,
 'methods': ['S', 'H', 'P', 'PS'],
 'package': 'graphmae2_v4_progress_dev1',
 'primary_comparison': 'P_minus_S',
 'queue_files_sha256': {'README.md': 'f8751d5ca1ac1edc64c8bf5de2effdb410ed2a9ca51729f7f44bef6ec1472430',
                        'graphmae2_v4_progress_dev1.patch': 'ad2e5c086f59946703943209eacdbc692c503c983a7dff6fde75251e9b451c7d',
                        'install_p.py': 'f7ef449582055768fed536d96bb67e57f1e4ddb9e970b3d8df813e2c67beb878',
                        'launch_p12.py': '62fd9afbd88637059698fb2c5dc7027f9fe21b5aa5c68bf9a1d64065ba758f43',
                        'local_validation.json': '2a97d98a56f9a7313e215a9af2c5bd61699377fd3bb9453b40a6f1de6d43dd6b',
                        'manifest.json': '0803e17fef83c947094245effb229cedb01c4430171818527fb229295ef54791',
                        'progress_math.py': '2ffb9626651f045fe5f07b915f9e55227ca0b553d126773cfa34e488743eeb93',
                        'protocol.py': '67547174cf76af6bb8f263decc850e1810ced6237ec9e02e0b07a7d90ab985b2',
                        'run_p12.py': '1f2bafdd0d07bc21355ab9cc3c897fcc5d600fb9efe6167f875fcc4157802482',
                        'run_stage1_p.py': 'bbf11f9882f869491fd5b6be041837a3ed38fb3568d2435e20852e6204e1c8d5',
                        'start.sh': 'd3b9946fd437e3599a1e91f350a1ef58673b7486136b5dad21ef1ec34082e54f',
                        'test_p_queue.py': 'f243e15c40d9ba92dd2f927d773bd211e107010bba4a7ab2feed5a2ac692e75c'},
 'secondary_comparisons': ['P_minus_H', 'P_minus_PS'],
 'server_test_count': 114,
 'source_files': 48,
 'test_metrics_used_for_selection': False,
 'training_seeds': [0, 1, 2],
 'version': 1}
EXPECTED_MANIFEST_SHA = '0803e17fef83c947094245effb229cedb01c4430171818527fb229295ef54791'
METHODS=('S','H','P','PS')
MODES=dict(S='density_stratified',H='rank_difficulty',P='rank_progress',PS='rank_progress_shuffled')
JOBS=[(m,s) for s,order in enumerate((('S','H','P','PS'),('H','P','PS','S'),('P','PS','S','H'))) for m in order]
ENV_KEYS=('versions','torch_cuda_runtime','torch_cpu_threads','deterministic_algorithms','cudnn_deterministic',
          'cudnn_benchmark','cuda_matmul_allow_tf32','cudnn_allow_tf32','gpu_name','visible_gpu_memory_bytes','runtime_environment')
COUNTERS=Counter()
EVIDENCE={}

def require(condition,message):
    if not condition: raise RuntimeError(message)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    path=Path(path); EVIDENCE[str(path)]=sha(path)
    return json.loads(path.read_text(encoding='utf-8'),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))

def text(path):
    path=Path(path); EVIDENCE[str(path)]=sha(path)
    return path.read_text(encoding='utf-8')

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def normal(args): return {k:v for k,v in args.items() if k not in {'seeds','input_mask_mode','random_group_seed','density_route_seed'}}

def snapshot(directory):
    require(read(directory/'source_sha256.json')==EXPECTED_SOURCE,'Source manifest mismatch: '+directory.name)
    found=set()
    with tarfile.open(directory/'source.tar','r:') as archive:
        for member in archive:
            path=PurePosixPath(member.name)
            require(member.isfile() and member.name in EXPECTED_SOURCE and member.name not in found
                    and not path.is_absolute() and '..' not in path.parts and '\\' not in member.name and member.size<=10*1024*1024,
                    'Unexpected source entry')
            with archive.extractfile(member) as stream:
                require(hashlib.sha256(stream.read()).hexdigest()==EXPECTED_SOURCE[member.name],'Source bytes mismatch: '+member.name)
            found.add(member.name)
    require(found==set(EXPECTED_SOURCE),'Incomplete source snapshot')
    EVIDENCE[str(directory/'source.tar')]=sha(directory/'source.tar'); COUNTERS['source_file_bytes']+=len(found)

def arrayfile(path):
    import numpy as np
    EVIDENCE[str(path)]=sha(path)
    with np.load(path,allow_pickle=False) as source:
        require(len(source.files)<=20 and len(set(source.files))==len(source.files),'Invalid array archive')
        arrays={name:source[name].copy() for name in source.files}
    require(all(v.dtype.kind in 'iufb' and v.size<=200000 and np.isfinite(v).all() for v in arrays.values()),'Invalid/oversize numeric array')
    return arrays

def correlation(a,b):
    import numpy as np
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and a.std()>0 and b.std()>0 else None

def audit_run(directory,method,seed,phase,anchor,reference_nodes,launch,reload):
    import numpy as np
    import torch
    status,metrics,args,env=map(lambda name:read(directory/name),('status.json','metrics.json','effective_args.json','environment.json'))
    for record in (status,metrics):
        for key,value in dict(status='complete',dataset='cora',method=method,seed=seed,phase=phase,
                              official_epoch_budget=phase=='development',recorder_version='4.0.0-progress-dev1').items():
            require(record.get(key)==value,'Identity mismatch: '+directory.name+'/'+key)
    expected=dict(anchor['arguments'])
    if phase=='diagnostic': expected.update(max_epoch=100,max_epoch_f=5,mask_log_interval=10)
    require(normal(args)==expected and args['seeds']==[seed] and args['input_mask_mode']==MODES[method],'Argument mismatch')
    require(metrics['dataset_info']==anchor['dataset_info'] and {k:env[k] for k in ENV_KEYS}==anchor['environment'],'Input/environment mismatch')
    snapshot(directory)
    epochs=args['max_epoch']; k=1354; n=2708
    feedback=[json.loads(row) for row in text(directory/'mask_feedback.jsonl').splitlines() if row.strip()]
    masks=[json.loads(row) for row in text(directory/'mask_stats.jsonl').splitlines() if row.strip()]
    steps=sorted(set(range(0,epochs,args['mask_log_interval']))|{epochs-1})
    require([r['step'] for r in feedback]==steps and [r['step'] for r in masks]==steps
            and metrics['feedback_records']==metrics['mask_records']==len(steps),'Logged steps mismatch')
    require(metrics['intermediate_probe_calls']==epochs//200,'Probe budget mismatch')
    training=text(directory/'train.log')
    require({int(x) for x in re.findall(r'# Epoch (\d+):',training)}==set(range(epochs)),'Incomplete pretraining epoch log')
    require({int(x) for x in re.findall(r'# Epoch: (\d+),',training)}==set(range(args['max_epoch_f'])),'Incomplete final probe log')
    require(not re.search(r'(?:test_loss|test_acc|early-stopping-TestAcc|TestAcc):\s*[0-9]',training),'Unredacted test values found')
    final=re.findall(r'Best ValAcc:\s*([0-9.]+)\s+in epoch\s+(\d+)',training)[-1]
    require(math.isclose(float(final[0]),metrics['best_val_acc'],abs_tol=1e-8) and int(final[1])==metrics['best_val_epoch'],'Final validation log mismatch')
    require(sha(directory/'mask_node_statistics.npz')==metrics['mask_node_statistics_sha256'],'Node artifact hash mismatch')
    nodes=arrayfile(directory/'mask_node_statistics.npz')
    common={'node_id','nonself_in_degree','reconstruction_error_ema','observation_count','local_feature_density','density_layer'}
    extra={'slow_reconstruction_error_ema','progress_score','last_observed_step'} if method!='S' else set()
    require(set(nodes)==common|extra and all(v.shape==(n,) for v in nodes.values()),'Node array schema mismatch')
    for name in ('node_id','nonself_in_degree','local_feature_density','density_layer'):
        require(np.array_equal(nodes[name],reference_nodes[name]),'Node input mismatch: '+name)
    counts=nodes['observation_count']; membership=nodes['density_layer']
    require(counts.dtype.kind in 'iu' and (counts>=0).all() and (counts<=epochs).all() and int(counts.sum())==epochs*k,'Observation budget mismatch')
    require((nodes['reconstruction_error_ema']>=0).all(),'Negative error')
    quotas=[339,339,338,338]
    for layer,q in enumerate(quotas): require(int(counts[membership==layer].sum())==epochs*q,'Cumulative stratum budget mismatch')
    for mask,row in zip(masks,feedback):
        step=row['step']; alpha=0. if method=='S' else .5*min(max((step/max(epochs-1,1)-.2)/(.8-.2),0.),1.)
        require(row['mode']==MODES[method] and row['nodes']==n and row['masked']==k and math.isclose(row['alpha'],alpha,abs_tol=1e-12),'Sampler metadata mismatch')
        require([x['nodes'] for x in row['density_layers']]==[677]*4 and [x['masked'] for x in row['density_layers']]==quotas,'Stratum quota mismatch')
        require(math.isclose(row['observation_count_min_mean_max'][1],k*(step+1)/n,abs_tol=1e-8),'Feedback clock mismatch')
        for key in ('step','mode','alpha','masked','nodes','actual_mask_rate'): require(mask[key]==row[key],'Mask/feedback disagreement')
    require(feedback[-1]['observation_count_min_mean_max']==[int(counts.min()),float(counts.mean()),int(counts.max())],'Final count summary mismatch')
    traces=None
    if method!='S':
        require(metrics.get('rank_protocol')=='fixed-rank-linear-v1' and metrics.get('fast_beta')==.9 and metrics.get('slow_beta')==.99
                and metrics.get('rank_tie_seed')==2026110504 and metrics.get('rank_shuffle_seed')==2026110505,'Rank protocol mismatch')
        require(sha(directory/'rank_trace.npz')==metrics['rank_trace_sha256'],'Rank trace hash mismatch')
        require(np.array_equal(nodes['progress_score'],np.maximum(nodes['slow_reconstruction_error_ema']-nodes['reconstruction_error_ema'],0)),
                'Exported progress mismatch')
        require((nodes['slow_reconstruction_error_ema']>=0).all(),'Negative slow error')
        traces=arrayfile(directory/'rank_trace.npz')
        require(set(traces)=={'steps','tie_priority','fast','slow','scores','weights','observation_count','last_observed_step','masked'},'Trace schema mismatch')
        require(traces['steps'].tolist()==steps,'Trace steps mismatch')
        for name in set(traces)-{'steps','tie_priority'}: require(traces[name].shape==(len(steps),n),'Trace shape mismatch')
        rng_before=torch.get_rng_state().clone()
        generator=torch.Generator(device='cpu').manual_seed(2026110504)
        require(np.array_equal(traces['tie_priority'],torch.randperm(n,generator=generator,device='cpu').numpy()),'Tie priority replay mismatch')
        groups=np.array_split(np.argsort(nodes['local_feature_density'],kind='stable'),4)
        for j,step in enumerate(steps):
            raw=traces['fast'][j] if method=='H' else np.maximum(traces['slow'][j]-traces['fast'][j],0)
            require(np.array_equal(raw,traces['scores'][j]),'Pre-feedback score mismatch')
            before=traces['observation_count'][j]; last=traces['last_observed_step'][j]; masked=traces['masked'][j]
            require(before.dtype.kind in 'iu' and last.dtype.kind in 'iu' and (before>=0).all() and (before<=step).all()
                    and int(before.sum())==step*k and ((before==0)==(last==-1)).all() and (last<step).all() and (last>=-1).all(),
                    'Current/future feedback or invalid pre-selection clock')
            require(np.isin(masked,[0,1]).all() and int(masked.sum())==k,'Trace mask budget mismatch')
            alpha=feedback[j]['alpha']
            for layer,group in enumerate(groups):
                order=np.lexsort((traces['tie_priority'][group],raw[group]))
                m=len(group); template=np.asarray([(1-alpha)/m+alpha*2*r/(m*(m+1)) for r in range(1,m+1)])
                weights=np.empty(m); weights[order]=template
                if method=='PS':
                    generator=torch.Generator(device='cpu').manual_seed(2026110505+4*int(step)+layer)
                    weights=weights[torch.randperm(m,generator=generator,device='cpu').numpy()]
                require(np.array_equal(weights,traces['weights'][j,group]),'Independent ranked weight/PS replay mismatch')
                require(int(masked[group].sum())==quotas[layer],'Actual trace mask quota mismatch')
            require(hashlib.sha256(traces['weights'][j].astype('<f8',copy=False).tobytes()).hexdigest()==feedback[j]['sampling_weight_sha256'],
                    'Pre-feedback weight hash mismatch')
            if j:
                difference=before-traces['observation_count'][j-1]; interval=step-int(steps[j-1])
                require((difference>=0).all() and (difference<=interval).all(),'Impossible count trajectory')
        require(torch.equal(rng_before,torch.get_rng_state()),'Audit replay unexpectedly changed global CPU RNG')
        require(np.array_equal(traces['observation_count'][-1]+traces['masked'][-1],counts),'Final observation is not one update after trace')
        COUNTERS['rank_trace_snapshots']+=len(steps)
    launch_log=text(launch); reload_log=text(reload)
    require(all(marker in launch_log for marker in ('RECORDED_RUN=PASS','CHECKPOINT_READBACK=PASS')),'Readback evidence missing')
    require(all(marker in reload_log for marker in ('MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS','CHECKPOINT_SHA256=PASS')),'CPU reload evidence missing')
    require(metrics['checkpoint_readback_equal'] is True and re.fullmatch('[a-f0-9]{64}',metrics['checkpoint_sha256']) is not None,'Checkpoint metadata mismatch')
    resources=read(directory/'resources.json')
    require(0<=metrics['best_val_acc']<=1 and 0<=metrics['best_val_epoch']<args['max_epoch_f'],'Invalid validation score')
    late=[r for r in feedback if r['step']/max(epochs-1,1)>=.8]
    diagnostic=dict(count_cv=float(counts.std()/counts.mean()),error_ema_mean=float(nodes['reconstruction_error_ema'].mean()),
                    late_zero_visible_two_hop=statistics.mean(r['masked_two_hop_zero_visible_fraction'] for r in late),
                    pretrain_seconds=resources['pretrain_including_intermediate_probes']['wall_seconds'])
    if method!='S':
        progress=nodes['progress_score']; ages=epochs-1-nodes['last_observed_step']
        diagnostic.update(progress_positive_fraction=float((progress>0).mean()),progress_age_pearson=correlation(progress,ages),
                          progress_count_pearson=correlation(progress,counts))
    COUNTERS['runs']+=1; COUNTERS['node_arrays']+=len(nodes); COUNTERS['feedback_records']+=len(feedback)
    return dict(method=method,seed=seed,best_val_acc=metrics['best_val_acc'],best_val_epoch=metrics['best_val_epoch'],diagnostic=diagnostic),traces

def compute_gate(traces):
    import numpy as np
    rows=[]
    for j,step in enumerate(traces['steps']):
        if step<50: continue
        scores=traces['scores'][j]; counts=traces['observation_count'][j]; ages=step-traces['last_observed_step'][j]; seen=counts>=5
        rows.append(dict(step=int(step),seen_five_fraction=float(seen.mean()),positive_fraction=float((scores[seen]>0).mean()) if seen.any() else 0.,
                         score_std=float(scores[seen].std()) if seen.any() else 0.,age_pearson=correlation(scores[seen],ages[seen]),
                         count_pearson=correlation(scores[seen],counts[seen]),error_pearson=correlation(scores[seen],traces['fast'][j,seen])))
    reasons=[]
    if len(rows)!=6: reasons.append('expected six late diagnostic snapshots')
    if not rows or any(x['seen_five_fraction']<.99 or x['positive_fraction']<.05 or x['score_std']<=1e-8 for x in rows):
        reasons.append('insufficient observation coverage or degenerate progress signal')
    for key in ('age_pearson','count_pearson'):
        values=[abs(x[key]) for x in rows if x[key] is not None]
        if values and statistics.mean(values)>=.8: reasons.append('strong '+key+' measurement confounding')
    return rows,reasons

def audit(repo,batch,output):
    import numpy as np
    require(batch.parent==repo/'outputs' and batch.name=='p_development12_by02jzbr','Use the agreed P12 batch')
    state=read(batch/'batch_status.json'); plan=read(batch/'plan.json')
    require((state['status'],state['completed'],state['total'])==('complete',12,12),'Batch is not complete')
    require(plan['dataset']=='cora' and plan['jobs']==[list(x) for x in JOBS] and plan['full_runs']==12
            and plan['source_sha256']==EXPECTED_SOURCE and plan['test_metrics_used_for_selection'] is False,'Batch plan mismatch')
    blog=text(batch/'batch.log').splitlines()
    require('ALL_DONE 12/12' in blog and blog[-1]=='BATCH_EXIT_CODE=0' and sum(x.startswith('DONE ') for x in blog)==12,'Batch completion log mismatch')
    queue=batch/'queue_snapshot'; identity=read(queue/'queue_identity.json')
    require(identity==EXPECTED_QUEUE and plan['queue_identity']==EXPECTED_QUEUE,'Frozen queue identity mismatch')
    for name,expected in EXPECTED_QUEUE['queue_files_sha256'].items(): require(sha(queue/name)==expected,'Frozen queue file mismatch: '+name)
    require(sha(queue/'manifest.json')==EXPECTED_MANIFEST_SHA,'Frozen manifest mismatch')
    anchor=read(batch/'input_protocol_anchor.json')
    base=repo/'outputs/v2_reference18_5AaMEP/v2_reference18_5AaMEP_cora_B0_s0'
    require(anchor['dataset_info']==read(base/'metrics.json')['dataset_info'] and anchor['arguments']==normal(read(base/'effective_args.json')),'Historical input/argument anchor mismatch')
    environment=read(base/'environment.json'); require(anchor['environment']=={k:environment[k] for k in ENV_KEYS},'Historical environment anchor mismatch')
    require((anchor['arguments']['max_epoch'],anchor['arguments']['max_epoch_f'],anchor['dataset_info']['nodes'])==(2000,300,2708),'Cora budget mismatch')
    reference=Path(anchor['s_directory']).resolve()
    require(reference.parent.parent==repo/'outputs' and reference.name=='v2_candidates18_p3TNwS_cora_S_s0','Unexpected S reference path')
    require(sha(reference/'mask_node_statistics.npz')==read(reference/'metrics.json')['mask_node_statistics_sha256'],'Historical S array hash mismatch')
    reference_nodes=arrayfile(reference/'mask_node_statistics.npz')
    stage=Path(plan['stage1_directory']).resolve()
    require(stage.parent==repo/'outputs' and stage.name.startswith('p_dev1_stage1_'),'Stage path mismatch')
    require(read(stage/'status.json')['status']=='complete','Stage incomplete')
    tests=text(stage/'tests.log').splitlines()
    require('P_DEV1_TEST_COUNT=114' in tests and 'P_DEV1_SKIPPED_COUNT=0' in tests and 'OK' in tests,'114-test evidence missing')
    diagnostic_traces={}
    for method in METHODS:
        directory=stage/(stage.name+'_cora_'+method+'_s0')
        _,traces=audit_run(directory,method,0,'diagnostic',anchor,reference_nodes,stage/('diagnostic_'+method+'.log'),stage/('reload_'+method+'.log'))
        diagnostic_traces[method]=traces
    gate=read(stage/'measurement_gate.json'); gate_rows,reasons=compute_gate(diagnostic_traces['P'])
    require(gate['rows']==gate_rows and gate['passed']==(not reasons) and gate['reasons']==reasons and gate['passed']
            and gate['test_scores_used'] is False and gate['validation_scores_used'] is False,'Independent measurement screening mismatch')
    require(gate['thresholds']==dict(min_seen_five_fraction=.99,min_positive_fraction=.05,min_std=1e-8,max_mean_abs_correlation=.8),'Measurement thresholds changed')
    rows=[]; traces_by_seed={}
    for method,seed in JOBS:
        name=batch.name+'_cora_'+method+'_s'+str(seed); directory=batch/name
        row,traces=audit_run(directory,method,seed,'development',anchor,reference_nodes,batch/(name+'_launch.log'),directory/'reload.log')
        rows.append(row); traces_by_seed[method,seed]=traces
    for data in [diagnostic_traces]+[{m:traces_by_seed[m,s] for m in METHODS} for s in range(3)]:
        for method in ('H','PS'):
            for group in np.array_split(np.argsort(reference_nodes['local_feature_density'],kind='stable'),4):
                require(np.array_equal(np.sort(data['P']['weights'][:,group],axis=1),np.sort(data[method]['weights'][:,group],axis=1)),
                        'H/P/PS matched weight multiset condition failed')
                COUNTERS['cross_arm_weight_multisets']+=len(data['P']['steps'])
    lookup={(x['method'],x['seed']):x['best_val_acc'] for x in rows}; groups=[]; comparisons=[]
    for method in METHODS:
        vector=[100*lookup[method,s] for s in range(3)]
        groups.append(dict(method=method,per_seed_val_percent=vector,mean_val_percent=statistics.mean(vector),population_std_percent=statistics.pstdev(vector)))
    for a,b in (('P','S'),('P','H'),('P','PS'),('H','S'),('PS','S')):
        diff=[float(100*(Decimal(str(lookup[a,s]))-Decimal(str(lookup[b,s])))) for s in range(3)]
        comparisons.append(dict(method=a,reference=b,differences_pp=diff,mean_pp=statistics.mean(diff),wins=sum(x>0 for x in diff),
                                ties=sum(x==0 for x in diff),losses=sum(x<0 for x in diff),
                                leave_one_seed_out_mean_pp=[statistics.mean(diff[:s]+diff[s+1:]) for s in range(3)]))
    stored=read(batch/'validation_summary.json')
    require(stored['test_metrics_opened'] is False and stored['primary_comparison']=='P_minus_S','Summary selection protocol mismatch')
    require(len(stored['groups'])==4 and len(stored['runs'])==12 and len(stored['comparisons'])==5,'Summary record counts mismatch')
    for group in stored['groups']:
        actual=next(x for x in groups if x['method']==group['method'])
        for key in ('per_seed_val_percent','mean_val_percent','population_std_percent'): require(group[key]==actual[key],'Saved group statistic mismatch')
    for old in stored['comparisons']:
        actual=next(x for x in comparisons if x['method']==old['method'] and x['reference']==old['reference'])
        for key in ('differences_pp','leave_one_seed_out_mean_pp'):
            require(all(math.isclose(a,b,abs_tol=1e-10) for a,b in zip(old[key],actual[key])),'Saved paired differences mismatch')
        require(math.isclose(old['mean_pp'],actual['mean_pp'],abs_tol=1e-10),'Saved paired mean mismatch')
    for row in read(batch/'validated_runs.json'): require(row['best_val_acc']==lookup[row['method'],row['seed']],'Saved run value mismatch')
    summary=dict(status='PASS',batch_directory=str(batch),stage_directory=str(stage),groups=groups,comparisons=comparisons,
                 measurement_gate=dict(status='PASS',rows=gate_rows),counters=dict(COUNTERS),runs=rows,
                 test_metrics_opened=False,model_weights_opened=False,training_started=False,
                 decision='Stop P expansion under the predeclared no-extra-benefit rule; n=3 on Cora does not establish general failure of learning progress.',
                 checkpoint_scope='Recorded server readback/hash/CPU reload evidence only; no weight files opened or local model reload.')
    write(output/'audit.json',summary)
    write(output/'evidence_sha256.json',EVIDENCE)
    report=['# P12服务器独立审计\n\n状态：PASS。只读复算，不读取模型权重或测试成绩JSON，不启动训练。\n\n',
            '| 方法 | seed0/1/2 验证% | 均值% | 总体标准差% |\n| --- | --- | ---: | ---: |\n']
    for row in groups: report.append('| '+row['method']+' | '+str(row['per_seed_val_percent'])+f" | {row['mean_val_percent']:.3f} | {row['population_std_percent']:.3f} |\n")
    report.append('\n核对配置、输入指纹、环境、源码快照、逐轮预算、反馈与前置选样轨迹、排名权重及PS私有随机重放；匹配权重数值多重集条件通过。测量筛查重新计算一致。工程测试114项、0跳过有记录。权重只核对已记录的保存读回、哈希和CPU重载证据，未读取权重本体。\n\n本报告不作显著性、等效或跨图推广声明。\n')
    (output/'P12独立审计.md').write_text(''.join(report),encoding='utf-8')
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('/home/jovyan/work/GraphMAE2'))
    parser.add_argument('--batch-dir',type=Path,default=Path('/home/jovyan/work/GraphMAE2/outputs/p_development12_by02jzbr'))
    cli=parser.parse_args(); repo=cli.repo.resolve(); batch=cli.batch_dir.resolve()
    require(repo==Path('/home/jovyan/work/GraphMAE2').resolve(),'Use the agreed server project')
    import fcntl
    with (repo/'outputs/.formal_record_v1.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        output=Path(tempfile.mkdtemp(prefix='p12_independent_audit_',dir=repo/'outputs'))
        print('P12_AUDIT_DIR='+str(output),flush=True)
        try:
            result=audit(repo,batch,output)
            write(Path(__file__).parent/'audit_receipt.json',dict(status='PASS',batch_directory=str(batch),audit_directory=str(output),
                        audit_sha256=sha(output/'audit.json'),auditor_sha256=sha(__file__),source_sha256=EXPECTED_SOURCE))
            print('P12_INDEPENDENT_AUDIT=PASS',flush=True)
            for group in result['groups']:
                print('SEEDS '+group['method']+'='+json.dumps(group['per_seed_val_percent']),flush=True)
            for comparison in result['comparisons']:
                print('PAIRED '+comparison['method']+'-'+comparison['reference']+' '+json.dumps(comparison),flush=True)
            print('MEASUREMENT '+json.dumps(result['measurement_gate']),flush=True)
            print('AUDIT_COUNTS '+json.dumps(result['counters']),flush=True)
        except BaseException as error:
            write(output/'audit_failure.json',dict(status='FAIL',error_type=type(error).__name__,error=str(error)))
            print('P12_INDEPENDENT_AUDIT=FAIL; '+str(error),flush=True); return 1
    return 0

if __name__=='__main__': raise SystemExit(main())
