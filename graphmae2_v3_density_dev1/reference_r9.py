"""Nine full-budget R runs, gated by completed N2 and matched historical anchors."""
import argparse
import contextlib
import datetime
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent
SEEDS = {'cora': 2026110301, 'citeseer': 2026110302, 'pubmed': 2026110303}
ENV_KEYS = ('versions', 'torch_cuda_runtime', 'torch_cpu_threads',
            'deterministic_algorithms', 'cudnn_deterministic', 'cudnn_benchmark',
            'cuda_matmul_allow_tf32', 'cudnn_allow_tf32', 'gpu_name',
            'visible_gpu_memory_bytes', 'runtime_environment')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def load_module(path, name, expected):
    require(sha(path) == expected, 'Helper identity differs: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def check_package():
    identity = read(PACKAGE/'queue_identity.json')
    for name, expected in identity['queue_files_sha256'].items():
        require(sha(PACKAGE/name) == expected, 'Queue package differs: '+name)
    return identity


def check_sources(repo, identity):
    package = repo/'graphmae2_v2_r_dev1'
    for name, expected in identity['accepted_r_package_files_sha256'].items():
        require(sha(package/name) == expected, 'Accepted R package differs: '+name)
    manifest = read(package/'manifest.json')
    for name, expected in manifest['after_sha256'].items():
        path = repo/name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(repo)
                and sha(path) == expected, 'Accepted R source differs: '+name)
    return manifest


def verify_snapshot(run_dir, expected):
    require(read(run_dir/'source_sha256.json') == expected, 'Run source manifest differs: '+str(run_dir))
    found = set()
    with tarfile.open(run_dir/'source.tar', 'r:') as archive:
        for item in archive:
            path = PurePosixPath(item.name)
            require(item.isfile() and not path.is_absolute() and '..' not in path.parts
                    and '\\' not in item.name and item.name in expected
                    and item.name not in found and item.size <= 10*1024*1024,
                    'Unexpected snapshot entry: '+item.name)
            stream = archive.extractfile(item)
            with stream:
                require(hashlib.sha256(stream.read()).hexdigest() == expected[item.name],
                        'Snapshot bytes differ: '+item.name)
            found.add(item.name)
    require(found == set(expected), 'Incomplete snapshot: '+str(run_dir))


def check_stage_status(directory):
    status = read(directory/'status.json')
    require(status.get('status') == 'complete' and status.get('stage') == 'implementation_acceptance_only',
            'N2 is not complete; do not restart it or launch full training')
    log = (directory/'tests.log').read_text(encoding='utf-8', errors='replace')
    require('R_DEV1_TEST_COUNT=80' in log.splitlines() and 'R_DEV1_SKIPPED_COUNT=0' in log.splitlines(),
            'N2 80-test/no-skip evidence missing')


def check_stage(repo, directory, manifest, identity):
    require(directory.parent == repo/'outputs', 'N2 directory must be in project outputs')
    check_stage_status(directory)
    run_dir = directory/(directory.name+'_cora_R_s0')
    metrics, args = read(run_dir/'metrics.json'), read(run_dir/'effective_args.json')
    require(metrics['status'] == 'complete' and metrics['phase'] == 'smoke'
            and metrics['method'] == 'R' and args['max_epoch'] == 5 and args['max_epoch_f'] == 5,
            'N2 R smoke identity/budget differs')
    require('MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS' in
            (directory/'reload_R.log').read_text(encoding='utf-8'), 'N2 CPU reload did not pass')
    require(sha(run_dir/metrics['mask_node_statistics_file']) == metrics['mask_node_statistics_sha256'],
            'N2 node arrays differ')
    require(sha(run_dir/metrics['random_group_plan_file']) == metrics['random_group_plan_sha256'],
            'N2 random group plan differs')
    verify_snapshot(run_dir, manifest['after_sha256'])
    helper = load_module(repo/'graphmae2_v2_r_dev1/run_stage1_r.py', '_accepted_r_stage',
                         identity['accepted_r_package_files_sha256']['run_stage1_r.py'])
    helper.validate_r_smoke(run_dir)


def env_signature(run_dir):
    environment = read(run_dir/'environment.json')
    return {key: environment[key] for key in ENV_KEYS}


def normalized_args(args):
    # This new field affects R only; all remaining original arguments must match.
    return {key:value for key,value in args.items()
            if key not in ('seeds', 'input_mask_mode', 'random_group_seed')}


def load_anchors(repo, manifest, identity):
    library = load_module(PACKAGE/'legacy_reference.py', '_approved_reference',
                          identity['queue_files_sha256']['legacy_reference.py'])
    candidates = load_module(PACKAGE/'legacy_candidate.py', '_approved_candidate',
                             identity['queue_files_sha256']['legacy_candidate.py'])
    ref = repo/'outputs/v2_reference18_5AaMEP'
    anchors, _ = candidates.load_references(library, repo, ref, {'after_sha256':manifest['before_sha256']})
    candidate = repo/'outputs/v2_candidates18_p3TNwS'
    state = read(candidate/'batch_status.json')
    require((state.get('status'),state.get('completed'),state.get('total')) == ('complete',18,18),
            'Historical candidate batch is incomplete')
    for (dataset, seed), pair in anchors.items():
        for method in ('B0','B2'):
            directory = Path(pair[method]['run_directory'])
            verify_snapshot(directory, manifest['before_sha256'])
            pair[method]['environment_signature'] = env_signature(directory)
        directory = candidate/('%s_%s_S_s%d'%(candidate.name,dataset,seed))
        # The approved S validator uses the original environment keys; the extended
        # comparison below also checks recorded runtime environment variables.
        old_reference = dict(pair['B0'], environment_signature={key:pair['B0']['environment_signature'][key]
                             for key in candidates.ENVIRONMENT_KEYS})
        pair['S'] = candidates.validate_candidate(library,directory,dataset,'S',seed,old_reference)
        pair['S']['environment_signature'] = env_signature(directory)
        verify_snapshot(directory, manifest['before_sha256'])
        require('MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS' in (directory/'reload.log').read_text(encoding='utf-8'),
                'Historical S reload evidence missing')
        require(pair['B0']['environment_signature'] == pair['B2']['environment_signature'] == pair['S']['environment_signature'],
                'Historical anchor environments differ')
    return library, anchors


def quotas(sizes, k):
    n = sum(sizes)
    raw = [Fraction(k*size,n) for size in sizes]
    result = [math.floor(value) for value in raw]
    for i in sorted(range(len(sizes)),key=lambda i:(-(raw[i]-result[i]),i))[:k-sum(result)]:
        result[i] += 1
    return result


def check_r(run_dir, dataset, seed, anchor, manifest):
    status, metrics, args = [read(run_dir/name) for name in ('status.json','metrics.json','effective_args.json')]
    for record in (status,metrics):
        for key,value in dict(status='complete',phase='development',dataset=dataset,method='R',seed=seed,
                              official_epoch_budget=True,recorder_version='2.0.0-r-dev1').items():
            require(record.get(key) == value, 'R metadata differs: '+key)
    require(args['seeds'] == [seed] and args['input_mask_mode'] == 'random_group_stratified'
            and args['random_group_seed'] == SEEDS[dataset], 'R seed/mode differs')
    require(normalized_args(args) == anchor['nonmethod_args'], 'R nonmethod arguments differ from frozen anchors')
    require(metrics['dataset_info'] == anchor['dataset_info'], 'R graph/features/splits differ')
    require(env_signature(run_dir) == anchor['environment_signature'], 'R runtime settings differ from anchors')
    verify_snapshot(run_dir,manifest['after_sha256'])
    n, epochs = anchor['dataset_info']['nodes'], args['max_epoch']
    k = int(n*args['mask_rate'])
    sizes = [n//4+(i<n%4) for i in range(4)]
    expected_quotas = quotas(sizes,k)
    plan_path=run_dir/metrics['random_group_plan_file']
    require(sha(plan_path) == metrics['random_group_plan_sha256'], 'R plan hash differs')
    plan=read(plan_path)
    require(plan['random_group_seed'] == SEEDS[dataset] and plan['nodes'] == n
            and plan['masked'] == k and plan['group_sizes'] == sizes and plan['quotas'] == expected_quotas,
            'R fixed grouping plan differs')
    feedback=[json.loads(line) for line in (run_dir/'mask_feedback.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    steps=sorted(set(range(0,epochs,100))|{epochs-1})
    require([row['step'] for row in feedback] == steps and metrics['feedback_records'] == len(steps),
            'R full-budget feedback steps differ')
    for row in feedback:
        require(row['mode'] == 'random_group_stratified' and row['nodes'] == n and row['masked'] == k
                and row['alpha'] == 0 and row['random_group_seed'] == SEEDS[dataset]
                and row['sampling_group_order_sha256'] == plan['ordered_group_nodes_sha256'], 'R sampler identity differs')
        require([x['group'] for x in row['sampling_groups']] == list(range(4))
                and [x['nodes'] for x in row['sampling_groups']] == sizes
                and [x['masked'] for x in row['sampling_groups']] == expected_quotas,
                'R actual group quota differs')
        require([x['layer'] for x in row['density_layers']] == list(range(4))
                and [x['nodes'] for x in row['density_layers']] == sizes
                and sum(x['masked'] for x in row['density_layers']) == k,
                'R true-density diagnostics differ')
        require(math.isclose(row['observation_count_min_mean_max'][1],k*(row['step']+1)/n,
                             rel_tol=1e-9,abs_tol=1e-8), 'R cumulative observation budget differs')
    node_path=run_dir/metrics['mask_node_statistics_file']
    require(sha(node_path) == metrics['mask_node_statistics_sha256'], 'R NPZ hash differs')
    import numpy as np
    with np.load(node_path,allow_pickle=False) as nodes, np.load(Path(anchor['s_directory'])/'mask_node_statistics.npz',allow_pickle=False) as s:
        require(np.array_equal(nodes['node_id'],np.arange(n)), 'R node ordering differs')
        for name in ('density_layer','local_feature_density','nonself_in_degree'):
            require(np.array_equal(nodes[name],s[name]), 'True density/graph statistics differ from S: '+name)
        order=nodes['sampling_group_node_order']
        require(np.array_equal(np.sort(order),np.arange(n)), 'Random group order is not a permutation')
        require(hashlib.sha256(order.astype('<i8',copy=False).tobytes()).hexdigest() == plan['ordered_group_nodes_sha256'],
                'Saved random group order fingerprint differs')
        membership=np.empty(n,dtype=np.int64)
        for i,group in enumerate(np.split(order,np.cumsum(sizes)[:-1])):
            membership[group]=i
        require(np.array_equal(nodes['sampling_group'],membership) and int(nodes['random_group_seed']) == SEEDS[dataset],
                'Saved random group membership differs')
        counts, errors = nodes['observation_count'], nodes['reconstruction_error_ema']
        require(counts.shape == (n,) and np.issubdtype(counts.dtype,np.integer)
                and (counts>=0).all() and (counts<=epochs).all() and int(counts.sum()) == k*epochs,
                'R full node observation budget differs')
        require(errors.shape == (n,) and np.isfinite(errors).all() and (errors>=0).all(), 'Invalid R error diagnostics')
    value=metrics['best_val_acc']
    require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and 0<=value<=1,
            'Invalid R validation accuracy')
    epoch=metrics['best_val_epoch']
    require(isinstance(epoch,int) and 0<=epoch<args['max_epoch_f'], 'Invalid R probe selection epoch')
    resources=read(run_dir/'resources.json')
    seconds=resources['pretrain_including_intermediate_probes']['wall_seconds']
    require(isinstance(seconds,(float,int)) and math.isfinite(seconds) and seconds>=0, 'Invalid R timing')
    return dict(dataset=dataset,method='R',seed=seed,best_val_acc=value,best_val_epoch=epoch,
                pretrain_wall_seconds=seconds,run_directory=str(run_dir),random_group_plan=plan,
                checkpoint_sha256=metrics['checkpoint_sha256'],resources=resources)


def summary(rows, anchors):
    groups=[]
    for dataset in SEEDS:
        ordered=sorted((row for row in rows if row['dataset']==dataset),key=lambda row:row['seed'])
        require([row['seed'] for row in ordered] == [0,1,2], 'Incomplete R group')
        values=[100*row['best_val_acc'] for row in ordered]
        mean=sum(values)/3
        item=dict(dataset=dataset,n=3,R_val_percent_mean=mean,
                  R_val_percent_population_std=math.sqrt(sum((x-mean)**2 for x in values)/3))
        for method in ('S','B0','B2'):
            differences=[100*(row['best_val_acc']-anchors[dataset,row['seed']][method]['best_val_acc']) for row in ordered]
            item['R_minus_'+method+'_paired_pp']=differences
            item['R_minus_'+method+'_mean_pp']=sum(differences)/3
            item['R_vs_'+method+'_win_tie_loss']=[sum(x>1e-10 for x in differences),sum(abs(x)<=1e-10 for x in differences),sum(x< -1e-10 for x in differences)]
        item['S_minus_R_paired_pp']=[-x for x in item['R_minus_S_paired_pp']]
        item['S_minus_R_mean_pp']=-item['R_minus_S_mean_pp']
        for method in ('B0','B2'):
            differences=[100*(anchors[dataset,seed]['S']['best_val_acc']-anchors[dataset,seed][method]['best_val_acc']) for seed in (0,1,2)]
            item['S_minus_'+method+'_paired_pp']=differences
            item['S_minus_'+method+'_mean_pp']=sum(differences)/3
        groups.append(item)
    return dict(groups=groups,runs=rows,primary_comparison='S_minus_R',
                equal_dataset_mean_S_minus_R_pp=sum(g['S_minus_R_mean_pp'] for g in groups)/3,
                test_metrics_opened=False,interpretation='Exploratory development validation, n=3; one fixed random grouping per dataset; no significance or equivalence claim')


@contextlib.contextmanager
def locked(repo):
    import fcntl
    path=repo/'outputs/.formal_record_v1.lock'
    inherited=os.environ.get('GRAPHMAE2_R_LOCK_FD')
    if inherited is not None:
        fd=int(inherited)
        a,b=os.fstat(fd),path.stat()
        require((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino),'Inherited lock descriptor differs')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield fd
    else:
        with path.open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield lock.fileno()


def preflight(repo, stage):
    require(Path(sys.prefix).resolve()==Path('/home/jovyan/work/envs/graphmae2'), 'Activate graphmae2 environment')
    identity=check_package()
    manifest=check_sources(repo,identity)
    check_stage(repo,stage,manifest,identity)
    library,anchors=load_anchors(repo,manifest,identity)
    smoke=stage/(stage.name+'_cora_R_s0')
    require(env_signature(smoke)==anchors['cora',0]['B0']['environment_signature'],
            'Accepted N2 runtime differs from historical anchors; stop before full training')
    return identity,manifest,library,anchors


def execute(command,repo,log,fd):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',DGLBACKEND='pytorch',
             WANDB_MODE='disabled',MPLBACKEND='Agg',GRAPHMAE2_R_LOCK_FD=str(fd))
    with log.open('x',encoding='utf-8') as stream:
        process=subprocess.run(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,
                               stdout=stream,stderr=subprocess.STDOUT,pass_fds=(fd,))
    if process.returncode:
        print('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-60:]),flush=True)
        raise RuntimeError('Child failed; log='+str(log))


def run(cli):
    repo,batch,stage=cli.repo.resolve(),cli.batch_dir.resolve(),cli.stage1_dir.resolve()
    require(batch.parent==repo/'outputs' and batch.name.startswith('r_development9_'), 'Invalid unique R output directory')
    require(batch.is_dir() and {p.name for p in batch.iterdir()} <= {'batch.log','batch.pid','launch.json'}, 'Never overwrite a previous batch')
    require(not cli.checkpoint_root.resolve().is_relative_to(repo), 'Weights must be outside the project')
    with locked(repo) as fd:
        state=dict(status='running',completed=0,total=9,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(batch/'batch_status.json',state)
        rows=[]
        started=time.perf_counter()
        try:
            identity,manifest,library,anchors=preflight(repo,stage)
            for name in sorted(set(identity['queue_files_sha256'])|{'queue_identity.json','SHA256SUMS.txt'}):
                dest=batch/'queue_snapshot'/name
                dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(PACKAGE/name,dest)
            jobs=[(dataset,seed) for dataset in SEEDS for seed in (0,1,2)]
            write(batch/'plan.json',dict(runs=jobs,method='R',training_seeds=[0,1,2],group_seeds=SEEDS,
                  official_configs=library.CONFIGS,source_sha256=manifest['after_sha256'],stage1_dir=str(stage),
                  queue_identity=identity,test_metrics_used_for_selection=False,
                  primary_comparison='S_minus_R',analysis='paired descriptive validation; ddof=0; all three datasets'))
            write(batch/'historical_anchors.json',[dict(dataset=d,seed=s,methods=p) for (d,s),p in anchors.items()])
            plans={}
            for dataset,seed in jobs:
                check_package()
                check_sources(repo,identity)
                name='%s_%s_R_s%d'%(batch.name,dataset,seed)
                directory=batch/name
                print('START dataset=%s method=R seed=%d'%(dataset,seed),flush=True)
                execute([sys.executable,'-u','record_full_batch_r.py','run','--dataset',dataset,'--method','R',
                         '--seed',str(seed),'--phase','development','--device','0','--output-dir',str(directory),
                         '--checkpoint-root',str(cli.checkpoint_root.resolve())],repo,batch/(name+'_launch.log'),fd)
                anchor=dict(anchors[dataset,seed]['B0'],s_directory=anchors[dataset,seed]['S']['run_directory'])
                row=check_r(directory,dataset,seed,anchor,manifest)
                if dataset in plans:
                    require(row['random_group_plan']==plans[dataset], 'Random grouping changed across training seeds')
                plans[dataset]=row['random_group_plan']
                execute([sys.executable,'-u','record_full_batch_r.py','verify','--output-dir',str(directory)],repo,directory/'reload.log',fd)
                require('MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS' in (directory/'reload.log').read_text(encoding='utf-8'), 'R CPU reload marker missing')
                rows.append(row)
                state['completed']=len(rows)
                write(batch/'validated_runs.json',rows)
                write(batch/'batch_status.json',state)
                print('DONE %d/9 dataset=%s seed=%d BestVal=%.4f'%(len(rows),dataset,seed,row['best_val_acc']),flush=True)
            check_package()
            check_sources(repo,identity)
            result=summary(rows,anchors)
            write(batch/'validation_summary.json',result)
            for group in result['groups']:
                print('VAL_COMPARE %s R=%.3f +/- %.3f %% S-minus-R=%+.3fpp'%(group['dataset'],group['R_val_percent_mean'],group['R_val_percent_population_std'],group['S_minus_R_mean_pp']),flush=True)
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
    cli=parser.parse_args()
    code=0
    try:
        run(cli)
    except BaseException as error:
        print('R9_ERROR='+str(error),flush=True)
        code=1
    print('BATCH_EXIT_CODE='+str(code),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
