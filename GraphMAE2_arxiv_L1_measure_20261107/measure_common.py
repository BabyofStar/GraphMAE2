"""Shared, import-safe helpers for a bounded B0 engineering measurement."""
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def versions():
    return dict(prefix=sys.prefix, python=sys.version,
                packages={name: metadata.version(name) for name in
                          ['torch','dgl','numpy','scipy','scikit-learn','ogb']})


def resource_snapshot(torch_module=None):
    result = {'time_beijing': now()}
    for name in ['memory.current','memory.peak','memory.max','memory.stat',
                 'memory.events','memory.pressure','cpu.max']:
        try:
            result[name] = Path('/sys/fs/cgroup', name).read_text('utf-8')
        except OSError:
            pass
    try:
        import resource
        result['process_lifetime_max_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        pass
    try:
        result['process_status'] = '\n'.join(line for line in Path('/proc/self/status').read_text().splitlines()
                                              if line.startswith(('VmRSS:','VmHWM:','Threads:')))
    except OSError:
        pass
    if torch_module is not None:
        try:
            torch = torch_module
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                result['cuda'] = dict(name=torch.cuda.get_device_name(0),
                                      visible_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                                      reported_free_bytes=free, reported_total_bytes=total,
                                      allocated_bytes=torch.cuda.memory_allocated(),
                                      reserved_bytes=torch.cuda.memory_reserved(),
                                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        except Exception as error:
            result['cuda_snapshot_error'] = type(error).__name__
    return result


def nvidia_snapshot():
    commands = [
        ['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv'],
        ['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']]
    results = []
    for command in commands:
        try:
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, errors='replace', timeout=8)
            results.append(dict(command=command, exit_code=process.returncode,
                                stdout=process.stdout[:12000], stderr=process.stderr[:2000]))
        except (OSError, subprocess.TimeoutExpired) as error:
            results.append(dict(command=command, error=type(error).__name__))
    return results


def is_cuda_oom(error):
    text = str(error).lower()
    return any(marker in text for marker in ['cuda out of memory','cuda_error_out_of_memory',
                                             'cudaerrormemoryallocation','cuda error: out of memory'])


def choose_capacity(attempts):
    """Only explicit device OOM allows trying a smaller batch."""
    for attempt in attempts:
        status = attempt['status']
        if status == 'PASS_CAPACITY_CHECK':
            return int(attempt['batch_size'])
        require(status == 'CUDA_OOM', 'Capacity check failed for a reason other than CUDA OOM')
    return None


def pilot_groups(clusters, batch_size, np):
    result = []
    for name, values in zip(['train','valid','test'], clusters):
        require(len(values) >= batch_size, 'Capacity pilot split is too small')
        indexes = np.linspace(0, len(values)-1, batch_size, dtype=np.int64)
        result.append((name, [values[int(i)] for i in indexes]))
    return result


def check_clusters(clusters, split_ids, pilot_path, np):
    require(isinstance(clusters, (tuple,list)) and len(clusters) == 3, 'Cluster split container differs')
    roots = []
    lengths = []
    flattened = []
    for name, values in zip(['train','valid','test'], clusters):
        ids = split_ids[name]
        require(len(values) == len(ids), 'Cluster split length differs: '+name)
        for root, value in zip(ids, values):
            require(isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == np.int32,
                    'Unexpected cluster array type')
            require(1 <= len(value) <= 256 and int(value[0]) == int(root), 'Cluster root or size differs')
            require(int(value.min()) >= 0 and int(value.max()) < sum(len(v) for v in split_ids.values()), 'Cluster node range differs')
            require(len(np.unique(value)) == len(value), 'Cluster contains repeated node positions')
            roots.append(int(root)); lengths.append(len(value)); flattened.append(value)
    require(len(set(roots)) == len(roots), 'Root coverage overlaps')
    lookup = {root: index for index, root in enumerate(roots)}
    with np.load(pilot_path, allow_pickle=False) as pilot:
        pr, offsets, nodes = (pilot[k] for k in ['root_ids','offsets','node_ids'])
        require(len(pr) == 256 and len(offsets) == 257 and offsets[0] == 0 and offsets[-1] == len(nodes), 'Pilot shape differs')
        for index, root in enumerate(pr):
            require(int(root) in lookup and np.array_equal(flattened[lookup[int(root)]], nodes[offsets[index]:offsets[index+1]]),
                    'Full file differs from returned pilot node list')
    root_array = np.asarray(roots, dtype=np.int64)
    return flattened, root_array, dict(roots=len(roots), split_counts={k:len(v) for k,v in split_ids.items()},
                                     node_count_min_mean_max=[min(lengths),float(np.mean(lengths)),max(lengths)],
                                     root_order_sha256=hashlib.sha256(root_array.tobytes()).hexdigest(),
                                     exact_returned_pilot_checks=256)


def effective_args(contract, snapshot):
    from utils import build_args, load_best_configs
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        sys.argv = ['measure_frozen_large', *contract['upstream_cli']]
        os.chdir(snapshot)
        args = build_args()
        if args.use_cfg:
            args = load_best_configs(args)
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
    for key, expected in contract['expected_effective_args'].items():
        require(getattr(args,key) == expected, 'Effective config differs for '+key)
    args.num_features = 128
    return args


def load_context(contract, snapshot, output):
    """No source installation, downloads, labels used for capacity selection, or LC generation."""
    import numpy as np
    import torch
    import dgl
    from datasets.lc_sampler import load_dataset
    from scipy.sparse import csr_matrix
    from utils import set_random_seed
    require(sys.prefix == contract['base_prefix'], 'Wrong training interpreter')
    require(versions()['packages'] == contract['base_versions'], 'Frozen training packages differ')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expected one visible CUDA device')
    require(torch.cuda.get_device_properties(0).total_memory == contract['visible_cuda_total_bytes'], 'Visible GPU memory changed; remeasure resources first')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    set_random_seed(0)
    args = effective_args(contract, snapshot)
    began = time.monotonic()
    asset = Path(contract['asset_root'])
    feats, graph, labels, split = load_dataset(str(asset), 'ogbn-arxiv')
    require(graph.num_nodes() == 169343 and graph.num_edges() == contract['graph']['edges_with_self_loops'], 'Processed training graph differs')
    require(list(feats.shape) == [169343,128] and torch.isfinite(feats).all().item(), 'Features have an unexpected shape or nonfinite values')
    ids = {k:split[k].cpu().numpy().astype(np.int64,copy=False) for k in ['train','valid','test']}
    require({k:len(v) for k,v in ids.items()} == contract['split_counts'], 'Official split counts differ')
    require(len(np.unique(np.concatenate(list(ids.values())))) == 169343, 'Official split coverage differs')
    require(all(hashlib.sha256(v.tobytes()).hexdigest() == contract['graph']['split_sha256'][k] for k,v in ids.items()), 'Official split identity differs')
    row, col = graph.edges()
    adj = csr_matrix((np.ones(len(row)), (row.numpy(),col.numpy())), shape=(169343,169343))
    require(hashlib.sha256(adj.indptr.tobytes()).hexdigest() == contract['graph']['csr_indptr_sha256'] and
            hashlib.sha256(adj.indices.tobytes()).hexdigest() == contract['graph']['csr_indices_sha256'], 'Processed CSR differs from clustering graph')
    del adj, row, col
    path = Path(contract['cluster_file']['path'])
    # Driver checks the complete SHA256 before any subprocess. This is a trusted,
    # locally generated numpy-container file, never an arbitrary downloaded pickle.
    require(path.stat().st_size == contract['cluster_file']['bytes'], 'Cluster file size changed')
    clusters = torch.load(path, map_location='cpu', weights_only=False)
    flat, roots, cluster_checks = check_clusters(clusters, ids, Path(__file__).parent/'pilot_clusters.npz', np)
    require(int(labels[split['train']].max().item())+1 == 40, 'Training label classes differ')
    info = dict(load_seconds=time.monotonic()-began, graph_nodes=graph.num_nodes(),graph_edges=graph.num_edges(),
                scaled_feature_sha256=hashlib.sha256(feats.numpy().tobytes()).hexdigest(),
                cluster_checks=cluster_checks, effective_args=vars(args), environment=versions())
    return dict(np=np,torch=torch,dgl=dgl,args=args,graph=graph,feats=feats,labels=labels,
                split=split,split_ids=ids,clusters=clusters,flat_clusters=flat,roots=roots,
                root_lookup={int(root):index for index,root in enumerate(roots)},info=info)


def budget_estimate(epoch_seconds, embedding_seconds, probe_epoch_seconds, checkpoint_seconds, test_forward_seconds):
    values = [epoch_seconds,embedding_seconds,probe_epoch_seconds,checkpoint_seconds,test_forward_seconds]
    require(all(math.isfinite(v) and v>=0 for v in values), 'Invalid timing estimate input')
    pretrain = epoch_seconds*60
    probe = probe_epoch_seconds*1000*10 + test_forward_seconds*10
    total = pretrain+embedding_seconds+probe+checkpoint_seconds*60
    return dict(pretrain_60_epoch_seconds=pretrain, full_embedding_seconds=embedding_seconds,
                ten_probes_1000_epochs_seconds=probe, model_save_60_times_seconds=checkpoint_seconds*60,
                B0_one_encoder_estimated_seconds=total,
                B0_B2_six_encoders_baseline_cost_estimated_seconds=total*6,
                uncertainty='Linear extrapolation only; includes measured worker/batch adaptation. One early epoch omits later EMA cost, run-to-run shared-GPU changes and B2 adaptation overhead. Six-run estimate is a baseline cost reference, not a launch authorization or validation of B2.')
