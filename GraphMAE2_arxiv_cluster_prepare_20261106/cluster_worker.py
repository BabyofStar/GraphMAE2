"""Measure and generate LC node lists using the frozen upstream algorithm.

Only the unused conductance calculation is omitted after exact node-list checks.
No labels are used to select roots and no model or classifier is trained.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import importlib.util
import inspect
import json
import multiprocessing as mp
from pathlib import Path
try:
    import resource
except ImportError:
    resource=None
import statistics as st
import sys
import time
import traceback
import numpy as np

_REF=None


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def nodes_only(root):
    start=time.monotonic();root=int(root);graph=_REF.graphlocal
    nodes,ppr=_REF.approximate_PageRank(graph,[root],iterations=1000,method='acl',normalize=False)
    inv=graph.dn[nodes];inv[inv>1.0]=1.0
    # Preserve upstream's order: take prefix first, then stable degree-adjusted sorting.
    choices=list(zip(nodes,ppr*inv))[:256]
    values=np.asarray([node for node,_ in sorted(choices,key=lambda pair:pair[1],reverse=True)],dtype=np.int32)
    require(len(values)>0 and values[0]==root and len(values)<=256 and len(np.unique(values))==len(values),'Upstream LC root/size/uniqueness invariant failed')
    return values,time.monotonic()-start


def sample_roots(split_ids,counts):
    roots=[];splits=[]
    for name in ['train','valid','test']:
        ids=split_ids[name];count=min(counts[name],len(ids))
        selected=np.linspace(0,len(ids)-1,count,dtype=np.int64)
        roots.extend(int(ids[i]) for i in selected);splits.extend([name]*len(selected))
    require(len(roots)==len(set(roots)),'Pilot roots overlap')
    return roots,splits


def packed_clusters(roots,clusters,path):
    offsets=np.zeros(len(clusters)+1,dtype=np.int64)
    offsets[1:]=np.cumsum([len(x) for x in clusters]);flat=np.concatenate(clusters).astype(np.int32,copy=False)
    with Path(path).open('xb') as f:np.savez_compressed(f,root_ids=np.asarray(roots,dtype=np.int64),offsets=offsets,node_ids=flat)


def memory():
    result={}
    for name in ['memory.current','memory.peak','memory.max','memory.stat','memory.events','memory.pressure','cpu.max']:
        try:result[name]=Path('/sys/fs/cgroup',name).read_text('utf-8')
        except OSError:pass
    if resource is not None:result['process_lifetime_max_rss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return result


def bounded_results(iterator,deadline):
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Declared full CPU preprocessing time cap reached')
        try:yield iterator.next(timeout=min(30.,remaining))
        except mp.TimeoutError:
            print('LC_WAITING_FOR_OWN_WORKER',flush=True)
            continue
        except StopIteration:return


def main():
    global _REF
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path);parser.add_argument('--contract',required=True,type=Path)
    args=parser.parse_args();project=args.project.resolve();out=args.output_dir.resolve();contract=json.loads(args.contract.read_text('utf-8'))
    report=dict(stage='LC CPU preprocessing',time_beijing=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
                training_started=False,model_constructed=False,library_version=metadata.version('localgraphclustering'),
                actual_environment=sys.prefix,contract=contract,memory_before=memory())
    began=time.monotonic()
    try:
        require(report['library_version']=='0.6.1','Wrong pinned LC version')
        source=project/'datasets/localclustering.py';require(sha(source)==contract['localclustering_source_sha256'],'Clustering reference source differs')
        import torch
        from scipy.sparse import csr_matrix
        from localgraphclustering import GraphLocal
        torch.set_num_threads(1);np.random.seed(0)
        spec=importlib.util.spec_from_file_location('graphmae2_lc_reference',source)
        _REF=importlib.util.module_from_spec(spec);spec.loader.exec_module(_REF)
        parameters=inspect.signature(_REF.approximate_PageRank).parameters
        require(parameters['alpha'].default==.15 and parameters['rho'].default==1e-6,'ACL defaults differ')
        # Compare exact returned node lists against the original function on a tiny graph.
        tiny=np.eye(6)
        for i in range(6):tiny[i,(i+1)%6]=tiny[(i+1)%6,i]=1.
        _REF.graphlocal=GraphLocal.from_sparse_adjacency(csr_matrix(tiny))
        for i in range(6):
            actual,_=nodes_only(i);expected,_=_REF.calc_local_clustering((i,10000,1000,256,'acl'))
            require(np.array_equal(actual,expected),'Tiny-graph node lists differ from upstream')
        report['tiny_graph_exact_node_checks']=6
        loaded=time.monotonic();asset=Path(contract['asset_root']).resolve()
        graph,split=_REF.load_dataset(str(asset),'ogbn-arxiv')
        require(graph.num_nodes()==contract['expected_nodes'],'Arxiv node count differs')
        split_ids={name:split[name].cpu().numpy().astype(np.int64,copy=False) for name in ['train','valid','test']}
        require({k:len(v) for k,v in split_ids.items()}==contract['expected_split_counts'],'Official time split counts differ')
        row,col=graph.edges();adj=csr_matrix((np.ones(len(row)),(row.numpy(),col.numpy())),shape=(graph.num_nodes(),graph.num_nodes()))
        require(np.all(adj.diagonal()==1) and (adj!=adj.T).nnz==0,'Upstream preprocessing did not produce the expected bidirected graph with one self-loop')
        _REF.graphlocal=GraphLocal.from_sparse_adjacency(adj)
        report['graph_loading_and_preparation_seconds']=time.monotonic()-loaded
        report['graph']={'nodes':graph.num_nodes(),'edges_with_self_loops':graph.num_edges(),
                         'feature_shape':list(graph.ndata['feat'].shape),
                         'csr_indptr_sha256':hashlib.sha256(adj.indptr.tobytes()).hexdigest(),
                         'csr_indices_sha256':hashlib.sha256(adj.indices.tobytes()).hexdigest(),
                         'split_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in split_ids.items()}}
        checks=[]
        for name,ids in split_ids.items():
            for position in [0,len(ids)//2,len(ids)-1]:
                i=int(ids[position]);actual,_=nodes_only(i);expected,_=_REF.calc_local_clustering((i,10000,1000,256,'acl'))
                require(np.array_equal(actual,expected),'Arxiv node lists differ from upstream')
                checks.append({'split':name,'root':i,'nodes':len(actual)})
        report['arxiv_exact_node_checks']=checks
        roots,root_splits=sample_roots(split_ids,contract['timing_pilot_counts'])
        serial=[nodes_only(root) for root in roots]
        for values,_ in serial:require(int(values.min())>=0 and int(values.max())<graph.num_nodes(),'Pilot ID range')
        timings={name:[seconds for (_,seconds),split_name in zip(serial,root_splits) if split_name==name] for name in split_ids}
        report['pilot']={'root_count':len(roots),'serial_mean_seconds_by_split':{k:st.mean(v) for k,v in timings.items()},
                         'serial_max_seconds':max(t for _,t in serial),'node_count_min_mean_max':[min(len(v) for v,_ in serial),st.mean(len(v) for v,_ in serial),max(len(v) for v,_ in serial)]}
        parallel_start=time.monotonic()
        with mp.get_context('fork').Pool(processes=2) as pool:parallel=pool.map(nodes_only,roots,chunksize=1)
        parallel_wall=time.monotonic()-parallel_start
        require(all(np.array_equal(a[0],b[0]) for a,b in zip(serial,parallel)),'Parallel preprocessing changed node lists')
        packed_clusters(roots,[v for v,_ in serial],out/'pilot_clusters.npz')
        estimated=sum(len(split_ids[k])*st.mean(timings[k]) for k in split_ids)/2
        observed_ratio=parallel_wall/max(sum(t for _,t in serial)/2,1e-8)
        conservative_estimate=estimated*max(1.,observed_ratio)
        report['pilot'].update(two_worker_wall_seconds=parallel_wall,parallel_exact_node_checks=len(roots),
                               estimated_full_two_worker_seconds=estimated,observed_overhead_adjusted_estimate_seconds=conservative_estimate,
                               estimate_scope='Extrapolated from fixed input-only pilot; no guarantee of complete runtime')
        print('LC_PILOT_ESTIMATED_FULL_SECONDS='+f'{conservative_estimate:.1f}',flush=True)
        if conservative_estimate>contract['max_full_preprocessing_seconds']:
            report['status']='PILOT_COMPLETE_FULL_GENERATION_HELD_FOR_COST'
        else:
            target=asset/'lc_ego_graphs/ogbn-arxiv-lc-ego-graphs-256.pt';target.parent.mkdir(exist_ok=True)
            require(not target.exists(),'Cluster target already exists; preserved without overwrite')
            all_roots=np.concatenate([split_ids[k] for k in ['train','valid','test']]);clusters=[];full_start=time.monotonic()
            try:
                with mp.get_context('fork').Pool(processes=2) as pool:
                    iterator=pool.imap(nodes_only,(int(i) for i in all_roots),chunksize=1)
                    for i,(values,_) in enumerate(bounded_results(iterator,full_start+contract['max_full_preprocessing_seconds']),1):
                        require(values[0]==all_roots[i-1] and 0<=values.min() and values.max()<graph.num_nodes(),'Full cluster identity/range')
                        clusters.append(values)
                        if i%5000==0:
                            elapsed=time.monotonic()-full_start;save(out/'progress.json',{'completed_roots':i,'total_roots':len(all_roots),'wall_seconds':elapsed})
                            print(f'LC_GENERATED {i}/{len(all_roots)} elapsed={elapsed:.1f}',flush=True)
            except TimeoutError:
                if clusters:packed_clusters(all_roots[:len(clusters)],clusters,out/'partial_clusters.npz')
                raise
            require(len(clusters)==graph.num_nodes(),'Full root coverage')
            a=len(split_ids['train']);b=a+len(split_ids['valid']);split_clusters=[tuple(clusters[:a]),tuple(clusters[a:b]),tuple(clusters[b:])]
            with target.open('xb') as f:torch.save(split_clusters,f)
            loaded_again=torch.load(target,map_location='cpu',weights_only=False)
            require(len(loaded_again)==3 and all(len(x)==len(y) and all(np.array_equal(i,j) for i,j in zip(x,y)) for x,y in zip(split_clusters,loaded_again)),
                    'Self-generated cluster save/readback mismatch')
            report['cluster_file']={'path':str(target),'bytes':target.stat().st_size,'sha256':sha(target),'nodes':len(clusters),
                                    'generation_seconds':time.monotonic()-full_start,'split_order':['train','valid','test'],
                                    'source':'Self-generated from frozen upstream ACL procedure; not the unavailable author precomputed file',
                                    'save_readback_equal':True}
            save(target.with_name(target.name+'.receipt.json'),report['cluster_file'])
            report['status']='COMPLETE_SELF_GENERATED_UPSTREAM_LC'
    except Exception as error:
        report.update(status='STOPPED_PRESERVED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    report['wall_seconds']=time.monotonic()-began;report['memory_after']=memory();save(out/'clustering.json',report)
    print('CLUSTERING_STATUS='+report['status'],flush=True)
    return 1 if report['status']=='STOPPED_PRESERVED' else 0


if __name__=='__main__':raise SystemExit(main())
