"""Checked helpers from L1v2; inference uses the final encoder and attempt filenames."""
import hashlib
import json
import math
import statistics
import time
from measure_common import require,save,sha,resource_snapshot

class CostHold(RuntimeError):
    pass

def check_batch(ctx, batch):
    torch, dgl = ctx['torch'], ctx['dgl']
    graph, targets, _, node_ids = batch[:4]
    require(torch.equal(graph.ndata[dgl.NID].long(), node_ids.long()), 'DGL subgraph order differs from feature/global node positions')
    require(targets.numel()>0 and int(targets.min())>=0 and int(targets.max())<graph.num_nodes(), 'Invalid root targets')
    roots=node_ids[targets].numpy().astype('int64',copy=False)
    values=[ctx['flat_clusters'][ctx['root_lookup'][int(root)]] for root in roots]
    # Upstream removes/adds self loops after batching, which can discard DGL's
    # batch metadata. Validate its explicit targets against the actual LC lists.
    expected=torch.from_numpy(ctx['np'].cumsum([0]+[len(value) for value in values])[:-1])
    require(torch.equal(targets.long(),expected.long()), 'Root target positions differ')
    expected_ids=torch.from_numpy(ctx['np'].concatenate(values)).long()
    require(torch.equal(node_ids.long(),expected_ids), 'Ego-node concatenation order differs')
    require(torch.equal(graph.ndata['feat'], ctx['feats'][node_ids]), 'Batch feature alignment differs')
    return roots

def loader(ctx, clusters, batch_size, training, workers, contract):
    from datasets.lc_sampler import OnlineLCLoader
    kwargs = dict(batch_size=batch_size, shuffle=training, drop_last=False,
                  num_workers=workers, persistent_workers=workers>0)
    if workers:
        kwargs.update(prefetch_factor=1, timeout=contract['loader_batch_timeout_seconds'])
    return OnlineLCLoader(clusters, ctx['graph'], ctx['feats'],
                          drop_edge_rate=ctx['args'].drop_edge_rate if training else 0,
                          **kwargs)

def shutdown(value):
    iterator = getattr(value, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()

def build(ctx):
    from models import build_model
    model = build_model(ctx['args'])
    require(not model.input_masker.enabled and not model.input_masker.requires_feedback,
            'Measurement requires the untouched random B0 input-mask branch')
    return model

def attach_mask_check(ctx, model):
    original = model.encoding_mask_noise
    last = {}
    def checked(graph, features, mask_rate):
        result = original(graph, features, mask_rate)
        masks, keeps = result[2]
        count = graph.num_nodes()
        require(masks.numel() == int(mask_rate*count) and masks.numel()+keeps.numel() == count,
                'B0 exact batch-position mask count differs')
        last.update(batch_positions=count, mask_count=masks.numel(), keep_count=keeps.numel())
        return result
    model.encoding_mask_noise = checked
    return last

def training_step(ctx, model, optimizer, batch, epoch, mask_check):
    torch = ctx['torch']
    model.train()
    graph, targets = batch[0].to('cuda:0'), batch[1]
    first_drop, second_drop = batch[4].to('cuda:0'), batch[5].to('cuda:0')
    features = graph.ndata.pop('feat')
    loss = model(graph, features, targets, epoch, first_drop, second_drop)
    require(torch.isfinite(loss).item(), 'Nonfinite pretraining loss')
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
    require(torch.isfinite(norm).item(), 'Nonfinite pretraining gradient norm')
    optimizer.step()
    torch.cuda.synchronize()
    return dict(loss=float(loss.item()), grad_norm=float(norm.item()),
                graph_nodes=graph.num_nodes(),graph_edges=graph.num_edges(),
                first_drop_edges=first_drop.num_edges(),second_drop_edges=second_drop.num_edges(),
                roots=targets.numel(),epoch_argument=epoch,**mask_check)

def inference(ctx, request, report, contract, output, assets):
    torch,np = ctx['torch'],ctx['np']
    payload=torch.load(assets/'final_B0.pt',map_location='cpu',weights_only=False)
    model=build(ctx)
    model.load_state_dict(payload['model_state'],strict=True)
    del payload
    model=model.to('cuda:0').eval()
    path=assets/f'all_root_embeddings_{request.attempt}.npy'
    require(not path.exists(),'Embedding target already exists')
    embeddings=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(169343,1024))
    data=loader(ctx,ctx['flat_clusters'],request.batch_size,False,contract['graph_workers'],contract)
    cursor=0;total_batches=math.ceil(169343/request.batch_size);times=[]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.monotonic()
    try:
        with (output/f'inference_batches_{request.attempt}.jsonl').open('x',encoding='utf-8') as trace, torch.no_grad():
            iterator=iter(data)
            for index in range(total_batches):
                if time.monotonic()-began>contract['inference_cap_seconds']:
                    raise CostHold('Full embedding engineering time cap reached')
                step_start=time.monotonic();batch=next(iterator);roots=check_batch(ctx,batch)
                require(np.array_equal(roots,ctx['roots'][cursor:cursor+len(roots)]),'Embedding row order differs from official split order')
                graph=batch[0].to('cuda:0');features=graph.ndata.pop('feat')
                values=model.embed(graph,features)[batch[1].to('cuda:0')].cpu()
                require(tuple(values.shape)==(len(roots),1024) and torch.isfinite(values).all().item(),'Nonfinite or malformed embeddings')
                embeddings[cursor:cursor+len(roots)]=values.numpy();cursor+=len(roots)
                seconds=time.monotonic()-step_start;times.append(seconds)
                metrics=dict(batch=index+1,rows=cursor,graph_nodes=graph.num_nodes(),graph_edges=graph.num_edges(),seconds=seconds)
                trace.write(json.dumps(metrics,allow_nan=False)+'\n');trace.flush()
                if (index+1)%10==0 or index<3:
                    state=dict(stage='inference',batches=index+1,total_batches=total_batches,rows=cursor,wall_seconds=time.monotonic()-began,resources=resource_snapshot(torch))
                    save(output/'progress.json',state)
                    print(f'EMBEDDING {index+1}/{total_batches} rows={cursor}/169343 elapsed={state["wall_seconds"]:.1f}',flush=True)
                if index==9:
                    projected=time.monotonic()-began+(total_batches-index-1)*statistics.median(times[-5:])
                    report['early_full_embedding_estimate_seconds']=projected
                    if projected>contract['inference_cap_seconds']:
                        raise CostHold('Fixed ten-batch timing projects beyond the embedding engineering cap')
                del batch,graph,features,values
            try:
                next(iterator)
                raise RuntimeError('Unexpected extra embedding batch')
            except StopIteration:
                pass
        torch.cuda.synchronize();require(cursor==169343,'Embedding coverage incomplete')
        compute_seconds=time.monotonic()-began
        flush_start=time.monotonic();embeddings.flush();del embeddings
        roots_path=assets/f'embedding_root_ids_{request.attempt}.npy'
        with roots_path.open('xb') as stream:np.save(stream,ctx['roots'],allow_pickle=False)
        saved=np.load(path,mmap_mode='r',allow_pickle=False)
        require(saved.shape==(169343,1024) and saved.dtype==np.float32 and np.isfinite(saved).all(),'Saved embedding shape/finite check failed')
        record=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),shape=list(saved.shape),dtype=str(saved.dtype),
                    root_ids_path=str(roots_path),root_ids_sha256=sha(roots_path),
                    root_order_sha256=hashlib.sha256(ctx['roots'].tobytes()).hexdigest(),
                    rows=cursor,compute_and_batch_IO_seconds=compute_seconds,
                    flush_verify_and_hash_seconds=time.monotonic()-flush_start)
        save(output/'embedding_receipt.json',record)
        report.update(status='PASS_FULL_EMBEDDING_EXTRACTION',embedding=record,embedding_seconds=time.monotonic()-began)
    finally:
        shutdown(data)
        report['partial_rows_observed']=cursor
