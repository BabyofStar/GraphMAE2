"""Bounded B0 capacity, one-epoch, embedding and classifier timing subprocesses."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

from measure_common import (require, save, sha, now, resource_snapshot, is_cuda_oom,
                            load_context, pilot_groups)


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


def capacity(ctx, request, report, contract):
    from utils import create_optimizer
    torch = ctx['torch']
    model = build(ctx)
    if request.stage == 'infer_capacity':
        payload = torch.load(Path(request.assets)/'one_epoch_B0.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(payload['model_state'],strict=True)
        del payload
    model = model.to('cuda:0')
    if request.stage == 'train_capacity':
        optimizer = create_optimizer(ctx['args'].optimizer,model,ctx['args'].lr,ctx['args'].weight_decay)
        mask_check = attach_mask_check(ctx,model)
    torch.cuda.reset_peak_memory_stats()
    measurements = []
    for index,(name,values) in enumerate(pilot_groups(ctx['clusters'],request.batch_size,ctx['np'])):
        began = time.monotonic()
        data = loader(ctx,values,request.batch_size,request.stage=='train_capacity',0,contract)
        try:
            batch = next(iter(data))
            roots = check_batch(ctx,batch)
            if request.stage == 'train_capacity':
                epoch = 40 if index==2 else 0
                metrics = training_step(ctx,model,optimizer,batch,epoch,mask_check)
            else:
                model.eval()
                with torch.no_grad():
                    graph = batch[0].to('cuda:0')
                    features = graph.ndata.pop('feat')
                    values = model.embed(graph,features)[batch[1].to('cuda:0')]
                    require(tuple(values.shape)==(request.batch_size,1024) and torch.isfinite(values).all().item(), 'Embedding capacity check differs')
                    torch.cuda.synchronize()
                    metrics = dict(graph_nodes=graph.num_nodes(),graph_edges=graph.num_edges(),embedding_shape=list(values.shape))
            metrics.update(split=name,seconds=time.monotonic()-began,
                           root_ids_sha256=hashlib.sha256(roots.tobytes()).hexdigest(),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
            measurements.append(metrics)
            print(f'CAPACITY {request.stage} batch={request.batch_size} split={name} PASS seconds={metrics["seconds"]:.3f}',flush=True)
        finally:
            shutdown(data)
    report.update(status='PASS_CAPACITY_CHECK',measurements=measurements,
                  pilot_protocol='Fixed evenly spaced roots in train/valid/test. Training probes use epochs 0/0/40, fresh initialization per candidate; no accuracy used.')


def save_checkpoint(ctx, model, optimizer, scheduler, assets, report, snapshot):
    torch = ctx['torch']
    from datasets.lc_sampler import OnlineLCLoader
    began = time.monotonic()
    model.cpu().eval()
    small = OnlineLCLoader(ctx['flat_clusters'][:2],ctx['graph'],ctx['feats'],batch_size=2,shuffle=False,num_workers=0)
    batch = next(iter(small))
    check_batch(ctx,batch)
    with torch.no_grad():
        before = model.embed(batch[0],batch[0].ndata['feat'])[batch[1]].clone()
    payload = dict(format_version=1,scope='B0 one-epoch engineering measurement; excluded from formal L2 results',
                   model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),
                   completed_epochs=1,pretrain_seed=0,frozen_effective_args=vars(ctx['args']),
                   measurement_contract_sha256=sha(Path(__file__).parent/'measurement_contract.json'),
                   source_identity_sha256=sha(Path(snapshot).parent/'source_identity.json'),
                   actual_runtime=report['actual_runtime'],input_identities=ctx['info'],
                   rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=ctx['np'].random.get_state(),python=random.getstate()))
    path = assets/'one_epoch_B0.pt'
    with path.open('xb') as stream:
        torch.save(payload,stream)
    write_seconds = time.monotonic()-began
    restored = torch.load(path,map_location='cpu',weights_only=False)
    require(restored['completed_epochs']==1 and restored['pretrain_seed']==0 and
            restored['scheduler_state']['last_epoch']==1,'Checkpoint epoch/seed/scheduler identity differs')
    reloaded = build(ctx)
    reloaded.load_state_dict(restored['model_state'],strict=True)
    reloaded.eval()
    require(all(torch.equal(value,restored['model_state'][name]) for name,value in model.state_dict().items()), 'Saved state readback differs')
    with torch.no_grad():
        after = reloaded.embed(batch[0],batch[0].ndata['feat'])[batch[1]]
    require(torch.equal(before,after), 'CPU reloaded model embedding differs')
    require(all(torch.equal(value,reloaded.state_dict()[name]) for name,value in model.state_dict().items()), 'CPU reloaded state differs')
    report['checkpoint'] = dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),
                                write_and_CPU_sample_seconds=write_seconds,total_save_reload_seconds=time.monotonic()-began,
                                strict_state_readback_equal=True,CPU_embedding_exact_equal=True,sample_roots=2,
                                excluded_from_formal_runs=True)


def pretrain(ctx, request, report, contract, output, assets):
    from utils import create_optimizer
    torch, np = ctx['torch'], ctx['np']
    model = build(ctx).to('cuda:0')
    optimizer = create_optimizer(ctx['args'].optimizer,model,ctx['args'].lr,ctx['args'].weight_decay)
    # Keep the frozen 60-epoch horizon; this measurement stops after epoch 0.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch:(1+np.cos(epoch*np.pi/60))*0.5)
    mask_check = attach_mask_check(ctx,model)
    data = loader(ctx,ctx['flat_clusters'],request.batch_size,True,contract['graph_workers'],contract)
    expected = math.ceil(len(ctx['roots'])/request.batch_size)
    seen = np.zeros(169343,dtype=np.bool_)
    observations = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    began = time.monotonic()
    try:
        with (output/'pretrain_batches.jsonl').open('x',encoding='utf-8') as trace:
            iterator = iter(data)
            for index in range(expected):
                if time.monotonic()-began > contract['pretrain_epoch_cap_seconds']:
                    raise CostHold('One-epoch engineering time cap reached')
                batch_start = time.monotonic()
                batch = next(iterator)
                roots = check_batch(ctx,batch)
                require(len(np.unique(roots))==len(roots) and not seen[roots].any(), 'Repeated pretraining roots in an epoch')
                seen[roots]=True
                compute_start = time.monotonic()
                metrics = training_step(ctx,model,optimizer,batch,0,mask_check)
                metrics.update(batch=index+1,load_and_checks_seconds=compute_start-batch_start,
                               compute_including_transfer_seconds=time.monotonic()-compute_start,
                               wall_seconds=time.monotonic()-batch_start,
                               root_ids_sha256=hashlib.sha256(roots.tobytes()).hexdigest())
                observations.append(metrics)
                trace.write(json.dumps(metrics,allow_nan=False)+'\n');trace.flush()
                if (index+1)%10==0 or index<3:
                    state=dict(stage='pretrain',batches=index+1,total_batches=expected,roots_seen=int(seen.sum()),
                               wall_seconds=time.monotonic()-began,resources=resource_snapshot(torch))
                    save(output/'progress.json',state)
                    print(f'PRETRAIN {index+1}/{expected} roots={int(seen.sum())}/169343 elapsed={state["wall_seconds"]:.1f}',flush=True)
                if index==9:
                    remaining = (expected-index-1)*statistics.median(m['wall_seconds'] for m in observations[-5:])
                    report['early_full_epoch_estimate_seconds'] = time.monotonic()-began+remaining
                    if report['early_full_epoch_estimate_seconds'] > contract['pretrain_epoch_cap_seconds']:
                        raise CostHold('Fixed ten-batch timing projects beyond the one-epoch engineering cap')
                del batch
            try:
                next(iterator)
                raise RuntimeError('Unexpected extra pretraining batch')
            except StopIteration:
                pass
        torch.cuda.synchronize()
        require(seen.all() and len(observations)==expected,'Incomplete pretraining epoch coverage')
        epoch_seconds = time.monotonic()-began
        scheduler.step()
        report.update(completed_epochs=1,completed_batches=expected,roots_seen=int(seen.sum()),
                      epoch_seconds=epoch_seconds,loss_mean=float(np.mean([m['loss'] for m in observations])),
                      batch_seconds_median=float(statistics.median(m['wall_seconds'] for m in observations)),
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      scheduler_horizon_epochs=60,epoch_argument=0,resources_after_epoch=resource_snapshot(torch))
    finally:
        shutdown(data)
        report['partial_batches_observed']=len(observations)
        report['partial_roots_seen']=int(seen.sum())
    save_checkpoint(ctx,model,optimizer,scheduler,assets,report,request.snapshot)
    report['status']='PASS_ONE_FULL_PRETRAIN_EPOCH'


def inference(ctx, request, report, contract, output, assets):
    torch,np = ctx['torch'],ctx['np']
    payload=torch.load(assets/'one_epoch_B0.pt',map_location='cpu',weights_only=False)
    model=build(ctx)
    model.load_state_dict(payload['model_state'],strict=True)
    del payload
    model=model.to('cuda:0').eval()
    path=assets/'all_root_embeddings.npy'
    require(not path.exists(),'Embedding target already exists')
    embeddings=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(169343,1024))
    data=loader(ctx,ctx['flat_clusters'],request.batch_size,False,contract['graph_workers'],contract)
    cursor=0;total_batches=math.ceil(169343/request.batch_size);times=[]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.monotonic()
    try:
        with (output/'inference_batches.jsonl').open('x',encoding='utf-8') as trace, torch.no_grad():
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
        roots_path=assets/'embedding_root_ids.npy'
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


def classifier(ctx, request, report, contract, output, assets):
    import torch.nn as nn
    from datasets.lc_sampler import LinearProbingDataLoader
    from models.finetune import LogisticRegression
    from utils import set_random_seed
    torch,np=ctx['torch'],ctx['np']
    receipt=json.loads((output/'embedding_receipt.json').read_text('utf-8'))
    require(sha(assets/'all_root_embeddings.npy')==receipt['sha256'],'Embedding input hash differs')
    values=np.load(assets/'all_root_embeddings.npy',mmap_mode='c',allow_pickle=False)
    tensors=torch.from_numpy(values)
    a,b=contract['split_counts']['train'],contract['split_counts']['valid']
    parts=[tensors[:a],tensors[a:a+b],tensors[a+b:]]
    train_labels=ctx['labels'][ctx['split']['train']].long()
    # No validation/test accuracy is computed in this short engineering run.
    labels=[train_labels,torch.zeros(b,dtype=torch.long),torch.zeros(len(parts[2]),dtype=torch.long)]
    loaders=[LinearProbingDataLoader(np.arange(len(x)),x,label,batch_size=5120,num_workers=0,
                                    persistent_workers=False,shuffle=index==0)
             for index,(x,label) in enumerate(zip(parts,labels))]
    set_random_seed(0)
    model=LogisticRegression(1024,40).to('cuda:0')
    optimizer=torch.optim.Adam(model.parameters(),lr=ctx['args'].lr_f,weight_decay=ctx['args'].weight_decay_f)
    criterion=nn.CrossEntropyLoss()
    def forward_all(data):
        predictions=[]
        for x,_ in data:predictions.append(model(None,x.to('cuda:0')).cpu())
        all_predictions=torch.cat(predictions)
        require(torch.isfinite(all_predictions).all().item(),'Nonfinite classifier logits')
        return len(all_predictions)
    epochs=[];began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with (output/'classifier_epochs.jsonl').open('x',encoding='utf-8') as trace:
        for epoch in range(contract['probe_measurement_epochs']):
            if time.monotonic()-began>contract['probe_cap_seconds']:
                raise CostHold('Classifier engineering time cap reached')
            torch.cuda.synchronize();start=time.monotonic();model.train();seen=0;losses=[]
            for features,label in loaders[0]:
                prediction=model(None,features.to('cuda:0'));loss=criterion(prediction,label.to('cuda:0'))
                require(torch.isfinite(loss).item(),'Nonfinite classifier loss')
                optimizer.zero_grad();loss.backward();optimizer.step();seen+=len(label);losses.append(float(loss.item()))
            require(seen==a,'Classifier training coverage incomplete');torch.cuda.synchronize()
            training_seconds=time.monotonic()-start;val_start=time.monotonic();model.eval()
            with torch.no_grad():require(forward_all(loaders[1])==b,'Validation forward coverage differs')
            torch.cuda.synchronize();val_seconds=time.monotonic()-val_start
            # Include an upper-bound copy cost for validation model selection without
            # producing or selecting on validation/test scores at one pretrain epoch.
            copy_start=time.monotonic();temporary_model=copy.deepcopy(model);del temporary_model
            torch.cuda.synchronize();copy_seconds=time.monotonic()-copy_start
            record=dict(epoch=epoch,train_seconds=training_seconds,validation_forward_seconds=val_seconds,
                        model_copy_seconds=copy_seconds,total_seconds=time.monotonic()-start,
                        training_rows=seen,validation_rows=b,last_train_loss=losses[-1])
            trace.write(json.dumps(record,allow_nan=False)+'\n');trace.flush();epochs.append(record)
            print(f'PROBE_TIMING epoch={epoch+1}/{contract["probe_measurement_epochs"]} seconds={record["total_seconds"]:.3f}',flush=True)
        test_start=time.monotonic();model.eval()
        with torch.no_grad():require(forward_all(loaders[2])==contract['split_counts']['test'],'Test-shape forward coverage differs')
        torch.cuda.synchronize();test_seconds=time.monotonic()-test_start
    report.update(status='PASS_CLASSIFIER_TIMING',probe_seed=0,probe_epochs_measured=len(epochs),
                  probe_batch_size=5120,probe_loader_workers=0,epoch_timings=epochs,
                  steady_probe_epoch_seconds=statistics.median(e['total_seconds'] for e in epochs[1:]),
                  cold_probe_epoch_seconds=epochs[0]['total_seconds'],test_forward_seconds=test_seconds,
                  no_validation_or_test_accuracy_computed=True,
                  classifier_protocol='Frozen Adam/linear head/full train rows and full validation forward per epoch; final full test-shape forward once. Dummy validation/test labels, no scores, copied head each epoch to include selection overhead.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['train_capacity','pretrain','infer_capacity','inference','classifier'])
    parser.add_argument('--batch-size',type=int,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    request=parser.parse_args();snapshot=request.snapshot.resolve();output=request.output_dir.resolve();assets=request.assets.resolve()
    contract=json.loads((Path(__file__).parent/'measurement_contract.json').read_text('utf-8'))
    report=dict(stage=request.stage,batch_size=request.batch_size,time_beijing=now(),pretrain_seed=0,
                status='STARTED',scope='Engineering measurement, not a full pretrain or a formal accuracy result',
                actual_runtime=dict(batch_size=request.batch_size,graph_workers=contract['graph_workers'],
                                    torch_CPU_threads=1,graph_prefetch_factor=1,probe_workers=0),resources_before=resource_snapshot())
    started=time.monotonic();torch=None
    try:
        sys.path.insert(0,str(snapshot))
        for relative,expected in contract['source_sha256'].items():
            require(sha(snapshot/relative)==expected,'Frozen snapshot differs: '+relative)
        ctx=load_context(contract,snapshot,output);torch=ctx['torch'];report['context']=ctx['info']
        if request.stage.endswith('_capacity'):
            capacity(ctx,request,report,contract)
        elif request.stage=='pretrain':pretrain(ctx,request,report,contract,output,assets)
        elif request.stage=='inference':inference(ctx,request,report,contract,output,assets)
        else:classifier(ctx,request,report,contract,output,assets)
    except CostHold as error:
        report.update(status='HELD_FOR_ENGINEERING_COST',error=str(error))
    except Exception as error:
        oom=(torch is not None and isinstance(error,torch.cuda.OutOfMemoryError)) or is_cuda_oom(error)
        report.update(status='CUDA_OOM' if oom else 'STOPPED_PRESERVED',error_type=type(error).__name__,
                      error=str(error),traceback=traceback.format_exc())
    report['resources_after']=resource_snapshot(torch);report['wall_seconds']=time.monotonic()-started
    save(request.result,report)
    print('MEASUREMENT_STAGE_STATUS='+report['status'],flush=True)
    return 0 if report['status'].startswith('PASS_') else 1


if __name__=='__main__':raise SystemExit(main())
