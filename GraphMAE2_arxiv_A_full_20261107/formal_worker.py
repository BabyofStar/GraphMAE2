"""One fixed B0 pretrain and ten fixed validation-selected linear probes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

from measure_common import require,save,sha,now,resource_snapshot,load_context
from formal_logic import validate_checkpoint,commit_checkpoint,summarize_probes
from runtime_core import build,loader,check_batch,training_step,attach_mask_check,shutdown,inference


def cpu_tree(value,torch):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {key:cpu_tree(item,torch) for key,item in value.items()}
    if isinstance(value,list):return [cpu_tree(item,torch) for item in value]
    if isinstance(value,tuple):return tuple(cpu_tree(item,torch) for item in value)
    return value


def capture(ctx,model,optimizer,scheduler,epoch,job):
    torch=ctx['torch']
    return dict(format_version=1,method='B0',pretrain_seed=0,completed_epochs=epoch,
                job_id=job['job_id'],contract_sha256=job['contract_sha256'],
                model_state=cpu_tree(model.state_dict(),torch),optimizer_state=cpu_tree(optimizer.state_dict(),torch),
                scheduler_state=scheduler.state_dict(),context=ctx['info'],
                rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),
                         numpy=ctx['np'].random.get_state(),python=random.getstate()))


def restore_rng(ctx,state):
    ctx['torch'].set_rng_state(state['torch']);ctx['torch'].cuda.set_rng_state_all(state['cuda'])
    ctx['np'].random.set_state(state['numpy']);random.setstate(state['python'])


def train(ctx,request,report,contract,job):
    from utils import create_optimizer
    torch,np=ctx['torch'],ctx['np'];assets=request.assets;out=request.output_dir
    model=build(ctx).to('cuda:0')
    optimizer=create_optimizer(ctx['args'].optimizer,model,ctx['args'].lr,ctx['args'].weight_decay)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch:(1+np.cos(epoch*np.pi/60))*0.5)
    mask_check=attach_mask_check(ctx,model)
    data=loader(ctx,ctx['flat_clusters'],256,True,0,contract)
    pointer=assets/'resume_pointer.json';start=0
    if pointer.exists():
        receipt=json.loads(pointer.read_text('utf-8'));path=Path(receipt['path'])
        require(path.resolve().parent==assets.resolve() and sha(path)==receipt['sha256'],'Committed checkpoint identity differs')
        state=torch.load(path,map_location='cpu',weights_only=False)
        start=validate_checkpoint(receipt,state,job)
        require(state['context']['scaled_feature_sha256']==ctx['info']['scaled_feature_sha256'],'Resume inputs differ')
        model.load_state_dict(state['model_state'],strict=True);optimizer.load_state_dict(state['optimizer_state']);scheduler.load_state_dict(state['scheduler_state'])
        if start>0 and not (out/f'epoch_{start:04d}.json').exists():
            require('last_epoch_summary' in state,'Committed checkpoint has no epoch summary')
            save(out/f'epoch_{start:04d}.json',dict(**state['last_epoch_summary'],checkpoint=receipt,
                                                 recovered_from_committed_checkpoint=True,rotated_checkpoints=[]))
        restore_rng(ctx,state['rng']);del state
    else:
        receipt,removed=commit_checkpoint(assets,capture(ctx,model,optimizer,scheduler,0,job),torch.save,request.attempt)
        save(out/f'checkpoint_e0000_{request.attempt}.json',dict(**receipt,rotated_checkpoints=removed))
    report['resumed_from_completed_epochs']=start
    batches=math.ceil(169343/256);began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(start,60):
            seen=np.zeros(169343,dtype=np.bool_);losses=[];epoch_start=time.monotonic();first_lr=optimizer.param_groups[0]['lr']
            trace_path=out/'traces'/f'pretrain_e{epoch+1:04d}_{request.attempt}.jsonl'
            with trace_path.open('x',encoding='utf-8') as trace:
                iterator=iter(data)
                for index in range(batches):
                    step_start=time.monotonic();batch=next(iterator);roots=check_batch(ctx,batch)
                    require(len(np.unique(roots))==len(roots) and not seen[roots].any(),'Repeated root within pretrain epoch')
                    seen[roots]=True;compute_start=time.monotonic()
                    metrics=training_step(ctx,model,optimizer,batch,epoch,mask_check)
                    metrics.update(batch=index+1,epoch=epoch+1,load_and_checks_seconds=compute_start-step_start,
                                   wall_seconds=time.monotonic()-step_start,root_ids_sha256=hashlib.sha256(roots.tobytes()).hexdigest())
                    trace.write(json.dumps(metrics,allow_nan=False)+'\n');trace.flush();losses.append(metrics['loss'])
                    if index<2 or (index+1)%20==0:
                        elapsed=time.monotonic()-epoch_start
                        save(out/'progress.json',dict(stage='pretrain',epoch=epoch+1,total_epochs=60,batches=index+1,total_batches=batches,
                             completed_epochs=epoch,wall_seconds=elapsed,estimated_current_epoch_remaining_seconds=elapsed/(index+1)*(batches-index-1),
                             estimated_pretrain_remaining_seconds=elapsed/(index+1)*(batches-index-1+(59-epoch)*batches)))
                        print(f'PRETRAIN epoch={epoch+1}/60 batch={index+1}/{batches} loss={metrics["loss"]:.5f} elapsed={elapsed:.1f}',flush=True)
                    del batch
                try:next(iterator);raise RuntimeError('Extra pretraining batch')
                except StopIteration:pass
            require(seen.all() and len(losses)==batches,'Incomplete full-epoch root coverage')
            torch.cuda.synchronize();seconds=time.monotonic()-epoch_start;scheduler.step()
            epoch_record=dict(epoch=epoch+1,batches=batches,roots=169343,loss_mean=statistics.mean(losses),
                 epoch_seconds=seconds,learning_rate=first_lr,next_learning_rate=optimizer.param_groups[0]['lr'],
                 trace_path=str(trace_path),trace_sha256=sha(trace_path),resources=resource_snapshot(torch),attempt=request.attempt)
            state=capture(ctx,model,optimizer,scheduler,epoch+1,job);state['last_epoch_summary']=epoch_record
            receipt,removed=commit_checkpoint(assets,state,torch.save,request.attempt);del state
            epoch_record.update(checkpoint=receipt,rotated_checkpoints=removed)
            save(out/f'epoch_{epoch+1:04d}.json',epoch_record)
            print(f'EPOCH_COMPLETE={epoch+1}/60 seconds={seconds:.1f} checkpoint={receipt["sha256"]}',flush=True)
        final=assets/'final_B0.pt'
        if not final.exists():
            temporary=assets/f'final_B0_{request.attempt}.tmp'
            with temporary.open('xb') as stream:
                torch.save(dict(model_state=cpu_tree(model.state_dict(),torch),completed_epochs=60,method='B0',pretrain_seed=0,
                                job_id=job['job_id'],contract_sha256=job['contract_sha256']),stream)
            temporary.replace(final)
        saved=torch.load(final,map_location='cpu',weights_only=False)
        require(saved['completed_epochs']==60 and saved['job_id']==job['job_id'] and saved['contract_sha256']==job['contract_sha256'],'Final checkpoint identity differs')
        original=cpu_tree(model.state_dict(),torch)
        require(all(torch.equal(value,saved['model_state'][key]) for key,value in original.items()),'Final saved model differs')
        model.cpu().eval();reloaded=build(ctx).eval();reloaded.load_state_dict(saved['model_state'],strict=True)
        small=loader(ctx,ctx['flat_clusters'][:2],2,False,0,contract);batch=next(iter(small));check_batch(ctx,batch)
        with torch.no_grad():
            before=model.embed(batch[0],batch[0].ndata['feat'])[batch[1]]
            after=reloaded.embed(batch[0],batch[0].ndata['feat'])[batch[1]]
        require(torch.equal(before,after),'Final CPU reload embedding differs')
        report.update(status='PASS_FULL_PRETRAIN',completed_epochs=60,checkpoint=dict(path=str(final),bytes=final.stat().st_size,
                      sha256=sha(final),strict_CPU_state_equal=True,CPU_embedding_exact_equal=True),
                      current_attempt_training_seconds=time.monotonic()-began)
    finally:shutdown(data)


def classify(ctx,request,report,contract,job):
    from datasets.lc_sampler import LinearProbingDataLoader
    from models.finetune import LogisticRegression
    from utils import set_random_seed
    torch,np=ctx['torch'],ctx['np'];out=request.output_dir;assets=request.assets
    embedding=json.loads((out/'embedding_receipt.json').read_text('utf-8'))
    require(sha(Path(embedding['path']))==embedding['sha256'],'Embedding changed before probes')
    final=json.loads((out/'pretrain.json').read_text('utf-8'))['checkpoint']
    require(sha(Path(final['path']))==final['sha256'],'Final encoder changed before probes')
    root_ids=np.load(embedding['root_ids_path'],allow_pickle=False)
    require(np.array_equal(root_ids,ctx['roots']),'Saved embedding row labels differ')
    values=np.load(embedding['path'],mmap_mode='c',allow_pickle=False);tensor=torch.from_numpy(values)
    a,b=90941,29799;parts=[tensor[:a],tensor[a:a+b],tensor[a+b:]]
    labels=[ctx['labels'][ctx['split'][name]].long() for name in ['train','valid','test']]
    results=[]
    for seed in range(10):
        result_path=out/f'probe_s{seed}.json'
        if result_path.exists():
            previous=json.loads(result_path.read_text('utf-8'))
            require(previous['status']=='PASS_FULL_PROBE' and previous['probe_seed']==seed and previous['epochs_completed']==1000 and
                    previous['job_id']==job['job_id'] and previous['contract_sha256']==job['contract_sha256'] and
                    previous['pretrain_checkpoint_sha256']==final['sha256'] and previous['embedding_sha256']==embedding['sha256'] and
                    sha(Path(previous['best_head']['path']))==previous['best_head']['sha256'],'Existing completed probe differs')
            results.append(previous);print(f'PROBE_REUSED seed={seed} completed=1000',flush=True);continue
        set_random_seed(seed);model=LogisticRegression(1024,40).to('cuda:0')
        optimizer=torch.optim.Adam(model.parameters(),lr=.005,weight_decay=.0001);criterion=torch.nn.CrossEntropyLoss()
        loaders=[LinearProbingDataLoader(np.arange(len(x)),x,label,batch_size=5120,num_workers=0,
                                        persistent_workers=False,shuffle=i==0) for i,(x,label) in enumerate(zip(parts,labels))]
        def evaluate(index):
            predictions=[]
            for features,_ in loaders[index]:predictions.append(model(None,features.to('cuda:0')).cpu())
            logits=torch.cat(predictions);require(torch.isfinite(logits).all().item(),'Nonfinite probe logits')
            return int((logits.argmax(1)==labels[index]).sum().item())/len(labels[index])
        best_val=-1.;best_epoch=-1;best_state=None;started=time.monotonic()
        with (out/'traces'/f'probe_s{seed}_{request.attempt}.jsonl').open('x',encoding='utf-8') as trace:
            for epoch in range(1000):
                model.train();seen=0;step_start=time.monotonic()
                for features,label in loaders[0]:
                    prediction=model(None,features.to('cuda:0'));loss=criterion(prediction,label.to('cuda:0'))
                    require(torch.isfinite(loss).item(),'Nonfinite probe loss')
                    optimizer.zero_grad();loss.backward();optimizer.step();seen+=len(label)
                require(seen==a,'Incomplete probe train rows')
                model.eval()
                with torch.no_grad():val=evaluate(1)
                if val>=best_val:
                    best_val=val;best_epoch=epoch;best_state=cpu_tree(model.state_dict(),torch)
                record=dict(epoch=epoch,probe_seed=seed,last_train_loss=float(loss.item()),val_acc=val,
                            best_val_acc=best_val,best_val_epoch=best_epoch,seconds=time.monotonic()-step_start)
                trace.write(json.dumps(record,allow_nan=False)+'\n');trace.flush()
                if (epoch+1)%100==0:
                    elapsed=time.monotonic()-started
                    save(out/'progress.json',dict(stage='linear_probe',probe_seed=seed,probe_index=seed+1,total_probes=10,
                         epochs=epoch+1,total_epochs=1000,wall_seconds=elapsed,
                         estimated_probe_stage_remaining_seconds=elapsed/(epoch+1)*(999-epoch+(9-seed)*1000)))
                    print(f'PROBE seed={seed} epoch={epoch+1}/1000 best_val={best_val:.6f} elapsed={elapsed:.1f}',flush=True)
        require(best_state is not None,'No validation-selected head')
        model.load_state_dict(best_state,strict=True);model.eval()
        with torch.no_grad():test=evaluate(2)
        head_path=assets/f'probe_s{seed}_best_{request.attempt}.pt'
        with head_path.open('xb') as stream:torch.save(dict(model_state=best_state,probe_seed=seed,best_val_epoch=best_epoch,
                                  best_val_acc=best_val,pretrain_checkpoint_sha256=final['sha256'],job_id=job['job_id']),stream)
        reloaded=torch.load(head_path,map_location='cpu',weights_only=False)
        require(all(torch.equal(value,reloaded['model_state'][name]) for name,value in best_state.items()),'Probe head saved state differs')
        result=dict(status='PASS_FULL_PROBE',probe_seed=seed,pretrain_seed=0,epochs_completed=1000,
                    best_val_epoch=best_epoch,best_val_acc=best_val,val_selected_test_acc=test,
                    pretrain_checkpoint_sha256=final['sha256'],embedding_sha256=embedding['sha256'],
                    job_id=job['job_id'],contract_sha256=job['contract_sha256'],attempt=request.attempt,
                    best_head=dict(path=str(head_path),sha256=sha(head_path),strict_state_readback_equal=True),wall_seconds=time.monotonic()-started)
        save(result_path,result);results.append(result);print(f'PROBE_COMPLETE seed={seed} test={test:.6f}',flush=True)
    summary=summarize_probes(results);save(out/'A_results.json',dict(**summary,method='B0',dataset='ogbn-arxiv',
         pretrain_epochs=60,probe_epochs=1000,pretrain_batch_size=256,graph_loader_workers=0,job_id=job['job_id'],results=results))
    report.update(status='PASS_TEN_FULL_PROBES',summary=summary)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pretrain','inference','classifier'],required=True)
    for name in ['snapshot','output-dir','assets','result']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--attempt',required=True);request=parser.parse_args()
    contract=json.loads((Path(__file__).parent/'formal_contract.json').read_text('utf-8'))
    job=json.loads((request.output_dir/'job_identity.json').read_text('utf-8'))
    request.batch_size=256;report=dict(stage=request.stage,status='STARTED',time_beijing=now(),attempt=request.attempt,
               job_id=job['job_id'],method='B0',pretrain_seed=0,resources_before=resource_snapshot())
    began=time.monotonic();torch=None
    try:
        require(job['contract_sha256']==sha(Path(__file__).parent/'formal_contract.json'),'Formal contract changed')
        require(job['kit_identity_sha256']==sha(Path(__file__).parent/'kit_identity.json'),'Runtime kit identity changed')
        for name,digest in json.loads((Path(__file__).parent/'kit_identity.json').read_text('utf-8')).items():
            require(sha(Path(__file__).parent/name)==digest,'Runtime kit file changed: '+name)
        sys.path.insert(0,str(request.snapshot))
        for name,digest in contract['source_sha256'].items():require(sha(request.snapshot/name)==digest,'Snapshot differs: '+name)
        ctx=load_context(contract,request.snapshot,request.output_dir);torch=ctx['torch'];report['context']=ctx['info']
        if request.stage=='pretrain':train(ctx,request,report,contract,job)
        elif request.stage=='inference':
            pre=json.loads((request.output_dir/'pretrain.json').read_text('utf-8'))
            require(pre['status']=='PASS_FULL_PRETRAIN' and sha(Path(pre['checkpoint']['path']))==pre['checkpoint']['sha256'],'Final pretrained encoder differs')
            inference(ctx,request,report,contract,request.output_dir,request.assets)
        else:classify(ctx,request,report,contract,job)
    except Exception as error:
        report.update(status='STOPPED_PRESERVED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    report.update(wall_seconds=time.monotonic()-began,resources_after=resource_snapshot(torch));save(request.result,report)
    print('FORMAL_STAGE_STATUS='+report['status'],flush=True)
    return 0 if report['status'].startswith('PASS_') else 1


if __name__=='__main__':raise SystemExit(main())
