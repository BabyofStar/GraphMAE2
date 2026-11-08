"""Pure helpers for selected A's fixed protocol and committed resume pointers."""
import math
from pathlib import Path
import re
import statistics
from measure_common import require,sha,save


def validate_checkpoint(receipt,payload,job):
    require(payload['job_id']==job['job_id'] and payload['contract_sha256']==job['contract_sha256'],'Checkpoint belongs to another run or contract')
    require(payload['method']=='B0' and payload['pretrain_seed']==0,'Checkpoint method/seed differs')
    epoch=payload['completed_epochs']
    require(type(epoch) is int and 0<=epoch<=60 and receipt['completed_epochs']==epoch,'Checkpoint epoch differs')
    require(payload['scheduler_state']['last_epoch']==epoch,'Checkpoint scheduler horizon/state differs')
    return epoch


def commit_checkpoint(assets,payload,save_fn,attempt):
    assets=Path(assets).resolve();epoch=payload['completed_epochs']
    require(re.fullmatch(r'a\d{4,}',attempt) is not None,'Invalid attempt name')
    path=assets/f'resume_e{epoch:04d}_{attempt}.pt'
    require(not path.exists(),'Checkpoint target already exists')
    with path.open('xb') as stream:save_fn(payload,stream)
    receipt=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),completed_epochs=epoch,
                 job_id=payload['job_id'],contract_sha256=payload['contract_sha256'])
    # Commit the pointer last. A crash before this keeps the older complete epoch.
    save(assets/'resume_pointer.json',receipt)
    removed=[]
    for old in assets.glob('resume_e*.pt'):
        match=re.fullmatch(r'resume_e(\d{4})_a\d{4,}\.pt',old.name)
        if match and int(match[1])<epoch-1:
            require(old.resolve().parent==assets and not old.is_symlink(),'Checkpoint rotation escaped this run asset directory')
            removed.append(dict(path=str(old),bytes=old.stat().st_size,sha256=sha(old)))
            old.unlink()
    return receipt,removed


def summarize_probes(results):
    require(len(results)==10 and [r['probe_seed'] for r in results]==list(range(10)),'Require all fixed probe seeds0..9 in order')
    require(all(r['status']=='PASS_FULL_PROBE' and r['epochs_completed']==1000 for r in results),'Incomplete fixed probe protocol')
    require(all(r['pretrain_seed']==0 and 0<=r['best_val_epoch']<1000 and 0<=r['best_val_acc']<=1 for r in results),'Probe selection or pretrain seed differs')
    require(len({r['pretrain_checkpoint_sha256'] for r in results})==1 and len({r['embedding_sha256'] for r in results})==1,'Probe inputs differ')
    values=[r['val_selected_test_acc'] for r in results]
    require(all(math.isfinite(v) and 0<=v<=1 for v in values),'Invalid test accuracy')
    return dict(metric='val_selected_test_acc',mean=statistics.mean(values),population_std=statistics.pstdev(values),
                independent_pretrain_runs=1,pretrain_seed=0,probe_count=10,probe_seeds=list(range(10)),
                statistical_scope='Variation across ten linear classifiers of ONE encoder, not ten independent pretrained encoders')
