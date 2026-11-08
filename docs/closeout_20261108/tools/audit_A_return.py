"""Independent finite audit of the existing complete A return; never trains or loads weights."""
import csv
import argparse
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import statistics as st
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/GraphMAE2_arxiv_A_audit_20261108'
KIT=ROOT/'outputs/GraphMAE2_arxiv_A_full_20261107'
INPUT=Path(r'C:\Users\18870\Downloads\A_return_a0001.zip')
MEASUREMENT=ROOT/'outputs/GraphMAE2_arxiv_L1_audit_20261107/inputs/measurement_return.zip'


def need(value,message):
    if not value:raise AssertionError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def close(a,b):return math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11)


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True);(OUT/'inputs').mkdir(exist_ok=True)
    path=OUT/'inputs/A_return_a0001.zip';source=INPUT
    if not path.exists():shutil.copyfile(source,path)
    need(path.read_bytes()==source.read_bytes(),'Input copy differs')
    digest=sha(path.read_bytes());need(digest=='e622b5c4599f14535ba592a784741c1ec5b6bab6247b360f8cde021562e845e2','Unexpected A input')
    contract=json.loads((KIT/'formal_contract.json').read_text('utf-8'))
    epochs=[];probe_rows=[];val_curves=[];assets=[];batch_count=0;mask_count=0;first_epoch_hashes=[]
    with zipfile.ZipFile(path) as z:
        need(z.testzip() is None,'ZIP CRC differs');names=z.namelist()
        need(len(names)==len(set(names))==176,'ZIP members differ')
        need(all(not n.startswith(('/', '\\')) and '..' not in Path(n).parts for n in names),'Unsafe ZIP path')
        need(sum(i.file_size for i in z.infolist())<40*1024**2,'Unexpected input size')
        manifest=json.loads(z.read('attempts/a0001/return_manifest.json'))
        need(len(manifest['files'])==168,'Unexpected manifest count')
        for item in manifest['files']:
            raw=z.read(item['path']);need(len(raw)==item['bytes'] and sha(raw)==item['sha256'],'Manifest mismatch: '+item['path'])
        for name in ['formal_contract.json','formal_worker.py','formal_driver.py','formal_logic.py','measure_common.py','runtime_core.py','kit_identity.json']:
            need(z.read('formal_kit/'+name)==(KIT/name).read_bytes(),'Delivered runtime differs: '+name)
        need(json.loads(z.read('frozen_contract.json'))==contract,'Frozen contract differs')
        job=json.loads(z.read('job_identity.json'));result=json.loads(z.read('A_results.json'));status=json.loads(z.read('formal_status.json'))
        need(status['status']=='PASS_A_FULL_RUN' and status['formal_pretrain_runs_completed']==1 and status['completed_pretrain_epochs']==60 and status['completed_probe_runs']==10,'A is not complete')
        need(job['job_id']=='arxiv_A_TxrQqJzj' and job['contract_sha256']==sha((KIT/'formal_contract.json').read_bytes()) and
             job['kit_identity_sha256']==sha((KIT/'kit_identity.json').read_bytes()),'Job or runtime identity differs')
        need(not job['measurement_checkpoint_reused'] and job['pretrain_seed']==0 and job['probe_seeds']==list(range(10)),'Run/seed protocol differs')
        need(job['source_sha256']==contract['source_sha256'] and job['actual_pretrain_batch_size']==job['actual_inference_batch_size']==256 and job['graph_workers']==0,'Adaptation or source identity differs')
        with tarfile.open(fileobj=io.BytesIO(z.read('source_snapshot.tar')),mode='r:*') as archive:
            members=archive.getmembers();need(len(members)==57 and {m.name for m in members}==set(contract['source_sha256']),'Snapshot member list differs')
            for member in members:need(member.isfile() and sha(archive.extractfile(member).read())==contract['source_sha256'][member.name],'Snapshot bytes differ')
        need(status['base_before']==status['base_after'] and status['base_versions_unchanged'] and not status['project_source_differences_at_end'],'Environment or project changed')
        pre=json.loads(z.read('pretrain.json'));infer=json.loads(z.read('inference.json'));classifier=json.loads(z.read('classifier.json'))
        need(pre==json.loads(z.read('pretrain_accepted.json')) and infer==json.loads(z.read('inference_accepted.json')),'Accepted stage differs')
        need(pre['status']=='PASS_FULL_PRETRAIN' and pre['completed_epochs']==60 and pre['resumed_from_completed_epochs']==0,'Pretrain protocol differs')
        need(pre['checkpoint']['strict_CPU_state_equal'] and pre['checkpoint']['CPU_embedding_exact_equal'],'Final save/reload receipts missing')
        for stage,wanted in [(pre,'PASS_FULL_PRETRAIN'),(infer,'PASS_FULL_EMBEDDING_EXTRACTION'),(classifier,'PASS_TEN_FULL_PROBES')]:
            need(stage['status']==wanted and stage['attempt']=='a0001','Unexpected stage status or attempt')
            need(stage['context']['effective_args']==contract['expected_effective_args']|{'num_features':128},'Effective args differ')
            need(stage['context']['cluster_checks']['exact_returned_pilot_checks']==256 and stage['context']['cluster_checks']['roots']==169343,'Data checks missing')
        need(len({x['context']['scaled_feature_sha256'] for x in [pre,infer,classifier]})==1,'Cross-stage feature identity differs')
        initial=json.loads(z.read('checkpoint_e0000_a0001.json'));receipts={0:{k:v for k,v in initial.items() if k!='rotated_checkpoints'}}
        removed=set()
        for epoch in range(1,61):
            info=json.loads(z.read(f'epoch_{epoch:04d}.json'));trace_name=f'traces/pretrain_e{epoch:04d}_a0001.jsonl';raw=z.read(trace_name)
            need(sha(raw)==info['trace_sha256'],'Epoch trace SHA differs')
            rows=[json.loads(line) for line in raw.splitlines()]
            need(len(rows)==info['batches']==662 and info['roots']==169343 and info['epoch']==epoch,'Epoch coverage differs')
            need(close(info['learning_rate'],.0025*(1+math.cos((epoch-1)*math.pi/60))/2) and
                 close(info['next_learning_rate'],.0025*(1+math.cos(epoch*math.pi/60))/2),'Cosine horizon differs')
            for index,row in enumerate(rows,1):
                need(row['batch']==index and row['epoch']==epoch and row['epoch_argument']==epoch-1,'Batch/EMA epoch argument differs')
                need(row['roots']==(256 if index<662 else 127),'Batch root count differs')
                need(row['mask_count']==row['graph_nodes']//2 and row['mask_count']+row['keep_count']==row['graph_nodes']==row['batch_positions'],'Exact mask quota differs')
                need(all(math.isfinite(row[key]) and row[key]>=0 for key in ['loss','grad_norm','wall_seconds','load_and_checks_seconds']),'Nonfinite training value')
                need(all(row['graph_nodes']<=row[key]<=row['graph_edges'] for key in ['first_drop_edges','second_drop_edges']),'Drop-edge range differs')
                need(len(row['root_ids_sha256'])==64,'Root digest missing')
            need(sum(row['roots'] for row in rows)==169343 and sum(row['graph_nodes'] for row in rows)==42884864,'Epoch root/ego-position totals differ')
            need(close(st.mean(row['loss'] for row in rows),info['loss_mean']),'Epoch loss summary differs')
            if epoch==1:first_epoch_hashes=[row['root_ids_sha256'] for row in rows]
            batch_count+=len(rows);mask_count+=sum(row['mask_count'] for row in rows)
            cp=info['checkpoint'];need(cp['completed_epochs']==epoch and cp['job_id']==job['job_id'] and cp['contract_sha256']==job['contract_sha256'],'Checkpoint identity differs')
            receipts[epoch]=cp
            for old in info['rotated_checkpoints']:
                previous=next((old_epoch for old_epoch,record in receipts.items() if record['path']==old['path']),None)
                need(previous is not None and previous<=epoch-2 and receipts[previous]['sha256']==old['sha256'],'Checkpoint rotation chain differs')
                removed.add(previous)
            epochs.append(dict(epoch=epoch,loss_mean=info['loss_mean'],epoch_seconds=info['epoch_seconds'],learning_rate=info['learning_rate'],
                 root_count=info['roots'],batch_count=len(rows),total_ego_positions=sum(row['graph_nodes'] for row in rows),
                 masks=sum(row['mask_count'] for row in rows),gradient_norm_max=max(row['grad_norm'] for row in rows),
                 checkpoint_sha256=cp['sha256'],trace_sha256=info['trace_sha256'],
                 peak_allocated_bytes=info['resources']['cuda']['peak_allocated_bytes'],peak_reserved_bytes=info['resources']['cuda']['peak_reserved_bytes']))
        need(removed==set(range(59)),'Unexpected checkpoint retention range')
        embed=infer['embedding'];need(embed==json.loads(z.read('embedding_receipt.json')) and embed['shape']==[169343,1024] and embed['dtype']=='float32','Embedding receipt differs')
        infer_traces=[n for n in names if n.startswith('inference_batches_') and n.endswith('.jsonl')];need(len(infer_traces)==1,'Unexpected inference attempts')
        rows=[json.loads(line) for line in z.read(infer_traces[0]).splitlines()]
        need(len(rows)==662 and sum(row['graph_nodes'] for row in rows)==42884864,'Inference coverage differs')
        for index,row in enumerate(rows,1):need(row['batch']==index and row['rows']==min(index*256,169343),'Inference row order differs')
        full_results=[]
        for seed in range(10):
            metric=json.loads(z.read(f'probe_s{seed}.json'));need(metric==result['results'][seed],'Probe result list differs')
            need(metric['status']=='PASS_FULL_PROBE' and metric['epochs_completed']==1000 and metric['probe_seed']==seed and metric['pretrain_seed']==0,'Probe protocol differs')
            need(metric['job_id']==job['job_id'] and metric['contract_sha256']==job['contract_sha256'] and metric['pretrain_checkpoint_sha256']==pre['checkpoint']['sha256'] and metric['embedding_sha256']==embed['sha256'],'Probe inputs differ')
            need(metric['best_head']['strict_state_readback_equal'],'Head readback missing')
            trace=[json.loads(line) for line in z.read(f'traces/probe_s{seed}_a0001.jsonl').splitlines()];need(len(trace)==1000,'Probe trace incomplete')
            best=-1.;best_epoch=-1
            for epoch,row in enumerate(trace):
                need(row['epoch']==epoch and row['probe_seed']==seed and 0<=row['val_acc']<=1 and math.isfinite(row['last_train_loss']) and row['last_train_loss']>=0,'Probe trace differs')
                need(abs(row['val_acc']*29799-round(row['val_acc']*29799))<1e-7,'Validation accuracy off count grid')
                need(not any('test' in key.lower() for key in row),'Test metric entered probe selection trace')
                if row['val_acc']>=best:best=row['val_acc'];best_epoch=epoch
                need(row['best_val_acc']==best and row['best_val_epoch']==best_epoch,'Sequential validation model selection differs')
                val_curves.append(dict(probe_seed=seed,epoch=epoch,validation_accuracy_percent=row['val_acc']*100))
            need(metric['best_val_acc']==best and metric['best_val_epoch']==best_epoch,'Best validation epoch mismatch')
            need(0<=metric['val_selected_test_acc']<=1 and abs(metric['val_selected_test_acc']*48603-round(metric['val_selected_test_acc']*48603))<1e-7,'Test accuracy off count grid')
            full_results.append(metric)
            probe_rows.append(dict(pretrain_seed=0,probe_seed=seed,epochs=1000,best_val_epoch_zero_based=best_epoch,
                 validation_accuracy_percent=best*100,test_accuracy_percent=metric['val_selected_test_acc']*100,
                 correct_test_nodes=round(metric['val_selected_test_acc']*48603),test_nodes=48603,wall_seconds=metric['wall_seconds']))
            assets.append(dict(kind='best_linear_head',path=metric['best_head']['path'],sha256=metric['best_head']['sha256'],bytes='not_in_receipt',state='server_receipt_not_downloaded'))
        values=[row['val_selected_test_acc'] for row in full_results];mean=st.mean(values);sd=st.pstdev(values)
        for summary in [result,classifier['summary'],status['result']]:
            need(close(summary['mean'],mean) and close(summary['population_std'],sd) and summary['independent_pretrain_runs']==1 and summary['probe_count']==10,'Summary or statistical unit differs')
        for stage in ['pretrain','inference','classifier']:
            assessed=json.loads(z.read(f'attempts/a0001/{stage}_assessment.json'));need('allocator_oom_log_lines' not in assessed and assessed['status'].startswith('PASS_'),'Allocator assessment failed')
            text=z.read(f'attempts/a0001/{stage}.log').decode('utf-8',errors='replace').lower()
            need('traceback (most recent' not in text and 'bus error' not in text and not any('hami-core' in line and ('oom' in line or 'out of memory' in line) for line in text.splitlines()),'Runtime error log')
        for point in [status['resources_before'],status['resources_after'],pre['resources_after'],infer['resources_after'],classifier['resources_after']]:
            events=dict(line.split() for line in point['memory.events'].splitlines());need(events['oom']==events['oom_kill']=='0','Cgroup OOM occurred')
        for epoch in [59,60]:assets.append(dict(kind=f'retained_resume_epoch_{epoch}',path=receipts[epoch]['path'],sha256=receipts[epoch]['sha256'],bytes=receipts[epoch]['bytes'],state='server_receipt_not_downloaded'))
        assets.extend([dict(kind='final_encoder',path=pre['checkpoint']['path'],sha256=pre['checkpoint']['sha256'],bytes=pre['checkpoint']['bytes'],state='server_receipt_not_downloaded'),
                       dict(kind='full_embeddings',path=embed['path'],sha256=embed['sha256'],bytes=embed['bytes'],state='server_receipt_not_downloaded'),
                       dict(kind='embedding_root_order',path=embed['root_ids_path'],sha256=embed['root_ids_sha256'],bytes='not_in_receipt',state='server_receipt_not_downloaded')])
        for name in ['A_results.json','formal_status.json','job_identity.json','pretrain.json','inference.json','classifier.json','embedding_receipt.json']:(OUT/name).write_bytes(z.read(name))
        with zipfile.ZipFile(MEASUREMENT) as measurement:
            first_measurement=[json.loads(line)['root_ids_sha256'] for line in measurement.read('pretrain_batches.jsonl').splitlines()]
        need(first_epoch_hashes==first_measurement,'Fresh seed0 batch order differs from the independent engineering run')
    report=dict(status='PASS_FINITE_A_ARTIFACT_AUDIT',date='2026-11-08 Asia/Shanghai',input_sha256=digest,
         manifest_files_checked=168,runtime_kit_files_checked=7,source_files_checked=57,epochs_checked=60,
         training_batches_checked=batch_count,training_root_exposures=169343*60,total_ego_positions=42884864*60,
         total_mask_positions=mask_count,inference_batches_checked=662,inference_rows=169343,
         classifier_runs_checked=10,classifier_epoch_records_checked=len(val_curves),
         validation_selection_rule='Last maximum validation epoch (>= tie rule), independently replayed over every epoch record',
         pretrain_seed=0,independent_pretrain_runs=1,probe_seeds=list(range(10)),
         test_mean_percent=mean*100,test_population_sd_percent=sd*100,
         whole_job_hours=status['wall_seconds_this_attempt']/3600,pretrain_phase_hours=pre['wall_seconds']/3600,
         inference_phase_minutes=infer['wall_seconds']/60,classifier_phase_minutes=classifier['wall_seconds']/60,
         pretrain_peak_allocated_GiB=max(row['peak_allocated_bytes'] for row in epochs)/1024**3,
         pretrain_peak_reserved_GiB=max(row['peak_reserved_bytes'] for row in epochs)/1024**3,
         no_allocator_or_cgroup_OOM=True,base_versions_unchanged=True,project_source_unchanged=True,
         measurement_checkpoint_reused=False,first_epoch_root_digest_sequence_matches_fresh_L1_seed0=True,
         limits=['Finite existing-artifact audit; no new training or test-based method/seed selection.',
                 'Full weights, embedding arrays and predictions remain on server. Their SHA256 and CPU readback receipts are checked; accuracy is recomputed from saved metrics, not independently re-inferred locally.',
                 'Root digests/counters and byte-matched runtime assertions support unique-root coverage; raw root identity arrays were not independently replayed locally.',
                 'One pretrained encoder with ten classifiers cannot establish pretraining-seed stability or B2 transfer.',
                 'Batch256/zero workers and self-generated ACL assets are resource adaptations, not exact original-batch reproduction.'])
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_csv(OUT/'epoch_summary_60.csv',epochs);write_csv(OUT/'probe_results_10.csv',probe_rows)
    write_csv(OUT/'validation_curves_10000.csv',val_curves);write_csv(OUT/'server_asset_receipts.csv',assets)
    (OUT/'核验结论.md').write_text(f'''# ogbn-arxiv方案A完整返回核验

2026-11-08：有限产物独立核验PASS。60轮预训练、39,720个训练批次、662个嵌入批次及10×1000轮分类器记录完整。168个清单文件、7个运行工具文件及57个实验源码逐字节一致，掩码数、学习率、EMA epoch参数、各轮统计及断点轮换链一致。10个分类器的最后最高验证轮次已从全部10,000条记录独立重选，吻合保存结果，测试值不进入选择轨迹。

最终测试准确率为**{mean*100:.3f}±{sd*100:.3f}%**（总体标准差ddof=0），即同一编码器的10个线性分类器；只有1次独立预训练（种子0）。没有运行B2大图迁移比较，不能由此声明B2迁移收益或预训练稳定性。

完整任务北京时间2026-11-07 22:57至2026-11-08 14:05结束，总计{status['wall_seconds_this_attempt']/3600:.3f}小时；预训练阶段{pre['wall_seconds']/3600:.3f}小时、嵌入阶段{infer['wall_seconds']/60:.2f}分钟、分类器阶段{classifier['wall_seconds']/60:.2f}分钟。共享负载下的15.14小时比13.1小时线性估计长，报告采用实测成本。256批/0加载子进程未见HAMI或cgroup OOM。

该实验是资源适配复现：训练/嵌入批次256、CPU加载0个子进程、自行生成ACL聚类；模型、损失、60/1000轮与10个探测种子冻结不变。此前单轮测量没有续入本实验。两个独立启动的首轮根节点散列序列一致，仅作种子及调度身份核对，不把两者当作两个正式预训练样本。

大模型和完整嵌入未下载，本地没有重跑推断。保存读回/严格CPU重载、子图与特征对齐、根节点唯一性由已逐字节核对的服务器入口执行；本地核对其回执、逐批计数和散列。测试均值/总体标准差由10个已保存测试值复算，不能称为本地重新测得准确率。

完整资产位于`/home/jovyan/work/GraphMAE2_artifacts/arxiv_A_TxrQqJzj`，本地已保留原始返回ZIP、复核脚本、逐轮/逐种子CSV及资产哈希回执。普通Git推送不自动包含这些外置大文件，独立备份仍须以实际保存清单为准。

用户选定的方案A实验部分现已完成，后续不默认追加实验；继续报告、README、数字/图表一致性及实际保存范围的收尾。
''',encoding='utf-8')
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
    (OUT/'SHA256SUMS.txt').write_text(''.join(sha(p.read_bytes())+'  '+p.relative_to(OUT).as_posix()+'\n' for p in files),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ['status','training_batches_checked','classifier_epoch_records_checked','test_mean_percent','test_population_sd_percent','whole_job_hours']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=INPUT);parser.add_argument('--outdir',type=Path,default=OUT)
    parser.add_argument('--kitdir',type=Path,default=KIT);parser.add_argument('--measurement-input',type=Path,default=MEASUREMENT)
    args=parser.parse_args();INPUT=args.input.resolve();OUT=args.outdir.resolve();KIT=args.kitdir.resolve();MEASUREMENT=args.measurement_input.resolve()
    main()
