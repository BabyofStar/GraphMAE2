"""Rebuild closeout tables from frozen evidence; uses Python standard library only.

Run from any directory: python /path/to/tools/rebuild_results.py
The script never opens server files, training code, checkpoints or dev test JSON.
"""
from pathlib import Path
from decimal import Decimal
import csv
import hashlib
import json
import math
import statistics as st
import sys

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / 'sources'
TABLES = ROOT / 'tables'
DATASETS = ('cora', 'citeseer', 'pubmed')
NAMES = dict(cora='Cora', citeseer='Citeseer', pubmed='PubMed')
LABELS = {'formal120': '已独立核验；本轮复算', 'v2_dev36': '已独立核验；本轮复算',
          'R9': '已独立核验；本轮复算', 'DS9': '已独立核验；本轮复算', 'C9': '已独立核验；本轮复算',
          'P12': '服务器独立审计回传PASS；本轮复算摘要', 'J12': '摘要复算；原始产物独立核验待完成'}


def read(rel):
    return json.loads((SOURCES / rel).read_text('utf-8-sig'))


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(name, rows):
    assert rows
    path = TABLES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        for row in rows:
            out.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def close(a, b, tol=1e-9):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) < tol, (a, b)


def decimal_mean(values):
    return float(sum(Decimal(str(v)) for v in values) / Decimal(len(values)))


def paired(values, baseline):
    deltas = [round(a-b, 10) for a, b in zip(values, baseline)]
    return dict(differences_pp=deltas, mean_delta_pp=decimal_mean(deltas),
                wins=sum(d > 1e-9 for d in deltas), ties=sum(abs(d) < 1e-9 for d in deltas),
                losses=sum(d < -1e-9 for d in deltas),
                leave_one_seed_out_mean_pp=[st.mean(deltas[:i]+deltas[i+1:]) for i in range(len(deltas))])


def t9_cdf_numerical(x):
    """Independent Simpson integration of Student-t density (df=9)."""
    n = 8192
    bound = abs(x)
    step = bound / n
    coeff = math.gamma(5) / (3 * math.sqrt(math.pi) * math.gamma(4.5))
    def density(z):
        return coeff * (1 + z*z/9)**-5
    total = density(0) + density(bound)
    total += 4 * sum(density(i*step) for i in range(1, n, 2))
    total += 2 * sum(density(i*step) for i in range(2, n, 2))
    area = total * step / 3
    return .5 + (area if x >= 0 else -area)


def holm(pvalues):
    adjusted, largest = [0.]*len(pvalues), 0.
    for rank, i in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        largest = max(largest, (len(pvalues)-rank)*pvalues[i])
        adjusted[i] = min(1., largest)
    return adjusted


def main():
    formal = read('formal120/analysis_results.json')
    v2 = read('v2_dev36/analysis_results.json')
    r9 = read('R9/analysis_results.json')
    dsc = read('DS_C9/analysis_results.json')
    p12 = read('P12/server_return.json')
    j12 = read('J12/analysis_results.json')
    source_manifest = read('source_manifest.json')
    for entry in source_manifest['files']:
        snapshot = ROOT / entry['snapshot']
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == entry['sha256'], snapshot
    for loc in ['formal120/audit.json', 'v2_dev36/audit.json', 'R9/audit.json', 'DS_C9/audit.json']:
        assert read(loc)['status'] == 'PASS'
    assert formal['primary_metric'] == 'val_selected_test_acc'
    assert formal['target_method'] == 'B2' and formal['report_sd_ddof'] == 0
    assert formal['seeds'] == list(range(10, 20))
    assert p12['status'] == 'SERVER_INDEPENDENT_AUDIT_PASS_USER_RETURN'
    assert 'PENDING' in j12['status']

    raw = read('selected_metric_records.json')['records']
    raw_metric, raw_tests = {}, {}
    for item in raw:
        record = item['record']
        if item['member'].endswith('/test_metrics.json'):
            assert item['batch'] == 'formal120'
            raw_tests[item['member'].rsplit('/', 1)[0]] = record
        else:
            assert record['status'] == 'complete'
            key = (item['batch'], record['dataset'], record['method'], record['seed'])
            assert key not in raw_metric
            raw_metric[key] = item
    assert len(raw_metric) == 183 and len(raw_tests) == 120

    formal_runs = []
    for r in formal['runs']:
        item = raw_metric[('formal120', r['dataset'], r['method'], r['seed'])]
        test = raw_tests[item['member'].rsplit('/', 1)[0]]
        assert test['primary'] == 'val_selected_test_acc' and test['accuracy_scale'] == 'fraction'
        close(test['val_selected_test_acc'], r['val_selected_test_acc'])
        close(item['record']['best_val_acc'], r['best_val_acc'])
        formal_runs.append(dict(batch='formal120', dataset=r['dataset'], method=r['method'], seed=r['seed'],
            metric='val_selected_test_acc', accuracy_percent=round(r['val_selected_test_acc']*100, 10),
            evidence_status=LABELS['formal120'], source_member=item['member'].rsplit('/',1)[0]+'/test_metrics.json'))
    assert len({(r['dataset'],r['method'],r['seed']) for r in formal_runs}) == 120
    formal_runs.sort(key=lambda r:(DATASETS.index(r['dataset']), r['method'], r['seed']))

    formal_groups = []
    for d in DATASETS:
        for m in ['B0','B1','B2','B3']:
            rows = [r for r in formal_runs if r['dataset']==d and r['method']==m]
            assert [r['seed'] for r in rows] == list(range(10,20))
            vals = [r['accuracy_percent'] for r in rows]
            saved = formal['groups'][d+':'+m]['val_selected_test_acc']
            close(decimal_mean(vals),saved['mean_percent']);close(st.pstdev(vals),saved['pop_sd_percent'])
            formal_groups.append(dict(batch='formal120',dataset=d,method=m,metric='val_selected_test_acc',n=10,
                seeds=list(range(10,20)),mean_percent=decimal_mean(vals),population_sd_percent=st.pstdev(vals),
                sd_ddof=0,evidence_status=LABELS['formal120']))

    # CI critical value and p-values recomputed independently from the old exact-polynomial implementation.
    lo, hi = 0., 8.
    for _ in range(52):
        mid=(lo+hi)/2
        if t9_cdf_numerical(mid)<.975:lo=mid
        else:hi=mid
    tcrit=(lo+hi)/2
    primary, paired_rows = [], []
    for d in DATASETS:
        values={m:[r['accuracy_percent'] for r in formal_runs if r['dataset']==d and r['method']==m] for m in ['B0','B2']}
        p = paired(values['B2'], values['B0'])
        saved=formal['primary_paired_comparisons'][d]
        se=st.stdev(p['differences_pp'])/math.sqrt(10)
        ci=[p['mean_delta_pp']-tcrit*se,p['mean_delta_pp']+tcrit*se]
        pv=2*(1-t9_cdf_numerical(abs(p['mean_delta_pp']/se)))
        close(p['mean_delta_pp'],saved['mean_delta_pp']);close(pv,saved['p_two_sided'],1e-10)
        for a,b in zip(ci,saved['ci95_unadjusted_pp']):close(a,b,1e-9)
        for a,b in zip(p['differences_pp'],saved['delta_pp']):close(a,b)
        primary.append(dict(dataset=d,comparison='B2-B0',n=10,mean_delta_pp=p['mean_delta_pp'],
            sample_sd_delta_pp=st.stdev(p['differences_pp']),ci95_unadjusted_low_pp=ci[0],ci95_unadjusted_high_pp=ci[1],
            p_two_sided=pv,p_holm_three_datasets=saved['p_holm_three_datasets'],
            wins=p['wins'],ties=p['ties'],losses=p['losses'],source='sources/formal120/analysis_results.json'))
        for i,seed in enumerate(range(10,20)):
            paired_rows.append(dict(dataset=d,seed=seed,B0_test_percent=values['B0'][i],B2_test_percent=values['B2'][i],delta_pp=p['differences_pp'][i]))
    for row,value in zip(primary,holm([r['p_two_sided'] for r in primary])):
        close(row['p_holm_three_datasets'],value,1e-10)

    groups, validation_runs = [], []
    specs=[('v2_dev36',v2['groups'], {'B0','B2','D','S'},'sources/v2_dev36/analysis_results.json'),
           ('R9',r9['groups'],{'R'},'sources/R9/analysis_results.json'),
           ('DS9',dsc['groups'],{'DS'},'sources/DS_C9/analysis_results.json'),
           ('C9',dsc['groups'],{'C'},'sources/DS_C9/analysis_results.json'),
           ('P12',p12['groups'],{'S','H','P','PS'},'sources/P12/server_return.json'),
           ('J12',j12['groups'],{'S','NA','NR','NC'},'sources/J12/analysis_results.json')]
    for batch, entries, methods, src in specs:
        for entry in entries:
            if entry['method'] not in methods:continue
            d=entry.get('dataset','cora');m=entry['method'];vals=entry['per_seed_val_percent']
            assert len(vals)==3
            close(decimal_mean(vals),entry['mean_val_percent']);close(st.pstdev(vals),entry['population_std_percent'])
            groups.append(dict(batch=batch,dataset=d,method=m,metric='best_val_acc',n=3,seeds=[0,1,2],
                mean_percent=decimal_mean(vals),population_sd_percent=st.pstdev(vals),sd_ddof=0,
                per_seed_percent=vals,evidence_status=LABELS[batch],source=src))
            for seed,value in enumerate(vals):
                key=(batch,d,m,seed)
                origin=src
                if key in raw_metric:
                    close(value,raw_metric[key]['record']['best_val_acc']*100)
                    origin=raw_metric[key]['member']
                validation_runs.append(dict(batch=batch,dataset=d,method=m,seed=seed,metric='best_val_acc',
                    accuracy_percent=value,evidence_status=LABELS[batch],source=src,source_member=origin))
    assert len(groups)==29 and len(validation_runs)==87
    assert len({(r['batch'],r['dataset'],r['method'],r['seed']) for r in validation_runs})==87
    # Shared historical anchors must stay identical across the derived analyses.
    base={(g['dataset'],g['method']):g for g in v2['groups']}
    for analysis in [r9,dsc]:
        for g in analysis['groups']:
            if (g['dataset'],g['method']) in base:
                for a,b in zip(g['per_seed_val_percent'],base[g['dataset'],g['method']]['per_seed_val_percent']):close(a,b)
    groupmap={(g['batch'],g['dataset'],g['method']):g for g in groups}
    comparison_specs=[('v2_dev36','v2_dev36','D','B2'),('v2_dev36','v2_dev36','S','B2'),
        ('v2_dev36','v2_dev36','S','B0'),('v2_dev36','v2_dev36','D','S'),
        ('v2_dev36','R9','S','R'),('R9','v2_dev36','R','B2'),('R9','v2_dev36','R','B0'),
        ('DS9','v2_dev36','DS','S'),('DS9','v2_dev36','DS','B2'),('DS9','v2_dev36','DS','D'),('DS9','v2_dev36','DS','B0'),
        ('C9','v2_dev36','C','S'),('C9','v2_dev36','C','B2'),('C9','v2_dev36','C','B0')]
    comparisons=[]
    for mb,rb,m,ref in comparison_specs:
        for d in DATASETS:
            a,b=groupmap[mb,d,m],groupmap[rb,d,ref]
            comparisons.append(dict(dataset=d,method=m,reference=ref,method_batch=mb,reference_batch=rb,n=3,
                metric='best_val_acc',**paired(a['per_seed_percent'],b['per_seed_percent']),
                evidence_status=LABELS[mb]+' / '+LABELS[rb],source=a['source']+'; '+b['source']))
    for batch, saved in [('P12',p12),('J12',j12)]:
        for comp in saved['comparisons']:
            a,b=groupmap[batch,'cora',comp['method']],groupmap[batch,'cora',comp['reference']]
            p=paired(a['per_seed_percent'],b['per_seed_percent'])
            close(p['mean_delta_pp'],comp['mean_pp'])
            for a0,b0 in zip(p['differences_pp'],comp['differences_pp']):close(a0,b0)
            comparisons.append(dict(dataset='cora',method=comp['method'],reference=comp['reference'],method_batch=batch,
                reference_batch=batch,n=3,metric='best_val_acc',**p,evidence_status=LABELS[batch],source=a['source']))

    mechanism=[]
    for d in DATASETS:
        for m in ['B0','S','R']:
            rows=[r for r in r9['diagnostics'] if r['dataset']==d and r['method']==m]
            assert len(rows)==3
            mechanism.append(dict(dataset=d,method=m,n=3,metric='true_density_layer_sampling_rate_temporal_std_pp',
                value_pp=st.mean(st.mean(r['true_density_layer_sampling_rate_temporal_std_pp']) for r in rows),
                aggregation='mean of 4 true density layers, then 3 seeds',source='sources/R9/analysis_results.json'))

    ledger=[dict(batch=b,full_pretrain_runs=n,scope=s,metric=met,evidence_status=status) for b,n,s,met,status in [
        ('historical60',60,'3 datasets x seeds 0..19','historical test summaries','交接README记录；本轮未逐种子复算'),
        ('formal120',120,'3 datasets x B0/B1/B2/B3 x seeds 10..19','val_selected_test_acc',LABELS['formal120']),
        ('v2_dev36',36,'3 datasets x B0/B2/D/S x seeds 0..2','best_val_acc',LABELS['v2_dev36']),
        ('R9',9,'3 datasets x R x seeds 0..2','best_val_acc',LABELS['R9']),
        ('DS9',9,'3 datasets x DS x seeds 0..2','best_val_acc',LABELS['DS9']),
        ('C9',9,'3 datasets x C x seeds 0..2','best_val_acc',LABELS['C9']),
        ('P12',12,'Cora x S/H/P/PS x seeds 0..2','best_val_acc',LABELS['P12']),
        ('J12',12,'Cora x S/NA/NR/NC x seeds 0..2','best_val_acc',LABELS['J12'])]]
    assert sum(r['full_pretrain_runs'] for r in ledger)==267
    write_csv('formal_test_runs_120.csv',formal_runs)
    write_csv('formal_test_summary.csv',formal_groups)
    write_csv('formal_B2_minus_B0_per_seed.csv',paired_rows)
    write_csv('formal_B2_minus_B0_statistics.csv',primary)
    write_csv('development_validation_runs_87.csv',validation_runs)
    write_csv('development_3graphs_summary.csv',[r for r in groups if r['batch'] not in ['P12','J12']])
    write_csv('development_P12_J12_summary.csv',[r for r in groups if r['batch'] in ['P12','J12']])
    write_csv('development_paired_comparisons.csv',comparisons)
    write_csv('mechanism_S_R.csv',mechanism)
    write_csv('run_ledger.csv',ledger)
    result=dict(date='2026-11-06 Asia/Shanghai',formal_groups=formal_groups,formal_primary=primary,
        formal_per_seed=paired_rows,development_groups=groups,development_comparisons=comparisons,
        mechanism=mechanism,run_ledger=ledger,
        formal_equal_dataset_mean_delta_pp=st.mean(r['mean_delta_pp'] for r in primary),
        note='Formal tests and exploratory validation are separate tables. P/J S controls remain batch-specific.')
    dump(TABLES/'unified_results.json',result)
    checks=dict(status='PASS',scope='Closeout transcription, raw metric crosschecks and numerical statistics; not a new full artifact audit.',
        source_snapshot_hash_checks=len(source_manifest['files']),archive_sha256_checks=2,raw_full_run_metric_crosschecks=183,
        formal_test_record_crosschecks=120,formal_rows=120,development_rows=87,ledger_full_runs=267,
        formal_independent_numeric_t9_check=True,formal_independent_holm_check=True,summary_sd_ddof=0,
        confidence_interval_delta_sd_ddof=1,development_test_metrics_opened=False,model_weights_opened=False,
        P12_status=LABELS['P12'],J12_status=LABELS['J12'],latest_github_sync_verified=False,
        limitation='Historical60 only has summary transcription here. J12 raw audit and latest remote commit inventory remain pending.')
    dump(ROOT/'validation_checks.json',checks)
    print(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':
    main()
