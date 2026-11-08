"""Check report/table/source consistency without executing any training code."""
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent.parent
def load(name):return json.loads((ROOT/name).read_text('utf-8-sig'))
def rows(name):
    with (ROOT/'tables'/name).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def main():
    u=load('tables/unified_results.json');audit=load('sources/arxiv_A/audit.json')
    report=(ROOT/'课程报告.md').read_text('utf-8');results=(ROOT/'02_统一结果总表.md').read_text('utf-8');readme=(ROOT/'README.md').read_text('utf-8')
    expected={'formal_test_runs_120.csv':120,'formal_test_summary.csv':12,'formal_B2_minus_B0_per_seed.csv':30,
              'formal_B2_minus_B0_statistics.csv':3,'development_validation_runs_87.csv':87,'development_3graphs_summary.csv':21,
              'development_P12_J12_summary.csv':8,'development_paired_comparisons.csv':52,'mechanism_S_R.csv':9,'run_ledger.csv':8,
              'run_ledger_all.csv':9,'arxiv_A_probe_results_10.csv':10,'arxiv_A_epochs_60.csv':60,'arxiv_A_validation_curves_10000.csv':10000,
              'arxiv_A_server_asset_receipts.csv':15}
    for name,count in expected.items():assert len(rows(name))==count,(name,count)
    assert sum(int(row['full_pretrain_runs']) for row in rows('run_ledger.csv'))==267
    assert sum(int(row['full_pretrain_runs']) for row in rows('run_ledger_all.csv'))==268
    assert {row['metric'] for row in rows('formal_test_runs_120.csv')}=={'val_selected_test_acc'}
    assert {row['metric'] for row in rows('development_validation_runs_87.csv')}=={'best_val_acc'}
    for group in u['formal_groups']:
        value=f"{group['mean_percent']:.2f}±{group['population_sd_percent']:.2f}"
        assert value in report and value in results,value
    for group in u['development_groups']:
        value=f"{group['mean_percent']:.3f}±{group['population_sd_percent']:.3f}"
        assert value in report and value in results,value
        if group['batch']=='J12':assert '核验PASS' in group['evidence_status']
        if group['batch']=='P12':assert '已本地核对' in group['evidence_status']
    for p in u['formal_primary']:assert f"{p['p_holm_three_datasets']:.4f}" in report
    assert '只有Cora通过' in report and '0.0466' in report and '0.0932' in report
    value=f"{audit['test_mean_percent']:.3f}±{audit['test_population_sd_percent']:.3f}%"
    assert value in report and value in results and value in readme
    for result in load('sources/arxiv_A/A_results.json')['results']:
        assert f"{result['val_selected_test_acc']*100:.4f}" in report
    assert audit['training_batches_checked']==39720 and audit['classifier_epoch_records_checked']==10000
    for stale in ['原始产物独立核验待完成','完整审计报告及原始产物尚未在本地重新读取','目前大图L0资源与数据预检']:
        assert stale not in report and stale not in readme,stale
    links=0
    for doc in list(ROOT.glob('*.md'))+[ROOT/'tools/复算说明.md']:
        text=doc.read_text('utf-8');assert '\ufffd' not in text and '{{' not in text,doc
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if link.startswith(('https://','http://','#')):continue
            target=(doc.parent/link.split('#',1)[0]).resolve()
            assert target.is_relative_to(ROOT.resolve()) and target.exists(),(doc.name,link)
            links+=1
    figures=[]
    for stem in ['formal_B2_minus_B0','mechanism_S_R','arxiv_A_training_and_probe']:
        ET.fromstring((ROOT/'figures'/f'{stem}.svg').read_text('utf-8'))
        raw=(ROOT/'figures'/f'{stem}.png').read_bytes();assert raw[:8]==b'\x89PNG\r\n\x1a\n'
        width,height=struct.unpack('>II',raw[16:24]);assert width>=2400 and height>=1200
        figures.append(dict(stem=stem,width=width,height=height))
    html=(ROOT/'课程报告.html').read_text('utf-8')
    assert html.count('src="data:image/png;base64,')==3 and '<script' not in html
    assert not re.search(r'<(?:img|link|iframe)[^>]+(?:src|href)="https?://',html)
    source_manifest=load('sources/source_manifest.json')
    for record in source_manifest['files']:assert hashlib.sha256((ROOT/record['snapshot']).read_bytes()).hexdigest()==record['sha256']
    assert hashlib.sha256((ROOT/'sources/arxiv_A/A_return_a0001.zip').read_bytes()).hexdigest()==audit['input_sha256']
    checks=dict(status='PASS',date='2026-11-08 Asia/Shanghai',csv_rows=expected,relative_links_checked=links,
          historical_source_hashes_unchanged=len(source_manifest['files']),figure_assets=figures,
          formal_development_arxiv_populations_separate=True,report_cells_match_tables=True,self_contained_HTML_images=3,
          audit_scope='Numeric/document consistency plus finite A artifact audit; no local model inference',
          unconfirmed=['Latest remote branch sync','Independent copies of complete external weights/data/environment','Course-specific final format/submission'])
    (ROOT/'document_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':main()
