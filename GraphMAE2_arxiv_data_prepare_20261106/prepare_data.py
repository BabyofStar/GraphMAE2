"""Prepare official arxiv raw data and GraphMAE2's pre-generated 256-node clusters.

No package installs, project source changes, model loads, GPU tasks or training.
Downloads are bounded, fingerprinted and written only below an explicit asset root.
"""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import html
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

OGB_URL='https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip'
SHARE='https://cloud.tsinghua.edu.cn/d/64f859f389ca43eda472/'
TOKEN='64f859f389ca43eda472'
CLUSTER_NAME='ogbn-arxiv-lc-ego-graphs-256.pt'
HOSTS={'snap.stanford.edu','cloud.tsinghua.edu.cn'}
MAX_CLUSTER_BYTES=512*1024**2


def require(value,message):
    if not value:raise ValueError(message)


def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):digest.update(data)
    return digest.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


class Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        require(urllib.parse.urlsplit(newurl).hostname in HOSTS,'Unexpected download redirect host')
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def open_url(url,method='GET',timeout=35):
    parts=urllib.parse.urlsplit(url)
    require(parts.scheme=='https' and parts.hostname in HOSTS and parts.username is None and parts.password is None,'Unexpected source URL')
    request=urllib.request.Request(url,method=method,headers={'User-Agent':'GraphMAE2-arxiv-data-preparation'})
    return urllib.request.build_opener(Redirects()).open(request,timeout=timeout)


def download(url,destination,max_bytes,expected_size=None,kind='binary'):
    destination=Path(destination);receipt=destination.with_name(destination.name+'.receipt.json')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        require(receipt.is_file(),'Existing file lacks verified receipt; left untouched: '+str(destination))
        old=json.loads(receipt.read_text('utf-8'))
        require(old['source_url']==url and old['sha256']==sha_file(destination),'Existing asset fingerprint/source differs')
        return dict(old,reused=True)
    partial=destination.with_name(destination.name+'.partial')
    require(not partial.exists(),'Partial download retained; use a fresh asset directory or review it: '+str(partial))
    started=time.monotonic();total=0;digest=hashlib.sha256();last=started
    with open_url(url) as response:
        content_type=response.headers.get('Content-Type','').lower()
        require('text/html' not in content_type and 'application/json' not in content_type,'Source returned a page instead of a binary file')
        announced=response.headers.get('Content-Length')
        if announced:require(0<int(announced)<=max_bytes,'Download exceeds file budget')
        with partial.open('xb') as f:
            first=True
            while True:
                block=response.read(1024*1024)
                if not block:break
                if first:
                    require(block.startswith(b'PK\x03\x04') if kind=='zip' else block.startswith((b'PK\x03\x04',b'\x80\x02',b'\x80\x03',b'\x80\x04',b'\x80\x05')),
                            'Unexpected binary format; no pickle/model was loaded')
                    first=False
                total+=len(block);require(total<=max_bytes,'Download exceeded streaming budget')
                f.write(block);digest.update(block)
                now=time.monotonic()
                if now-last>=5:print(f'DOWNLOADING {destination.name} {total/1024**2:.1f} MiB',flush=True);last=now
        require(total>0 and (not announced or total==int(announced)),'Incomplete download')
        if expected_size is not None:require(total==expected_size,'Official arxiv size changed from the frozen HEAD observation; review before using')
        result=dict(source_url=url,final_url=response.url,bytes=total,sha256=digest.hexdigest(),
                    content_type=content_type,wall_seconds=time.monotonic()-started,path=str(destination),reused=False)
    os.rename(partial,destination)
    save(receipt,result)
    print(f'DOWNLOAD_COMPLETE {destination.name} bytes={total}',flush=True)
    return result


def unpack_ogb(archive,asset_root):
    target=asset_root/'dataset/ogbn_arxiv';marker=target/'raw_archive_identity.json'
    digest=sha_file(archive)
    if target.exists():
        require(marker.is_file() and json.loads(marker.read_text('utf-8'))['archive_sha256']==digest,'Existing dataset has no matching raw-archive identity; left untouched')
        return target
    with zipfile.ZipFile(archive) as z:
        members=[];seen=set();total=0
        for item in z.infolist():
            name=item.filename;p=PurePosixPath(name)
            require(name not in seen and not p.is_absolute() and '..' not in p.parts and '\\' not in name and not re.match(r'^[A-Za-z]:',name),'Unsafe dataset ZIP path')
            seen.add(name);require(not stat.S_ISLNK(item.external_attr>>16),'Dataset ZIP symlink')
            require(p.parts and p.parts[0]=='arxiv','Unexpected dataset root')
            total+=item.file_size;require(total<=1024**3 and item.file_size<=512*1024**2,'Dataset extraction budget')
            if not item.is_dir():members.append(item)
        require(z.testzip() is None,'Dataset ZIP CRC failure')
        target.mkdir(parents=True)
        for item in members:
            rel=PurePosixPath(item.filename).parts[1:];require(bool(rel),'Empty dataset leaf')
            dest=target.joinpath(*rel);require(dest.resolve().is_relative_to(target.resolve()),'Dataset path escaped target')
            dest.parent.mkdir(parents=True,exist_ok=True)
            with z.open(item) as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst,length=1024*1024)
    save(marker,dict(archive_sha256=digest,archive_bytes=Path(archive).stat().st_size,source_url=OGB_URL,
                     contents='Official raw graph/features/labels and time splits; no experiment scores.'))
    return target


def csv_values(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        for row in csv.reader(f):
            if row:yield row


def validate_raw(root,expected_nodes=169343,expected_edges=1166243,expected_features=128):
    required=['raw/edge.csv.gz','raw/node-feat.csv.gz','raw/node-label.csv.gz','raw/num-node-list.csv.gz',
              'raw/num-edge-list.csv.gz','split/time/train.csv.gz','split/time/valid.csv.gz','split/time/test.csv.gz']
    require(all((root/name).is_file() for name in required),'Official raw data or time-split file missing')
    nodes=sum(int(row[0]) for row in csv_values(root/'raw/num-node-list.csv.gz'))
    edges=sum(int(row[0]) for row in csv_values(root/'raw/num-edge-list.csv.gz'))
    require(nodes==expected_nodes and edges==expected_edges,'Raw OGB node/edge counts differ')
    feature_sample=next(csv_values(root/'raw/node-feat.csv.gz'))
    require(len(feature_sample)==expected_features and all(__import__('math').isfinite(float(x)) for x in feature_sample),'Feature dimension/sample finite check')
    counts={};seen=set()
    for split in ['train','valid','test']:
        count=0
        for row in csv_values(root/f'split/time/{split}.csv.gz'):
            require(len(row)==1,'Split schema')
            i=int(row[0]);require(0<=i<nodes and i not in seen,'Duplicate, overlapping or invalid official split ID')
            seen.add(i);count+=1
        counts[split]=count
    require(len(seen)==nodes,'Official split coverage differs')
    return dict(nodes=nodes,raw_directed_edges=edges,feature_dim_sample=expected_features,split='official time',split_counts=counts,
                raw_files=[dict(path=name,bytes=(root/name).stat().st_size,sha256=sha_file(root/name)) for name in required],
                feature_all_rows_parsed=False,graph_preprocessing_executed=False,experiment_test_scores_read=False,
                release_marker_present=(root/'RELEASE_v1.txt').is_file())


def file_link(path):
    require(path.startswith('/') and '..' not in PurePosixPath(path).parts and PurePosixPath(path).name==CLUSTER_NAME,'Unexpected cluster path')
    return SHARE+'files/?'+urllib.parse.urlencode({'p':path,'dl':'1'})


def binary_head(url):
    try:
        with open_url(url,'HEAD',timeout=15) as response:
            ct=response.headers.get('Content-Type','').lower();size=response.headers.get('Content-Length')
            return dict(url=url,status=response.status,content_type=ct,bytes=int(size) if size else None,
                        binary='text/html' not in ct and 'application/json' not in ct and response.status==200)
    except (OSError,ValueError,urllib.error.URLError) as e:return dict(url=url,error=type(e).__name__,binary=False)


def find_official_cluster(asset_root):
    candidates=['/'+CLUSTER_NAME];attempts=[]
    first=binary_head(file_link(candidates[0]));attempts.append(first)
    if first.get('binary'):return file_link(candidates[0]),attempts
    listing_urls=[SHARE, f'https://cloud.tsinghua.edu.cn/api/v2.1/share-links/{TOKEN}/dirents/?path=%2F']
    for index,url in enumerate(listing_urls):
        try:
            with open_url(url,timeout=20) as response:
                payload=response.read(512*1024+1);require(len(payload)<=512*1024,'Listing metadata budget')
            (asset_root/f'cluster_listing_{index}.txt').write_bytes(payload)
            text=html.unescape(payload.decode('utf-8',errors='replace'))
            for value in re.findall(r'''["']([^"'<>\r\n]{0,512}ogbn-arxiv-lc-ego-graphs-256\.pt)["']''',text):
                value=urllib.parse.unquote(value.replace('\\/','/'))
                if '?' in value:value=urllib.parse.parse_qs(urllib.parse.urlsplit(value).query).get('p',[value])[0]
                if not value.startswith('/'):value='/'+value
                if '..' not in PurePosixPath(value).parts and PurePosixPath(value).name==CLUSTER_NAME:candidates.append(value)
            try:
                obj=json.loads(text)
                def visit(value):
                    if isinstance(value,dict):
                        if value.get('name')==CLUSTER_NAME:candidates.append(value.get('path') or '/'+CLUSTER_NAME)
                        for child in value.values():visit(child)
                    elif isinstance(value,list):
                        for child in value:visit(child)
                visit(obj)
            except json.JSONDecodeError:pass
            attempts.append({'url':url,'metadata_bytes':len(payload),'status':'listing_read'})
        except (OSError,ValueError,urllib.error.URLError) as e:attempts.append({'url':url,'error':type(e).__name__})
    for path in dict.fromkeys(candidates[1:]):
        if not path.startswith('/'):path='/'+path
        result=binary_head(file_link(path));attempts.append(result)
        if result.get('binary'):return file_link(path),attempts
    return None,attempts


def memory_observation():
    result={}
    for name in ['memory.current','memory.max','memory.events','memory.pressure','memory.stat','cpu.max']:
        try:result[name]=Path('/sys/fs/cgroup',name).read_text('utf-8')
        except OSError:pass
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',required=True,type=Path)
    args=parser.parse_args();asset=args.asset_root.expanduser().resolve()
    require(asset.is_dir() and not asset.is_symlink(),'Use an existing fresh artifact directory')
    started=time.monotonic()
    report=dict(stage='arxiv data preparation',time_beijing=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
                asset_root=str(asset),training_started=False,dependencies_installed=False,project_source_modified=False,
                model_or_pickle_loaded=False,personal_disk_quota_established=False,memory_before=memory_observation())
    try:
        print('PREPARE_ASSET_ROOT='+str(asset),flush=True)
        report['ogb_archive']=download(OGB_URL,asset/'downloads/arxiv.zip',128*1024**2,83058288,'zip')
        dataset=unpack_ogb(asset/'downloads/arxiv.zip',asset)
        report['raw_data']=validate_raw(dataset)
        require(report['raw_data']['release_marker_present'],'Official version marker missing; preserved data, review before loading OGB')
        print('RAW_DATA_AND_OFFICIAL_SPLITS=PASS',flush=True)
        url,attempts=find_official_cluster(asset);report['cluster_source_attempts']=attempts
        if url:
            report['cluster_file']=download(url,asset/'lc_ego_graphs'/CLUSTER_NAME,MAX_CLUSTER_BYTES,kind='cluster')
            report['status']='FILES_PREPARED_CLUSTER_CONTENT_AND_L1_MEASUREMENT_PENDING'
        else:
            report['status']='RAW_DATA_PREPARED_CLUSTER_LOCATION_PENDING'
            report['next_action']='Retain prepared data and return preparation.json plus cluster_listing_*.txt; do not install LocalGraphClustering in the frozen environment.'
        report['next_stage_paths']=dict(data_dir=str(asset),ogb_dataset_root=str(asset/'dataset'),
            ego_graph_file_path=str(asset/'lc_ego_graphs'/CLUSTER_NAME),sampling_method='lc')
    except Exception as error:
        report.update(status='PREPARATION_STOPPED',error_type=type(error).__name__,error=str(error),
                      next_action='Preserve partial assets and return preparation.json; no automatic training or environment changes.')
    report['wall_seconds']=time.monotonic()-started;report['memory_after']=memory_observation()
    save(asset/'preparation.json',report)
    with zipfile.ZipFile(asset/'preparation_return.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(asset/'preparation.json','preparation.json')
        for p in sorted(asset.glob('cluster_listing_*.txt')):z.write(p,p.name)
    print('PREPARATION_STATUS='+report['status'],flush=True)
    print('RETURN_FILE='+str(asset/'preparation_return.zip'),flush=True)
    print('No training has been started. Prepared paths will be reused for the next engineering stage.',flush=True)
    return 1 if report['status']=='PREPARATION_STOPPED' else 0


if __name__=='__main__':raise SystemExit(main())
