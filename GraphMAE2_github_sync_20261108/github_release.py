"""Pinned CLI login bootstrap and resumable, verified GitHub Release upload."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from common import *


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,url):return None


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,url):
        parts=urllib.parse.urlsplit(url)
        need(parts.scheme=='https' and parts.hostname in ['github.com','api.github.com','uploads.github.com',
                 'release-assets.githubusercontent.com','objects.githubusercontent.com'],'Unexpected GitHub download redirect')
        result=super().redirect_request(request,fp,code,msg,headers,url)
        if result is not None and parts.hostname!=urllib.parse.urlsplit(request.full_url).hostname:
            result.remove_header('Authorization')
        return result


def request_json(method,path,token=None,payload=None,missing_ok=False):
    need(path.startswith('/') and not path.startswith('//'),'Invalid GitHub API path')
    allowed=path=='/user' or path.startswith('/repos/'+CONTRACT['repo']+'/') or path=='/repos/'+CONTRACT['repo'] or path.startswith('/repos/cli/cli/releases/')
    need(allowed,'API path escaped authorized repositories')
    headers={'Accept':'application/vnd.github+json','User-Agent':'GraphMAE2-authorized-backup','X-GitHub-Api-Version':'2022-11-28'}
    if token:headers['Authorization']='Bearer '+token
    data=None if payload is None else json.dumps(payload).encode('utf-8')
    if data is not None:headers['Content-Type']='application/json'
    req=urllib.request.Request('https://api.github.com'+path,headers=headers,data=data,method=method)
    try:
        with urllib.request.build_opener(NoRedirect()).open(req,timeout=45) as response:
            raw=response.read(8*1024**2+1);need(len(raw)<=8*1024**2,'API metadata exceeds its budget')
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        if missing_ok and error.code==404:return None
        raise RuntimeError(f'GitHub API {method} failed with HTTP {error.code}; credential value withheld') from None


def download(url,target=None,token=None):
    parsed=urllib.parse.urlsplit(url)
    need(parsed.scheme=='https' and parsed.hostname in ['github.com','api.github.com'],'Unexpected initial download host')
    headers={'User-Agent':'GraphMAE2-authorized-backup','Accept':'application/octet-stream'}
    if token:headers['Authorization']='Bearer '+token
    request=urllib.request.Request(url,headers=headers);digest=hashlib.sha256();size=0
    final=Path(target) if target is not None else None
    if final is not None:need(not final.exists(),'Download target already exists')
    temporary=final.with_name(final.name+'.incomplete') if final is not None else None
    stream=temporary.open('wb') if temporary is not None else None
    try:
        with urllib.request.build_opener(PublicRedirect()).open(request,timeout=120) as response:
            need('application/json' not in response.headers.get('Content-Type',''),'Download returned metadata instead of asset bytes')
            for block in iter(lambda:response.read(4*1024**2),b''):
                digest.update(block);size+=len(block)
                if stream:stream.write(block)
    finally:
        if stream:stream.close()
    if final is not None:temporary.replace(final)
    return dict(bytes=size,sha256=digest.hexdigest())


def install_cli(out):
    job(out);version=CONTRACT['gh_version'];arch={'x86_64':'amd64','amd64':'amd64','aarch64':'arm64','arm64':'arm64'}.get(platform.machine().lower())
    need(platform.system()=='Linux' and arch is not None,'Unsupported CLI host platform')
    tools=out.parent/'tools';tools.mkdir(exist_ok=True);directory=tools/f'gh_{version}_linux_{arch}'
    receipt_path=out/'gh_tool.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text('utf-8'));need(sha(Path(receipt['path']))==receipt['binary_sha256'],'Installed helper binary changed')
        return
    meta=request_json('GET','/repos/cli/cli/releases/tags/v'+version)
    need(meta['tag_name']=='v'+version,'Wrong pinned CLI release')
    assets={item['name']:item for item in meta['assets']};name=f'gh_{version}_linux_{arch}.tar.gz';checksum_name=f'gh_{version}_checksums.txt'
    temp=tools/f'gh_download_{out.name}';temp.mkdir(exist_ok=True)
    archive_path=temp/name;checksums=temp/checksum_name
    if not checksums.exists():download(assets[checksum_name]['browser_download_url'],checksums)
    lines=checksums.read_text('utf-8').splitlines();expected=next(line.split()[0] for line in lines if line.split() and line.split()[-1].lstrip('*')==name)
    need(re.fullmatch('[0-9a-f]{64}',expected) is not None,'Invalid official CLI checksum')
    if not archive_path.exists():
        print('Downloading pinned official GitHub CLI helper...',flush=True);download(assets[name]['browser_download_url'],archive_path)
    need(sha(archive_path)==expected,'Official CLI archive checksum mismatch')
    with tarfile.open(archive_path,'r:gz') as archive:
        member=next(m for m in archive.getmembers() if m.name==f'gh_{version}_linux_{arch}/bin/gh')
        need(member.isfile() and member.size<150*1024**2,'Unexpected CLI binary member')
        directory.mkdir(exist_ok=True);(directory/'bin').mkdir(exist_ok=True)
        binary=directory/'bin/gh'
        if binary.exists():
            with archive.extractfile(member) as source:
                digest=hashlib.sha256()
                for block in iter(lambda:source.read(1024**2),b''):digest.update(block)
            need(sha(binary)==digest.hexdigest(),'Cached CLI differs from its official archive')
        else:
            installing=binary.with_name('gh.installing')
            with archive.extractfile(member) as source,installing.open('wb') as target:
                for block in iter(lambda:source.read(1024**2),b''):target.write(block)
            installing.chmod(0o700);installing.replace(binary)
    binary.chmod(0o700)
    save(out/'gh_download_provenance.json',dict(repository='cli/cli',tag='v'+version,archive=name,archive_sha256=expected,
         source_url=assets[name]['browser_download_url'],checksum_url=assets[checksum_name]['browser_download_url']))
    binary=directory/'bin/gh';save(receipt_path,dict(path=str(binary),version=version,binary_sha256=sha(binary),experiment_environment_modified=False))


def token_from_cli(binary):
    result=subprocess.run([str(binary),'auth','token','--hostname','github.com'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    need(result.returncode==0,'GitHub browser login is required before Release upload')
    token=result.stdout.decode('utf-8').strip();need(bool(token),'GitHub login returned no usable credential')
    return token


class ProgressBody:
    def __init__(self,path,out,name):self.stream=Path(path).open('rb');self.sent=0;self.last=0.;self.out=out;self.name=name;self.started=time.monotonic()
    def read(self,size=-1):
        block=self.stream.read(size);self.sent+=len(block);clock=time.monotonic()
        if clock-self.last>10:
            save(self.out/'upload_progress.json',dict(asset=self.name,bytes_sent=self.sent,elapsed_seconds=clock-self.started,
                 scope='Bytes sent for current request; not committed remote bytes until SHA verification'))
            print(f'UPLOAD_BYTES asset={self.name} sent={self.sent}',flush=True);self.last=clock
        return block
    def close(self):self.stream.close()


def upload_file(url,path,token,out):
    parsed=urllib.parse.urlsplit(url)
    need(parsed.scheme=='https' and parsed.hostname=='uploads.github.com' and parsed.path.startswith('/repos/'+CONTRACT['repo']+'/releases/'),'Upload URL escaped the authorized Release')
    body=ProgressBody(path,out,Path(path).name)
    request=urllib.request.Request(url,data=body,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
        'Content-Type':'application/octet-stream','Content-Length':str(Path(path).stat().st_size),'User-Agent':'GraphMAE2-authorized-backup'},method='POST')
    try:
        with urllib.request.build_opener(NoRedirect()).open(request,timeout=120) as response:
            need(response.status==201,'Unexpected asset upload response');raw=response.read(2*1024**2+1)
            need(len(raw)<=2*1024**2,'Asset metadata response too large');return json.loads(raw)
    except urllib.error.HTTPError as error:raise RuntimeError(f'GitHub asset upload failed with HTTP {error.code}; completed assets retained') from None
    finally:body.close()


def list_assets(release_id,token):
    found=[]
    for page in range(1,12):
        items=request_json('GET',f'/repos/{CONTRACT["repo"]}/releases/{release_id}/assets?per_page=100&page={page}',token)
        found.extend(items)
        if len(items)<100:return {item['name']:item for item in found}
    raise RuntimeError('Unexpectedly many assets in this backup Release')


def verify_remote_asset(item,path,token):
    expected=sha(path);need(item['state']=='uploaded' and item['size']==Path(path).stat().st_size,'Remote asset state/size differs')
    digest=item.get('digest')
    if digest:
        need(digest=='sha256:'+expected,'GitHub-computed asset SHA256 differs');method='github_computed_sha256'
    else:
        actual=download(f'https://api.github.com/repos/{CONTRACT["repo"]}/releases/assets/{item["id"]}',token=token)
        need(actual['sha256']==expected and actual['bytes']==item['size'],'Downloaded remote asset differs');method='streamed_download_sha256'
    return dict(name=item['name'],id=item['id'],bytes=item['size'],sha256=expected,verification_method=method,browser_download_url=item['browser_download_url'])


def publish(out):
    identity=job(out);tool=json.loads((out/'gh_tool.json').read_text('utf-8'));need(sha(Path(tool['path']))==tool['binary_sha256'],'GitHub CLI helper changed')
    token=token_from_cli(Path(tool['path']))
    try:
        user=request_json('GET','/user',token);need(user.get('login','').lower()=='babyofstar','Log in to the intended BabyofStar account')
        repo=request_json('GET','/repos/'+CONTRACT['repo'],token);need(repo.get('permissions',{}).get('push') and not repo.get('private'),'Unexpected target repository permissions/visibility')
        source=json.loads((out/'source_synced.json').read_text('utf-8'));weights=json.loads((out/'weights_manifest.json').read_text('utf-8'))
        part_data=json.loads((out/'parts_manifest.json').read_text('utf-8'));directory=Path(part_data['directory'])
        need(weights['job_id']==identity['id'] and weights['file_count']==235 and weights['weight_bytes']==CONTRACT['weight_bytes'],'Weight scope differs')
        need(weights['local_archive_member_readback_verified'],'Local archive has not passed per-file readback')
        marker='GraphMAE2 backup job '+identity['id'];release_path='/repos/'+CONTRACT['repo']+'/releases'
        release=request_json('GET',release_path+'/tags/'+urllib.parse.quote(identity['tag'],safe=''),token,missing_ok=True)
        if release is None:
            notes=marker+f'\n\nIndependent backup of {weights["file_count"]} model files ({weights["weight_bytes"]} bytes), bound to source snapshot `{source["sha"]}`.\n\nWeights are in lossless tar parts with a per-file SHA256 manifest and restore tool. Raw datasets, clustering, full embeddings and environment binaries are excluded. One arxiv encoder has ten linear heads. Archive readback is checked before upload; remote asset SHA256 is verified before publication.'
            release=request_json('POST',release_path,token,dict(tag_name=identity['tag'],target_commitish=source['sha'],name='GraphMAE2 model backup 2026-11-08',body=notes,draft=True,prerelease=False,make_latest='false'))
        need(release['tag_name']==identity['tag'] and marker in (release.get('body') or ''),'Existing Release does not belong to this backup job')
        save(out/'release_created.json',dict(id=release['id'],tag=identity['tag'],url=release['html_url'],source_commit=source['sha'],job_id=identity['id']))
        upload_base=release['upload_url'].split('{',1)[0]
        files=[directory/part['name'] for part in part_data['parts']]+[out/name for name in ['weights_manifest.json','parts_manifest.json','SHA256SUMS.weights.txt','RESTORE.md','restore_weights.py']]
        need(len(files)<=1000 and all(path.stat().st_size<2*1024**3 for path in files),'Release file count or size exceeds supported bounds')
        assets=list_assets(release['id'],token);verified=[]
        for index,path in enumerate(files,1):
            if path.name.startswith('weights.tar.part'):
                expected=next(part for part in part_data['parts'] if part['name']==path.name)
                need(sha(path)==expected['sha256'],'Local part changed before upload')
            item=assets.get(path.name)
            if item and item.get('state')=='starter' and item.get('size')==0:
                request_json('DELETE',f'/repos/{CONTRACT["repo"]}/releases/assets/{item["id"]}',token);item=None
            if item is None:
                print(f'UPLOAD_ASSET={index}/{len(files)} {path.name} bytes={path.stat().st_size}',flush=True)
                item=upload_file(upload_base+'?name='+urllib.parse.quote(path.name,safe=''),path,token,out)
            record=verify_remote_asset(item,path,token);verified.append(record)
            save(out/'remote_assets_progress.json',dict(release_id=release['id'],verified=verified,total_assets=len(files)))
            print(f'ASSET_VERIFIED={index}/{len(files)} {path.name}',flush=True)
        if release['draft']:
            release=request_json('PATCH',release_path+'/'+str(release['id']),token,dict(draft=False,make_latest='false'))
        live=request_json('GET',release_path+'/'+str(release['id']),token);need(not live['draft'],'Release is not published')
        # Verify that the Release tag resolves to the precise synchronized source commit.
        ref=request_json('GET',f'/repos/{CONTRACT["repo"]}/git/ref/tags/'+urllib.parse.quote(identity['tag'],safe=''),token)['object']
        for _ in range(5):
            if ref['type']=='commit':break
            need(ref['type']=='tag','Unexpected Release tag object type')
            ref=request_json('GET',f'/repos/{CONTRACT["repo"]}/git/tags/{ref["sha"]}',token)['object']
        need(ref['type']=='commit' and ref['sha']==source['sha'],'Release tag points to another source snapshot')
        fresh=list_assets(release['id'],token)
        need(all(fresh[row['name']]['id']==row['id'] and fresh[row['name']]['state']=='uploaded' and fresh[row['name']]['size']==row['bytes'] for row in verified),'Published assets changed after verification')
        save(out/'release_verified.json',dict(status='PASS_INDEPENDENT_GITHUB_WEIGHT_BACKUP',independent_backup_complete=True,
             job_id=identity['id'],url=live['html_url'],release_id=release['id'],tag=identity['tag'],source_commit=source['sha'],
             weight_files=weights['file_count'],weight_bytes=weights['weight_bytes'],parts=len(part_data['parts']),assets=verified,
             local_archive_per_file_readback_verified=True,model_loaded=False,environment_binary_backup=False,
             full_embedding_backup=False,time_beijing=now()))
        print('RELEASE_BACKUP_VERIFIED='+live['html_url'],flush=True)
    finally:token=None


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['install-cli','publish']);parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();out=args.output_dir.resolve()
    install_cli(out) if args.action=='install-cli' else publish(out)
