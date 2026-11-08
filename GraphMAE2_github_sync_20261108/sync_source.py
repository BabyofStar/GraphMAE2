"""Authorized source push in an isolated worktree; preserve the experiment worktree."""
import argparse
import json
import os
from pathlib import Path
import shutil
import zipfile
from common import *


def remote_head(tree):
    raw=text(git(tree,'ls-remote','--heads',CONTRACT['remote'],'refs/heads/main')).split()
    need(len(raw)==2 and len(raw[0])==40 and raw[1]=='refs/heads/main','Cannot identify remote main')
    return raw[0]


def commit_if_needed(tree,message):
    if git(tree,'diff','--cached','--quiet',check=False).returncode==0:return text(git(tree,'rev-parse','HEAD'))
    git(tree,'var','GIT_AUTHOR_IDENT');git(tree,'commit','-m',message)
    return text(git(tree,'rev-parse','HEAD'))


def reconcile_and_push(tree,out,candidate_readme):
    for attempt in range(3):
        git(tree,'fetch',CONTRACT['remote'],'refs/heads/main')
        remote=text(git(tree,'rev-parse','FETCH_HEAD'));local=text(git(tree,'rev-parse','HEAD'))
        need(git(tree,'merge-base',local,remote,check=False).returncode==0,'Unrelated repository histories; no overwrite attempted')
        if git(tree,'merge-base','--is-ancestor',remote,local,check=False).returncode!=0:
            merged=git(tree,'merge','--no-edit',remote,check=False)
            if merged.returncode:
                conflicts=text(git(tree,'diff','--name-only','--diff-filter=U')).splitlines()
                if conflicts!=['README.md']:
                    save(out/'merge_review_required.json',dict(remote=remote,conflicts=conflicts))
                    raise RuntimeError('Non-README merge conflict; preserved for targeted review')
                history=tree/'docs/backup-history';history.mkdir(parents=True,exist_ok=True)
                for label,commit in [('local',local),('remote',remote)]:
                    raw=git(tree,'show',commit+':README.md',check=False)
                    if raw.returncode==0:
                        path=history/f'README.{label}.{commit}.md'
                        if not path.exists():path.write_bytes(raw.stdout)
                (tree/'README.md').write_bytes(candidate_readme)
                git(tree,'--literal-pathspecs','add','--force','--','README.md','docs/backup-history')
                git(tree,'commit','--no-edit')
        # The audited experiment implementation must not be silently replaced by remote changes.
        expected=json.loads((PACKAGE/'expected_experiment_source.json').read_text('utf-8'))
        for name,digest in expected.items():need(sha(tree/name)==digest,'Remote merge changes audited experiment source: '+name)
        for record in json.loads((PACKAGE/'source_inventory.json').read_text('utf-8'))['files']:
            if record['path']=='README.md':continue
            need(sha(tree/record['path'])==record['sha256'],'Remote merge changes the selected source snapshot: '+record['path'])
        head=text(git(tree,'rev-parse','HEAD'))
        # Inspect all newly reachable blobs before sending, including pre-existing unpublished history.
        objects=git(tree,'rev-list','--objects',head,'^'+remote).stdout.splitlines()
        ids=b'\n'.join(line.split(b' ',1)[0] for line in objects)+b'\n'
        if objects:
            sizes=git(tree,'cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)',input=ids).stdout.splitlines()
            need(all(not (p[1]==b'blob' and int(p[2])>100*1024**2) for p in (line.split() for line in sizes)),'Unpublished history has an oversized Git blob')
        save(out/'source_candidate_commit.json',dict(sha=head,remote_before=remote,worktree=str(tree),time_beijing=now()))
        pushed=git(tree,'push',CONTRACT['remote'],'HEAD:refs/heads/main',check=False)
        observed=remote_head(tree)
        if observed==head:
            need(git(tree,'diff','--quiet',check=False).returncode==0 and git(tree,'diff','--cached','--quiet',check=False).returncode==0,'Synchronized worktree changed during push')
            return head
        if pushed.returncode==0:raise RuntimeError('Remote main moved after push; snapshot is retained, final verification pending')
    raise RuntimeError('Remote main changed repeatedly or push failed; no force push attempted')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args();out=args.output_dir.resolve();identity=job(out);tree=out/'worktree'
    need(text(git(PROJECT,'rev-parse','HEAD'))==CONTRACT['original_HEAD'] and text(git(PROJECT,'branch','--show-current'))==CONTRACT['original_branch'],'Original experiment branch changed')
    need(git(PROJECT,'diff','--cached','--quiet',check=False).returncode==0 and git(PROJECT,'diff','--quiet',check=False).returncode==0,'Original tracked files or staging changed since preflight')
    for args2 in [('remote','get-url','--all','course_backup'),('remote','get-url','--push','--all','course_backup')]:
        need(text(git(PROJECT,*args2))==CONTRACT['remote_url'],'Backup remote URL changed')
    source=json.loads((PACKAGE/'source_inventory.json').read_text('utf-8'))
    if args.finalize:
        receipt=json.loads((out/'release_verified.json').read_text('utf-8'))
        need(receipt['independent_backup_complete'],'Remote weight backup not yet verified')
        previous=json.loads((out/'source_synced.json').read_text('utf-8'))
        candidate=(PACKAGE/'README.sync_candidate.md').read_text('utf-8')
        pending='外置权重计划发布在本仓库独立Release中，附文件清单、SHA256及恢复说明；普通推送不自动包含外置权重。当前同步/上传尚在准备，完成状态以最终远端完整提交哈希与Release逐项校验回执为准，不仅依据README。'
        need(candidate.count(pending)==1,'README backup status replacement is ambiguous')
        candidate=candidate.replace(pending,
            f"外置权重独立备份已完成远端内容校验：{receipt['weight_files']}个文件，原始总大小{receipt['weight_bytes']}字节。[GitHub Release]({receipt['url']})保存权重分卷、清单与恢复工具；每个分卷匹配GitHub资产SHA256，缺少服务端摘要时流式下载重算。本地已逐文件读回归档，未额外加载模型。备份不含原始数据、完整嵌入和环境二进制。源码快照提交为`{previous['sha']}`，随后本提交仅补充备份验收记录。")
        (tree/'README.md').write_text(candidate,encoding='utf-8')
        destination=tree/'backup'/identity['tag'];destination.mkdir(parents=True,exist_ok=True)
        for name in ['release_verified.json','weights_manifest.json','parts_manifest.json','RESTORE.md']:
            shutil.copyfile(out/name,destination/name)
        notice=tree/'docs/closeout_20261108/BACKUP_STATUS.md'
        notice.write_text(f"# 后续备份状态\n\n报告内容快照生成于备份之前。最新权重备份验收见[Release]({receipt['url']})及仓库backup/{identity['tag']}/。课程排版仍暂缓。\n",encoding='utf-8')
        git(tree,'--literal-pathspecs','add','--force','--','README.md','backup','docs/closeout_20261108/BACKUP_STATUS.md')
        commit_if_needed(tree,CONTRACT['metadata_commit_message'])
        head=reconcile_and_push(tree,out,candidate.encode('utf-8'))
        save(out/'final_sync_verified.json',dict(status='PASS',sha=head,remote_main=head,source_snapshot_commit=previous['sha'],release=receipt['url'],
             original_experiment_branch_preserved=True,time_beijing=now()))
        print('FINAL_MAIN_VERIFIED='+head,flush=True);return
    if (out/'source_synced.json').exists():
        previous=json.loads((out/'source_synced.json').read_text('utf-8'))
        need(remote_head(tree)==previous['sha'],'Remote main changed after the source snapshot; review before continuing')
        print('SOURCE_SYNC_REUSED='+previous['sha'],flush=True);return
    if not tree.exists():git(PROJECT,'worktree','add','--detach',str(tree),CONTRACT['original_HEAD'])
    require_no_conflicts=git(tree,'diff','--name-only','--diff-filter=U');need(not text(require_no_conflicts),'Existing merge conflict requires review')
    for index,record in enumerate(source['files'],1):
        path=PROJECT/record['path'];need(path.is_file() and not path.is_symlink() and sha(path)==record['sha256'],'Source changed since audited preflight: '+record['path'])
        target=tree/record['path'];target.parent.mkdir(parents=True,exist_ok=True)
        need(not target.is_symlink(),'Candidate target is a symlink')
        shutil.copyfile(path,target);target.chmod(0o755 if record['mode']=='100755' else 0o644)
        if index%500==0:print(f'SOURCE_COPIED={index}/{len(source["files"])}',flush=True)
    history=tree/'docs/backup-history';history.mkdir(parents=True,exist_ok=True)
    (history/f'README.server.{CONTRACT["original_HEAD"]}.md').write_bytes((PROJECT/'README.md').read_bytes())
    with zipfile.ZipFile(PACKAGE/'closeout_content_20261108.zip') as archive:
        for name in archive.namelist():
            relative=Path(name);need(not relative.is_absolute() and '..' not in relative.parts,'Unsafe content archive path')
            target=tree/'docs/closeout_20261108'/Path(*relative.parts[1:]);need(target.resolve().is_relative_to(tree.resolve()),'Content target escaped worktree')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(name))
    candidate=(PACKAGE/'README.sync_candidate.md').read_bytes();(tree/'README.md').write_bytes(candidate)
    source_paths=[r['path'] for r in source['files']]+['docs/closeout_20261108','docs/backup-history']
    git(tree,'--literal-pathspecs','add','--force','--pathspec-from-file=-','--pathspec-file-nul',input=b'\0'.join(p.encode('utf-8') for p in source_paths)+b'\0')
    commit_if_needed(tree,CONTRACT['source_commit_message'])
    head=reconcile_and_push(tree,out,candidate)
    save(out/'source_synced.json',dict(status='PASS',sha=head,remote_main=head,original_head=CONTRACT['original_HEAD'],
         original_branch=CONTRACT['original_branch'],synchronization_worktree=str(tree),selected_files=len(source['files']),
         content_payload_sha256=sha(PACKAGE/'closeout_content_20261108.zip'),original_experiment_branch_preserved=True,time_beijing=now()))
    print('SOURCE_MAIN_VERIFIED='+head,flush=True)


if __name__=='__main__':main()
