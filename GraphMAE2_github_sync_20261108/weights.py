"""Lossless streaming tar parts and per-file archive readback; no model imports."""
import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import tarfile
from common import *


class PartWriter:
    def __init__(self,directory,limit):
        self.directory=Path(directory);self.limit=limit;self.parts=[];self.stream=None;self.digest=None;self.size=0
    def writable(self):return True
    def _finish(self):
        if self.stream is not None:
            self.stream.close();self.parts.append(dict(name=self.path.name,bytes=self.size,sha256=self.digest.hexdigest()))
            self.stream=None
    def write(self,data):
        total=len(data);data=memoryview(data)
        while data:
            if self.stream is None:
                self.path=self.directory/f'weights.tar.part{len(self.parts)+1:04d}'
                self.stream=self.path.open('xb');self.digest=hashlib.sha256();self.size=0
            take=min(len(data),self.limit-self.size);piece=data[:take]
            self.stream.write(piece);self.digest.update(piece);self.size+=take;data=data[take:]
            if self.size==self.limit:self._finish()
        return total
    def flush(self):
        if self.stream is not None:self.stream.flush()
    def close(self):self._finish()


class PartReader:
    def __init__(self,paths):self.paths=iter(paths);self.stream=None
    def read(self,size=-1):
        need(size>=0,'Bounded reads required');pieces=[];remaining=size
        while remaining:
            if self.stream is None:
                try:self.stream=Path(next(self.paths)).open('rb')
                except StopIteration:break
            block=self.stream.read(remaining)
            if not block:self.stream.close();self.stream=None;continue
            pieces.append(block);remaining-=len(block)
        return b''.join(pieces)
    def close(self):
        if self.stream is not None:self.stream.close()


class HashedReader:
    def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256();self.size=0
    def read(self,size=-1):
        block=self.stream.read(size);self.digest.update(block);self.size+=len(block);return block


def safe_name(name):
    path=PurePosixPath(name)
    return not path.is_absolute() and '..' not in path.parts and '\\' not in name and path.parts and path.parts[0]=='GraphMAE2_artifacts'


def verify_archive(directory,parts,files):
    expected={'GraphMAE2_artifacts/'+row['archive_path']:row for row in files};seen=set()
    for part in parts:need(sha(Path(directory)/part['name'])==part['sha256'] and (Path(directory)/part['name']).stat().st_size==part['bytes'],'Part identity differs')
    reader=PartReader([Path(directory)/part['name'] for part in parts])
    try:
        with tarfile.open(fileobj=reader,mode='r|') as archive:
            for member in archive:
                need(safe_name(member.name) and member.isfile() and member.name in expected and member.name not in seen,'Unexpected archive member')
                expected_file=expected[member.name];need(member.size==expected_file['bytes'],'Archive member size differs')
                digest=hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
                need(digest.hexdigest()==expected_file['sha256'],'Archive member SHA differs');seen.add(member.name)
    finally:reader.close()
    need(seen==set(expected),'Archive member coverage differs')


def support_files(out,parts):
    (out/'SHA256SUMS.weights.txt').write_text(''.join(p['sha256']+'  '+p['name']+'\n' for p in parts),encoding='utf-8')
    for source,target in [('restore_weights.py','restore_weights.py'),('RESTORE.template.md','RESTORE.md')]:
        (out/target).write_bytes((PACKAGE/source).read_bytes())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    out=args.output_dir.resolve();identity=job(out);manifest_path=out/'weights_manifest.json';parts_path=out/'parts_manifest.json'
    inventory=json.loads((PACKAGE/'weight_inventory.json').read_text('utf-8'));files=inventory['files']
    need(len(files)==235 and sum(f['bytes'] for f in files)==CONTRACT['weight_bytes'],'Frozen weight scope differs')
    if manifest_path.exists() and parts_path.exists():
        manifest=json.loads(manifest_path.read_text('utf-8'));parts=json.loads(parts_path.read_text('utf-8'))
        need(manifest['files']==files and manifest['job_id']==identity['id'],'Existing archive belongs to a different job')
        verify_archive(Path(parts['directory']),parts['parts'],files)
        support_files(out,parts['parts'])
        print('ARCHIVE_REUSED_AND_VERIFIED',flush=True);return
    index=1
    while (out/f'weight_parts_a{index:04d}').exists():index+=1
    directory=out/f'weight_parts_a{index:04d}';directory.mkdir()
    writer=PartWriter(directory,CONTRACT['part_bytes'])
    try:
        with tarfile.open(fileobj=writer,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for index,row in enumerate(files,1):
                path=Path(row['path']);need(path.resolve().is_relative_to(ARTIFACTS.resolve()) and not path.is_symlink(),'Weight source escaped approved root')
                before=path.stat();need(before.st_size==row['bytes'],'Weight size changed since preflight')
                arcname='GraphMAE2_artifacts/'+row['archive_path'];need(safe_name(arcname),'Unsafe weight archive path')
                member=tarfile.TarInfo(arcname);member.size=row['bytes'];member.mode=0o600;member.mtime=int(before.st_mtime)
                with path.open('rb') as stream:
                    hashed=HashedReader(stream);archive.addfile(member,hashed)
                    need(hashed.digest.hexdigest()==row['sha256'] and hashed.size==row['bytes'],'Weight changed since preflight: '+row['archive_path'])
                after=path.stat();need((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Weight changed during archiving')
                print(f'WEIGHT_PACKED={index}/{len(files)}',flush=True)
    finally:writer.close()
    verify_archive(directory,writer.parts,files)
    save(manifest_path,dict(format_version=1,job_id=identity['id'],repository=CONTRACT['repo'],release_tag=identity['tag'],
         file_count=len(files),weight_bytes=sum(f['bytes'] for f in files),files=files,exclusions=inventory['excluded'],
         local_archive_member_readback_verified=True,model_loaded=False))
    save(parts_path,dict(format_version=1,directory=str(directory),part_order='Concatenate named parts in listed order',parts=writer.parts))
    support_files(out,writer.parts)
    print(f'ARCHIVE_VERIFIED files={len(files)} parts={len(writer.parts)} bytes={sum(p["bytes"] for p in writer.parts)}',flush=True)


if __name__=='__main__':main()
