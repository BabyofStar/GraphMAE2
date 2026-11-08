"""Verify and restore weight parts to a NEW directory without loading models."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile


class Reader:
    def __init__(self,paths):self.paths=iter(paths);self.stream=None
    def read(self,size=-1):
        if size<0:raise ValueError('Bounded reads required')
        pieces=[]
        while size:
            if self.stream is None:
                try:self.stream=Path(next(self.paths)).open('rb')
                except StopIteration:break
            block=self.stream.read(size)
            if not block:self.stream.close();self.stream=None;continue
            pieces.append(block);size-=len(block)
        return b''.join(pieces)
    def close(self):
        if self.stream is not None:self.stream.close()


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):value.update(block)
    return value.hexdigest()


def restore(manifest,parts,directory,dest):
    expected={'GraphMAE2_artifacts/'+row['archive_path']:row for row in manifest['files']}
    if len(expected)!=manifest['file_count']:raise ValueError('Duplicate or missing manifest files')
    if dest.exists():raise ValueError('Destination already exists; choose a new directory')
    paths=[]
    for part in parts['parts']:
        if Path(part['name']).name!=part['name']:raise ValueError('Unsafe part name')
        path=directory/part['name']
        if path.stat().st_size!=part['bytes'] or sha(path)!=part['sha256']:raise ValueError('Part content differs: '+part['name'])
        paths.append(path)
    dest.mkdir(parents=True,exist_ok=False);dest=dest.resolve();seen=set();reader=Reader(paths)
    try:
        with tarfile.open(fileobj=reader,mode='r|') as archive:
            for member in archive:
                relative=PurePosixPath(member.name)
                if relative.is_absolute() or '..' in relative.parts or '\\' in member.name or not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError('Unsafe or unexpected archive member')
                row=expected[member.name]
                if member.size!=row['bytes']:raise ValueError('Member size differs')
                path=dest.joinpath(*relative.parts)
                if not path.resolve().is_relative_to(dest):raise ValueError('Member escaped destination')
                path.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256()
                with archive.extractfile(member) as source,path.open('xb') as target:
                    for block in iter(lambda:source.read(4*1024**2),b''):target.write(block);digest.update(block)
                if digest.hexdigest()!=row['sha256']:raise ValueError('Restored file hash differs')
                seen.add(member.name)
    finally:reader.close()
    if seen!=set(expected):raise ValueError('Restoration is incomplete')
    return dict(status='PASS_RESTORED_FILE_HASHES',files=len(seen),destination=str(dest),model_loaded=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--parts',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--dest',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(restore(json.loads(args.manifest.read_text('utf-8')),json.loads(args.parts.read_text('utf-8')),args.directory,args.dest)))
