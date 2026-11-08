"""Small on-server archive roundtrip before any remote mutation."""
import argparse
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
from common import job,need,save
from weights import PartWriter,verify_archive
from restore_weights import restore


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True,type=Path);args=parser.parse_args()
    out=args.output_dir.resolve();job(out)
    temporary=tempfile.TemporaryDirectory(dir=out,prefix='archive_self_check_')
    root=Path(temporary.name)
    try:
        directory=root/'parts';directory.mkdir();writer=PartWriter(directory,257)
        payload=b'GraphMAE2 lossless backup check\n'*101
        files=[dict(archive_path='checkpoints/self_check.bin',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())]
        with tarfile.open(fileobj=writer,mode='w|') as archive:
            member=tarfile.TarInfo('GraphMAE2_artifacts/'+files[0]['archive_path']);member.size=len(payload);archive.addfile(member,io.BytesIO(payload))
        writer.close();verify_archive(directory,writer.parts,files)
        restore(dict(file_count=1,files=files),dict(parts=writer.parts),directory,root/'restored')
        need((root/'restored/GraphMAE2_artifacts/checkpoints/self_check.bin').read_bytes()==payload,'Roundtrip payload differs')
        save(out/'runtime_self_check.json',dict(status='PASS',scope='Small split archive roundtrip and restored file SHA256',experiment_files_changed=False))
        print('SERVER_ARCHIVE_ROUNDTRIP=PASS',flush=True)
    finally:
        need(root.resolve().is_relative_to(out),'Self-check cleanup escaped its staging directory');temporary.cleanup()


if __name__=='__main__':main()
