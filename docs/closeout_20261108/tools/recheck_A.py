"""Portable finite replay of A's saved records, without loading model binaries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parent.parent


def main():
    scratch=ROOT/'tools/.verification_work';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as folder:
        temp=Path(folder).resolve();assert temp.is_relative_to(scratch.resolve())
        with zipfile.ZipFile(ROOT/'sources/formal_A_runtime_20261107.zip') as archive:
            assert archive.testzip() is None
            assert len(archive.namelist())==15
            for name in archive.namelist():
                relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
                target=temp/relative;assert target.resolve().is_relative_to(temp)
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(name))
        spec=importlib.util.spec_from_file_location('finite_A_auditor',ROOT/'tools/audit_A_return.py')
        auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
        auditor.INPUT=ROOT/'sources/arxiv_A/A_return_a0001.zip'
        auditor.MEASUREMENT=ROOT/'sources/arxiv_L1/measurement_return.zip'
        auditor.OUT=temp/'replayed_audit';auditor.KIT=temp/'GraphMAE2_arxiv_A_full_20261107'
        auditor.main()
        expected=json.loads((ROOT/'sources/arxiv_A/audit.json').read_text('utf-8'))
        actual=json.loads((auditor.OUT/'audit.json').read_text('utf-8'))
        assert actual==expected,'Recomputed finite A audit differs from the saved report'
    print('PASS: all A records replayed; no weights loaded, no GPU training performed.')


if __name__=='__main__':main()
