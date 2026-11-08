"""Install a pinned official helper for the existing backup job, without networking."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import tarfile

PACKAGE=Path(__file__).resolve().parent
ORIGINAL=Path('/home/jovyan/work/GraphMAE2/GraphMAE2_github_sync_20261108')
STAGING=Path('/home/jovyan/work/GraphMAE2_backup_staging')
EXPECTED_JOB='sync_u0Vtv5j5'
VERSION='2.103.3'
ARCHIVE='gh_2.103.3_linux_amd64.tar.gz'
ARCHIVE_SHA='4c806978ffce1cec70853302a56d70ac6da4f19bae7adca4e1a8ab00633dd91d'
CHECKSUMS='gh_2.103.3_checksums.txt'
CHECKSUMS_SHA='f756e7859d90dbb5e85aae5194206c37da220bec9bac23d3d659078a8e766fdc'


def require(ok,message):
    if not ok:raise RuntimeError(message)


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def install(package,out,original,identity):
    require(platform.system()=='Linux' and platform.machine().lower() in ['x86_64','amd64'],'This offline helper is for Linux x86_64')
    require(identity['id']==EXPECTED_JOB and out.name==EXPECTED_JOB,'Refusing to switch to a different backup job')
    require(digest(package/ARCHIVE)==ARCHIVE_SHA and digest(package/CHECKSUMS)==CHECKSUMS_SHA,'Offline official files changed')
    lines=(package/CHECKSUMS).read_text('utf-8').splitlines()
    checks=[line.split()[0] for line in lines if line.split() and line.split()[-1].lstrip('*')==ARCHIVE]
    require(checks==[ARCHIVE_SHA],'Official checksum document disagrees')
    provenance=json.loads((package/'official_provenance.json').read_text('utf-8'))
    require(provenance['version']==VERSION and provenance['archive_sha256']==ARCHIVE_SHA,'Offline provenance differs')
    require(identity['kit_identity_sha256']==digest(original/'kit_identity.json'),'Original execution package identity differs')
    require(identity['kit_identity_sha256']==provenance['original_package_identity_sha256'],'Unexpected original execution package version')
    require((out/'source_synced.json').is_file() and (out/'weights_manifest.json').is_file() and (out/'parts_manifest.json').is_file(),
            'Original source/archive receipts missing; no new task will be started')
    directory=out.parent/'tools'/f'gh_{VERSION}_linux_amd64';binary=directory/'bin/gh'
    with tarfile.open(package/ARCHIVE,'r:gz') as archive:
        matches=[member for member in archive.getmembers() if member.name==f'gh_{VERSION}_linux_amd64/bin/gh']
        require(len(matches)==1 and matches[0].isfile() and matches[0].size<150*1024**2,'Unexpected official binary member')
        member=matches[0];expected=hashlib.sha256()
        with archive.extractfile(member) as source:
            for block in iter(lambda:source.read(1024**2),b''):expected.update(block)
        expected=expected.hexdigest();require(expected==provenance['binary_sha256'],'Official binary identity differs')
        directory.mkdir(parents=True,exist_ok=True,mode=0o700);binary.parent.mkdir(exist_ok=True,mode=0o700)
        if binary.exists():require(binary.is_file() and not binary.is_symlink() and digest(binary)==expected,'Existing helper differs; preserved for review')
        else:
            temporary=binary.with_name('gh.offline_installing')
            require(not temporary.is_symlink(),'Unexpected helper staging link')
            with archive.extractfile(member) as source,temporary.open('wb') as target:
                for block in iter(lambda:source.read(1024**2),b''):target.write(block)
            require(digest(temporary)==expected,'Helper install readback differs')
            temporary.chmod(0o700);temporary.replace(binary)
    binary.chmod(0o700)
    receipt=dict(path=str(binary),version=VERSION,binary_sha256=expected,experiment_environment_modified=False)
    receipt_path=out/'gh_tool.json'
    if receipt_path.exists():
        previous=json.loads(receipt_path.read_text('utf-8'))
        require(previous['path']==receipt['path'] and previous['binary_sha256']==expected,'Existing helper receipt differs')
    def save(name,value):
        path=out/name;temporary=path.with_name(path.name+'.tmp')
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temporary.replace(path)
    save('gh_download_provenance.json',dict(repository='cli/cli',tag='v'+VERSION,archive=ARCHIVE,archive_sha256=ARCHIVE_SHA,
         checksum_sha256=CHECKSUMS_SHA,binary_sha256=expected,source_url=provenance['archive_url'],checksum_url=provenance['checksum_url'],
         transport='Official files downloaded and verified locally, then transferred offline by user',
         existing_job_reused=True,original_package_files_modified=False))
    save('gh_tool.json',receipt)
    print('OFFLINE_OFFICIAL_HELPER=VERIFIED',flush=True)
    print('RESUME_EXISTING_JOB='+str(out),flush=True)
    return receipt


def main():
    require(ORIGINAL.is_dir(),'Original synchronization package not found')
    launch=json.loads((ORIGINAL/'.launch_receipt.json').read_text('utf-8'));out=Path(launch['directory']).resolve()
    require(out==STAGING/EXPECTED_JOB,'Original launch receipt does not match the stopped job')
    sys.path.insert(0,str(ORIGINAL));from common import job
    identity=job(out)  # Original frozen contract and runtime files must still match.
    install(PACKAGE,out,ORIGINAL,identity)


if __name__=='__main__':main()
