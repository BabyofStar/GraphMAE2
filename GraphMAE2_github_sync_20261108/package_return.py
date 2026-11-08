"""Export small receipts only; never include credentials or weight parts."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True,type=Path);parser.add_argument('--exit-code',type=int,required=True)
    args=parser.parse_args();out=args.output_dir.resolve();package=Path(__file__).resolve().parent
    allowed=['job.json','runtime_self_check.json','source_candidate_commit.json','source_synced.json','weights_manifest.json','parts_manifest.json',
             'gh_tool.json','gh_download_provenance.json','release_created.json','upload_progress.json','remote_assets_progress.json',
             'release_verified.json','final_sync_verified.json','merge_review_required.json','execution_status.json',
             'SHA256SUMS.weights.txt','RESTORE.md','restore_weights.py','execution.log']
    state=dict(exit_code=args.exit_code,source_synced=(out/'source_synced.json').exists(),
               independent_weight_backup_verified=(out/'release_verified.json').exists(),
               final_metadata_pushed=(out/'final_sync_verified.json').exists(),
               credentials_included=False,weight_parts_included=False)
    (out/'execution_status.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    files=[(out/name,name) for name in allowed if (out/name).is_file()]
    files.extend((package/name,'inputs/'+name) for name in ['contract.json','kit_identity.json','preparation.json','remote_main_result.json'])
    manifest=dict(files=[dict(path=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path,name in files])
    target=out/'github_sync_return.zip';temp=out/'github_sync_return.finalizing'
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path,name in files:archive.write(path,name)
        archive.writestr('return_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip() is not None:raise RuntimeError('Return ZIP CRC failed')
    temp.replace(target);print('RETURN_FILE='+str(target),flush=True)


if __name__=='__main__':main()
