"""Validate anchors under the common lock, then detach exactly one C9 batch."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode=True
import run_c9 as queue


def launch(repo,stage):
    receipt=queue.PACKAGE/'launch_receipt.json'
    queue.check_package()
    if receipt.exists():
        previous=queue.read(receipt)
        print('EXISTING_BATCH='+previous['batch_directory'])
        print('No additional training was started. Inspect this existing batch before any retry.')
        return
    with queue.locked(repo) as fd:
        # Run all source, stage, 27 historical-run and matching checks before detaching.
        queue.preflight(repo,stage)
        directory=Path(tempfile.mkdtemp(prefix='c_development9_',dir=repo/'outputs'))
        value=dict(status='starting',batch_directory=str(directory),
                   created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   stage1_directory=str(stage),total=9,test_metrics_used_for_selection=False)
        queue.write(receipt,value)
        try:
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',DGLBACKEND='pytorch',
                     WANDB_MODE='disabled',MPLBACKEND='Agg',GRAPHMAE2_R_LOCK_FD=str(fd))
            with (directory/'batch.log').open('x',encoding='utf-8') as log:
                process=subprocess.Popen([sys.executable,'-u',str(queue.PACKAGE/'run_c9.py'),
                    '--repo',str(repo),'--batch-dir',str(directory),'--stage1-dir',str(stage)],
                    cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,pass_fds=(fd,))
            with (directory/'batch.pid').open('x',encoding='ascii') as stream:
                stream.write(str(process.pid)+'\n')
            value.update(status='launched',pid=process.pid)
            queue.write(receipt,value)
            print('C9_BATCH_DIR='+str(directory))
            print('C9_PID='+str(process.pid))
            print('C9_BACKGROUND_DISPATCHED=YES; completion is not yet verified')
            print('Monitor: tail -n 25 '+str(directory/'batch.log'))
            print('Status: cat '+str(directory/'batch_status.json'))
        except BaseException as error:
            value.update(status='launch_failed',error_type=type(error).__name__,error=str(error))
            queue.write(receipt,value)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('/home/jovyan/work/GraphMAE2'))
    parser.add_argument('--stage1-dir',type=Path,default=None)
    args=parser.parse_args()
    try:
        stage=args.stage1_dir or Path(queue.read(queue.PACKAGE/'stage_receipt.json')['directory'])
        launch(args.repo.resolve(),stage.resolve())
    except BaseException as error:
        print('C9_LAUNCH_ERROR='+str(error),flush=True)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
