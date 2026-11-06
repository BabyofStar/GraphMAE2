"""Detach one serial twelve-run pilot, retaining the common lock in its child."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode=True
import protocol as p
from run_p12 import preflight

def launch(repo):
    p.check_package(); receipt=p.PACKAGE/'launch_receipt.json'
    if receipt.exists():
        value=p.read(receipt)
        print('EXISTING_BATCH='+value['batch_directory'],flush=True)
        print('No additional training started; inspect the existing batch.',flush=True); return
    stage=Path(p.read(p.PACKAGE/'stage_receipt.json')['directory']).resolve()
    with p.locked(repo) as fd:
        preflight(repo,stage)
        batch=Path(tempfile.mkdtemp(prefix='p_development12_',dir=repo/'outputs'))
        value=dict(status='starting',batch_directory=str(batch),stage1_directory=str(stage),total=12,
                   created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_metrics_used_for_selection=False)
        p.write(receipt,value)
        try:
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',DGLBACKEND='pytorch',WANDB_MODE='disabled',
                     MPLBACKEND='Agg',GRAPHMAE2_R_LOCK_FD=str(fd))
            with (batch/'batch.log').open('x',encoding='utf-8') as log:
                process=subprocess.Popen([sys.executable,'-u',str(p.PACKAGE/'run_p12.py'),'--repo',str(repo),
                                          '--batch-dir',str(batch),'--stage1-dir',str(stage)],cwd=repo,env=env,
                                          stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(fd,))
            (batch/'batch.pid').write_text(str(process.pid)+'\n',encoding='ascii')
            value.update(status='launched',pid=process.pid); p.write(receipt,value)
            print('P12_BATCH_DIR='+str(batch),flush=True); print('P12_PID='+str(process.pid),flush=True)
            print('P12_BACKGROUND_DISPATCHED=YES; completion not yet verified',flush=True)
            print('Monitor: tail -n 25 '+str(batch/'batch.log'),flush=True)
            print('Status: cat '+str(batch/'batch_status.json'),flush=True)
        except BaseException as error:
            value.update(status='launch_failed',error_type=type(error).__name__,error=str(error)); p.write(receipt,value); raise

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--repo',type=Path,default=Path('/home/jovyan/work/GraphMAE2'))
    try: launch(parser.parse_args().repo.resolve())
    except Exception as error: print('P12_LAUNCH_ERROR='+str(error),flush=True); return 1
    return 0

if __name__=='__main__': raise SystemExit(main())
