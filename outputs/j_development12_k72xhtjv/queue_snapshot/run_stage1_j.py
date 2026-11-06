"""134 real tests, four 100/5 Cora diagnostic runs, then a input-only structural gate."""
import argparse
import datetime
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode=True
import protocol as p

TEST_CODE='''
import unittest,sys
loader=unittest.TestLoader(); suite=unittest.TestSuite()
for pattern in ('test_input_masking.py','test_masking_math_v2.py','test_adaptive_masking_v2.py',
                'test_random_group_masking.py','test_density_route_math.py','test_density_curriculum.py',
                'test_progress_math.py','test_progress_masking.py','test_joint_math.py','test_joint_masking.py'):
    suite.addTests(loader.discover('tests',pattern=pattern))
suite.addTests(loader.loadTestsFromNames(['test_record_full_batch','test_record_full_batch_j']))
result=unittest.TextTestRunner(verbosity=2).run(suite)
print('J_DEV1_TEST_COUNT='+str(result.testsRun))
print('J_DEV1_SKIPPED_COUNT='+str(len(result.skipped)))
sys.exit(0 if result.wasSuccessful() and not result.skipped and result.testsRun==134 else 1)
'''

def accept(repo,weights):
    p.require(Path(sys.prefix).resolve()==Path('/home/jovyan/work/envs/graphmae2').resolve(),'Activate the frozen graphmae2 environment')
    import torch,dgl,numpy
    p.require((str(torch.__version__),str(dgl.__version__),str(numpy.__version__))==('2.1.2+cu118','1.1.3+cu118','1.26.4'),'Runtime versions differ')
    p.require(torch.cuda.is_available(),'Actual CUDA is required')
    p.require(not weights.is_relative_to(repo),'Keep weights outside the Git project')
    with p.locked(repo) as fd:
        manifest=p.sources(repo); anchor=p.anchors(repo); p.prior_audit(repo)
        if (p.PACKAGE/'launch_receipt.json').exists():
            p.require(False,'A batch was already dispatched; use launch_j12.py to inspect the receipt')
        if (p.PACKAGE/'stage_receipt.json').exists():
            stage=Path(p.read(p.PACKAGE/'stage_receipt.json')['directory']).resolve()
            p.check_stage(repo,stage,manifest,anchor)
            print('J_STAGE_REUSED='+str(stage),flush=True); return stage
        p.require(not (p.PACKAGE/'structural_hold_receipt.json').exists(),
                  'Earlier structural contrast is on HOLD; preserve it and review the pairing design before retrying')
        subprocess.run(['git','cat-file','-e','bc473e8b750d6e90b7cba1c6bb20d858f617d90c^{commit}'],cwd=repo,check=True)
        stage=Path(tempfile.mkdtemp(prefix='j_dev1_stage1_',dir=repo/'outputs'))
        print('J_DEV1_STAGE1_DIR='+str(stage),flush=True)
        print('GPU='+torch.cuda.get_device_name(0),flush=True)
        print('VISIBLE_GPU_GIB='+str(torch.cuda.get_device_properties(0).total_memory/1024**3),flush=True)
        state=dict(status='running',stage='tests_and_structural_validation',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   diagnostic_pretrain_epochs=100,diagnostic_probe_epochs=5,diagnostic_methods=list(p.METHODS),
                   test_scores_used_for_selection=False,validation_scores_used_for_gate=False)
        p.write(stage/'status.json',state); started=time.perf_counter()
        try:
            p.execute([sys.executable,'-u','-c',TEST_CODE],repo,stage/'tests.log',fd)
            print('J_DEV1_ALL_TESTS_PASSED; count=134; skipped=0',flush=True)
            for method in p.METHODS:
                p.sources(repo)
                run=stage/(stage.name+'_cora_'+method+'_s0')
                p.execute(p.command(run,weights,method,0,'diagnostic'),repo,stage/('diagnostic_'+method+'.log'),fd)
                p.execute([sys.executable,'-u','record_full_batch_j.py','verify','--output-dir',str(run)],repo,stage/('reload_'+method+'.log'),fd)
                p.validate(run,method,0,'diagnostic',anchor,manifest,stage/('diagnostic_'+method+'.log'),stage/('reload_'+method+'.log'))
                print('J_DIAGNOSTIC_'+method+'=PASS (100 pretrain / 5 probe; CPU reload)',flush=True)
            gate=p.structural_gate(stage)
            p.write(stage/'structural_gate.json',gate)
            if not gate['passed']:
                state.update(status='structural_hold',wall_seconds=time.perf_counter()-started,reasons=gate['reasons'])
                p.write(stage/'status.json',state)
                p.write(p.PACKAGE/'structural_hold_receipt.json',dict(directory=str(stage),gate_sha256=p.sha(stage/'structural_gate.json')))
                raise RuntimeError('J_STRUCTURE_GATE=HOLD; no full training: '+'; '.join(gate['reasons']))
            p.sources(repo)
            state.update(status='complete',wall_seconds=time.perf_counter()-started)
            p.write(stage/'status.json',state)
            p.write(p.PACKAGE/'stage_receipt.json',dict(directory=str(stage),test_count=134,source_sha256=manifest['after_sha256'],
                                                     gate_sha256=p.sha(stage/'structural_gate.json')))
            print('J_STRUCTURE_GATE=PASS; heuristic screen, no performance conclusion',flush=True)
            print('J_DEV1_STAGE1_PASSED',flush=True)
            return stage
        except BaseException as error:
            if state['status']!='structural_hold':
                state.update(status='failed',error_type=type(error).__name__,error=str(error),wall_seconds=time.perf_counter()-started)
                p.write(stage/'status.json',state)
            raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('/home/jovyan/work/GraphMAE2'))
    parser.add_argument('--checkpoint-root',type=Path,default=Path('/home/jovyan/work/GraphMAE2_artifacts/checkpoints'))
    cli=parser.parse_args()
    try: accept(cli.repo.resolve(),cli.checkpoint_root.resolve())
    except Exception as error:
        print('J_DEV1_STAGE1_ERROR='+str(error),flush=True); return 1
    return 0

if __name__=='__main__': raise SystemExit(main())
