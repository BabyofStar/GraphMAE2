"""Local queue/selection guards only; real DS recording smoke runs on the server."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_ds9 as queue

WORK=Path(__file__).resolve().parents[2]/'work'


class DSQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='ds9_guard_',dir=WORK)
        self.root=Path(self.tmp.name).resolve()
        assert self.root.parent==WORK.resolve()
        self.addCleanup(self.tmp.cleanup)

    def examples(self):
        rows=[]
        anchors={}
        for d,value in [('cora',.82),('citeseer',.72),('pubmed',.84)]:
            for seed in (0,1,2):
                rows.append(dict(dataset=d,seed=seed,best_val_acc=value))
                anchors[d,seed]={m:dict(best_val_acc=reference) for m,reference in [('S',.81),('B2',.73),('D',.80),('B0',.74)]}
        return rows,anchors

    def test_commands_use_ds_and_preserve_requested_phase(self):
        for phase in ('smoke','development'):
            command=queue.command(self.root,self.root/'run',self.root/'weights','cora',0,phase)
            self.assertEqual(command[command.index('--method')+1],'DS')
            self.assertEqual(command[command.index('--phase')+1],phase)
            self.assertIn('record_full_batch_r.py',command)

    def test_course_uses_fixed_original_schedule(self):
        for step,expected in [(0,0.),(20,0.),(50,.25),(80,.5),(100,.5)]:
            self.assertAlmostEqual(queue.alpha(step,101),expected)

    def test_summary_keeps_negative_dataset(self):
        rows,anchors=self.examples()
        result=queue.summary(rows,anchors)
        self.assertEqual([x['dataset'] for x in result['groups']],['cora','citeseer','pubmed'])
        self.assertLess(result['groups'][1]['DS_minus_S_mean_pp'],0)
        self.assertFalse(result['test_metrics_opened'])
        self.assertEqual(result['primary_comparison'],'DS_minus_S')

    def test_missing_and_duplicate_seeds_rejected(self):
        rows,anchors=self.examples()
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaisesRegex(RuntimeError,'Incomplete DS group'):
                queue.summary(bad,anchors)

    def test_r_result_cannot_pass_as_ds(self):
        for name in ('status.json','metrics.json'):
            (self.root/name).write_text(json.dumps(dict(status='complete',dataset='cora',method='R',seed=0,phase='development',official_epoch_budget=True,recorder_version='2.0.0-r-dev1')),encoding='utf-8')
        (self.root/'effective_args.json').write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'DS metadata differs: method'):
            queue.validate(self.root,'cora',0,'development',{}, {})

    def test_incomplete_r_batch_is_rejected_before_dispatch(self):
        outputs=self.root/'outputs/r_development9_yy6rh7vj'
        outputs.mkdir(parents=True)
        (outputs/'batch_status.json').write_text(json.dumps(dict(status='running',completed=8,total=9)),encoding='utf-8')
        with mock.patch.object(queue.support,'preflight',return_value=({'queue_files_sha256':{'legacy_candidate.py':'unused'}},{},None,{})),mock.patch.object(queue.support,'load_module'):
            with self.assertRaisesRegex(RuntimeError,'R9 must be complete'):
                queue.preflight(self.root,self.root)

    def test_existing_receipt_prevents_duplicate_dispatch(self):
        import launch_ds9
        (self.root/'launch_receipt.json').write_text(json.dumps({'batch_directory':'existing-DS-batch'}),encoding='utf-8')
        with mock.patch.object(queue,'PACKAGE',self.root),mock.patch.object(queue,'check_package'),mock.patch.object(queue,'preflight') as preflight,mock.patch.object(launch_ds9.subprocess,'Popen') as popen,contextlib.redirect_stdout(io.StringIO()):
            launch_ds9.launch(self.root,self.root)
            preflight.assert_not_called()
            popen.assert_not_called()

    def test_current_model_source_matches_accepted_r_version(self):
        accepted=json.loads((Path(__file__).parent.parent/'graphmae2_r_development9/queue_identity.json').read_text(encoding='utf-8'))
        repo=WORK/'r9_github_audit_20261105/files'
        manifest=queue.support.check_sources(repo,accepted)
        self.assertEqual(len(manifest['after_sha256']),34)


if __name__=='__main__':
    unittest.main(verbosity=2)
