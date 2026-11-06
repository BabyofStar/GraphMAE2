"""Small local behavioral checks; no model runtime or training."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import protocol as p
import run_p12 as queue
import launch_p12 as launcher

TEMP_ROOT=Path(__file__).resolve().parent/'.test_tmp'
TEMP_ROOT.mkdir(exist_ok=True)

class QueueTests(unittest.TestCase):
    def rows(self):
        return [dict(method=m,seed=s,best_val_acc=.70+.01*s+({'S':0,'H':.004,'P':.008,'PS':-.002}[m])) for m,s in p.JOBS]

    def test_budget_covers_each_method_seed_exactly_once(self):
        self.assertEqual(len(p.JOBS),12)
        self.assertEqual(set(p.JOBS),{(m,s) for m in ('S','H','P','PS') for s in (0,1,2)})
        self.assertNotEqual([m for m,s in p.JOBS if s==0],[m for m,s in p.JOBS if s==1])

    def test_paired_summary_is_independent_of_run_order(self):
        rows=self.rows(); result=queue.summary(list(reversed(rows)))
        self.assertEqual(result['primary_comparison'],'P_minus_S')
        main=next(x for x in result['comparisons'] if x['method']=='P' and x['reference']=='S')
        self.assertAlmostEqual(main['mean_pp'],.8)
        self.assertEqual((main['wins'],main['ties'],main['losses']),(3,0,0))
        self.assertFalse(result['test_metrics_opened'])

    def test_incomplete_or_duplicate_seed_groups_do_not_get_a_summary(self):
        rows=self.rows()
        with self.assertRaises(RuntimeError): queue.summary(rows[:-1])
        rows[-1]=dict(rows[0])
        with self.assertRaises(RuntimeError): queue.summary(rows)

    def test_launch_receipt_prevents_any_new_process_or_preflight(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as tmp:
            assert Path(tmp).resolve().is_relative_to(TEMP_ROOT)
            directory=Path(tmp)
            (directory/'launch_receipt.json').write_text(json.dumps(dict(batch_directory='/existing/batch')),encoding='utf-8')
            with (mock.patch.object(p,'PACKAGE',directory),mock.patch.object(p,'check_package'),mock.patch.object(launcher,'preflight') as gate,
                  mock.patch.object(launcher.subprocess,'Popen') as process,contextlib.redirect_stdout(io.StringIO())):
                launcher.launch(Path('/unused'))
                process.assert_not_called(); gate.assert_not_called()

    def test_frozen_package_rejects_changed_file_before_launch(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as tmp:
            assert Path(tmp).resolve().is_relative_to(TEMP_ROOT)
            directory=Path(tmp); f=directory/'data.txt'; f.write_text('accepted',encoding='utf-8')
            identity=dict(approved_full_runs=12,dataset='cora',jobs=p.JOBS,queue_files_sha256={'data.txt':p.sha(f)})
            (directory/'queue_identity.json').write_text(json.dumps(identity),encoding='utf-8')
            with mock.patch.object(p,'PACKAGE',directory):
                p.check_package(); f.write_text('changed',encoding='utf-8')
                with self.assertRaises(RuntimeError): p.check_package()

    def test_wrong_full_run_budget_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as tmp:
            assert Path(tmp).resolve().is_relative_to(TEMP_ROOT)
            directory=Path(tmp)
            (directory/'queue_identity.json').write_text(json.dumps(dict(approved_full_runs=36,dataset='cora',jobs=p.JOBS)),encoding='utf-8')
            with mock.patch.object(p,'PACKAGE',directory):
                with self.assertRaises(RuntimeError): p.check_package()

if __name__=='__main__': unittest.main(verbosity=2)
