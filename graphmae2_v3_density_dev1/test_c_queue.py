import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import c_validation as protocol
import run_c9 as queue

WORK=Path(__file__).resolve().parents[2]/'work'


class CQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='c9_contract_',dir=WORK)
        self.root=Path(self.tmp.name).resolve()
        assert self.root.parent==WORK.resolve()
        self.addCleanup(self.tmp.cleanup)

    def plan(self):
        routes=[0,0,1,1,1]
        return dict(density_route_seed=2026110501,epochs=5,warmup_ratio=.2,ramp_end_ratio=.8,probability_max=1.,
                    routes=routes,route_probabilities=[0.,1/12,.5,11/12,1.],density_epochs=3,uniform_epochs=2,
                    routes_sha256=hashlib.sha256(bytes(routes)).hexdigest())

    def examples(self):
        rows=[]
        anchors={}
        for d,value in [('cora',.82),('citeseer',.72),('pubmed',.84)]:
            for seed in (0,1,2):
                rows.append(dict(dataset=d,seed=seed,best_val_acc=value))
                anchors[d,seed]={m:dict(best_val_acc=v) for m,v in [('S',.81),('B2',.73),('B0',.74)]}
        return rows,anchors

    def test_only_c_full_development_command(self):
        command=queue.command(self.root,self.root/'weights','cora',0)
        self.assertEqual(command[command.index('--method')+1],'C')
        self.assertEqual(command[command.index('--phase')+1],'development')
        self.assertIn('record_full_batch_c.py',command)

    def test_probability_max_is_not_degree_alpha(self):
        p=self.plan()
        protocol.validate_plan(p,'cora',5)
        p['probability_max']=.5
        with self.assertRaisesRegex(RuntimeError,'parameters differ'):
            protocol.validate_plan(p,'cora',5)

    def test_forced_uniform_and_density_endpoints(self):
        for index,value in ((0,1),(4,0)):
            p=self.plan()
            p['routes'][index]=value
            with self.assertRaisesRegex(RuntimeError,'endpoint'):
                protocol.validate_plan(p,'cora',5)

    def test_route_count_and_fingerprint_are_checked(self):
        for key,value in [('density_epochs',4),('routes_sha256','0'*64)]:
            p=self.plan()
            p[key]=value
            with self.assertRaisesRegex(RuntimeError,'count/hash'):
                protocol.validate_plan(p,'cora',5)

    def test_summary_preserves_negative_dataset_and_no_control_claim(self):
        rows,anchors=self.examples()
        result=queue.summary(rows,anchors)
        self.assertEqual([g['dataset'] for g in result['groups']],['cora','citeseer','pubmed'])
        self.assertLess(result['groups'][1]['C_minus_S_mean_pp'],0.)
        self.assertEqual(result['route_order_control_runs'],0)
        self.assertFalse(result['test_metrics_opened'])

    def test_missing_and_duplicate_seed_rejected(self):
        rows,anchors=self.examples()
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaisesRegex(RuntimeError,'Incomplete C group'):
                queue.summary(bad,anchors)

    def test_existing_receipt_does_not_dispatch_more_runs(self):
        import launch_c9
        (self.root/'launch_receipt.json').write_text(json.dumps({'batch_directory':'existing-C-batch'}),encoding='utf-8')
        with mock.patch.object(queue,'PACKAGE',self.root),mock.patch.object(queue,'check_package'),mock.patch.object(queue,'preflight') as preflight,mock.patch.object(launch_c9.subprocess,'Popen') as popen,contextlib.redirect_stdout(io.StringIO()):
            launch_c9.launch(self.root,self.root)
            preflight.assert_not_called()
            popen.assert_not_called()

    def test_independent_source_contract_and_budget(self):
        manifest=protocol.sources(WORK/'graphmae2_v3_density_dev1/src')
        self.assertEqual(len(manifest['after_sha256']),41)
        self.assertEqual(manifest['approved_full_runs'],9)
        self.assertEqual(manifest['route_order_control_runs'],0)

    def test_old_arguments_remain_required(self):
        args=dict(dataset='cora',max_epoch=2000,lr=.001,seeds=[0],input_mask_mode='density_curriculum',
                  random_group_seed=2026110301,density_route_seed=2026110501)
        expected=dict(dataset='cora',max_epoch=2000,lr=.001)
        self.assertEqual(protocol.normal(args),expected)
        args['lr']=.002
        self.assertNotEqual(protocol.normal(args),expected)


if __name__=='__main__':
    unittest.main(verbosity=2)
