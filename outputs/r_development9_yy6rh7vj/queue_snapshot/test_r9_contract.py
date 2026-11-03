"""Standard-library-only queue guard and analysis tests; no training."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import run_r9 as queue


class QueueContractTests(unittest.TestCase):
    def setUp(self):
        work=Path(__file__).resolve().parents[2]/'work'
        work.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(prefix='r9_contract_',dir=work)
        self.root=Path(self.tmp.name).resolve()
        assert self.root.parent==work.resolve()
        self.addCleanup(self.tmp.cleanup)

    def status(self,value,log='R_DEV1_TEST_COUNT=80\nR_DEV1_SKIPPED_COUNT=0\n'):
        (self.root/'status.json').write_text(json.dumps(dict(status=value,stage='implementation_acceptance_only')),encoding='utf-8')
        (self.root/'tests.log').write_text(log,encoding='utf-8')

    def test_incomplete_stage_cannot_pass(self):
        for status in ('running','failed'):
            self.status(status)
            with self.assertRaisesRegex(RuntimeError,'N2 is not complete'):
                queue.check_stage_status(self.root)

    def test_test_count_and_skip_evidence_are_required(self):
        for log in ('','R_DEV1_TEST_COUNT=72\nR_DEV1_SKIPPED_COUNT=0\n','R_DEV1_TEST_COUNT=80\nR_DEV1_SKIPPED_COUNT=1\n'):
            self.status('complete',log)
            with self.assertRaisesRegex(RuntimeError,'evidence missing'):
                queue.check_stage_status(self.root)
        self.status('complete')
        queue.check_stage_status(self.root)

    def test_new_seed_field_is_the_only_new_nonmethod_exclusion(self):
        original=dict(dataset='cora',lr=.001,max_epoch=2000)
        new=dict(original,seeds=[2],input_mask_mode='random_group_stratified',random_group_seed=2026110301)
        self.assertEqual(queue.normalized_args(new),original)
        new['lr']=.002
        self.assertNotEqual(queue.normalized_args(new),original)

    def test_integer_quotas_for_all_three_datasets(self):
        for sizes,k,expected in [([677]*4,1354,[339,339,338,338]),
                                  ([832,832,832,831],1663,[416,416,416,415]),
                                  ([4930,4929,4929,4929],17745,[4437,4436,4436,4436]),
                                  ([1,1,1],0,[0,0,0]),([1,1,1],3,[1,1,1])]:
            self.assertEqual(queue.quotas(sizes,k),expected)

    def snapshot(self,entries,manifest):
        (self.root/'source_sha256.json').write_text(json.dumps(manifest),encoding='utf-8')
        with tarfile.open(self.root/'source.tar','w') as archive:
            for name,content in entries:
                item=tarfile.TarInfo(name)
                item.size=len(content)
                archive.addfile(item,io.BytesIO(content))

    def test_snapshot_bytes_are_verified_without_executing(self):
        content=b'raise RuntimeError("must never execute")\n'
        manifest={'a.py':hashlib.sha256(content).hexdigest()}
        self.snapshot([('a.py',content)],manifest)
        queue.verify_snapshot(self.root,manifest)
        self.snapshot([('a.py',b'changed')],manifest)
        with self.assertRaisesRegex(RuntimeError,'bytes differ'):
            queue.verify_snapshot(self.root,manifest)

    def test_snapshot_duplicate_and_traversal_entries_fail(self):
        content=b'x'
        manifest={'a.py':hashlib.sha256(content).hexdigest()}
        for entries in ([('a.py',content),('a.py',content)],[('../a.py',content)]):
            self.snapshot(entries,manifest)
            with self.assertRaisesRegex(RuntimeError,'Unexpected snapshot entry'):
                queue.verify_snapshot(self.root,manifest)

    def test_accepted_source_hash_drift_fails(self):
        package=self.root/'graphmae2_v2_r_dev1'
        package.mkdir()
        source=self.root/'a.py'
        source.write_bytes(b'a=1\n')
        manifest={'after_sha256':{'a.py':queue.sha(source)}}
        (package/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        identity={'accepted_r_package_files_sha256':{'manifest.json':queue.sha(package/'manifest.json')}}
        queue.check_sources(self.root,identity)
        source.write_bytes(b'a=2\n')
        with self.assertRaisesRegex(RuntimeError,'source differs'):
            queue.check_sources(self.root,identity)

    def examples(self):
        rows=[]
        anchors={}
        for dataset,rv,sv in [('cora',.81,.82),('citeseer',.74,.72),('pubmed',.83,.835)]:
            for seed in (0,1,2):
                rows.append(dict(dataset=dataset,seed=seed,best_val_acc=rv))
                anchors[dataset,seed]={key:dict(best_val_acc=value) for key,value in [('S',sv),('B0',.70),('B2',.71)]}
        return rows,anchors

    def test_summary_preserves_negative_dataset_and_pairing(self):
        rows,anchors=self.examples()
        result=queue.summary(rows,anchors)
        self.assertEqual([x['dataset'] for x in result['groups']],['cora','citeseer','pubmed'])
        self.assertAlmostEqual(result['groups'][0]['S_minus_R_mean_pp'],1.)
        self.assertAlmostEqual(result['groups'][1]['S_minus_R_mean_pp'],-2.)
        self.assertAlmostEqual(result['equal_dataset_mean_S_minus_R_pp'],-1/6)
        self.assertFalse(result['test_metrics_opened'])
        self.assertEqual(result['groups'][1]['R_vs_S_win_tie_loss'],[3,0,0])

    def test_summary_rejects_missing_or_duplicate_seed(self):
        rows,anchors=self.examples()
        for bad in (rows[:-1],rows+[copy.deepcopy(rows[0])]):
            with self.assertRaisesRegex(RuntimeError,'Incomplete R group'):
                queue.summary(bad,anchors)

    def test_existing_launch_receipt_prevents_another_dispatch(self):
        import launch_r9
        (self.root/'launch_receipt.json').write_text(json.dumps({'batch_directory':'existing-batch'}),encoding='utf-8')
        with mock.patch.object(queue,'PACKAGE',self.root),mock.patch.object(queue,'check_package'),mock.patch.object(queue,'preflight') as preflight,mock.patch.object(launch_r9.subprocess,'Popen') as popen:
            launch_r9.launch(self.root,self.root)
            preflight.assert_not_called()
            popen.assert_not_called()


if __name__=='__main__':
    unittest.main(verbosity=2)
