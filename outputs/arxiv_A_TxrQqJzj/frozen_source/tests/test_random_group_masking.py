"""Run on both real CPU and CUDA in the frozen server environment."""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import dgl

from models.adaptive_masking import AdaptiveInputMask
from models.edcoder import PreModel
from models.random_group_masking import RandomGroupInputMask
from models.masking_math import proportional_quotas


def states():
    return [torch.get_rng_state().clone()] + [x.clone() for x in torch.cuda.get_rng_state_all()]


def restore(value):
    torch.set_rng_state(value[0])
    if len(value) > 1:
        torch.cuda.set_rng_state_all(value[1:])


def devices():
    return ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])


def graph(n, device):
    nodes = torch.arange(n)
    g = dgl.add_self_loop(dgl.graph((nodes, (nodes + 1) % n), num_nodes=n)).to(device)
    g.ndata['feat'] = ((torch.arange(n*8).reshape(n,8) % 13).float()/13).to(device)
    return g


def fixture(name, module):
    spec = importlib.util.spec_from_file_location(module, Path(__file__).parent/'fixtures'/name)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


class RandomGroupTests(unittest.TestCase):
    def assert_states(self, a, b):
        self.assertEqual(len(a),len(b))
        for x,y in zip(a,b):
            self.assertTrue(torch.equal(x,y))

    def assert_feedback(self, a, b):
        if isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_feedback(a[key], b[key])
        elif isinstance(a, list):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_feedback(x, y)
        elif isinstance(a, float):
            self.assertTrue(math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-6), (a, b))
        else:
            self.assertEqual(a, b)

    def test_prepare_is_private_reproducible_and_preserves_density(self):
        for device in devices():
            g = graph(23,device)
            before = states()
            a = RandomGroupInputMask().to(device)
            a.prepare(g,device,g.ndata['feat'])
            self.assert_states(before,states())
            b = RandomGroupInputMask().to(device)
            torch.manual_seed(222)
            b.prepare(g,device,g.ndata['feat'])
            np.testing.assert_array_equal(a._sampling_membership,b._sampling_membership)
            self.assertEqual(a.group_plan(.5),b.group_plan(.5))
            s = AdaptiveInputMask('density_stratified').to(device)
            s.prepare(g,device,g.ndata['feat'])
            np.testing.assert_array_equal(a._membership,s._membership)
            self.assertEqual([len(x) for x in a._sampling_groups],[len(x) for x in s._partitions])
            self.assertEqual(dict(a.state_dict()),{})
            self.assertEqual(list(a.parameters()),[])
            before=states()
            a.prepare(g,device,g.ndata['feat'])
            self.assert_states(before,states())

    def test_exact_budget_quotas_rounding_and_sampling_rng_match_s(self):
        for device in devices():
            for n in (1,2,3,4,5,11,23):
                g=graph(n,device)
                a=RandomGroupInputMask(log_interval=0).to(device)
                s=AdaptiveInputMask('density_stratified',log_interval=0).to(device)
                for item in (a,s):
                    item.prepare(g,device,g.ndata['feat'])
                    item.set_progress(0,1)
                for rate in (0.,.1,.5,.9,1.):
                    with self.subTest(device=device,n=n,rate=rate):
                        before=states()
                        perm=a.permutation(g,device,rate)
                        after=states()
                        restore(before)
                        s.permutation(g,device,rate)
                        self.assert_states(after,states())
                        self.assertEqual(perm.unique().numel(),n)
                        self.assertEqual(perm.numel(),n)
                        k=int(rate*n)
                        actual=np.bincount(a._sampling_membership[perm[:k].cpu().numpy()],minlength=min(4,n))
                        self.assertEqual(actual.tolist(),proportional_quotas([len(x) for x in a._sampling_groups],k))

    def test_labels_splits_and_feature_values_cannot_choose_random_groups(self):
        g=graph(23,'cpu')
        a=RandomGroupInputMask()
        a.prepare(g,'cpu',g.ndata['feat'])
        g.ndata['label']=torch.full((23,),999)
        for key in ('train_mask','val_mask','test_mask'):
            g.ndata[key]=torch.ones(23,dtype=torch.bool)
        g.ndata['feat']=g.ndata['feat'].flip(0)
        b=RandomGroupInputMask()
        b.prepare(g,'cpu',g.ndata['feat'])
        np.testing.assert_array_equal(a._sampling_membership,b._sampling_membership)

    def test_feedback_logs_and_npz_keep_both_group_meanings(self):
        for device in devices():
            g=graph(23,device)
            a=RandomGroupInputMask(log_interval=1).to(device)
            a.prepare(g,device,g.ndata['feat'])
            a.set_progress(0,1)
            ids=a.permutation(g,device,.5)[:11]
            with self.assertLogs(level='INFO') as logs:
                a.observe_reconstruction(ids,g.ndata['feat'][ids],g.ndata['feat'][ids])
            self.assertIn('sampling_groups',' '.join(logs.output))
            self.assertEqual([x['masked'] for x in a.last_feedback['sampling_groups']],a.group_plan(.5)['quotas'])
            self.assertEqual(sum(x['masked'] for x in a.last_feedback['density_layers']),11)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'nodes.npz'
                before=states()
                a.export_node_statistics(path)
                self.assert_states(before,states())
                with np.load(path,allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['density_layer'],a._membership)
                    np.testing.assert_array_equal(data['sampling_group'],a._sampling_membership)
                    self.assertEqual(int(data['observation_count'].sum()),11)
                with self.assertRaises(FileExistsError):
                    a.export_node_statistics(path)

    def test_invalid_seed_and_prepare_fail_early(self):
        for seed in (True,-1,2**63,1.2):
            with self.assertRaises(ValueError):
                RandomGroupInputMask(random_group_seed=seed)
        with self.assertRaises(RuntimeError):
            RandomGroupInputMask().permutation(graph(4,'cpu'),'cpu',.5)
        with self.assertRaises(ValueError):
            RandomGroupInputMask().prepare(graph(4,'cpu'),'cpu')

    def test_group_membership_is_identical_on_cpu_and_cuda(self):
        reference=None
        for device in devices():
            g=graph(23,device)
            sampler=RandomGroupInputMask().to(device)
            sampler.prepare(g,device,g.ndata['feat'])
            if reference is not None:
                np.testing.assert_array_equal(reference,sampler._sampling_membership)
            reference=sampler._sampling_membership.copy()

    def test_all_v2_modes_match_pinned_losses_weights_rng_and_feedback(self):
        pinned=fixture('adaptive_masking_v2_dev1.py','models._pinned_adaptive_v2')
        old=fixture('edcoder_v2_dev1.py','models._pinned_edcoder_v2')
        old.AdaptiveInputMask=pinned.AdaptiveInputMask
        from test_adaptive_masking_v2 import kwargs
        previous=torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for device in devices():
                g=graph(12,device)
                for mode in AdaptiveInputMask.MODES:
                    results=[]
                    for factory in (old.PreModel,PreModel):
                        torch.manual_seed(127)
                        with contextlib.redirect_stdout(io.StringIO()):
                            model=factory(**dict(kwargs(),input_mask_mode=mode,input_mask_diagnostics=True,mask_log_interval=1)).to(device)
                        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                        model.prepare_input_mask(g,device)
                        losses=[]
                        for step in range(3):
                            model.set_input_mask_progress(step,3)
                            loss=model(g,g.ndata['feat'],targets=torch.arange(12,device=device))
                            losses.append(loss.detach())
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        results.append((model,torch.stack(losses),states(),model.input_masker.last_feedback))
                    a,b=results
                    torch.testing.assert_close(a[1],b[1],rtol=1e-4,atol=1e-6)
                    self.assert_states(a[2],b[2])
                    self.assert_feedback(a[3],b[3])
                    for key,value in a[0].state_dict().items():
                        torch.testing.assert_close(value,b[0].state_dict()[key],rtol=1e-4,atol=1e-6)
        finally:
            torch.set_num_threads(previous)

    def test_r_real_forward_backward_checkpoint_and_cli(self):
        import sys
        from unittest import mock
        from models import build_model
        from utils import build_args,load_best_configs
        for device in devices():
            with mock.patch.object(sys,'argv',['test','--dataset','cora','--use_cfg','--input_mask_mode','random_group_stratified','--random_group_seed','2026110301']):
                args=load_best_configs(build_args())
            args.num_features,args.num_hidden,args.num_heads=8,16,2
            with contextlib.redirect_stdout(io.StringIO()):
                model=build_model(args).to(device)
            self.assertIsInstance(model.input_masker,RandomGroupInputMask)
            g=graph(12,device)
            model.prepare_input_mask(g,device)
            model.set_input_mask_progress(0,1)
            loss=model(g,g.ndata['feat'],targets=torch.arange(12,device=device))
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item()>0 for p in model.encoder.parameters()))
            with contextlib.redirect_stdout(io.StringIO()):
                fresh=build_model(args)
            fresh.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()},strict=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
