"""Real CPU/CUDA curriculum tests for the frozen graphmae2 environment."""
import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import unittest

import numpy as np
import torch

from models.adaptive_masking import AdaptiveInputMask
from models.density_curriculum import DensityCurriculumInputMask
from models.edcoder import PreModel
from models.masking_math import proportional_quotas


def helpers():
    from test_random_group_masking import graph,states,restore,devices
    return graph,states,restore,devices


class CurriculumTests(unittest.TestCase):
    def assert_states(self,a,b):
        self.assertEqual(len(a),len(b))
        for x,y in zip(a,b):
            self.assertTrue(torch.equal(x,y))

    def test_private_plan_and_boundaries_do_not_change_training_rng(self):
        graph,states,_,devices=helpers()
        for device in devices():
            g=graph(23,device)
            before=states()
            sampler=DensityCurriculumInputMask().to(device)
            sampler.prepare(g,device,g.ndata['feat'])
            sampler.set_progress(0,2000)
            self.assert_states(before,states())
            plan=sampler.route_plan()
            self.assertEqual(plan['routes'][:400],[0]*400)
            self.assertEqual(plan['routes'][1600:],[1]*400)
            self.assertEqual(plan['routes_sha256'],hashlib.sha256(bytes(plan['routes'])).hexdigest())
            self.assertEqual(plan['density_epochs']+plan['uniform_epochs'],2000)
            self.assertEqual(dict(sampler.state_dict()),{})

    def test_plan_is_fixed_across_training_seeds_and_progress_cache(self):
        plans=[]
        for seed in (0,1,2):
            torch.manual_seed(seed)
            sampler=DensityCurriculumInputMask()
            sampler.set_progress(0,500)
            plans.append(sampler.route_plan())
            saved=sampler._routes
            sampler.set_progress(499,500)
            self.assertIs(sampler._routes,saved)
        self.assertEqual(plans[0],plans[1])
        self.assertEqual(plans[1],plans[2])

    def test_both_routes_match_original_masks_and_rng_exactly(self):
        graph,states,restore,devices=helpers()
        for device in devices():
            g=graph(23,device)
            c=DensityCurriculumInputMask(log_interval=0).to(device)
            c.prepare(g,device,g.ndata['feat'])
            for step,mode in ((0,'random'),(10,'density_stratified')):
                original=AdaptiveInputMask(mode,diagnostics=True,log_interval=0).to(device)
                original.prepare(g,device,g.ndata['feat'])
                original.set_progress(step,11)
                c.set_progress(step,11)
                for rate in (0.,.5,.9,1.):
                    before=states()
                    expected=original.permutation(g,device,rate)
                    after=states()
                    restore(before)
                    actual=c.permutation(g,device,rate)
                    self.assertTrue(torch.equal(expected,actual))
                    self.assert_states(after,states())

    def test_each_epoch_exact_budget_and_density_route_quotas(self):
        graph,_,_,devices=helpers()
        for device in devices():
            g=graph(11,device)
            c=DensityCurriculumInputMask(log_interval=0).to(device)
            c.prepare(g,device,g.ndata['feat'])
            for step in range(11):
                c.set_progress(step,11)
                perm=c.permutation(g,device,.5)
                self.assertEqual(perm.numel(),11)
                self.assertEqual(perm.unique().numel(),11)
                if c._routes[step]:
                    counts=np.bincount(c._membership[perm[:5].cpu().numpy()],minlength=4).tolist()
                    self.assertEqual(counts,proportional_quotas([len(p) for p in c._partitions],5))

    def test_feedback_has_actual_route_probability_and_single_mask_record(self):
        graph,_,_,_=helpers()
        g=graph(11,'cpu')
        c=DensityCurriculumInputMask(log_interval=1)
        c.prepare(g,'cpu',g.ndata['feat'])
        for step,route in ((0,'uniform'),(10,'density')):
            c.set_progress(step,11)
            with self.assertLogs(level='INFO') as logs:
                ids=c.permutation(g,'cpu',.5)[:5]
                c.observe_reconstruction(ids,g.ndata['feat'][ids],g.ndata['feat'][ids])
            self.assertEqual(sum('INPUT_MASK {' in line for line in logs.output),1)
            self.assertEqual(c.last_feedback['density_route'],route)
            self.assertEqual(c.last_feedback['mode'],'density_curriculum')
            self.assertEqual(c.last_feedback['alpha'],0.)

    def test_invalid_seed_and_missing_progress_fail(self):
        graph,_,_,_=helpers()
        for seed in (True,-1,2**63,1.5):
            with self.assertRaises(ValueError):
                DensityCurriculumInputMask(density_route_seed=seed)
        g=graph(11,'cpu')
        c=DensityCurriculumInputMask()
        c.prepare(g,'cpu',g.ndata['feat'])
        with self.assertRaises(RuntimeError):
            c.permutation(g,'cpu',.5)

    def test_labels_and_split_masks_do_not_change_route_or_density(self):
        graph,_,_,_=helpers()
        g=graph(23,'cpu')
        a=DensityCurriculumInputMask()
        a.prepare(g,'cpu',g.ndata['feat'])
        a.set_progress(0,11)
        g.ndata['label']=torch.full((23,),999)
        for name in ('train_mask','val_mask','test_mask'):
            g.ndata[name]=torch.ones(23,dtype=torch.bool)
        b=DensityCurriculumInputMask()
        b.prepare(g,'cpu',g.ndata['feat'])
        b.set_progress(0,11)
        self.assertEqual(a.route_plan(),b.route_plan())
        np.testing.assert_array_equal(a._membership,b._membership)

    def test_all_old_methods_match_pinned_r_model_losses_weights_and_rng(self):
        graph,states,_,devices=helpers()
        from test_adaptive_masking_v2 import kwargs
        path=Path(__file__).parent/'fixtures/edcoder_r_dev1_reference.py'
        spec=importlib.util.spec_from_file_location('models._pinned_r_model',path)
        reference=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        previous=torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for device in devices():
                g=graph(12,device)
                for mode in AdaptiveInputMask.MODES+('random_group_stratified',):
                    results=[]
                    for factory in (reference.PreModel,PreModel):
                        torch.manual_seed(127)
                        with contextlib.redirect_stdout(io.StringIO()):
                            model=factory(**dict(kwargs(),input_mask_mode=mode,input_mask_diagnostics=True,mask_log_interval=0)).to(device)
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
                        results.append((model,torch.stack(losses),states()))
                    a,b=results
                    torch.testing.assert_close(a[1],b[1],rtol=1e-4,atol=1e-6)
                    self.assert_states(a[2],b[2])
                    for key,value in a[0].state_dict().items():
                        torch.testing.assert_close(value,b[0].state_dict()[key],rtol=1e-4,atol=1e-6)
        finally:
            torch.set_num_threads(previous)

    def test_cli_factory_real_backward_and_cpu_weight_reload(self):
        import sys
        from unittest import mock
        from models import build_model
        from utils import build_args,load_best_configs
        graph,_,_,devices=helpers()
        for device in devices():
            argv=['test','--dataset','cora','--use_cfg','--input_mask_mode','density_curriculum','--density_route_seed','2026110501']
            with mock.patch.object(sys,'argv',argv):
                args=load_best_configs(build_args())
            args.num_features,args.num_hidden,args.num_heads=8,16,2
            with contextlib.redirect_stdout(io.StringIO()):
                model=build_model(args).to(device)
            self.assertIsInstance(model.input_masker,DensityCurriculumInputMask)
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
