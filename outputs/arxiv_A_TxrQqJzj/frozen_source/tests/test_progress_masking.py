"""Run on real CPU and CUDA; no fake tensor/model runtime."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from models.adaptive_masking import AdaptiveInputMask
from models.progress_masking import ProgressInputMask
from models.edcoder import PreModel
from models.masking_math import proportional_quotas


def helpers():
    from test_random_group_masking import graph,states,restore,devices
    return graph,states,restore,devices


class ProgressMaskTests(unittest.TestCase):
    def sampler(self,g,device,mode='rank_progress',interval=1):
        obj=ProgressInputMask(mode,log_interval=interval).to(device)
        obj.prepare(g,device,g.ndata['feat'])
        return obj

    def test_private_ties_prepare_and_empty_model_state(self):
        graph,states,_,devices=helpers()
        for device in devices():
            g=graph(23,device); before=states(); obj=self.sampler(g,device)
            for x,y in zip(before,states()): self.assertTrue(torch.equal(x,y))
            self.assertEqual(dict(obj.state_dict()),{})
            self.assertEqual(sorted(obj._tie_priority.tolist()),list(range(23)))

    def test_all_rank_modes_match_S_during_warmup_and_rng(self):
        graph,states,restore,devices=helpers()
        for device in devices():
            g=graph(23,device)
            for mode in ProgressInputMask.PROGRESS_MODES:
                for rate in (0.,.5,.9,1.):
                    s=AdaptiveInputMask('density_stratified',diagnostics=True,log_interval=0).to(device)
                    s.prepare(g,device,g.ndata['feat']); s.set_progress(0,100)
                    obj=self.sampler(g,device,mode,0); obj.set_progress(0,100)
                    before=states(); expected=s.permutation(g,device,rate); after=states()
                    restore(before); actual=obj.permutation(g,device,rate)
                    self.assertTrue(torch.equal(expected,actual))
                    for x,y in zip(after,states()): self.assertTrue(torch.equal(x,y))

    def test_exact_mask_budget_and_fixed_layer_quotas(self):
        graph,_,_,devices=helpers()
        for device in devices():
            g=graph(11,device)
            for mode in ProgressInputMask.PROGRESS_MODES:
                for rate in (0.,.5,.9,1.):
                    obj=self.sampler(g,device,mode); obj.set_progress(9,10)
                    perm=obj.permutation(g,device,rate); k=int(rate*11)
                    self.assertEqual(perm.unique().numel(),11)
                    self.assertEqual(np.bincount(obj._membership[perm[:k].cpu().numpy()],minlength=4).tolist(),
                                     proportional_quotas([len(x) for x in obj._partitions],k))

    def test_both_ema_updates_have_correct_observation_clock(self):
        graph,_,_,devices=helpers()
        for device in devices():
            g=graph(8,device); obj=self.sampler(g,device)
            for step,reconstructed in ((0,0.),(1,.5)):
                obj.set_progress(step,10); ids=obj.permutation(g,device,1.)[:8]
                target=torch.ones((8,8),device=device)
                obj.observe_reconstruction(ids,target,torch.full_like(target,reconstructed),'mse')
            torch.testing.assert_close(obj.error_ema,torch.full_like(obj.error_ema,.925))
            torch.testing.assert_close(obj.slow_error_ema,torch.full_like(obj.slow_error_ema,.9925))
            self.assertTrue((obj.last_observed_step==1).all().item())

    def test_duplicate_feedback_does_not_change_any_adaptive_state(self):
        graph,_,_,_=helpers(); g=graph(8,'cpu'); obj=self.sampler(g,'cpu')
        obj.set_progress(0,10); ids=obj.permutation(g,'cpu',.5)[:4]; target=g.ndata['feat'][ids]
        obj.observe_reconstruction(ids,target,target)
        saved=[x.clone() for x in (obj.error_ema,obj.slow_error_ema,obj.observation_count,obj.last_observed_step)]
        with self.assertRaises(RuntimeError): obj.observe_reconstruction(ids,target,target)
        for a,b in zip(saved,(obj.error_ema,obj.slow_error_ema,obj.observation_count,obj.last_observed_step)):
            self.assertTrue(torch.equal(a,b))

    def test_trace_is_pre_feedback_and_does_not_alias_later_state(self):
        graph,_,_,_=helpers(); g=graph(12,'cpu'); obj=self.sampler(g,'cpu')
        obj.set_progress(0,10); ids=obj.permutation(g,'cpu',.5)[:6]
        self.assertEqual(int(obj._rank_trace[0]['observation_count'].sum()),0)
        obj.observe_reconstruction(ids,torch.ones((6,8)),torch.zeros((6,8)),'mse')
        self.assertEqual(int(obj._rank_trace[0]['observation_count'].sum()),0)
        self.assertEqual(float(obj._rank_trace[0]['fast'].sum()),0.)
        self.assertEqual(int(obj.observation_count.sum().item()),6)

    def test_H_P_PS_have_identical_weight_multisets_despite_different_scores(self):
        graph,_,_,devices=helpers()
        for device in devices():
            g=graph(23,device); results=[]
            for mode in ProgressInputMask.PROGRESS_MODES:
                obj=self.sampler(g,device,mode); obj.set_progress(9,10)
                obj.error_ema.copy_(torch.linspace(.2,2.,23,device=device))
                obj.slow_error_ema.copy_(torch.linspace(2.,.2,23,device=device))
                obj.permutation(g,device,.5)
                results.append([sorted(obj._rank_trace[-1]['weights'][group].tolist()) for group in obj._partitions])
            self.assertEqual(results[0],results[1]); self.assertEqual(results[1],results[2])

    def test_trace_export_is_numeric_and_preserves_rng(self):
        graph,states,_,_=helpers(); g=graph(12,'cpu'); obj=self.sampler(g,'cpu')
        obj.set_progress(0,10); ids=obj.permutation(g,'cpu',.5)[:6]
        obj.observe_reconstruction(ids,g.ndata['feat'][ids],g.ndata['feat'][ids])
        before=states()
        with tempfile.TemporaryDirectory() as tmp:
            obj.export_rank_trace(Path(tmp)/'trace.npz'); obj.export_node_statistics(Path(tmp)/'nodes.npz')
            with np.load(Path(tmp)/'trace.npz',allow_pickle=False) as data:
                self.assertEqual(data['weights'].shape,(1,12)); self.assertEqual(data['steps'].tolist(),[0])
            with np.load(Path(tmp)/'nodes.npz',allow_pickle=False) as data:
                self.assertEqual(int(data['observation_count'].sum()),6)
                np.testing.assert_array_equal(data['progress_score'],np.maximum(data['slow_reconstruction_error_ema']-data['reconstruction_error_ema'],0))
        for x,y in zip(before,states()): self.assertTrue(torch.equal(x,y))

    def test_missing_progress_and_repeated_sampling_fail(self):
        graph,_,_,_=helpers(); g=graph(12,'cpu'); obj=self.sampler(g,'cpu')
        with self.assertRaises(RuntimeError): obj.permutation(g,'cpu',.5)
        obj.set_progress(0,10); obj.permutation(g,'cpu',.5)
        with self.assertRaises(RuntimeError): obj.permutation(g,'cpu',.5)

    def test_labels_and_split_masks_do_not_affect_density_ties_or_scores(self):
        graph,_,_,_=helpers(); g=graph(12,'cpu'); a=self.sampler(g,'cpu')
        g.ndata['label']=torch.arange(12)
        for name in ('train_mask','val_mask','test_mask'): g.ndata[name]=torch.ones(12,dtype=torch.bool)
        b=self.sampler(g,'cpu')
        np.testing.assert_array_equal(a._membership,b._membership); np.testing.assert_array_equal(a._tie_priority,b._tie_priority)

    def test_old_methods_preserve_C_model_training_and_rng(self):
        graph,states,_,devices=helpers()
        from test_adaptive_masking_v2 import kwargs
        path=Path(__file__).parent/'fixtures/edcoder_c_dev1_reference.py'
        spec=importlib.util.spec_from_file_location('models._frozen_c_model',path)
        reference=importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
        previous=torch.get_num_threads(); torch.set_num_threads(1)
        try:
            for device in devices():
                g=graph(12,device)
                for mode in AdaptiveInputMask.MODES+('random_group_stratified','density_curriculum'):
                    results=[]
                    for factory in (reference.PreModel,PreModel):
                        torch.manual_seed(127)
                        with contextlib.redirect_stdout(io.StringIO()):
                            model=factory(**dict(kwargs(),input_mask_mode=mode,input_mask_diagnostics=True,mask_log_interval=0)).to(device)
                        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                        model.prepare_input_mask(g,device); losses=[]
                        for step in range(3):
                            model.set_input_mask_progress(step,3)
                            loss=model(g,g.ndata['feat'],targets=torch.arange(12,device=device)); losses.append(loss.detach())
                            optimizer.zero_grad(); loss.backward(); optimizer.step()
                        results.append((model,torch.stack(losses),states()))
                    a,b=results; torch.testing.assert_close(a[1],b[1],rtol=1e-4,atol=1e-6)
                    for x,y in zip(a[2],b[2]): self.assertTrue(torch.equal(x,y))
                    for name,value in a[0].state_dict().items(): torch.testing.assert_close(value,b[0].state_dict()[name],rtol=1e-4,atol=1e-6)
        finally: torch.set_num_threads(previous)

    def test_cli_factory_new_methods_backward_and_cpu_weight_reload(self):
        import sys
        from unittest import mock
        from models import build_model
        from utils import build_args,load_best_configs
        graph,_,_,devices=helpers()
        for device in devices():
            for mode in ProgressInputMask.PROGRESS_MODES:
                with mock.patch.object(sys,'argv',['test','--dataset','cora','--use_cfg','--input_mask_mode',mode]):
                    args=load_best_configs(build_args())
                args.num_features,args.num_hidden,args.num_heads=8,16,2
                with contextlib.redirect_stdout(io.StringIO()): model=build_model(args).to(device)
                g=graph(12,device); model.prepare_input_mask(g,device); model.set_input_mask_progress(0,1)
                loss=model(g,g.ndata['feat'],targets=torch.arange(12,device=device)); loss.backward()
                self.assertTrue(torch.isfinite(loss).item())
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item()>0 for p in model.encoder.parameters()))
                with contextlib.redirect_stdout(io.StringIO()): fresh=build_model(args)
                fresh.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()},strict=True)

if __name__=='__main__': unittest.main(verbosity=2)
