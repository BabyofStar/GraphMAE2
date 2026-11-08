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
from models.joint_masking import JointInputMask
from models.edcoder import PreModel
from models.masking_math import proportional_quotas

def helpers():
    from test_random_group_masking import graph,states,restore,devices
    return graph,states,restore,devices

class JointMaskTests(unittest.TestCase):
    def sampler(self,g,device,mode='pair_semantic_antithetic'):
        obj=JointInputMask(mode,log_interval=1).to(device); obj.prepare(g,device,g.ndata['feat']); return obj

    def test_private_pairing_preserves_rng_and_empty_weight_state(self):
        graph,states,_,devices=helpers()
        for device in devices():
            g=graph(23,device); before=states(); obj=self.sampler(g,device)
            self.assertEqual(dict(obj.state_dict()),{})
            for x,y in zip(before,states()): self.assertTrue(torch.equal(x,y))

    def test_uniform_reference_matches_original_S_masks_and_rng(self):
        graph,states,restore,devices=helpers()
        for device in devices():
            g=graph(23,device)
            for rate in (0.,.5,.9,1.):
                original=AdaptiveInputMask('density_stratified',diagnostics=True,log_interval=0).to(device)
                original.prepare(g,device,g.ndata['feat']); original.set_progress(0,1)
                obj=self.sampler(g,device,'pair_uniform_reference'); obj.set_progress(0,1)
                before=states(); expected=original.permutation(g,device,rate); after=states()
                restore(before); actual=obj.permutation(g,device,rate)
                self.assertTrue(torch.equal(expected,actual))
                for x,y in zip(after,states()): self.assertTrue(torch.equal(x,y))

    def test_pair_plans_are_fixed_across_training_seeds(self):
        graph,_,_,_=helpers(); g=graph(23,'cpu'); plans=[]
        for seed in (0,1,2):
            torch.manual_seed(seed); plans.append(self.sampler(g,'cpu').pair_plan())
        self.assertEqual(plans[0],plans[1]); self.assertEqual(plans[1],plans[2])

    def test_semantic_reference_antithetic_and_comask_use_identical_pair_plan(self):
        graph,_,_,_=helpers(); g=graph(23,'cpu')
        plans=[self.sampler(g,'cpu',m).pair_plan() for m in ('pair_uniform_reference','pair_semantic_antithetic','pair_semantic_comask')]
        self.assertEqual(plans[0],plans[1]); self.assertEqual(plans[1],plans[2])

    def test_all_pairings_cover_every_node_exactly_once(self):
        graph,_,_,_=helpers(); g=graph(23,'cpu')
        for mode in JointInputMask.JOINT_MODES:
            obj=self.sampler(g,'cpu',mode); covered=[]
            for group in obj.pair_plan()['groups']:
                nodes=[i for pair in group['pairs'] for i in pair]
                if group['singleton'] is not None: nodes.append(group['singleton'])
                self.assertEqual(len(nodes),group['nodes']); covered+=nodes
            self.assertEqual(sorted(covered),list(range(23)))

    def test_every_mode_rate_has_exact_layer_quotas(self):
        graph,_,_,devices=helpers()
        for device in devices():
            g=graph(23,device)
            for mode in JointInputMask.JOINT_MODES:
                for rate in (0.,.5,.9,1.):
                    obj=self.sampler(g,device,mode); obj.set_progress(0,1)
                    perm=obj.permutation(g,device,rate); k=int(rate*23)
                    self.assertEqual(perm.unique().numel(),23)
                    self.assertEqual(np.bincount(obj._membership[perm[:k].cpu().numpy()],minlength=4).tolist(),
                                     proportional_quotas([len(p) for p in obj._partitions],k))

    def test_half_masking_has_expected_opposite_joint_dependence(self):
        graph,_,_,_=helpers(); g=graph(24,'cpu')
        for mode,expected in (('pair_semantic_antithetic',1.),('pair_random_antithetic',1.),('pair_semantic_comask',None)):
            obj=self.sampler(g,'cpu',mode)
            obj.set_progress(0,1); ids=obj.permutation(g,'cpu',.5)[:12]
            mask=np.zeros(24,dtype=bool); mask[ids.numpy()]=True
            for group in obj.pair_plan()['groups']:
                pairs=np.asarray(group['pairs']); fraction=float((mask[pairs[:,0]]!=mask[pairs[:,1]]).mean())
                if expected is not None: self.assertEqual(fraction,expected)
                else: self.assertLessEqual(fraction,1/len(pairs))

    def test_feedback_and_exports_preserve_rng_and_numeric_trace(self):
        graph,states,_,_=helpers(); g=graph(12,'cpu'); obj=self.sampler(g,'cpu')
        obj.set_progress(0,1); ids=obj.permutation(g,'cpu',.5)[:6]
        obj.observe_reconstruction(ids,g.ndata['feat'][ids],g.ndata['feat'][ids])
        before=states()
        with tempfile.TemporaryDirectory() as tmp:
            obj.export_joint_trace(Path(tmp)/'trace.npz'); obj.export_node_statistics(Path(tmp)/'nodes.npz')
            with np.load(Path(tmp)/'trace.npz',allow_pickle=False) as data:
                self.assertEqual(data['masked'].shape,(1,12)); self.assertEqual(int(data['masked'].sum()),6)
            with np.load(Path(tmp)/'nodes.npz',allow_pickle=False) as data: self.assertEqual(int(data['observation_count'].sum()),6)
        for x,y in zip(before,states()): self.assertTrue(torch.equal(x,y))

    def test_labels_and_splits_do_not_affect_pair_plans(self):
        graph,_,_,_=helpers(); g=graph(23,'cpu'); a=self.sampler(g,'cpu').pair_plan()
        g.ndata['label']=torch.full((23,),999)
        for name in ('train_mask','val_mask','test_mask'): g.ndata[name]=torch.ones(23,dtype=torch.bool)
        self.assertEqual(a,self.sampler(g,'cpu').pair_plan())

    def test_missing_progress_and_repeated_sampling_rejected(self):
        graph,_,_,_=helpers(); g=graph(12,'cpu'); obj=self.sampler(g,'cpu')
        with self.assertRaises(RuntimeError): obj.permutation(g,'cpu',.5)
        obj.set_progress(0,1); obj.permutation(g,'cpu',.5)
        with self.assertRaises(RuntimeError): obj.permutation(g,'cpu',.5)

    def test_old_P_model_methods_preserve_losses_weights_and_rng(self):
        graph,states,_,devices=helpers()
        from test_adaptive_masking_v2 import kwargs
        spec=importlib.util.spec_from_file_location('models._frozen_p_model',Path(__file__).parent/'fixtures/edcoder_p_dev1_reference.py')
        reference=importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
        previous=torch.get_num_threads(); torch.set_num_threads(1)
        try:
            for device in devices():
                g=graph(12,device)
                for mode in AdaptiveInputMask.MODES+('random_group_stratified','density_curriculum')+ProgressInputMask.PROGRESS_MODES:
                    results=[]
                    for factory in (reference.PreModel,PreModel):
                        torch.manual_seed(127)
                        with contextlib.redirect_stdout(io.StringIO()):
                            model=factory(**dict(kwargs(),input_mask_mode=mode,input_mask_diagnostics=True,mask_log_interval=0)).to(device)
                        optimizer=torch.optim.Adam(model.parameters(),lr=.001); model.prepare_input_mask(g,device); losses=[]
                        for step in range(3):
                            model.set_input_mask_progress(step,3); loss=model(g,g.ndata['feat'],targets=torch.arange(12,device=device))
                            losses.append(loss.detach()); optimizer.zero_grad(); loss.backward(); optimizer.step()
                        results.append((model,torch.stack(losses),states()))
                    a,b=results; torch.testing.assert_close(a[1],b[1],rtol=1e-4,atol=1e-6)
                    for x,y in zip(a[2],b[2]): self.assertTrue(torch.equal(x,y))
                    for name,value in a[0].state_dict().items(): torch.testing.assert_close(value,b[0].state_dict()[name],rtol=1e-4,atol=1e-6)
        finally: torch.set_num_threads(previous)

    def test_cli_factory_backward_and_cpu_reload_for_all_new_modes(self):
        import sys
        from unittest import mock
        from models import build_model
        from utils import build_args,load_best_configs
        graph,_,_,devices=helpers()
        for device in devices():
            for mode in JointInputMask.JOINT_MODES:
                with mock.patch.object(sys,'argv',['test','--dataset','cora','--use_cfg','--input_mask_mode',mode]): args=load_best_configs(build_args())
                args.num_features,args.num_hidden,args.num_heads=8,16,2
                with contextlib.redirect_stdout(io.StringIO()): model=build_model(args).to(device)
                g=graph(12,device); model.prepare_input_mask(g,device); model.set_input_mask_progress(0,1)
                loss=model(g,g.ndata['feat'],targets=torch.arange(12,device=device)); loss.backward()
                self.assertTrue(torch.isfinite(loss).item())
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item()>0 for p in model.encoder.parameters()))
                with contextlib.redirect_stdout(io.StringIO()): fresh=build_model(args)
                fresh.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()},strict=True)

if __name__=='__main__': unittest.main(verbosity=2)
