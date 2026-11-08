"""C: private fixed epoch routing from uniform input masks to density strata."""
import hashlib

import torch

from .adaptive_masking import AdaptiveInputMask
from .density_route_math import probability_schedule


class DensityCurriculumInputMask(AdaptiveInputMask):
    MODE='density_curriculum'
    MODES=AdaptiveInputMask.MODES+(MODE,)
    NEW_MODES=AdaptiveInputMask.NEW_MODES+(MODE,)
    STRATIFIED=AdaptiveInputMask.STRATIFIED+(MODE,)

    def __init__(self,mode=MODE,density_route_seed=2026110501,**kwargs):
        if mode!=self.MODE:
            raise ValueError('DensityCurriculumInputMask requires density_curriculum')
        if (isinstance(density_route_seed,bool) or not isinstance(density_route_seed,int)
                or not 0<=density_route_seed<2**63):
            raise ValueError('density_route_seed must be an integer in [0, 2**63)')
        super().__init__(mode=mode,**kwargs)
        self.density_route_seed=density_route_seed
        self._routes=None
        self._route_probabilities=None
        self._route_epochs=None
        self._sampling_route=False

    def set_progress(self,step,total_steps):
        super().set_progress(step,total_steps)
        if self._route_epochs==total_steps:
            return
        probability=probability_schedule(total_steps,self.warmup_ratio,self.ramp_end_ratio)
        generator=torch.Generator(device='cpu')
        generator.manual_seed(self.density_route_seed)
        draws=torch.rand(total_steps,generator=generator,device='cpu').tolist()
        self._routes=[int(value<p) for value,p in zip(draws,probability)]
        self._route_probabilities=probability
        self._route_epochs=total_steps

    def effective_alpha(self):
        # No degree/difficulty weighting. Route probability is a separate quantity.
        return 0.0

    def _record(self,perm,k,alpha):
        if not self._sampling_route:
            return super()._record(perm,k,alpha)

    @torch.no_grad()
    def permutation(self,graph,device,mask_rate):
        device=self._device(device)
        if self._adaptive_key!=(self._key(graph),str(device)):
            raise RuntimeError('Call prepare for this fixed graph/device first')
        if self._routes is None or self._step is None:
            raise RuntimeError('Call set_progress before curriculum sampling')
        mode=self.mode
        try:
            self._sampling_route=True
            self.mode='density_stratified' if self._routes[self._step] else 'random'
            perm=super().permutation(graph,device,mask_rate)
        finally:
            self.mode=mode
            self._sampling_route=False
        self._record(perm,int(mask_rate*graph.num_nodes()),0.)
        return perm

    def route_plan(self):
        if self._routes is None:
            raise RuntimeError('No prepared density route plan')
        return dict(density_route_seed=self.density_route_seed,epochs=self._route_epochs,
                    warmup_ratio=self.warmup_ratio,ramp_end_ratio=self.ramp_end_ratio,
                    probability_max=1.0,routes=self._routes.copy(),
                    route_probabilities=self._route_probabilities.copy(),
                    density_epochs=sum(self._routes),uniform_epochs=self._route_epochs-sum(self._routes),
                    routes_sha256=hashlib.sha256(bytes(self._routes)).hexdigest(),
                    route_encoding='0=uniform; 1=true density quotas; one fixed route plan per dataset')

    def extra_feedback(self,ids,keep):
        if self._routes is None:
            raise RuntimeError('No route plan for reconstruction feedback')
        return dict(density_route='density' if self._routes[self._step] else 'uniform',
                    density_route_probability=self._route_probabilities[self._step],
                    density_route_seed=self.density_route_seed,
                    density_routes_sha256=hashlib.sha256(bytes(self._routes)).hexdigest())
