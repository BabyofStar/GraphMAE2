"""Observation-time EMA progress proxy; all new arms have identical rank weights."""
import hashlib
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .adaptive_masking import AdaptiveInputMask
from .masking_math import finite_correlation,proportional_quotas
from .progress_math import FAST_BETA,SLOW_BETA,TIE_SEED,SHUFFLE_SEED,rank_weights,shuffle_seed


class ProgressInputMask(AdaptiveInputMask):
    PROGRESS_MODES=('rank_difficulty','rank_progress','rank_progress_shuffled')
    MODES=AdaptiveInputMask.MODES+PROGRESS_MODES
    NEW_MODES=AdaptiveInputMask.NEW_MODES+PROGRESS_MODES
    STRATIFIED=AdaptiveInputMask.STRATIFIED+PROGRESS_MODES

    def __init__(self,mode='rank_progress',**kwargs):
        if mode not in self.PROGRESS_MODES:
            raise ValueError('ProgressInputMask requires a ranked method')
        super().__init__(mode=mode,**kwargs)
        if self.ema_beta!=FAST_BETA or self.density_bins!=4:
            raise ValueError('Frozen progress protocol requires fast beta=.9 and four strata')
        self.register_buffer('slow_error_ema',torch.empty(0),persistent=False)
        self.register_buffer('last_observed_step',torch.empty(0,dtype=torch.long),persistent=False)
        self._tie_priority=None
        self._rank_trace=[]
        self._last_sampling_step=None
        self._last_weight_sha256=None

    @torch.no_grad()
    def prepare(self,graph,device,features=None):
        key=(self._key(graph),str(self._device(device)))
        if self._adaptive_key==key:
            return
        started=time.perf_counter()
        super().prepare(graph,device,features)
        n=graph.num_nodes()
        self.slow_error_ema=torch.zeros_like(self.error_ema)
        self.last_observed_step=torch.full((n,),-1,dtype=torch.long,device=self.error_ema.device)
        generator=torch.Generator(device='cpu').manual_seed(TIE_SEED)
        self._tie_priority=torch.randperm(n,generator=generator,device='cpu').numpy()
        self._rank_trace=[]
        self._last_sampling_step=None
        self.preparation_seconds=time.perf_counter()-started

    def _should_log(self):
        return self.log_interval and self._step is not None and (self._step%self.log_interval==0 or self._step==self._total_steps-1)

    @torch.no_grad()
    def permutation(self,graph,device,mask_rate):
        device=self._device(device); n=graph.num_nodes()
        if self._adaptive_key!=(self._key(graph),str(device)) or self._step is None:
            raise RuntimeError('Prepare this graph/device and set progress first')
        if not math.isfinite(mask_rate) or not 0<=mask_rate<=1:
            raise ValueError('Invalid mask rate')
        if self._last_sampling_step==self._step:
            raise RuntimeError('Exactly one ranked sampling call per training epoch')
        alpha=self.effective_alpha(); k=int(mask_rate*n)
        fast=self.error_ema.detach().cpu().numpy().copy()
        slow=self.slow_error_ema.detach().cpu().numpy().copy()
        raw=fast.copy() if self.mode=='rank_difficulty' else np.maximum(slow-fast,0.)
        weights=np.zeros(n,dtype=np.float64)
        quotas=proportional_quotas([len(g) for g in self._partitions],k)
        selected=[]
        for layer,(group,ids,quota) in enumerate(zip(self._partitions,self._partition_tensors,quotas)):
            local=np.asarray(rank_weights(raw[group].tolist(),self._tie_priority[group].tolist(),alpha),dtype=np.float64)
            if self.mode=='rank_progress_shuffled':
                generator=torch.Generator(device='cpu').manual_seed(shuffle_seed(self._step,layer))
                permutation=torch.randperm(len(group),generator=generator,device='cpu').numpy()
                local=local[permutation]
            weights[group]=local
            if quota:
                chosen=(torch.randperm(len(group),device=device)[:quota] if alpha==0 else
                        torch.multinomial(torch.as_tensor(local,dtype=torch.float64,device=device),quota,replacement=False))
                selected.append(ids[chosen])
        masked=torch.cat(selected) if selected else torch.empty(0,dtype=torch.long,device=device)
        keep=torch.ones(n,dtype=torch.bool,device=device); keep[masked]=False
        perm=torch.cat((masked,keep.nonzero(as_tuple=False).flatten()))
        if k==0:
            perm=torch.randperm(n,device=device)
        self._last_weight_sha256=hashlib.sha256(weights.astype('<f8',copy=False).tobytes()).hexdigest()
        if self._should_log():
            observed=self.observation_count.detach().cpu().numpy().copy()
            last=self.last_observed_step.detach().cpu().numpy().copy()
            masked_array=np.zeros(n,dtype=np.uint8); masked_array[masked.cpu().numpy()]=1
            self._rank_trace.append(dict(step=self._step,fast=fast,slow=slow,scores=raw,weights=weights,
                                         observation_count=observed,last_observed_step=last,masked=masked_array))
        self._last_sampling_step=self._step
        self._record(perm,k,alpha)
        return perm

    @torch.no_grad()
    def observe_reconstruction(self,mask_nodes,target,reconstruction,loss_name='sce',alpha_l=2):
        if self._last_sampling_step!=self._step:
            raise RuntimeError('Feedback must follow the current epoch selection')
        if self._feedback_step==self._step:
            raise RuntimeError('Duplicate feedback')
        if (mask_nodes.ndim!=1 or mask_nodes.numel()==0 or mask_nodes.unique().numel()!=mask_nodes.numel()
                or target.ndim!=2 or target.shape!=reconstruction.shape or target.shape[0]!=mask_nodes.numel()):
            raise ValueError('Invalid paired reconstruction feedback')
        # Superclass owns validation and fast EMA; slow state updates only after it accepts feedback.
        previous=self.observation_count[mask_nodes].clone()
        if loss_name=='sce':
            error=(1-(F.normalize(target.detach(),p=2,dim=-1)*F.normalize(reconstruction.detach(),p=2,dim=-1)).sum(-1)).clamp_min(0).pow(alpha_l)
        elif loss_name=='mse':
            error=(target.detach()-reconstruction.detach()).square().mean(-1)
        else:
            raise ValueError('Unsupported loss')
        # extra_feedback is called during super(), so prepare validated post-update state first;
        # rollback if superclass rejects a duplicate/invalid update.
        old_slow=self.slow_error_ema[mask_nodes].clone()
        old_last=self.last_observed_step[mask_nodes].clone()
        self.slow_error_ema[mask_nodes]=torch.where(previous>0,SLOW_BETA*old_slow+(1-SLOW_BETA)*error,error)
        self.last_observed_step[mask_nodes]=self._step
        try:
            super().observe_reconstruction(mask_nodes,target,reconstruction,loss_name,alpha_l)
        except BaseException:
            self.slow_error_ema[mask_nodes]=old_slow
            self.last_observed_step[mask_nodes]=old_last
            raise

    def extra_feedback(self,ids,keep):
        fast=self.error_ema.detach().cpu().numpy(); slow=self.slow_error_ema.detach().cpu().numpy()
        counts=self.observation_count.detach().cpu().numpy(); seen=counts>0
        progress=np.maximum(slow-fast,0.)
        ages=self._step-self.last_observed_step.detach().cpu().numpy()
        return dict(rank_protocol='fixed-rank-linear-v1',fast_beta=FAST_BETA,slow_beta=SLOW_BETA,
                    rank_tie_seed=TIE_SEED,rank_shuffle_seed=SHUFFLE_SEED,
                    sampling_weight_sha256=self._last_weight_sha256,
                    progress_positive_fraction=float((progress[seen]>0).mean()) if seen.any() else 0.,
                    progress_ema_mean=float(progress[seen].mean()) if seen.any() else 0.,
                    progress_age_pearson=finite_correlation(progress[seen],ages[seen]),
                    progress_count_pearson=finite_correlation(progress[seen],counts[seen]),
                    measurement='observation-time fast/slow EMA proxy; feedback is post-update; rank_trace is pre-feedback')

    @torch.no_grad()
    def export_rank_trace(self,path):
        if not self._rank_trace:
            raise RuntimeError('No logged ranked selections')
        arrays={'steps':np.asarray([r['step'] for r in self._rank_trace],dtype=np.int64),
                'tie_priority':self._tie_priority.copy()}
        for name in ('fast','slow','scores','weights','observation_count','last_observed_step','masked'):
            arrays[name]=np.stack([r[name] for r in self._rank_trace])
        with open(path,'xb') as stream:
            np.savez_compressed(stream,**arrays)

    @torch.no_grad()
    def export_node_statistics(self,path):
        arrays=dict(node_id=np.arange(self.error_ema.numel(),dtype=np.int64),
                    nonself_in_degree=self.degree.cpu().numpy(),reconstruction_error_ema=self.error_ema.cpu().numpy(),
                    slow_reconstruction_error_ema=self.slow_error_ema.cpu().numpy(),
                    progress_score=np.maximum((self.slow_error_ema-self.error_ema).cpu().numpy(),0.),
                    observation_count=self.observation_count.cpu().numpy(),last_observed_step=self.last_observed_step.cpu().numpy(),
                    local_feature_density=self._density,density_layer=self._membership)
        with open(path,'xb') as stream:
            np.savez_compressed(stream,**arrays)
