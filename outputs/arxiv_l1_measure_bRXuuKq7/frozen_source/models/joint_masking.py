"""Joint-mask correlation at fixed density quotas and equal node marginals."""
import hashlib
import time

import numpy as np
import torch

from .adaptive_masking import AdaptiveInputMask
from .masking_math import proportional_quotas
from .joint_math import pairing,selected_pair_positions,SEMANTIC_SEED,RANDOM_SEED


class JointInputMask(AdaptiveInputMask):
    JOINT_MODES=('pair_uniform_reference','pair_semantic_antithetic','pair_random_antithetic','pair_semantic_comask')
    MODES=AdaptiveInputMask.MODES+JOINT_MODES
    NEW_MODES=AdaptiveInputMask.NEW_MODES+JOINT_MODES
    STRATIFIED=AdaptiveInputMask.STRATIFIED+JOINT_MODES

    def __init__(self,mode='pair_semantic_antithetic',**kwargs):
        if mode not in self.JOINT_MODES: raise ValueError('Require a joint-mask mode')
        super().__init__(mode=mode,**kwargs)
        if self.density_bins!=4: raise ValueError('Joint pilot uses four density strata')
        self._plan=None; self._pairs=(); self._partner=None; self._trace=[]
        self._reference_sampling=False; self._sampled_step=None

    def effective_alpha(self): return 0.

    @torch.no_grad()
    def prepare(self,graph,device,features=None):
        key=(self._key(graph),str(self._device(device)))
        if self._adaptive_key==key: return
        started=time.perf_counter(); super().prepare(graph,device,features)
        n=graph.num_nodes(); x=features.detach().to(device='cpu',dtype=torch.float32).numpy()
        normalized=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),np.float32(1e-12))
        src,dst=graph.edges(); edges=sorted({(min(int(u),int(v)),max(int(u),int(v))) for u,v in zip(src.cpu().tolist(),dst.cpu().tolist()) if u!=v})
        edge_set=set(edges)
        def cosine(u,v): return float(np.sum(normalized[u]*normalized[v],dtype=np.float64))
        candidates=[[] for _ in range(4)]
        for u,v in edges:
            if self._membership[u]==self._membership[v]:
                candidates[int(self._membership[u])].append((cosine(u,v),u,v))
        random=self.mode=='pair_random_antithetic'; seed=RANDOM_SEED if random else SEMANTIC_SEED
        plans=[]; tensors=[]; partner=np.full(n,-1,dtype=np.int64); from_matching=np.zeros(n,dtype=np.uint8)
        for layer,group in enumerate(self._partitions):
            generator=torch.Generator(device='cpu').manual_seed(seed+layer)
            fallback=group[torch.randperm(len(group),generator=generator,device='cpu').numpy()].tolist()
            pairs,singleton,matched=pairing(group.tolist(),[] if random else candidates[layer],fallback)
            for (u,v),flag in zip(pairs,matched):
                partner[u]=v; partner[v]=u; from_matching[u]=from_matching[v]=int(flag)
            plans.append(dict(layer=layer,nodes=len(group),pairs=[list(p) for p in pairs],singleton=singleton,
                              from_graph_matching=matched,pair_feature_cosine=[cosine(u,v) for u,v in pairs],
                              pair_is_graph_neighbor=[(min(u,v),max(u,v)) in edge_set for u,v in pairs]))
            tensors.append(torch.as_tensor(pairs,dtype=torch.long,device=self.error_ema.device).reshape(-1,2))
        if not any(len(tensor) for tensor in tensors):
            raise ValueError('Pair diagnostics require at least one complete pair')
        self._pairs=tuple(tensors); self._partner=partner; self._from_matching=from_matching
        self._plan=dict(protocol='equal-marginal-pair-v1',pairing='random' if random else 'feature_graph_matching',
                        pairing_seed=seed,nodes=n,density_bins=4,groups=plans,
                        marginal='quota/stratum_size for every node, including odd singleton',
                        features='original features; positive cosine on same-stratum undirected nonself edges; greedy disjoint matching; private random fallback')
        self._trace=[]; self._sampled_step=None; self.preparation_seconds=time.perf_counter()-started

    def _record(self,perm,k,alpha):
        if not self._reference_sampling: super()._record(perm,k,alpha)

    @torch.no_grad()
    def permutation(self,graph,device,mask_rate):
        device=self._device(device)
        if self._adaptive_key!=(self._key(graph),str(device)) or self._step is None: raise RuntimeError('Prepare and set progress first')
        if self._sampled_step==self._step: raise RuntimeError('One joint sample per epoch')
        n=graph.num_nodes(); k=int(mask_rate*n)
        if not 0<=mask_rate<=1: raise ValueError('Invalid mask rate')
        if self.mode=='pair_uniform_reference':
            old=self.mode
            try:
                self.mode='density_stratified'; self._reference_sampling=True
                perm=super().permutation(graph,device,mask_rate)
            finally:
                self.mode=old; self._reference_sampling=False
        else:
            quotas=proportional_quotas([len(x) for x in self._partitions],k); selections=[]
            for tensor,plan,q in zip(self._pairs,self._plan['groups'],quotas):
                count=len(tensor); size=plan['nodes']; singleton=plan['singleton']
                include=(int(torch.randint(size,(1,),device=device).item())<q) if singleton is not None else False
                order=torch.randperm(count,device=device).cpu().tolist()
                coins=torch.randint(2,(count,),device=device).cpu().tolist()
                positions=selected_pair_positions(count,order,coins,q-int(include),'co' if self.mode=='pair_semantic_comask' else 'anti')
                chosen=tensor.flatten()[torch.as_tensor(positions,dtype=torch.long,device=device)]
                if include: chosen=torch.cat((chosen,torch.tensor([singleton],dtype=torch.long,device=device)))
                selections.append(chosen)
            masked=torch.cat(selections); keep=torch.ones(n,dtype=torch.bool,device=device); keep[masked]=False
            perm=torch.cat((masked,keep.nonzero(as_tuple=False).flatten()))
        # Full mask trace permits independent per-epoch quota/coupling/count checks.
        selected=np.zeros(n,dtype=np.uint8); selected[perm[:k].cpu().numpy()]=1
        self._trace.append(dict(step=self._step,masked=selected))
        self._sampled_step=self._step; self._record(perm,k,0.)
        return perm

    def extra_feedback(self,ids,keep):
        pairs=np.concatenate([np.asarray(x['pairs'],dtype=np.int64).reshape(-1,2) for x in self._plan['groups']])
        flags=np.concatenate([np.asarray(x['from_graph_matching'],dtype=bool) for x in self._plan['groups']])
        mask=~keep; differing=mask[pairs[:,0]]!=mask[pairs[:,1]]
        seen_pair=mask[pairs].sum(axis=1)>0
        return dict(joint_protocol='equal-marginal-pair-v1',pair_disagreement_fraction=float(differing.mean()),
                    graph_matched_pair_fraction=float(flags.mean()),
                    graph_matched_pair_disagreement_fraction=float(differing[flags].mean()) if flags.any() else None,
                    selected_pair_has_visible_partner_fraction=float(differing[seen_pair].mean()) if seen_pair.any() else None,
                    pairing_seed=self._plan['pairing_seed'])

    def pair_plan(self):
        if self._plan is None: raise RuntimeError('Prepare pairing before export')
        return self._plan

    @torch.no_grad()
    def export_joint_trace(self,path):
        if not self._trace: raise RuntimeError('No logged masks')
        with open(path,'xb') as stream:
            np.savez_compressed(stream,steps=np.asarray([x['step'] for x in self._trace],dtype=np.int64),
                                masked=np.stack([x['masked'] for x in self._trace]))

    @torch.no_grad()
    def export_node_statistics(self,path):
        with open(path,'xb') as stream:
            np.savez_compressed(stream,node_id=np.arange(self.error_ema.numel(),dtype=np.int64),
                                nonself_in_degree=self.degree.cpu().numpy(),reconstruction_error_ema=self.error_ema.cpu().numpy(),
                                observation_count=self.observation_count.cpu().numpy(),local_feature_density=self._density,
                                density_layer=self._membership,pair_partner=self._partner,pair_from_graph_matching=self._from_matching)
