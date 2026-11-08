"""Fixed random-group control; true density diagnostics retain their meaning."""
import hashlib
import time

import numpy as np
import torch

from .adaptive_masking import AdaptiveInputMask
from .masking_math import proportional_quotas


class RandomGroupInputMask(AdaptiveInputMask):
    MODE = "random_group_stratified"
    NEW_MODES = AdaptiveInputMask.NEW_MODES + (MODE,)
    MODES = AdaptiveInputMask.MODES + (MODE,)
    STRATIFIED = AdaptiveInputMask.STRATIFIED + (MODE,)

    def __init__(self, mode=MODE, random_group_seed=2026110301, **kwargs):
        if mode != self.MODE:
            raise ValueError("RandomGroupInputMask requires random_group_stratified")
        if (isinstance(random_group_seed, bool) or not isinstance(random_group_seed, int)
                or not 0 <= random_group_seed < 2**63):
            raise ValueError("random_group_seed must be an integer in [0, 2**63)")
        super().__init__(mode=mode, **kwargs)
        self.random_group_seed = random_group_seed
        self._sampling_groups = ()
        self._sampling_tensors = ()
        self._sampling_membership = None
        self._sampling_key = None

    @torch.no_grad()
    def prepare(self, graph, device, features=None):
        device = self._device(device)
        key = self._key(graph), str(device)
        if self._sampling_key == key:
            return
        super().prepare(graph, device, features)
        started = time.perf_counter()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_group_seed)
        order = torch.randperm(graph.num_nodes(), generator=generator, device="cpu").numpy()
        sizes = [len(group) for group in self._partitions]
        boundaries = np.cumsum(sizes)[:-1]
        self._sampling_groups = tuple(np.split(order, boundaries))
        self._sampling_membership = np.empty(graph.num_nodes(), dtype=np.int64)
        for i, group in enumerate(self._sampling_groups):
            self._sampling_membership[group] = i
        self._sampling_tensors = tuple(torch.as_tensor(group, dtype=torch.long, device=device)
                                       for group in self._sampling_groups)
        self._sampling_key = key
        self.preparation_seconds += time.perf_counter() - started

    def effective_alpha(self):
        return 0.0

    @torch.no_grad()
    def permutation(self, graph, device, mask_rate):
        if self._sampling_key != (self._key(graph), str(self._device(device))):
            raise RuntimeError("Call prepare for this fixed graph/device first")
        # Reuse the exact S sampler, with only the membership changed. Restore the
        # true density tensors before any feedback or diagnostic can observe them.
        density_tensors = self._partition_tensors
        try:
            self._partition_tensors = self._sampling_tensors
            return super().permutation(graph, device, mask_rate)
        finally:
            self._partition_tensors = density_tensors

    def group_plan(self, mask_rate):
        if self._sampling_key is None:
            raise RuntimeError("Call prepare before exporting groups")
        n = len(self._sampling_membership)
        sizes = [len(group) for group in self._sampling_groups]
        k = int(mask_rate * n)
        order = np.concatenate(self._sampling_groups).astype('<i8', copy=False)
        return {"random_group_seed": self.random_group_seed, "nodes": n,
                "group_sizes": sizes, "masked": k,
                "quotas": proportional_quotas(sizes, k),
                "ordered_group_nodes_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
                "fingerprint_encoding": "concatenated groups, little-endian int64 node IDs",
                "density_layer_semantics": "true local-feature-density partitions"}

    def extra_feedback(self, ids, keep):
        plan = self.group_plan(len(ids) / len(keep))
        # Count actual selections independently; do not infer them from the plan.
        return {"random_group_seed": self.random_group_seed,
                "sampling_group_order_sha256": plan["ordered_group_nodes_sha256"],
                "sampling_groups": [
                    {"group": i, "nodes": len(group),
                     "masked": int(np.count_nonzero(~keep[group]))}
                    for i, group in enumerate(self._sampling_groups)]}

    @torch.no_grad()
    def export_node_statistics(self, path):
        if self._sampling_key is None:
            raise RuntimeError("No prepared random groups to export")
        arrays = {
            "node_id": np.arange(self.error_ema.numel(), dtype=np.int64),
            "nonself_in_degree": self.degree.detach().cpu().numpy(),
            "reconstruction_error_ema": self.error_ema.detach().cpu().numpy(),
            "observation_count": self.observation_count.detach().cpu().numpy(),
            "local_feature_density": self._density,
            "density_layer": self._membership,
            "sampling_group": self._sampling_membership,
            "sampling_group_node_order": np.concatenate(self._sampling_groups),
            "random_group_seed": np.asarray(self.random_group_seed, dtype=np.int64),
        }
        with open(path, "xb") as stream:
            np.savez_compressed(stream, **arrays)
