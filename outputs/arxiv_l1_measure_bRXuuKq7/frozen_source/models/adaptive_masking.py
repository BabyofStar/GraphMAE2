"""v2: detached reconstruction difficulty and density-stratified input masks.

Legacy B0--B3 sampling is delegated to the unchanged v1.2 implementation.
All adaptive state is ephemeral: saved model weights do not resume training.
"""

import logging
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .input_masking import StructureInputMask
from .masking_math import (context_summary, density_partitions, finite_correlation,
                           local_density, neighborhoods, proportional_quotas)


class AdaptiveInputMask(StructureInputMask):
    DIFFICULTY = ("reconstruction_curriculum", "reconstruction_density_curriculum")
    STRATIFIED = ("density_stratified", "reconstruction_density_curriculum")
    NEW_MODES = ("reconstruction_curriculum", "density_stratified",
                 "reconstruction_density_curriculum")
    MODES = StructureInputMask.MODES + NEW_MODES

    def __init__(self, mode="random", alpha_max=0.5, warmup_ratio=0.2,
                 ramp_end_ratio=0.8, log_interval=100, shuffle_seed=20261025,
                 diagnostics=False, ema_beta=0.9, clip_quantile=0.95,
                 density_bins=4, density_chunk_size=256):
        if mode not in self.MODES:
            raise ValueError("Unknown input_mask_mode: " + str(mode))
        super().__init__(mode="random" if mode in self.NEW_MODES else mode,
                         alpha_max=alpha_max, warmup_ratio=warmup_ratio,
                         ramp_end_ratio=ramp_end_ratio, log_interval=log_interval,
                         shuffle_seed=shuffle_seed)
        if not math.isfinite(ema_beta) or not 0 <= ema_beta < 1:
            raise ValueError("difficulty_ema_beta must be finite and in [0, 1)")
        if not math.isfinite(clip_quantile) or not 0 < clip_quantile <= 1:
            raise ValueError("difficulty_clip_quantile must be in (0, 1]")
        for name, value in (("density_bins", density_bins),
                            ("density_chunk_size", density_chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(name + " must be a positive integer")
        self.mode, self.diagnostics = mode, bool(diagnostics)
        self.ema_beta, self.clip_quantile = float(ema_beta), float(clip_quantile)
        self.density_bins, self.density_chunk_size = density_bins, density_chunk_size
        self.register_buffer("error_ema", torch.empty(0), persistent=False)
        self.register_buffer("observation_count", torch.empty(0, dtype=torch.long),
                             persistent=False)
        self._adaptive_key = None
        self._feedback_step = None
        self._contexts = None
        self._density = None
        self._membership = None
        self._partitions = ()
        self._partition_tensors = ()
        self.preparation_seconds = 0.0
        self.feedback_records = 0
        self.last_feedback = None

    @property
    def enabled(self):
        return self.mode in self.STRATIFIED or super().enabled

    @property
    def requires_feedback(self):
        return self.mode in self.NEW_MODES or self.diagnostics

    @property
    def needs_features(self):
        return self.mode in self.STRATIFIED or self.diagnostics

    @torch.no_grad()
    def prepare(self, graph, device, features=None):
        if not self.requires_feedback:
            return super().prepare(graph, device)
        device = self._device(device)
        key = self._key(graph), str(device)
        if self._adaptive_key == key:
            return
        started = time.perf_counter()
        n = graph.num_nodes()
        if n <= 0:
            raise ValueError("A nonempty fixed full-batch graph is required")
        # Calls the original scorer (including private B3 RNG) for legacy modes.
        if self.mode not in self.NEW_MODES:
            super().prepare(graph, device)
        src, dst = graph.edges()
        self.degree = torch.bincount(dst[src != dst], minlength=n).to(
            device=device, dtype=torch.float32)
        self.error_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.observation_count = torch.zeros(n, dtype=torch.long, device=device)
        if self.needs_features:
            if features is None or tuple(features.shape[:1]) != (n,):
                raise ValueError("Density/diagnostics require original node features")
            self._contexts = neighborhoods(
                n, src.detach().cpu().numpy(), dst.detach().cpu().numpy())
            self._density = local_density(
                features.detach().to(device="cpu", dtype=torch.float32).numpy(),
                self._contexts[1], self.density_chunk_size)
            self._partitions, self._membership = density_partitions(
                self._density, self.density_bins)
            self._partition_tensors = tuple(torch.as_tensor(
                group, dtype=torch.long, device=device) for group in self._partitions)
        self._graph_key = self._key(graph)
        self._adaptive_key = key
        self._feedback_step = None
        self._last_logged_step = None
        self.last_feedback, self.feedback_records = None, 0
        self.preparation_seconds = time.perf_counter() - started

    def effective_alpha(self):
        if self.mode == "density_stratified":
            return 0.0
        return super().effective_alpha()

    @torch.no_grad()
    def difficulty_scores(self):
        if not self.error_ema.numel():
            raise RuntimeError("Call prepare before calculating difficulty")
        seen = self.observation_count > 0
        if not seen.any().item():
            return torch.ones_like(self.error_ema)
        observed = self.error_ema[seen]
        upper = torch.quantile(observed, self.clip_quantile)
        scores = torch.minimum(self.error_ema, upper).clamp_min(1e-8)
        # Previously unseen nodes get a neutral observed median, not zero weight.
        scores[~seen] = torch.minimum(observed.median().clamp_min(1e-8),
                                     upper.clamp_min(1e-8))
        return scores

    @torch.no_grad()
    def permutation(self, graph, device, mask_rate):
        if self.mode not in self.NEW_MODES:
            return super().permutation(graph, device, mask_rate)
        n = graph.num_nodes()
        if n <= 0 or not math.isfinite(mask_rate) or not 0 <= mask_rate <= 1:
            raise ValueError("Require nonempty graph and mask_rate in [0, 1]")
        device = self._device(device)
        if self._adaptive_key != (self._key(graph), str(device)):
            raise RuntimeError("Call prepare for this fixed graph/device first")
        alpha, k = self.effective_alpha(), int(mask_rate * n)
        if k == 0 or (self.mode not in self.STRATIFIED and alpha == 0):
            perm = torch.randperm(n, device=device)
        else:
            scores = self.difficulty_scores() if self.mode in self.DIFFICULTY else None
            if self.mode in self.STRATIFIED:
                groups = self._partition_tensors
                quotas = proportional_quotas([len(g) for g in groups], k)
            else:
                groups, quotas = (torch.arange(n, device=device),), [k]
            selections = []
            for group, quota in zip(groups, quotas):
                if quota == 0:
                    continue
                if alpha == 0:
                    local = torch.randperm(len(group), device=device)[:quota]
                else:
                    raw = scores[group]
                    weights = (1 - alpha) / len(group) + alpha * raw / raw.sum()
                    local = torch.multinomial(weights, quota, replacement=False)
                selections.append(group[local])
            masked = torch.cat(selections)
            keep = torch.ones(n, dtype=torch.bool, device=device)
            keep[masked] = False
            perm = torch.cat((masked, keep.nonzero(as_tuple=False).flatten()))
        self._record(perm, k, alpha)
        return perm

    @torch.no_grad()
    def observe_reconstruction(self, mask_nodes, target, reconstruction,
                               loss_name="sce", alpha_l=2):
        """Update only AFTER this mask has been sampled and its loss computed."""
        if not self.requires_feedback:
            return
        if self._adaptive_key is None or self._step is None:
            raise RuntimeError("Feedback requires prepare and set_progress")
        if self._feedback_step == self._step:
            raise RuntimeError("Exactly one feedback update per training step is supported")
        if (mask_nodes.ndim != 1 or mask_nodes.numel() == 0
                or mask_nodes.unique().numel() != mask_nodes.numel()
                or target.ndim != 2 or reconstruction.shape != target.shape
                or target.shape[0] != mask_nodes.numel()):
            raise ValueError("Feedback needs nonempty, unique masked nodes and paired features")
        if loss_name == "sce":
            a = F.normalize(target.detach(), p=2, dim=-1)
            b = F.normalize(reconstruction.detach(), p=2, dim=-1)
            # Clamp only the diagnostic score; the original training loss is untouched.
            error = (1 - (a * b).sum(dim=-1)).clamp_min(0).pow(alpha_l)
        elif loss_name == "mse":
            error = (target.detach() - reconstruction.detach()).square().mean(dim=-1)
        else:
            raise ValueError("Unsupported feedback loss: " + str(loss_name))
        if not torch.isfinite(error).all().item():
            raise ValueError("Nonfinite reconstruction difficulty")
        previous = self.observation_count[mask_nodes]
        old = self.error_ema[mask_nodes]
        self.error_ema[mask_nodes] = torch.where(
            previous > 0, self.ema_beta * old + (1 - self.ema_beta) * error, error)
        self.observation_count[mask_nodes] += 1
        self._feedback_step = self._step
        if not self.log_interval or (self._step % self.log_interval
                                     and self._step != self._total_steps - 1):
            return
        ids = mask_nodes.detach().cpu().numpy()
        degree = self.degree.detach().cpu().numpy()
        counts = self.observation_count.detach().cpu().numpy()
        ema = self.error_ema.detach().cpu().numpy()
        seen = counts > 0
        keep = np.ones(len(counts), dtype=bool)
        keep[ids] = False
        stats = {
            "step": self._step, "mode": self.mode, "alpha": self.effective_alpha(),
            "masked": len(ids), "nodes": len(counts),
            "actual_mask_rate": len(ids) / len(counts),
            "batch_reconstruction_error_mean": float(error.mean().item()),
            "observed_nodes": int(seen.sum()),
            "observation_count_min_mean_max": [int(counts.min()),
                                                float(counts.mean()), int(counts.max())],
            "degree_error_pearson_seen": finite_correlation(degree[seen], ema[seen]),
            "masked_degree_mean": float(degree[ids].mean()),
            "kept_degree_mean": float(degree[keep].mean()) if keep.any() else None,
        }
        if self._contexts is not None:
            stats.update(context_summary(ids, *self._contexts))
            stats["density_layers"] = [
                {"layer": i, "nodes": len(group),
                 "masked": int(np.count_nonzero(~keep[group])),
                 "observed": int(np.count_nonzero(seen[group])),
                 "error_ema_seen_mean": (float(ema[group][seen[group]].mean())
                                         if seen[group].any() else None)}
                for i, group in enumerate(self._partitions)]
        stats.update(self.extra_feedback(ids, keep))
        self.last_feedback = stats
        self.feedback_records += 1
        logging.info("INPUT_MASK_FEEDBACK %s", stats)

    def extra_feedback(self, ids, keep):
        return {}

    @torch.no_grad()
    def export_node_statistics(self, path):
        if not self.requires_feedback or self._adaptive_key is None:
            raise RuntimeError("No adaptive/diagnostic state to export")
        arrays = {
            "node_id": np.arange(self.error_ema.numel(), dtype=np.int64),
            "nonself_in_degree": self.degree.detach().cpu().numpy(),
            "reconstruction_error_ema": self.error_ema.detach().cpu().numpy(),
            "observation_count": self.observation_count.detach().cpu().numpy(),
        }
        if self._density is not None:
            arrays.update(local_feature_density=self._density,
                          density_layer=self._membership)
        with open(path, "xb") as stream:
            np.savez_compressed(stream, **arrays)
