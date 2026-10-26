"""Input masking for a fixed full-batch graph; no labels or feature values are read."""

import logging
import math

import torch
from torch import nn


class StructureInputMask(nn.Module):
    MODES = ("random", "structure_static", "structure_curriculum")

    def __init__(self, mode="random", alpha_max=0.5, warmup_ratio=0.2,
                 ramp_end_ratio=0.8, log_interval=100):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("Unknown input_mask_mode: %s" % mode)
        if not math.isfinite(alpha_max) or not 0 <= alpha_max <= 1:
            raise ValueError("structure_alpha_max must be finite and in [0, 1]")
        if not 0 <= warmup_ratio < ramp_end_ratio <= 1:
            raise ValueError("Require 0 <= warmup_ratio < ramp_end_ratio <= 1")
        if not isinstance(log_interval, int) or log_interval < 0:
            raise ValueError("mask_log_interval must be a nonnegative integer")
        self.mode = mode
        self.alpha_max = float(alpha_max)
        self.warmup_ratio = float(warmup_ratio)
        self.ramp_end_ratio = float(ramp_end_ratio)
        self.log_interval = log_interval
        # No parameters, no random initialization, and no additional checkpoint keys.
        self.register_buffer("degree", torch.empty(0), persistent=False)
        self.register_buffer("score_weights", torch.empty(0), persistent=False)
        self._graph_key = None
        self._step = None
        self._total_steps = None
        self.last_stats = None
        self._last_logged_step = None

    @property
    def enabled(self):
        return self.mode != "random" and self.alpha_max > 0

    @staticmethod
    def _key(graph):
        return id(graph), graph.num_nodes(), graph.num_edges()

    @staticmethod
    def _device(device):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    @torch.no_grad()
    def prepare(self, graph, device):
        if not self.enabled:
            return
        device = self._device(device)
        key = self._key(graph)
        if self._graph_key == key and self.score_weights.device == device:
            return
        n = graph.num_nodes()
        if n <= 0:
            raise ValueError("Input masking requires at least one node")
        src, dst = graph.edges()
        # Count incoming non-self edges, including any parallel edges.
        degree = torch.bincount(dst[src != dst], minlength=n).to(
            device=device, dtype=torch.float32)
        raw = torch.log1p(degree) + 1e-8
        self.degree = degree
        self.score_weights = raw / raw.sum()
        self._graph_key = key
        self._last_logged_step = None

    def set_progress(self, step, total_steps):
        if not isinstance(step, int) or not isinstance(total_steps, int):
            raise ValueError("Masking progress must use integer steps")
        if total_steps <= 0 or not 0 <= step < total_steps:
            raise ValueError("Require 0 <= step < total_steps and total_steps > 0")
        self._step, self._total_steps = step, total_steps

    def effective_alpha(self):
        if not self.enabled:
            return 0.0
        if self.mode == "structure_static":
            return self.alpha_max
        if self._step is None:
            raise RuntimeError("Curriculum requires set_progress(step, total_steps)")
        progress = self._step / max(self._total_steps - 1, 1)
        fraction = (progress - self.warmup_ratio) / (
            self.ramp_end_ratio - self.warmup_ratio)
        return self.alpha_max * min(max(fraction, 0.0), 1.0)

    @torch.no_grad()
    def permutation(self, graph, device, mask_rate):
        n = graph.num_nodes()
        if n <= 0 or not math.isfinite(mask_rate) or not 0 <= mask_rate <= 1:
            raise ValueError("Require a nonempty graph and mask_rate in [0, 1]")
        # Important: alpha=0 uses randperm, NOT multinomial with uniform weights.
        if not self.enabled:
            return torch.randperm(n, device=device)
        device = self._device(device)
        if self._graph_key != self._key(graph) or self.score_weights.device != device:
            raise RuntimeError("Call prepare(graph, x.device) for this fixed graph first")
        alpha = self.effective_alpha()
        k = int(mask_rate * n)
        if alpha == 0.0 or k == 0:
            perm = torch.randperm(n, device=device)
        else:
            weights = (1.0 - alpha) / n + alpha * self.score_weights
            masked = torch.multinomial(weights, k, replacement=False)
            keep = torch.ones(n, dtype=torch.bool, device=device)
            keep[masked] = False
            perm = torch.cat((masked, keep.nonzero(as_tuple=False).flatten()))
        self._record(perm, k, alpha)
        return perm

    @torch.no_grad()
    def _record(self, perm, k, alpha):
        if self.log_interval == 0 or self._step is None:
            return
        if self._last_logged_step == self._step:
            return
        if self._step % self.log_interval and self._step != self._total_steps - 1:
            return
        masked_degree = self.degree[perm[:k]]
        keep_degree = self.degree[perm[k:]]
        stats = {
            "step": self._step,
            "mode": self.mode,
            "alpha": alpha,
            "masked": k,
            "nodes": perm.numel(),
            "actual_mask_rate": k / perm.numel(),
            "masked_degree_mean": masked_degree.mean().item() if k else None,
            "kept_degree_mean": keep_degree.mean().item() if keep_degree.numel() else None,
            "masked_degree_q25_q50_q75": (
                torch.quantile(masked_degree, masked_degree.new_tensor([.25, .5, .75]))
                .cpu().tolist() if k else None
            ),
        }
        self.last_stats = stats
        self._last_logged_step = self._step
        logging.info("INPUT_MASK %s", stats)
