"""Deterministic CPU helpers for the three small-graph v2 experiments.

No labels, train/validation/test masks, RNG, or dense N-by-N matrices are used.
"""

import math

import numpy as np


def proportional_quotas(sizes, total):
    """Hamilton apportionment, with a stable layer-index tie break."""
    if (isinstance(total, bool) or not isinstance(total, (int, np.integer))
            or total < 0):
        raise ValueError("total must be a nonnegative integer")
    values = list(sizes)
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0
           for x in values):
        raise ValueError("Layer sizes must be nonnegative integers")
    n = sum(values)
    if total > n:
        raise ValueError("Cannot mask more nodes than exist")
    if n == 0:
        return [0] * len(values)
    # Integer arithmetic avoids rounding errors and respects layer capacities.
    quotas = [int(total * size // n) for size in values]
    order = sorted(range(len(values)),
                   key=lambda i: (-(total * values[i] % n), i))
    for i in order[:int(total - sum(quotas))]:
        quotas[i] += 1
    if sum(quotas) != total or any(q > s for q, s in zip(quotas, values)):
        raise RuntimeError("Invalid layer quotas")
    return quotas


def neighborhoods(num_nodes, src, dst):
    """Unique undirected 1-hop and <=2-hop contexts, excluding self loops.

    Contexts are calculated once, not resampled. Two-hop lists may use O(N^2)
    storage on very dense graphs; this implementation is scoped to Cora,
    Citeseer and PubMed, not arbitrary large-graph training.
    """
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer")
    src, dst = np.asarray(src), np.asarray(dst)
    if src.ndim != 1 or dst.ndim != 1 or src.shape != dst.shape:
        raise ValueError("Edges must be equal-length vectors")
    if not np.issubdtype(src.dtype, np.integer) or not np.issubdtype(dst.dtype, np.integer):
        raise ValueError("Edge indices must be integers")
    if src.size and (min(src.min(), dst.min()) < 0
                     or max(src.max(), dst.max()) >= num_nodes):
        raise ValueError("Edge endpoint out of range")
    adjacent = [set() for _ in range(num_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            adjacent[u].add(v)
            adjacent[v].add(u)
    first, second = [], []
    for i, neighbors in enumerate(adjacent):
        reached = set(neighbors)
        for j in neighbors:
            reached.update(adjacent[j])
        reached.discard(i)
        first.append(np.asarray(sorted(neighbors), dtype=np.int64))
        second.append(np.asarray(sorted(reached), dtype=np.int64))
    return tuple(first), tuple(second)


def local_density(features, contexts, chunk_size=256):
    """Mean feature cosine similarity over unique <=2-hop neighbors."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != len(contexts) or not x.shape[1]:
        raise ValueError("Features must have shape [nodes, positive feature count]")
    if (isinstance(chunk_size, bool) or not isinstance(chunk_size, int)
            or chunk_size <= 0):
        raise ValueError("chunk_size must be a positive integer")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = x / np.maximum(norms, np.float32(1e-12))
    result = np.zeros(x.shape[0], dtype=np.float64)
    for i, neighbors in enumerate(contexts):
        neighbors = np.asarray(neighbors, dtype=np.int64)
        if not neighbors.size:
            continue
        summed = 0.0
        for start in range(0, neighbors.size, chunk_size):
            rows = normalized[neighbors[start:start + chunk_size]]
            similarities = np.einsum("ij,j->i", rows, normalized[i], optimize=False)
            summed += float(similarities.sum(dtype=np.float64))
        result[i] = summed / neighbors.size
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite local density")
    return result


def density_partitions(scores, num_bins=4):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Density must be a finite, nonempty vector")
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError("num_bins must be a positive integer")
    # Ascending density; equal values are resolved by node index, without RNG.
    order = np.argsort(values, kind="stable")
    groups = tuple(np.array_split(order, min(num_bins, values.size)))
    membership = np.empty(values.size, dtype=np.int64)
    for i, group in enumerate(groups):
        membership[group] = i
    return groups, membership


def context_summary(masked, first, second):
    n = len(first)
    ids = np.asarray(masked, dtype=np.int64)
    if ids.ndim != 1 or len(second) != n or (ids.size and (
            ids.min() < 0 or ids.max() >= n or np.unique(ids).size != ids.size)):
        raise ValueError("Invalid masked nodes/contexts")
    visible = np.ones(n, dtype=bool)
    visible[ids] = False
    result = {}
    for name, contexts in (("one_hop", first), ("two_hop", second)):
        counts = np.asarray([np.count_nonzero(visible[contexts[i]]) for i in ids],
                            dtype=np.int64)
        result["masked_" + name + "_visible_mean"] = (
            float(counts.mean()) if ids.size else None)
        result["masked_" + name + "_zero_visible_fraction"] = (
            float(np.mean(counts == 0)) if ids.size else None)
    return result


def finite_correlation(a, b):
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        return None
    x, y = x - x.mean(), y - y.mean()
    divisor = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / divisor) if divisor > 0 else None
