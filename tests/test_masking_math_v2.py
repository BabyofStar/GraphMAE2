"""CPU-only numerical tests; importing the models package is unnecessary."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

_path = Path(__file__).resolve().parents[1] / "models" / "masking_math.py"
_spec = importlib.util.spec_from_file_location("masking_math_cpu_v2", _path)
maths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(maths)


class MaskingMathTests(unittest.TestCase):
    def test_quotas_exhaustive_small_capacity_and_exact_total(self):
        for sizes in ([0], [1], [3, 3, 3, 2], [0, 2, 0, 5], [8, 1, 1, 1]):
            for k in range(sum(sizes) + 1):
                q = maths.proportional_quotas(sizes, k)
                self.assertEqual(sum(q), k)
                for quota, capacity in zip(q, sizes):
                    self.assertGreaterEqual(quota, 0)
                    self.assertLessEqual(quota, capacity)
                    if sum(sizes):
                        self.assertLessEqual(abs(quota - k * capacity / sum(sizes)), 1)

    def test_quota_rounding_is_stable_and_handles_empty_groups(self):
        self.assertEqual(maths.proportional_quotas([1, 1, 1, 1], 2), [1, 1, 0, 0])
        self.assertEqual(maths.proportional_quotas([0, 0], 0), [0, 0])
        for sizes, k in (([2, 3], 6), ([-1], 0), ([1.2], 0), ([2], True)):
            with self.assertRaises(ValueError):
                maths.proportional_quotas(sizes, k)

    def test_neighborhoods_match_shortest_path_distance_oracle(self):
        src = np.array([0, 0, 1, 2, 3, 3], dtype=np.int64)
        dst = np.array([1, 1, 2, 3, 3, 4], dtype=np.int64)
        first, second = maths.neighborhoods(6, src, dst)
        distance = np.full((6, 6), 99, dtype=int)
        np.fill_diagonal(distance, 0)
        for u, v in zip(src, dst):
            if u != v:
                distance[u, v] = distance[v, u] = 1
        for k in range(6):
            distance = np.minimum(distance, distance[:, k:k + 1] + distance[k:k + 1, :])
        for i in range(6):
            np.testing.assert_array_equal(first[i], np.flatnonzero(distance[i] == 1))
            np.testing.assert_array_equal(second[i], np.flatnonzero(
                (distance[i] > 0) & (distance[i] <= 2)))

    def test_bad_edges_are_rejected(self):
        for src, dst in (([0], [4]), ([0], [-1]), ([0], []), ([0.5], [1.0])):
            with self.assertRaises(ValueError):
                maths.neighborhoods(4, np.array(src), np.array(dst))

    def test_density_matches_full_cosine_oracle_on_tiny_graph(self):
        x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
        contexts = maths.neighborhoods(4, np.array([0, 1]), np.array([1, 2]))[1]
        expected_normalized = np.array([[1, 0], [0, 1],
                                        [1 / np.sqrt(2), 1 / np.sqrt(2)], [0, 0]])
        cosine = expected_normalized @ expected_normalized.T
        expected = [cosine[i, ids].mean() if ids.size else 0
                    for i, ids in enumerate(contexts)]
        for chunk_size in (1, 2, 256):
            np.testing.assert_allclose(maths.local_density(x, contexts, chunk_size),
                                       expected, atol=1e-7)

    def test_density_does_not_change_input_or_numpy_rng(self):
        x = np.array([[1, 2], [2, 1], [0, 0]], dtype=np.float32)
        original, before = x.copy(), np.random.get_state()
        contexts = maths.neighborhoods(3, np.array([0]), np.array([1]))[1]
        maths.local_density(x, contexts)
        after = np.random.get_state()
        np.testing.assert_array_equal(original, x)
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_density_rejects_nonfinite_features_and_bad_shapes(self):
        contexts = (np.array([1]), np.array([0]))
        for x in (np.array([[1.0], [np.nan]]), np.ones((3, 2)), np.ones((2, 0))):
            with self.assertRaises(ValueError):
                maths.local_density(x, contexts)

    def test_partitions_cover_all_nodes_and_ties_are_stable(self):
        scores = np.array([0.4, -0.1, 0.4, 0.2, 0.4, 0.0, -0.3])
        groups, membership = maths.density_partitions(scores, 4)
        np.testing.assert_array_equal(np.concatenate(groups), [6, 1, 5, 3, 0, 2, 4])
        np.testing.assert_array_equal(np.sort(np.concatenate(groups)), np.arange(7))
        for layer, ids in enumerate(groups):
            self.assertTrue(np.all(membership[ids] == layer))

    def test_partition_bins_are_capped_by_node_count(self):
        groups, _ = maths.density_partitions([1, 1], 4)
        self.assertEqual([len(g) for g in groups], [1, 1])
        with self.assertRaises(ValueError):
            maths.density_partitions([float("inf")])

    def test_context_summary_counts_visible_nodes_not_edges(self):
        first, second = maths.neighborhoods(5, np.array([0, 0, 1, 2]),
                                           np.array([1, 1, 2, 3]))
        stats = maths.context_summary([1, 4], first, second)
        self.assertEqual(stats["masked_one_hop_visible_mean"], 1.0)
        self.assertEqual(stats["masked_two_hop_visible_mean"], 1.5)
        self.assertEqual(stats["masked_one_hop_zero_visible_fraction"], 0.5)
        self.assertEqual(stats["masked_two_hop_zero_visible_fraction"], 0.5)

    def test_empty_and_fully_masked_contexts(self):
        first, second = maths.neighborhoods(2, np.array([0]), np.array([1]))
        self.assertIsNone(maths.context_summary([], first, second)[
            "masked_two_hop_visible_mean"])
        self.assertEqual(maths.context_summary([0, 1], first, second)[
            "masked_two_hop_zero_visible_fraction"], 1.0)

    def test_correlation_has_defined_constant_vector_behavior(self):
        self.assertIsNone(maths.finite_correlation([1, 1], [0, 1]))
        self.assertAlmostEqual(maths.finite_correlation([0, 1, 2], [2, 1, 0]), -1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
