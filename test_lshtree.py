"""
Unit tests for the LSH prefix tree and its shared pruning logic.

Focus of step 1: the pure `prune_to_leaves` / `prune_tree` helpers are the single
source of truth for turning a per-node (noisy) count map into surviving leaves,
shared by the centralized `LSHTree` and (later) the federated server/clients.

Run:
    python -m pytest test_lshtree.py -v
    # or
    python test_lshtree.py
"""

import unittest

import numpy as np

from utils.LSHTree import (
    _prefix_seed,
    build_lsh_tree,
    leaves_of,
    prune_to_leaves,
    prune_tree,
)
from utils.ortho_clustering import orthogonal_basis


def count_oracle_from_data(points, basis, count_sigma, base_seed):
    """Per-prefix noisy-count oracle that mirrors LSHTreeNode.get_private_count.

    This is exactly what the federated server reconstructs from the aggregated
    leaf histogram: count(prefix) = (#points whose hash starts with prefix)
    + N(0, count_sigma^2) seeded deterministically by the prefix. A node's points
    are precisely those whose first len(prefix) hash bits equal prefix, so this
    reproduces the centralized node counts without building any node objects.
    """
    bits = np.where(points @ basis >= 0, "0", "1")          # (n, d') sign bits
    hashes = ["".join(row) for row in bits]

    def get_count(prefix):
        true = sum(1 for h in hashes if h.startswith(prefix))
        rng = np.random.RandomState(_prefix_seed(base_seed, prefix))
        return true + rng.normal(0, count_sigma)

    return get_count


class TestPruneToLeavesHandcrafted(unittest.TestCase):
    """Pruning behaves correctly on hand-specified count maps."""

    def test_basic_branch_and_keep(self):
        counts = {"": 100, "0": 60, "1": 40,
                  "00": 50, "01": 10, "10": 30, "11": 10}
        leaves = prune_to_leaves(counts.__getitem__, d_prime=2,
                                 min_count_to_branch=20, min_count_in_node=20)
        # "01" and "11" pruned (count 10 < 20); their parents are leaves only if
        # they have no surviving child -- "0" keeps "00", "1" keeps "10".
        self.assertEqual(leaves, ["00", "10"])

    def test_kept_but_unbranched_node_is_a_leaf(self):
        # "0" is kept (>= min_node) but not branched (< min_branch) -> leaf.
        counts = {"": 100, "0": 15, "1": 80,
                  "10": 40, "11": 40}
        leaves = prune_to_leaves(counts.__getitem__, d_prime=2,
                                 min_count_to_branch=20, min_count_in_node=10)
        self.assertEqual(leaves, ["0", "10", "11"])

    def test_root_below_branch_threshold_gives_single_leaf(self):
        counts = {"": 5}
        leaves = prune_to_leaves(counts.__getitem__, d_prime=3,
                                 min_count_to_branch=20, min_count_in_node=1)
        self.assertEqual(leaves, [""])

    def test_max_depth_zero_returns_root(self):
        leaves = prune_to_leaves({"": 100}.__getitem__, d_prime=0,
                                 min_count_to_branch=0, min_count_in_node=0)
        self.assertEqual(leaves, [""])

    def test_no_pruning_full_binary_tree(self):
        # All counts huge, thresholds 0 -> full tree of depth 2 -> 4 leaves.
        counts = {p: 100 for p in
                  ["", "0", "1", "00", "01", "10", "11"]}
        levels = prune_tree(counts.__getitem__, 2, 0, 0)
        self.assertEqual(levels[2], ["00", "01", "10", "11"])
        self.assertEqual(leaves_of(levels), ["00", "01", "10", "11"])

    def test_leaves_partition_is_prefix_free(self):
        counts = {"": 100, "0": 60, "1": 40,
                  "00": 50, "01": 10, "10": 30, "11": 10}
        leaves = prune_to_leaves(counts.__getitem__, 2, 20, 20)
        # No leaf is a prefix of another (a valid partition of hashed space).
        for a in leaves:
            for b in leaves:
                if a is not b:
                    self.assertFalse(b.startswith(a))


class TestLSHTreeUsesSharedPruning(unittest.TestCase):
    """The centralized LSHTree selects exactly the leaves the helper computes."""

    def setUp(self):
        rng = np.random.RandomState(0)
        d = 8
        a = rng.randn(400, d) + np.array([5] + [0] * (d - 1))
        b = rng.randn(400, d) - np.array([5] + [0] * (d - 1))
        X = np.vstack([a, b])
        self.X = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.basis = orthogonal_basis(self.X, d_prime=5, method="random", seed=1)

    def test_tree_leaves_match_count_oracle(self):
        for sigma in (0.0, 5.0, 15.0):
            tree = build_lsh_tree(self.X, self.basis, max_depth=5,
                                  min_count_to_branch=50, min_count_in_node=20,
                                  count_sigma=sigma, base_seed=7)
            tree_leaves = [leaf.hash_prefix for leaf in tree.leaves]

            d_prime = self.basis.shape[1]
            get_count = count_oracle_from_data(self.X, self.basis, sigma, 7)
            helper_leaves = prune_to_leaves(get_count, d_prime,
                                            min_count_to_branch=50,
                                            min_count_in_node=20)
            self.assertEqual(tree_leaves, helper_leaves,
                             msg=f"mismatch at count_sigma={sigma}")

    def test_leaf_points_consistent_with_prefixes(self):
        # Each surviving leaf's retained points must all hash to its prefix.
        tree = build_lsh_tree(self.X, self.basis, max_depth=5,
                              min_count_to_branch=50, min_count_in_node=20,
                              count_sigma=0.0, base_seed=7)
        for leaf in tree.leaves:
            p = leaf.hash_prefix
            for pt in leaf.points:
                bits = "".join("0" if pt @ self.basis[:, j] >= 0 else "1"
                               for j in range(len(p)))
                self.assertEqual(bits, p)

    def test_determinism_same_seed(self):
        kw = dict(max_depth=5, min_count_to_branch=50, min_count_in_node=20,
                  count_sigma=10.0, base_seed=3)
        t1 = build_lsh_tree(self.X, self.basis, **kw)
        t2 = build_lsh_tree(self.X, self.basis, **kw)
        self.assertEqual([l.hash_prefix for l in t1.leaves],
                         [l.hash_prefix for l in t2.leaves])
        np.testing.assert_array_equal(t1.private_centers(1.0),
                                      t2.private_centers(1.0))

    def test_centers_shape_and_no_nan(self):
        tree = build_lsh_tree(self.X, self.basis, max_depth=5,
                              min_count_to_branch=50, min_count_in_node=20,
                              count_sigma=10.0, base_seed=3)
        centers = tree.private_centers(center_sigma=1.0)
        self.assertEqual(centers.shape, (len(tree.leaves), self.basis.shape[0]))
        self.assertFalse(np.any(np.isnan(centers)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
