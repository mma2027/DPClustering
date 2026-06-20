"""Differentially private LSH prefix tree (SimHash tree).

A simple port of the structure in
https://github.com/google/differential-privacy/blob/main/learning/clustering/lsh_tree.py
adapted to this repo's orthogonal-projection basis.

Idea
----
Each point is hashed bit-by-bit by projecting it onto the columns of a basis
matrix and taking the sign -- exactly the SimHash mechanism from
https://github.com/google/differential-privacy/blob/main/learning/clustering/lsh.py
The projection vectors there (one per hash bit) are the *columns* of our
orthonormal basis ``Q`` of shape ``(d, max_hash_len)``: bit ``j`` of a point
``x`` is ``sign(x @ Q[:, j])``.

The tree groups points by their hash prefix. The root holds all points (empty
prefix); a node at depth ``j`` holds the points sharing its ``j``-bit prefix and
splits them on bit ``j`` into a "0" child (projection >= 0) and a "1" child
(projection < 0). Every node carries a *noisy* count of its points; a branch is
pruned -- never added to the tree -- when its noisy count falls below a
threshold. This keeps small (and therefore privacy-sensitive) groups from being
released or refined further.

The leaves of the surviving tree partition the data into buckets that can then
be turned into private cluster centers.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


def _prefix_seed(base_seed: int, hash_prefix: str) -> int:
    """Deterministic per-node seed so every node draws independent count noise.

    The leading "1" makes the mapping injective over all prefixes (including the
    empty root prefix and prefixes with leading zeros).
    """
    return base_seed + int("1" + hash_prefix, 2)


def node_count_noise(base_seed: int, hash_prefix: str, count_sigma: float) -> float:
    """Gaussian noise added to one node's count (scalar, sensitivity 1).

    Shared by the centralized tree and the federated server so both add the
    identical, prefix-seeded noise to a node's count.
    """
    rng = np.random.RandomState(_prefix_seed(base_seed, hash_prefix))
    return rng.normal(0, count_sigma)


def leaf_sum_noise(base_seed: int, hash_prefix: str, center_sigma: float,
                   dim: int) -> np.ndarray:
    """Gaussian noise added to one leaf's sum vector (the centroid numerator).

    Uses ``_prefix_seed + 1`` so it is independent of the count noise for the
    same node. Shared by the centralized tree and the federated server.
    """
    rng = np.random.RandomState(_prefix_seed(base_seed, hash_prefix) + 1)
    return rng.normal(0, center_sigma, size=dim)


def hash_leaf_ids(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Full d'-bit SimHash leaf id for every point (vectorized, no Python loop).

    Bit ``j`` is 1 when ``x . basis[:, j] < 0``, packed MSB-first so that a prefix
    of length ``L`` with integer value ``v`` covers the contiguous leaf-id range
    ``[v << (d'-L), (v+1) << (d'-L))``. Returns an ``(n,)`` int64 array in
    ``[0, 2^d')``.
    """
    X = np.asarray(X, dtype=float)
    basis = np.asarray(basis, dtype=float)
    d_prime = basis.shape[1]
    neg = (X @ basis) < 0                       # (n, d')  bit 1 where projection < 0
    weights = (1 << np.arange(d_prime - 1, -1, -1)).astype(np.int64)
    return neg.astype(np.int64) @ weights


@dataclass
class LSHTreeNode:
    """A surviving leaf of the LSH tree (only leaves are materialized).

    Attributes:
        hash_prefix: the bit string this leaf represents ("" for the root).
        points: the points hashing to ``hash_prefix`` (used for the centroid only).
        base_seed: base RNG seed; combined with the prefix for the centroid noise.
        private_count: the noisy count used for pruning (== get_count(prefix)).
        dim: ambient dimension (centroid noise size).
    """

    hash_prefix: str
    points: np.ndarray
    base_seed: int
    private_count: float
    dim: int

    @property
    def depth(self) -> int:
        return len(self.hash_prefix)

    def private_center(self, center_sigma: float) -> np.ndarray:
        """Noisy centroid: (sum + N(0, center_sigma^2 I)) / noisy_count.

        Divides the noisy sum by the (already noisy) ``private_count`` so the
        true count is never released; the count is clamped to >= 1 to avoid
        division blow-ups on tiny / negative noisy counts.
        """
        noisy_sum = self.points.sum(axis=0) + leaf_sum_noise(
            self.base_seed, self.hash_prefix, center_sigma, self.dim)
        return noisy_sum / max(self.private_count, 1.0)

    def __repr__(self) -> str:
        return f"{self.private_count:.0f}({self.hash_prefix or 'root'})"


# ---------------------------------------------------------------------------
# Pure pruning logic (no points, no noise) -- the single source of truth for
# how a (noisy) per-node count map turns into a pruned set of surviving leaves.
# Used by the centralized LSHTree below and by the federated server/clients,
# which obtain the per-node counts over the network instead of from local points.
# ---------------------------------------------------------------------------

def prune_tree(get_count: Callable[[str], float], d_prime: int,
               min_count_to_branch: float, min_count_in_node: float
               ) -> Dict[int, List[str]]:
    """Grow the pruned prefix tree level by level from a per-node count oracle.

    Args:
        get_count: maps a node's hash-prefix to its (noisy) count. Must be
            deterministic; it is queried for the root, every surviving node
            (branch test), and both children of every branched node (keep test).
        d_prime: maximum tree depth (number of hash bits).
        min_count_to_branch: only nodes whose count is >= this are expanded.
        min_count_in_node: a child is kept only if its count is >= this.

    Returns:
        ``{level: [surviving prefixes]}`` with the root ("") always at level 0,
        in the same order the centralized tree produces (parent order, "0"<"1").
    """
    levels: Dict[int, List[str]] = {0: [""]}
    level = 0
    while level < d_prime:
        nxt: List[str] = []
        for prefix in levels[level]:
            if get_count(prefix) >= min_count_to_branch:
                for ch in ("0", "1"):
                    child = prefix + ch
                    if get_count(child) >= min_count_in_node:
                        nxt.append(child)
        if not nxt:
            break
        level += 1
        levels[level] = nxt
    return levels


def leaves_of(levels: Dict[int, List[str]]) -> List[str]:
    """Surviving leaves: nodes with no surviving child one level below."""
    max_level = max(levels)
    leaves: List[str] = []
    for lvl in sorted(levels):
        parents_below = {p[:-1] for p in levels.get(lvl + 1, [])}
        for prefix in levels[lvl]:
            if lvl == max_level or prefix not in parents_below:
                leaves.append(prefix)
    return leaves


def prune_to_leaves(get_count: Callable[[str], float], d_prime: int,
                    min_count_to_branch: float, min_count_in_node: float
                    ) -> List[str]:
    """Convenience: surviving leaf prefixes for the given per-node count oracle."""
    return leaves_of(prune_tree(get_count, d_prime,
                                min_count_to_branch, min_count_in_node))


class LSHTree:
    """Container for a pruned LSH tree: surviving leaves + the level structure.

    Built by ``build_lsh_tree``. Memory is O(n*d): points are bucketed by their
    integer leaf id, so no per-node copies are kept across the tree's depth.

    Attributes:
        leaves: surviving leaf nodes (each with .points for its centroid).
        tree:   maps level index -> list of surviving prefixes at that level.
    """

    def __init__(self, levels: Dict[int, List[str]], leaves: List[LSHTreeNode],
                 min_count_to_branch: float, min_count_in_node: float):
        self.min_count_to_branch = min_count_to_branch
        self.min_count_in_node = min_count_in_node
        self.tree = levels
        self.leaves = leaves

    def private_centers(self, center_sigma: float) -> np.ndarray:
        """Noisy centroid of every leaf bucket: ``(num_leaves, dim)`` array."""
        return np.array([leaf.private_center(center_sigma) for leaf in self.leaves])

    def __repr__(self) -> str:
        per_level = {lvl: len(p) for lvl, p in self.tree.items()}
        return (f"LSHTree(levels={per_level}, leaves={len(self.leaves)}, "
                f"branch>={self.min_count_to_branch}, keep>={self.min_count_in_node})")


def build_lsh_tree(points: np.ndarray, basis: np.ndarray, max_depth: int,
                   min_count_to_branch: float, min_count_in_node: float,
                   count_sigma: float, base_seed: int = 0) -> LSHTree:
    """Grow the pruned LSH tree in O(n*d) memory (leaf-id bucketing, no copies).

    Each point is hashed to an integer leaf id once; node counts come from
    counting leaf ids in a contiguous range (binary search on the sorted ids),
    and leaf points are gathered by range -- so unlike a recursive split, the
    data is never copied once per level. Produces results identical (up to
    floating-point summation order) to the previous implementation.

    Args:
        points: ``(n, d)`` data.
        basis: ``(d, max_hash_len)`` projection vectors (columns).
        max_depth: maximum tree depth (clamped to the basis width).
        min_count_to_branch: noisy-count threshold for expanding a node.
        min_count_in_node: noisy-count threshold for keeping a child branch.
        count_sigma: Gaussian noise std added to every node's count.
        base_seed: base RNG seed for the per-node count noise.
    """
    X = np.asarray(points, dtype=float)
    basis = np.asarray(basis, dtype=float)
    n, dim = X.shape
    d_prime = basis.shape[1]
    max_depth = min(max_depth, d_prime)

    leaf_ids = hash_leaf_ids(X, basis)            # (n,) in [0, 2^d')
    order = np.argsort(leaf_ids, kind="stable")
    sorted_ids = leaf_ids[order]

    def _range(prefix):
        L = len(prefix)
        if L == 0:
            return 0, 1 << d_prime
        v = int(prefix, 2)
        shift = d_prime - L
        return v << shift, (v + 1) << shift

    def get_count(prefix):
        lo, hi = _range(prefix)
        true = int(np.searchsorted(sorted_ids, hi) - np.searchsorted(sorted_ids, lo))
        return true + node_count_noise(base_seed, prefix, count_sigma)

    levels = prune_tree(get_count, max_depth, min_count_to_branch, min_count_in_node)

    leaves = []
    for p in leaves_of(levels):
        lo, hi = _range(p)
        a = int(np.searchsorted(sorted_ids, lo))
        b = int(np.searchsorted(sorted_ids, hi))
        leaves.append(LSHTreeNode(
            hash_prefix=p, points=X[order[a:b]], base_seed=base_seed,
            private_count=get_count(p), dim=dim))
    return LSHTree(levels, leaves, min_count_to_branch, min_count_in_node)
