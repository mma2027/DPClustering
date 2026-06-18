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

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


class LSHHasher:
    """SimHash over a fixed basis matrix.

    Args:
        basis: ``(d, max_hash_len)`` array whose columns are the projection
               vectors (e.g. an orthonormal basis from
               ``utils.ortho_clustering.orthogonal_basis``). Column ``j`` defines
               hash bit ``j``.
    """

    def __init__(self, basis: np.ndarray):
        self.basis = np.asarray(basis, dtype=float)
        self.dim, self.max_hash_len = self.basis.shape

    def group_by_next_hash(self, points: np.ndarray,
                           hash_prefix: str = "") -> Dict[str, np.ndarray]:
        """Split ``points`` on the next hash bit after ``hash_prefix``.

        Returns a dict mapping the next hash character ("0" / "1") to the subset
        of points taking that bit. Mirrors SimHash.group_by_next_hash: "0" for a
        non-negative projection, "1" for a negative one.
        """
        depth = len(hash_prefix)
        if depth >= self.max_hash_len:
            raise ValueError(
                f"Hash prefix {hash_prefix!r} has length >= max hash length "
                f"({self.max_hash_len})"
            )
        projected = points @ self.basis[:, depth]
        return {
            "0": points[projected >= 0],
            "1": points[projected < 0],
        }


def _prefix_seed(base_seed: int, hash_prefix: str) -> int:
    """Deterministic per-node seed so every node draws independent count noise.

    The leading "1" makes the mapping injective over all prefixes (including the
    empty root prefix and prefixes with leading zeros).
    """
    return base_seed + int("1" + hash_prefix, 2)


@dataclass
class LSHTreeNode:
    """A node corresponding to a single hash prefix.

    Attributes:
        hash_prefix: the bit string this node represents ("" for the root).
        points: the (non-private) points hashing to ``hash_prefix``.
        hasher: the shared LSHHasher used to generate child splits.
        count_sigma: std dev of the Gaussian noise added to this node's count.
                     The count query has L2 sensitivity 1, so for an
                     (eps, delta)-DP count use ``sqrt(2 ln(1.25/delta)) / eps``.
        base_seed: base RNG seed; combined with the prefix for the count noise.
        private_count: noisy count of ``points`` (computed on init if not given).
    """

    hash_prefix: str
    points: np.ndarray
    hasher: LSHHasher
    count_sigma: float
    base_seed: int = 0
    private_count: Optional[float] = None

    def __post_init__(self):
        if self.private_count is None:
            self.private_count = self.get_private_count()

    def get_private_count(self) -> float:
        """Return (and cache) the noisy count of points in this node."""
        if self.private_count is not None:
            return self.private_count
        rng = np.random.RandomState(_prefix_seed(self.base_seed, self.hash_prefix))
        self.private_count = len(self.points) + rng.normal(0, self.count_sigma)
        return self.private_count

    @property
    def depth(self) -> int:
        return len(self.hash_prefix)

    def children(self) -> List["LSHTreeNode"]:
        """All children of this node (one per next hash bit), before pruning."""
        groups = self.hasher.group_by_next_hash(self.points, self.hash_prefix)
        return [
            LSHTreeNode(self.hash_prefix + ch, pts, self.hasher,
                        self.count_sigma, self.base_seed)
            for ch, pts in groups.items()
        ]

    def private_center(self, center_sigma: float) -> np.ndarray:
        """Noisy centroid of this node: (sum + N(0, center_sigma^2 I)) / count.

        Divides the noisy sum by the (already noisy) ``private_count`` so the
        true count is never released. ``private_count`` is clamped to >= 1 to
        avoid division blow-ups on tiny / negative noisy counts.
        """
        rng = np.random.RandomState(
            _prefix_seed(self.base_seed, self.hash_prefix) + 1
        )
        noisy_sum = self.points.sum(axis=0) + rng.normal(
            0, center_sigma, size=self.hasher.dim
        )
        return noisy_sum / max(self.private_count, 1.0)

    def __repr__(self) -> str:
        return f"{self.private_count:.0f}({self.hash_prefix or 'root'})"


class LSHTree:
    """LSH prefix tree built level by level, pruning low-count branches.

    Args:
        root: root node (whole dataset, empty prefix). Built via ``build`` below
              in the typical case.
        max_depth: maximum tree depth (also bounded by the basis width).
        min_count_to_branch: only nodes whose noisy count is at least this are
            expanded into children.
        min_count_in_node: a child is kept only if its noisy count is at least
            this; otherwise the branch is pruned.

    Attributes:
        tree: maps level index -> list of nodes at that level.
        leaves: nodes with no surviving children.
    """

    def __init__(self, root: LSHTreeNode, max_depth: int,
                 min_count_to_branch: float, min_count_in_node: float):
        self.min_count_to_branch = min_count_to_branch
        self.min_count_in_node = min_count_in_node
        max_depth = min(max_depth, root.hasher.max_hash_len)

        self.tree: Dict[int, List[LSHTreeNode]] = {0: [root]}
        level = 0
        while level < max_depth:
            next_level = self._next_level(self.tree[level])
            if not next_level:
                break
            level += 1
            self.tree[level] = next_level

        self.leaves = [
            node for nodes in self.tree.values()
            for node in nodes if self._is_leaf(node)
        ]

    def _next_level(self, level_nodes: List[LSHTreeNode]) -> List[LSHTreeNode]:
        """Branch eligible nodes and keep only children above the threshold."""
        children: List[LSHTreeNode] = []
        for node in level_nodes:
            if node.private_count >= self.min_count_to_branch:
                children.extend(node.children())
        return [c for c in children if c.private_count >= self.min_count_in_node]

    def _is_leaf(self, node: LSHTreeNode) -> bool:
        """A node is a leaf if no node one level below extends its prefix."""
        below = node.depth + 1
        if below > max(self.tree):
            return True
        return not any(
            child.hash_prefix[:-1] == node.hash_prefix
            for child in self.tree[below]
        )

    def private_centers(self, center_sigma: float) -> np.ndarray:
        """Noisy centroid of every leaf bucket: ``(num_leaves, dim)`` array."""
        return np.array([leaf.private_center(center_sigma) for leaf in self.leaves])

    def __repr__(self) -> str:
        per_level = {lvl: len(nodes) for lvl, nodes in self.tree.items()}
        return (f"LSHTree(levels={per_level}, leaves={len(self.leaves)}, "
                f"branch>={self.min_count_to_branch}, keep>={self.min_count_in_node})")


def build_lsh_tree(points: np.ndarray, basis: np.ndarray, max_depth: int,
                   min_count_to_branch: float, min_count_in_node: float,
                   count_sigma: float, base_seed: int = 0) -> LSHTree:
    """Convenience builder: wrap ``basis`` in a hasher and grow the pruned tree.

    Args:
        points: ``(n, d)`` data.
        basis: ``(d, max_hash_len)`` projection vectors (columns), e.g. the
               orthonormal basis from ``orthogonal_basis``.
        max_depth: maximum tree depth.
        min_count_to_branch: noisy-count threshold for expanding a node.
        min_count_in_node: noisy-count threshold for keeping a child branch.
        count_sigma: Gaussian noise std added to every node's count.
        base_seed: base RNG seed for the per-node count noise.
    """
    hasher = LSHHasher(basis)
    root = LSHTreeNode("", np.asarray(points, dtype=float), hasher,
                       count_sigma, base_seed)
    return LSHTree(root, max_depth, min_count_to_branch, min_count_in_node)
