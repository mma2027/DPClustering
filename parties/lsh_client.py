"""Client side of the federated LSH prefix-tree protocol (sparse + vectorized).

Each client holds a data shard. Once the basis is broadcast, a client hashes its
points to integer leaf ids (vectorized) and contributes:
  - a SPARSE local leaf histogram (only occupied leaf ids + counts) in the count
    round -- communication is O(#occupied) not O(2^d'), and
  - per-surviving-leaf sums in the sum round, computed with a vectorized
    scatter-add (no Python per-point loop).

The server aggregates these, adds the DP noise, prunes, and forms the centroids;
see parties/lsh_server.py. (The earlier masked variant materialised dense 2^d'
vectors per client, which OOMed at large d'; this sparse design removes that.)
"""

from copy import copy

import numpy as np

from configs import Params
from data_io import unscale
from utils.LSHTree import hash_leaf_ids

# Cap on the total #points pooled at the server to fit a dpsgd_pca basis. Bounds
# the (raw-data) subsample communication regardless of n; svd_pca avoids the
# subsample entirely (it sends a d x d scatter matrix instead).
# Raised from 2000 -> 50000: the 2000 cap starved DP-SGD of steps (~80 steps at 10
# epochs), under-converging the basis (esp. glove100); the larger pool gives far more
# steps. Trade-off: the basis round now gathers more raw points (higher comm) and the
# DP-SGD training is heavier -- both fall inside the timed protocol for `timing` runs.
BASIS_MAX_SUBSAMPLE = 50000


def basis_subsample_rate(params):
    """Data-independent Poisson keep-probability ``gamma`` for the dpsgd_pca basis.

    Uses only the public total size ``params.data_size`` and the cap, so it leaks
    nothing about the data:

        gamma = min(basis_data_fraction, BASIS_MAX_SUBSAMPLE / N), clipped to <= 1.
    """
    N = params.data_size
    gamma = min(params.basis_data_fraction, BASIS_MAX_SUBSAMPLE / N) if N > 0 else 1.0
    return min(1.0, gamma)


def basis_calib_size(params, batch_size=256):
    """Deterministic pool size (from public params only) at which to calibrate the dpsgd
    basis noise: ``round(gamma * data_size)``, the *expected* Poisson pool.

    The realized pool is ``Binomial(N, gamma)``, so its size wobbles ~1% around this mean
    every seed. Calibrating sigma on the realized (data-dependent) count both (a) leaks a
    little through the noise level and (b) recomputes a ~identical sigma every seed. Keying
    the calibration on this public mean instead makes sigma fully data-independent (standard
    DP-SGD fixes T / q from public parameters, not realized counts) and lets the memoized
    calibration hit across all seeds. sigma is smooth in the pool size, so the <1% realized
    deviation changes it negligibly. Floored at batch_size so T >= epochs.
    """
    gamma = basis_subsample_rate(params)
    return max(batch_size, int(round(gamma * params.data_size)))


def poisson_subsample(values, index, params):
    """Per-shard Poisson sub-sample for the (dpsgd_pca) server-side basis.

    Each point is kept independently with probability ``gamma`` (see
    ``basis_subsample_rate``), so the pool across shards is a Poisson sub-sample of
    the full dataset at rate ``gamma``. This is exactly the sampling that makes the
    basis ``(eps, delta)``-DP **w.r.t. the full dataset** via amplification by
    sub-sampling (see ``utils.ortho_clustering._find_sigma_autodp_full``).

    Shared by ``LshClient.subsample`` (MPI/federated) and ``lsh_proto`` (centralized)
    so both paths realize the **identical** basis mechanism for a given seed and
    shard split: same per-shard RNG stream (``params.seed + index + 1``), same rate.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    gamma = basis_subsample_rate(params)
    rng = np.random.RandomState(params.seed + index + 1)
    keep = rng.random(n) < gamma            # Bernoulli(gamma) per point == Poisson sub-sampling
    return values[keep]


class LshClient:
    """A data-shard client: hashes locally, contributes sparse counts and sums."""

    def __init__(self, index: int, values, params: Params):
        self.params = copy(params)
        self.index = index
        v = np.asarray(values, dtype=float)
        self.values = unscale(v) if params.fixed else v
        self.basis = None
        self.leaf_ids = np.empty(0, dtype=np.int64)

    # --- step 1: basis ------------------------------------------------------
    def local_moments(self):
        """Augmented scatter matrix for federated SVD-PCA, shape (d+1, d+1).

        ``[X | 1]^T [X | 1] = [[X^T X, sum], [sum^T, n]]`` encodes the local Gram
        matrix, the sum vector, and the count in one small (d+1)x(d+1) matrix.
        Summed across clients it gives the exact pooled centered scatter, so the
        server recovers the full-data PCA basis without any raw points crossing
        the wire (O(d^2) communication instead of O(fraction * n * d)).
        """
        n = len(self.values)
        Xa = np.hstack([self.values, np.ones((n, 1))]) if n else np.zeros((0, self.values.shape[1] + 1))
        return Xa.T @ Xa

    def subsample(self):
        """A capped, random **Poisson** sub-sample of this shard for the dpsgd_pca basis.

        Delegates to the shared :func:`poisson_subsample` so the centralized
        ``lsh_proto`` realizes the identical basis mechanism (same rate, same
        per-shard RNG stream). ``gamma`` is data-independent and capped so the
        expected pool stays ``<= BASIS_MAX_SUBSAMPLE``.
        """
        return poisson_subsample(self.values, self.index, self.params)

    def set_basis(self, basis):
        """Receive the basis and hash every local point to its integer leaf id."""
        self.basis = np.asarray(basis, dtype=float)
        self.d_prime = self.basis.shape[1]
        if len(self.values):
            self.leaf_ids = hash_leaf_ids(self.values, self.basis)
        else:
            self.leaf_ids = np.empty(0, dtype=np.int64)

    # --- step 2: counts (sparse) -------------------------------------------
    def local_leaf_hist(self):
        """Sparse local histogram as a (2, m) int64 array: [leaf ids; counts]."""
        if len(self.leaf_ids) == 0:
            return np.empty((2, 0), dtype=np.int64)
        uids, counts = np.unique(self.leaf_ids, return_counts=True)
        return np.stack([uids, counts]).astype(np.int64)

    # --- step 3: sums (vectorized over surviving leaves) -------------------
    def local_leaf_sums(self, leaf_ranges):
        """Local sum of points per surviving leaf, shape (L, dim).

        `leaf_ranges` is an (L, 2) int array of half-open leaf-id ranges
        [lo, hi), ascending and non-overlapping (leaves are prefix-free). Each
        point is mapped to its leaf via binary search; points in pruned regions
        match no leaf and are dropped. Fully vectorized (scatter-add).
        """
        L = len(leaf_ranges)
        dim = self.values.shape[1]
        sums = np.zeros((L, dim))
        if L == 0 or len(self.leaf_ids) == 0:
            return sums
        starts = leaf_ranges[:, 0]
        ends = leaf_ranges[:, 1]
        cand = np.searchsorted(starts, self.leaf_ids, side="right") - 1
        valid = (cand >= 0) & (cand < L)
        within = np.zeros(len(self.leaf_ids), dtype=bool)
        within[valid] = self.leaf_ids[valid] < ends[cand[valid]]
        if within.any():
            np.add.at(sums, cand[within], self.values[within])
        return sums
