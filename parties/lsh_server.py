"""Server side of the federated LSH prefix-tree protocol (sparse + vectorized).

The server aggregates the clients' sparse leaf histograms, builds per-node counts
ONLY for nodes that are ancestors of occupied leaves (so memory is O(#occupied),
never O(2^d')), adds the DP noise, prunes the tree, and forms the leaf centroids.

This mirrors FastLloyd's server (which sees noisy aggregates) and produces
results identical -- up to floating-point summation order -- to the centralized
`lsh_proto` under the same seed.
"""

from copy import copy
from time import perf_counter

import numpy as np

from configs import Params
from utils.LSHTree import leaf_sum_noise, leaves_of, node_count_noise, prune_tree
from parties.lsh_client import basis_calib_size
from utils.ortho_clustering import (
    compute_dp_sigmas_zcdp, dpsgd_calibrate_sigma, orthogonal_basis)


class LshServer:
    """Aggregation / noise / pruning server for the federated LSH protocol."""

    def __init__(self, params: Params):
        self.params = copy(params)
        n = params.data_size
        self.delta = 1.0 / (n * np.log(n))
        self.basis = None
        self.max_depth = None
        self.eps_basis = 0.0
        self.eps_agg = params.eps
        self.delta_agg = self.delta
        self.sigma_centers = 0.0
        self.sigma_count = 0.0
        self._leaf_prefixes = []
        # Wall-time (seconds) of the two basis phases, filled by build_basis. Split so the
        # timing harness can report the (dominant, precomputable) sigma calibration
        # separately from the actual basis construction (SGD / eigh / random draw).
        self.t_basis_calib = 0.0
        self.t_basis_build = 0.0

    def build_basis(self, subsample=None, moments=None):
        """Build the SimHash basis and calibrate the zCDP noise.

        Basis input depends on the method (set by the caller / mpi_lsh_proto):
          - random   : no data (seed only)
          - svd_pca  : `moments` = summed (d+1, d+1) augmented scatter -> exact
                       full-data PCA, no raw points communicated
          - dpsgd_pca: `subsample` = small pooled subsample (capped)
        """
        p = self.params
        if p.basis_method == "dpsgd_pca":
            assert p.eps > 0, "dpsgd_pca basis needs eps > 0"
            assert 0.0 < p.basis_epsilon < 1.0, "basis_epsilon must be in (0, 1)"
            self.eps_basis = p.basis_epsilon * p.eps
            delta_basis = p.basis_epsilon * self.delta
            self.eps_agg = p.eps - self.eps_basis
            self.delta_agg = self.delta - delta_basis
        else:
            self.eps_basis, delta_basis = 0.0, 0.0
            self.eps_agg, self.delta_agg = p.eps, self.delta

        self.t_basis_calib = 0.0
        if p.basis_method == "svd_pca":
            t0 = perf_counter()
            self.basis = self._svd_basis_from_moments(moments)
            self.t_basis_build = perf_counter() - t0
        elif p.basis_method == "dpsgd_pca":
            X = subsample if subsample is not None else np.empty((1, p.dim))
            # Split calibration from the SGD loop so each is timed on its own. full_n = full
            # dataset size: dpsgd_pca calibrates its noise w.r.t. the FULL dataset via
            # amplification by sub-sampling. Calibrate at the DETERMINISTIC public pool size
            # (basis_calib_size) -- not the realized Poisson count X.shape[0] -- so sigma is
            # data-independent and the memoized calibration hits across seeds (see
            # basis_calib_size). The SGD still runs on the realized pool X.
            sigma, self.t_basis_calib = dpsgd_calibrate_sigma(
                self.eps_basis, delta_basis, basis_calib_size(p), 256, p.basis_epochs,
                full_n=p.data_size)
            t0 = perf_counter()
            self.basis = orthogonal_basis(
                X, p.d_prime, method="dpsgd_pca", seed=p.seed,
                epsilon=self.eps_basis, delta=delta_basis,
                clip_norm=p.basis_clip_norm, data_fraction=1.0,
                epochs=p.basis_epochs, lr=p.basis_lr,
                full_n=p.data_size, sigma=sigma)
            self.t_basis_build = perf_counter() - t0
        else:  # random
            t0 = perf_counter()
            self.basis = orthogonal_basis(
                np.empty((1, p.dim)), p.d_prime, method="random", seed=p.seed)
            self.t_basis_build = perf_counter() - t0

        self.max_depth = min(p.tree_max_depth or p.d_prime, self.basis.shape[1])
        if p.eps > 0:
            self.sigma_centers, self.sigma_count = compute_dp_sigmas_zcdp(
                self.eps_agg, self.delta_agg, p.sigma_fraction,
                count_levels=self.max_depth + 1)
        else:
            self.sigma_centers, self.sigma_count = 0.0, 0.0
        return self.basis

    def _svd_basis_from_moments(self, moments):
        """Top-d' principal components from the summed augmented scatter matrix.

        `moments` = sum_i [X_i | 1]^T [X_i | 1]. The top eigenvectors of the
        centered scatter `G - sum sum^T / n` are exactly the right singular
        vectors of the centered pooled data (PCA), so this equals `svd_pca_basis`
        on the full data -- but computed from O(d^2) aggregates, not raw points.
        """
        d = self.params.dim
        M = np.asarray(moments, dtype=float)
        G = M[:d, :d]
        s = M[:d, d]
        n = M[d, d]
        cov = G - np.outer(s, s) / n
        cov = (cov + cov.T) / 2                      # symmetrize for eigh
        w, V = np.linalg.eigh(cov)                   # ascending eigenvalues
        d_eff = min(self.params.d_prime, d)
        top = np.argsort(w)[::-1][:d_eff]            # largest-variance directions
        return V[:, top]

    def aggregate_and_prune(self, client_hists):
        """Count round: merge sparse client histograms, add noise, prune.

        Args:
            client_hists: list of (2, m_i) int arrays [leaf ids; counts].

        Returns:
            leaf_ranges_sorted: (L, 2) int leaf-id ranges, ascending by lo
                (what the clients need for their sums).
            order: permutation mapping the sorted ranges back to the canonical
                leaf order (so centroids match the centralized tree's order).
        Also stores leaf prefixes / noisy counts (canonical order) on self.
        """
        p = self.params
        dp = self.basis.shape[1]

        # --- merge sparse leaf counts across clients -> global occupied hist ---
        uid_parts = [h[0] for h in client_hists if h.shape[1] > 0]
        cnt_parts = [h[1] for h in client_hists if h.shape[1] > 0]
        if uid_parts:
            all_uids = np.concatenate(uid_parts)
            all_cnts = np.concatenate(cnt_parts)
            gids, inv = np.unique(all_uids, return_inverse=True)
            gcounts = np.zeros(len(gids), dtype=np.int64)
            np.add.at(gcounts, inv, all_cnts)
        else:
            gids = np.empty(0, dtype=np.int64)
            gcounts = np.empty(0, dtype=np.int64)

        # --- range-count oracle over the sorted aggregated leaf ids -----------
        # O(#occupied) memory: a node's count is the #ids in its contiguous
        # leaf-id range (binary search + prefix sums), like the centralized tree.
        # Avoids the per-level Python dicts (~2^d' nodes), which OOMed rank 0.
        order = np.argsort(gids, kind="stable")
        sgids = gids[order]
        cum = np.concatenate(([0], np.cumsum(gcounts[order].astype(np.int64))))
        seed, sigma = p.seed, self.sigma_count

        def get_count(prefix):
            L = len(prefix)
            if L == 0:
                lo, hi = 0, 1 << dp
            else:
                v = int(prefix, 2)
                shift = dp - L
                lo, hi = v << shift, (v + 1) << shift
            a = int(np.searchsorted(sgids, lo))
            b = int(np.searchsorted(sgids, hi))
            return int(cum[b] - cum[a]) + node_count_noise(seed, prefix, sigma)

        levels = prune_tree(get_count, self.max_depth,
                            p.min_count_to_branch, p.min_count_in_node)
        leaf_prefixes = leaves_of(levels)          # canonical order

        ranges = np.empty((len(leaf_prefixes), 2), dtype=np.int64)
        noisy_counts = np.empty(len(leaf_prefixes))
        for i, pref in enumerate(leaf_prefixes):
            L = len(pref)
            if L == 0:
                lo, hi = 0, 1 << dp
            else:
                v = int(pref, 2)
                shift = dp - L
                lo, hi = v << shift, (v + 1) << shift
            ranges[i] = (lo, hi)
            noisy_counts[i] = get_count(pref)

        self._leaf_prefixes = leaf_prefixes
        self._noisy_counts = noisy_counts

        # clients need ascending, non-overlapping ranges for their binary search
        order = np.argsort(ranges[:, 0], kind="stable")
        return ranges[order], order

    def centroids(self, summed_sorted, order):
        """Sum round: un-sort the gathered leaf sums, add noise, divide by count.

        `summed_sorted` is (L, dim) aligned to the sorted ranges; `order` maps
        those back to the canonical leaf order. Returns (L, dim) centroids in the
        canonical order (matching the centralized tree).
        """
        dim = summed_sorted.shape[1]
        summed = np.empty_like(summed_sorted)
        summed[order] = summed_sorted               # back to canonical leaf order
        seed, sigma = self.params.seed, self.sigma_centers
        out = np.empty_like(summed)
        for i, pref in enumerate(self._leaf_prefixes):
            noisy_sum = summed[i] + leaf_sum_noise(seed, pref, sigma, dim)
            out[i] = noisy_sum / max(self._noisy_counts[i], 1.0)
        return out
