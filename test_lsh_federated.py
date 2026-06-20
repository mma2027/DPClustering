"""
Tests: the sparse federated LSH server/client reproduce the centralized
`lsh_proto` (up to floating-point summation order), independent of shard count.

These run the protocol *in process* (no MPI): the driver sequences the same
server/client methods `mpi_lsh_proto` calls, replacing gather/bcast with plain
sums/concats. test_mpi_lsh.py exercises the real MPI path.

Run:
    python -m pytest test_lsh_federated.py -v
    # or
    python test_lsh_federated.py
"""

import unittest

import numpy as np

from configs.params import Params
from parties import LshClient, LshServer
from utils.protocols import lsh_proto


def run_lsh_inprocess(value_lists, params):
    """In-process simulation of the sparse three-step federated LSH protocol.

    Returns (centroids, num_leaves).
    """
    server = LshServer(params)
    clients = [LshClient(i, value_lists[i], params) for i in range(len(value_lists))]

    # Step 1 — basis at the server: seed (random) / scatter moments (svd_pca) /
    # capped subsample (dpsgd_pca).
    if params.basis_method == "random":
        basis = server.build_basis()
    elif params.basis_method == "svd_pca":
        basis = server.build_basis(moments=sum(c.local_moments() for c in clients))
    else:
        basis = server.build_basis(subsample=np.vstack([c.subsample() for c in clients]))
    for c in clients:
        c.set_basis(basis)

    # Step 2 — count round: gather sparse local histograms; server prunes.
    hists = [c.local_leaf_hist() for c in clients]
    ranges, order = server.aggregate_and_prune(hists)

    # Step 3 — sum round: gather per-leaf local sums; server forms centroids.
    summed = sum(c.local_leaf_sums(ranges) for c in clients)
    centers = server.centroids(summed, order)
    return centers, len(ranges)


def make_params(**kw):
    p = Params(k=2, dim=8, data_size=800, num_clients=4, d_prime=5,
               fixed=False, basis_method="random", eps=2.0)
    p.sigma_fraction = 10.0
    p.min_count_in_node = 20
    p.min_count_to_branch = 50
    p.tree_max_depth = 0
    p.seed = 7
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def make_data(n=800, d=8, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(n // 2, d) + np.array([5] + [0] * (d - 1))
    b = rng.randn(n // 2, d) - np.array([5] + [0] * (d - 1))
    X = np.vstack([a, b])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def split(X, n_clients):
    return [X[i::n_clients] for i in range(n_clients)]   # interleaved shards


class TestFederatedEqualsCentralized(unittest.TestCase):
    """Federated run == centralized lsh_proto (random basis)."""

    def _centralized(self, X, params):
        centers, _ = lsh_proto([X], params)   # lsh_proto pools value_lists itself
        return centers

    # Agreement is up to floating-point summation order only (per-shard sums then
    # combined vs one pass); leaf sets and noise are identical, so the tol is tiny.
    EXACT = dict(rtol=1e-11, atol=1e-12)

    def test_matches_lsh_proto_eps_positive(self):
        X = make_data()
        params = make_params(eps=2.0)
        ref = self._centralized(X, params)
        fed, _ = run_lsh_inprocess(split(X, 4), params)
        self.assertEqual(fed.shape, ref.shape)
        np.testing.assert_allclose(fed, ref, **self.EXACT)

    def test_matches_lsh_proto_eps_zero(self):
        X = make_data()
        params = make_params(eps=0.0)
        ref = self._centralized(X, params)
        fed, _ = run_lsh_inprocess(split(X, 4), params)
        np.testing.assert_allclose(fed, ref, **self.EXACT)

    def test_shard_count_does_not_change_result(self):
        X = make_data()
        ref = self._centralized(X, make_params(eps=2.0))
        for ncl in (2, 4, 8):
            p = make_params(eps=2.0, num_clients=ncl)
            fed, _ = run_lsh_inprocess(split(X, ncl), p)
            np.testing.assert_allclose(fed, ref, err_msg=f"ncl={ncl}", **self.EXACT)


class TestFederatedBasisMethods(unittest.TestCase):
    """svd_pca / dpsgd_pca run federated (basis from a server-side subsample)."""

    def test_svd_pca_runs_and_shapes(self):
        X = make_data()
        params = make_params(eps=2.0, basis_method="svd_pca")
        fed, n_leaves = run_lsh_inprocess(split(X, 4), params)
        self.assertEqual(fed.shape[1], 8)
        self.assertEqual(fed.shape[0], n_leaves)

    def test_dpsgd_pca_runs(self):
        X = make_data()
        params = make_params(eps=2.0, basis_method="dpsgd_pca", basis_epsilon=0.5)
        try:
            fed, n_leaves = run_lsh_inprocess(split(X, 4), params)
        except ModuleNotFoundError:
            self.skipTest("autodp not installed")
        self.assertEqual(fed.shape[0], n_leaves)
        self.assertFalse(np.any(np.isnan(fed)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
