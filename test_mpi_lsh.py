"""
MPI equivalence test for the federated LSH protocol (step 3).

Run under MPI (rank 0 = server, ranks 1..N = clients):
    mpirun -np 5 python test_mpi_lsh.py      # 4 clients
    mpirun -np 3 python test_mpi_lsh.py      # 2 clients

On rank 0 it compares the federated result against the centralized lsh_proto and
prints PASS/FAIL plus per-rank communication stats. Exits non-zero on failure.
"""

import sys

import numpy as np

from configs.params import Params
from data_io.comm import comm
from utils.protocols import lsh_proto, mpi_lsh_proto


def make_data(n=800, d=8, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(n // 2, d) + np.array([5] + [0] * (d - 1))
    b = rng.randn(n // 2, d) - np.array([5] + [0] * (d - 1))
    X = np.vstack([a, b])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def make_params(ncl, **kw):
    p = Params(k=2, dim=8, data_size=800, num_clients=ncl, d_prime=5,
               fixed=False, basis_method="random", eps=2.0)
    p.sigma_fraction = 10.0
    p.min_count_in_node = 20
    p.min_count_to_branch = 50
    p.tree_max_depth = 0
    p.seed = 7
    p.delay = 0.0
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def split(X, ncl):
    return [X[i::ncl] for i in range(ncl)]


def main():
    ncl = comm.world_size - 1
    if ncl < 1:
        if comm.rank == 0:
            print("Need >= 2 MPI processes (1 server + >= 1 client).")
        return 0

    X = make_data()
    failures = 0
    params = make_params(ncl)
    fed, stats = mpi_lsh_proto(split(X, ncl), params)
    if comm.rank == 0:
        ref, _ = lsh_proto([X], params)
        try:
            assert fed.shape == ref.shape, f"shape {fed.shape} != {ref.shape}"
            np.testing.assert_allclose(fed, ref, rtol=1e-9, atol=1e-9)
            print(f"[PASS] ncl={ncl} leaves={stats['num_leaves']} shape={fed.shape}")
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] ncl={ncl}: {e}")

    comm.comm.Barrier()
    print(f"  rank {comm.rank}: {comm.get_comm_stats()}")
    if comm.rank == 0:
        print("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
