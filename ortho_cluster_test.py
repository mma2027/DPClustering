"""
Test script for orthogonal projection clustering.

Runs ortho_assign across all accuracy datasets, varying d_prime and seed,
and saves timing and cluster distribution results to CSV.

Usage:
    python ortho_cluster_test.py
"""

import os
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from data_io.data_handler import load_txt, normalize
from utils.ortho_clustering import ortho_assign

# Datasets to test (same as FastLloyd accuracy experiments)
DATASETS = ["iris", "s1", "house", "adult", "lsun", "birch2", "wine", "yeast", "breast", "mnist"]
D_PRIMES = [1, 2, 3, 4, 5]
NUM_SEEDS = 10
OUTPUT_DIR = Path("ortho_results")


def run_tests():
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []

    for dataset in DATASETS:
        path = Path("data") / f"{dataset}.txt"
        if not path.is_file():
            print(f"  Skipping {dataset} (file not found)")
            continue

        values = load_txt(str(path))
        values = normalize(values)
        n, d = values.shape
        print(f"{dataset}: n={n}, d={d}")

        for d_prime in D_PRIMES:
            for seed in range(NUM_SEEDS):
                start = timer()
                labels = ortho_assign(values, d_prime, seed=seed)
                elapsed = timer() - start

                sizes = np.bincount(labels, minlength=2**d_prime)
                occupied = np.sum(sizes > 0)
                nonempty_sizes = sizes[sizes > 0]

                rows.append({
                    "dataset": dataset,
                    "n": n,
                    "d": d,
                    "d_prime": d_prime,
                    "seed": seed,
                    "num_clusters": 2**d_prime,
                    "num_occupied": occupied,
                    "elapsed": elapsed,
                    "cluster_size_min": nonempty_sizes.min(),
                    "cluster_size_max": nonempty_sizes.max(),
                    "cluster_size_std": nonempty_sizes.std(),
                })

            # Print summary for this d_prime
            recent = rows[-NUM_SEEDS:]
            avg_time = np.mean([r["elapsed"] for r in recent])
            avg_occupied = np.mean([r["num_occupied"] for r in recent])
            print(f"  d'={d_prime}: {2**d_prime} clusters, "
                  f"{avg_occupied:.1f} occupied, "
                  f"{avg_time:.6f}s avg")

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path} ({len(df)} rows)")


def get_basis(d, d_prime, seed=42):
    """Replicates the basis computation from ortho_assign so we can build
    test points with known projections."""
    d_eff = min(d_prime, d)
    rng = np.random.RandomState(seed)
    R = rng.randn(d, d_eff)
    Q, _, _ = np.linalg.svd(R, full_matrices=False)
    Q = Q / np.linalg.norm(Q, axis=0)
    return Q  # (d, d_eff), orthonormal columns, Q^T @ Q = I


def run_verification_tests():
    """Hand-checkable correctness tests for ortho_assign.

    Key idea: if point = Q @ s  where s is a sign vector (+1/-1),
    then projection = point @ Q = s^T @ Q^T @ Q = s^T  (since Q^T Q = I).
    So we know the exact projection, and therefore the exact label.
    """
    print("\n" + "=" * 60)
    print("  VERIFICATION TESTS")
    print("=" * 60)
    passed = 0
    failed = 0

    def check(name, expected, actual, condition):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        else:
            failed += 1
        print(f"    [{status}] {name}")
        print(f"           expected: {expected}   actual: {actual}")

    # ------------------------------------------------------------------
    # Test 1: Opposite points with d'=1 (two halves)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  Test 1: Opposite points, d'=1")
    print("-" * 60)
    print("  Setup: d=3, d'=1, seed=42")
    print("  d'=1 means one basis vector q splits space into 2 halves.")
    print("  Points along +q should get label 1, along -q should get label 0.")
    d, d_prime, seed = 3, 1, 42
    Q = get_basis(d, d_prime, seed)
    print(f"  Basis vector q = {Q[:, 0].round(4).tolist()}")
    p = Q[:, 0] * 10
    values = np.vstack([p, -p])
    print(f"  Point A (+10*q) = {values[0].round(4).tolist()}")
    print(f"  Point B (-10*q) = {values[1].round(4).tolist()}")
    labels = ortho_assign(values, d_prime, seed=seed)
    proj = values @ Q
    print(f"  Projections: A={proj[0, 0]:.4f}, B={proj[1, 0]:.4f}")
    print(f"  Signs (>=0): A={int(proj[0, 0] >= 0)}, B={int(proj[1, 0] >= 0)}")
    check("Point A (+q) -> label 1", expected=1, actual=labels[0], condition=labels[0] == 1)
    check("Point B (-q) -> label 0", expected=0, actual=labels[1], condition=labels[1] == 0)

    # ------------------------------------------------------------------
    # Test 2: Four quadrants with d'=2
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  Test 2: Four quadrants, d=4, d'=2")
    print("-" * 60)
    print("  Setup: d=4, d'=2, seed=42")
    print("  d'=2 gives 2 basis vectors -> 2^2 = 4 quadrants.")
    print("  point = s0*Q[:,0] + s1*Q[:,1]  =>  projection = [s0, s1]")
    print("  label = (s0>=0)*1 + (s1>=0)*2")
    d, d_prime, seed = 4, 2, 42
    Q = get_basis(d, d_prime, seed)
    print(f"  Q[:,0] = {Q[:, 0].round(4).tolist()}")
    print(f"  Q[:,1] = {Q[:, 1].round(4).tolist()}")
    combos =    [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
    expected =  [       3,        1,        2,        0]
    for (s0, s1), exp in zip(combos, expected):
        point = (s0 * Q[:, 0] + s1 * Q[:, 1]).reshape(1, -1)
        proj = point @ Q
        label = ortho_assign(point, d_prime, seed=seed)[0]
        print(f"  Point = {s0:+d}*Q[:,0] + {s1:+d}*Q[:,1] = {point[0].round(4).tolist()}")
        print(f"    projection = {proj[0].round(4).tolist()} -> signs = [{int(proj[0,0]>=0)}, {int(proj[0,1]>=0)}]")
        check(f"signs ({s0:+d},{s1:+d}) -> label {exp}", expected=exp, actual=label, condition=label == exp)

    # ------------------------------------------------------------------
    # Test 3: Eight octants with d'=3
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  Test 3: Eight octants, d=3, d'=3")
    print("-" * 60)
    print("  Setup: d=3, d'=3, seed=42")
    print("  d'=3 in 3D -> 2^3 = 8 octants. Each sign combo maps to a unique label.")
    print("  point = Q @ [s0,s1,s2] => projection = [s0,s1,s2]")
    print("  label = bit0*1 + bit1*2 + bit2*4 = the octant index itself")
    d, d_prime, seed = 3, 3, 42
    Q = get_basis(d, d_prime, seed)
    print(f"  Q[:,0] = {Q[:, 0].round(4).tolist()}")
    print(f"  Q[:,1] = {Q[:, 1].round(4).tolist()}")
    print(f"  Q[:,2] = {Q[:, 2].round(4).tolist()}")
    for i in range(8):
        signs_vec = np.array([(1 if (i >> j) & 1 else -1) for j in range(3)])
        point = (Q @ signs_vec).reshape(1, -1)
        proj = point @ Q
        label = ortho_assign(point, d_prime, seed=seed)[0]
        bits = [int(proj[0, j] >= 0) for j in range(3)]
        print(f"  Octant {i}: signs={signs_vec.tolist()} -> proj={proj[0].round(4).tolist()} -> bits={bits}")
        check(f"octant {i} -> label {i}", expected=i, actual=label, condition=label == i)

    # ------------------------------------------------------------------
    # Test 4: d_prime > d gets capped to d
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  Test 4: d_prime > d is capped")
    print("-" * 60)
    print("  Setup: d=2, d'=5, seed=42")
    print("  SVD can only produce min(d', d) = 2 orthogonal vectors from 2D data.")
    print("  So effective d' = 2, max possible clusters = 2^2 = 4.")
    values = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    print(f"  Points: {values.tolist()}")
    labels = ortho_assign(values, d_prime=5, seed=42)
    print(f"  Labels: {labels.tolist()}")
    print(f"  Unique labels: {sorted(np.unique(labels).tolist())}  (max possible: [0,1,2,3])")
    check("max label < 4", expected="< 4", actual=labels.max(), condition=labels.max() < 4)
    check("at most 4 unique labels", expected="<= 4", actual=len(np.unique(labels)), condition=len(np.unique(labels)) <= 4)

    # ------------------------------------------------------------------
    # Test 5: Identical points all get the same label
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  Test 5: Identical points -> same label")
    print("-" * 60)
    print("  Setup: 100 copies of [1, 1, 1], d'=2, seed=0")
    print("  All points are the same, so all projections are identical -> same label.")
    values = np.ones((100, 3))
    labels = ortho_assign(values, d_prime=2, seed=0)
    unique = np.unique(labels)
    print(f"  All labels: {labels[0]}  (unique: {unique.tolist()})")
    check("100 identical points -> 1 unique label", expected=1, actual=len(unique), condition=len(unique) == 1)

    # ------------------------------------------------------------------
    # Test 6: Single point returns shape (1,)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  Test 6: Single point")
    print("-" * 60)
    print("  Setup: 1 point [3, -2, 1], d'=2, seed=42")
    values = np.array([[3.0, -2.0, 1.0]])
    labels = ortho_assign(values, d_prime=2, seed=42)
    Q = get_basis(3, 2, seed=42)
    proj = values @ Q
    print(f"  Projection: {proj[0].round(4).tolist()}")
    print(f"  Label: {labels[0]}  shape: {labels.shape}")
    check("output shape is (1,)", expected="(1,)", actual=labels.shape, condition=labels.shape == (1,))

    # ------------------------------------------------------------------
    # Test 7: Deterministic (same seed -> same labels)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  Test 7: Determinism")
    print("-" * 60)
    values = np.array([[1, 2, 3], [4, 5, 6], [-1, -2, -3]], dtype=float)
    print(f"  Points: {values.tolist()}")

    print("  Run 1: d'=2, seed=99")
    l1 = ortho_assign(values, d_prime=2, seed=99)
    print(f"    labels = {l1.tolist()}")
    print("  Run 2: d'=2, seed=99 (same seed)")
    l2 = ortho_assign(values, d_prime=2, seed=99)
    print(f"    labels = {l2.tolist()}")
    check("same seed -> identical labels", expected=l1.tolist(), actual=l2.tolist(), condition=np.array_equal(l1, l2))

    print("  Run 3: d'=2, seed=100 (different seed)")
    l3 = ortho_assign(values, d_prime=2, seed=100)
    print(f"    labels = {l3.tolist()}")
    check("different seed -> different labels", expected=f"!= {l1.tolist()}", actual=l3.tolist(), condition=not np.array_equal(l1, l3))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed out of {passed + failed} checks")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    # --- Correctness verification ---
    ok = run_verification_tests()
    if not ok:
        print("\nVerification FAILED — skipping dataset tests.")
        exit(1)

    # --- Real dataset benchmarks ---
    run_tests()
