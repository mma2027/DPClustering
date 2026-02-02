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


if __name__ == "__main__":
    run_tests()
