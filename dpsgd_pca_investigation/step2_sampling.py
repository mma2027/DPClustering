"""
Step 2 — Effect of the training subsample size on basis quality (still non-private).

The basis is LEARNED on a fraction of the data but its EVR is always measured on the
FULL data (the deployment target). This tests how many points PCA estimation needs.
We also include the production cap BASIS_MAX_SUBSAMPLE=2000 (parties/lsh_client.py) as
an absolute-count point, because that is what the real pipeline actually feeds dpsgd_pca.

Outputs:
  results/step2_sampling.csv
  console table: efficiency = EVR(dpsgd on subsample) / EVR_pca(full)
"""

import csv
import os

import numpy as np

from common import (load_dataset, VarianceOracle, dpsgd_pca_basis_instrumented,
                    subsample, svd_pca_basis, RESULTS_DIR)

DATASETS = ["mnist784", "glove100"]
LR = {"mnist784": 0.01, "glove100": 0.05}
D_PRIMES = [5, 10, 20, 50]
FRACTIONS = [0.10, 0.25, 0.50, 1.0]
ABS_COUNTS = [2000]          # production cap (BASIS_MAX_SUBSAMPLE)
EPOCHS = 10
BATCH = 256
SEEDS = [0, 1, 2]


def run_subsample(X, orc, d_prime, Xsub, lr):
    """Mean efficiency over seeds for dpsgd learned on Xsub, plus SVD-on-subsample."""
    effs = []
    for s in SEEDS:
        W, _ = dpsgd_pca_basis_instrumented(
            Xsub, d_prime, clip_norm=None, sigma=0.0, lr=lr,
            batch_size=min(BATCH, len(Xsub)), epochs=EPOCHS, seed=s)
        effs.append(orc.evr(W) / orc.evr_pca(d_prime))
    # SVD computed on the same subsample (data-dependent oracle on a sample)
    svd_eff = orc.evr(svd_pca_basis(Xsub, d_prime)) / orc.evr_pca(d_prime)
    return float(np.mean(effs)), float(np.std(effs)), svd_eff


def main():
    rows = []
    for name in DATASETS:
        X = load_dataset(name)
        orc = VarianceOracle(X)
        lr = LR[name]
        n = len(X)
        print(f"\n==== {name} (n={n}, d={X.shape[1]}, lr={lr}) ====")
        print("efficiency = EVR(dpsgd on subsample)/EVR_pca(full); [SVD on subsample] in brackets")
        # build subsample specs: fractions + absolute counts
        specs = [(f"{int(f*100)}%", f, None) for f in FRACTIONS]
        specs += [(f"{c}pts", c / n, c) for c in ABS_COUNTS]
        header = "d'  | " + " | ".join(lbl for lbl, _, _ in specs)
        print(header)
        for d_prime in D_PRIMES:
            cells = []
            for lbl, frac, _ in specs:
                Xsub, m = subsample(X, frac, seed=0)
                eff, sd, svd_eff = run_subsample(X, orc, d_prime, Xsub, lr)
                cells.append(f"{eff:.3f}[{svd_eff:.3f}]")
                rows.append(dict(dataset=name, d_prime=d_prime, subsample=lbl,
                                 n_sub=m, dpsgd_eff=eff, dpsgd_eff_std=sd,
                                 svd_eff=svd_eff, evr_pca=orc.evr_pca(d_prime)))
            print(f"{d_prime:>3} | " + " | ".join(cells))

    _write(os.path.join(RESULTS_DIR, "step2_sampling.csv"), rows)
    print("\nWrote results/step2_sampling.csv")


def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
