"""
Step 1 — Non-private confirmation + learning-rate selection.

Question: with NO privacy (no clipping, no noise) and the FULL dataset, does the
DP-SGD PCA iteration actually recover the true PCA subspace, i.e. does
EVR(dpsgd) -> EVR(pca)?  And which learning rate gets there?  We also check whether
large d' is intrinsically harder even without privacy.

Outputs:
  results/step1_lr_sweep.csv    final EVR for each (dataset, d', lr)
  results/step1_convergence.csv EVR vs step for the chosen lr
  console summary table (efficiency = EVR(dpsgd)/EVR(pca), 1.0 == optimal)
"""

import csv
import os

import numpy as np

from common import (load_dataset, VarianceOracle, dpsgd_pca_basis_instrumented, RESULTS_DIR)

DATASETS = ["mnist784", "glove100"]
D_PRIMES = [2, 5, 10, 20, 50]
LRS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
EPOCHS = 10
BATCH = 256


def main():
    rows = []
    conv_rows = []
    for name in DATASETS:
        X = load_dataset(name)
        orc = VarianceOracle(X)
        print(f"\n==== {name}  (n={X.shape[0]}, d={X.shape[1]}) ====")
        header = "d'  | EVR_pca | " + " | ".join(f"lr={lr}" for lr in LRS)
        print(header)
        for d_prime in D_PRIMES:
            evr_pca = orc.evr_pca(d_prime)
            effs = []
            for lr in LRS:
                W, _ = dpsgd_pca_basis_instrumented(
                    X, d_prime, clip_norm=None, sigma=0.0, lr=lr,
                    batch_size=BATCH, epochs=EPOCHS, seed=0)
                evr = orc.evr(W)
                eff = evr / evr_pca
                effs.append(eff)
                rows.append(dict(dataset=name, d_prime=d_prime, lr=lr,
                                 evr=evr, evr_pca=evr_pca, efficiency=eff))
            print(f"{d_prime:>3} | {evr_pca:.4f}  | " +
                  " | ".join(f"{e:.3f}" for e in effs))

        # Convergence trace at a representative d' with a good lr
        for d_prime in [10, 50]:
            W, trace = dpsgd_pca_basis_instrumented(
                X, d_prime, clip_norm=None, sigma=0.0, lr=0.1,
                batch_size=BATCH, epochs=EPOCHS, seed=0,
                oracle=orc, record_every=20)
            for step, evr in trace:
                conv_rows.append(dict(dataset=name, d_prime=d_prime, step=step,
                                      evr=evr, evr_pca=orc.evr_pca(d_prime)))

    _write(os.path.join(RESULTS_DIR, "step1_lr_sweep.csv"), rows)
    _write(os.path.join(RESULTS_DIR, "step1_convergence.csv"), conv_rows)
    print("\nWrote results/step1_lr_sweep.csv and results/step1_convergence.csv")


def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
