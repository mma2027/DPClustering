"""
Step 3 — Private regime (epsilon = 0.25 spent on the basis).

We sweep the moving parts the user flagged — clipping, epochs(=#steps), batch size,
and d' — measuring EVR on the full data. delta = 1/(n ln n). lr from Step 1
(mnist784=0.01, glove100=0.05). Noise is random so we average over seeds.

Sub-experiments (selectable via argv; default: all):
  clip   3A  clip_norm sweep            (d'=10, batch=256, epochs=10)
  epochs 3B  epochs/#steps sweep        (d'=10, batch=256, best clip)
  batch  3C  batch-size sweep           (d'=10, epochs=10, best clip)
  dprime 3D  d' sweep -> rule of thumb  (batch=256, epochs=10, best clip)

Baselines reported alongside: EVR_pca (optimal) and EVR_random (~d'/d).
Outputs: results/step3_<exp>.csv
"""

import csv
import os
import sys

import numpy as np

from common import (load_dataset, VarianceOracle, dpsgd_pca_basis_instrumented,
                    sigma_for_epsilon, RESULTS_DIR)

EPS = 0.25
LR = {"mnist784": 0.01, "glove100": 0.05}
# clip candidates span the measured per-sample grad-norm distributions
CLIPS = {"mnist784": [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0],
         "glove100": [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]}
BEST_CLIP = {"mnist784": 5.0, "glove100": 2.0}   # peak of 3A inverted-U (corrected sigma); see report
SEEDS = [0, 1, 2]


def _delta(n):
    return 1.0 / (n * np.log(n))


def _run(X, orc, d_prime, *, clip, batch, epochs, lr, n_priv):
    """Mean/std EVR over seeds at eps=0.25 (sigma calibrated to n_priv)."""
    sigma = sigma_for_epsilon(EPS, _delta(n_priv), n_priv, batch, epochs)
    evrs = []
    for s in SEEDS:
        W, _ = dpsgd_pca_basis_instrumented(
            X, d_prime, clip_norm=clip, sigma=sigma, lr=lr,
            batch_size=batch, epochs=epochs, seed=s)
        evrs.append(orc.evr(W))
    return float(np.mean(evrs)), float(np.std(evrs)), sigma


def exp_clip(name, X, orc, rows):
    lr, n = LR[name], len(X)
    print(f"\n[3A clip] {name}  EVR_pca(10)={orc.evr_pca(10):.4f} EVR_rand(10)={orc.evr_random(10):.4f}")
    for clip in CLIPS[name]:
        m, sd, sigma = _run(X, orc, 10, clip=clip, batch=256, epochs=10, lr=lr, n_priv=n)
        print(f"   clip={clip:>6}: EVR={m:.4f} ±{sd:.4f}  (sigma={sigma:.1f})")
        rows.append(dict(exp="clip", dataset=name, d_prime=10, clip=clip, batch=256,
                         epochs=10, sigma=sigma, evr=m, evr_std=sd,
                         evr_pca=orc.evr_pca(10), evr_random=orc.evr_random(10)))


def exp_epochs(name, X, orc, rows):
    lr, n, clip = LR[name], len(X), BEST_CLIP[name]
    print(f"\n[3B epochs] {name} clip={clip}  EVR_pca(10)={orc.evr_pca(10):.4f} EVR_rand={orc.evr_random(10):.4f}")
    for ep in [1, 2, 3, 5, 10, 20]:
        m, sd, sigma = _run(X, orc, 10, clip=clip, batch=256, epochs=ep, lr=lr, n_priv=n)
        print(f"   epochs={ep:>2}: EVR={m:.4f} ±{sd:.4f}  (sigma={sigma:.1f})")
        rows.append(dict(exp="epochs", dataset=name, d_prime=10, clip=clip, batch=256,
                         epochs=ep, sigma=sigma, evr=m, evr_std=sd,
                         evr_pca=orc.evr_pca(10), evr_random=orc.evr_random(10)))


def exp_batch(name, X, orc, rows):
    lr, n, clip = LR[name], len(X), BEST_CLIP[name]
    print(f"\n[3C batch] {name} clip={clip}")
    for bs in [64, 128, 256, 512, 1024]:
        m, sd, sigma = _run(X, orc, 10, clip=clip, batch=bs, epochs=10, lr=lr, n_priv=n)
        print(f"   batch={bs:>4}: EVR={m:.4f} ±{sd:.4f}  (sigma={sigma:.1f})")
        rows.append(dict(exp="batch", dataset=name, d_prime=10, clip=clip, batch=bs,
                         epochs=10, sigma=sigma, evr=m, evr_std=sd,
                         evr_pca=orc.evr_pca(10), evr_random=orc.evr_random(10)))


def exp_dprime(name, X, orc, rows):
    lr, n, clip = LR[name], len(X), BEST_CLIP[name]
    print(f"\n[3D d'] {name} clip={clip}  (best epochs from 3B applied)")
    best_ep = 10 if name == "mnist784" else 5
    for d_prime in [2, 5, 8, 10, 15, 20, 30, 50]:
        m, sd, sigma = _run(X, orc, d_prime, clip=clip, batch=256, epochs=best_ep, lr=lr, n_priv=n)
        epca, erand = orc.evr_pca(d_prime), orc.evr_random(d_prime)
        gain = (m - erand) / (epca - erand) if epca > erand else float("nan")
        print(f"   d'={d_prime:>2}: EVR={m:.4f}  pca={epca:.4f} rand={erand:.4f}  "
              f"captured_gain={gain:.2f}  beats_random={'Y' if m>erand else 'n'}")
        rows.append(dict(exp="dprime", dataset=name, d_prime=d_prime, clip=clip, batch=256,
                         epochs=best_ep, sigma=sigma, evr=m, evr_std=sd,
                         evr_pca=epca, evr_random=erand, captured_gain=gain))


EXPS = {"clip": exp_clip, "epochs": exp_epochs, "batch": exp_batch, "dprime": exp_dprime}


def main():
    which = sys.argv[1:] or list(EXPS.keys())
    for exp in which:
        rows = []
        for name in ["mnist784", "glove100"]:
            X = load_dataset(name)
            orc = VarianceOracle(X)
            EXPS[exp](name, X, orc, rows)
        path = os.path.join(RESULTS_DIR, f"step3_{exp}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
