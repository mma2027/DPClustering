"""
Shared utilities for the DP-SGD PCA basis investigation.

We ISOLATE the basis-generation step from the LSH protocol and measure a single,
LSH-independent quantity: how much of the data's total variance a basis W explains,

    EVR(W) = trace(W^T C W) / trace(C),      C = covariance of the (centered) data.

This is exactly the objective DP-SGD PCA optimizes (max trace(W^T X^T X W)) and the
quantity true PCA maximizes, so it lets us compare the private basis to the optimum
without any clustering noise/quantization in the way.

Baselines for a given d':
  - PCA (optimal):  EVR = sum(top-d' eigenvalues) / sum(all eigenvalues)   [upper bound]
  - random basis:   EVR ~ d'/d in expectation                              [lower bound]

The DP-SGD PCA implementation here is a faithful, *vectorized* reimplementation of
utils.ortho_clustering.dpsgd_pca_basis (same per-sample gradient, same Frobenius
clipping, same Gaussian noise on the gradient sum, same QR re-orthonormalization),
with explicit toggles for clip_norm (None = off) and sigma (0.0 = off) and a returned
convergence trace so we can study each effect in isolation.
"""

import os
import sys
import time

import numpy as np

# Production code (privacy accounting + reference baselines)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_io.data_handler import load_txt, normalize  # noqa: E402
from utils.ortho_clustering import svd_pca_basis, random_orthogonal_basis  # noqa: E402
from autodp.mechanism_zoo import NoisySGD_Mechanism  # noqa: E402
from autodp.transformer_zoo import AmplificationBySampling  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading (cached, normalized exactly as the pipeline does: min-max to [-1,1])
# ---------------------------------------------------------------------------

def load_dataset(name):
    """Load and normalize a dataset to [-1,1]^d, cached as .npy for fast reuse."""
    cache = os.path.join(CACHE_DIR, f"{name}_norm.npy")
    if os.path.exists(cache):
        return np.load(cache)
    path = os.path.join(DATA_DIR, f"{name}.txt")
    t0 = time.time()
    try:
        import pandas as pd
        raw = pd.read_csv(path, sep=r"\s+", header=None).to_numpy(dtype=float)
    except Exception:
        raw = load_txt(path)
    X = normalize(raw, fixed=False).astype(np.float64)
    np.save(cache, X)
    print(f"[load] {name}: {X.shape} loaded+normalized in {time.time()-t0:.1f}s -> {cache}")
    return X


# ---------------------------------------------------------------------------
# Explained-variance metric and PCA reference
# ---------------------------------------------------------------------------

class VarianceOracle:
    """Precomputes the full-data covariance spectrum for a dataset once.

    Provides:
      - total_var: trace(C)
      - evr_pca(d'): optimal explained-variance ratio at d' (top-d' eigenvalues)
      - evr(W): explained-variance ratio of an arbitrary orthonormal basis W
      - evr_random(d'): expected random-basis EVR (averaged over seeds)
    """

    def __init__(self, X):
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        n, d = Xc.shape
        self.d = d
        # Covariance C = Xc^T Xc / n  (symmetric, d x d)
        self.C = (Xc.T @ Xc) / n
        evals = np.linalg.eigvalsh(self.C)        # ascending
        self.evals_desc = evals[::-1].copy()      # descending
        self.total_var = float(self.evals_desc.sum())

    def evr_pca(self, d_prime):
        d_eff = min(d_prime, self.d)
        return float(self.evals_desc[:d_eff].sum() / self.total_var)

    def evr(self, W):
        # trace(W^T C W) = sum_j w_j^T C w_j
        return float(np.einsum("ij,jk,ik->", W.T, self.C, W.T) / self.total_var)

    def evr_random(self, d_prime, n_seeds=5):
        vals = [self.evr(random_orthogonal_basis(self.d, d_prime, seed=s))
                for s in range(n_seeds)]
        return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Instrumented DP-SGD PCA (vectorized; faithful to production algorithm)
# ---------------------------------------------------------------------------

def dpsgd_pca_basis_instrumented(
    X, d_prime, *, clip_norm=None, sigma=0.0, lr=0.01, batch_size=256, epochs=10,
    seed=None, oracle=None, record_every=0,
):
    """Vectorized DP-SGD PCA with explicit clip/noise toggles.

    Args:
        X: (n, d) data the basis is LEARNED on (already centered internally).
        d_prime: number of components.
        clip_norm: per-sample Frobenius clip threshold; None disables clipping
                   (true non-private gradient).
        sigma: Gaussian noise multiplier on the gradient SUM (std = sigma*clip_norm
               per entry, matching production). 0.0 disables noise. Requires clip_norm
               set when > 0 (noise scale is sigma*clip_norm).
        lr: learning rate.
        batch_size, epochs: SGD schedule.
        seed: RNG seed (init + shuffles + noise).
        oracle: optional VarianceOracle; if given together with record_every>0,
                records EVR(W) on the oracle's full data every `record_every` steps.
        record_every: if >0, store a convergence trace (step, evr).

    Returns:
        (W, trace) where W is (d, d_eff) orthonormal and trace is a list of
        (step, evr) tuples (empty if record_every == 0).
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape
    d_eff = min(d_prime, d)

    Xc = X - X.mean(axis=0)
    W = random_orthogonal_basis(d, d_eff, seed=(seed if seed is not None else 42))

    noise_std = (sigma * clip_norm) if (sigma > 0 and clip_norm is not None) else 0.0
    trace = []
    step = 0
    for _ in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            Xb = Xc[idx[start:start + batch_size]]
            b = len(Xb)
            if b == 0:
                continue
            P = Xb @ W                                   # (b, d_eff)
            # Per-sample gradient g_i = -2 outer(x_i, p_i); ||g_i||_F = 2||x_i|| ||p_i||
            if clip_norm is not None:
                gnorm = 2.0 * np.linalg.norm(Xb, axis=1) * np.linalg.norm(P, axis=1)
                c = np.minimum(1.0, clip_norm / (gnorm + 1e-8))   # (b,)
                agg = -2.0 * (Xb * c[:, None]).T @ P              # (d, d_eff)
            else:
                agg = -2.0 * Xb.T @ P
            if noise_std > 0.0:
                agg = agg + rng.normal(0.0, noise_std, size=W.shape)
            noisy_grad = agg / b
            W = W - lr * noisy_grad
            W, _ = np.linalg.qr(W)
            step += 1
            if record_every and oracle is not None and (step % record_every == 0):
                trace.append((step, oracle.evr(W)))

    return W[:, :d_eff], trace


def _calibrate_sigma(cost_fn, epsilon, lo=0.05, hi=5000.0, iters=50):
    """Binary-search the smallest sigma with cost_fn(sigma) <= epsilon."""
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if cost_fn(mid) > epsilon:
            lo = mid
        else:
            hi = mid
    return hi


def sigma_for_epsilon(epsilon, delta, n, batch_size, epochs):
    """Smallest Gaussian sigma for (epsilon, delta)-DP DP-SGD on n points.

    Uses autodp's NoisySGD_Mechanism = T-fold composition of the *Poisson-subsampled*
    Gaussian (rate q = batch/n), i.e. proper privacy AMPLIFICATION BY SUBSAMPLING.

    NOTE: this REPLACES utils.ortho_clustering._find_sigma_autodp, which calls a
    nonexistent rdp_bank.RDP_gaussian_subsampled, silently falls back to the
    unamplified bound T*q*RDP_gaussian, and returns sigma ~20x too large.
    """
    T = epochs * max(1, n // batch_size)
    q = min(1.0, batch_size / n)

    def cost(sigma):
        return NoisySGD_Mechanism(prob=q, sigma=sigma, niter=T).get_approxDP(delta)

    return _calibrate_sigma(cost, epsilon)


def sigma_for_epsilon_full(epsilon, delta, m, N, batch_size, epochs):
    """Sigma s.t. learning on a size-m random subsample of an N-point dataset is
    (epsilon, delta)-DP **with respect to the full N-point dataset**.

    NOTE: this investigation prototype uses without-replacement amplification
    (PoissonSampling=False, remove_only) — a one-sided bound. The PRODUCTION version is
    `utils.ortho_clustering._find_sigma_autodp_full`, which uses **Poisson** outer
    sub-sampling (PoissonSampling=True) for a true **add/remove** guarantee. Prefer that.

    Composes the per-step subsampled Gaussian (inner mini-batch rate q=batch/m) and
    then applies the one-time amplification-by-subsampling at rate gamma=m/N — the
    accounting that lets a (forced) subsample spend less noise than calibrating epsilon
    to the subsample alone.
    """
    if m >= N:
        return sigma_for_epsilon(epsilon, delta, N, batch_size, epochs)
    T = epochs * max(1, m // batch_size)
    q = min(1.0, batch_size / m)
    amp = AmplificationBySampling(PoissonSampling=False)

    def cost(sigma):
        inner = NoisySGD_Mechanism(prob=q, sigma=sigma, niter=T, name="inner")
        inner.neighboring = "remove_only"  # required for without-replacement amplification
        # improved_bound_flag=False: the simpler (looser but fast) subsampling bound;
        # the tight bound is pathologically slow under repeated calibration queries.
        outer = amp.amplify(inner, prob=m / N, improved_bound_flag=False)
        return outer.get_approxDP(delta)

    return _calibrate_sigma(cost, epsilon)


def subsample(X, fraction, seed=0):
    """Return a random row-subsample of X (fraction in (0,1]) plus its size."""
    if fraction >= 1.0:
        return X, len(X)
    rng = np.random.RandomState(seed)
    m = max(1, int(round(fraction * len(X))))
    idx = rng.choice(len(X), size=m, replace=False)
    return X[idx], m
