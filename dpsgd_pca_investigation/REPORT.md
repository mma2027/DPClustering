# DP-SGD PCA basis investigation

Branch: `dpsgdpca-accuracy`. Scope: **isolate the basis-generation step from LSH** and
compare DP-SGD PCA to true (non-private) PCA purely by how much of the data variance the
basis explains. Datasets: `glove100` (n=400k, d=100) and `mnist784` (n=70k, d=784).

## Metric

For an orthonormal basis `W ∈ R^{d×d'}` we report the **explained-variance ratio**

> `EVR(W) = trace(Wᵀ C W) / trace(C)`,  with `C` the full-data covariance.

This is exactly the objective DP-SGD PCA maximizes and what true PCA optimizes, so it
compares the private basis to the optimum with no clustering noise in the way.

References per d':
- **PCA optimum** `EVR_pca(d') = Σ top-d' eigenvalues / Σ all eigenvalues` (upper bound).
- **Random basis** `EVR_rand(d') ≈ d'/d` (lower bound).
- We report **efficiency = EVR(method) / EVR_pca(d')** ∈ [~0, 1]; 1.0 == optimal.

Code: `dpsgd_pca_investigation/common.py` (cached loaders, `VarianceOracle`, a vectorized
instrumented DP-SGD PCA faithful to `utils/ortho_clustering.dpsgd_pca_basis` with explicit
clip/noise toggles). Run with the `fastlloyd` conda env.

---

## Step 0 — The spectra (why the two datasets behave differently)

| d' | mnist784 EVR_pca | mnist784 EVR_rand | glove100 EVR_pca | glove100 EVR_rand |
|----|------|------|------|------|
| 2  | 0.169 | 0.003 | 0.095 | 0.020 |
| 5  | 0.333 | 0.006 | 0.163 | 0.050 |
| 10 | 0.489 | 0.013 | 0.245 | 0.099 |
| 20 | 0.645 | 0.026 | 0.375 | 0.200 |
| 50 | 0.825 | 0.063 | 0.675 | 0.498 |

**Key structural fact.** mnist784 has a **concentrated** spectrum: PCA explains hugely more
than random (e.g. 49% vs 1.3% at d'=10), so a good basis is very valuable. glove100 has a
**flat** spectrum: random already captures ≈ d'/d, close to PCA (e.g. 24.5% vs 9.9% at
d'=10, and 67.5% vs 49.8% at d'=50). The PCA-over-random gap is the prize DP-SGD PCA is
fighting for — large on mnist784, small on glove100. This alone predicts that even a
*perfect* private PCA can only modestly beat random on glove100, and that DP noise erasing
a few points of EVR is fatal to the glove100 advantage but survivable on mnist784.

---

## Step 1 — Non-private confirmation + learning-rate selection

No clipping, no noise, full data, 10 epochs, batch 256. Efficiency = EVR(dpsgd)/EVR_pca.

**mnist784** (total var 211, large eigenvalues)

| d' | lr=.001 | .005 | .01 | .05 | .1 | .5 |
|----|------|------|------|------|------|------|
| 2  | 0.999 | 0.996 | 0.990 | 0.964 | 0.952 | 0.936 |
| 10 | 1.000 | 0.998 | 0.995 | 0.976 | 0.962 | 0.933 |
| 50 | 0.995 | 0.999 | 0.998 | 0.992 | 0.985 | 0.957 |

**glove100** (total var 2.2, tiny eigenvalues)

| d' | lr=.001 | .005 | .01 | .05 | .1 | .5 |
|----|------|------|------|------|------|------|
| 2  | 0.761 | 1.000 | 1.000 | 0.999 | 0.999 | 0.994 |
| 10 | 0.734 | 0.966 | 0.994 | 1.000 | 0.999 | 0.996 |
| 50 | 0.868 | 0.964 | 0.988 | 0.999 | 0.999 | 0.998 |

**Findings.**
1. **The algorithm is correct.** With the right lr, non-private DP-SGD PCA reaches
   **99–100% of the PCA optimum** at every d'. The gradient/QR iteration is subspace
   (orthogonal) iteration on `I + 2·lr·C` and converges to the true top-d' eigenspace.
2. **Large d' is NOT intrinsically hard.** Efficiency stays ≥0.98 up to d'=50 on both
   datasets. ⟹ any "large d' is bad" effect seen later must come from **privacy**, not
   the optimizer.
3. **lr must match the data scale.** The effective power-iteration step is `2·lr·λ`.
   mnist784 has large λ ⟹ small lr (≤0.01) is best; large lr injects minibatch-noise
   wobble. glove100 has tiny λ ⟹ needs larger lr (0.05–0.1) or it under-converges in 10
   epochs (lr=0.001 only reaches 0.72–0.87).
   - **Rule of thumb:** for data normalized to [-1,1], `lr=0.01` is the robust default
     (≥0.99 on both). If the data variance is small/flat, scale lr up (∝ 1/λ_max) or run
     more epochs. We use **lr=0.01 (mnist784)** and **lr=0.05 (glove100)** downstream.

---

## Step 2 — Effect of the training subsample (still non-private)

Basis learned on a subsample, EVR measured on full data. 10 epochs, batch 256, mean over
3 seeds. `efficiency = EVR(dpsgd on subsample)/EVR_pca(full)`; `[ ]` = SVD on same subsample.

**mnist784**

| d' | 10% | 25% | 50% | 100% | 2000 pts |
|----|------|------|------|------|------|
| 5  | 0.993 [0.998] | 0.992 [0.999] | 0.995 [1.000] | 0.993 [1.000] | 0.992 [0.993] |
| 20 | 0.994 [0.998] | 0.995 [0.999] | 0.997 [1.000] | 0.997 [1.000] | 0.974 [0.991] |
| 50 | 0.991 [0.998] | 0.997 [0.999] | 0.999 [1.000] | 0.998 [1.000] | 0.906 [0.992] |

**glove100**

| d' | 10% | 25% | 50% | 100% | 2000 pts |
|----|------|------|------|------|------|
| 5  | 0.971 [0.999] | 0.999 [1.000] | 1.000 [1.000] | 1.000 [1.000] | **0.381** [0.969] |
| 10 | 0.965 [0.998] | 0.995 [0.999] | 0.999 [1.000] | 1.000 [1.000] | **0.477** [0.951] |
| 50 | 0.961 [0.997] | 0.990 [0.999] | 0.997 [1.000] | 0.999 [1.000] | **0.795** [0.963] |

**Findings.**
1. **PCA estimation is statistically cheap.** SVD on a subsample is ≈optimal (0.95–1.0)
   *everywhere*, even at 2000 points. Direction estimation needs very few samples; 10% is
   already plenty (dpsgd 0.95–0.99). So sample size per se is **not** the bottleneck.
2. **dpsgd quality is governed by the NUMBER OF SGD STEPS = epochs·(n/batch), not by n.**
   Subsampling cuts steps, causing under-convergence. At the production cap of **2000
   points** (`BASIS_MAX_SUBSAMPLE`, `parties/lsh_client.py:26`) with 10 epochs you get only
   ~80 steps → glove100 collapses to **0.38–0.80** while SVD on the *same* 2000 points is
   0.95–0.97. Confirmed by holding the sample at 2000 and increasing epochs:

   | epochs (≈steps) | 10 (80) | 50 (400) | 100 (800) | 200 (1600) | 400 (3200) |
   |---|---|---|---|---|---|
   | glove100 d'=10 eff | 0.50 | 0.77 | 0.88 | 0.94 | 0.95 |

   i.e. the collapse is **under-iteration, fully recoverable** — it reaches the SVD/sample
   ceiling (~0.95) once enough steps are taken. Flat-spectrum glove100 needs more steps
   (small eigengaps ⟹ slow power iteration); concentrated mnist784 is far more forgiving.
3. **Implications for the pipeline.**
   - The 2000-point cap is fine *statistically* but starves the SGD of steps. Either (a)
     scale epochs to keep total steps ≈ constant (target ≥1500), e.g. `epochs ≈ max(10,
     1500·batch/n_sub)`, or (b) raise the cap, or (c) for non-private use the exact
     **SVD-from-moments** path the server already supports (`_svd_basis_from_moments`).
   - Take-away for d': non-privately, **more steps fix everything** — d' is not the issue
     here. The d' sensitivity appears only under privacy (Step 3).

---

## Step 3 — Private regime (ε = 0.25 spent on the basis)

`δ = 1/(n ln n)`. Full data (best for privacy: small q ⟹ strong subsampling
amplification). lr from Step 1. EVR averaged over 3 noise seeds. Figures in
`results/fig_*.png`.

### ⚠ Privacy-accounting correction (this revision)

The first version of Step 3 used `utils/ortho_clustering._find_sigma_autodp`, which calls
`rdp_bank.RDP_gaussian_subsampled` — **a function that does not exist in the installed
autodp**. The `try/except` silently falls back to `T·q·RDP_gaussian`, the *unamplified*
linear-in-q bound, which **ignores privacy amplification by subsampling** and returns σ
**~20× too large** (e.g. mnist784 ε=0.25: σ≈66 instead of the correct ≈3.3). This is a
real bug in the production accounting (the LSH pipeline calls the same function).

Fixed here by `common.sigma_for_epsilon`, which uses autodp's `NoisySGD_Mechanism`
(T-fold composition of the **Poisson-subsampled** Gaussian, rate q=batch/n). Corrected σ
at ε=0.25, batch 256: mnist784 1.9 (1 ep) → 3.3 (10) → 4.5 (20); glove100 1.8 → 1.9 → 2.2.
**All Step-3 numbers below are the corrected runs.** Correct accounting also restores the
true sampling-rate dependence: σ now *grows* with batch (mnist784 batch 64→σ1.9,
1024→σ6.4) because larger batches mean a larger per-step rate q and thus weaker
amplification — the broken version had been ~batch-independent.

### Setup facts
- **Per-sample gradient norms**: mnist784 median ‖g‖≈43 (p99≈113), glove100 median≈1.08.
- **The mechanism that degrades the basis**: each step injects noise of Frobenius norm
  `η_noise ≈ lr·σ·clip/b·√(d·d')` into the orthonormal W, which QR then re-orthonormalizes.
  When `η_noise ≳ 1` a single step scrambles W toward random; when the per-step *signal* is
  too weak W never leaves its random init. With the corrected (small) σ this window is wide
  and the achievable quality is high.

### Headline: at ε=0.25 the basis is near-optimal
Best-tuned EVR vs the PCA optimum (d'=10): **mnist784 0.460 / 0.489 = 94%**,
**glove100 0.240 / 0.245 = 98%**. (Under the broken σ these were 22% / 74% — the
"hard privacy wall" was an accounting artifact.)

### 3A — clip_norm (d'=10, batch 256, 10 epochs): broad inverted-U

| mnist784 clip | 1 | 2 | **5** | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|
| EVR (pca=0.489, rand=0.013) | .376 | .439 | **.460** | .441 | .398 | .283 | .159 | .046 |

| glove100 clip | 0.1 | 0.2 | 0.5 | 1 | **2** | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|---|
| EVR (pca=0.245, rand=0.099) | .221 | .233 | .239 | .240 | **.240** | .237 | .214 | .160 |

Still a two-sided trade (too large ⟹ noise scrambles W; too small ⟹ weak signal), but with
correct σ the peak is **high and broad**: **clip≈5 (mnist784), ≈1–2 (glove100)** — i.e.
near (mnist: a few× below) the typical per-sample gradient norm, the textbook DP-SGD
choice. The production `clip_norm=1.0` is now only mildly suboptimal for glove100 but still
~5× too small for mnist784. Only the product `lr·clip` matters in the clipped regime.

### 3B — epochs (=#steps): mild optimum

| epochs (σ mnist / glove) | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| mnist784 EVR | .272 | .372 | .415 | .451 | **.460** | .443 |
| glove100 EVR | .229 | .237 | .239 | **.240** | .240 | .240 |

More epochs = more optimization vs slowly-growing σ. mnist784 wants **≈10 epochs**;
glove100 (flat spectrum, small d) is **converged by ≈3–5** and essentially flat thereafter.

### 3C — batch size: shallow optimum

| batch (σ mnist) | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| mnist784 EVR | .380 | .441 | **.460** | .453 | .401 |
| glove100 EVR | .225 | .237 | **.240** | .241 | .240 |

Now genuinely two-sided: small batch ⟹ stronger amplification (lower σ) but fewer steps &
higher η_noise/step; large batch ⟹ more noise (higher σ). Sweet spot **batch 256–512**.

### 3D — d' sweep → is large d' bad? (best clip, best epochs)

captured-gain `= (EVR_dpsgd − EVR_rand)/(EVR_pca − EVR_rand)` = fraction of the PCA-over-
random advantage captured.

| d' | 2 | 5 | 8 | 10 | 15 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|---|
| **mnist784** captured-gain | .97 | .96 | .95 | .94 | .90 | .87 | .80 | .75 |
| **mnist784** EVR (pca) | .16(.17) | .32(.33) | .42(.44) | .46(.49) | .53(.58) | .56(.64) | .60(.73) | .63(.83) |
| **glove100** captured-gain | .98 | .98 | .97 | .97 | .96 | .96 | .94 | .92 |
| **glove100** EVR (pca) | .09(.10) | .16(.16) | .21(.21) | .24(.24) | .31(.31) | .37(.38) | .47(.49) | .66(.68) |

The basis is **high-quality at every d'** — captured-gain ≥0.75 even at d'=50. There is a
**gentle, monotone decline** with d' (noise ∝√(d·d'), and high-d' columns have the smallest
eigengaps), and it is steeper for high-d mnist784 (d=784) than for glove100 (d=100). So
"too large d' is bad" is true only in a soft sense at ε=0.25: you lose a little efficiency,
not the whole basis. (The earlier, much harsher decline 0.45→0.15 was the σ artifact.)

---

## Rule of thumb — choosing d' (and the other knobs) for a new dataset

With **correct privacy accounting**, at ε=0.25 the private basis is high-quality at every
d' tested; the residual loss vs PCA grows slowly with d' and with ambient dimension d.

**Governing quantities:**
1. **Spectral concentration** (cheap to estimate from a DP moment matrix): how fast
   `EVR_pca(d')` saturates and how big the PCA-over-random prize is (large for concentrated
   mnist784, small for flat glove100).
2. **The η_noise budget**: per-step noise into W is `η_noise = lr·σ·clip/b·√(d·d')`. With
   the *correct* (small) σ, η_noise stays well below the scrambling threshold across a wide
   range, so the basis degrades only **gently** with d' rather than hitting a wall. Higher d
   and higher d' raise η_noise (∝√(d·d')), which is why high-d mnist784 loses a bit more
   per added dimension than glove100.

**Practical recipe (per dataset):**
- **Use correct subsampled accounting** (`NoisySGD_Mechanism` / a working subsampled-RDP),
  NOT the buggy `_find_sigma_autodp` — otherwise σ is ~20× too big and everything below is
  moot.
- **Tune the effective step `lr·clip`** (only the product matters in the clipped regime).
  Good target: **clip ≈ a few× below the median per-sample gradient norm** (mnist784 clip≈5
  with lr 0.01; glove100 clip≈1–2 with lr 0.05). The peak is broad — being within ~2× is
  fine. Production `clip_norm=1.0` is ~5× low for mnist784, ~OK for glove100.
- **Batch 256–512; epochs ≈10 for concentrated/high-d, ≈3–5 for flat/low-d.** Use the
  **full dataset** — full data gives the smallest per-step q (max amplification) *and* the
  most steps; pre-subsampling never helps privacy here (see note below), only communication.
- **d':** at ε=0.25 you may use a fairly large d' — captured-gain ≥0.9 (glove100) / ≥0.87
  (mnist784) through d'≈20, ≥0.75 even at d'=50. Pick d' from the **downstream task's**
  needs; the basis will track PCA closely. If you must economize, smaller d' has marginally
  higher per-dimension quality, especially for high-d data.
- **Sanity gate:** compare the produced `EVR(W)` to `EVR_random ≈ d'/d`; if within noise,
  the accounting or tuning is off (or ε is genuinely too small) — fall back to `random`.

### On dataset-level subsampling amplification (the reviewer's question)
For these datasets the right move is to train on the **full** data: per-step Poisson
mini-batch sampling (rate q=batch/n) already gives maximal amplification and the most
steps, so there is no σ benefit to pre-subsampling. Pre-subsampling is only forced for
**communication** (production's 2000-pt `BASIS_MAX_SUBSAMPLE`). In that forced case, the
mechanism is "draw m of N once, then DP-SGD," and the guarantee *w.r.t. the full N-point
dataset* enjoys an **additional one-time amplification by γ=m/N** on top of the per-step
amplification. Calibrating σ to ε *on the m-subsample* (what `_find_sigma_autodp(n=m)` does)
ignores this and is over-conservative. `common.sigma_for_epsilon_full` composes the
subsampled-Gaussian steps and then applies `AmplificationBySampling` (without replacement)
at rate γ — the correct full-dataset accounting, which lets a forced subsample spend less
noise. (Benefit is roughly an additive `log(N/m)` discount on the subsample-level ε, so it
softens but does not erase the step-starvation cost of a tiny cap; full data remains best.)

---

## Executive summary

1. **The algorithm is correct.** Non-privately it recovers true PCA to 99–100% at all d'
   (Step 1). The earlier "no better than random" result is **not** an algorithmic bug.
2. **A privacy-accounting bug dominated everything.** `_find_sigma_autodp` calls a
   nonexistent `rdp_bank.RDP_gaussian_subsampled`, silently falls back to an unamplified
   bound, and returns **σ ~20× too large**. This affects the production LSH pipeline too.
   With correct subsampled accounting, σ at ε=0.25 is ≈3.3 (mnist784) / ≈1.9 (glove100),
   not ≈66 / ≈71.
3. **With correct accounting, DP-SGD PCA at ε=0.25 is near-optimal:** EVR captures
   **94% (mnist784)** and **98% (glove100)** of the PCA optimum at d'=10, and **≥0.75 of the
   PCA-over-random gap even at d'=50**. The previously-reported "hard privacy wall"
   (15–45%) was the σ artifact.
4. **Remaining real failure modes (orthogonal to the bug):** (a) `clip_norm=1.0` is ~5×
   too small for high-variance data (mnist784 grads ~43); (b) fixed `lr=0.01` under-
   converges flat/low-variance data (glove100 wants ~0.05); (c) the **2000-pt cap starves
   the SGD of steps** (glove100 → 0.38 EVR vs a 0.95 SVD ceiling on the same points), and
   also leaves dataset-level amplification on the table.
5. **d' guidance (ε=0.25):** quality declines only gently with d'; choose d' from the
   downstream task — the basis tracks PCA closely (captured-gain ≥0.87 through d'≈20).
   Smaller d' is marginally higher quality per dimension, more so for high-d data.
6. **Action items for the codebase:** fix `_find_sigma_autodp` (use a real subsampled-RDP /
   `NoisySGD_Mechanism`); raise/replace the 2000-pt cap or scale epochs to hold steps; set
   `clip_norm`≈median gradient norm and `lr` per data scale. After the σ fix, dpsgd_pca is a
   genuinely viable private basis — re-run the LSH accuracy comparison before drawing
   conclusions from the old numbers.

Reproduce: `cd dpsgd_pca_investigation && <fastlloyd-python> step1_nonprivate.py`,
`step2_sampling.py`, `step3_privacy.py [clip|epochs|batch|dprime]`, `make_plots.py`.
(`<fastlloyd-python>` = `/homes/dnguyen1/.conda/envs/fastlloyd/bin/python`.)
