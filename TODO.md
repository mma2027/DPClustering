# TODO

## Privacy

- [x] **Subsample → full-dataset privacy amplification for the DP-SGD PCA basis.** DONE
  (branch `dpsgdpca-accuracy`): `utils.ortho_clustering._find_sigma_autodp_full` (Poisson
  outer amplification, add/remove) + `dpsgd_pca_basis(full_n=…)` + `lsh_server` passes
  `full_n=data_size` + `lsh_client.subsample` switched to Poisson. Tested in
  `test_pca_privacy.py`. Plan: `dpsgd_pca_investigation/PLAN_subsample_amplification.md`.
  Follow-ups: (a) consider hard-capping the realized Poisson pool for communication
  determinism; (b) port to `main` with the rest; (c) calibration speed — see
  **Performance → DP-SGD basis calibration** below (turned out to be ~2000s/query, not
  ~12s; Approach A landed, Approach B planned).

- [x] **Unit-norm sensitivity assumption in the LSH aggregation. DONE**
  `compute_dp_sigmas_zcdp` assumes the leaf-centroid sum has L2 sensitivity 1, but the
  aggregation summed values in `[-1,1]^d` (norm up to `√d`), so it under-noised.
  Fix: `data_io.data_handler.ensure_unit_norm` projects every point onto the unit ball
  (already-unit rows pass through unchanged; zero rows stay at the origin, norm 0 ≤ 1).
  Preprocessing is now done **in data preparation** (`scripts/download_data.save_dataset`
  does per-feature min-max → `ensure_unit_norm`, writing final unit-norm files), so it is
  **outside the timed region**; `experiments.py` loads files as-is and applies the same
  safeguard idempotently as a guarantee for any legacy/un-prepared file. Existing
  `data/*.txt` were migrated in place (guarded by `data/.unit_norm_migrated`).
  Regression tests: `test_pca_privacy.TestUnitNormSafeguard` (sensitivity ≤ assumed
  bound 1; min-max-alone can exceed it; idempotence; zero-row handling).

- [ ] **`n < batch_size` floor bug** in `dpsgd_pca_basis` (Caveat B): the `max(batch_size, …)`
  subsample floor crashes when `n < batch_size`. Fix before enabling `data_fraction`.

- [ ] **End-to-end empirical DP audit** of the full DP-SGD-PCA mechanism
  (`test_pca_privacy.py:TestEmpiricalSingleStep.scale_up_audit` scaffold).

## Performance

### n=8 timing on large-d datasets (per-rank shard loading — Fix A) [FUTURE]

The scalability sweep runs all MPI ranks on ONE host (`mpirun --oversubscribe`), and every
rank loads the **full** dataset (`experiments.process_dataset` → `load_txt`). At n=8 (9 ranks)
on glove300 (400k × 300) this OOMs a 32 GB host even after the memory-lean `load_txt` (Fix B,
commit `9b23971`): the residual per-rank footprint (`to_fixed` transient ~2.4 GB + eval copies)
× 9 still exceeds 32 GB. So **n=8 is currently capped out** for glove300 (`NCLIENTS="2 4"`);
mnist784 / glove100 n=8 completed fine (fewer cells).

- [ ] **Fix A — per-rank shard loading.** In the MPI path, the server (rank 0) should load
  no data and each client rank should read only its row-slice, instead of every rank loading
  the whole file and then `shuffle_and_split`-ing. Removes the 9× full-dataset multiplier,
  so n=8 (and larger) works on any dataset size. Then restore `NCLIENTS="2 4 8"` and re-run
  the glove300 (and huge-tier) n=8 timing.
- Current n≤4 timing deliverable: `large/timing/timing_compare_nle4/` (mnist784, glove100;
  glove300 timing still incomplete — finish via the n≤4 relaunch).

### DP-SGD basis calibration

Calibrating the DP-SGD-PCA noise multiplier (`utils.ortho_clustering._find_sigma_autodp_full`)
was the dominant cost of the whole timing pipeline: **~2000 s per unique config**.

- [x] **Approach A — search-from-below on autodp (DONE, branch `dpsgdpca-performance`).**
  Root cause (profiled): autodp's RDP→approxDP conversion runs an *unbounded* Brent search
  over the Rényi order; the optimal order grows with σ, each amplified RDP evaluation is
  O(order) **and** the amplified RDP overflows past a σ-dependent numerical cliff (→ garbage
  values, eventual segfault). The old code seeded the σ bisection at the loose *unamplified*
  σ (5–6× too large) and probed downward, so most queries landed in that slow/fragile
  large-σ regime. Fix: keep autodp's exact conversion but bracket σ by **exponential search
  upward** from a small σ, so every query stays near the (small) answer; plus a 0.1%
  relative-σ bisection tolerance. Result: ε=0.5 (worst) 210 s, looser ε tens of s (~9–37×);
  identical σ up to tolerance; safe + tight; 96 tests pass. Commits `3d7a880`, `382c98c`.

- [x] **Approach B — closed-form Poisson amplification (DONE, branch `dpsgdpca-perfB`).**
  The PLD / Fourier route planned here turned out to be **unnecessary and non-viable
  off-the-shelf** in this env: autodp's phi/FFT path is doubly broken (its
  `SubsampleGaussianMechanism(phi_off=False)` passes `prob` where `phi_bank` reads `gamma`,
  *and* `phi_bank` calls `scipy.integrate.quadrature`, removed in SciPy 1.16); Google
  `dp-accounting` is not installed and its outer-amplification-of-a-composition is not native
  either. A simpler route dominates both.

  Insight (profiled): autodp's amplified `get_approxDP` returns `min(closed-form approx-DP
  amplification, RDP→approxDP)`, and the **closed form always wins** in our regime — so A was
  computing the expensive, cliff-prone RDP order-search *and then discarding it*. Compute
  only the closed form:

      ε = log(1 + γ·(e^{ε'} − 1)) = log1p(γ·expm1(ε')),   ε' = inner.get_approxDP(δ/γ)

  where `inner` is the T-fold subsampled-Gaussian composition (`NoisySGD_Mechanism`, cheap and
  cliff-free) and `γ = m/full_n`. This is the standard subsampling amplification theorem — a
  proven **upper bound** (calibrating to it can only over-noise, never under-noise). No PLD
  machinery, no new dependency.

  Result: worst case (ε=0.5, epochs=40, m=7000) **0.39 s** (vs ~210–640 s under A, ~600–1600×);
  σ **identical** to A at production scale (12.3909 vs 12.389) and ≤0.04% larger at tiny T (safe
  direction); cliff-free at σ up to ≥26. Implemented in `utils.ortho_clustering._find_sigma_autodp_full`
  (Approach A retained verbatim as `_find_sigma_autodp_full_amplified_rdp`, an oracle/fallback).
  Validation gate `test_pca_privacy.TestApproachBEquivalence` (B meets full budget under the
  autodp amplified oracle; σ_B ≥ σ_A; tight). Full suite: 99 tests pass.
  Note: identical σ ⟹ identical basis ⟹ **accuracy numbers unchanged** (no accuracy re-run
  needed); only the timing plots' DP-SGD calibration offset shrinks to sub-second.
