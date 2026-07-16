# Privacy pipeline — DP-SGD PCA basis + LSH aggregation

**Privacy model:** add/remove one record, i.e. unbounded `(ε, δ)`-DP. Total budget `(eps,
delta)` with `delta = 1/(n ln n)` (`lsh_server.py:27`). The full mechanism is the
**sequential composition** of two releases — the basis and the aggregation — so their
`(ε, δ)` budgets add.

```
                         total (eps, delta)
                                │  split (lsh_server.build_basis)
              ┌─────────────────┴───────────────────┐
   basis  (eps_basis, delta_basis)        aggregation (eps_agg, delta_agg)
   DP-SGD PCA  (this fix)                  zCDP: leaf-centroid sums + level counts
```

This note explains each privacy-related step, what was fixed, what is verified by tests,
and the open caveats for a thorough audit.

---

## Step 0 — Budget split (`lsh_server.build_basis`, lines 47–56)

For `basis_method == "dpsgd_pca"`:
- `eps_basis  = basis_epsilon · eps`,  `delta_basis = basis_epsilon · delta`
- `eps_agg    = eps − eps_basis`,      `delta_agg   = delta − delta_basis`

`basis_epsilon ∈ (0,1)` is the **fraction** of the total budget spent on the basis. The two
parts sum back to `(eps, delta)`, so sequential composition gives the claimed total.
For `random` / `svd_pca`, `eps_basis = 0` (the basis is treated as spending nothing — see
caveat C). Tested: `TestBudgetComposition.test_basis_aggregation_split_sums_to_total`.

---

## Step 1 — Basis privacy: DP-SGD PCA  (`dpsgd_pca_basis`, `_find_sigma_autodp`)

The basis is the output of DP-SGD on the projected-variance objective. Three sub-parts:

**1a. Per-sample gradient clipping → sensitivity.** Each per-sample gradient
`g_i = −2 xᵢ (xᵢᵀW)` is scaled by `min(1, clip_norm / ‖g_i‖_F)`, so every clipped gradient
has Frobenius norm ≤ `clip_norm`. Under add/remove of one record, the per-step gradient
**sum** therefore changes by ≤ `clip_norm` in L2 — the **L2 sensitivity is `clip_norm`**.
Tested for arbitrary/adversarial points: `TestClippingSensitivity.*`.

**1b. Gaussian mechanism per step.** Noise `N(0, (σ·clip_norm)² I)` is added to the
gradient sum (then divided by the batch size — deterministic post-processing, privacy-free).
So the **noise multiplier** (noise std / sensitivity) is `σ`, independent of `clip_norm`.
`_find_sigma_autodp` therefore returns the multiplier and takes no `clip_norm` argument.

**1c. Composition + subsampling amplification → calibrate σ.** Over an epoch the data is
shuffled into mini-batches, so each step is a **Poisson-subsampled Gaussian** with rate
`q = batch/n`, and the run is `T = epochs·⌊n/batch⌋` such steps. `_find_sigma_autodp`
binary-searches the smallest `σ` whose **subsampled-RDP composition** meets `eps_basis` at
`delta_basis`, using autodp's `NoisySGD_Mechanism` under the **add/remove** neighboring
relation, then converting RDP → `(ε, δ)`.

> **Fix applied this revision.** The previous `_find_sigma_autodp` called
> `rdp_bank.RDP_gaussian_subsampled`, which **does not exist** in the installed autodp; the
> `try/except` silently fell back to `T·q·RDP_gaussian`, the *unamplified* bound, returning
> **σ ~20× too large** (ε=0.25, n=70k, batch 256, ep 10: σ≈66 vs the correct ≈3.3). It now
> uses `NoisySGD_Mechanism`, and the search **expands its upper bound** until the budget is
> met, so it can never silently under-noise. Regression-guarded by
> `TestBasisAccounting.test_amplification_actually_applied`.

Verified: returned σ meets the budget and is tight (σ/2 violates), monotone in ε/δ/epochs,
and privacy-safe in edge cases (`q=1` full batch, `batch>n`, tiny ε, small n) — see
`test_pca_privacy.py` groups 1 and 3. The QR re-orthonormalization after each step is
data-independent post-processing and spends no budget.

---

## Step 2 — Aggregation privacy: zCDP  (`compute_dp_sigmas_zcdp`, `zcdp_rho_from_epsilon`)

The aggregation releases, under add/remove with **assumed L2 sensitivity 1**:
- **one** leaf-centroid **sum** vector (leaves partition the points → a single Gaussian
  release, `σ_centers`), and
- **one noisy count histogram per tree level** (`L = max_depth+1` levels; within a level
  the nodes partition the points → parallel composition, one Gaussian per level, `σ_count`).

These compose **additively in zCDP**: `ρ_total = ρ_centers + L·ρ_per_level`. The target
`(eps_agg, delta_agg)` is converted once to `ρ_total` via `zcdp_rho_from_epsilon` (zCDP
spends δ a single time at conversion — no per-release δ splitting), then `ρ_total` is split
between centers and counts by the `sigma_fraction` knob. Round-trip verified to spend
≤ budget: `TestBudgetComposition.test_aggregation_zcdp_round_trip_within_budget`.

---

## Step 3 — Total

`(eps_basis, delta_basis)` ⊕ `(eps_agg, delta_agg)` = `(eps, delta)` by basic sequential
composition. (One could tighten this with RDP/zCDP composition across the two stages, but
basic composition is valid and is what the split assumes.)

---

## What the tests check (`test_pca_privacy.py`, `test_dpsgd_pca.py`)

- Accounting: calibrated σ meets and is tight to the budget; monotone in ε, δ, epochs;
  subsampling amplification is actually applied (regression guard for the fixed bug).
- Sensitivity: clipped per-sample gradient ≤ clip_norm for any point; add/remove of one
  record moves the clipped sum by ≤ clip_norm; σ is a clip-independent multiplier.
- Edge cases: `q=1`, `batch>n`, tiny/large ε, small n, `d'>d` — all meet the budget; the
  search expands its bound rather than under-noising.
- Composition: basis+aggregation ε/δ split sums to total; zCDP aggregation round-trips
  within budget; `sigma_fraction` trades centroid vs count noise as intended.
- Empirical single-step audit of the Gaussian building block, with a documented hook
  (`scale_up_audit`) for a full end-to-end DP audit.

All 16 privacy tests in `test_dpsgd_pca.py` and 20 in `test_pca_privacy.py` pass under the
`fastlloyd` env.

---

## Caveats / open items for a thorough audit

**A. Basis guarantee w.r.t. the full dataset — IMPLEMENTED (amplification by sub-sampling).**
The server runs DP-SGD on a **Poisson sub-sample** of the full pool (`lsh_client.subsample`
now keeps each point i.i.d. w.p. `γ = min(basis_data_fraction, BASIS_MAX_SUBSAMPLE/N)`).
`utils.ortho_clustering._find_sigma_autodp_full(eps, delta, m, full_n, batch, epochs)`
composes the inner T-step subsampled-Gaussian (`NoisySGD_Mechanism`, add/remove) and applies
the one-time outer **Poisson amplification** at rate `γ = m/N`
(`AmplificationBySampling(PoissonSampling=True)`), giving `(eps_basis, delta_basis)`-DP
**w.r.t. add/remove one record in the full N-point dataset**. `lsh_server.build_basis` passes
`full_n = params.data_size`, so the basis noise is now calibrated to the full dataset and is
**smaller** than the sub-sample-level noise (e.g. m=2000, N=70k, ε=0.25: σ 17.8 → 2.1).
`γ ≥ 1` reduces exactly to `_find_sigma_autodp`. Tested in
`test_pca_privacy.py:TestFullDatasetAmplification` / `TestClientPoissonSubsample`.
Design: `PLAN_subsample_amplification.md`. *(Calibration is a one-time per-run cost; each
amplified accounting query is ~1s, so the search is capped at ~12 iterations.)*

**B. `data_fraction` double-subsampling.** `dpsgd_pca_basis` can additionally subsample by
`data_fraction` and (a separate bug) floors the subsample at `batch_size`, crashing when
`n < batch_size`. `lsh_server` calls it with `data_fraction=1.0`, so the pipeline avoids
this, but the floor should be fixed before enabling `data_fraction`.

**C. `svd_pca` is the oracle for the *maximum potential of a private basis*, not a private
method.** The SVD basis is computed from the exact (noiseless) pooled moment matrix, so it
spends `eps_basis = 0` and the *basis itself is not DP*. **Its purpose is to mark the ceiling**
— the best explained variance any basis (private or not) could achieve at a given `d'` — so we
can measure how close the private DP-SGD PCA basis gets to that maximum (e.g. 94–98% at
ε=0.25; see REPORT.md). A run using `svd_pca` is therefore **not** end-to-end `(eps, delta)`-DP
and must never be reported as a private method; it is the upper-bound reference only.

**D. Aggregation sensitivity assumes unit-norm data.** `compute_dp_sigmas_zcdp` assumes the
centroid-sum has L2 sensitivity 1, but the centroids sum **raw normalized values**
(`lsh_client.py:105`, data in `[-1,1]^d`), whose norm can reach `√d`. The protocol must scale
points into the unit ball before the sum (or `σ_centers` must carry the `√d` factor);
otherwise the **aggregation** under-noises. *This is in the FastLloyd-style aggregation, not
the PCA basis.* **To be fixed in `main` with proper tests — tracked in `TODO.md`.** It is a
**precondition/assumption**, so the unit-norm normalization's calculation and running time
must be performed **outside the measured/benchmarked performance region** (it is data prep,
not part of the algorithm being timed).

**E. End-to-end empirical audit** of the full DP-SGD-PCA mechanism (composition + QR) is
scaffolded but not run (`TestEmpiricalSingleStep.scale_up_audit`). Recommended before
publishing the privacy claim.
