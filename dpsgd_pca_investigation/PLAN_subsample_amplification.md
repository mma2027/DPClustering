# Plan — translate basis privacy from the sub-sample to the full dataset

## Problem

The DP-SGD PCA basis is trained on a **sub-sample** of `m` points pooled at the server
(`BASIS_MAX_SUBSAMPLE = 2000`, `parties/lsh_client.py:26`), and `_find_sigma_autodp` is
currently calibrated with `n = m`. The resulting guarantee is `(ε_basis, δ_basis)`-DP **with
respect to the m-point sub-sample**, not the full `N`-point dataset. Because the sub-sample
selection is itself random, the *full-dataset* guarantee is **stronger** — drawing a record
into the sub-sample at all is an extra random event (amplification by sub-sampling). We want
to claim the budget w.r.t. the full dataset, which lets us spend **less noise** (smaller σ)
for the same target `(ε_basis, δ_basis)`, under the **add/remove-one-record** model.

## Theory

The basis mechanism is `M(D) = DPSGD( Subsample_γ(D) )`, where `Subsample_γ` selects the
training set from the `N`-point candidate pool **once**, and `DPSGD` internally does its own
Poisson mini-batch sampling (rate `q = batch/m`) over `T = epochs·⌊m/batch⌋` steps.

If the inner `DPSGD` is `(ε', δ')`-DP w.r.t. its training set, then by amplification by
sub-sampling at rate `γ`:

> `ε_full ≈ log(1 + γ (e^{ε'} − 1))`  (≈ `γ·ε'` for small `ε'`),  `δ_full = γ·δ'`.

So to hit a target `ε_full`, the inner mechanism may use a larger `ε'` ⟹ smaller σ. The
benefit is roughly an **additive `log(N/m)` discount** on `ε'` in the large-`ε'` regime, and
multiplicative `γ` in the small-`ε'` regime — meaningful but bounded (it does not undo the
step-starvation of a tiny `m`; full data remains best — see REPORT.md Step 2).

## The add/remove subtlety (the key design decision)

Amplification depends on **how the sub-sample is drawn**:

- **Poisson sub-sampling** — include each of the `N` candidates independently w.p. `γ`.
  Cleanly amplifies **add/remove** DP and composes with the inner Poisson mini-batching.
  autodp: `AmplificationBySampling(PoissonSampling=True)`, compatible with `neighboring='add_remove'`.
  Cost: the training-set size is random (`~Binomial(N, γ)`), not exactly `m`.
- **Without-replacement (fixed `m`)** — draw exactly `m` of `N`. autodp's
  `AmplificationBySampling(PoissonSampling=False)` requires the inner mechanism to be
  `remove_only` (or `add_only`), giving a **one-sided** guarantee, NOT add/remove directly.

The current prototype `common.sigma_for_epsilon_full` uses `PoissonSampling=False` +
`neighboring='remove_only'` → it yields a **remove-only** bound, which is **not** the
requested add/remove guarantee. **Decision: adopt Poisson dataset sub-sampling** so the
full-dataset guarantee is add/remove and rigorous; keep without-replacement only as a
documented, one-sided fallback.

## Integration points

1. `parties/lsh_client.py:basis_subsample` — change the fixed-`m` draw to **Poisson(γ)**
   with `γ = BASIS_MAX_SUBSAMPLE / N` (hard-cap the realized size at some `M_max` for
   communication; account the truncation — see Edge cases). Document that selection is now
   randomized per run.
2. `utils/ortho_clustering.py` — add `_find_sigma_autodp_amplified(epsilon, delta, m_or_gamma,
   N, batch_size, epochs)` that composes the inner `NoisySGD_Mechanism` (add/remove) and
   applies `AmplificationBySampling(PoissonSampling=True)` at rate `γ`, then binary-searches σ.
   Port the working logic from `common.sigma_for_epsilon_full`, switching to the Poisson
   variant. `γ = 1` (full data, `m ≥ N`) must reduce exactly to `_find_sigma_autodp`.
3. `parties/lsh_server.build_basis` — pass `N = params.data_size` (the full pooled count)
   alongside the sub-sample, and call the amplified calibrator for `dpsgd_pca`.
4. `dpsgd_pca_basis` — unchanged (it already takes the already-σ-calibrated noise via
   `_find_sigma_autodp`); only the calibrator it calls changes.

## Edge cases to handle

- `γ = 1` (`m ≥ N`): no outer sub-sampling → identical to `_find_sigma_autodp`.
- `γ → 0` (huge `N`, tiny cap): amplification large but step-starvation dominates; ensure σ
  search still meets the budget (expanding upper bound already guards this).
- Realized Poisson size `0` or `< batch_size`: define behaviour (skip/relax; do **not** crash
  — cf. the existing `n < batch_size` floor bug, Caveat B).
- Hard cap `M_max` truncation under Poisson: either pick `γ` so overflow probability is
  negligible, or account the deterministic truncation (it only removes points → does not
  weaken privacy, but changes the realized rate). Document the choice.
- `δ` bookkeeping: `δ_full = γ·δ'`; calibrate so the realized `δ_full ≤ δ_basis`.

## Testing plan (mirror `test_pca_privacy.py`)

- Amplified σ **meets the full-dataset budget** via an independent re-derivation, and is
  **smaller** than the sub-sample-level σ (amplification helps).
- `γ = 1` reproduces `_find_sigma_autodp` to numerical tolerance.
- Monotone: σ_full increases with `γ` (less amplification) and decreases with `N`.
- Add/remove correctness: confirm the Poisson path uses `neighboring='add_remove'` (regression
  guard against silently falling back to a one-sided bound).
- Edge cases above; empirical single-step / scaled audit hook.

## Performance note

σ calibration is a **one-time** cost per protocol run (not per step / per point), so the
amplification accounting — even the slower tight bound — is amortized and negligible vs the
SGD itself. Use the fast bound (`improved_bound_flag=False`) by default; the tight bound is
pathologically slow under repeated binary-search queries (observed) and should be opt-in.

## Open decisions / risks

- **Poisson vs without-replacement** sampling (recommend Poisson for clean add/remove).
- Whether randomized per-run sub-sample selection is acceptable operationally (reproducibility
  / communication variance).
- Keep this work on a branch with the test suite green before porting to main.
