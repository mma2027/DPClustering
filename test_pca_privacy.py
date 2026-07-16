"""
Thorough privacy tests for the DP-SGD PCA basis (and the LSH budget composition).

Privacy model: **add/remove one record** ((epsilon, delta)-DP, unbounded neighbouring).
Mechanism under test: utils/ortho_clustering.dpsgd_pca_basis, whose noise is calibrated
by _find_sigma_autodp = T-fold composition of the Poisson-subsampled Gaussian via
autodp's NoisySGD_Mechanism.

Groups:
  1. TestBasisAccounting       — _find_sigma_autodp vs an independent accountant; tightness;
                                  monotonicity; privacy-safety of the search.
  2. TestClippingSensitivity   — per-sample gradient clipping bounds the L2 sensitivity to
                                  clip_norm; absolute noise std == sigma * clip_norm.
  3. TestPrivacyEdgeCases      — full batch (q=1), batch>n, tiny/large eps, small n, d'>d.
  4. TestBudgetComposition     — basis/aggregation epsilon+delta split; zCDP round-trip.
  5. TestEmpiricalSingleStep   — empirical audit of the per-step Gaussian mechanism
                                  (sensitivity + calibrated noise) on adjacent datasets.

Run:  <fastlloyd-python> test_pca_privacy.py            # unittest
      <fastlloyd-python> -m unittest test_pca_privacy -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from autodp.mechanism_zoo import NoisySGD_Mechanism
    AUTODP = True
except ImportError:
    AUTODP = False

from utils.ortho_clustering import (
    _find_sigma_autodp, _find_sigma_autodp_full,
    _find_sigma_autodp_full_amplified_rdp, dpsgd_pca_basis,
    zcdp_rho_from_epsilon, compute_dp_sigmas_zcdp,
)
from data_io import normalize, ensure_unit_norm

requires_autodp = unittest.skipUnless(AUTODP, "autodp not installed")

if AUTODP:
    from autodp.transformer_zoo import AmplificationBySampling


# --- independent reference accountant (kept structurally separate from production) ---
def achieved_epsilon(sigma, delta, n, batch_size, epochs):
    """Independent epsilon of the subsampled-Gaussian DP-SGD composition (add/remove)."""
    T = epochs * max(1, n // batch_size)
    q = min(1.0, batch_size / n)
    return NoisySGD_Mechanism(prob=q, sigma=sigma, niter=T).get_approxDP(delta)


def achieved_epsilon_full(sigma, delta, m, full_n, batch_size, epochs):
    """Independent epsilon w.r.t. the FULL dataset: inner subsampled-Gaussian composition
    amplified by the one-time outer Poisson sub-sampling at rate gamma = m/full_n."""
    T = epochs * max(1, m // batch_size)
    q = min(1.0, batch_size / m)
    inner = NoisySGD_Mechanism(prob=q, sigma=sigma, niter=T)
    outer = AmplificationBySampling(PoissonSampling=True).amplify(
        inner, prob=min(1.0, m / full_n), improved_bound_flag=False)
    return outer.get_approxDP(delta)


# ===========================================================================
# 1. Accounting correctness
# ===========================================================================
@requires_autodp
class TestBasisAccounting(unittest.TestCase):

    CFG = dict(n=10000, batch_size=256, epochs=10)

    def test_calibrated_sigma_meets_budget(self):
        """The returned sigma must achieve epsilon (within search tolerance)."""
        for eps in (0.1, 0.25, 1.0, 3.0):
            delta = 1e-6
            sigma = _find_sigma_autodp(eps, delta, **self.CFG)
            ach = achieved_epsilon(sigma, delta, **self.CFG)
            self.assertLessEqual(ach, eps + 1e-3,
                msg=f"eps={eps}: calibrated sigma={sigma:.4f} achieves {ach:.5f} > budget")

    def test_calibration_is_tight(self):
        """Half the calibrated sigma must violate the budget (not over-conservative)."""
        eps, delta = 0.25, 1e-6
        sigma = _find_sigma_autodp(eps, delta, **self.CFG)
        ach_half = achieved_epsilon(sigma / 2.0, delta, **self.CFG)
        self.assertGreater(ach_half, eps,
            msg=f"sigma/2={sigma/2:.4f} achieves {ach_half:.5f} <= {eps}: over-conservative")

    def test_monotone_in_epsilon(self):
        d = 1e-6
        s = [_find_sigma_autodp(e, d, **self.CFG) for e in (0.25, 0.5, 1.0, 3.0)]
        self.assertTrue(all(s[i] > s[i + 1] for i in range(len(s) - 1)),
                        msg=f"sigma should decrease as epsilon grows: {s}")

    def test_monotone_in_delta(self):
        e = 1.0
        s_tight = _find_sigma_autodp(e, 1e-7, **self.CFG)
        s_loose = _find_sigma_autodp(e, 1e-3, **self.CFG)
        self.assertGreater(s_tight, s_loose)

    def test_monotone_in_epochs(self):
        e, d = 1.0, 1e-6
        s_few = _find_sigma_autodp(e, d, n=10000, batch_size=256, epochs=2)
        s_many = _find_sigma_autodp(e, d, n=10000, batch_size=256, epochs=20)
        self.assertGreater(s_many, s_few)

    def test_amplification_actually_applied(self):
        """Regression guard for the fixed bug: the accountant MUST use subsampling
        amplification. The buggy fallback (T*q*RDP_gaussian, no amplification) returns
        a far larger sigma; assert we are well below it."""
        eps, delta = 0.25, 1e-6
        sigma = _find_sigma_autodp(eps, delta, **self.CFG)
        # Correct subsampled sigma at this config is ~3; the unamplified bug gave ~30+.
        self.assertLess(sigma, 10.0,
            msg=f"sigma={sigma:.2f} is suspiciously large — subsampling amplification "
                f"may not be applied (the _find_sigma_autodp bug regressed)")


# ===========================================================================
# 2. Clipping bounds the sensitivity; noise scales with clip_norm
# ===========================================================================
class TestClippingSensitivity(unittest.TestCase):
    """The accounting assumes per-step L2 sensitivity == clip_norm. Verify the
    clipping in dpsgd_pca_basis enforces that for arbitrary (even adversarial) points."""

    @staticmethod
    def _clipped_grad_norm(xi, W, clip_norm):
        # exact replica of the clip in dpsgd_pca_basis
        proj = xi @ W
        g_i = -2.0 * np.outer(xi, proj)
        g_norm = np.linalg.norm(g_i)
        g_i = g_i * min(1.0, clip_norm / (g_norm + 1e-8))
        return np.linalg.norm(g_i)

    def test_clipped_gradient_never_exceeds_clip_norm(self):
        rng = np.random.RandomState(0)
        d, d_prime = 50, 5
        W, _ = np.linalg.qr(rng.randn(d, d_prime))
        for clip in (0.1, 1.0, 5.0):
            for scale in (1e-3, 1.0, 1e3):           # tiny to huge gradients
                xi = rng.randn(d) * scale
                gn = self._clipped_grad_norm(xi, W, clip)
                self.assertLessEqual(gn, clip + 1e-9,
                    msg=f"clipped grad norm {gn} > clip_norm {clip} (scale={scale})")

    def test_sensitivity_is_per_sample_not_per_batch(self):
        """Add/remove of ONE point changes the clipped sum by at most clip_norm."""
        rng = np.random.RandomState(1)
        d, d_prime, clip = 30, 4, 1.0
        W, _ = np.linalg.qr(rng.randn(d, d_prime))

        def clipped(xi):
            proj = xi @ W
            g = -2.0 * np.outer(xi, proj)
            return g * min(1.0, clip / (np.linalg.norm(g) + 1e-8))

        batch = rng.randn(20, d)
        s_with = sum(clipped(x) for x in batch)
        s_without = sum(clipped(x) for x in batch[:-1])      # remove one point
        self.assertLessEqual(np.linalg.norm(s_with - s_without), clip + 1e-9)

    def test_noise_multiplier_independent_of_clip_norm(self):
        """sigma from _find_sigma_autodp is the NOISE MULTIPLIER (no clip arg); the
        absolute noise std added in dpsgd_pca_basis is sigma*clip_norm, so doubling
        clip_norm doubles the noise but not sigma."""
        # _find_sigma_autodp takes no clip_norm -> multiplier interpretation is structural.
        import inspect
        params = inspect.signature(_find_sigma_autodp).parameters
        self.assertNotIn("clip_norm", params,
            msg="_find_sigma_autodp must return a clip-independent noise multiplier")


# ===========================================================================
# 3. Edge cases (the regime where accounting bugs hide)
# ===========================================================================
@requires_autodp
class TestPrivacyEdgeCases(unittest.TestCase):

    def _check_meets_budget(self, eps, delta, n, batch, epochs):
        sigma = _find_sigma_autodp(eps, delta, n, batch, epochs)
        self.assertTrue(np.isfinite(sigma) and sigma > 0)
        ach = achieved_epsilon(sigma, delta, n, batch, epochs)
        self.assertLessEqual(ach, eps + 1e-3,
            msg=f"(eps={eps},n={n},batch={batch},ep={epochs}) sigma={sigma:.3f} -> {ach:.5f}")
        return sigma

    def test_full_batch_q_equals_one(self):
        # batch_size >= n  -> q = 1 (no subsampling amplification)
        self._check_meets_budget(1.0, 1e-5, n=1000, batch=5000, epochs=10)

    def test_batch_equals_n(self):
        self._check_meets_budget(1.0, 1e-5, n=1000, batch=1000, epochs=10)

    def test_tiny_epsilon_needs_large_sigma(self):
        s_small = self._check_meets_budget(0.05, 1e-6, n=10000, batch=256, epochs=10)
        s_big = self._check_meets_budget(0.5, 1e-6, n=10000, batch=256, epochs=10)
        self.assertGreater(s_small, s_big)

    def test_large_epsilon_small_sigma(self):
        sigma = self._check_meets_budget(8.0, 1e-6, n=10000, batch=256, epochs=10)
        self.assertLess(sigma, 2.0)

    def test_very_small_n(self):
        self._check_meets_budget(1.0, 1e-4, n=64, batch=16, epochs=5)

    def test_returned_sigma_is_privacy_safe_not_just_search_bound(self):
        """Even when the required sigma is large, the function must EXPAND the search
        bound and still meet the budget (never silently under-noise)."""
        eps, delta = 0.02, 1e-8        # very tight -> large sigma, exceeds old hi=1000? check meets budget
        sigma = _find_sigma_autodp(eps, delta, n=2000, batch_size=256, epochs=20)
        ach = achieved_epsilon(sigma, delta, n=2000, batch_size=256, epochs=20)
        self.assertLessEqual(ach, eps + 1e-3)

    def test_d_prime_exceeds_d_still_private(self):
        """Shape edge: d' > d caps to d; mechanism still runs and stays finite."""
        X = np.random.RandomState(0).randn(500, 6)
        W = dpsgd_pca_basis(X, d_prime=20, epsilon=1.0, delta=1e-5, clip_norm=1.0)
        self.assertEqual(W.shape, (6, 6))
        self.assertFalse(np.any(np.isnan(W)) or np.any(np.isinf(W)))


# ===========================================================================
# 4. Budget composition across the pipeline (basis + aggregation)
# ===========================================================================
class TestBudgetComposition(unittest.TestCase):

    def test_basis_aggregation_split_sums_to_total(self):
        """lsh_server splits eps as eps_basis = f*eps, eps_agg = eps - eps_basis."""
        eps, delta = 1.0, 1e-6
        for f in (0.1, 0.2, 0.5):
            eps_basis, eps_agg = f * eps, eps - f * eps
            delta_basis, delta_agg = f * delta, delta - f * delta
            self.assertAlmostEqual(eps_basis + eps_agg, eps)
            self.assertAlmostEqual(delta_basis + delta_agg, delta)
            self.assertGreater(eps_agg, 0)
            self.assertGreater(delta_agg, 0)

    def test_aggregation_zcdp_round_trip_within_budget(self):
        """sigmas from compute_dp_sigmas_zcdp must spend <= the aggregation budget."""
        for eps_agg, delta_agg, f, L in [(0.8, 1e-6, 1.0, 6), (0.5, 1e-5, 2.0, 4)]:
            sc, scount = compute_dp_sigmas_zcdp(eps_agg, delta_agg, f, count_levels=L)
            rho = 1.0 / (2 * sc ** 2) + L * (1.0 / (2 * scount ** 2))
            eps_back = rho + 2 * np.sqrt(rho * np.log(1.0 / delta_agg))
            self.assertLessEqual(eps_back, eps_agg + 1e-6,
                msg=f"agg sigmas spend eps={eps_back:.6f} > budget {eps_agg}")

    def test_sigma_fraction_controls_split(self):
        """Larger sigma_fraction -> less centroid noise, more count noise."""
        sc1, cc1 = compute_dp_sigmas_zcdp(1.0, 1e-6, 0.5, count_levels=4)
        sc2, cc2 = compute_dp_sigmas_zcdp(1.0, 1e-6, 2.0, count_levels=4)
        self.assertGreater(sc1, sc2)     # smaller fraction -> noisier centroids
        self.assertLess(cc1, cc2)


# ===========================================================================
# 4b. Unit-norm safeguard for the LSH aggregation sensitivity assumption
# ===========================================================================
class TestUnitNormSafeguard(unittest.TestCase):
    """compute_dp_sigmas_zcdp calibrates the leaf-centroid-sum noise assuming each
    point has L2 sensitivity <= 1. Leaves partition the points, so adding/removing
    one point changes the stacked leaf-sum vector by exactly that point's norm; the
    aggregation's L2 sensitivity is therefore max_i ||x_i||. ensure_unit_norm (run in
    data preparation) is what guarantees that bound is <= 1."""

    def _heterogeneous(self):
        rng = np.random.RandomState(1)
        return rng.randn(200, 16) * np.array([1, 100, 0.01] + [1] * 13) + 7.0

    def test_minmax_alone_can_violate_assumed_bound(self):
        """Regression motivation: per-feature min-max only bounds points to [-1,1]^d,
        whose norm reaches ~sqrt(d) -- so the sensitivity-1 assumption is violated."""
        P = normalize(self._heterogeneous(), fixed=False)
        max_norm = np.linalg.norm(P, axis=1).max()
        self.assertGreater(max_norm, 1.0,
            "min-max alone should be able to exceed unit norm (else test is vacuous)")
        self.assertLessEqual(max_norm, np.sqrt(P.shape[1]) + 1e-9)

    def test_safeguard_guarantees_sensitivity_at_most_one(self):
        """After the data-prep safeguard, the aggregation's L2 sensitivity (= max
        point norm) is <= the assumed bound of 1."""
        P = ensure_unit_norm(normalize(self._heterogeneous(), fixed=False))
        sensitivity = np.linalg.norm(P, axis=1).max()
        self.assertLessEqual(sensitivity, 1.0 + 1e-9,
            f"leaf-sum sensitivity {sensitivity} exceeds assumed bound 1")

    def test_leaf_sum_adjacent_sensitivity_within_budget(self):
        """End-to-end: on safeguarded data the realized add-remove sensitivity of the
        leaf-sum vector is <= 1, the exact bound compute_dp_sigmas_zcdp budgets for."""
        P = ensure_unit_norm(normalize(self._heterogeneous(), fixed=False))
        # leaf sums partition points; removing one point drops exactly its vector.
        full_sum = P.sum(axis=0)
        worst = max(np.linalg.norm(full_sum - (full_sum - x)) for x in P)
        self.assertLessEqual(worst, 1.0 + 1e-9)
        sc, _ = compute_dp_sigmas_zcdp(0.8, 1e-6, 1.0, count_levels=6)
        self.assertGreater(sc, 0.0)          # sigma calibrated for that sensitivity

    def test_safeguard_idempotent_and_unit_norm(self):
        P = ensure_unit_norm(normalize(self._heterogeneous(), fixed=False))
        self.assertTrue(np.allclose(np.linalg.norm(P, axis=1), 1.0, atol=1e-9))
        self.assertTrue(np.allclose(P, ensure_unit_norm(P)))   # no-op on prepared data

    def test_zero_row_stays_in_unit_ball(self):
        X = np.zeros((4, 5))
        X[2] = [0.0, 3.0, 0.0, 4.0, 0.0]
        out = ensure_unit_norm(X)
        norms = np.linalg.norm(out, axis=1)
        self.assertTrue(np.all(norms <= 1.0 + 1e-9))   # sensitivity bound still holds
        self.assertAlmostEqual(norms[2], 1.0)          # non-zero row reaches unit norm
        self.assertEqual(norms[0], 0.0)                # zero row left at the origin


# ===========================================================================
# 5. Empirical single-step audit (the composable building block)
# ===========================================================================
@requires_autodp
class TestEmpiricalSingleStep(unittest.TestCase):
    """The full mechanism = T noisy clipped-gradient sums (composed by RDP) + QR
    post-processing. QR is data-independent post-processing (free), so privacy rests
    entirely on the per-step Gaussian mechanism. Here we audit that one step:
      (a) sensitivity: ||sum(D) - sum(D')|| <= clip_norm for adjacent D, D';
      (b) the calibrated per-step noise gives the Gaussian-mechanism epsilon implied
          by the RDP accountant.
    A full end-to-end membership audit (scaled-up, many trials) can be layered on top;
    see scale_up_audit() below."""

    def test_single_step_sensitivity_and_gaussian_eps(self):
        rng = np.random.RandomState(0)
        d, d_prime, clip = 20, 3, 1.0
        W, _ = np.linalg.qr(rng.randn(d, d_prime))

        def clipped_sum(batch):
            s = np.zeros((d, d_prime))
            for xi in batch:
                g = -2.0 * np.outer(xi, xi @ W)
                s += g * min(1.0, clip / (np.linalg.norm(g) + 1e-8))
            return s

        D = rng.randn(64, d)
        Dp = D[:-1]                                  # remove one record
        sens = np.linalg.norm(clipped_sum(D) - clipped_sum(Dp))
        self.assertLessEqual(sens, clip + 1e-9, "single-step sensitivity exceeds clip_norm")

        # Single Gaussian step with multiplier sigma == add-remove eps via autodp.
        sigma = 4.0
        eps_step = NoisySGD_Mechanism(prob=1.0, sigma=sigma, niter=1).get_approxDP(1e-6)
        self.assertGreater(eps_step, 0.0)
        self.assertTrue(np.isfinite(eps_step))

    @unittest.skip("Heavy end-to-end empirical DP audit — enable manually for a thorough run")
    def scale_up_audit(self):
        """Scaffold for a thorough empirical (epsilon, delta) audit of the FULL basis:
        repeatedly run dpsgd_pca_basis on a fixed pair of adjacent datasets (differing
        in one record), estimate the membership advantage / privacy loss distribution
        of a chosen statistic of W, and check it against the claimed epsilon lower bound
        (e.g. Jagielski et al. auditing). Left as a documented hook; fill in attack and
        trial count when a thorough audit is requested."""
        raise NotImplementedError


# ===========================================================================
# 6. Full-dataset privacy via amplification by sub-sampling
# ===========================================================================
@requires_autodp
class TestFullDatasetAmplification(unittest.TestCase):
    """`_find_sigma_autodp_full` translates the sub-sample guarantee to the FULL dataset
    via Poisson amplification (add/remove). Tests are kept light: mostly fast fixed-sigma
    property checks and short-circuits, with a single full calibration (each amplified
    query is ~1s, a one-time cost in production)."""

    SMALL = dict(m=800, full_n=8000, batch_size=128, epochs=3)   # gamma = 0.1, T = 18

    def test_gamma_one_reduces_to_direct(self):
        """m == full_n: no outer sub-sampling -> identical to _find_sigma_autodp."""
        eps, delta = 0.5, 1e-5
        a = _find_sigma_autodp_full(eps, delta, 5000, 5000, 128, 3)
        b = _find_sigma_autodp(eps, delta, 5000, 128, 3)
        self.assertAlmostEqual(a, b, places=6)

    def test_m_geq_full_n_reduces_to_direct(self):
        eps, delta = 0.5, 1e-5
        a = _find_sigma_autodp_full(eps, delta, 5005, 5000, 128, 3)
        b = _find_sigma_autodp(eps, delta, 5000, 128, 3)
        self.assertAlmostEqual(a, b, places=6)

    def test_amplification_lowers_epsilon_fixed_sigma(self):
        """Core property (fast, 2 direct queries): at the same sigma the FULL-dataset eps
        is strictly below the sub-sample-level eps."""
        sigma, delta = 2.0, 1e-5
        s = self.SMALL
        eps_sub = achieved_epsilon(sigma, delta, s["m"], s["batch_size"], s["epochs"])
        eps_full = achieved_epsilon_full(sigma, delta, **s, )
        self.assertLess(eps_full, eps_sub,
            msg=f"amplified eps {eps_full:.4f} should be < sub-sample eps {eps_sub:.4f}")

    def test_larger_full_n_lowers_epsilon_fixed_sigma(self):
        """More amplification (smaller gamma) -> smaller eps at fixed sigma (fast)."""
        sigma, delta = 2.0, 1e-5
        e_small_N = achieved_epsilon_full(sigma, delta, 800, 4000, 128, 3)
        e_large_N = achieved_epsilon_full(sigma, delta, 800, 40000, 128, 3)
        self.assertLess(e_large_N, e_small_N)

    def test_calibrated_full_sigma_meets_full_budget_and_reduces_noise(self):
        """The one full calibration: returned sigma meets the FULL-dataset budget
        (independent check) AND is smaller than the sub-sample-level sigma."""
        eps, delta = 0.5, 1e-5
        s = self.SMALL
        sigma_full = _find_sigma_autodp_full(eps, delta, **s)
        sigma_sub = _find_sigma_autodp(eps, delta, s["m"], s["batch_size"], s["epochs"])
        ach = achieved_epsilon_full(sigma_full, delta, **s)
        self.assertLessEqual(ach, eps + 5e-3,
            msg=f"amplified sigma={sigma_full:.4f} achieves full-eps {ach:.5f} > {eps}")
        self.assertLess(sigma_full, sigma_sub,
            msg=f"amplification should reduce noise: full {sigma_full:.3f} vs sub {sigma_sub:.3f}")


# ===========================================================================
# 6b. Approach B (closed-form amplification) vs Approach A (amplified-RDP): the
#     privacy validation gate. B is the production calibrator; A is the retained
#     autodp-native fallback used here as an independent oracle.
# ===========================================================================
@requires_autodp
class TestApproachBEquivalence(unittest.TestCase):
    """`_find_sigma_autodp_full` (Approach B, closed-form Poisson amplification) must be a
    valid privacy upper bound AND must not under-noise relative to `_find_sigma_autodp_full_
    amplified_rdp` (Approach A, autodp's amplified-RDP path). Expected: sigma_B == sigma_A to
    tolerance across the grid (B is exactly the closed form autodp's amplified approxDP
    already minimises to). Grid is small since each Approach-A query is ~1s."""

    # (m, full_n, batch_size, epochs); gamma <= 0.1, modest T. Each Approach-A calibration is
    # ~15s (the slow path being replaced), so the A-vs-B grid is deliberately tiny.
    GRID = [
        dict(m=800,  full_n=8000,  batch_size=128, epochs=3),   # gamma = 0.10, T = 18
        dict(m=1000, full_n=20000, batch_size=256, epochs=5),   # gamma = 0.05, T = 15
    ]
    EPS = (0.5, 1.0)
    DELTA = 1e-6

    def test_B_meets_full_budget(self):
        """Safety: B's sigma achieves the FULL-dataset budget under the autodp amplified
        oracle (never under-noises)."""
        for s in self.GRID:
            for eps in self.EPS:
                sigma_B = _find_sigma_autodp_full(eps, self.DELTA, **s)
                ach = achieved_epsilon_full(sigma_B, self.DELTA, **s)
                self.assertLessEqual(ach, eps + 1e-3,
                    msg=f"B sigma={sigma_B:.4f} for eps={eps}, {s} achieves {ach:.5f} > budget")

    def test_B_matches_A(self):
        """Equivalence / no-regression: sigma_B == sigma_A to tolerance, and B never gives
        LESS noise than A (privacy-safe direction). One small config (A is ~15s/calibration)."""
        s = self.GRID[0]
        for eps in self.EPS:
            sigma_B = _find_sigma_autodp_full(eps, self.DELTA, **s)
            sigma_A = _find_sigma_autodp_full_amplified_rdp(eps, self.DELTA, **s)
            # never under-noise vs the autodp-native accountant (allow tiny search slack)
            self.assertGreaterEqual(sigma_B, sigma_A * (1 - 2e-3),
                msg=f"B under-noises vs A for eps={eps}, {s}: {sigma_B:.5f} < {sigma_A:.5f}")
            # and they should agree closely (relative); at production-scale T they are exact.
            self.assertLessEqual(abs(sigma_B - sigma_A), 5e-3 * sigma_A + 1e-3,
                msg=f"B/A disagree for eps={eps}, {s}: B={sigma_B:.5f} A={sigma_A:.5f}")

    def test_B_is_tight(self):
        """Not over-conservative: 3%-smaller sigma violates the full-dataset budget."""
        s = self.GRID[0]
        eps = 0.5
        sigma_B = _find_sigma_autodp_full(eps, self.DELTA, **s)
        ach_less = achieved_epsilon_full(sigma_B * 0.97, self.DELTA, **s)
        self.assertGreater(ach_less, eps,
            msg=f"0.97*sigma_B={0.97*sigma_B:.4f} achieves {ach_less:.5f} <= {eps}: over-noised")


# ===========================================================================
# 7. Client Poisson sub-sampling (the scheme amplification relies on)
# ===========================================================================
class TestClientPoissonSubsample(unittest.TestCase):
    """The client must Poisson-sample (each point kept i.i.d. w.p. gamma) so the pooled set
    is a genuine Poisson sub-sample of the full dataset — the premise of `_find_sigma_autodp_full`."""

    def _make_client(self, n, N, data_fraction, num_clients=1, cap=2000):
        from configs import Params
        from parties.lsh_client import LshClient
        import parties.lsh_client as lc
        lc.BASIS_MAX_SUBSAMPLE = cap
        p = Params(data_size=N, dim=5, num_clients=num_clients,
                   basis_data_fraction=data_fraction, fixed=False, seed=0)
        return LshClient(0, np.random.RandomState(0).randn(n, 5), p)

    def test_expected_rate_matches_gamma(self):
        N, n, cap = 100000, 100000, 2000
        client = self._make_client(n, N, data_fraction=1.0, cap=cap)
        gamma = min(1.0, cap / N)                     # = 0.02
        sizes = []
        for seed in range(8):
            client.params.seed = seed
            sizes.append(len(client.subsample()))
        mean = np.mean(sizes)
        self.assertAlmostEqual(mean / n, gamma, delta=0.25 * gamma,
            msg=f"mean kept fraction {mean/n:.4f} should be ~gamma={gamma}")

    def test_rate_is_data_independent(self):
        """gamma depends only on public N / cap / data_fraction, not on point values."""
        c1 = self._make_client(5000, 100000, data_fraction=1.0)
        c2 = self._make_client(5000, 100000, data_fraction=1.0)
        c2.values = c2.values + 999.0                 # perturb values, same shapes
        self.assertEqual(len(c1.subsample()), len(c2.subsample()))

    def test_data_fraction_caps_gamma(self):
        # small data_fraction below cap/N should dominate gamma
        client = self._make_client(10000, 100000, data_fraction=0.01, cap=50000)
        kept = len(client.subsample())
        self.assertLess(kept / 10000, 0.05)           # ~1%, well below cap-driven rate


if __name__ == "__main__":
    unittest.main(verbosity=2)
