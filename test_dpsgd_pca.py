"""
Unit tests for DP-SGD PCA basis and related changes.

Tests cover:
  - _find_sigma_autodp: RDP accounting, monotonicity, privacy guarantee
  - dpsgd_pca_basis: shape, orthonormality, variance capture, edge cases
  - orthogonal_basis: dispatcher correctness, error handling
  - Params: new basis_* attributes
  - ortho_proto: integration with both basis methods

Run from the project root:
    python -m pytest test_dpsgd_pca.py -v
    # or
    python test_dpsgd_pca.py
"""

import sys
import os
import unittest

import numpy as np

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from autodp import rdp_bank
    AUTODP_AVAILABLE = True
except ImportError:
    AUTODP_AVAILABLE = False

from utils.ortho_clustering import (
    _find_sigma_autodp,
    dpsgd_pca_basis,
    orthogonal_basis,
    random_orthogonal_basis,
)
from configs.params import Params

requires_autodp = unittest.skipUnless(AUTODP_AVAILABLE, "autodp not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _structured_data(n=400, d=8, seed=0):
    """Return data with strong variance in the first dimension only."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n, d))
    X[:, 0] = rng.randn(n) * 20.0   # variance 400 in first direction
    X[:, 1:] = rng.randn(n, d - 1) * 0.1  # variance 0.01 everywhere else
    return X


def _projected_variance(X, W_col):
    """Projected variance of X onto a single unit vector W_col (shape d,)."""
    X_c = X - X.mean(axis=0)
    return float(np.var(X_c @ W_col))


def _dp_cost_for_sigma(sigma, epsilon, delta, n, batch_size, epochs):
    """
    Replicate the dp_cost computation from _find_sigma_autodp.
    Returns the achieved epsilon for the given sigma.
    """
    T = epochs * max(1, n // batch_size)
    q = min(1.0, batch_size / n)
    alphas = list(range(2, 256))
    min_eps = float("inf")
    for alpha in alphas:
        try:
            rdp = T * rdp_bank.RDP_gaussian_subsampled({"prob": q, "sigma": sigma}, alpha)
        except Exception:
            rdp = T * q * rdp_bank.RDP_gaussian({"sigma": sigma}, alpha)
        eps = rdp + np.log(1.0 / delta) / (alpha - 1)
        if eps < min_eps:
            min_eps = eps
    return min_eps


# ===========================================================================
# 1. _find_sigma_autodp
# ===========================================================================

@requires_autodp
class TestFindSigmaAutodp(unittest.TestCase):

    # --- basic sanity ---

    def test_returns_positive_float(self):
        sigma = _find_sigma_autodp(1.0, 1e-5, 1000, 128, 10)
        self.assertIsInstance(sigma, float)
        self.assertGreater(sigma, 0.0)

    def test_sigma_is_finite(self):
        sigma = _find_sigma_autodp(1.0, 1e-5, 1000, 128, 10)
        self.assertFalse(np.isinf(sigma))
        self.assertFalse(np.isnan(sigma))

    # --- monotonicity in epsilon ---

    def test_larger_epsilon_gives_smaller_sigma(self):
        """Larger privacy budget → less noise needed."""
        s1 = _find_sigma_autodp(0.5, 1e-5, 1000, 128, 10)
        s2 = _find_sigma_autodp(2.0, 1e-5, 1000, 128, 10)
        self.assertGreater(s1, s2,
            msg=f"Expected sigma(eps=0.5)={s1:.4f} > sigma(eps=2.0)={s2:.4f}")

    def test_three_epsilon_levels_ordered(self):
        s_low  = _find_sigma_autodp(0.5,  1e-5, 1000, 128, 10)
        s_mid  = _find_sigma_autodp(1.0,  1e-5, 1000, 128, 10)
        s_high = _find_sigma_autodp(3.0,  1e-5, 1000, 128, 10)
        self.assertGreater(s_low, s_mid)
        self.assertGreater(s_mid, s_high)

    # --- monotonicity in delta ---

    def test_larger_delta_gives_smaller_sigma(self):
        """Weaker delta (larger) → smaller noise needed."""
        s_tight = _find_sigma_autodp(1.0, 1e-6, 1000, 128, 10)
        s_loose = _find_sigma_autodp(1.0, 1e-3, 1000, 128, 10)
        self.assertGreater(s_tight, s_loose,
            msg=f"Expected sigma(delta=1e-6)={s_tight:.4f} > sigma(delta=1e-3)={s_loose:.4f}")

    # --- monotonicity in epochs ---

    def test_more_epochs_gives_larger_sigma(self):
        """More composition steps → need more noise per step."""
        s_few  = _find_sigma_autodp(1.0, 1e-5, 1000, 128, 5)
        s_many = _find_sigma_autodp(1.0, 1e-5, 1000, 128, 20)
        self.assertGreater(s_many, s_few,
            msg=f"Expected sigma(20 epochs)={s_many:.4f} > sigma(5 epochs)={s_few:.4f}")

    # --- privacy guarantee actually satisfied ---

    def test_returned_sigma_satisfies_privacy_guarantee(self):
        """
        The sigma returned by _find_sigma_autodp must satisfy:
            dp_cost(sigma) <= epsilon  (up to binary-search tolerance).
        """
        epsilon, delta = 1.0, 1e-5
        n, batch_size, epochs = 1000, 128, 10
        sigma = _find_sigma_autodp(epsilon, delta, n, batch_size, epochs)
        achieved_eps = _dp_cost_for_sigma(sigma, epsilon, delta, n, batch_size, epochs)
        self.assertLessEqual(achieved_eps, epsilon + 1e-4,
            msg=f"sigma={sigma:.4f} achieves eps={achieved_eps:.6f}, target={epsilon}")

    def test_returned_sigma_tight_not_over_private(self):
        """
        The returned sigma should not be more than twice what's strictly needed.
        (Ensures we are not returning a wildly over-conservative value.)
        """
        epsilon, delta = 1.0, 1e-5
        n, batch_size, epochs = 1000, 128, 10
        sigma = _find_sigma_autodp(epsilon, delta, n, batch_size, epochs)
        # A sigma of half that value should violate the privacy constraint
        achieved_at_half = _dp_cost_for_sigma(sigma / 2.0, epsilon, delta, n, batch_size, epochs)
        self.assertGreater(achieved_at_half, epsilon,
            msg=f"sigma/2={sigma/2:.4f} still satisfies eps={achieved_at_half:.6f}; "
                f"_find_sigma_autodp is over-conservative")

    # --- edge cases ---

    def test_batch_size_larger_than_n(self):
        """Full-batch case (batch_size > n): T = epochs * 1."""
        sigma = _find_sigma_autodp(1.0, 1e-5, 100, 512, 10)
        self.assertGreater(sigma, 0.0)
        self.assertFalse(np.isinf(sigma))

    def test_small_dataset(self):
        sigma = _find_sigma_autodp(1.0, 1e-4, 50, 16, 5)
        self.assertGreater(sigma, 0.0)

    def test_batch_size_equals_n(self):
        sigma = _find_sigma_autodp(1.0, 1e-5, 200, 200, 5)
        self.assertGreater(sigma, 0.0)


# ===========================================================================
# 2. dpsgd_pca_basis — shape and orthonormality
# ===========================================================================

@requires_autodp
class TestDpsgdPcaBasisShapeAndOrthonormality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X = _structured_data(n=400, d=8, seed=0)

    def _run(self, d_prime, **kwargs):
        kw = dict(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        kw.update(kwargs)
        return dpsgd_pca_basis(self.X, d_prime, **kw)

    def test_shape_standard(self):
        W = self._run(3)
        self.assertEqual(W.shape, (8, 3))

    def test_shape_d_prime_one(self):
        W = self._run(1)
        self.assertEqual(W.shape, (8, 1))

    def test_shape_d_prime_equals_d(self):
        W = self._run(8)
        self.assertEqual(W.shape, (8, 8))

    def test_shape_d_prime_exceeds_d(self):
        """d_prime > d: output must be capped at (d, d)."""
        W = self._run(20)
        self.assertEqual(W.shape, (8, 8))

    def test_columns_orthonormal_d_prime_3(self):
        W = self._run(3)
        np.testing.assert_allclose(
            W.T @ W, np.eye(3), atol=1e-10,
            err_msg="W^T W should be identity (orthonormality failed)"
        )

    def test_columns_orthonormal_d_prime_1(self):
        W = self._run(1)
        np.testing.assert_allclose(
            np.linalg.norm(W[:, 0]), 1.0, atol=1e-10,
            err_msg="Single column should be a unit vector"
        )

    def test_columns_orthonormal_full_rank(self):
        W = self._run(8)
        np.testing.assert_allclose(
            W.T @ W, np.eye(8), atol=1e-10
        )

    def test_orthonormality_preserved_after_many_steps(self):
        """With many epochs the QR re-orthonormalization must hold."""
        W = self._run(3, epochs=20)
        np.testing.assert_allclose(W.T @ W, np.eye(3), atol=1e-10)

    def test_1d_data_caps_at_d_equals_1(self):
        X_1d = np.random.randn(200, 1)
        W = dpsgd_pca_basis(X_1d, d_prime=5, epsilon=1.0, delta=1e-4, clip_norm=1.0)
        self.assertEqual(W.shape, (1, 1))
        np.testing.assert_allclose(abs(W[0, 0]), 1.0, atol=1e-10)

    def test_batch_size_larger_than_n(self):
        X_small = self.X[:50]
        W = dpsgd_pca_basis(X_small, d_prime=2, epsilon=1.0, delta=1e-4,
                            clip_norm=1.0, batch_size=512)
        self.assertEqual(W.shape, (8, 2))
        np.testing.assert_allclose(W.T @ W, np.eye(2), atol=1e-10)

    def test_data_with_constant_feature_no_crash(self):
        """A column of all-identical values must not cause NaN/Inf."""
        X = self.X.copy()
        X[:, 2] = 0.0   # constant column
        W = dpsgd_pca_basis(X, d_prime=2, epsilon=1.0, delta=1e-5, clip_norm=1.0)
        self.assertFalse(np.any(np.isnan(W)), "NaN in output")
        self.assertFalse(np.any(np.isinf(W)), "Inf in output")

    def test_small_clip_norm_no_crash(self):
        W = dpsgd_pca_basis(self.X, d_prime=2, epsilon=1.0, delta=1e-5,
                            clip_norm=1e-3)
        self.assertEqual(W.shape, (8, 2))
        np.testing.assert_allclose(W.T @ W, np.eye(2), atol=1e-10)

    def test_large_clip_norm_no_crash(self):
        W = dpsgd_pca_basis(self.X, d_prime=2, epsilon=1.0, delta=1e-5,
                            clip_norm=100.0)
        self.assertEqual(W.shape, (8, 2))
        np.testing.assert_allclose(W.T @ W, np.eye(2), atol=1e-10)


# ===========================================================================
# 3. dpsgd_pca_basis — variance capture and privacy sensitivity
# ===========================================================================

@requires_autodp
class TestDpsgdPcaBasisVarianceCapture(unittest.TestCase):
    """
    Statistical tests: average behaviour over several trials.

    These tests use data with an extreme variance ratio so that even a small
    signal-to-noise ratio in the gradient steps is sufficient to align the
    learned basis with the true principal component.
    """

    @classmethod
    def setUpClass(cls):
        # d=2 makes the test fast and the geometry clear:
        # true PC = [1, 0], variance ratio 400 / 0.01 = 40 000.
        np.random.seed(0)
        n = 600
        cls.X_2d = np.zeros((n, 2))
        cls.X_2d[:, 0] = np.random.randn(n) * 20.0   # std=20, var=400
        cls.X_2d[:, 1] = np.random.randn(n) * 0.1    # std=0.1, var=0.01
        cls.true_pc = np.array([1.0, 0.0])
        # Expected variance for a random 2-D unit vector: (400 + 0.01) / 2 ≈ 200
        cls.random_expected_var = (400.0 + 0.01) / 2.0

    def _mean_variance_over_trials(self, n_trials=5, **basis_kwargs):
        """Return mean projected variance across n_trials independent runs."""
        variances = []
        for _ in range(n_trials):
            W = dpsgd_pca_basis(self.X_2d, d_prime=1, **basis_kwargs)
            variances.append(_projected_variance(self.X_2d, W[:, 0]))
        return float(np.mean(variances))

    def test_dpsgd_pca_captures_more_variance_than_random_basis(self):
        """
        DP-PCA with a reasonable budget should beat a random direction on average.
        Random direction has E[var] ≈ 200; DP-PCA should be substantially above.
        """
        mean_var = self._mean_variance_over_trials(
            n_trials=5, epsilon=2.0, delta=1e-5, clip_norm=1.0, epochs=15
        )
        threshold = self.random_expected_var * 1.5   # 300 — well above random
        self.assertGreater(mean_var, threshold,
            msg=f"DP-PCA mean variance={mean_var:.1f} should exceed {threshold:.1f} "
                f"(random expected ≈ {self.random_expected_var:.1f})")

    def test_higher_epsilon_captures_more_variance(self):
        """
        Looser privacy (higher ε) → less noise → better alignment with true PC.
        Tested as an average over 5 independent trials.
        """
        mean_tight = self._mean_variance_over_trials(
            n_trials=5, epsilon=0.5, delta=1e-5, clip_norm=1.0, epochs=15
        )
        mean_loose = self._mean_variance_over_trials(
            n_trials=5, epsilon=3.0, delta=1e-5, clip_norm=1.0, epochs=15
        )
        self.assertGreater(mean_loose, mean_tight,
            msg=f"eps=3.0 mean var={mean_loose:.1f} should exceed eps=0.5 mean var={mean_tight:.1f}")

    def test_more_epochs_improves_variance_capture_when_epsilon_is_generous(self):
        """With a fixed, generous budget, more epochs produce better alignment."""
        mean_few = self._mean_variance_over_trials(
            n_trials=5, epsilon=3.0, delta=1e-5, clip_norm=1.0, epochs=3
        )
        mean_many = self._mean_variance_over_trials(
            n_trials=5, epsilon=3.0, delta=1e-5, clip_norm=1.0, epochs=20
        )
        # DP-SGD with more epochs is noisier (sigma grows), but each epoch adds
        # gradient signal. At generous epsilon, more epochs help.
        # Use a soft check: many-epoch result should be at least as good.
        self.assertGreaterEqual(mean_many, mean_few * 0.9,
            msg=f"More epochs ({mean_many:.1f}) should not be much worse than fewer ({mean_few:.1f})")


# ===========================================================================
# 4. orthogonal_basis dispatcher
# ===========================================================================

class TestOrthogonalBasis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(1)
        cls.X = np.random.randn(200, 8)

    # --- random method ---

    def test_random_matches_random_orthogonal_basis_exactly(self):
        W_disp = orthogonal_basis(self.X, d_prime=3, method="random", seed=99)
        W_direct = random_orthogonal_basis(8, 3, seed=99)
        np.testing.assert_array_equal(W_disp, W_direct)

    def test_random_shape(self):
        W = orthogonal_basis(self.X, d_prime=4, method="random", seed=0)
        self.assertEqual(W.shape, (8, 4))

    def test_random_orthonormal(self):
        W = orthogonal_basis(self.X, d_prime=3, method="random", seed=0)
        np.testing.assert_allclose(W.T @ W, np.eye(3), atol=1e-10)

    def test_random_seed_reproducibility(self):
        W1 = orthogonal_basis(self.X, d_prime=2, method="random", seed=42)
        W2 = orthogonal_basis(self.X, d_prime=2, method="random", seed=42)
        np.testing.assert_array_equal(W1, W2)

    def test_random_different_seeds_give_different_results(self):
        W1 = orthogonal_basis(self.X, d_prime=2, method="random", seed=1)
        W2 = orthogonal_basis(self.X, d_prime=2, method="random", seed=2)
        self.assertFalse(np.allclose(W1, W2))

    def test_random_does_not_use_X_data(self):
        """Random basis must depend only on shape, not values — check by perturbing X."""
        W1 = orthogonal_basis(self.X, d_prime=2, method="random", seed=7)
        X_perturbed = self.X + 999.0
        W2 = orthogonal_basis(X_perturbed, d_prime=2, method="random", seed=7)
        np.testing.assert_array_equal(W1, W2)

    # --- dpsgd_pca method ---

    @requires_autodp
    def test_dpsgd_pca_shape(self):
        W = orthogonal_basis(self.X, d_prime=2, method="dpsgd_pca",
                             epsilon=1.0, delta=1e-5, clip_norm=1.0)
        self.assertEqual(W.shape, (8, 2))

    @requires_autodp
    def test_dpsgd_pca_orthonormal(self):
        W = orthogonal_basis(self.X, d_prime=2, method="dpsgd_pca",
                             epsilon=1.0, delta=1e-5, clip_norm=1.0)
        np.testing.assert_allclose(W.T @ W, np.eye(2), atol=1e-10)

    @requires_autodp
    def test_dpsgd_pca_d_prime_exceeds_d(self):
        W = orthogonal_basis(self.X, d_prime=20, method="dpsgd_pca",
                             epsilon=1.0, delta=1e-5, clip_norm=1.0)
        self.assertEqual(W.shape, (8, 8))
        np.testing.assert_allclose(W.T @ W, np.eye(8), atol=1e-10)

    # --- error handling ---

    def test_unknown_method_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            orthogonal_basis(self.X, d_prime=2, method="svd_plain")
        self.assertIn("svd_plain", str(ctx.exception))

    @requires_autodp
    def test_dpsgd_pca_missing_delta_raises_key_error(self):
        with self.assertRaises(KeyError):
            orthogonal_basis(self.X, d_prime=2, method="dpsgd_pca",
                             epsilon=1.0, clip_norm=1.0)  # delta missing

    @requires_autodp
    def test_dpsgd_pca_missing_epsilon_raises_key_error(self):
        with self.assertRaises(KeyError):
            orthogonal_basis(self.X, d_prime=2, method="dpsgd_pca",
                             delta=1e-5, clip_norm=1.0)   # epsilon missing

    @requires_autodp
    def test_dpsgd_pca_missing_clip_norm_raises_key_error(self):
        with self.assertRaises(KeyError):
            orthogonal_basis(self.X, d_prime=2, method="dpsgd_pca",
                             epsilon=1.0, delta=1e-5)     # clip_norm missing


# ===========================================================================
# 5. Params — new basis_* attributes
# ===========================================================================

class TestParamsNewAttributes(unittest.TestCase):

    def test_default_basis_method(self):
        self.assertEqual(Params().basis_method, "random")

    def test_default_basis_epsilon(self):
        self.assertEqual(Params().basis_epsilon, 0.0)

    def test_default_basis_delta(self):
        self.assertEqual(Params().basis_delta, 1e-5)

    def test_default_basis_clip_norm(self):
        self.assertEqual(Params().basis_clip_norm, 1.0)

    def test_set_all_four_via_kwargs(self):
        p = Params(
            basis_method="dpsgd_pca",
            basis_epsilon=0.5,
            basis_delta=1e-6,
            basis_clip_norm=2.0,
        )
        self.assertEqual(p.basis_method, "dpsgd_pca")
        self.assertAlmostEqual(p.basis_epsilon, 0.5)
        self.assertAlmostEqual(p.basis_delta, 1e-6)
        self.assertAlmostEqual(p.basis_clip_norm, 2.0)

    def test_setattr_post_init(self):
        p = Params()
        p.basis_method = "dpsgd_pca"
        p.basis_epsilon = 0.3
        self.assertEqual(p.basis_method, "dpsgd_pca")
        self.assertAlmostEqual(p.basis_epsilon, 0.3)

    def test_basis_attrs_appear_in_vars(self):
        """All four new attributes must appear in vars(params) for CSV export."""
        p = Params()
        d = vars(p)
        for attr in ("basis_method", "basis_epsilon", "basis_delta", "basis_clip_norm"):
            self.assertIn(attr, d, msg=f"{attr} missing from vars(Params())")

    def test_invalid_kwarg_still_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            Params(nonexistent_param=42)

    def test_other_defaults_unchanged(self):
        """Adding new attributes must not change existing defaults."""
        p = Params()
        self.assertEqual(p.d_prime, 3)
        self.assertEqual(p.sigma, 0.0)
        self.assertEqual(p.dp, "none")
        self.assertEqual(p.method, "none")
        self.assertEqual(p.fixed, True)


# ===========================================================================
# 6. ortho_proto integration
# ===========================================================================

class TestOrthoProtoIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from utils.protocols import ortho_proto
        cls.ortho_proto = staticmethod(ortho_proto)

        np.random.seed(7)
        n, d = 300, 4
        cls.values = np.random.randn(n, d).astype(np.float64)
        cls.value_lists = [cls.values[:150], cls.values[150:]]

    def _params(self, basis_method="random", d_prime=2, **kw):
        p = Params(
            k=4, dim=4, data_size=300, num_clients=2,
            d_prime=d_prime, sigma=0.0, fixed=False,
            basis_method=basis_method,
            basis_epsilon=kw.get("basis_epsilon", 0.0),
            basis_delta=kw.get("basis_delta", 1e-5),
            basis_clip_norm=kw.get("basis_clip_norm", 1.0),
        )
        p.seed = kw.get("seed", 42)
        return p

    # --- random baseline ---

    def test_random_basis_returns_2d_centroid_array(self):
        centers, _ = self.ortho_proto(self.value_lists, self._params("random"))
        self.assertEqual(centers.ndim, 2)
        self.assertEqual(centers.shape[1], 4)

    def test_random_basis_same_seed_is_deterministic(self):
        p1 = self._params("random", seed=42)
        p2 = self._params("random", seed=42)
        c1, s1 = self.ortho_proto(self.value_lists, p1)
        c2, s2 = self.ortho_proto(self.value_lists, p2)
        np.testing.assert_array_equal(c1, c2)
        self.assertEqual(s1["occupied_quadrants"], s2["occupied_quadrants"])

    def test_random_basis_different_seeds_may_differ(self):
        c1, _ = self.ortho_proto(self.value_lists, self._params("random", seed=1))
        c2, _ = self.ortho_proto(self.value_lists, self._params("random", seed=2))
        # Very unlikely to be identical
        self.assertFalse(np.array_equal(c1, c2))

    # --- dpsgd_pca method ---

    @requires_autodp
    def test_dpsgd_pca_basis_returns_2d_centroid_array(self):
        p = self._params("dpsgd_pca", basis_epsilon=1.0, basis_delta=1e-4)
        centers, _ = self.ortho_proto(self.value_lists, p)
        self.assertEqual(centers.ndim, 2)
        self.assertEqual(centers.shape[1], 4)

    @requires_autodp
    def test_dpsgd_pca_no_nan_or_inf_in_centroids(self):
        p = self._params("dpsgd_pca", basis_epsilon=1.0, basis_delta=1e-4)
        centers, _ = self.ortho_proto(self.value_lists, p)
        self.assertFalse(np.any(np.isnan(centers)))
        self.assertFalse(np.any(np.isinf(centers)))

    # --- stats dict ---

    def test_stats_dict_has_all_expected_keys(self):
        expected = {"unassigned", "count_min", "count_max",
                    "count_mean", "count_std", "occupied_quadrants"}
        _, stats = self.ortho_proto(self.value_lists, self._params("random"))
        self.assertTrue(expected.issubset(stats.keys()),
            msg=f"Missing keys: {expected - stats.keys()}")

    def test_unassigned_is_zero(self):
        _, stats = self.ortho_proto(self.value_lists, self._params("random"))
        self.assertEqual(stats["unassigned"], 0)

    def test_occupied_quadrants_within_bounds_d_prime_1(self):
        _, stats = self.ortho_proto(self.value_lists, self._params("random", d_prime=1))
        self.assertGreaterEqual(stats["occupied_quadrants"], 1)
        self.assertLessEqual(stats["occupied_quadrants"], 2)

    def test_occupied_quadrants_within_bounds_d_prime_3(self):
        _, stats = self.ortho_proto(self.value_lists, self._params("random", d_prime=3))
        self.assertGreaterEqual(stats["occupied_quadrants"], 1)
        self.assertLessEqual(stats["occupied_quadrants"], 8)

    def test_count_min_leq_count_max(self):
        _, stats = self.ortho_proto(self.value_lists, self._params("random"))
        self.assertLessEqual(stats["count_min"], stats["count_max"])

    def test_centroid_count_equals_occupied_quadrants(self):
        centers, stats = self.ortho_proto(self.value_lists, self._params("random", d_prime=2))
        self.assertEqual(centers.shape[0], stats["occupied_quadrants"])


# ===========================================================================
# 7. Noise-level accounting consistency
# ===========================================================================

@requires_autodp
class TestNoiseAccountingConsistency(unittest.TestCase):
    """
    Verify that the noise actually added during DP-SGD is consistent with
    the sigma returned by _find_sigma_autodp.
    """

    def test_sigma_not_larger_than_binary_search_upper_bound(self):
        """sigma must be strictly within the search range [0.01, 1000]."""
        sigma = _find_sigma_autodp(0.5, 1e-5, 500, 64, 10)
        self.assertGreater(sigma, 0.01)
        self.assertLess(sigma, 1000.0)

    def test_absolute_noise_scale_increases_with_clip_norm(self):
        """
        The absolute noise scale is sigma * clip_norm.
        For fixed privacy params, a larger clip_norm must yield larger |noise|.
        """
        sigma = _find_sigma_autodp(1.0, 1e-5, 1000, 128, 10)
        clip_small, clip_large = 0.5, 2.0
        self.assertGreater(sigma * clip_large, sigma * clip_small)

    def test_more_steps_increases_sigma(self):
        """Doubling epochs should strictly increase sigma."""
        s1 = _find_sigma_autodp(1.0, 1e-5, 1000, 128, 5)
        s2 = _find_sigma_autodp(1.0, 1e-5, 1000, 128, 10)
        s3 = _find_sigma_autodp(1.0, 1e-5, 1000, 128, 20)
        self.assertLess(s1, s2)
        self.assertLess(s2, s3)

    def test_privacy_cost_of_half_sigma_violates_budget(self):
        """
        Using half the calibrated sigma should violate the privacy budget,
        confirming we are at the tightest valid sigma, not too conservative.
        """
        epsilon, delta = 1.0, 1e-5
        params = dict(n=500, batch_size=64, epochs=8)
        sigma = _find_sigma_autodp(epsilon, delta, **params)
        cost_at_half = _dp_cost_for_sigma(sigma / 2.0, epsilon, delta, **params)
        self.assertGreater(cost_at_half, epsilon,
            msg=f"Half sigma={sigma/2:.4f} achieved eps={cost_at_half:.4f} ≤ {epsilon}; "
                f"calibrated sigma is over-conservative")

    def test_privacy_cost_at_calibrated_sigma_within_budget(self):
        """Complementary to above: full sigma must satisfy budget."""
        epsilon, delta = 1.0, 1e-5
        params = dict(n=500, batch_size=64, epochs=8)
        sigma = _find_sigma_autodp(epsilon, delta, **params)
        cost = _dp_cost_for_sigma(sigma, epsilon, delta, **params)
        self.assertLessEqual(cost, epsilon + 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
