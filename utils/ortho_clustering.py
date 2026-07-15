from functools import lru_cache

import numpy as np


# The autodp calibrations below are PURE functions of their (public, data-independent)
# arguments — eps, delta, sub-sample size, full_n, batch_size, epochs. They sit inside the
# (timed) basis build, so memoizing makes a repeated config free. NOTE: the noise multiplier
# is fully determined by public parameters, so in a real deployment it is precomputed once
# from a table; the calibration is a constant, data-independent cost that does NOT scale with
# n, d, or the number of clients (it only adds a fixed offset to the dpsgd-PCA line, leaving
# the scalability slopes/rounds/bytes unaffected). See large/EXPERIMENT_PLAN.md.
#
# The amplified calibration (`_find_sigma_autodp_full`, used when DP-SGD trains on a random
# sub-sample of a larger pool) accounts the one-time outer Poisson sub-sampling with the
# CLOSED-FORM approx-DP amplification bound (Approach B, the default). An earlier, much
# slower variant that called autodp's amplified-RDP accountant directly is preserved as
# `_find_sigma_autodp_full_amplified_rdp` (Approach A) for future comparison / cross-checks.
# See the docstrings of both for the derivation and the ~100x speed difference.
@lru_cache(maxsize=None)
def _find_sigma_autodp(epsilon, delta, n, batch_size, epochs):
    """Binary-search for the Gaussian noise multiplier that achieves (epsilon, delta)-DP.

    ## Privacy accounting overview

    DP-SGD adds Gaussian noise N(0, sigma² · clip_norm² · I) to the gradient sum at
    every mini-batch step.  The sensitivity of the sum (over one adjacent dataset
    differing in one point) is clip_norm, so the noise-to-sensitivity ratio is sigma.

    Naively composing T steps each with cost (epsilon_0, delta_0) would give total cost
    roughly T * epsilon_0, which is very loose.  Instead we use the Rényi DP (RDP)
    framework, which composes additively and converts to (epsilon, delta)-DP at the end:

        R(alpha)_total = T * R(alpha)_per_step
        epsilon(alpha) = R(alpha)_total + log(1/delta) / (alpha - 1)
        epsilon = min_alpha  epsilon(alpha)

    Subsampling amplification: because each step draws a fresh mini-batch of size
    batch_size from n points (sampling rate q = batch_size / n), the per-step Rényi cost
    is that of the *subsampled* Gaussian mechanism, which is proportional to q² for small
    q — much cheaper than the non-subsampled version.  This is the dominant source of
    tightness compared with naive composition.

    Accounting is delegated to autodp's NoisySGD_Mechanism (the standard tight
    subsampled-RDP / moments accountant) under the **add/remove-one-record** neighboring
    relation, which is the natural notion for Poisson mini-batch subsampling. The binary
    search finds the smallest sigma whose total cost is ≤ epsilon; it first expands the
    upper bound until the budget is met (so it can never silently under-noise) and then
    bisects 64 times.

    Args:
        epsilon: target (epsilon, delta)-DP privacy budget epsilon
        delta: target delta; standard choice is 1 / (n * log(n))
        n: number of points in the dataset passed to DP-SGD (after any subsampling)
        batch_size: mini-batch size; controls sampling rate q = batch_size / n
        epochs: number of full passes over the data;
                total steps T = epochs * floor(n / batch_size)

    Returns:
        float: smallest sigma s.t. the total privacy cost ≤ (epsilon, delta)

    Note:
        Larger sigma → more noise → more privacy → worse utility.
        To reduce sigma (less noise) for a fixed epsilon budget you can:
          - decrease epochs (fewer compositions)
          - increase batch_size (fewer steps, but weaker subsampling amplification)
          - subsample the dataset before calling (fewer points → fewer steps)
    """
    from autodp.mechanism_zoo import NoisySGD_Mechanism

    T = epochs * max(1, n // batch_size)
    q = min(1.0, batch_size / n)

    def dp_cost(sigma):
        # (epsilon at the given delta) of the T-fold composition of the
        # *Poisson-subsampled* Gaussian mechanism with sampling rate q. autodp's
        # NoisySGD_Mechanism uses the tight subsampled-RDP accountant and the
        # add/remove-one-record neighboring relation, then converts RDP -> (eps, delta).
        # This REPLACES a previous hand-rolled accountant that called a nonexistent
        # rdp_bank.RDP_gaussian_subsampled, silently fell back to the *unamplified*
        # bound T*q*RDP_gaussian, and returned sigma ~20x too large.
        return NoisySGD_Mechanism(prob=q, sigma=sigma, niter=T).get_approxDP(delta)

    # Invariant for the bisection: dp_cost is decreasing in sigma, so we keep an
    # interval [lo, hi] with dp_cost(lo) > epsilon (under-noised) and
    # dp_cost(hi) <= epsilon (budget met). Expand hi first so we can never return
    # an under-noised sigma for tiny epsilon / many steps (privacy safety).
    lo, hi = 1e-3, 1.0
    while dp_cost(hi) > epsilon:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError(
                f"Cannot meet (eps={epsilon}, delta={delta}) for n={n}, "
                f"batch_size={batch_size}, epochs={epochs}: required sigma exceeds 1e6")
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if dp_cost(mid) > epsilon:
            lo = mid
        else:
            hi = mid
    return hi  # satisfies dp_cost(hi) <= epsilon by construction


@lru_cache(maxsize=None)
def _find_sigma_autodp_full(epsilon, delta, m, full_n, batch_size, epochs):
    """Smallest noise multiplier for (epsilon, delta)-DP w.r.t. the FULL `full_n`-point
    dataset, when DP-SGD trains on a random size-`m` Poisson sub-sample of it.

    This is **Approach B** (the default): the outer one-time Poisson sub-sampling is
    accounted with the CLOSED-FORM approx-DP amplification bound. It returns the same sigma
    as the amplified-RDP variant `_find_sigma_autodp_full_amplified_rdp` (Approach A) but is
    ~100x faster and free of the large-sigma numerical cliff. See that function for the
    slower autodp-native variant kept for comparison.

    ## Why this differs from `_find_sigma_autodp`

    `_find_sigma_autodp` calibrates the guarantee w.r.t. the points actually fed to DP-SGD.
    When those points are a *random* sub-sample of a larger pool, the guarantee w.r.t. the
    full pool is **stronger**: being selected into the sub-sample at all is an extra random
    event (privacy amplification by sub-sampling), so the same epsilon-vs-full-dataset can be
    met with **less noise**.

    ## Mechanism and accounting (add/remove one record)

    The basis mechanism is `M(D) = DPSGD( PoissonSample_gamma(D) )` with `gamma = m / full_n`:
      1. the training set is drawn ONCE from the full pool — each record kept independently
         with probability `gamma` (Poisson sub-sampling), then
      2. DP-SGD runs `T = epochs * floor(m / batch_size)` steps, each a Poisson-subsampled
         Gaussian with inner mini-batch rate `q = batch_size / m`.

    We build the inner T-step composition (`NoisySGD_Mechanism`, add/remove neighboring) and
    amplify the WHOLE composition by the one-time outer Poisson sub-sampling at rate `gamma`.

    ## Closed-form outer amplification

    For Poisson sub-sampling under the add/remove relation, if the base (composed) mechanism
    `M` is `(eps', delta')`-DP then `PoissonSample_gamma(M)` is `(eps, delta)`-DP with

        eps   = log(1 + gamma * (exp(eps') - 1))          # = log1p(gamma * expm1(eps'))
        delta = gamma * delta'      <=>   query M at delta' = delta / gamma

    (the standard subsampling amplification theorem; a proven **upper bound**, so calibrating
    to it can only ever *over*-noise, never under-noise -> privacy-safe by construction). The
    inner `M.get_approxDP(delta/gamma)` is autodp's tight subsampled-RDP accountant, which is
    cheap and cliff-free at every sigma (unlike the AMPLIFIED RDP query, whose optimal Renyi
    order grows with sigma, costs O(order) per evaluation, and overflows / can segfault at
    large sigma). Empirically this closed form equals autodp's own amplified `get_approxDP`
    (`AmplificationBySampling(...).amplify(inner, gamma, improved_bound_flag=False)`) to
    machine precision across our whole operating grid -- because autodp's amplified approxDP
    is exactly `min(this closed form, RDP->approxDP)` and the closed form is what wins; the
    expensive RDP path is computed and then discarded. Here we skip it. See the
    validation-gate test in `test_pca_privacy.py` and the plan notes.

    For `gamma >= 1` (`m >= full_n`, no sub-sampling) this reduces exactly to
    `_find_sigma_autodp(epsilon, delta, full_n, batch_size, epochs)`.

    Args:
        epsilon, delta: target budget **w.r.t. the full dataset**.
        m: number of points actually fed to DP-SGD (the realized sub-sample size).
        full_n: size of the full dataset the sub-sample was drawn from.
        batch_size, epochs: DP-SGD schedule (define T and q over the sub-sample).

    Returns:
        float: smallest sigma (noise / clip_norm multiplier) meeting the full-dataset budget.

    Note:
        Assumes the size-`m` set was obtained by Poisson sub-sampling of the full pool at the
        rate `gamma = m / full_n` (see `parties/lsh_client.LshClient.subsample`). `gamma` is
        the realized sampling rate; the sub-sample size is data-independent in value, and the
        effect of add/remove on it is exactly what the amplification term accounts for.
    """
    import math

    from autodp.mechanism_zoo import NoisySGD_Mechanism

    gamma = min(1.0, m / full_n) if full_n > 0 else 1.0
    if gamma >= 1.0:
        return _find_sigma_autodp(epsilon, delta, full_n, batch_size, epochs)

    T = epochs * max(1, m // batch_size)
    q = min(1.0, batch_size / m)
    delta_inner = delta / gamma
    if not delta_inner < 1.0:
        # log(1/delta') must be finite for the inner accountant; gamma is O(0.1) and
        # delta ~ 1e-7 here, so delta/gamma << 1 always. Guard defensively regardless.
        raise ValueError(
            f"delta/gamma = {delta_inner:.3g} >= 1 (delta={delta}, gamma={gamma}); "
            "cannot apply subsampling amplification")

    def dp_cost(sigma):
        # eps' of the T-fold inner composition at the amplified delta' = delta / gamma
        # (cheap, cliff-free), then the closed-form outer Poisson amplification.
        eps_inner = NoisySGD_Mechanism(prob=q, sigma=sigma, niter=T).get_approxDP(delta_inner)
        return math.log1p(gamma * math.expm1(eps_inner))

    # dp_cost is cheap, monotone-decreasing in sigma, AND cliff-free, so a plain robust
    # bisection suffices (the delicate exponential-search-from-below that Approach A needed
    # to dodge the large-sigma RDP cliff is no longer necessary). Expand hi until the budget
    # is met -- so we can never return an under-noised sigma -- then bisect to a 0.1% relative
    # sigma tolerance (far below any privacy-relevant difference).
    lo, hi = 1e-3, 1.0
    while dp_cost(hi) > epsilon:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError(
                f"Cannot meet (eps={epsilon}, delta={delta}) for m={m}, full_n={full_n}, "
                f"batch_size={batch_size}, epochs={epochs}: required sigma exceeds 1e6")
    for _ in range(64):
        if hi - lo <= 1e-3 * hi:
            break
        mid = (lo + hi) / 2.0
        if dp_cost(mid) > epsilon:
            lo = mid
        else:
            hi = mid
    return hi  # satisfies dp_cost(hi) <= epsilon by construction


@lru_cache(maxsize=None)
def _find_sigma_autodp_full_amplified_rdp(epsilon, delta, m, full_n, batch_size, epochs):
    """**Approach A** (fallback, kept for comparison): same guarantee as
    `_find_sigma_autodp_full` but the outer Poisson sub-sampling is accounted with autodp's
    native amplified-RDP path instead of the closed-form approx-DP bound.

    This calls `AmplificationBySampling(PoissonSampling=True).amplify(inner, gamma,
    improved_bound_flag=False).get_approxDP(delta)`. autodp's amplified `get_approxDP` returns
    `min(closed-form approx-DP amplification, RDP->approxDP)`; in our operating regime the
    closed form (what Approach B computes directly) always wins, so this returns the SAME
    sigma -- but it *also* evaluates the RDP->approxDP conversion, whose optimal Renyi order
    grows with sigma, costs O(order) per query, and overflows / can segfault at large sigma.
    That makes it ~100x slower per query (seconds vs ~0.02s) and forces the fragile
    exponential-search-from-below bracketing below.

    Retained so future work can (a) cross-check Approach B's sigma against autodp's own
    accountant as an oracle and (b) compare tightness should a regime ever arise where the
    RDP path is tighter than the closed form (in which case this returns a smaller/less
    conservative sigma -- Approach B stays a valid upper bound either way).

    NOT used in the pipeline; `dpsgd_calibrate_sigma` calls `_find_sigma_autodp_full`.
    See that function's docstring for the mechanism and the closed-form derivation.
    """
    from autodp.mechanism_zoo import NoisySGD_Mechanism
    from autodp.transformer_zoo import AmplificationBySampling

    gamma = min(1.0, m / full_n) if full_n > 0 else 1.0
    if gamma >= 1.0:
        return _find_sigma_autodp(epsilon, delta, full_n, batch_size, epochs)

    T = epochs * max(1, m // batch_size)
    q = min(1.0, batch_size / m)
    amplify = AmplificationBySampling(PoissonSampling=True)

    def dp_cost(sigma):
        inner = NoisySGD_Mechanism(prob=q, sigma=sigma, niter=T)   # add/remove
        outer = amplify.amplify(inner, prob=gamma, improved_bound_flag=False)
        return outer.get_approxDP(delta)

    # Exponential search UPWARD from a small sigma so every (expensive) amplified dp_cost
    # query stays at/below ~ the small answer, avoiding the large-sigma regime where autodp's
    # RDP->approxDP cost explodes and can crash. sigma_cap = the UNAMPLIFIED sigma: a
    # guaranteed-sufficient upper bound (amplification can only lower epsilon) that caps the
    # growth and is the fallback when amplification barely helps.
    sigma_cap = _find_sigma_autodp(epsilon, delta, m, batch_size, epochs)

    GROWTH = 1.5
    sigma0 = min(0.5, sigma_cap)

    if dp_cost(sigma0) <= epsilon:
        # sigma0 already suffices (loose budget / very private): shrink to tighten, staying
        # at small sigma (cheap). Keep the last sufficient value as hi.
        hi, lo = sigma0, sigma0 / GROWTH
        cost_lo = dp_cost(lo)
        while lo > 1e-3 and cost_lo <= epsilon:
            hi = lo
            lo /= GROWTH
            cost_lo = dp_cost(lo)
        if cost_lo <= epsilon:          # even ~0 noise meets the budget (degenerate)
            return lo
        # invariant now: dp_cost(lo) > epsilon >= dp_cost(hi)
    else:
        # typical case: grow from below until sufficient, capped by sigma_cap.
        lo, hi = sigma0, sigma0 * GROWTH
        while hi < sigma_cap and dp_cost(hi) > epsilon:
            lo = hi
            hi *= GROWTH
        if hi >= sigma_cap:
            # the answer is close to the unamplified cap -> amplification barely helps here;
            # return the (sufficient) cap rather than probing the expensive large-sigma zone.
            return sigma_cap
        # invariant now: dp_cost(lo) > epsilon >= dp_cost(hi), with hi <~ 1.5x the answer

    # Refine within the moderate bracket [lo, hi]; dp_cost(hi) <= epsilon is invariant, so
    # the returned hi is always safe (never under-noised). Stop at a RELATIVE sigma tolerance
    # (0.1%), with a hard cap of 16 as a safety net.
    for _ in range(16):
        if hi - lo <= 1e-3 * hi:
            break
        mid = (lo + hi) / 2.0
        if dp_cost(mid) > epsilon:
            lo = mid
        else:
            hi = mid
    return hi


# Memoize (sigma, calibration_seconds) per unique arg-tuple. The calibration is
# deterministic in its public args, so we time it ONCE and replay both the value and its
# measured duration on later calls -- letting every timing row report the true, consistent
# calibration cost without re-paying it (and independent of the network-delay sweep, which
# previously polluted the measurement by attributing the whole calibration to the first delay).
_SIGMA_TIMED = {}


def dpsgd_calibrate_sigma(epsilon, delta, n, batch_size, epochs, full_n=None):
    """Return (sigma, calib_seconds): the DP-SGD noise multiplier and the wall-time its
    autodp calibration cost. Splits the (dominant) sigma search out of the SGD loop so the
    two can be timed separately. Duration is memoized alongside sigma (see _SIGMA_TIMED)."""
    from time import perf_counter
    key = (epsilon, delta, n, batch_size, epochs, full_n)
    if key not in _SIGMA_TIMED:
        t0 = perf_counter()
        if full_n is not None and full_n > n:
            sigma = _find_sigma_autodp_full(epsilon, delta, n, full_n, batch_size, epochs)
        else:
            sigma = _find_sigma_autodp(epsilon, delta, n, batch_size, epochs)
        _SIGMA_TIMED[key] = (sigma, perf_counter() - t0)
    return _SIGMA_TIMED[key]


def dpsgd_pca_basis(X, d_prime, epsilon, delta, clip_norm, epochs=10, lr=0.1, batch_size=256, data_fraction=1.0, full_n=None, sigma=None):
    """Compute a differentially private PCA basis via DP-SGD.

    ## Objective

    Find W ∈ R^{d × d'} with orthonormal columns (W on the Stiefel manifold St(d, d'))
    that maximizes the projected variance:

        max_{W: W^T W = I}  trace(W^T X^T X W)

    Equivalently, minimize L(W) = -trace(W^T X^T X W).  At the optimum, the columns of
    W are the top-d' principal components of X (the leading right singular vectors of X).

    ## Algorithm

    1. **Center** the data: X_c = X - mean(X, axis=0).

    2. **Initialize** W as a random (d, d') orthonormal matrix via random_orthogonal_basis.

    3. **Calibrate noise**: call _find_sigma_autodp to find the smallest Gaussian noise
       multiplier sigma such that the full SGD run is (epsilon, delta)-DP.

    4. For each epoch, shuffle the data and iterate over mini-batches:

       a. **Per-sample gradient**: for each point x_i in the batch,
              g_i = ∂L_i/∂W = -2 · outer(x_i, x_i @ W)    shape (d, d')
          This is the per-sample contribution to the full gradient -2 X^T X W.

       b. **Clip**: bound the Frobenius norm of each g_i:
              g_i ← g_i · min(1, clip_norm / ||g_i||_F)
          This ensures the L2 sensitivity of the gradient sum over one mini-batch is
          at most clip_norm regardless of the data point.

          These two steps are implemented vectorized (no per-sample Python loop): since g_i
          is rank-1, ||g_i||_F = 2||x_i|| ||x_i @ W|| exactly, so the clipped sum is
              Σ_i min(1, clip_norm/||g_i||_F) · g_i = -2 · (c_i x_i)^T @ (x_i @ W),
          two matmuls that match the naive loop to floating-point round-off.

       c. **Add noise**: draw Z ~ N(0, sigma² · clip_norm² · I) (shape (d, d'))
          and form the noisy average gradient:
              g_noisy = (Σ_i g_i^clipped + Z) / b

       d. **Gradient step**: W ← W - lr · g_noisy

       e. **Re-orthonormalize**: W, _ = qr(W)
          This projects W back onto the Stiefel manifold after the noisy gradient
          perturbs it.  Without this step the basis vectors would drift, become
          collinear, and the sign-based cluster assignment would degrade.

    ## Privacy guarantee

    The output W is (epsilon, delta)-differentially private with respect to X: replacing
    any single row of X changes the distribution of W by at most a factor e^epsilon (with
    probability 1 - delta).  The guarantee holds for the specific epsilon and delta passed
    in; these are spent entirely on basis computation and are independent of any subsequent
    DP mechanism applied to cluster centroids.

    ## Choosing hyperparameters

    - **clip_norm**: should match the typical per-sample gradient norm.  For data
      normalized to [-1, 1]^d the gradient norm ||g_i||_F = 2 ||x_i||_2 · ||x_i @ W||_2
      is at most 2·sqrt(d)·sqrt(d'), so clip_norm=1.0 is conservative but safe.  Larger
      values allow more signal through at the cost of needing more noise.

    - **epsilon / delta**: standard choices are epsilon ∈ {0.1, 0.5, 1.0} and
      delta = 1 / (n · log n).  Tighter epsilon forces larger sigma and degrades the
      basis quality.

    - **epochs**: more epochs = more SGD compositions = higher sigma for the same epsilon.
      With the subsampling amplification, 10 epochs is usually a good balance.

    - **batch_size**: smaller batches give stronger subsampling amplification (sigma grows
      more slowly with epochs) but more iterations, which can slow wall-clock time.
      256 works well for datasets up to ~100k points; consider larger batches for very
      small datasets where batch_size > n/2 collapses to full-batch GD.

    - **data_fraction** (if subsampling): using a fraction f of the data reduces n to
      f·n, cutting T by f and allowing sigma to shrink accordingly, which improves utility
      at the cost of using less data for the basis.  10% (f=0.1) is a practical default.

    Args:
        X: (n, d) data matrix; should be normalized (e.g., to [-1, 1]^d) so that
           the default clip_norm=1.0 is meaningful
        d_prime: number of private principal components to return
        epsilon: (epsilon, delta)-DP privacy budget for this computation; independent
                 of any clustering epsilon
        delta: (epsilon, delta)-DP delta; standard: 1 / (n · log(n))
        clip_norm: per-sample gradient clipping threshold (Frobenius norm);
                   bounds the sensitivity of the sum query to clip_norm
        epochs: number of full passes over X during SGD (default: 10)
        lr: SGD learning rate; 0.01 works for normalized data (default: 0.01)
        batch_size: mini-batch size; controls both step count and subsampling rate
                    q = batch_size / n (default: 256)
        data_fraction: fraction of X to subsample before running DP-SGD, in (0, 1].
                       Reduces the number of SGD steps proportionally, letting the
                       noise calibration return a smaller sigma for the same budget.
                       E.g. 0.1 uses 10% of the data (default: 1.0, use all data)
        full_n: size of the full dataset that X was Poisson-sub-sampled from. When given
                and > n, the (epsilon, delta) budget is calibrated **w.r.t. the full
                dataset** via amplification by sub-sampling (`_find_sigma_autodp_full`),
                i.e. less noise for the same full-dataset guarantee. None ⟹ the guarantee
                is w.r.t. X itself (default: None).

    Returns:
        W: (d, d_eff) orthonormal matrix whose columns approximate the top-d' principal
           components of X under (epsilon, delta)-DP, where d_eff = min(d_prime, d)
    """
    n, d = X.shape
    d_eff = min(d_prime, d)

    # Subsample the dataset before running DP-SGD.  Using a fraction f of n reduces the
    # number of SGD steps by f, which lets _find_sigma_autodp return a smaller sigma for
    # the same (epsilon, delta) budget — improving utility at the cost of a smaller basis
    # training set.  The DP guarantee holds over the subsampled dataset; the additional
    # subsampling from the full dataset only strengthens privacy.
    rng = np.random.RandomState(None)
    if data_fraction < 1.0:
        n_sub = max(batch_size, int(n * data_fraction))
        idx = rng.choice(n, size=n_sub, replace=False)
        X = X[idx]
        n = n_sub

    # Center the data
    X_c = X - X.mean(axis=0)

    # Initialize W as a random orthonormal matrix
    W = random_orthogonal_basis(d, d_eff)

    # Find noise multiplier via moments accountant (uses subsampled n after data_fraction).
    # If full_n is given and X is a sub-sample of a larger pool, calibrate the budget w.r.t.
    # the FULL pool via amplification by sub-sampling (less noise for the same guarantee).
    # This is a data-independent, memoized, one-time cost (see the _find_sigma_* notes):
    # constant w.r.t. n / d / #clients, so it does not affect timing scalability slopes.
    # A precomputed `sigma` may be passed in (e.g. so the timing harness can measure the
    # calibration and the SGD loop separately); None ⟹ calibrate here.
    if sigma is None:
        sigma, _ = dpsgd_calibrate_sigma(epsilon, delta, n, batch_size, epochs, full_n)

    for _ in range(epochs):
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = X_c[indices[start: start + batch_size]]
            b = len(batch)
            if b == 0:
                continue

            # Sum of per-sample clipped gradients, vectorized (identical to the naive
            # per-sample loop up to floating point). Each sample's gradient is the rank-1
            # matrix g_i = -2 * outer(x_i, x_i @ W). The Frobenius norm of a rank-1 outer
            # product is the product of the two vector norms, so ||g_i||_F = 2||x_i|| ||x_i@W||
            # exactly -- letting us clip WITHOUT ever forming the (d, d_eff) matrix g_i:
            #   c_i     = min(1, clip_norm / ||g_i||_F)              (per-sample clip factor)
            #   agg     = sum_i c_i g_i = -2 * (c_i x_i)^T @ (x_i @ W)
            # This replaces the O(b) Python loop (which dominated the build ~95%) with two
            # BLAS matmuls; the RNG call sequence (permutation per epoch, noise per step) is
            # unchanged, so results match the loop to round-off (~1e-15).
            proj = batch @ W                                                    # (b, d_eff)
            g_norm = 2.0 * np.linalg.norm(batch, axis=1) * np.linalg.norm(proj, axis=1)  # (b,)
            clip_factor = np.minimum(1.0, clip_norm / (g_norm + 1e-8))          # (b,)
            agg_grad = -2.0 * (batch * clip_factor[:, None]).T @ proj           # (d, d_eff)

            # Add calibrated Gaussian noise (sensitivity = clip_norm)
            noise = rng.normal(0.0, sigma * clip_norm, size=W.shape)
            noisy_grad = (agg_grad + noise) / b

            # Gradient descent step
            W = W - lr * noisy_grad

            # Re-orthonormalize so sign-based assignment stays meaningful
            W, _ = np.linalg.qr(W)

    return W[:, :d_eff]


def orthogonal_basis(X, d_prime, method="random", seed=42, **kwargs):
    """Dispatcher for orthonormal basis generation.

    Args:
        X: (n, d) data matrix (used for dpsgd_pca and svd_pca; only shape used for random)
        d_prime: desired number of basis vectors
        method: "random", "dpsgd_pca", or "svd_pca"
        seed: random seed (used for random; ignored for dpsgd_pca and svd_pca)
        **kwargs: for dpsgd_pca — epsilon, delta, clip_norm required;
                  data_fraction optional (default 1.0)

    Returns:
        (d, d_eff) orthonormal matrix, d_eff = min(d_prime, d)
    """
    if method == "random":
        return random_orthogonal_basis(X.shape[1], d_prime, seed=seed)
    elif method == "dpsgd_pca":
        return dpsgd_pca_basis(
            X, d_prime,
            epsilon=kwargs["epsilon"],
            delta=kwargs["delta"],
            clip_norm=kwargs["clip_norm"],
            epochs=kwargs.get("epochs", 10),
            lr=kwargs.get("lr", 0.1),
            data_fraction=kwargs.get("data_fraction", 1.0),
            full_n=kwargs.get("full_n", None),
            sigma=kwargs.get("sigma", None),
        )
    elif method == "svd_pca":
        return svd_pca_basis(X, d_prime)
    else:
        raise ValueError(f"Unknown basis method: {method!r}. Choose 'random', 'dpsgd_pca', or 'svd_pca'.")


def orthogonalize_svd(R):
    """
    Orthogonalize a matrix via economy SVD.

    Args:
        R: (d, k) matrix to orthogonalize

    Returns:
        (d, k) matrix with orthonormal columns
    """
    Q, _, _ = np.linalg.svd(R, full_matrices=False)
    Q = Q / np.linalg.norm(Q, axis=0)
    return Q


def svd_pca_basis(X, d_prime):
    """Compute the top-d' principal components via non-private SVD.

    Centers X and returns the d' right singular vectors of X corresponding to the
    largest singular values (i.e. the true top principal components).  This is the
    non-private oracle baseline — it uses the full data without any noise, so it
    gives the best possible basis quality but provides no differential privacy
    guarantee for the basis itself.

    Args:
        X: (n, d) data matrix
        d_prime: number of principal components to return

    Returns:
        (d, d_eff) orthonormal matrix of top principal components,
        where d_eff = min(d_prime, d)
    """
    d_eff = min(d_prime, X.shape[1])
    X_c = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    return Vt[:d_eff].T  # (d, d_eff)


def random_orthogonal_basis(d, d_prime, seed=42, orthogonalize=None):
    """
    Generate a random orthonormal basis.

    Draws a random Gaussian matrix and orthogonalizes it using the
    provided method (defaults to SVD).

    Args:
        d: ambient dimensionality (number of rows)
        d_prime: desired number of orthogonal basis vectors
        seed: random seed for reproducibility
        orthogonalize: callable (R) -> Q that takes a (d, k) matrix and
                       returns a (d, k) orthonormal matrix. Defaults to
                       orthogonalize_svd.

    Returns:
        (d, d_eff) array with orthonormal columns, where d_eff = min(d_prime, d)
    """
    if orthogonalize is None:
        orthogonalize = orthogonalize_svd
    d_eff = min(d_prime, d)
    rng = np.random.RandomState(seed)
    R = rng.randn(d, d_eff)
    return orthogonalize(R)


def zcdp_rho_from_epsilon(epsilon, delta):
    """Largest rho such that rho-zCDP still implies (epsilon, delta)-DP.

    Inverts the standard conversion (Bun & Steinke 2016): a rho-zCDP mechanism
    is (rho + 2 sqrt(rho ln(1/delta)), delta)-DP. Solving
        epsilon = rho + 2 sqrt(rho ln(1/delta))
    for rho (a quadratic in sqrt(rho)) gives

        sqrt(rho) = sqrt(ln(1/delta) + epsilon) - sqrt(ln(1/delta)).

    Since the conversion is an upper bound on epsilon, a mechanism with total
    zCDP at most this rho is guaranteed to be (epsilon, delta)-DP.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if not (0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    ln_inv_delta = np.log(1.0 / delta)
    return (np.sqrt(ln_inv_delta + epsilon) - np.sqrt(ln_inv_delta)) ** 2


def compute_dp_sigmas_zcdp(epsilon, delta, sigma_fraction, count_levels):
    """Rigorous zCDP budget split for the LSH-tree aggregation.

    The aggregation releases, on unit-norm data (L2 sensitivity 1 throughout):
      - ONE leaf-centroid sum query. Leaves partition the points, so the whole
        vector of leaf sums is a single Gaussian mechanism -> rho_centers.
      - ONE noisy-count histogram PER TREE LEVEL. Within a level the points are
        partitioned across nodes (parallel composition -> one mechanism per
        level, sensitivity 1); across the `count_levels` levels the releases
        compose sequentially. Pass the data-independent upper bound
        `count_levels = max_depth + 1` (the realized depth is itself private).

    All releases compose additively in zCDP:
        rho_total = rho_centers + count_levels * rho_per_level.
    We convert the target (epsilon, delta) to rho_total via
    `zcdp_rho_from_epsilon` (no delta splitting needed -- zCDP spends delta once
    at conversion), then split it so that
        sigma_count / sigma_centers == sigma_fraction
    (larger sigma_fraction -> less center noise, noisier counts).

    For a sensitivity-1 Gaussian, rho = 1/(2 sigma^2), i.e. sigma = 1/sqrt(2 rho).
    Writing L = count_levels, f = sigma_fraction, the ratio constraint gives
        rho_centers      = rho_total / (1 + L / f^2)
        rho_counts_total = rho_total - rho_centers
        sigma_centers    = 1 / sqrt(2 rho_centers)
        sigma_count      = sqrt(L / (2 rho_counts_total))   (per-level sigma)

    Args:
        epsilon:        total privacy budget for the aggregation.
        delta:          total delta budget for the aggregation.
        sigma_fraction: ratio sigma_count / sigma_centers (> 0); larger -> less
                        center noise (better centroids), noisier counts.
        count_levels:   number of sequential count releases (tree levels),
                        i.e. max_depth + 1.

    Returns:
        sigma_centers: noise std for the leaf-centroid sums (single release).
        sigma_count:   noise std for each per-level count release.
    """
    if sigma_fraction <= 0:
        raise ValueError(f"sigma_fraction must be positive, got {sigma_fraction}")
    if count_levels < 1:
        raise ValueError(f"count_levels must be >= 1, got {count_levels}")

    rho_total = zcdp_rho_from_epsilon(epsilon, delta)

    L = count_levels
    f = sigma_fraction
    rho_centers = rho_total / (1.0 + L / f ** 2)
    rho_counts_total = rho_total - rho_centers

    sigma_centers = 1.0 / np.sqrt(2 * rho_centers)
    sigma_count = np.sqrt(L / (2 * rho_counts_total))

    return sigma_centers, sigma_count
