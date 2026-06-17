import numpy as np


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

    The binary search finds the smallest sigma such that the above accounting yields a
    total cost ≤ epsilon.  It converges in 64 iterations to precision ~10^{-14} over the
    search range [0.01, 1000].

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
    from autodp import rdp_bank

    T = epochs * max(1, n // batch_size)
    q = min(1.0, batch_size / n)
    alphas = list(range(2, 256))

    def dp_cost(sigma):
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

    lo, hi = 0.01, 1000.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if dp_cost(mid) > epsilon:
            lo = mid
        else:
            hi = mid
    return hi


def dpsgd_pca_basis(X, d_prime, epsilon, delta, clip_norm, epochs=10, lr=0.01, batch_size=256, data_fraction=1.0):
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

    # Find noise multiplier via moments accountant (uses subsampled n after data_fraction)
    sigma = _find_sigma_autodp(epsilon, delta, n, batch_size, epochs)

    for _ in range(epochs):
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = X_c[indices[start: start + batch_size]]
            b = len(batch)
            if b == 0:
                continue

            # Accumulate per-sample clipped gradients
            agg_grad = np.zeros_like(W)
            for xi in batch:
                proj = xi @ W                          # (d_eff,)
                g_i = -2.0 * np.outer(xi, proj)        # (d, d_eff)
                g_norm = np.linalg.norm(g_i)
                g_i *= min(1.0, clip_norm / (g_norm + 1e-8))
                agg_grad += g_i

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
            data_fraction=kwargs.get("data_fraction", 1.0),
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


def ortho_assign(values, d_prime, seed=42, basis=None):
    """
    Assign points to clusters via orthogonal projection quadrants.

    Projects each point onto an orthonormal basis and assigns it to a
    quadrant based on the sign of each projection. Produces up to
    2^d_prime clusters.

    Args:
        values: (n, d) array of data points
        d_prime: number of orthogonal basis vectors
        seed: random seed for reproducibility (ignored when basis is provided)
        basis: optional (d, d_prime) orthonormal matrix to use instead of
               generating one via SVD. When provided, seed is ignored.

    Returns:
        (n,) integer array of cluster labels in [0, 2^d_eff - 1]
    """
    n, d = values.shape
    d_eff = min(d_prime, d)

    if basis is not None:
        Q = basis
    else:
        Q = random_orthogonal_basis(d, d_prime, seed=seed)

    # Project points onto orthogonal basis
    projections = values @ Q  # (n, d_eff)

    # Convert sign pattern to cluster ID
    signs = (projections >= 0).astype(int)  # (n, d_eff) binary
    labels = signs @ (2 ** np.arange(Q.shape[1]))  # (n,) integer

    return labels


# This is not private, since the cluster_points isn't private
def noisy_cluster_centers(values, labels, sigma, seed=42):
    """
    Compute cluster centroids with Gaussian noise added to sums.

    Adds isotropic Gaussian noise N(0, sigma^2 I) to each cluster's
    sum vector before dividing by count, following the DP mechanism
    from "Improved Private DP Clustering via Projections".

    Args:
        values: (n, d) array of data points
        labels: (n,) integer cluster labels
        sigma: standard deviation of Gaussian noise added to each cluster sum
        seed: random seed for noise generation

    Returns:
        (k, d) array of noisy cluster centroids
        unique_labels: (k,) sorted array of the label ids
    """
    unique_labels = np.unique(labels)
    d = values.shape[1]
    rng = np.random.RandomState(seed)
    centers = np.empty((len(unique_labels), d))
    for i, lab in enumerate(unique_labels):
        cluster_points = values[labels == lab]
        cluster_sum = cluster_points.sum(axis=0)
        noise = rng.normal(0, sigma, size=d)
        centers[i] = (cluster_sum + noise) / cluster_points.shape[0]
    return centers, unique_labels


def cluster_counts(labels):
    """
    Compute the number of points in each cluster.

    Args:
        labels: (n,) integer cluster labels

    Returns:
        counts: (k,) array of point counts per cluster
        unique_labels: (k,) sorted array of the label ids
    """
    unique_labels = np.unique(labels)
    counts = np.array([np.sum(labels == lab) for lab in unique_labels])
    return counts, unique_labels



def noisy_cluster_counts(labels, sigma, seed=42):
    """
    Compute cluster counts with Gaussian noise added to each count.

    Follows the same DP mechanism structure as noisy_cluster_centers:
    adds isotropic Gaussian noise N(0, sigma^2) to each cluster's scalar
    count. L2 sensitivity of a count query is 1 (adding/removing one
    unit-norm point changes exactly one cluster's count by 1).

    Args:
        labels:         (n,) integer cluster labels
        sigma:          std dev of Gaussian noise added to each count
        seed:           random seed for noise generation

    Returns:
        counts:         (k,) array of noisy counts (floats; may be negative
                        for very small clusters with large sigma)
        unique_labels:  (k,) sorted array of label ids
    """
    unique_labels = np.unique(labels)
    rng = np.random.RandomState(seed)

    counts = np.array(
        [np.sum(labels == lab) for lab in unique_labels],
        dtype=float
    )
    noise = rng.normal(0, sigma, size=len(unique_labels))
    return counts + noise, unique_labels


def cluster_centers(values, labels):
    """
    Compute the centroid of each cluster.

    Args:
        values: (n, d) array of data points
        labels: (n,) integer cluster labels

    Returns:
        (k, d) array of cluster centroids, where k = number of unique labels.
              Row i is the mean of all points with label unique_labels[i].
        unique_labels: (k,) sorted array of the label ids
    """
    unique_labels = np.unique(labels)
    centers = np.empty((len(unique_labels), values.shape[1]))
    for i, lab in enumerate(unique_labels):
        centers[i] = values[labels == lab].mean(axis=0)
    return centers, unique_labels



def compute_dp_sigmas(epsilon, delta, sigma_fraction):
    """
    Split a total (epsilon, delta) DP budget between noisy_cluster_centers
    and noisy_cluster_counts such that:

        epsilon_centers = sigma_fraction * epsilon_count

    Assumes:
    - Unit-norm data points  => L2 sensitivity of per-cluster sum query = 1
    - Scalar count query     => L2 sensitivity = 1
    - Gaussian mechanism:      sigma = sqrt(2 ln(1.25 / delta_i)) / epsilon_i
    - Basic composition:       epsilon_centers + epsilon_count = epsilon
    - Equal delta split:       delta_centers = delta_count = delta / 2

    Derivation:
        epsilon_count   = epsilon / (1 + sigma_fraction)
        epsilon_centers = epsilon * sigma_fraction / (1 + sigma_fraction)

        gauss_const = sqrt(2 * ln(1.25 / (delta / 2)))
        sigma_count   = gauss_const / epsilon_count
        sigma_centers = gauss_const / epsilon_centers
                      = sigma_count / sigma_fraction

    Note: larger sigma_fraction allocates more epsilon to centers,
    reducing sigma_centers (better centroid utility) at the cost of
    increased sigma_count (noisier counts).

    Args:
        epsilon:          total privacy budget (epsilon_centers + epsilon_count)
        delta:            total delta budget, split equally as delta/2 each
        sigma_fraction:   ratio epsilon_centers / epsilon_count  (> 0)

    Returns:
        sigma_centers:  noise std for noisy_cluster_centers  (added to each cluster sum)
        sigma_count:    noise std for noisy_cluster_counts   (added to each cluster count)
    """
    if sigma_fraction <= 0:
        raise ValueError(f"sigma_fraction must be positive, got {sigma_fraction}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if not (0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    delta_each = delta / 2
    gauss_const = np.sqrt(2 * np.log(1.25 / delta_each))

    eps_count   = epsilon / (1 + sigma_fraction)
    eps_centers = epsilon * sigma_fraction / (1 + sigma_fraction)

    sigma_count   = gauss_const / eps_count    # = gauss_const * (1 + sigma_fraction) / epsilon
    sigma_centers = gauss_const / eps_centers  # = sigma_count / sigma_fraction

    return sigma_centers, sigma_count


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
    at conversion), then split it so that, as in `compute_dp_sigmas`,
        sigma_count / sigma_centers == sigma_fraction.

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


def noisy_cluster_centers_and_counts(values, labels, sigma_centers, sigma_count, seed=42):
    """DP per-orthant centroids and counts from a single noise source.
 
    For each orthant releases the noisy sum and the noisy count, then forms the
    centroid as  noisy_sum / noisy_count. The centroid is therefore pure
    post-processing of two Gaussian-mechanism releases and never divides by the
    true count, so it does not leak the exact count. Both noises are drawn from
    ONE RandomState (sum first, then count) so they are independent -- unlike
    calling separately-seeded noisy_cluster_centers / noisy_cluster_counts with
    the same seed, which correlates the two noise vectors and leaves a
    no-noise direction in [sum, count] space.
 
    Privacy: with sigma_centers calibrated to (eps_centers, delta/2) for the sum
    query (L2 sensitivity 1 on unit-norm data) and sigma_count to
    (eps_count, delta/2) for the count query (sensitivity 1), the pair composes
    to (eps_centers + eps_count, delta) by sequential composition.
 
    Args:
        values: (n, d) data points (unit-norm assumed for the sensitivity above)
        labels: (n,) integer orthant ids from a basis fixed before this call
        sigma_centers: noise std for the cluster sums (centroid numerator)
        sigma_count: noise std for the cluster counts (centroid denominator)
        seed: RNG seed
 
    Returns:
        centers: (k, d) noisy centroids (noisy_sum / noisy_count)
        noisy_counts: (k,) noisy counts (float; may be non-integer or <= 0)
        unique_labels: (k,) sorted orthant ids present in labels
    """
    unique_labels = np.unique(labels)
    d = values.shape[1]
    rng = np.random.RandomState(seed)
 
    centers = np.empty((len(unique_labels), d))
    noisy_counts = np.empty(len(unique_labels))
    for i, lab in enumerate(unique_labels):
        pts = values[labels == lab]
        noisy_sum = pts.sum(axis=0) + rng.normal(0.0, sigma_centers, size=d)
        noisy_count = pts.shape[0] + rng.normal(0.0, sigma_count)
 
        noisy_counts[i] = noisy_count
        denom = noisy_count if noisy_count >= 1.0 else 1.0  # guard tiny/<=0 counts
        centers[i] = noisy_sum / denom
 
    return centers, noisy_counts, unique_labels
